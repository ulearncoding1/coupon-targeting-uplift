# -*- coding: utf-8 -*-
"""SQL/Spark SQL 对账层：用 DuckDB 在本地执行 sql/*.sql，并与现有 Pandas 结果对账。

设计原则：
1. SQL 只做特征聚合、描述统计和对账，不替代 04/05 的模型估计。
2. SQL 文件不写 DuckDB 专有 read_csv_auto；由本脚本注册视图，方便平移到 Spark。
3. 每条查询输出到 outputs/tables/sql/，并生成 _manifest.csv 与 _validation.csv。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

from config import CLEAN, TABLE, LOG

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
SQL_OUT = TABLE / "sql"
VALIDATION_PATH = LOG / "07_sql_validation.txt"

# SQL 里引用的视图 -> CSV 文件
VIEWS = {
    "raw_transcript": CLEAN / "transcript.csv",
    "raw_profile": CLEAN / "profile.csv",
    "raw_portfolio": CLEAN / "portfolio.csv",
    "user_features": CLEAN / "user_features.csv",
    "user_segments": CLEAN / "user_segments.csv",
    "offer_dataset": TABLE / "04_offer_dataset.csv",
    "send_windows": TABLE / "05_send_windows.csv",
    "policy_v2": TABLE / "05_policy_v2.csv",
    "offer_matrix": TABLE / "03_offer_matrix.csv",
    "segment_profile": TABLE / "02_segment_profile.csv",
    "segment_uplift": TABLE / "05_segment_uplift.csv",
    "uplift_deciles": TABLE / "05_uplift_deciles.csv",
}


def sql_path(path: Path) -> str:
    """DuckDB read_csv_auto 使用正斜杠，避免 Windows 反斜杠转义。"""
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    for name, path in VIEWS.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件：{path}")
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_csv_auto('{sql_path(path)}', header=true)"
        )


def run_queries(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    SQL_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted(SQL_DIR.glob("*.sql")):
        sql = file.read_text(encoding="utf-8")
        df = con.execute(sql).fetch_df()
        out = SQL_OUT / f"{file.stem}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "query": file.stem,
                "rows": len(df),
                "columns": len(df.columns),
                "output": str(out.relative_to(TABLE.parent)),
            }
        )
        print(f"[SQL] {file.stem}: {len(df):,} rows, {len(df.columns)} cols -> {out.name}")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(SQL_OUT / "_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def _check(name: str, actual: float, expected: float, tol: float = 0.0, note: str = "") -> dict:
    ok = abs(float(actual) - float(expected)) <= tol
    return {
        "check": name,
        "actual": actual,
        "expected": expected,
        "tolerance": tol,
        "pass": bool(ok),
        "note": note,
    }


def validate() -> pd.DataFrame:
    checks = []

    # 01 事件漏斗
    ev = pd.read_csv(SQL_OUT / "01_event_funnel.csv")
    counts = dict(zip(ev["event"], ev["n_events"]))
    expected_events = {
        "transaction": 138953,
        "offer received": 76277,
        "offer viewed": 57725,
        "offer completed": 33579,
    }
    for event, n in expected_events.items():
        checks.append(_check(f"event::{event}", counts.get(event, -1), n, 0, "原始事件数"))
    checks.append(_check("event::total", ev["n_events"].sum(), 306534, 0, "事件总数"))

    # 02 用户特征
    uf = pd.read_csv(SQL_OUT / "02_user_features_rebuild.csv")
    base = pd.read_csv(VIEWS["user_features"], usecols=["user_id", "n_tx", "monetary", "recency_h"])
    checks.append(_check("user_features::rows", len(uf), len(base), 0, "用户行数"))
    checks.append(_check("user_features::n_tx_sum", uf["n_tx"].sum(), base["n_tx"].sum(), 0, "交易数合计"))
    checks.append(_check("user_features::monetary_sum", uf["monetary"].sum(), base["monetary"].sum(), 1.0, "消费额合计"))
    checks.append(_check("user_features::recency_mean", uf["recency_h"].mean(), base["recency_h"].mean(), 0.02, "距上次交易小时均值"))
    checks.append(_check("user_features::recency_null", uf["recency_h"].isna().sum(), base["recency_h"].isna().sum(), 0, "无交易用户数"))

    # 03 分群画像
    sp = pd.read_csv(SQL_OUT / "03_segment_profile.csv")
    expected_sp = pd.read_csv(VIEWS["segment_profile"])
    checks.append(_check("segment_profile::rows", len(sp), len(expected_sp), 0, "人群数"))
    checks.append(_check("segment_profile::n_sum", sp["n"].sum(), expected_sp["n"].sum(), 0, "覆盖用户数"))

    # 04 券型矩阵
    om = pd.read_csv(SQL_OUT / "04_offer_matrix.csv")
    expected_om = pd.read_csv(VIEWS["offer_matrix"])
    checks.append(_check("offer_matrix::sends", om["sends"].sum(), expected_om["sends"].sum(), 0, "发券次数"))
    checks.append(_check("offer_matrix::comps", om["comps"].sum(), expected_om["comps"].sum(), 0, "完成次数"))
    checks.append(_check("offer_matrix::rows", len(om), len(expected_om), 0, "人群×券型行数"))

    # 05 描述性增量
    su = pd.read_csv(SQL_OUT / "05_segment_uplift_raw.csv")
    sw = pd.read_csv(VIEWS["send_windows"], usecols=["coupon"])
    checks.append(_check("raw_uplift::sends", su["n_sends"].sum(), len(sw), 0, "窗口记录数"))
    checks.append(_check("raw_uplift::coupon", su["n_coupon"].sum(), (sw["coupon"] == 1).sum(), 0, "券记录数"))
    checks.append(_check("raw_uplift::info", su["n_info"].sum(), (sw["coupon"] == 0).sum(), 0, "信息型记录数"))

    # 06 十分组（描述性）
    dec = pd.read_csv(SQL_OUT / "06_uplift_deciles.csv")
    checks.append(_check("deciles::rows", len(dec), 10, 0, "十组"))
    checks.append(_check("deciles::n_sum", dec["n"].sum(), len(sw), 0, "全量窗口"))

    # 07 购买间隔
    seq = pd.read_csv(SQL_OUT / "07_purchase_sequence.csv")
    checks.append(_check("purchase_sequence::n_transitions", seq.loc[0, "n_transitions"], 122375, 0, "交易数 - 有交易用户数"))
    checks.append(_check("purchase_sequence::n_users", seq.loc[0, "n_users"], 16177, 0, "至少有两次交易的用户"))

    # 08 数据质量
    dq = pd.read_csv(SQL_OUT / "08_data_quality.csv")
    for _, row in dq.iterrows():
        checks.append(_check(f"quality::{row['check_name']}", row["failures"], 0, 0, f"total={row['total']}"))

    # 09 v2 策略汇总
    pol = pd.read_csv(SQL_OUT / "09_policy_summary.csv").iloc[0]
    checks.append(_check("policy::subsidy", pol["subsidy"], 72258, 1, "v2 期望核销补贴"))
    checks.append(_check("policy::inc_gmv", pol["inc_gmv"], 191880, 2, "v2 增量 GMV"))
    checks.append(_check("policy::inc_net", pol["inc_net"], 119622, 2, "v2 增量净额"))
    checks.append(_check("policy::roi", pol["incremental_roi"], 2.655, 0.01, "增量 ROI"))

    result = pd.DataFrame(checks)
    result.to_csv(SQL_OUT / "_validation.csv", index=False, encoding="utf-8-sig")
    return result


def write_report(result: pd.DataFrame) -> int:
    failed = result[~result["pass"]]
    lines = []
    lines.append("07 SQL 对账报告")
    lines.append("=" * 60)
    lines.append(f"检查项：{len(result)}；通过：{len(result) - len(failed)}；失败：{len(failed)}")
    lines.append("")
    lines.append(result.to_string(index=False))
    lines.append("")
    if len(failed):
        lines.append("失败项：")
        lines.append(failed.to_string(index=False))
    else:
        lines.append("结论：SQL 重算与现有 Pandas 结果对账通过。")
    text = "\n".join(lines)
    VALIDATION_PATH.write_text(text, encoding="utf-8")
    print(text)
    return 1 if len(failed) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DuckDB SQL layer for coupon project")
    parser.add_argument("--skip-run", action="store_true", help="只对已有 SQL 输出做对账")
    args = parser.parse_args()

    con = duckdb.connect(database=":memory:")
    create_views(con)
    if not args.skip_run:
        run_queries(con)
    con.close()

    result = validate()
    return write_report(result)


if __name__ == "__main__":
    sys.exit(main())
