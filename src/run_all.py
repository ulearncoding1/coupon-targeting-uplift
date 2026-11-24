# -*- coding: utf-8 -*-
"""一键运行入口：按 00 -> 09 顺序执行，统一输出到控制台和 outputs/logs/。

用法：
    python src/run_all.py                 # 从 00 开始完整跑
    python src/run_all.py --from-step 7   # 只跑 SQL / A/B / 监控
    python src/run_all.py --dry-run        # 只检查输入文件和命令，不执行
    python src/run_all.py --list           # 列出步骤

设计原则：
1. 用 subprocess 调每个脚本，保持各脚本可单独运行，不强行改成 import。
2. 每一步有前置文件检查，缺文件时给出明确提示，不跑到一半才报错。
3. 每个步骤的 stdout/stderr 同时写入日志，方便复现和排错。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
TABLE = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "报告"
DASH = ROOT / "outputs" / "看板"
LOG = ROOT / "outputs" / "logs"


STEPS = [
    {
        "id": 0,
        "script": "00_parse_json.py",
        "title": "解析 JSONL 原始数据",
        "requires": [RAW / "portfolio.json", RAW / "profile.json", RAW / "transcript.json"],
    },
    {
        "id": 1,
        "script": "01_profile_rfm.py",
        "title": "用户画像与 RFM/行为特征",
        "requires": [CLEAN / "portfolio.csv", CLEAN / "profile.csv", CLEAN / "transcript.csv"],
    },
    {
        "id": 2,
        "script": "02_cluster.py",
        "title": "价值×行为双聚类与人群画像",
        "requires": [CLEAN / "user_features.csv"],
    },
    {
        "id": 3,
        "script": "03_strategy.py",
        "title": "人群×券型策略矩阵",
        "requires": [CLEAN / "user_features.csv", CLEAN / "user_segments.csv"],
    },
    {
        "id": 4,
        "script": "04_response_model.py",
        "title": "LightGBM 响应模型",
        "requires": [TABLE / "03_offer_matrix.csv", CLEAN / "user_features.csv"],
    },
    {
        "id": 5,
        "script": "05_uplift.py",
        "title": "增量评估与预算策略 v2",
        "requires": [TABLE / "04_offer_dataset.csv", CLEAN / "user_segments.csv"],
    },
    {
        "id": 6,
        "script": "06_build_dashboard.py",
        "title": "构建 HTML 六视图看板",
        "requires": [TABLE / "05_policy_v2.csv", FIG / "05_policy_compare.png"],
    },
    {
        "id": 7,
        "script": "07_sql_pipeline.py",
        "title": "DuckDB / Spark SQL 对账",
        "requires": [
            CLEAN / "transcript.csv",
            CLEAN / "user_features.csv",
            CLEAN / "user_segments.csv",
            TABLE / "03_offer_matrix.csv",
            TABLE / "04_offer_dataset.csv",
            TABLE / "05_send_windows.csv",
        ],
    },
    {
        "id": 8,
        "script": "08_ab_power.py",
        "title": "A/B 实验设计与样本量",
        "requires": [TABLE / "04_offer_dataset.csv", TABLE / "05_send_windows.csv"],
    },
    {
        "id": 9,
        "script": "09_monitoring.py",
        "title": "监控与稳定性回放",
        "requires": [TABLE / "04_offer_dataset.csv", TABLE / "05_send_windows.csv", CLEAN / "transcript.csv"],
    },
]


def ensure_dirs() -> None:
    for path in (LOG, TABLE, FIG, REPORT, DASH):
        path.mkdir(parents=True, exist_ok=True)


def check_requires(step: dict) -> list[Path]:
    return [path for path in step["requires"] if not path.exists()]


def stream_process(cmd: list[str], log_file) -> int:
    """实时打印并写日志；避免 capture_output 导致长任务没有进度。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
    rc = proc.wait()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="优惠券投放复现一键运行")
    parser.add_argument("--from-step", type=int, default=0, help="从第几步开始（0-9）")
    parser.add_argument("--list", action="store_true", help="列出步骤")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入和命令，不执行")
    args = parser.parse_args()

    if args.list:
        for step in STEPS:
            print(f"{step['id']:>2}  {step['script']:<24} {step['title']}")
        return 0

    if not 0 <= args.from_step <= 9:
        parser.error("--from-step 必须在 0-9 之间")

    selected = [step for step in STEPS if step["id"] >= args.from_step]
    ensure_dirs()

    missing_any = False
    print(f"项目根：{ROOT}")
    print(f"计划执行：{len(selected)} 步，从 {selected[0]['script']} 开始\n")
    for step in selected:
        missing = check_requires(step)
        if missing:
            missing_any = True
            print(f"[缺输入] {step['script']}：")
            for path in missing:
                print(f"  - {path}")
    if missing_any:
        print("\n请先补齐输入文件，或用更小的 --from-step 从上游步骤开始。")
        return 2

    if args.dry_run:
        for step in selected:
            print(f"[dry-run] python src/{step['script']}")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG / f"run_all_{stamp}.log"
    print(f"运行日志：{log_path}\n")

    started_all = time.time()
    results = []
    with log_path.open("w", encoding="utf-8") as log_file:
        header = f"run_all {stamp} | from_step={args.from_step}\n"
        print(header, end="")
        log_file.write(header)

        for index, step in enumerate(selected, start=1):
            cmd = [sys.executable, str(SRC / step["script"])]
            title = f"[{index}/{len(selected)}] {step['script']} | {step['title']}"
            print(f"\n{title}")
            log_file.write(f"\n{title}\n")
            started = time.time()
            rc = stream_process(cmd, log_file)
            elapsed = time.time() - started
            results.append((step, rc, elapsed))
            status = "PASS" if rc == 0 else "FAIL"
            summary = f"{status}  {step['script']}  {elapsed:.1f}s  (exit={rc})"
            print(summary)
            log_file.write(summary + "\n")
            if rc != 0:
                print(f"\n已停止：{step['script']} 返回 {rc}。完整日志见 {log_path}")
                return rc

    total = time.time() - started_all
    print(f"\n全部完成：{len(results)} 步，总耗时 {total:.1f}s")
    print(f"日志：{log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
