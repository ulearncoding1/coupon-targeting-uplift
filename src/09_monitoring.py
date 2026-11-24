# -*- coding: utf-8 -*-
"""09 监控与稳定性：PSI、模型衰减、d7 券效、数据量异常。

这不是生产实时监控，而是用 30 天历史数据做离线回放：
- 参考期：send_day 0–13
- 监控期：D14–20、D21–29
- 模型评估只用 test split，避免把训练集分数当线上表现
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from config import CLEAN, FIG, TABLE
from utils_viz import save_fig, use_style

PERIODS = ["D0-6", "D7-13", "D14-20", "D21-29"]
PERIOD_BINS = [-1, 6, 13, 20, 29]
REFERENCE = "D7-13"
TARGET_SEGMENTS = ["高价值·全响应型", "中价值·全响应型"]
D7_SUBSIDY = 3.0
PROFILE_FEATURES = {"age", "income", "tenure_days"}

FEATURE_CN = {
    "avg_order_prior": "历史客单价",
    "tenure_days": "会员天数",
    "max_order_prior": "历史最大单笔",
    "income": "收入",
    "age": "年龄",
    "reward": "券面奖励",
    "difficulty_to_aov": "门槛/客单价",
    "threshold_gap": "门槛缺口",
    "difficulty": "券门槛",
    "spend_prior": "历史消费额",
    "n_tx_prior": "历史交易笔数",
    "duration_days": "券有效期",
    "recency_h": "距上次交易小时",
    "p_complete": "完成概率分数",
}

PSI_COLORS = LinearSegmentedColormap.from_list(
    "psi", ["#F7F4ED", "#F2C14E", "#C44536"]
)


def add_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = pd.cut(df["send_day"], PERIOD_BINS, labels=PERIODS)
    return df


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float | None:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref) < 20 or len(cur) < 20:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    rb = pd.cut(ref, edges, include_lowest=True, duplicates="drop")
    cb = pd.cut(cur, edges, include_lowest=True, duplicates="drop")
    r = rb.value_counts(normalize=True).sort_index()
    c = cb.value_counts(normalize=True).sort_index()
    r = r.reindex(c.index, fill_value=0).clip(1e-4)
    c = c.reindex(r.index, fill_value=0).clip(1e-4)
    return float(((c - r) * np.log(c / r)).sum())


def psi_level(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "unknown"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "warn"
    return "severe"


def compute_psi(offer: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    # 与模型训练口径一致：去掉信息型/不可完成券，只看可评分券
    scored = offer[offer["split"] != "not_completable"].copy()
    ref = scored[(scored["send_day"] >= 7) & (scored["send_day"] <= 13)]
    rows = []
    for period in ["D14-20", "D21-29", "D14-29"]:
        if period == "D14-29":
            cur = scored[scored["send_day"] >= 14]
        else:
            cur = scored[scored["period"] == period]
        for feature in features:
            value = psi(ref[feature], cur[feature])
            if feature in PROFILE_FEATURES:
                category = "profile"
                note = "稳定画像特征"
            else:
                category = "behavior_history"
                note = "离线回放中受历史回看窗口左截断影响；需先核查特征回看口径，不能直接判线上漂移"
            rows.append(
                {
                    "feature": feature,
                    "feature_cn": FEATURE_CN.get(feature, feature),
                    "category": category,
                    "period": period,
                    "ref_n": len(ref),
                    "cur_n": len(cur),
                    "psi": round(value, 4) if value is not None else np.nan,
                    "level": psi_level(value),
                    "note": note,
                }
            )
    # 分数 PSI
    for period in ["D14-20", "D21-29", "D14-29"]:
        cur = scored[scored["send_day"] >= 14] if period == "D14-29" else scored[scored["period"] == period]
        value = psi(ref["p_complete"], cur["p_complete"])
        rows.append(
            {
                "feature": "p_complete",
                "feature_cn": "完成概率分数",
                "category": "score",
                "period": period,
                "ref_n": len(ref),
                "cur_n": len(cur),
                "psi": round(value, 4) if value is not None else np.nan,
                "level": psi_level(value),
                "note": "模型分数稳定性",
            }
        )
    return pd.DataFrame(rows)


def model_metrics(df: pd.DataFrame, label: str) -> dict:
    y = df["y_completed"].astype(int).to_numpy()
    p = df["p_complete"].astype(float).to_numpy()
    both = len(np.unique(y)) == 2
    return {
        "period": label,
        "n": len(df),
        "pos_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if both else np.nan,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if both else np.nan,
        "brier": round(float(brier_score_loss(y, p)), 4),
        "logloss": round(float(log_loss(y, p)), 4),
        "calib_gap": round(float(abs(p.mean() - y.mean())), 4),
    }


def compute_model_decay(offer: pd.DataFrame) -> pd.DataFrame:
    # 只看 test，避免 in-sample 分数把监控结果抬高
    test = offer[(offer["split"] == "test") & offer["p_complete"].notna()].copy()
    ref = test[(test["send_day"] >= 7) & (test["send_day"] <= 13)]
    ref_m = model_metrics(ref, REFERENCE)
    rows = [ref_m]
    for period in ["D0-6", "D14-20", "D21-29"]:
        cur = test[test["period"] == period]
        m = model_metrics(cur, period)
        m["auc_drop_vs_ref"] = round(ref_m["auc"] - m["auc"], 4) if not math.isnan(m["auc"]) else np.nan
        m["brier_increase_vs_ref"] = round(m["brier"] - ref_m["brier"], 4)
        m["calib_gap_vs_ref"] = round(m["calib_gap"] - ref_m["calib_gap"], 4)
        if period == "D0-6":
            m["alert"] = "cold_start"
            m["note"] = "活动冷启动，历史特征回看不足；不作模型衰减结论"
        else:
            m["alert"] = (
                "review"
                if (m["auc_drop_vs_ref"] > 0.05 or m["brier_increase_vs_ref"] > 0.03 or m["calib_gap"] > 0.05)
                else "ok"
            )
            m["note"] = "与 D7-13 基准比较"
        rows.append(m)
    return pd.DataFrame(rows)


def compute_coupon_monitor(offer: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_csv(TABLE / "05_send_windows.csv")
    y = offer[["person", "offer_id", "send_time", "y_completed"]]
    w = w.merge(y, on=["person", "offer_id", "send_time"], how="left")
    d7 = w[
        (w["offer_group"] == "d7")
        & w["segment_name"].isin(TARGET_SEGMENTS)
    ].copy()
    d7["period"] = pd.cut(d7["send_day"], PERIOD_BINS, labels=PERIODS)
    d7["subsidy"] = D7_SUBSIDY * d7["y_completed"].fillna(0)
    d7["net_gmv"] = d7["gmv72"] - d7["subsidy"]

    def agg(df: pd.DataFrame, label: str) -> dict:
        sends = len(df)
        users = df["person"].nunique()
        return {
            "period": label,
            "sends": sends,
            "users": users,
            "redemption_rate": round(float(df["y_completed"].mean()), 4) if sends else np.nan,
            "gmv_per_send": round(float(df["gmv72"].mean()), 2) if sends else np.nan,
            "subsidy_per_send": round(float(df["subsidy"].mean()), 2) if sends else np.nan,
            "net_gmv_per_send": round(float(df["net_gmv"].mean()), 2) if sends else np.nan,
            "net_gmv_total": round(float(df["net_gmv"].sum()), 2) if sends else np.nan,
        }

    rows = [agg(d7[(d7["send_day"] >= 7) & (d7["send_day"] <= 13)], REFERENCE)]
    for period in ["D0-6", "D14-20", "D21-29"]:
        rows.append(agg(d7[d7["period"] == period], period))

    out = pd.DataFrame(rows)
    ref_net = out.loc[out.period == REFERENCE, "net_gmv_per_send"].iloc[0]
    out["net_gmv_change_vs_ref"] = (out["net_gmv_per_send"] / ref_net - 1).round(4)
    out["alert"] = np.where(out["net_gmv_change_vs_ref"] < -0.20, "review", "ok")
    out.loc[out.period == REFERENCE, "alert"] = "reference"
    out.loc[out.period == "D0-6", "alert"] = "cold_start"
    return out


def compute_daily_quality() -> pd.DataFrame:
    trans = pd.read_csv(CLEAN / "transcript.csv", usecols=["time", "event", "person"])
    trans["day"] = np.floor(trans["time"] / 24).astype(int)
    daily = (
        trans.groupby("day")
        .agg(
            events=("event", "size"),
            users=("person", "nunique"),
            transactions=("event", lambda s: int((s == "transaction").sum())),
            offers=("event", lambda s: int((s == "offer received").sum())),
        )
        .reset_index()
    )
    ref = daily[(daily["day"] >= 7) & (daily["day"] <= 13)]
    mu, sd = ref["events"].mean(), ref["events"].std(ddof=1)
    daily["events_z"] = (daily["events"] - mu) / sd
    daily["alert"] = np.where(daily["events_z"].abs() > 3, "review", "ok")
    daily.loc[daily["day"] <= 6, "alert"] = "cold_start"
    return daily


def plot_psi(df: pd.DataFrame) -> Path:
    periods = ["D14-20", "D21-29", "D14-29"]
    pivot = df[df["period"].isin(periods)].pivot(
        index="feature_cn", columns="period", values="psi"
    )[periods]
    # 按最大 PSI 排序，最不稳定放上面
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    im = ax.imshow(
        pivot.values.astype(float),
        aspect="auto",
        cmap=PSI_COLORS,
        vmin=0,
        vmax=max(0.25, float(np.nanmax(pivot.values))),
    )
    ax.set_xticks(np.arange(len(periods)), labels=periods)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("特征 PSI：基准 D7–13 vs 后两周", loc="left")
    ax.set_xlabel("监控窗口")
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if value >= 0.25 else "#2F2A24",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("PSI（>0.25 严重，0.1–0.25 关注）")
    fig.text(0.01, -0.02, "口径：非信息型可评分券；基准 D7–13（避开冷启动），当前 D14–20 / D21–29。行为历史特征 PSI 高时先查回看窗口，不直接判线上漂移。", fontsize=8.2, color="#666666")
    fig.tight_layout()
    out = FIG / "09_psi_heatmap.png"
    save_fig(fig, out)
    return out


def plot_model_decay(df: pd.DataFrame) -> Path:
    weekly = df[df["period"].isin(PERIODS)].copy()
    weekly["period"] = pd.Categorical(weekly["period"], categories=PERIODS, ordered=True)
    weekly = weekly.sort_values("period")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    ax.plot(weekly["period"], weekly["auc"], marker="o", color="#1D3A6E", linewidth=2.0, label="AUC")
    ax.plot(weekly["period"], weekly["pr_auc"], marker="s", color="#4CC38A", linewidth=2.0, label="PR-AUC")
    for x, y in zip(weekly["period"], weekly["auc"]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_title("按周模型排序能力", loc="left")
    ax.set_ylim(0.70, 0.96)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(weekly["period"], weekly["brier"], marker="o", color="#F08C4A", linewidth=2.0, label="Brier")
    ax.plot(weekly["period"], weekly["logloss"], marker="s", color="#8A6BBE", linewidth=2.0, label="LogLoss")
    for x, y in zip(weekly["period"], weekly["brier"]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_title("按周概率质量", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("模型衰减监控：test split，基准 D7–13", x=0.01, ha="left", fontsize=14)
    fig.text(0.01, -0.02, "口径：只用 test split，避免训练集分数虚高；早期样本量约 2,000/周。", fontsize=8.5, color="#666666")
    fig.tight_layout()
    out = FIG / "09_model_decay.png"
    save_fig(fig, out)
    return out


def plot_coupon_monitor(df: pd.DataFrame) -> Path:
    weekly = df[df["period"].isin(PERIODS)].copy()
    weekly["period"] = pd.Categorical(weekly["period"], categories=PERIODS, ordered=True)
    weekly = weekly.sort_values("period")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    bars = ax.bar(weekly["period"], weekly["net_gmv_per_send"], color="#1D3A6E", width=0.62)
    for b, v in zip(bars, weekly["net_gmv_per_send"]):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8.5)
    ax.set_title("d7 每发送净 GMV", loc="left")
    ax.set_ylabel("元 / 发送")

    ax = axes[1]
    ax.plot(weekly["period"], weekly["redemption_rate"] * 100, marker="o", color="#4CC38A", linewidth=2.2)
    for x, y in zip(weekly["period"], weekly["redemption_rate"] * 100):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_title("d7 核销率", loc="left")
    ax.set_ylabel("%")
    ax.set_ylim(95, 100.8)

    fig.suptitle("d7 券效周报：高价值/中价值全响应人群", x=0.01, ha="left", fontsize=14)
    fig.text(0.01, -0.02, "口径：net_gmv = gmv72 - 3 × 实际核销；每发送均值；基准 D7–13，D0–6 仅作冷启动参考。", fontsize=8.5, color="#666666")
    fig.tight_layout()
    out = FIG / "09_coupon_monitor.png"
    save_fig(fig, out)
    return out


def main() -> None:
    use_style()
    offer = add_period(pd.read_csv(TABLE / "04_offer_dataset.csv"))
    importance = pd.read_csv(TABLE / "04_feature_importance.csv")

    features = [
        f for f in importance["feature"].tolist()
        if f in offer.columns and f not in ("send_day", "send_day.1")
    ][:10]
    # 补上 p_complete 作为 score PSI；features 只保留连续/离散特征
    features = [f for f in features if f != "p_complete"]

    psi_df = compute_psi(offer, features)
    psi_df.to_csv(TABLE / "09_psi.csv", index=False, encoding="utf-8-sig")

    decay = compute_model_decay(offer)
    decay.to_csv(TABLE / "09_model_decay.csv", index=False, encoding="utf-8-sig")

    coupon = compute_coupon_monitor(offer)
    coupon.to_csv(TABLE / "09_coupon_monitor.csv", index=False, encoding="utf-8-sig")

    daily = compute_daily_quality()
    daily.to_csv(TABLE / "09_daily_quality.csv", index=False, encoding="utf-8-sig")

    # 汇总告警：PSI severe/warn、模型 review、券效 review、日事件量 review
    alerts = []
    for _, row in psi_df[psi_df["level"].isin(["warn", "severe"])].iterrows():
        alerts.append({
            "type": "PSI",
            "category": row["category"],
            "object": f"{row['feature_cn']} / {row['period']}",
            "value": row["psi"],
            "level": row["level"],
            "action": "check_lookback" if row["category"] == "behavior_history" else "investigate",
        })
    for _, row in decay[decay["alert"] == "review"].iterrows():
        alerts.append({"type": "MODEL", "category": "model", "object": row["period"], "value": row["auc"], "level": "review", "action": "retrain_or_investigate"})
    for _, row in coupon[coupon["alert"] == "review"].iterrows():
        alerts.append({"type": "COUPON", "category": "coupon", "object": row["period"], "value": row["net_gmv_change_vs_ref"], "level": "review", "action": "check_coupon_mix"})
    for _, row in daily[daily["alert"] == "review"].iterrows():
        alerts.append({"type": "DATA", "category": "data", "object": f"day {int(row['day'])}", "value": row["events_z"], "level": "review", "action": "check_data_pipeline"})
    alert_df = pd.DataFrame(alerts)
    alert_df.to_csv(TABLE / "09_alerts.csv", index=False, encoding="utf-8-sig")

    psi_fig = plot_psi(psi_df)
    decay_fig = plot_model_decay(decay)
    coupon_fig = plot_coupon_monitor(coupon)

    meta = {
        "reference": "send_day 7-13（避开 D0-6 冷启动）",
        "monitor_windows": ["D14-20", "D21-29"],
        "psi_thresholds": {"stable": 0.10, "warn": 0.25},
        "model_eval_split": "test only",
        "note": "历史离线回放，不是生产实时监控。行为历史特征受 30 天窗口左截断影响，PSI 高需先核查回看口径。",
        "figures": [str(psi_fig), str(decay_fig), str(coupon_fig)],
    }
    with open(TABLE / "09_monitor_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("PSI severe/warn:")
    print(psi_df[psi_df["level"].isin(["warn", "severe"])].to_string(index=False))
    print("\nModel decay:")
    print(decay.to_string(index=False))
    print("\nCoupon monitor:")
    print(coupon.to_string(index=False))
    print(f"\nAlerts: {len(alert_df)}")
    print(f"Figures: {psi_fig.name}, {decay_fig.name}, {coupon_fig.name}")


if __name__ == "__main__":
    main()
