# -*- coding: utf-8 -*-
"""A/B 实验样本量与功效计算。

口径：
- 实验单位：用户。历史数据里同一用户会收到多张券，本脚本先按用户聚合，避免把多次曝光当成多个独立样本。
- 对照组：信息型（当前高/中全响应人群的实际主投券型）。
- 处理组：折扣 7-3（d7），补贴 3 元。
- 主指标：72h 净 GMV / 用户 = gmv72 - 3 * redeemed；redeemed 是 d7 实际核销，不是“发券即扣 3 元”。
- 次指标：72h 是否购买（any72）、转化率、ROI。
- 功效：双侧 alpha=0.05，power=80%。

注意：这是用历史观测值做“规划假设”，不是已经做过的线上 A/B。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from config import TABLE

SEGMENTS = ["高价值·全响应型", "中价值·全响应型"]
SUBSIDY_D7 = 3.0
ALPHA = 0.05
POWER = 0.80
Z_ALPHA = NormalDist().inv_cdf(1 - ALPHA / 2)
Z_BETA = NormalDist().inv_cdf(POWER)
K = (Z_ALPHA + Z_BETA) ** 2  # 7.849 for 80% power, two-sided 5%


def required_n_binary(p0: float, p1: float) -> float:
    """两比例检验，每臂样本量。"""
    return K * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2


def power_binary(p0: float, p1: float, n: float) -> float:
    se = math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
    z = abs(p1 - p0) / se
    return NormalDist().cdf(z - Z_ALPHA) + NormalDist().cdf(-z - Z_ALPHA)


def mde_binary(p0: float, n: float) -> float:
    """在给定每臂样本量下，80% power 能检出的最小绝对差。"""
    lo, hi = 1e-9, min(p0, 1 - p0) - 1e-9
    # required_n 随 delta 增大而下降；找最小的 delta 使 required_n <= n
    for _ in range(100):
        mid = (lo + hi) / 2
        if required_n_binary(p0, p0 + mid) > n:
            lo = mid
        else:
            hi = mid
    return hi


def required_n_cont(sd0: float, sd1: float, delta: float) -> float:
    """两独立样本均值检验，每臂样本量。"""
    return K * (sd0 ** 2 + sd1 ** 2) / delta ** 2


def power_cont(sd0: float, sd1: float, delta: float, n: float) -> float:
    se = math.sqrt((sd0 ** 2 + sd1 ** 2) / n)
    z = abs(delta) / se
    return NormalDist().cdf(z - Z_ALPHA) + NormalDist().cdf(-z - Z_ALPHA)


def mde_cont(sd0: float, sd1: float, n: float) -> float:
    return math.sqrt(K * (sd0 ** 2 + sd1 ** 2) / n)


def user_level(df: pd.DataFrame) -> pd.DataFrame:
    """同一用户多次曝光 -> 先按用户求均值，保证实验单位是用户。"""
    return (
        df.groupby(["segment_name", "person"], as_index=False)
        .agg(
            any72=("any72", "mean"),
            gmv72=("gmv72", "mean"),
            redeemed=("y_completed", "mean"),
            n_sends=("any72", "size"),
        )
    )


def stats_for(df_info: pd.DataFrame, df_d7: pd.DataFrame, n_users: int, label: str) -> dict:
    p0 = float(df_info["any72"].mean())
    p1 = float(df_d7["any72"].mean())
    mu0 = float(df_info["gmv72"].mean())
    mu1_net = float((df_d7["gmv72"] - SUBSIDY_D7 * df_d7["redeemed"]).mean())
    sd0 = float(df_info["gmv72"].std(ddof=1))
    sd1_net = float((df_d7["gmv72"] - SUBSIDY_D7 * df_d7["redeemed"]).std(ddof=1))
    redemption = float(df_d7["redeemed"].mean())

    n_per_arm = n_users / 2.0
    delta_conv = p1 - p0
    delta_net = mu1_net - mu0

    mde_conv = mde_binary(p0, n_per_arm)
    mde_net = mde_cont(sd0, sd1_net, n_per_arm)
    power_conv = power_binary(p0, p1, n_per_arm)
    power_net = power_cont(sd0, sd1_net, delta_net, n_per_arm)

    req_conv_5pp = required_n_binary(p0, p0 + 0.05)
    req_net_2_5 = required_n_cont(sd0, sd1_net, 2.5)

    return {
        "segment": label,
        "n_users": n_users,
        "n_per_arm": int(n_users // 2),
        "n_info_users": len(df_info),
        "n_coupon_users": len(df_d7),
        "d7_redemption_rate": round(redemption, 4),
        "p0_info_conv": round(p0, 4),
        "p1_coupon_conv": round(p1, 4),
        "delta_conv_pp": round(delta_conv * 100, 2),
        "mu0_info_gmv": round(mu0, 2),
        "mu1_coupon_net_gmv": round(mu1_net, 2),
        "delta_net_gmv": round(delta_net, 2),
        "sd0_info_gmv": round(sd0, 2),
        "sd1_coupon_net_gmv": round(sd1_net, 2),
        "mde_conv_pp_80": round(mde_conv * 100, 2),
        "mde_net_gmv_80": round(mde_net, 2),
        "power_conv_observed": round(power_conv, 3),
        "power_net_observed": round(power_net, 3),
        "required_n_per_arm_conv_5pp": int(math.ceil(req_conv_5pp)),
        "required_n_per_arm_net_2_5": int(math.ceil(req_net_2_5)),
        "shortfall_users_conv_5pp": max(0, int(math.ceil(2 * req_conv_5pp) - n_users)),
        "shortfall_users_net_2_5": max(0, int(math.ceil(2 * req_net_2_5) - n_users)),
        "conv_5pp_power_ok": bool(req_conv_5pp <= n_per_arm),
        "net_2_5_power_ok": bool(req_net_2_5 <= n_per_arm),
    }


def main() -> None:
    w = pd.read_csv(TABLE / "05_send_windows.csv")
    o = pd.read_csv(
        TABLE / "04_offer_dataset.csv",
        usecols=["person", "offer_id", "send_time", "y_completed"],
    )
    w = w.merge(o, on=["person", "offer_id", "send_time"], how="left")
    w = w[w["segment_name"].isin(SEGMENTS)].copy()
    info = user_level(w[w["coupon"] == 0])
    d7 = user_level(w[(w["offer_group"] == "d7") & w["y_completed"].notna()])

    rows = []
    for seg in SEGMENTS:
        rows.append(
            stats_for(
                info[info["segment_name"] == seg],
                d7[d7["segment_name"] == seg],
                n_users=int(w.loc[w["segment_name"] == seg, "person"].nunique()),
                label=seg,
            )
        )

    rows.append(
        stats_for(
            info,
            d7,
            n_users=int(w["person"].nunique()),
            label="pooled（高全 + 中全）",
        )
    )

    result = pd.DataFrame(rows)
    out = TABLE / "08_ab_power.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")

    meta = {
        "segments": SEGMENTS,
        "subsidy_d7": SUBSIDY_D7,
        "alpha": ALPHA,
        "power": POWER,
        "z_alpha": Z_ALPHA,
        "z_beta": Z_BETA,
        "note": "基于 05_send_windows 与 04_offer_dataset 的用户级历史观测；处理组只取 d7，按实际核销扣 3 元补贴。仅作规划假设，不是线上 A/B 结果。",
    }
    with open(TABLE / "08_ab_power_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(result.round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
