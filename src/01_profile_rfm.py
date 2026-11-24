# -*- coding: utf-8 -*-
"""
01_profile_rfm.py —— 从事件流水构建「用户级特征宽表」，作为后续双聚类的输入

输入（上一步 00_parse_json.py 产出）：
    data/clean/portfolio.csv    10 种券配置
    data/clean/profile.csv      17,000 个用户画像
    data/clean/transcript.csv   306,534 条事件

输出：
    data/clean/user_features.csv   17,000 行 × 52 列（一个用户一行）
    outputs/tables/01_feature_stats.csv    所有特征的描述统计，方便自查

特征分成 6 组，对应简历项目里的三个问题：
    A 价值维度  → 「给谁发」（RFM：最近一次、频次、金额）
    B 探索维度  → 「给谁发」（消费时段分散度、渠道广度、券型尝鲜）
    C 触达响应  → 「发完是否有效」（收→看→用的漏斗率）
    D 券型偏好  → 「给谁发什么」（BOGO / 折扣 / 信息型分别的完成率）
    E 门槛耐受  → 「发什么力度」（高门槛 vs 低门槛券的完成率差）
    F 人口属性  → 辅助分层（年龄、收入、会员时长）

口径说明（踩坑点）：
1. 活动窗口是 30 天（time 0~714 小时），recency 用 720 - 最近一次交易时间。
2. 同一用户同一张券会重复收到（约 86.5% 的用户×券组合出现多次），
   所以这里的收/看/用都是「事件计数」，不是去重计数——这正好对应"触达次数"的业务含义。
3. 信息型券（informational）没有 reward、也不会被 completed，
   只参与"查看率"，不参与完成率，避免出现假指标。
4. age=118 / income 缺失的 2,175 个用户保留 NaN，另给 demo_missing 标记，
   由后面聚类时再决定插补还是单独成簇。
5. 有 6 个用户没有任何券事件，其券类特征为 NaN（不是 0），
   避免把"没被触达"和"触达了没反应"混为一谈。
"""

import os
import numpy as np
import pandas as pd

from config import CLEAN, TABLE

OUT = TABLE

WINDOW_HOURS = 720          # 活动窗口 30 天
HARD_DIFFICULTY = 10        # 门槛阈值：券的 difficulty 中位数（5/7/10/20 中位数=10）


# ---------------------------------------------------------------- 工具函数
def row_entropy(count_df):
    """按行计算香农熵（单位 bit），用于衡量行为的分散程度"""
    total = count_df.sum(axis=1)
    p = count_df.div(total.replace(0, np.nan), axis=0)
    ent = -(p.where(p > 0) * np.log2(p.where(p > 0))).sum(axis=1)
    return ent


# ---------------------------------------------------------------- 1. 读数据
portfolio = pd.read_csv(os.path.join(CLEAN, "portfolio.csv"))
profile = pd.read_csv(os.path.join(CLEAN, "profile.csv"), parse_dates=["became_member_on"])
transcript = pd.read_csv(os.path.join(CLEAN, "transcript.csv"))

USER_IDS = profile["id"].unique()

tx = transcript[transcript["event"] == "transaction"].copy()
tx["hour"] = (tx["time"] % 24).astype(int)
oe = transcript[transcript["event"] != "transaction"].copy()

# 券配置并进来（reward 与 transcript 的 reward 重名，改名区分）
offer_meta = portfolio.rename(columns={"id": "offer_id", "reward": "offer_reward"})
oe = oe.merge(offer_meta[["offer_id", "offer_type", "difficulty", "offer_reward",
                          "ch_email", "ch_mobile", "ch_social"]], on="offer_id", how="left")


# ---------------------------------------------------------------- A. 价值维度 R/F/M
tx_agg = tx.groupby("person").agg(
    n_tx=("amount", "size"),
    monetary=("amount", "sum"),
    avg_order_value=("amount", "mean"),
    max_order_value=("amount", "max"),
    first_tx_h=("time", "min"),
    last_tx_h=("time", "max"),
    n_active_days=("day", "nunique"),
)
tx_agg["recency_h"] = WINDOW_HOURS - tx_agg["last_tx_h"]          # 距最近一次消费的小时数
tx_agg["tx_per_active_day"] = tx_agg["n_tx"] / tx_agg["n_active_days"]
tx_agg["active_day_ratio"] = tx_agg["n_active_days"] / 30.0       # 30 天里有几天在消费

# 时段分散度（探索偏好代理指标 1）
hour_counts = pd.crosstab(tx["person"], tx["hour"])
hour_entropy = row_entropy(hour_counts).rename("hour_entropy")
night_mask = [c for c in hour_counts.columns if 0 <= c <= 5]
night_share = (hour_counts[night_mask].sum(axis=1) / hour_counts.sum(axis=1)).rename("night_share")

# 消费日期分散度（探索偏好代理指标 2）
day_counts = pd.crosstab(tx["person"], tx["day"])
day_entropy = row_entropy(day_counts).rename("day_entropy")

value_feats = pd.concat([tx_agg, hour_entropy, night_share, day_entropy], axis=1)


# ---------------------------------------------------------------- B/C/D/E. 券维度
# 券事件按「券型 × 事件」计数
oe["key"] = oe["offer_type"] + "|" + oe["event"].str.replace("offer ", "", regex=False)
offer_counts = pd.crosstab(oe["person"], oe["key"]).reindex(USER_IDS, fill_value=0)

expected_keys = [f"{t}|{e}" for t in ["bogo", "discount", "informational"]
                 for e in ["received", "viewed", "completed"]]
for k in expected_keys:
    if k not in offer_counts.columns:
        offer_counts[k] = 0
offer_counts = offer_counts[expected_keys]


def rate(num, den):
    """分母为 0 时返回 NaN —— 没触达 ≠ 触达后没反应"""
    return (num / den.replace(0, np.nan)).astype(float)


offer_feats = pd.DataFrame(index=offer_counts.index)
# C 触达响应（整体漏斗）
offer_feats["n_offer_recv"] = offer_counts["bogo|received"] + offer_counts["discount|received"] + offer_counts["informational|received"]
offer_feats["n_offer_view"] = offer_counts["bogo|viewed"] + offer_counts["discount|viewed"] + offer_counts["informational|viewed"]
offer_feats["n_offer_comp"] = offer_counts["bogo|completed"] + offer_counts["discount|completed"]

recv_all = offer_feats["n_offer_recv"]
view_all = offer_feats["n_offer_view"]
comp_all = offer_feats["n_offer_comp"]

# 关键口径：信息型券不可能被"完成"，如果算进完成率分母会系统性压低指标。
# 所以完成率只在「可完成券」= BOGO + 折扣 上计算，信息型券单独用 view_rate_info 衡量。
recv_completable = offer_counts["bogo|received"] + offer_counts["discount|received"]
view_completable = offer_counts["bogo|viewed"] + offer_counts["discount|viewed"]

offer_feats["view_rate"] = rate(view_all, recv_all)                    # 收→看（衡量注意力）
offer_feats["comp_rate"] = rate(comp_all, recv_completable)            # 收→用（仅可完成券）
offer_feats["view_to_comp_rate"] = rate(comp_all, view_completable)    # 看→用（仅可完成券）
offer_feats["unviewed_rate"] = 1 - offer_feats["view_rate"]            # 触达不足
offer_feats["viewed_not_comp_rate"] = (1 - offer_feats["view_to_comp_rate"]).clip(lower=0)  # 看了不买
offer_feats["info_share"] = rate(offer_counts["informational|received"], recv_all)  # 信息型券在触达里的占比

# D 券型偏好
offer_feats["recv_bogo"] = offer_counts["bogo|received"]
offer_feats["comp_bogo"] = offer_counts["bogo|completed"]
offer_feats["comp_rate_bogo"] = rate(offer_feats["comp_bogo"], offer_feats["recv_bogo"])

offer_feats["recv_discount"] = offer_counts["discount|received"]
offer_feats["comp_discount"] = offer_counts["discount|completed"]
offer_feats["comp_rate_discount"] = rate(offer_feats["comp_discount"], offer_feats["recv_discount"])

offer_feats["recv_info"] = offer_counts["informational|received"]
offer_feats["view_info"] = offer_counts["informational|viewed"]
offer_feats["view_rate_info"] = rate(offer_feats["view_info"], offer_feats["recv_info"])

# 折扣偏好强度 = 折扣券完成率 − BOGO 完成率（价格敏感度的核心代理）
offer_feats["pref_discount"] = offer_feats["comp_rate_discount"] - offer_feats["comp_rate_bogo"]

# E 门槛耐受：高门槛（difficulty>=10）vs 低门槛（5/7）券的完成率差
he = oe[oe["event"].isin(["offer received", "offer completed"]) & (oe["offer_type"] != "informational")].copy()
he["level"] = np.where(he["difficulty"] >= HARD_DIFFICULTY, "hard", "easy")
he["level_key"] = he["level"] + "|" + he["event"].str.replace("offer ", "", regex=False)
he_counts = pd.crosstab(he["person"], he["level_key"]).reindex(USER_IDS, fill_value=0)
for k in ["hard|received", "hard|completed", "easy|received", "easy|completed"]:
    if k not in he_counts.columns:
        he_counts[k] = 0

offer_feats["recv_hard"] = he_counts["hard|received"]
offer_feats["comp_hard"] = he_counts["hard|completed"]
offer_feats["comp_rate_hard"] = rate(offer_feats["comp_hard"], offer_feats["recv_hard"])
offer_feats["recv_easy"] = he_counts["easy|received"]
offer_feats["comp_easy"] = he_counts["easy|completed"]
offer_feats["comp_rate_easy"] = rate(offer_feats["comp_easy"], offer_feats["recv_easy"])
offer_feats["hard_minus_easy"] = offer_feats["comp_rate_hard"] - offer_feats["comp_rate_easy"]

# 收到券的平均让利强度 reward/difficulty（信息型券让利为 0，不参与）
recv_only = oe[(oe["event"] == "offer received") & (oe["difficulty"] > 0)].copy()
recv_only["ratio"] = recv_only["offer_reward"] / recv_only["difficulty"]
offer_feats["avg_offer_ratio"] = recv_only.groupby("person")["ratio"].mean().reindex(USER_IDS)

# 已领取的补贴总额（完成券拿到的 reward）
offer_feats["total_reward_earned"] = (transcript.loc[transcript["event"] == "offer completed", "reward"]
                                      .groupby(transcript.loc[transcript["event"] == "offer completed", "person"])
                                      .sum().reindex(USER_IDS).fillna(0))

# 渠道广度（探索偏好代理指标 3）：收到的券覆盖了几种触达渠道
recv_ch = oe[oe["event"] == "offer received"]
channel_breadth = (recv_ch.groupby("person")[["ch_email", "ch_mobile", "ch_social"]]
                   .max().sum(axis=1)).reindex(USER_IDS)
offer_feats["channel_breadth"] = channel_breadth

# 收到几种券型（1~3）
offer_feats["n_offer_types_recv"] = ((offer_feats["recv_bogo"] > 0).astype(int)
                                     + (offer_feats["recv_discount"] > 0).astype(int)
                                     + (offer_feats["recv_info"] > 0).astype(int))

# B 标签（后面响应模型要用）
offer_feats["y_comp_any"] = (offer_feats["n_offer_comp"] > 0).astype(int)
offer_feats["y_view_any"] = (offer_feats["n_offer_view"] > 0).astype(int)


# ---------------------------------------------------------------- F. 人口属性
profile_feats = profile.rename(columns={"id": "person"}).set_index("person")
profile_feats["demo_missing"] = profile_feats["age"].isna().astype(int)
# 会员时长：数据集没有绝对活动日期，用「最晚注册日 − 本人注册日」作相对新老
profile_feats["tenure_days"] = (profile_feats["became_member_on"].max()
                                - profile_feats["became_member_on"]).dt.days
profile_feats = profile_feats.drop(columns=["became_member_on"])


# ---------------------------------------------------------------- 合并 & 收尾
user = (profile_feats
        .join(value_feats, how="left")
        .join(offer_feats, how="left"))

user["y_tx_any"] = (user["n_tx"].fillna(0) > 0).astype(int)
user["n_tx"] = user["n_tx"].fillna(0).astype(int)
user["monetary"] = user["monetary"].fillna(0.0)
user["n_active_days"] = user["n_active_days"].fillna(0).astype(int)
user["active_day_ratio"] = user["active_day_ratio"].fillna(0.0)

# 事件计数类：没有券事件 = 0 次触达
count_cols = ["n_offer_recv", "n_offer_view", "n_offer_comp", "recv_bogo", "comp_bogo",
              "recv_discount", "comp_discount", "recv_info", "view_info",
              "recv_hard", "comp_hard", "recv_easy", "comp_easy",
              "n_offer_types_recv", "channel_breadth"]
user[count_cols] = user[count_cols].fillna(0).astype(int)

user = user.reset_index().rename(columns={"person": "user_id"})

col_order = ["user_id",
             "recency_h", "n_tx", "monetary", "avg_order_value", "max_order_value",
             "n_active_days", "tx_per_active_day", "active_day_ratio", "first_tx_h", "last_tx_h",
             "hour_entropy", "day_entropy", "night_share", "channel_breadth", "n_offer_types_recv",
             "n_offer_recv", "n_offer_view", "n_offer_comp",
             "view_rate", "comp_rate", "view_to_comp_rate", "unviewed_rate", "viewed_not_comp_rate",
             "info_share",
             "recv_bogo", "comp_bogo", "comp_rate_bogo",
             "recv_discount", "comp_discount", "comp_rate_discount",
             "recv_info", "view_info", "view_rate_info", "pref_discount",
             "recv_hard", "comp_hard", "comp_rate_hard", "recv_easy", "comp_easy", "comp_rate_easy",
             "hard_minus_easy", "avg_offer_ratio", "total_reward_earned",
             "age", "income", "gender", "tenure_days", "demo_missing",
             "y_comp_any", "y_view_any", "y_tx_any"]
user = user[col_order]

user.to_csv(os.path.join(CLEAN, "user_features.csv"), index=False, encoding="utf-8-sig")
user.describe().T.to_csv(os.path.join(OUT, "01_feature_stats.csv"), encoding="utf-8-sig")

# ---------------------------------------------------------------- 体检报告
print(f"[OK] user_features.csv  {user.shape[0]} 行 × {user.shape[1]} 列")
print("\n缺失率最高的 10 个特征：")
print((user.isna().mean().sort_values(ascending=False).head(10) * 100).round(1).to_string())
print("\n关键特征分布：")
key = ["n_tx", "monetary", "view_rate", "comp_rate", "comp_rate_bogo",
       "comp_rate_discount", "pref_discount", "hour_entropy", "channel_breadth"]
print(user[key].describe().T.round(3).to_string())
print("\n标签占比：")
print(f"  有过消费 y_tx_any      : {user['y_tx_any'].mean():.1%}")
print(f"  看过券   y_view_any    : {user['y_view_any'].mean():.1%}")
print(f"  完成过券 y_comp_any    : {user['y_comp_any'].mean():.1%}")
print(f"  无任何券事件用户       : {(user['n_offer_recv'] == 0).sum()}")
