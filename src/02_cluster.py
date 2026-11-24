# -*- coding: utf-8 -*-
"""
02_cluster.py —— 双聚类：价值维 × 行为维，把 1.7 万人切成可运营的人群

思路（对应简历里的「采用双聚类的方法进行用户分层」）：
    第一层 价值维（RFM）：recency / 频次 / 金额 / 客单价 / 活跃天数 → KMeans 三档（高/中/低价值）
    第二层 行为维（响应 + 偏好）：收→看→用率 / 折扣偏好 / 门槛耐受 / 归一化熵
                          → GMM 四档（全响应型 / 选择性响应型 / 零完成型 / 低活跃型）
    两层交叉 → 3×3 共 9 个人群，每个人群都能落到「发不发、发什么、发多大力度」上

为什么不用一个 12 维 KMeans 直接切 9 类：
    单次聚类出来的类无法解释「是价值高还是响应高」，业务上没法下指令；
    两维分开切，每个轴都能单独讲清楚，交叉后人群含义是确定的。

输入：data/clean/user_features.csv
输出：data/clean/user_segments.csv        每个用户的人群归属
      outputs/tables/02_value_profile.csv         价值三档画像
      outputs/tables/02_behavior_profile.csv      行为四档画像
      outputs/tables/02_segment_profile.csv       9 个人群画像
      outputs/figures/02_k_selection.png / 02_value_scatter.png / 02_behavior_pca.png
      outputs/figures/02_behavior_radar.png / 02_segment_matrix.png
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

from utils_viz import use_style, save_fig, VALUE_COLORS, BEHAVIOR_COLORS, SEGMENT_CMAP
from config import CLEAN, TABLE, FIG

OUT = TABLE

K_VALUE = 3          # 价值三档：高 / 中 / 低
K_BEHAVIOR = 4       # 行为四档：全响应 / 选择性响应 / 零完成 / 低活跃
RANDOM_STATE = 42
FONT = use_style()
print(f"[style] 中文字体：{FONT}")


# ---------------------------------------------------------------- 1. 读宽表
user = pd.read_csv(os.path.join(CLEAN, "user_features.csv"))

# ---------------------------------------------------------------- 2. 价值维矩阵
# 金额、频次右偏严重，先 log1p 再标准化，否则 KMeans 会被极少数大额用户带跑
value_df = pd.DataFrame({
    "recency_h": user["recency_h"].fillna(720),                 # 从没消费 = 最久未消费
    "log_n_tx": np.log1p(user["n_tx"]),
    "log_monetary": np.log1p(user["monetary"]),
    "log_aov": np.log1p(user["avg_order_value"].fillna(0)),
    "active_day_ratio": user["active_day_ratio"].fillna(0),
}, index=user.index)
value_scaled = StandardScaler().fit_transform(value_df)

# ---------------------------------------------------------------- 3. 价值维 k 选择
k_range = [2, 3, 4, 5, 6]
value_sil, value_ch = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(value_scaled)
    value_sil.append(silhouette_score(value_scaled, km.labels_, sample_size=6000, random_state=RANDOM_STATE))
    value_ch.append(calinski_harabasz_score(value_scaled, km.labels_))

# 业务上需要「高/中/低」三档，k 固定为 3；k 曲线用来证明 3 处于平台区，不是拍脑袋
km_value = KMeans(n_clusters=K_VALUE, n_init=20, random_state=RANDOM_STATE).fit(value_scaled)
user["value_cluster"] = km_value.labels_

# 按消费中位数给簇贴「高/中/低价值」标签（升序 → 低、中、高）
value_prof = user.groupby("value_cluster").agg(
    n=("user_id", "size"),
    monetary_median=("monetary", "median"),
    n_tx_median=("n_tx", "median"),
    recency_median=("recency_h", "median"),
    active_day_ratio=("active_day_ratio", "mean"),
).sort_values("monetary_median")
value_labels = {cid: lab for cid, lab in zip(value_prof.index, ["低价值", "中价值", "高价值"])}
user["value_label"] = user["value_cluster"].map(value_labels)
value_prof["label"] = value_prof.index.map(value_labels)


# ---------------------------------------------------------------- 4. 行为维矩阵
# 只对「收到过券」的 16,994 人做行为聚类；6 个从未被触达的用户单列，不硬塞进人群
# 重要修正：直接用 hour_entropy / day_entropy 会把「交易笔数多」误当成「消费分散」
#   （熵与交易笔数的相关系数：时段 0.60、日期 0.88）。
#   这里改成「归一化熵」= 熵 / log2(可观测档位数)，抵消样本量带来的偏差。
behavior_ok = user["n_offer_recv"] > 0
behavior_df = user.loc[behavior_ok, [
    "view_rate", "comp_rate", "pref_discount", "hard_minus_easy",
    "hour_entropy", "day_entropy", "n_tx",
]].copy()

# 缺失口径：
behavior_df["view_rate"] = behavior_df["view_rate"].fillna(behavior_df["view_rate"].median())
behavior_df["comp_rate"] = behavior_df["comp_rate"].fillna(behavior_df["comp_rate"].median())
# 只收到一种券型时比不出偏好 → 记 0（无差异证据），而不是当缺失丢掉这个人
behavior_df["pref_discount"] = behavior_df["pref_discount"].fillna(0)
behavior_df["hard_minus_easy"] = behavior_df["hard_minus_easy"].fillna(0)

n_eff = behavior_df["n_tx"].clip(lower=1)
behavior_df["hour_entropy_norm"] = (behavior_df["hour_entropy"] /
                                    np.log2(np.minimum(n_eff, 4).clip(lower=2))).fillna(0)
behavior_df["day_entropy_norm"] = (behavior_df["day_entropy"] /
                                   np.log2(np.minimum(n_eff, 30).clip(lower=2))).fillna(0)

# 注意：comp_rate 已修正为「仅可完成券」的口径（信息型券不进分母），见 01 脚本注释
behavior_feats = ["comp_rate", "view_rate", "pref_discount", "hard_minus_easy",
                  "hour_entropy_norm", "day_entropy_norm"]
behavior_scaled = StandardScaler().fit_transform(behavior_df[behavior_feats])

# ---------------------------------------------------------------- 5. 行为维 k 选择 + GMM
b_sil, b_bic = [], []
for k in [2, 3, 4, 5]:
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         random_state=RANDOM_STATE, n_init=3).fit(behavior_scaled)
    lab = gm.predict(behavior_scaled)
    b_sil.append(silhouette_score(behavior_scaled, lab, sample_size=6000, random_state=RANDOM_STATE))
    b_bic.append(gm.bic(behavior_scaled))

gmm = GaussianMixture(n_components=K_BEHAVIOR, covariance_type="full",
                      random_state=RANDOM_STATE, n_init=5).fit(behavior_scaled)
user.loc[behavior_ok, "behavior_cluster"] = gmm.predict(behavior_scaled)
user["behavior_cluster"] = user["behavior_cluster"].astype("Int64")
user.loc[behavior_ok, "hour_entropy_norm"] = behavior_df["hour_entropy_norm"]
user.loc[behavior_ok, "day_entropy_norm"] = behavior_df["day_entropy_norm"]

# 行为簇画像（原始量纲，方便解释）
behavior_prof = user.loc[behavior_ok].groupby("behavior_cluster").agg(
    n=("user_id", "size"),
    view_rate=("view_rate", "mean"),
    comp_rate=("comp_rate", "mean"),
    comp_rate_bogo=("comp_rate_bogo", "mean"),
    comp_rate_discount=("comp_rate_discount", "mean"),
    pref_discount=("pref_discount", "mean"),
    hard_minus_easy=("hard_minus_easy", "mean"),
    hour_entropy_norm=("hour_entropy_norm", "mean"),
    day_entropy_norm=("day_entropy_norm", "mean"),
    n_tx_median=("n_tx", "median"),
    monetary_median=("monetary", "median"),
)

# 标签规则（按簇的实际画像命名）：
#   完成率≈100%            → 全响应型（券必用，补贴浪费审计对象）
#   完成率≈0 且有消费      → 零完成型（只看不用，触达/说服问题）
#   几乎无交易（中位为 0）  → 低活跃型（沉睡/未激活）
#   其余                   → 选择性响应型（约一半券会完成，主力增量池）
b_labels = {}
for cid in behavior_prof.index[behavior_prof["comp_rate"] >= 0.99]:
    b_labels[cid] = "全响应型"
for cid in behavior_prof.index[(behavior_prof["comp_rate"] <= 0.005) & (~behavior_prof.index.isin(b_labels))]:
    b_labels[cid] = "零完成型"
for cid in behavior_prof.index:
    if cid in b_labels:
        continue
    if behavior_prof.loc[cid, "n_tx_median"] <= 0 or behavior_prof.loc[cid, "monetary_median"] <= 0:
        b_labels[cid] = "低活跃型"
    else:
        b_labels[cid] = "选择性响应型"
user["behavior_label"] = user["behavior_cluster"].map(b_labels)
user.loc[~behavior_ok, "behavior_label"] = "未触达"
behavior_prof["label"] = behavior_prof.index.map(b_labels)
behavior_prof = behavior_prof.sort_values("comp_rate", ascending=False)


# ---------------------------------------------------------------- 6. 交叉成人群
def seg_code(row):
    if pd.isna(row["behavior_cluster"]):
        return f"V{ {'低价值':3,'中价值':2,'高价值':1}[row['value_label']] }-NA"
    v = {"高价值": 1, "中价值": 2, "低价值": 3}[row["value_label"]]
    b = {"全响应型": 1, "选择性响应型": 2, "零完成型": 3, "低活跃型": 4}[row["behavior_label"]]
    return f"V{v}B{b}"


user["segment_code"] = user.apply(seg_code, axis=1)
user["segment_name"] = np.where(
    user["behavior_label"] == "未触达",
    user["value_label"] + "·未触达",
    user["value_label"] + "·" + user["behavior_label"],
)

# 9 个人群的画像（排除 6 个未触达）
seg = user[user["behavior_label"] != "未触达"].copy()
seg_profile = seg.groupby(["value_label", "behavior_label"]).agg(
    n=("user_id", "size"),
    share=("user_id", lambda s: len(s) / len(user)),
    monetary_median=("monetary", "median"),
    n_tx_median=("n_tx", "median"),
    recency_median=("recency_h", "median"),
    view_rate=("view_rate", "mean"),
    comp_rate=("comp_rate", "mean"),
    comp_rate_bogo=("comp_rate_bogo", "mean"),
    comp_rate_discount=("comp_rate_discount", "mean"),
    pref_discount=("pref_discount", "mean"),
    total_reward_mean=("total_reward_earned", "mean"),
).reset_index()

user[["user_id", "value_cluster", "value_label", "behavior_cluster", "behavior_label",
      "segment_code", "segment_name"]].to_csv(
    os.path.join(CLEAN, "user_segments.csv"), index=False, encoding="utf-8-sig")
value_prof.to_csv(os.path.join(OUT, "02_value_profile.csv"), encoding="utf-8-sig")
behavior_prof.to_csv(os.path.join(OUT, "02_behavior_profile.csv"), encoding="utf-8-sig")
seg_profile.to_csv(os.path.join(OUT, "02_segment_profile.csv"), index=False, encoding="utf-8-sig")

print(f"\n[OK] user_segments.csv  {user.shape[0]} 人，{seg_profile.shape[0]} 个人群")
print("\n价值三档：")
print(value_prof.round(2).to_string())
print("\n行为四档：")
print(behavior_prof.round(3).to_string())
print("\n人群规模：")
print(seg_profile.assign(share_pct=(seg_profile["share"] * 100).round(1))[
    ["value_label", "behavior_label", "n", "share_pct", "comp_rate", "pref_discount"]
].round(3).to_string(index=False))


# ---------------------------------------------------------------- 7. 图表
# 7.1 k 选择曲线
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
for ax, ks, sils, title, chosen in [
    (axes[0], k_range, value_sil, "价值维（RFM）", K_VALUE),
    (axes[1], [2, 3, 4, 5], b_sil, "行为维（响应/偏好）", K_BEHAVIOR),
]:
    colors = ["#1D3A6E" if k == chosen else "#C9D6EA" for k in ks]
    bars = ax.bar([str(k) for k in ks], sils, color=colors, width=0.62)
    for b, s in zip(bars, sils):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.006, f"{s:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{title}：k 选择", pad=10)
    ax.set_xlabel("簇数 k")
    ax.set_ylabel("轮廓系数（越高越分得开）")
    ax.set_ylim(0, max(sils) * 1.18)
axes[0].text(0.98, 0.02, "k=3 与 k=4 仅差 0.02，取三档便于业务落地", transform=axes[0].transAxes,
             ha="right", va="bottom", fontsize=8.5, color="#555555")
axes[1].text(0.98, 0.02, "k=4：全响应 / 选择性 / 零完成 / 低活跃", transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=8.5, color="#555555")
fig.suptitle("双聚类：价值三档 × 行为四档", fontsize=15, fontweight="bold")
save_fig(fig, os.path.join(FIG, "02_k_selection.png"))

# 7.2 价值维散点
fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
for lab in ["低价值", "中价值", "高价值"]:
    m = user["value_label"] == lab
    ax.scatter(np.log1p(user.loc[m, "monetary"]), user.loc[m, "recency_h"],
               s=16, alpha=0.45, color=VALUE_COLORS[lab], label=f"{lab}（{m.sum():,} 人）",
               edgecolors="none")
for lab in ["低价值", "中价值", "高价值"]:
    m = user["value_label"] == lab
    cx = np.log1p(user.loc[m, "monetary"]).mean()
    cy = user.loc[m, "recency_h"].mean()
    ax.scatter([cx], [cy], s=280, marker="*", color=VALUE_COLORS[lab],
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate(lab, (cx, cy), xytext=(8, 8), textcoords="offset points",
                fontsize=10, fontweight="bold", color=VALUE_COLORS[lab],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
ax.set_xlabel("消费总额（log(1+x) 变换）")
ax.set_ylabel("距最近一次消费（小时，越小越近）")
ax.set_title("价值维：三档人群在「金额—近度」平面上的位置", pad=10)
ax.legend(loc="upper right", markerscale=1.6)
save_fig(fig, os.path.join(FIG, "02_value_scatter.png"))

# 7.3 行为维：关键指标分布
# 说明：comp_rate 高度离散（全响应簇恒为 1.0），直接做 PCA 散点会被拉成直线，
#      改用箱线图看完成率分布 + 分组柱看偏好差异，信息量更大也更诚实。
order_labs = [lab for lab in ["全响应型", "选择性响应型", "零完成型", "低活跃型"]
              if lab in set(b_labels.values())]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), constrained_layout=True)

comp_data = [user.loc[behavior_ok & (user["behavior_label"] == lab), "comp_rate"].dropna().values
             for lab in order_labs]
bp = axes[0].boxplot(comp_data, patch_artist=True, widths=0.52, showfliers=False,
                     medianprops=dict(color="#333333", linewidth=1.6))
for patch, lab in zip(bp["boxes"], order_labs):
    patch.set_facecolor(BEHAVIOR_COLORS.get(lab, "#999999"))
    patch.set_alpha(0.75)
    patch.set_edgecolor("#444444")
    patch.set_linewidth(0.8)
for i, lab in enumerate(order_labs):
    axes[0].text(i + 1, 1.05, f"均值 {np.mean(comp_data[i]):.0%}", ha="center", va="bottom",
                 fontsize=9, color="#333333")
axes[0].set_xticks(range(1, len(order_labs) + 1))
axes[0].set_xticklabels(order_labs)
axes[0].set_ylim(0, 1.12)
axes[0].set_ylabel("券完成率（仅可完成券）")
axes[0].set_title("完成率分布：全响应簇恒为 100%", pad=10)

x = np.arange(len(order_labs))
w = 0.36
prof_idx = behavior_prof.set_index("label").reindex(order_labs)
axes[1].bar(x - w / 2, prof_idx["pref_discount"], w, color="#F08C4A", label="折扣偏好（折扣−BOGO 完成率）")
axes[1].bar(x + w / 2, prof_idx["hard_minus_easy"], w, color="#5B8FF9", label="门槛耐受（高门槛−低门槛）")
axes[1].axhline(0, color="#888888", linewidth=0.9)
for xi, v in zip(x - w / 2, prof_idx["pref_discount"]):
    axes[1].text(xi, v + (0.018 if v >= 0 else -0.018), f"{v:.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8.5)
for xi, v in zip(x + w / 2, prof_idx["hard_minus_easy"]):
    axes[1].text(xi, v + (0.018 if v >= 0 else -0.018), f"{v:.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(order_labs)
axes[1].set_ylim(-0.45, 0.30)
axes[1].set_ylabel("均值（正 = 更偏好 / 更耐受）")
axes[1].set_title("偏好差异：选择性响应型最吃折扣", pad=10)
axes[1].legend(loc="upper left", fontsize=8.5)
fig.suptitle("行为四档：完成率分布与偏好差异", fontsize=15, fontweight="bold")
save_fig(fig, os.path.join(FIG, "02_behavior_profile.png"))

# 7.4 人群矩阵
order_value = ["高价值", "中价值", "低价值"]
order_behavior = [b for b in ["全响应型", "选择性响应型", "零完成型", "低活跃型"]
                  if b in seg_profile["behavior_label"].unique()]
pivot_comp = seg_profile.pivot(index="value_label", columns="behavior_label", values="comp_rate").reindex(
    index=order_value, columns=order_behavior)
pivot_n = seg_profile.pivot(index="value_label", columns="behavior_label", values="n").reindex(
    index=order_value, columns=order_behavior)
pivot_share = seg_profile.pivot(index="value_label", columns="behavior_label", values="share").reindex(
    index=order_value, columns=order_behavior)

arr = pivot_comp.values.astype(float)
fig, ax = plt.subplots(figsize=(9.6, 6.8), constrained_layout=True)
im = ax.imshow(np.ma.masked_invalid(arr), cmap=SEGMENT_CMAP, vmin=0.0, vmax=1.0, aspect="auto")

# 没有该组合的人群 → 浅灰底 + 「无此人群」，不能显示 nan
from matplotlib.patches import Rectangle
for i, j in zip(*np.where(np.isnan(arr))):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F4F4F4",
                           edgecolor="white", linewidth=2, zorder=2))
for i in range(len(order_value)):
    for j in range(len(order_behavior)):
        n = pivot_n.values[i, j]
        share = pivot_share.values[i, j] * 100
        comp = arr[i, j]
        if np.isnan(comp):
            ax.text(j, i, f"{order_value[i]}·{order_behavior[j]}\n无此人群",
                    ha="center", va="center", fontsize=9.5, color="#999999", linespacing=1.5)
            continue
        text_color = "white" if comp > 0.55 else "#1A1A1A"
        ax.text(j, i, f"{order_value[i]}·{order_behavior[j]}\n{n:,.0f} 人 · {share:.1f}%\n完成率 {comp:.0%}",
                ha="center", va="center", fontsize=9.5, color=text_color, linespacing=1.5)

# 白色分隔线，让格子边界清楚
for k in range(1, len(order_value)):
    ax.axhline(k - 0.5, color="white", linewidth=2, zorder=3)
for k in range(1, len(order_behavior)):
    ax.axvline(k - 0.5, color="white", linewidth=2, zorder=3)

ax.set_xticks(range(len(order_behavior)))
ax.set_xticklabels(order_behavior, fontsize=11)
ax.set_yticks(range(len(order_value)))
ax.set_yticklabels(order_value, fontsize=11)
ax.set_xlabel("行为维（响应与偏好）", labelpad=8)
ax.set_ylabel("价值维（RFM）", labelpad=8)
ax.set_title(f"人群矩阵：{len(seg_profile)} 个实际人群与券完成率", pad=12)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("人均券完成率", fontsize=9.5)
save_fig(fig, os.path.join(FIG, "02_segment_matrix.png"))

print("\n[OK] 图表已输出到 outputs/figures/：02_k_selection / 02_value_scatter / 02_behavior_profile / 02_segment_matrix")
