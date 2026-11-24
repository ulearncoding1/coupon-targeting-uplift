# -*- coding: utf-8 -*-
"""
03_strategy.py —— 策略矩阵与预算模拟

目标：
1. 现状审计：每个细分人群实际收到多少券、完成多少、补贴多少、换来多少 GMV、ROI 多少；
2. 策略矩阵：9 个人群 × 6 类券的「触达→完成率」和「每千次触达净 GMV」，看清谁该发什么；
3. 预算模拟：S0 现状 vs S1 优化（保持触达量、切换券型结构），算补贴、GMV、净额；
4. 输出 CSV / 图表 / 策略文档所需的数据底座。

关键口径（写进文档，面试要能讲）：
- 补贴 = 完成券对应的 reward 之和；
- GMV = 完成券对应的 difficulty 之和（即「门槛口径」成交额，和 01/02 保持一致）；
- 账面净额 = GMV − 补贴；
- 增量净额（保守）= 只把「选择性响应型」的 GMV 视为增量，其余人群的 GMV 记为 0（他们本来就会买 / 根本没买）。
  S1 只对选择性人群发可完成券，因此 S1 的账面净额 = 增量净额。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils_viz import use_style, save_fig, VALUE_COLORS
from config import CLEAN, TABLE, FIG

DATA = CLEAN
OUT = TABLE
use_style()

# ------------------------------------------------------------------
# 1. 读取数据
# ------------------------------------------------------------------
portfolio = pd.read_csv(os.path.join(DATA, "portfolio.csv"))
trans = pd.read_csv(os.path.join(DATA, "transcript.csv"))
seg = pd.read_csv(os.path.join(DATA, "user_segments.csv"))

# 券种归并：同一经济结构的券合并（4 个 BOGO 实际只有 5/5 和 10/10 两档）
def offer_group(row):
    if row.offer_type == "informational":
        return "信息型"
    if row.offer_type == "bogo":
        return f"BOGO {int(row.difficulty)}/{int(row.reward)}"
    return f"折扣 {int(row.difficulty)}-{int(row.reward)}"

portfolio["offer_group"] = portfolio.apply(offer_group, axis=1)
portfolio["offer_short"] = portfolio["offer_group"].replace({
    "BOGO 5/5": "BOGO\n5/5",
    "BOGO 10/10": "BOGO\n10/10",
    "折扣 7-3": "折扣\n7-3",
    "折扣 10-2": "折扣\n10-2",
    "折扣 20-5": "折扣\n20-5",
    "信息型": "信息型\n0 成本",
})

GROUP_ORDER = ["BOGO 5/5", "BOGO 10/10", "折扣 7-3", "折扣 10-2", "折扣 20-5", "信息型"]
GROUP_SHORT = {g: g.replace("BOGO ", "BOGO\n").replace("折扣 ", "折扣\n").replace("信息型", "信息型\n0 成本") for g in GROUP_ORDER}

# 事件表合并券属性 + 人群标签（避免与 transcript 自带的 reward 列重名）
pmeta = portfolio[["id", "offer_type", "difficulty", "reward", "offer_group"]].rename(
    columns={"difficulty": "offer_difficulty", "reward": "offer_reward"}
)
recv = trans[trans.event == "offer received"].merge(
    pmeta, left_on="offer_id", right_on="id", how="left"
)
comp = trans[trans.event == "offer completed"].merge(
    pmeta, left_on="offer_id", right_on="id", how="left"
)
recv = recv.merge(seg[["user_id", "value_label", "behavior_label", "segment_name"]],
                  left_on="person", right_on="user_id", how="left")
comp = comp.merge(seg[["user_id", "value_label", "behavior_label", "segment_name"]],
                  left_on="person", right_on="user_id", how="left")

print(f"[1/6] 数据读取完成：触达 {len(recv):,} 条，完成 {len(comp):,} 条，人群 {seg.segment_name.nunique()} 个")

# ------------------------------------------------------------------
# 2. 现状审计
# ------------------------------------------------------------------
users = seg.groupby("segment_name")["user_id"].nunique().rename("users")
sends = recv.groupby("segment_name").size().rename("sends")
comps = comp.groupby("segment_name").size().rename("completions")
subsidy = comp.groupby("segment_name")["offer_reward"].sum().rename("subsidy")
gmv = comp.groupby("segment_name")["offer_difficulty"].sum().rename("gmv")

audit = pd.concat([users, sends, comps, subsidy, gmv], axis=1).fillna(0)
audit["send_per_user"] = audit["sends"] / audit["users"]
audit["comp_rate"] = audit["completions"] / audit["sends"].replace(0, np.nan)
audit["subsidy_per_comp"] = audit["subsidy"] / audit["completions"].replace(0, np.nan)
audit["net"] = audit["gmv"] - audit["subsidy"]
audit["roi"] = audit["gmv"] / audit["subsidy"].replace(0, np.nan)

# 现状增量口径：只有选择性响应型的 GMV 算增量
incr_mask = comp["behavior_label"] == "选择性响应型"
incr_gmv = comp[incr_mask].groupby("segment_name")["offer_difficulty"].sum().rename("gmv_incr")
audit = audit.join(incr_gmv).fillna({"gmv_incr": 0})
audit["net_incr"] = audit["gmv_incr"] - audit["subsidy"]

audit = audit.sort_values("subsidy", ascending=False)
audit.to_csv(os.path.join(OUT, "03_current_audit.csv"), encoding="utf-8-sig")

print("\n[2/6] 现状审计（按补贴降序）")
print(audit[["users", "sends", "completions", "subsidy", "gmv", "net", "roi", "net_incr"]]
      .round(2).to_string())

# ------------------------------------------------------------------
# 3. 策略矩阵：人群 × 券种
# ------------------------------------------------------------------
m_send = recv.groupby(["segment_name", "offer_group"]).size().rename("sends")
m_comp = comp.groupby(["segment_name", "offer_group"]).size().rename("comps")
matrix = pd.concat([m_send, m_comp], axis=1).fillna(0).reset_index()

# 补齐券种的 difficulty / reward / type
offer_meta = portfolio.groupby("offer_group").agg(
    offer_type=("offer_type", "first"),
    difficulty=("difficulty", "first"),
    reward=("reward", "first")
)
matrix = matrix.merge(offer_meta, left_on="offer_group", right_index=True, how="left")
matrix["comp_rate"] = matrix["comps"] / matrix["sends"].replace(0, np.nan)
matrix["net_per_send"] = matrix["comp_rate"] * (matrix["difficulty"] - matrix["reward"])
matrix["net_per_1k"] = matrix["net_per_send"] * 1000
matrix["segment_sends"] = matrix.groupby("segment_name")["sends"].transform("sum")
matrix["send_share"] = matrix["sends"] / matrix["segment_sends"]

matrix.to_csv(os.path.join(OUT, "03_offer_matrix.csv"), index=False, encoding="utf-8-sig")

# 现状主投券种（按触达量）
main_offer = (matrix.sort_values(["segment_name", "sends"], ascending=[True, False])
                    .groupby("segment_name").first()[["offer_group"]]
                    .rename(columns={"offer_group": "current_main"}))

print("\n[3/6] 策略矩阵（每千次触达净 GMV，仅列正收益券种）")
show = matrix[matrix.net_per_1k > 0].sort_values(["segment_name", "net_per_1k"], ascending=[True, False])
print(show[["segment_name", "offer_group", "sends", "comp_rate", "net_per_1k"]].round(3).to_string(index=False))

# ------------------------------------------------------------------
# 4. 策略规则
# ------------------------------------------------------------------
SEG_ORDER = [
    "高价值·全响应型", "高价值·选择性响应型", "高价值·零完成型",
    "中价值·全响应型", "中价值·选择性响应型", "中价值·零完成型", "中价值·低活跃型",
    "低价值·零完成型", "低价值·低活跃型", "中价值·未触达",
]
SEG_SHORT = {
    "高价值·全响应型": "高·全响应", "高价值·选择性响应型": "高·选择性",
    "高价值·零完成型": "高·零完成", "中价值·全响应型": "中·全响应",
    "中价值·选择性响应型": "中·选择性", "中价值·零完成型": "中·零完成",
    "中价值·低活跃型": "中·低活跃", "低价值·零完成型": "低·零完成",
    "低价值·低活跃型": "低·低活跃", "中价值·未触达": "中·未触达",
}

policies = []
for s in SEG_ORDER:
    if s not in audit.index:
        continue
    row = audit.loc[s]
    if s == "中价值·未触达":
        policies.append(dict(segment_name=s, policy_type="不触达", offer_group="不触达",
                             reason="无任何券事件，缺行为数据；先补一次触达再分层", est_sends=0))
        continue
    beh = s.split("·")[1]
    if beh == "全响应型":
        policies.append(dict(segment_name=s, policy_type="信息型", offer_group="信息型",
                             reason="收到可完成券 100% 完成，券不是增量；停发可完成券，仅保留 0 成本信息触达",
                             est_sends=int(row.sends)))
    elif beh == "选择性响应型":
        cand = matrix[(matrix.segment_name == s) & (matrix.offer_type == "discount") & (matrix.sends >= 50)]
        best_net = cand.net_per_1k.max()
        # 稳健性规则：净 GMV 在最优值 95% 以内的券种视为「同一档」，
        # 同档内优先完成率更高的（客户体验更好、方差更小），避免为了 2% 的净额去选高门槛券。
        shortlist = cand[cand.net_per_1k >= best_net * 0.95]
        best = shortlist.sort_values(["comp_rate", "net_per_1k"], ascending=False).iloc[0]
        max_row = cand.sort_values("net_per_1k", ascending=False).iloc[0]
        reason = (f"选择性人群；{best.offer_group} 每千次净 GMV {best.net_per_1k:,.0f}、完成率 {best.comp_rate:.0%}。"
                  f"{max_row.offer_group} 净 GMV 最高（{max_row.net_per_1k:,.0f}）但仅高 "
                  f"{(max_row.net_per_1k/best.net_per_1k-1)*100:.0f}%，完成率却低 "
                  f"{(best.comp_rate-max_row.comp_rate)*100:.0f}pp，且该人群偏好低门槛，故不选")
        policies.append(dict(segment_name=s, policy_type="折扣券", offer_group=best.offer_group,
                             reason=reason, est_sends=int(row.sends)))
    elif beh == "零完成型":
        policies.append(dict(segment_name=s, policy_type="信息型/停发", offer_group="信息型",
                             reason="完成率 0%，查看率约 70%；门槛结构性不可达，停发可完成券，另做低门槛 A/B",
                             est_sends=int(row.sends)))
    else:  # 低活跃
        if row.completions >= 50:
            reason = ("交易中位仅 1 笔、消费 14 元；历史完成 218 次但样本小。主方案不配预算，"
                      "建议 10% 小流量测试折扣 10-2（该人群每千次净 GMV 1,514）")
        else:
            reason = "30 天几乎无交易，发券无核销基础；只保留信息触达与产品激活"
        policies.append(dict(segment_name=s, policy_type="信息型", offer_group="信息型",
                             reason=reason, est_sends=int(row.sends)))

policy = pd.DataFrame(policies)
policy = policy.merge(main_offer, left_on="segment_name", right_index=True, how="left")

# 估算 S1：保持当前触达量，按人群目标券种的历史完成率外推
def est_row(r):
    if r.offer_group in ("信息型", "不触达"):
        return pd.Series({"est_comps": 0.0, "est_subsidy": 0.0, "est_gmv": 0.0, "est_net": 0.0})
    m = matrix[(matrix.segment_name == r.segment_name) & (matrix.offer_group == r.offer_group)].iloc[0]
    c = r.est_sends * m.comp_rate
    sub = c * m.reward
    gmv = c * m.difficulty
    return pd.Series({"est_comps": c, "est_subsidy": sub, "est_gmv": gmv, "est_net": gmv - sub})

policy = pd.concat([policy, policy.apply(est_row, axis=1)], axis=1)
policy.to_csv(os.path.join(OUT, "03_policy.csv"), index=False, encoding="utf-8-sig")

print("\n[4/6] 策略规则")
print(policy[["segment_name", "current_main", "offer_group", "est_sends", "est_subsidy", "est_net"]]
      .round(1).to_string(index=False))

# ------------------------------------------------------------------
# 5. 情景对比 S0 vs S1
# ------------------------------------------------------------------
def scen(name, sends_, comps_, subsidy_, gmv_, gmv_incr_):
    return dict(
        scenario=name, sends=sends_, completions=comps_, subsidy=subsidy_,
        gmv=gmv_, net_book=gmv_ - subsidy_,
        gmv_incr=gmv_incr_, net_incr=gmv_incr_ - subsidy_,
        roi_book=gmv_ / subsidy_ if subsidy_ else np.nan,
        roi_incr=gmv_incr_ / subsidy_ if subsidy_ else np.nan,
    )

s0 = scen("S0 现状",
          audit.sends.sum(), audit.completions.sum(), audit.subsidy.sum(), audit.gmv.sum(), audit.gmv_incr.sum())
s1 = scen("S1 优化",
          policy.est_sends.sum(), policy.est_comps.sum(), policy.est_subsidy.sum(), policy.est_gmv.sum(), policy.est_gmv.sum())

compare = pd.DataFrame([s0, s1])
compare.to_csv(os.path.join(OUT, "03_scenario_compare.csv"), index=False, encoding="utf-8-sig")

print("\n[5/6] 情景对比")
print(compare.round(1).to_string(index=False))
print(f"\n补贴变化：{s1['subsidy'] - s0['subsidy']:,.0f}（{(s1['subsidy']/s0['subsidy']-1)*100:.1f}%）")
print(f"增量净额变化：{s1['net_incr'] - s0['net_incr']:,.0f}")

# 人群级 S1 汇总
seg_s1 = policy.groupby("segment_name").agg(
    est_sends=("est_sends", "sum"), est_comps=("est_comps", "sum"),
    est_subsidy=("est_subsidy", "sum"), est_gmv=("est_gmv", "sum"), est_net=("est_net", "sum")
).reindex(SEG_ORDER)
seg_s1.to_csv(os.path.join(OUT, "03_segment_scenario.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------
# 6. 图表
# ------------------------------------------------------------------
mat_rows = [s for s in SEG_ORDER if s != "中价值·未触达"]
rate_df = matrix.pivot(index="segment_name", columns="offer_group", values="comp_rate").reindex(mat_rows)[GROUP_ORDER]
net_df = matrix.pivot(index="segment_name", columns="offer_group", values="net_per_1k").reindex(mat_rows)[GROUP_ORDER]
send_df = matrix.pivot(index="segment_name", columns="offer_group", values="sends").reindex(mat_rows)[GROUP_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6), gridspec_kw={"width_ratios": [1.05, 1]})
for ax, df, cmap_name, title, fmt, cbar_label in [
    (axes[0], rate_df, "Blues", "A. 触达→完成率", lambda v: f"{v*100:.0f}%", "完成率"),
    (axes[1], net_df, "Greens", "B. 每千次触达净 GMV", lambda v: f"{v:,.0f}", "净 GMV / 千次"),
]:
    data = df.values.astype(float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#EDEDED")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([GROUP_SHORT[c] for c in df.columns], fontsize=8.8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([SEG_SHORT[s] for s in df.index], fontsize=9.2)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(False)
    if ax is axes[1]:
        ax.set_yticklabels([])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999999")
            elif ax is axes[1] and v == 0:
                continue  # 净 GMV 为 0 的格子留空，避免满屏 “0”
            else:
                color = "white" if (v - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-9) > 0.55 else "#222222"
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.6, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=8)

fig.suptitle("策略矩阵：9 个人群 × 6 类券（数值为历史观测，BOGO 净 GMV 恒为 0）", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, os.path.join(FIG, "03_offer_matrix.png"))

# S0 / S1 补贴重分配 + 净额对比
fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.3), gridspec_kw={"width_ratios": [1.25, 1]})

ax = axes[0]
s0_sub = audit["subsidy"].reindex(SEG_ORDER).fillna(0)
s1_sub = seg_s1["est_subsidy"].reindex(SEG_ORDER).fillna(0)
seg_colors = plt.get_cmap("tab10")(np.arange(len(SEG_ORDER)))
left0 = 0
left1 = 0
for i, s in enumerate(SEG_ORDER):
    if s0_sub[s] > 0:
        ax.barh(1, s0_sub[s], left=left0, color=seg_colors[i], height=0.55, edgecolor="white", linewidth=0.5)
        left0 += s0_sub[s]
    if s1_sub[s] > 0:
        ax.barh(0, s1_sub[s], left=left1, color=seg_colors[i], height=0.55, edgecolor="white", linewidth=0.5)
        left1 += s1_sub[s]
ax.set_yticks([0, 1])
ax.set_yticklabels(["S1 优化", "S0 现状"], fontsize=11)
ax.set_xlabel("补贴金额（元）")
ax.set_title("A. 补贴收缩：只保留选择性人群的可完成券", fontsize=12, pad=10)
ax.text(left0 + s0_sub.max() * 0.02, 1, f"合计 {s0_sub.sum():,.0f}", va="center", fontsize=10, color="#333333")
ax.text(left1 + s1_sub.max() * 0.02, 0, f"合计 {s1_sub.sum():,.0f}", va="center", fontsize=10, color="#333333")
ax.set_xlim(0, max(left0, left1) * 1.18)
ax.grid(axis="x")
handles = [Patch(facecolor=seg_colors[i], edgecolor="white", label=SEG_SHORT[s]) for i, s in enumerate(SEG_ORDER)]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=8.2,
          handlelength=1.2, columnspacing=1.2)

ax = axes[1]
metrics = ["subsidy", "gmv", "net_book", "net_incr"]
labels = ["补贴", "GMV", "账面净额", "增量净额"]
x = np.arange(len(metrics))
w = 0.36
b0 = ax.bar(x - w/2, [s0[m] for m in metrics], w, color="#B9C6DD", label="S0 现状")
b1 = ax.bar(x + w/2, [s1[m] for m in metrics], w, color="#1D3A6E", label="S1 优化")
for bars in (b0, b1):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + (4 if h >= 0 else -4), f"{h:,.0f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=8.6)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("B. 补贴、GMV 与净额：账面 vs 增量", fontsize=12, pad=10)
ax.legend(loc="upper right")
ax.grid(axis="y")

fig.suptitle("预算模拟：S0 现状 vs S1 优化（保持触达量，只切换券型结构）", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, os.path.join(FIG, "03_budget_compare.png"))

print("\n[6/6] 已输出：03_current_audit.csv / 03_offer_matrix.csv / 03_policy.csv / "
      "03_scenario_compare.csv / 03_segment_scenario.csv / 03_offer_matrix.png / 03_budget_compare.png")
