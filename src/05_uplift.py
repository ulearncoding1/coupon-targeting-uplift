# -*- coding: utf-8 -*-
"""
05_uplift.py —— 增量评估（准实验 + 分人群 CATE）

背景：
    这份数据没有真正的「无券对照组」。但有三类可比窗口：
    A. 券 vs 信息型券：同样触达、无购买激励，是最干净的「消息对照」；
    B. 同一用户内，券窗口 vs 信息窗口：用户固定效应，控制稳定画像；
    C. 券窗口 vs 完全没有触达的窗口：衡量「触达本身」的总效果，但时间分布不均，只作描述。

主口径：
    - 结果变量：发券后 72 小时内是否发生交易（any_tx）、72 小时内 GMV；
    - 稳健性：168 小时；
    - 主估计：用户固定效应 + 活动日/日平方/第几次收券/时段控制，按用户聚类标准误；
    - 分人群 CATE：每个 02 人群单独做 FE；
    - 分券型：BOGO5/BOGO10/折扣7/折扣10/折扣20 相对信息型券的增量；
    - T-learner：券模型 vs 信息模型的预测差，做 uplift 十分组检验。

重要局限（写进文档）：
    1. 信息型券不是「无触达」，它本身有提醒效应；
    2. 72 小时窗口可能包含「把未来购买提前」的 pull-forward，不等于长期净增量；
    3. 人群标签由全窗口行为聚类得到，只用于分群描述，不作为因果调整变量。
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from utils_viz import use_style, save_fig
from config import CLEAN, TABLE, FIG

warnings.filterwarnings("ignore")
DATA = CLEAN
OUT = TABLE
use_style()

# ------------------------------------------------------------------
# 1. 构造发券窗口与结果变量
# ------------------------------------------------------------------
d = pd.read_csv(os.path.join(OUT, "04_offer_dataset.csv"), encoding="utf-8-sig")
d = d.drop(columns=[c for c in d.columns if c.endswith(".1")])
portfolio = pd.read_csv(os.path.join(DATA, "portfolio.csv"))
d = d.merge(portfolio[["id", "offer_type"]].rename(columns={"id": "offer_id"}),
            on="offer_id", how="left")
d["coupon"] = d["offer_type"].isin(["bogo", "discount"]).astype(int)

trans = pd.read_csv(os.path.join(DATA, "transcript.csv"))
tx = trans[trans.event == "transaction"]
txs = {uid: (g.time.to_numpy(float), g.amount.fillna(0).to_numpy(float))
       for uid, g in tx.groupby("person")}

def window_outcome(uid, start, hours):
    if uid not in txs:
        return 0, 0.0
    ts, am = txs[uid]
    a = np.searchsorted(ts, start, side="left")
    b = np.searchsorted(ts, start + hours, side="left")
    return int(b - a), float(am[a:b].sum())

for h in [72, 168]:
    res = [window_outcome(r.person, r.send_time, h) for r in d.itertuples()]
    d[f"tx{h}"], d[f"gmv{h}"] = zip(*res)
    d[f"any{h}"] = (d[f"tx{h}"] > 0).astype(int)

d["offer_group"] = "info"
d.loc[(d.offer_type == "bogo") & (d.difficulty == 5), "offer_group"] = "bogo5"
d.loc[(d.offer_type == "bogo") & (d.difficulty == 10), "offer_group"] = "bogo10"
d.loc[(d.offer_type == "discount") & (d.difficulty == 7), "offer_group"] = "d7"
d.loc[(d.offer_type == "discount") & (d.difficulty == 10), "offer_group"] = "d10"
d.loc[(d.offer_type == "discount") & (d.difficulty == 20), "offer_group"] = "d20"

print(f"[1/7] 发券窗口 {len(d):,}：券 {d.coupon.sum():,} / 信息型 {(1-d.coupon).sum():,}")
print("72h 购买率：券 {:.1%} / 信息 {:.1%}".format(
    d[d.coupon == 1].any72.mean(), d[d.coupon == 0].any72.mean()))

# ------------------------------------------------------------------
# 2. 准随机性检查：发券前的协变量平衡
# ------------------------------------------------------------------
balance_cols = ["age", "income", "tenure_days", "n_tx_prior", "avg_order_prior",
                "recency_h", "n_recv_prior", "comp_rate_prior", "tx_7d",
                "send_day", "seq_recv", "duration_days"]
bal = []
for c in balance_cols:
    a = d.loc[d.coupon == 1, c].astype(float).dropna()
    b = d.loc[d.coupon == 0, c].astype(float).dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    bal.append({"covariate": c, "coupon_mean": a.mean(), "info_mean": b.mean(), "smd": smd})
bal = pd.DataFrame(bal)
bal.to_csv(os.path.join(OUT, "05_balance.csv"), index=False, encoding="utf-8-sig")
print("\n[2/7] 发券前协变量平衡（|SMD|<0.1 视为平衡）")
print(bal.round(3).to_string(index=False))

# ------------------------------------------------------------------
# 3. 回归工具：用户固定效应 + 聚类标准误
# ------------------------------------------------------------------
def ols_cluster(df, ycol, xcols, cluster="person", fe=False):
    """普通/用户固定效应 OLS，按 cluster 聚类标准误。"""
    sub = df.dropna(subset=[ycol] + xcols).reset_index(drop=True).copy()
    X = sub[xcols].to_numpy(float).copy()
    y = sub[ycol].to_numpy(float).copy()
    groups = sub[cluster]
    if fe:
        for j in range(X.shape[1]):
            X[:, j] -= pd.Series(X[:, j]).groupby(groups).transform("mean").to_numpy()
        y -= pd.Series(y).groupby(groups).transform("mean").to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    XtX = np.linalg.pinv(X.T @ X)
    meat = np.zeros_like(XtX)
    for _, idx in pd.Series(np.arange(len(sub))).groupby(groups.values):
        Xg = X[idx.to_numpy()]
        ug = resid[idx.to_numpy()]
        meat += Xg.T @ np.outer(ug, ug) @ Xg
    V = XtX @ meat @ XtX
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta, se, len(sub)

def add_est(rows, method, outcome, beta, se, n, note=""):
    rows.append({
        "method": method, "outcome": outcome,
        "estimate": float(beta), "se": float(se),
        "ci_low": float(beta - 1.96 * se), "ci_high": float(beta + 1.96 * se),
        "n": int(n), "note": note,
    })

ests = []
# 3.1 朴素对比（不控制任何东西，仅按用户聚类；带截距，系数才是差值）
d["const"] = 1.0
for y in ["any72", "gmv72", "any168", "gmv168"]:
    beta, se, n = ols_cluster(d, y, ["const", "coupon"], fe=False)
    add_est(ests, "naive_coupon_vs_info", y, beta[1], se[1], n, "未控制任何协变量")

# 3.2 同一用户内：券 vs 信息（用户 FE + 时间控制）
both_users = d.groupby("person").coupon.nunique()
paired = d[d.person.isin(both_users[both_users == 2].index)].copy()
controls = ["send_day", "send_day_sq", "seq_recv", "hour12", "hour18"]
for df_ in [d, paired]:
    df_["send_day_sq"] = df_["send_day"] ** 2
    df_["hour12"] = (df_["send_hour"] == 12).astype(int)
    df_["hour18"] = (df_["send_hour"] == 18).astype(int)

for y in ["any72", "gmv72", "any168", "gmv168"]:
    beta, se, n = ols_cluster(paired, y, ["coupon"] + controls, fe=True)
    add_est(ests, "within_user_fe", y, beta[0], se[0], n, "用户 FE + 活动日/日平方/收券次序/时段")

est_df = pd.DataFrame(ests)
est_df.to_csv(os.path.join(OUT, "05_estimates.csv"), index=False, encoding="utf-8-sig")
print("\n[3/7] 增量估计（券 vs 信息型券）")
print(est_df.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 4. 分人群 CATE：每个 02 人群单独做用户 FE
# ------------------------------------------------------------------
seg_rows = []
for s, g in paired.groupby("segment_name"):
    if len(g) < 100:
        continue
    beta, se, n = ols_cluster(g, "any72", ["coupon"] + controls, fe=True)
    beta_g, se_g, _ = ols_cluster(g, "gmv72", ["coupon"] + controls, fe=True)
    seg_rows.append({
        "segment_name": s, "n_sends": n, "n_users": g.person.nunique(),
        "effect_any72": beta[0], "se_any72": se[0],
        "ci_low": beta[0] - 1.96 * se[0], "ci_high": beta[0] + 1.96 * se[0],
        "effect_gmv72": beta_g[0], "se_gmv72": se_g[0],
        "actual_coupon_rate": g.loc[g.coupon == 1, "any72"].mean(),
        "actual_info_rate": g.loc[g.coupon == 0, "any72"].mean(),
        "median_aov": g.loc[g.coupon == 1, "avg_order_prior"].median(),
        "median_ratio": g.loc[g.coupon == 1, "difficulty_to_aov"].median(),
        "avg_reward_current": g.loc[g.coupon == 1, "reward"].mean(),
    })
seg = pd.DataFrame(seg_rows)
audit_seg = pd.read_csv(os.path.join(OUT, "03_current_audit.csv"), encoding="utf-8-sig")
audit_seg["avg_subsidy_current"] = audit_seg.subsidy / audit_seg.sends.replace(0, np.nan)
seg = seg.merge(audit_seg[["segment_name", "avg_subsidy_current"]], on="segment_name", how="left")
seg["significant"] = (seg.ci_low > 0) | (seg.ci_high < 0)
seg["net_gmv72"] = seg.effect_gmv72 - seg.avg_subsidy_current
seg["roi_incremental"] = seg.effect_gmv72 / seg.avg_subsidy_current
seg = seg.sort_values("effect_any72", ascending=False)
seg.to_csv(os.path.join(OUT, "05_segment_uplift.csv"), index=False, encoding="utf-8-sig")

print("\n[4/7] 分人群增量（72h 购买率，券 - 信息）")
print(seg[["segment_name", "n_sends", "effect_any72", "se_any72", "effect_gmv72",
           "actual_coupon_rate", "actual_info_rate", "median_aov", "median_ratio", "significant"]]
      .round(3).to_string(index=False))

# ------------------------------------------------------------------
# 5. 分券型增量：BOGO5/BOGO10/折扣7/折扣10/折扣20 相对信息
# ------------------------------------------------------------------
for t in ["bogo5", "bogo10", "d7", "d10", "d20"]:
    paired[f"t_{t}"] = (paired.offer_group == t).astype(int)
type_cols = ["t_bogo5", "t_bogo10", "t_d7", "t_d10", "t_d20"]
sub_map = {"BOGO 5/5": 5.0, "BOGO 10/10": 10.0, "折扣 7-3": 3.0, "折扣 10-2": 2.0, "折扣 20-5": 5.0}
type_rows = []
for y in ["any72", "gmv72"]:
    beta, se, n = ols_cluster(paired, y, type_cols + controls, fe=True)
    for i, t in enumerate(["BOGO 5/5", "BOGO 10/10", "折扣 7-3", "折扣 10-2", "折扣 20-5"]):
        type_rows.append({"offer_type": t, "outcome": y, "estimate": beta[i], "se": se[i],
                          "ci_low": beta[i] - 1.96 * se[i], "ci_high": beta[i] + 1.96 * se[i],
                          "n": n, "n_type": int((paired.offer_group == ["bogo5", "bogo10", "d7", "d10", "d20"][i]).sum())})
type_df = pd.DataFrame(type_rows)
type_df["subsidy"] = type_df.offer_type.map(sub_map)
type_df["net_gmv72"] = np.where(type_df.outcome == "gmv72", type_df.estimate - type_df.subsidy, np.nan)
type_df["roi_incremental"] = np.where(type_df.outcome == "gmv72", type_df.estimate / type_df.subsidy, np.nan)
type_df.to_csv(os.path.join(OUT, "05_type_uplift.csv"), index=False, encoding="utf-8-sig")
print("\n[5/7] 分券型增量（相对信息型券，用户 FE）")
print(type_df.round(4).to_string(index=False))

# 5b. 关键人群内部的分券型增量：回答「不是发不发，而是发哪种」
seg_type_rows = []
for s in ["高价值·全响应型", "高价值·选择性响应型", "高价值·零完成型",
          "中价值·全响应型", "中价值·选择性响应型", "中价值·零完成型",
          "中价值·低活跃型", "低价值·低活跃型"]:
    sub = paired[paired.segment_name == s]
    if len(sub) < 100:
        continue
    for y in ["any72", "gmv72"]:
        beta, se, n = ols_cluster(sub, y, type_cols + controls, fe=True)
        for i, t in enumerate(["BOGO 5/5", "BOGO 10/10", "折扣 7-3", "折扣 10-2", "折扣 20-5"]):
            seg_type_rows.append({"segment_name": s, "offer_type": t, "outcome": y,
                                  "estimate": beta[i], "se": se[i],
                                  "ci_low": beta[i] - 1.96 * se[i], "ci_high": beta[i] + 1.96 * se[i],
                                  "n": n, "n_type": int((sub.offer_group == ["bogo5", "bogo10", "d7", "d10", "d20"][i]).sum())})
seg_type_df = pd.DataFrame(seg_type_rows)
seg_type_df["subsidy"] = seg_type_df.offer_type.map(sub_map)
seg_type_df["net_gmv72"] = np.where(seg_type_df.outcome == "gmv72", seg_type_df.estimate - seg_type_df.subsidy, np.nan)
seg_type_df["roi_incremental"] = np.where(seg_type_df.outcome == "gmv72", seg_type_df.estimate / seg_type_df.subsidy, np.nan)
seg_type_df.to_csv(os.path.join(OUT, "05_segment_type_uplift.csv"), index=False, encoding="utf-8-sig")
print("\n[5b/7] 关键人群分券型增量")
print(seg_type_df.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 6. T-learner：券模型 vs 信息模型，做 uplift 十分组
# ------------------------------------------------------------------
imp = pd.read_csv(os.path.join(OUT, "04_feature_importance.csv"), encoding="utf-8-sig")
FEATURES = imp.feature.tolist()
X = d[FEATURES].astype(float)
y = d["any72"].astype(int)
groups = d["person"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, groups=groups))
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=5, verbosity=-1, seed=42, num_threads=4)

tr_c = tr[d.iloc[tr].coupon.values == 1]
tr_i = tr[d.iloc[tr].coupon.values == 0]
m1 = lgb.train(params, lgb.Dataset(X.iloc[tr_c], y.iloc[tr_c]), num_boost_round=250)
m0 = lgb.train(params, lgb.Dataset(X.iloc[tr_i], y.iloc[tr_i]), num_boost_round=250)
p1 = m1.predict(X)
p0 = m0.predict(X)
d["p_coupon_model"] = p1
d["p_info_model"] = p0
d["uplift_score"] = p1 - p0

te_df = d.iloc[te].copy()
te_df["decile"] = pd.qcut(te_df.uplift_score, 10, labels=False, duplicates="drop") + 1
dec = te_df.groupby("decile").apply(
    lambda g: pd.Series({
        "n": len(g), "n_coupon": int(g.coupon.sum()), "n_info": int((1 - g.coupon).sum()),
        "coupon_rate": g.loc[g.coupon == 1, "any72"].mean(),
        "info_rate": g.loc[g.coupon == 0, "any72"].mean(),
        "gap": g.loc[g.coupon == 1, "any72"].mean() - g.loc[g.coupon == 0, "any72"].mean(),
        "pred_uplift": g.uplift_score.mean(),
    }), include_groups=False
).reset_index()
dec.to_csv(os.path.join(OUT, "05_uplift_deciles.csv"), index=False, encoding="utf-8-sig")
print("\n[6/7] T-learner uplift 十分组（测试集）")
print(dec.round(3).to_string(index=False))

# 6b. T-learner 的简单验收指标
from sklearn.metrics import roc_auc_score
te_c = te_df[te_df.coupon == 1]
te_i = te_df[te_df.coupon == 0]
auc_c = roc_auc_score(te_c.any72, te_c.p_coupon_model)
auc_i = roc_auc_score(te_i.any72, te_i.p_info_model)
overall_gap = te_c.any72.mean() - te_i.any72.mean()
top20 = te_df[te_df.uplift_score >= te_df.uplift_score.quantile(0.8)]
bot20 = te_df[te_df.uplift_score <= te_df.uplift_score.quantile(0.2)]
def gap_of(g):
    return g.loc[g.coupon == 1, "any72"].mean() - g.loc[g.coupon == 0, "any72"].mean()
metrics = pd.DataFrame([
    {"metric": "auc_coupon_model", "value": auc_c, "note": "测试集券窗口"},
    {"metric": "auc_info_model", "value": auc_i, "note": "测试集信息窗口"},
    {"metric": "overall_gap_pp", "value": overall_gap * 100, "note": "测试集整体券-信息"},
    {"metric": "top20_gap_pp", "value": gap_of(top20) * 100, "note": f"n={len(top20)}"},
    {"metric": "bottom20_gap_pp", "value": gap_of(bot20) * 100, "note": f"n={len(bot20)}"},
])
metrics.to_csv(os.path.join(OUT, "05_uplift_metrics.csv"), index=False, encoding="utf-8-sig")
print("\nT-learner 验收：")
print(metrics.round(3).to_string(index=False))

# 保存窗口数据
d[["person", "offer_id", "send_time", "send_day", "send_hour", "seq_recv",
   "segment_name", "offer_type", "offer_group", "coupon",
   "tx72", "gmv72", "any72", "tx168", "gmv168", "any168",
   "p_coupon_model", "p_info_model", "uplift_score"]].to_csv(
    os.path.join(OUT, "05_send_windows.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
# 7. 策略 v2 与图表
# ------------------------------------------------------------------
pol = pd.read_csv(os.path.join(OUT, "03_policy.csv"), encoding="utf-8-sig")
audit = pd.read_csv(os.path.join(OUT, "03_current_audit.csv"), encoding="utf-8-sig")
seg_map = seg.set_index("segment_name")
st_gmv = seg_type_df[seg_type_df.outcome == "gmv72"].copy()
st_any = seg_type_df[seg_type_df.outcome == "any72"].copy()
rows = []
for _, pr in pol.iterrows():
    s = pr.segment_name
    est_sends = float(pr.est_sends)
    row = {"segment_name": s, "current_main": pr.current_main, "offer_group": pr.offer_group,
           "est_sends": est_sends}
    if s == "中价值·未触达":
        row.update({"v2_offer": "不触达", "v2_reason": "无券事件，保持不触达",
                    "effect_any72_pp": np.nan, "effect_gmv72": np.nan,
                    "avg_subsidy_current": np.nan, "current_inc_net_per_send": np.nan,
                    "v2_subsidy_per_send": 0.0, "v2_inc_gmv_per_send": 0.0,
                    "v2_inc_net_per_send": 0.0,
                    "v2_effect_any72_pp": 0.0, "v2_est_subsidy": 0.0,
                    "v2_est_inc_gmv": 0.0, "v2_est_inc_net": 0.0,
                    "current_inc_net_est": 0.0, "v2_delta_net_vs_current": 0.0})
        rows.append(row); continue
    if s not in seg_map.index:
        row.update({"v2_offer": "信息型", "v2_reason": "样本不足，保持信息型",
                    "effect_any72_pp": np.nan, "effect_gmv72": np.nan,
                    "avg_subsidy_current": np.nan, "current_inc_net_per_send": np.nan,
                    "v2_subsidy_per_send": 0.0, "v2_inc_gmv_per_send": 0.0,
                    "v2_inc_net_per_send": 0.0,
                    "v2_effect_any72_pp": 0.0, "v2_est_subsidy": 0.0,
                    "v2_est_inc_gmv": 0.0, "v2_est_inc_net": 0.0,
                    "current_inc_net_est": 0.0, "v2_delta_net_vs_current": 0.0})
        rows.append(row); continue
    sr = seg_map.loc[s]
    g = st_gmv[st_gmv.segment_name == s].sort_values("net_gmv72", ascending=False)
    # 只有「增量 GMV 显著为正，且扣掉补贴后仍为正」的券型才值得发
    ok = g[(g.estimate > 0) & (g.ci_low > 0) & (g.net_gmv72 > 0)]
    best = ok.iloc[0] if len(ok) else None
    if best is not None:
        a = st_any[(st_any.segment_name == s) & (st_any.offer_type == best.offer_type)].iloc[0]
        v2_offer = best.offer_type
        v2_sub = float(best.subsidy)
        v2_gmv = float(best.estimate)
        v2_net = float(best.net_gmv72)
        v2_any = float(a.estimate) * 100
        reason = (f"{best.offer_type} 增量净 GMV {best.net_gmv72:+.2f}/次"
                  f"（增量 ROI {best.roi_incremental:.2f}），人群内最优；改发此券")
    else:
        v2_offer, v2_sub, v2_gmv, v2_net, v2_any = "信息型", 0.0, 0.0, 0.0, 0.0
        if len(g):
            b = g.iloc[0]
            reason = f"券型最优 {b.offer_type} 增量净 {b.net_gmv72:+.2f}/次，仍为负；改信息型"
        else:
            reason = "样本不足，保持信息型"
    current_net = float(sr.net_gmv72)
    current_net_est = est_sends * current_net
    row.update({
        "v2_offer": v2_offer, "v2_reason": reason,
        "effect_any72_pp": float(sr.effect_any72) * 100,
        "effect_gmv72": float(sr.effect_gmv72),
        "avg_subsidy_current": float(sr.avg_subsidy_current),
        "current_inc_net_per_send": current_net,
        "v2_subsidy_per_send": v2_sub, "v2_inc_gmv_per_send": v2_gmv,
        "v2_inc_net_per_send": v2_net, "v2_effect_any72_pp": v2_any,
        "v2_est_subsidy": est_sends * v2_sub,
        "v2_est_inc_gmv": est_sends * v2_gmv,
        "v2_est_inc_net": est_sends * v2_net,
        "current_inc_net_est": current_net_est,
        "v2_delta_net_vs_current": est_sends * v2_net - current_net_est,
    })
    rows.append(row)
pol_v2 = pd.DataFrame(rows)
pol_v2.to_csv(os.path.join(OUT, "05_policy_v2.csv"), index=False, encoding="utf-8-sig")

current_subsidy = float(audit.subsidy.sum())
current_inc_gmv = float(audit.gmv_incr.sum())
current_inc_net = float(audit.net_incr.sum())
v2_subsidy = float(pol_v2.v2_est_subsidy.sum())
v2_inc_gmv = float(pol_v2.v2_est_inc_gmv.sum())
v2_inc_net = float(pol_v2.v2_est_inc_net.sum())
current_inc_gmv_fe = float((pol_v2.est_sends * pol_v2.effect_gmv72.fillna(0)).sum())
current_inc_net_fe = float(pol_v2.current_inc_net_est.sum())
# 把 03 的 S1（只对两个选择性人群发折扣 10-2）用 05 口径重算增量净额
s1_net = 0.0
for s in ["高价值·选择性响应型", "中价值·选择性响应型"]:
    b = seg_type_df[(seg_type_df.segment_name == s) & (seg_type_df.offer_type == "折扣 10-2")
                    & (seg_type_df.outcome == "gmv72")].iloc[0]
    sends = float(pol.loc[pol.segment_name == s, "est_sends"].iloc[0])
    s1_net += sends * (b.estimate - b.subsidy)
summary = pd.DataFrame([
    {"metric": "current_subsidy", "value": current_subsidy, "note": "03 现状审计"},
    {"metric": "v2_subsidy", "value": v2_subsidy, "note": "只保留增量净 GMV 为正的人群"},
    {"metric": "current_inc_gmv", "value": current_inc_gmv, "note": "03 账面口径：增量 GMV"},
    {"metric": "current_inc_gmv_fe", "value": current_inc_gmv_fe, "note": "05 用户 FE 口径：增量 GMV"},
    {"metric": "v2_inc_gmv", "value": v2_inc_gmv, "note": "05 口径：只算券相对信息的增量"},
    {"metric": "current_inc_net", "value": current_inc_net, "note": "03 账面口径：增量 GMV − 补贴"},
    {"metric": "current_inc_net_fe", "value": current_inc_net_fe, "note": "05 用户 FE 口径：增量 GMV − 补贴"},
    {"metric": "v2_inc_net", "value": v2_inc_net, "note": "增量 GMV − 补贴"},
    {"metric": "delta_inc_net", "value": v2_inc_net - current_inc_net, "note": "v2 − 03 现状"},
    {"metric": "delta_inc_net_fe", "value": v2_inc_net - current_inc_net_fe, "note": "v2 − 05 FE 现状"},
    {"metric": "s1_inc_net_05", "value": s1_net, "note": "03 S1 用 05 口径重算"},
    {"metric": "v2_vs_s1", "value": v2_inc_net - s1_net, "note": "v2 − S1（05 口径）"},
    {"metric": "current_inc_roi", "value": current_inc_gmv / current_subsidy, "note": "03 增量 GMV / 补贴"},
    {"metric": "v2_inc_roi", "value": v2_inc_gmv / v2_subsidy, "note": "05 增量 GMV / 补贴"},
    {"metric": "v2_coupon_sends", "value": float(pol_v2[pol_v2.v2_subsidy_per_send > 0].est_sends.sum()), "note": ""},
    {"metric": "v2_info_sends", "value": float(pol_v2[pol_v2.v2_subsidy_per_send == 0].est_sends.sum()), "note": ""},
])
summary.to_csv(os.path.join(OUT, "05_policy_v2_summary.csv"), index=False, encoding="utf-8-sig")

print("\n策略 v1 → v2：")
print(pol_v2[["segment_name", "current_main", "v2_offer", "effect_any72_pp",
              "v2_subsidy_per_send", "v2_inc_net_per_send", "v2_est_inc_net",
              "v2_delta_net_vs_current", "v2_reason"]].round(3).to_string(index=False))
print("\nv2 预算汇总：")
print(summary.round(2).to_string(index=False))

# 7.1 分人群 forest plot
sp = seg.sort_values("effect_any72").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.6, 6.2))
yy = np.arange(len(sp))
colors = ["#C0392B" if v < 0 else "#1D3A6E" for v in sp.effect_any72]
ax.errorbar(sp.effect_any72 * 100, yy, xerr=1.96 * sp.se_any72 * 100, fmt="o",
            color="#333333", ecolor="#999999", capsize=3, markersize=0, zorder=2)
ax.scatter(sp.effect_any72 * 100, yy, s=np.clip(sp.n_sends / 100, 30, 260),
           c=colors, zorder=3, edgecolor="white")
ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--")
ax.set_yticks(yy)
ax.set_yticklabels([s.replace("价值", "").replace("响应型", "").replace("型", "") for s in sp.segment_name], fontsize=9)
ax.set_xlabel("72 小时购买率增量（券 − 信息型，pp）")
ax.set_title("分人群增量（当前券型组合）：高·选择性平均为负")
for i, r in sp.iterrows():
    ax.text(r.effect_any72 * 100 + (2.2 if r.effect_any72 >= 0 else -2.2), i,
            f"{r.effect_any72*100:+.1f}pp", va="center",
            ha="left" if r.effect_any72 >= 0 else "right", fontsize=8, color="#333333")
ax.set_xlim(sp.effect_any72.min() * 100 - 8, sp.effect_any72.max() * 100 + 12)
ax.grid(axis="x")
save_fig(fig, os.path.join(FIG, "05_segment_effects.png"))

# 7.2 机制：门槛/客单比 vs 增量（对数横轴，避免低客单组把 0-3 区间压扁）
fig, ax = plt.subplots(figsize=(8.8, 5.8))
colors2 = ["#C0392B" if v < 0 else "#4CC38A" for v in seg.effect_any72]
ax.scatter(seg.median_ratio, seg.effect_any72 * 100, s=np.clip(seg.n_sends / 120, 35, 260),
           c=colors2, edgecolor="#333333", linewidth=0.6, zorder=3, alpha=0.85)
offsets = {
    "中价值·全响应型": (8, -12),
    "高价值·全响应型": (8, 6),
    "中价值·选择性响应型": (8, 8),
    "高价值·选择性响应型": (8, -16),
    "中价值·零完成型": (8, 10),
    "高价值·零完成型": (8, -16),
    "中价值·低活跃型": (-10, 10),
    "低价值·低活跃型": (-10, -16),
}
for _, r in seg.iterrows():
    dx, dy = offsets.get(r.segment_name, (8, 6))
    ax.annotate(r.segment_name.replace("价值", "").replace("响应型", "").replace("型", ""),
                (r.median_ratio, r.effect_any72 * 100), textcoords="offset points",
                xytext=(dx, dy), fontsize=8, color="#333333")
ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
ax.axvline(1, color="#AAAAAA", linestyle=":", linewidth=1)
ax.set_xscale("log")
ax.set_xlim(0.45, 25)
ax.set_xlabel("该人群券的「门槛 / 客单价」中位数（对数轴；>1 表示门槛高于平均一单）")
ax.set_ylabel("72h 购买率增量（pp）")
ax.set_title("门槛越高于客单，券的增量越差（甚至为负）")
ax.grid(axis="both")
save_fig(fig, os.path.join(FIG, "05_mechanism.png"))

# 7.3 分券型
tp = type_df[type_df.outcome == "any72"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.0, 4.8))
cols = ["#C0392B" if v < 0 else "#1D3A6E" for v in tp.estimate]
ax.bar(tp.offer_type, tp.estimate * 100, color=cols, width=0.62)
ax.errorbar(tp.offer_type, tp.estimate * 100, yerr=1.96 * tp.se * 100, fmt="none",
            ecolor="#666666", capsize=4, linewidth=1)
ax.axhline(0, color="#333333", linewidth=0.9)
ax.set_ylabel("72h 购买率增量（相对信息型，pp）")
ax.set_title("分券型增量：并非越深越好")
for i, r in tp.iterrows():
    top = max(r.estimate, r.ci_high) * 100 + 0.6
    ax.text(i, top, f"{r.estimate*100:+.1f}pp", ha="center", va="bottom", fontsize=8)
ax.set_ylim(tp.ci_low.min() * 100 - 1.5, tp.ci_high.max() * 100 + 2.5)
save_fig(fig, os.path.join(FIG, "05_type_effects.png"))

# 7.4 uplift 十分组
fig, ax = plt.subplots(figsize=(9.6, 5.4))
x = dec.decile.to_numpy()
ax.bar(x - 0.18, dec.coupon_rate * 100, width=0.34, color="#1D3A6E", label="券窗口购买率")
ax.bar(x + 0.18, dec.info_rate * 100, width=0.34, color="#B9C6DD", label="信息窗口购买率")
ax.plot(x, dec.gap * 100, color="#F08C4A", marker="o", linewidth=1.8, label="差值（pp）")
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_xticks(x)
ax.set_xlabel("按 T-learner 预测增量从低到高分组（1=最低，10=最高）")
ax.set_ylabel("72h 购买率（%）/ 差值（pp）")
ax.set_title("T-learner uplift 十分组：高增量组的券-信息差距更大")
ax.legend(loc="upper left", ncol=3, fontsize=8)
save_fig(fig, os.path.join(FIG, "05_uplift_deciles.png"))

# 7.5 方案对比：补贴 vs 增量净额
sm = summary.set_index("metric").value
schemes = ["S0 现状", "S1\n03：选择性发 d10", "v2\n只给全响应发 d7"]
subs = [sm["current_subsidy"] / 10000, 41959 / 10000, sm["v2_subsidy"] / 10000]
nets = [sm["current_inc_net_fe"] / 10000, sm["s1_inc_net_05"] / 10000, sm["v2_inc_net"] / 10000]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
ax.bar(schemes, subs, color=["#8C8C8C", "#C0392B", "#1D3A6E"], width=0.6)
for i, v in enumerate(subs):
    ax.text(i, v + 0.4, f"{v:.1f} 万", ha="center", fontsize=8)
ax.set_title("补贴")
ax.set_ylabel("万元")
ax.set_ylim(0, max(subs) * 1.22)
ax = axes[1]
colors3 = ["#C0392B" if v < 0 else "#4CC38A" for v in nets]
ax.bar(schemes, nets, color=colors3, width=0.6)
ax.axhline(0, color="#333333", linewidth=0.9)
for i, v in enumerate(nets):
    ax.text(i, v + (0.35 if v >= 0 else -0.35), f"{v:+.2f} 万", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8)
ax.set_title("增量净额（05 口径）")
ax.set_ylabel("万元")
ax.set_ylim(min(nets) * 1.25, max(nets) * 1.25)
fig.suptitle("账面净额会骗人：S1 的增量净额约 −5.5 万，v2 约 +11.9 万", fontsize=12, y=1.02)
fig.tight_layout()
save_fig(fig, os.path.join(FIG, "05_policy_compare.png"))

print("\n[7/7] 已输出：05_send_windows.csv / 05_balance.csv / 05_estimates.csv / "
      "05_segment_uplift.csv / 05_type_uplift.csv / 05_uplift_deciles.csv / 05_policy_v2.csv / 5 张图")
print("\n策略 v1 → v2（明细）：")
print(pol_v2[["segment_name", "offer_group", "v2_offer", "v2_reason"]].to_string(index=False))
