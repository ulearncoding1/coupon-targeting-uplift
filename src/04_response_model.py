# -*- coding: utf-8 -*-
"""
04_response_model.py —— 券级响应模型（LightGBM + SHAP）

目标：
    预测「一次发券会不会被完成」，输出可解释的驱动因素，并回答：
    什么人、收到什么券、在什么时机，更可能响应？

样本构造：
    一行 = 一次 offer received（76,277 行），标签 = 该次发券后是否发生对应 offer completed。
    同一用户可能多次收到同一张券，按「完成事件回溯到最近一次发券」匹配，避免重复计数。

防泄漏（这是本脚本最重要的设计）：
    1. 不用 01 里的全窗口 comp_rate / monetary 等汇总特征——它们包含当前这张券的结果；
    2. 所有历史特征只取「发券时刻之前」的事件（严格 time < send_time）；
    3. 训练/测试按用户分组切分（GroupShuffleSplit），同一用户不会同时出现在两边；
    4. 标签只用于监督，不作为特征；segment 标签只用于分人群评估，不作为模型输入。

输出：
    04_offer_dataset.csv / 04_metrics.csv / 04_feature_importance.csv /
    04_segment_eval.csv / 04_calibration.csv / 04_lift.csv /
    04_lgbm_model.txt / 04_model_eval.png / 04_lift_chart.png /
    04_feature_importance.png / 04_shap_summary.png
"""

import os
import shutil
import tempfile
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss, precision_recall_curve)

from utils_viz import use_style, save_fig
from config import CLEAN, TABLE, FIG, LOG

warnings.filterwarnings("ignore")
DATA = CLEAN
OUT = TABLE
use_style()

plt.rcParams["figure.dpi"] = 120

# ------------------------------------------------------------------
# 1. 读取与券属性
# ------------------------------------------------------------------
trans = pd.read_csv(os.path.join(DATA, "transcript.csv"))
portfolio = pd.read_csv(os.path.join(DATA, "portfolio.csv"))
profile = pd.read_csv(os.path.join(DATA, "profile.csv"))
seg = pd.read_csv(os.path.join(DATA, "user_segments.csv"))
uf = pd.read_csv(os.path.join(DATA, "user_features.csv"))

pmap = portfolio.rename(columns={"id": "offer_id", "duration": "duration_days",
                                 "difficulty": "offer_difficulty", "reward": "offer_reward"})[
    ["offer_id", "offer_type", "offer_difficulty", "offer_reward", "duration_days"]
]

ev = trans.merge(pmap, on="offer_id", how="left")
ev["amount"] = ev["amount"].fillna(0.0)
ev["is_tx"] = ev["event"] == "transaction"
ev["is_recv"] = ev["event"] == "offer received"
ev["is_view"] = ev["event"] == "offer viewed"
ev["is_comp"] = ev["event"] == "offer completed"

for t in ["bogo", "discount", "informational"]:
    ev[f"is_{t}"] = ev["offer_type"] == t
ev["is_hard"] = ev["offer_difficulty"] >= 10
ev["is_easy"] = (ev["offer_difficulty"] > 0) & (ev["offer_difficulty"] < 10)

print(f"[1/6] 事件 {len(ev):,} 条，发券实例 {ev.is_recv.sum():,} 条")

# ------------------------------------------------------------------
# 2. 标签：完成事件回溯到最近一次发券
# ------------------------------------------------------------------
recv = ev[ev.is_recv].copy().reset_index(drop=True)
recv["instance_id"] = np.arange(len(recv))
comp = ev[ev.is_comp].copy().reset_index(drop=True)
comp["completion_id"] = np.arange(len(comp))

match = comp[["completion_id", "person", "offer_id", "time"]].merge(
    recv[["instance_id", "person", "offer_id", "time", "duration_days"]],
    on=["person", "offer_id"], suffixes=("_c", "_s")
)
match = match[(match.time_s <= match.time_c) &
              (match.time_c <= match.time_s + match.duration_days * 24)]
latest = match.sort_values("time_s").groupby("completion_id").tail(1)
completed_ids = set(latest["instance_id"].unique())
recv["y_completed"] = recv["instance_id"].isin(completed_ids).astype(int)

# 把标签挂回事件表（发券行 (person, offer_id, time) 唯一）
ev = ev.merge(recv[["person", "offer_id", "time", "y_completed"]],
              on=["person", "offer_id", "time"], how="left")

print(f"[2/6] 完成事件 {len(comp):,} 条，全部匹配到发券；正样本实例 {recv.y_completed.sum():,} "
      f"（{recv.y_completed.mean():.1%}）")

# ------------------------------------------------------------------
# 3. 发券时刻的历史特征（严格只看过去）
# ------------------------------------------------------------------
# 静态画像
prof = profile.copy()
prof["became_member_on"] = pd.to_datetime(prof["became_member_on"], errors="coerce")
prof["tenure_base"] = (prof["became_member_on"].max() - prof["became_member_on"]).dt.days
prof["age"] = pd.to_numeric(prof["age"], errors="coerce")
prof["income"] = pd.to_numeric(prof["income"], errors="coerce")
prof["demo_missing"] = prof["age"].isna().astype(int)
for g in ["F", "M", "O", "Unknown"]:
    prof[f"gender_{g}"] = (prof["gender"].fillna("Unknown") == g).astype(int)
prof = prof[["id", "age", "income", "demo_missing", "tenure_base",
             "gender_F", "gender_M", "gender_O", "gender_Unknown"]].rename(columns={"id": "person"})

rows = []
for uid, g in ev.groupby("person", sort=False):
    g = g.sort_values("time", kind="stable")
    t = g["time"].to_numpy(dtype=float)
    tx = g["is_tx"].to_numpy()
    recv_f = g["is_recv"].to_numpy()
    view_f = g["is_view"].to_numpy()
    comp_f = g["is_comp"].to_numpy()
    bogo = g["is_bogo"].to_numpy()
    disc = g["is_discount"].to_numpy()
    info = g["is_informational"].to_numpy()
    hard = g["is_hard"].to_numpy()
    easy = g["is_easy"].to_numpy()
    amount = g["amount"].to_numpy(dtype=float)
    day = g["day"].to_numpy()

    def cs(x):
        return np.r_[0, np.cumsum(np.asarray(x, dtype=float))]

    cum_tx, cum_spend = cs(tx), cs(amount * tx)
    cum_view, cum_comp = cs(view_f), cs(comp_f)
    cum_recv = cs(recv_f)
    cum_rb, cum_cb = cs(recv_f & bogo), cs(comp_f & bogo)
    cum_rd, cum_cd = cs(recv_f & disc), cs(comp_f & disc)
    cum_ri, cum_vi = cs(recv_f & info), cs(view_f & info)
    cum_rh, cum_ch = cs(recv_f & hard), cs(comp_f & hard)
    cum_re, cum_ce = cs(recv_f & easy), cs(comp_f & easy)
    # 活跃天数：每个有交易的 day 只记一次（先只保留交易日的 day，再对 day 去重）
    tx_day = np.where(tx, day, -1)
    day_first = tx & ~pd.Series(tx_day).duplicated().to_numpy()
    cum_active = cs(day_first)

    run_max = np.r_[0, np.maximum.accumulate(np.where(tx, amount, 0))]
    last_tx = np.r_[-1, np.maximum.accumulate(np.where(tx, t, -1))]
    first_recv = np.r_[np.inf, np.minimum.accumulate(np.where(recv_f, t, np.inf))]

    ridx = np.where(recv_f)[0]
    if len(ridx) == 0:
        continue
    ts = t[ridx]
    k = np.searchsorted(t, ts, side="left")          # 严格早于发券的事件数
    k7 = np.searchsorted(t, ts - 168, side="left")   # 7 天窗口起点

    ss = g.iloc[ridx]
    df = pd.DataFrame({
        "person": uid,
        "offer_id": ss["offer_id"].to_numpy(),
        "send_time": ts,
        "send_day": ss["day"].to_numpy(),
        "send_hour": np.mod(ts, 24),
        "seq_recv": cum_recv[k] + 1,

        "n_tx_prior": cum_tx[k],
        "spend_prior": cum_spend[k],
        "avg_order_prior": np.where(cum_tx[k] > 0, cum_spend[k] / np.maximum(cum_tx[k], 1), np.nan),
        "max_order_prior": run_max[k],
        "n_active_days_prior": cum_active[k],
        "recency_h": np.where(last_tx[k] >= 0, ts - last_tx[k], 999),
        "tx_7d": cum_tx[k] - cum_tx[k7],
        "spend_7d": cum_spend[k] - cum_spend[k7],

        "n_recv_prior": cum_recv[k],
        "n_view_prior": cum_view[k],
        "n_comp_prior": cum_comp[k],
        "view_rate_prior": np.where(cum_recv[k] > 0, cum_view[k] / np.maximum(cum_recv[k], 1), np.nan),
        "comp_rate_prior": np.where(cum_rb[k] + cum_rd[k] > 0,
                                    cum_comp[k] / np.maximum(cum_rb[k] + cum_rd[k], 1), np.nan),
        "bogo_recv_prior": cum_rb[k],
        "bogo_comp_prior": cum_cb[k],
        "disc_recv_prior": cum_rd[k],
        "disc_comp_prior": cum_cd[k],
        "pref_discount_prior": np.where((cum_rb[k] > 0) & (cum_rd[k] > 0),
                                        cum_cd[k] / np.maximum(cum_rd[k], 1) - cum_cb[k] / np.maximum(cum_rb[k], 1),
                                        np.nan),
        "info_recv_prior": cum_ri[k],
        "info_view_prior": cum_vi[k],
        "n_types_seen": (cum_rb[k] > 0).astype(int) + (cum_rd[k] > 0).astype(int) + (cum_ri[k] > 0).astype(int),
        "days_since_first_offer": (ts - np.minimum(first_recv[k], ts)) / 24,
        "hard_recv_prior": cum_rh[k],
        "hard_comp_prior": cum_ch[k],
        "easy_recv_prior": cum_re[k],
        "easy_comp_prior": cum_ce[k],

        "is_bogo": ss["is_bogo"].to_numpy().astype(int),
        "is_discount": ss["is_discount"].to_numpy().astype(int),
        "is_info": ss["is_informational"].to_numpy().astype(int),
        "difficulty": ss["offer_difficulty"].to_numpy(),
        "reward": ss["offer_reward"].to_numpy(),
        "net_benefit": ss["offer_difficulty"].to_numpy() - ss["offer_reward"].to_numpy(),
        "duration_days": ss["duration_days"].to_numpy(),
        "discount_rate": np.where(ss["offer_difficulty"].to_numpy() > 0,
                                  ss["offer_reward"].to_numpy() / ss["offer_difficulty"].to_numpy(), np.nan),
        "y_completed": ss["y_completed"].to_numpy(),
    })
    # 门槛相对客单价：>1 说明这张券的门槛高于该用户平均一单
    df["difficulty_to_aov"] = np.minimum(
        df["difficulty"] / (df["avg_order_prior"].fillna(0) + 0.5), 100)
    df["threshold_gap"] = df["difficulty"] - df["avg_order_prior"].fillna(0)
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
data = data.merge(prof, on="person", how="left")
data["tenure_days"] = data["tenure_base"].fillna(0) + data["send_time"] / 24
data = data.drop(columns=["tenure_base"])

# 分人群标签只用于评估，不进模型
data = data.merge(seg[["user_id", "segment_name"]].rename(columns={"user_id": "person"}),
                  on="person", how="left")

print(f"[3/6] 券级特征表：{data.shape[0]:,} 行 × {data.shape[1]} 列，"
      f"正样本率 {data.y_completed.mean():.1%}")

# ------------------------------------------------------------------
# 4. 特征、切分、训练
# ------------------------------------------------------------------
FEATURES = [
    "age", "income", "demo_missing",
    "gender_F", "gender_M", "gender_O", "gender_Unknown", "tenure_days",
    "n_tx_prior", "spend_prior", "avg_order_prior", "max_order_prior",
    "n_active_days_prior", "recency_h", "tx_7d", "spend_7d",
    "n_recv_prior", "n_view_prior", "n_comp_prior", "view_rate_prior", "comp_rate_prior",
    "bogo_recv_prior", "bogo_comp_prior", "disc_recv_prior", "disc_comp_prior",
    "pref_discount_prior", "info_recv_prior", "info_view_prior",
    "n_types_seen", "days_since_first_offer",
    "hard_recv_prior", "hard_comp_prior", "easy_recv_prior", "easy_comp_prior",
    "is_bogo", "is_discount", "difficulty", "reward", "net_benefit",
    "duration_days", "discount_rate", "difficulty_to_aov", "threshold_gap",
    "send_hour", "send_day", "seq_recv",
]
DISPLAY = {
    "age": "年龄", "income": "收入", "demo_missing": "人口属性缺失",
    "gender_F": "性别 F", "gender_M": "性别 M", "gender_O": "性别 O", "gender_Unknown": "性别未知",
    "tenure_days": "会员天数",
    "n_tx_prior": "历史交易笔数", "spend_prior": "历史消费额", "avg_order_prior": "历史客单价",
    "max_order_prior": "历史最大单笔", "n_active_days_prior": "历史活跃天数",
    "recency_h": "距上次交易小时", "tx_7d": "近7天交易笔数", "spend_7d": "近7天消费额",
    "n_recv_prior": "历史收券数", "n_view_prior": "历史查看数", "n_comp_prior": "历史完成数",
    "view_rate_prior": "历史查看率", "comp_rate_prior": "历史可完成券完成率",
    "bogo_recv_prior": "历史BOGO收券", "bogo_comp_prior": "历史BOGO完成",
    "disc_recv_prior": "历史折扣收券", "disc_comp_prior": "历史折扣完成",
    "pref_discount_prior": "历史折扣偏好", "info_recv_prior": "历史信息型收券",
    "info_view_prior": "历史信息型查看", "n_types_seen": "历史券种数",
    "days_since_first_offer": "距首次收券天数",
    "hard_recv_prior": "历史高门槛收券", "hard_comp_prior": "历史高门槛完成",
    "easy_recv_prior": "历史低门槛收券", "easy_comp_prior": "历史低门槛完成",
    "is_bogo": "BOGO券", "is_discount": "折扣券", "is_info": "信息型券",
    "difficulty": "门槛", "reward": "补贴", "net_benefit": "单次净额",
    "duration_days": "有效期(天)", "discount_rate": "补贴/门槛",
    "difficulty_to_aov": "门槛/客单价", "threshold_gap": "门槛-客单价",
    "send_hour": "发送时段", "send_day": "活动第几天", "seq_recv": "第几次收券",
}

# 只对「可完成券」（BOGO + 折扣）建模：信息型券按规则 p=0，不参与训练，
# 否则 is_info 会成为一个"白送"的强特征，把 AUC 虚高到 0.91。
md = data[data.is_info == 0].copy()
X = md[FEATURES].astype(float)
y = md["y_completed"].astype(int)
groups = md["person"]

# 测试集按用户分组 20%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
# 训练集内部再按用户切 15% 做早停
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=7)
sub_idx, val_idx = next(gss2.split(X.iloc[train_idx], y.iloc[train_idx], groups=groups.iloc[train_idx]))
tr = train_idx[sub_idx]
va = train_idx[val_idx]

params = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
    min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, lambda_l2=1.0, verbosity=-1, seed=42, num_threads=4,
)
dtrain = lgb.Dataset(X.iloc[tr], y.iloc[tr], feature_name=FEATURES)
dval = lgb.Dataset(X.iloc[va], y.iloc[va], reference=dtrain)
model0 = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval],
                   callbacks=[lgb.early_stopping(80, verbose=False)])
best_iter = model0.best_iteration or 800
model = lgb.train(params, lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx], feature_name=FEATURES),
                  num_boost_round=best_iter)
print(f"[4/6] 可完成券样本 {len(md):,}（占全部发券 {len(md)/len(data):.0%}）；"
      f"训练完成：best_iter={best_iter}，训练用户 {groups.iloc[train_idx].nunique():,}，"
      f"测试用户 {groups.iloc[test_idx].nunique():,}")

pred_test = model.predict(X.iloc[test_idx])
pred_md = model.predict(X)
# 信息型券规则上不可能完成，直接置 0
data["p_complete"] = 0.0
data.loc[md.index, "p_complete"] = pred_md

metrics = []
def add(name, value):
    metrics.append({"metric": name, "value": value})
add("n_train", len(train_idx))
add("n_test", len(test_idx))
add("positive_rate_completable", float(y.mean()))
add("auc", roc_auc_score(y.iloc[test_idx], pred_test))
add("pr_auc", average_precision_score(y.iloc[test_idx], pred_test))
add("baseline_pr_auc", float(y.iloc[test_idx].mean()))
add("brier", brier_score_loss(y.iloc[test_idx], pred_test))
add("logloss", log_loss(y.iloc[test_idx], pred_test))

prec, rec, thr = precision_recall_curve(y.iloc[test_idx], pred_test)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = float(thr[np.argmax(f1[:-1])])
best_f1 = float(np.max(f1[:-1]))
add("best_threshold", best_thr)
add("best_f1", best_f1)

# top decile lift
test_df = md.iloc[test_idx].copy()
test_df["p"] = pred_test
test_df = test_df.sort_values("p", ascending=False)
top10 = test_df.head(int(len(test_df) * 0.1))
add("top10_lift", float(top10.y_completed.mean() / test_df.y_completed.mean()))
add("top10_capture", float(top10.y_completed.sum() / test_df.y_completed.sum()))

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(OUT, "04_metrics.csv"), index=False, encoding="utf-8-sig")
print("\n[5/6] 模型指标")
print(metrics_df.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 5. 分人群评估 / 校准 / 提升
# ------------------------------------------------------------------
seg_eval = []
for s, g in test_df.groupby("segment_name"):
    both = g.y_completed.nunique() == 2
    seg_eval.append({
        "segment_name": s, "n": len(g),
        "actual_rate": g.y_completed.mean(),
        "pred_rate": g.p.mean(),
        "auc": roc_auc_score(g.y_completed, g.p) if both and len(g) >= 20 else np.nan,
    })
seg_eval = pd.DataFrame(seg_eval).sort_values("n", ascending=False)
seg_eval.to_csv(os.path.join(OUT, "04_segment_eval.csv"), index=False, encoding="utf-8-sig")
print("\n分人群：实际完成率 vs 模型预测")
print(seg_eval.round(3).to_string(index=False))

# 校准分箱
test_df["bin"] = pd.qcut(test_df.p, 10, duplicates="drop")
calib = test_df.groupby("bin", observed=True).agg(
    n=("p", "size"), pred=("p", "mean"), actual=("y_completed", "mean")
).reset_index(drop=True)
calib["bin_id"] = np.arange(1, len(calib) + 1)
calib.to_csv(os.path.join(OUT, "04_calibration.csv"), index=False, encoding="utf-8-sig")

# 累计提升
lift = test_df[["y_completed", "p"]].copy()
lift["frac"] = np.arange(1, len(lift) + 1) / len(lift)
lift["capture"] = lift.y_completed.cumsum() / lift.y_completed.sum()
lift_curve = lift.iloc[::max(1, len(lift) // 300)]
lift_curve.to_csv(os.path.join(OUT, "04_lift.csv"), index=False, encoding="utf-8-sig")

# 特征重要性
imp = pd.DataFrame({
    "feature": FEATURES,
    "gain": model.feature_importance("gain"),
    "split": model.feature_importance("split"),
})
imp["gain_pct"] = imp.gain / imp.gain.sum()
imp = imp.sort_values("gain", ascending=False)

# SHAP（取测试集 3,000 条解释）
sample = md.iloc[test_idx].sample(min(3000, len(test_idx)), random_state=42)
X_sample = sample[FEATURES].astype(float)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_sample)
if isinstance(sv, list):
    sv_pos = np.asarray(sv[1])
elif np.ndim(sv) == 3:
    sv_pos = np.asarray(sv)[:, :, 1]
else:
    sv_pos = np.asarray(sv)

shap_mean = np.abs(sv_pos).mean(axis=0)
imp = imp.merge(pd.DataFrame({"feature": FEATURES, "shap_mean_abs": shap_mean}), on="feature", how="left")
imp.to_csv(os.path.join(OUT, "04_feature_importance.csv"), index=False, encoding="utf-8-sig")

# 把预测分数写回全量样本
out_cols = ["person", "offer_id", "send_time", "send_day", "send_hour", "seq_recv",
            "segment_name", "y_completed"] + FEATURES
scored = data[out_cols].copy()
scored["p_complete"] = data["p_complete"]
scored["split"] = "not_completable"
scored.loc[md.index, "split"] = "train"
scored.loc[md.index[test_idx], "split"] = "test"
scored.to_csv(os.path.join(OUT, "04_offer_dataset.csv"), index=False, encoding="utf-8-sig")

# LightGBM 的 C++ 文件写入在中文路径下会失败，先写临时 ASCII 路径再移动
tmp_model = os.path.join(tempfile.gettempdir(), "04_lgbm_model.txt")
model.save_model(tmp_model)
shutil.move(tmp_model, os.path.join(LOG, "04_lgbm_model.txt"))

# ------------------------------------------------------------------
# 6. 图表
# ------------------------------------------------------------------
# 6.1 ROC / PR / 校准
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
ax = axes[0]
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y.iloc[test_idx], pred_test)
ax.plot(fpr, tpr, color="#1D3A6E", linewidth=2.2, label=f"LightGBM (AUC={roc_auc_score(y.iloc[test_idx], pred_test):.3f})")
ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1, label="随机")
ax.set_xlabel("假阳性率")
ax.set_ylabel("真阳性率")
ax.set_title("A. ROC 曲线")
ax.legend(loc="lower right")

ax = axes[1]
ax.plot(rec[::-1], prec[::-1], color="#F08C4A", linewidth=2.2,
        label=f"LightGBM (AP={average_precision_score(y.iloc[test_idx], pred_test):.3f})")
ax.axhline(y.iloc[test_idx].mean(), color="#AAAAAA", linestyle="--", linewidth=1,
           label=f"基准 ({y.iloc[test_idx].mean():.2f})")
ax.set_xlabel("召回率")
ax.set_ylabel("精确率")
ax.set_title("B. PR 曲线")
ax.legend(loc="upper right")

ax = axes[2]
ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1)
ax.scatter(calib.pred, calib.actual, s=calib.n / calib.n.max() * 160 + 25,
           color="#4CC38A", edgecolor="#1D3A6E", linewidth=0.8, zorder=3)
ax.set_xlabel("预测完成率")
ax.set_ylabel("实际完成率")
ax.set_title("C. 校准：预测 vs 实际")
for _, r in calib.iterrows():
    ax.annotate(f"{r.actual:.0%}", (r.pred, r.actual), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7.5, color="#333333")
fig.suptitle("券级响应模型：判别力与校准", fontsize=14, y=1.03)
fig.tight_layout()
save_fig(fig, os.path.join(FIG, "04_model_eval.png"))

# 6.2 累计提升
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot(lift.frac, lift.capture, color="#1D3A6E", linewidth=2.2, label="模型排序")
ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1, label="随机")
top10_frac = 0.1
top10_cap = test_df.head(int(len(test_df) * top10_frac)).y_completed.sum() / test_df.y_completed.sum()
ax.scatter([top10_frac], [top10_cap], color="#F08C4A", s=60, zorder=3)
ax.annotate(f"前10%捕获 {top10_cap:.0%}", (top10_frac, top10_cap), textcoords="offset points",
            xytext=(8, -18), fontsize=9, color="#F08C4A")
ax.set_xlabel("按预测分数从高到低覆盖的用户比例")
ax.set_ylabel("累计捕获的完成次数占比")
ax.set_title("累计提升曲线：模型把高响应人群排在前面")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
save_fig(fig, os.path.join(FIG, "04_lift_chart.png"))

# 6.3 分人群校准（模型在零完成人群的高估问题）
def short_seg(s):
    v, b = s.split("·")
    return v.replace("价值", "") + "·" + b.replace("响应型", "").replace("型", "")

sp = seg_eval.sort_values("actual_rate").copy()
fig, ax = plt.subplots(figsize=(10.6, 5.2))
yy = np.arange(len(sp))
ax.barh(yy - 0.18, sp.actual_rate, height=0.34, color="#B9C6DD", label="实际完成率")
ax.barh(yy + 0.18, sp.pred_rate, height=0.34, color="#F08C4A", label="模型预测")
ax.set_yticks(yy)
ax.set_yticklabels([short_seg(s) for s in sp.segment_name], fontsize=9)
ax.set_xlabel("完成率")
ax.set_title("分人群校准：模型在零完成/低活跃人群明显高估")
for i, (a, p) in enumerate(zip(sp.actual_rate, sp.pred_rate)):
    ax.text(max(a, p) + 0.012, i, f"实际 {a:.0%} / 预测 {p:.0%}", va="center", fontsize=8, color="#333333")
ax.set_xlim(0, 1.02)
ax.legend(loc="lower right")
ax.grid(axis="x")
save_fig(fig, os.path.join(FIG, "04_segment_response.png"))

# 6.4 特征重要性（gain + SHAP）
top_gain = imp.head(15).iloc[::-1]
top_shap = imp.sort_values("shap_mean_abs", ascending=False).head(15).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
ax = axes[0]
ax.barh([DISPLAY[f] for f in top_gain.feature], top_gain.gain_pct, color="#5B8FF9", height=0.68)
ax.set_xlabel("增益占比")
ax.set_title("A. LightGBM 增益重要性（Top 15）")
for i, (v, f) in enumerate(zip(top_gain.gain_pct, top_gain.feature)):
    ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=8, color="#333333")
ax.set_xlim(0, top_gain.gain_pct.max() * 1.18)
ax.grid(axis="x")

ax = axes[1]
ax.barh([DISPLAY[f] for f in top_shap.feature], top_shap.shap_mean_abs, color="#4CC38A", height=0.68)
ax.set_xlabel("平均 |SHAP|")
ax.set_title("B. SHAP 平均绝对贡献（Top 15）")
ax.grid(axis="x")
fig.suptitle("什么在驱动券的完成：模型视角", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, os.path.join(FIG, "04_feature_importance.png"))

# 6.5 SHAP beeswarm
plt.figure(figsize=(9.6, 6.8))
shap.summary_plot(sv_pos, X_sample, feature_names=[DISPLAY[f] for f in FEATURES],
                  max_display=15, show=False, plot_type="dot", color_bar_label="特征值")
ax = plt.gca()
ax.set_title("SHAP 值分布：每个特征把完成概率推高还是拉低", fontsize=13, pad=14)
save_fig(plt.gcf(), os.path.join(FIG, "04_shap_summary.png"))

print("\nTop 10 增益特征：")
print(imp.head(10)[["feature", "gain_pct", "shap_mean_abs"]].round(3).to_string(index=False))

print("\n[6/6] 已输出：04_offer_dataset.csv / 04_metrics.csv / 04_feature_importance.csv / "
      "04_segment_eval.csv / 04_calibration.csv / 04_lift.csv / 04_lgbm_model.txt / 5 张图")
