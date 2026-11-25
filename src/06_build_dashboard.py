# -*- coding: utf-8 -*-
"""
06_build_dashboard.py
把 02–05 的分析结果和 07–09 的验证结果预生成一个单文件 HTML 看板。
不依赖 Streamlit，不依赖 fetch/CSV；双击 outputs/看板/06_dashboard.html 即可打开。
构建顺序：02–05 → 07–09 → 06（看板必须最后构建）。
"""
from pathlib import Path
import base64
import json
import pandas as pd
from PIL import Image

from config import TABLE, FIG, DASH, ASSETS, ROOT

OUT = TABLE


def read_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(
            f"缺少看板输入：{path}\n"
            "请先运行 07–09 生成验证数据，或直接运行 python src/run_all.py。"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def records(df: pd.DataFrame) -> list[dict]:
    # to_json 会把 NaN 写成 null，避免 json.dumps 产生非法 NaN
    return json.loads(df.to_json(force_ascii=False, orient="records"))


def data_uri(name: str) -> str:
    p = FIG / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def panel_uri(src_name: str, out_name: str, box: tuple[float, float, float, float]) -> str:
    """从多面板原图中裁出单面板，供看板全宽显示；箱线按原图比例，重建时自动更新。"""
    src = FIG / src_name
    im = Image.open(src).convert("RGB")
    w, h = im.size
    left, top, right, bottom = box
    crop = im.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))
    out_dir = FIG / "dashboard"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / out_name
    crop.save(out, optimize=True)
    return data_uri(f"dashboard/{out_name}")


# ---------- 数据 ----------
segment = read_csv("02_segment_profile.csv")
value = read_csv("02_value_profile.csv")
behavior = read_csv("02_behavior_profile.csv")
audit = read_csv("03_current_audit.csv")
offer = read_csv("03_offer_matrix.csv")
policy_v1 = read_csv("03_policy.csv")
scenario = read_csv("03_scenario_compare.csv")
metrics = read_csv("04_metrics.csv")
segment_eval = read_csv("04_segment_eval.csv")
lift = read_csv("04_lift.csv")
calibration = read_csv("04_calibration.csv")
seg_uplift = read_csv("05_segment_uplift.csv")
type_uplift = read_csv("05_type_uplift.csv")
seg_type = read_csv("05_segment_type_uplift.csv")
deciles = read_csv("05_uplift_deciles.csv")
policy_v2 = read_csv("05_policy_v2.csv")
policy_summary = read_csv("05_policy_v2_summary.csv")
uplift_metrics = read_csv("05_uplift_metrics.csv")
sql_validation = read_csv("sql/_validation.csv")
ab_power = read_csv("08_ab_power.csv")
psi = read_csv("09_psi.csv")
model_decay = read_csv("09_model_decay.csv")
coupon_monitor = read_csv("09_coupon_monitor.csv")
alerts = read_csv("09_alerts.csv")

metric_dict = {str(k): (None if pd.isna(v) else float(v)) for k, v in zip(metrics.metric, metrics.value)}
summary_dict = {str(k): (None if pd.isna(v) else float(v)) for k, v in zip(policy_summary.metric, policy_summary.value)}
uplift_metric_dict = {str(k): (None if pd.isna(v) else float(v)) for k, v in zip(uplift_metrics.metric, uplift_metrics.value)}

DATA = {
    "kpi": {
        "users": 17000,
        "events": 306534,
        "sends": 76277,
        "completions": 33579,
        "overallAnyPp": 2.36,
        "overallGmv": 1.478,
        "v2Net": summary_dict.get("v2_inc_net"),
        "s1Net": summary_dict.get("s1_inc_net_05"),
        "currentNet": summary_dict.get("current_inc_net_fe"),
        "auc": metric_dict.get("auc"),
        "top20Gap": 14.38,
    },
    "segments": records(segment),
    "value": records(value),
    "behavior": records(behavior),
    "audit": records(audit),
    "offer": records(offer),
    "policyV1": records(policy_v1),
    "scenario": records(scenario),
    "metrics": metric_dict,
    "segmentEval": records(segment_eval),
    "calibration": records(calibration),
    "segUplift": records(seg_uplift),
    "typeUplift": records(type_uplift),
    "segType": records(seg_type),
    "deciles": records(deciles),
    "policyV2": records(policy_v2),
    "summary": summary_dict,
    "upliftMetrics": uplift_metric_dict,
    "sqlValidation": records(sql_validation),
    "sqlQueryCount": len(list((ROOT / "sql").glob("*.sql"))),
    "abPower": records(ab_power),
    "psi": records(psi),
    "modelDecay": records(model_decay),
    "couponMonitor": records(coupon_monitor),
    "alerts": records(alerts),
}

IMAGES = {
    "behavior_profile": data_uri("02_behavior_profile.png"),
    "model_eval_roc": panel_uri("04_model_eval.png", "04_model_eval_roc.png", (0.0, 0.09, 0.334, 1.0)),
    "model_eval_pr": panel_uri("04_model_eval.png", "04_model_eval_pr.png", (0.334, 0.09, 0.6655, 1.0)),
    "model_eval_cal": panel_uri("04_model_eval.png", "04_model_eval_cal.png", (0.6655, 0.09, 1.0, 1.0)),
    "lift_chart": data_uri("04_lift_chart.png"),
    "feature_importance_gain": panel_uri("04_feature_importance.png", "04_feature_importance_gain.png", (0.0, 0.08, 0.5095, 1.0)),
    "feature_importance_shap": panel_uri("04_feature_importance.png", "04_feature_importance_shap.png", (0.5095, 0.08, 1.0, 1.0)),
    "shap_summary": data_uri("04_shap_summary.png"),
    "seg_effects": data_uri("05_segment_effects.png"),
    "mechanism": data_uri("05_mechanism.png"),
    "type_effects": data_uri("05_type_effects.png"),
    "uplift_deciles": data_uri("05_uplift_deciles.png"),
    "policy_compare": data_uri("05_policy_compare.png"),
}

ECHARTS = (ASSETS / "echarts.min.js").read_text(encoding="utf-8").replace("</script>", "<\\/script>")

# ---------- HTML ----------
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>优惠券精准投放与增量效果评估</title>
<style>
:root{
  --bg:#F7F5EF; --paper:#FFFDF8; --ink:#1C1C1A; --muted:#6F695F; --muted2:#827B70;
  --navy:#2F4A6D; --navy2:#3C526F;
  --green:#2F5D50; --red:#8A3B45; --ochre:#B06A3C; --plum:#5B5478; --teal:#3F6B73;
  --gold:#B06A3C; --line:#D8D3C7; --soft:#EEEAE0;
  --accent-overview:#2F4A6D; --accent-segments:#2F5D50; --accent-offers:#8F5B36;
  --accent-budget:#7E3038; --accent-response:#514A70; --accent-uplift:#35636A; --accent-validation:#6A5D45;
  --serif:"Source Han Serif SC","Noto Serif SC","STSong","SimSun",serif;
  --sans:"Microsoft YaHei","PingFang SC","Noto Sans SC","Segoe UI",sans-serif;
  --mono:"Consolas","Cascadia Mono",monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.68;font-size:14px;overflow-x:hidden}
a{color:inherit}
button,input,select{font-family:inherit}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:236px;background:rgba(247,245,239,.97);border-right:1px solid var(--line);color:var(--ink);padding:30px 24px 26px;display:flex;flex-direction:column;gap:0;z-index:20}
.brand{font-size:11px;letter-spacing:.05em;color:var(--muted);font-weight:500}
.side-title{font-family:var(--serif);font-size:26px;line-height:1.4;margin:12px 0 8px;font-weight:600;letter-spacing:-.01em}
.side-sub{margin:0;color:var(--muted);font-size:12px;line-height:1.65}
.nav{margin-top:26px;display:flex;flex-direction:column;gap:0;border-top:1px solid var(--line)}
.nav-item{display:flex;gap:12px;align-items:center;border:0;border-left:2px solid transparent;border-bottom:1px solid var(--line);background:transparent;color:var(--muted);padding:11px 0 11px 12px;border-radius:0;text-align:left;cursor:pointer;font-size:13px;transition:none}
.nav-item:hover{color:var(--ink);border-left-color:var(--line)}
.nav-item[data-view=overview]{--nav-accent:var(--accent-overview)}
.nav-item[data-view=segments]{--nav-accent:var(--accent-segments)}
.nav-item[data-view=offers]{--nav-accent:var(--accent-offers)}
.nav-item[data-view=budget]{--nav-accent:var(--accent-budget)}
.nav-item[data-view=response]{--nav-accent:var(--accent-response)}
.nav-item[data-view=uplift]{--nav-accent:var(--accent-uplift)}
.nav-item[data-view=validation]{--nav-accent:var(--accent-validation)}
.nav-item.active{background:transparent;color:var(--ink);font-weight:700;border-left-color:var(--nav-accent,var(--navy))}
.nav-index{width:auto;height:auto;border-radius:0;background:none;font-family:var(--mono);font-size:10.5px;color:var(--nav-accent,var(--muted));opacity:.6;letter-spacing:.04em;flex:0 0 auto}
.nav-item.active .nav-index{background:none;color:var(--nav-accent,var(--navy));font-weight:700;opacity:1}
.side-note{margin-top:auto;font-size:11px;color:var(--muted2);border-top:1px solid var(--line);padding-top:14px;line-height:1.85}
.main{margin-left:236px;padding:38px 44px 72px;max-width:1420px}
.hero{background:transparent;border:0;border-bottom:1px solid var(--line);border-radius:0;padding:0 0 24px;position:relative;overflow:visible}
.hero:after{display:none}
.eyebrow{font-size:12px;letter-spacing:.04em;color:var(--muted);font-weight:400}
.hero h2{font-family:var(--serif);font-size:38px;line-height:1.3;margin:12px 0 10px;font-weight:600;letter-spacing:-.015em}
.hero p{color:var(--muted);max-width:900px;margin:0;font-size:14px;line-height:1.75}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.badge{font-size:11px;padding:3px 9px;border-radius:0;background:transparent;border:1px solid var(--line);color:var(--muted)}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:20px 0 30px;border-top:2px solid var(--view-accent,var(--navy))}
.kpi{background:transparent;border:0;border-bottom:1px solid var(--line);border-radius:0;padding:14px 18px 14px 0}
.kpi:not(:first-child){border-left:1px solid var(--line);padding-left:18px}
.kpi .k{font-size:11px;color:var(--muted);letter-spacing:.02em}
.kpi .v{font-family:var(--serif);font-size:26px;font-weight:600;line-height:1.3;margin-top:4px}
#kpiGrid .kpi:nth-child(1) .v{color:var(--accent-overview)}
#kpiGrid .kpi:nth-child(2) .v{color:var(--accent-segments)}
#kpiGrid .kpi:nth-child(3) .v{color:var(--accent-offers)}
#kpiGrid .kpi:nth-child(4) .v{color:var(--accent-budget)}
.view .kpi .v{color:var(--view-accent,var(--ink))}
.kpi .s{font-size:11px;color:var(--muted);margin-top:2px}
.view{display:none}
.view.active{display:block}
#view-overview{--view-accent:var(--accent-overview)}
#view-segments{--view-accent:var(--accent-segments)}
#view-offers{--view-accent:var(--accent-offers)}
#view-budget{--view-accent:var(--accent-budget)}
#view-response{--view-accent:var(--accent-response)}
#view-uplift{--view-accent:var(--accent-uplift)}
#view-validation{--view-accent:var(--accent-validation)}
.section-head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin:36px 0 18px;border-top:2px solid var(--view-accent,var(--line));padding-top:18px}
.section-head h3{font-family:var(--serif);font-size:25px;font-weight:600;margin:0;letter-spacing:-.01em}
.section-head p{margin:4px 0 0;color:var(--muted);font-size:12.5px;max-width:840px;line-height:1.7}
.grid-1{display:grid;grid-template-columns:1fr;gap:18px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:2px;padding:20px 22px;box-shadow:none}
.card h4{font-family:var(--serif);font-size:16px;font-weight:600;margin:0 0 12px;color:var(--ink);letter-spacing:0}
.card h4:after{content:"";display:block;width:26px;height:2px;background:var(--view-accent,var(--navy));margin-top:8px;opacity:.9}
.card p{margin:0;color:var(--muted);font-size:12.5px;line-height:1.7}
.callout{border:0;border-left:2px solid var(--line);border-radius:0;padding:2px 0 2px 14px;font-size:12.5px;background:transparent;line-height:1.75}
.callout.green{border-left-color:var(--green);background:transparent}
.callout.red{border-left-color:var(--red);background:transparent}
.callout.navy{border-left-color:var(--navy);background:transparent}
.callout.gold{border-left-color:var(--gold);background:transparent}
.muted{color:var(--muted)}
.small{font-size:12px}
.pill{display:inline-block;padding:2px 8px;border-radius:2px;font-size:11px;background:transparent;border:1px solid var(--line);color:var(--muted)}
.pill.green{background:transparent;border-color:transparent;color:var(--green);padding-left:0}
.pill.red{background:transparent;border-color:transparent;color:var(--red);padding-left:0}
.pill.navy{background:transparent;border-color:transparent;color:var(--view-accent,var(--muted));padding:0;font-family:var(--mono);font-size:11px;letter-spacing:.06em}
.table-wrap{width:100%;overflow-x:auto;padding-bottom:2px}
.data-table{width:100%;min-width:620px;border-collapse:collapse;font-size:12.5px;background:transparent}
.data-table th{background:#F1EDE3;background:color-mix(in srgb, var(--view-accent, #33383A) 9%, var(--paper));color:var(--ink);font-weight:600;text-align:left;padding:9px 10px;border-top:2px solid var(--view-accent,var(--line));border-bottom:1px solid var(--line);white-space:nowrap;font-size:11.5px}
.data-table td{padding:9px 10px;border-bottom:1px solid #E6E1D5;vertical-align:top;background:var(--paper)}
.data-table tbody tr:nth-child(even) td{background:#F3F0E7}
.data-table tr:hover td{background:#EDE8DC}
.data-table td:first-child,.data-table th:first-child{white-space:nowrap}
.num{font-variant-numeric:tabular-nums}
.heatmap{height:460px;width:100%;max-width:900px}
.chart{height:420px;width:100%;max-width:900px}
.grid-2>*,.grid-3>*,.card{min-width:0}
.chart-img{width:100%;max-width:900px;border:0;border-radius:0;background:transparent;display:block;cursor:zoom-in;filter:saturate(1) contrast(1);mix-blend-mode:multiply}
.chart-img:hover{filter:saturate(1) contrast(1)}
.matrix-table{min-width:0;table-layout:fixed}
.matrix-table th{white-space:normal;font-size:10.5px;line-height:1.25;padding:8px 6px}
.matrix-table td{text-align:center;min-width:0;padding:8px 6px;font-size:11px}
.matrix-table th:first-child,.matrix-table td:first-child{width:82px}
.matrix-n{font-size:16px;font-weight:700;color:var(--view-accent,var(--navy))}
.matrix-s{font-size:11px;color:var(--muted)}
.matrix-cell.empty{color:var(--muted2);background:transparent}
.profile-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
.profile-item{background:transparent;border:0;border-top:1px solid var(--line);border-radius:0;padding:10px 0 0}
.profile-item .k{font-size:10px;color:var(--muted)}
.profile-item .v{font-size:15px;font-weight:700;margin-top:2px}
.controls{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.control{background:transparent;border:0;border-top:1px solid var(--line);border-radius:0;padding:12px 0 0}
.control label{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:7px;gap:10px}
.control input[type=range]{width:100%;accent-color:var(--view-accent,var(--navy))}
.control input[type=number]{width:100%;padding:7px 8px;border:1px solid var(--line);border-radius:2px;font-size:13px;background:var(--paper);color:var(--ink)}
.control .val{font-weight:700;color:var(--view-accent,var(--navy))}
.subtabs{display:flex;gap:24px;flex-wrap:wrap;margin:0 0 18px;border-bottom:1px solid var(--line)}
.subtab{border:0;border-bottom:2px solid transparent;border-radius:0;background:transparent;color:var(--muted);padding:8px 0 7px;font-size:12.5px;cursor:pointer}
.subtab.active{background:transparent;border-bottom-color:var(--view-accent,var(--navy));color:var(--ink);font-weight:600}
.panel{display:none}
.panel.active{display:block}
.eval-panel{display:none}
.eval-panel.active{display:block}
.note{font-size:11px;color:var(--muted);margin-top:8px;line-height:1.65}
.pipeline{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
.step{background:transparent;border:0;border-top:1px solid var(--line);border-radius:0;padding:12px 0 0}
.step .n{font-family:var(--mono);font-size:10.5px;color:var(--view-accent,var(--navy));font-weight:600;letter-spacing:.04em}
.step .t{font-weight:700;font-size:13px;margin:4px 0 2px}
.step .d{font-size:11px;color:var(--muted);line-height:1.6}
.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.insight{background:transparent;border:0;border-top:1px solid var(--line);border-radius:0;padding:14px 0 0}
.insight .tag{font-size:11px;color:var(--view-accent,var(--muted));letter-spacing:.06em;font-weight:600}
.insight h5{font-family:var(--serif);font-size:16px;font-weight:600;margin:8px 0 6px}
.insight p{font-size:12.5px;color:var(--muted);margin:0;line-height:1.7}
.footer{margin-top:38px;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:11px;line-height:1.8}
.lightbox{position:fixed;inset:0;background:rgba(28,28,26,.9);display:none;place-items:center;padding:34px;z-index:90}
.lightbox.open{display:grid}
.lightbox img{max-width:94vw;max-height:86vh;background:#fff;border:1px solid #fff;box-shadow:none;mix-blend-mode:normal;filter:none}
.lightbox-close{position:absolute;top:18px;right:22px;width:36px;height:36px;border:1px solid rgba(255,255,255,.35);background:transparent;color:#fff;font-size:22px;line-height:1;cursor:pointer}
select{border-radius:2px!important;background:var(--paper)!important;color:var(--ink)!important;border:1px solid var(--line)!important}
@media(max-width:1100px){
  .sidebar{width:210px;padding:26px 18px}
  .main{margin-left:210px;padding:30px 28px 60px}
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  .kpi:not(:first-child){border-left:0;padding-left:0}
  .grid-2,.grid-3,.controls,.insight-grid{grid-template-columns:1fr}
  .pipeline{grid-template-columns:repeat(3,1fr)}
}
@media(max-width:820px){
  .sidebar{position:static;width:auto;border-right:0;border-bottom:1px solid var(--line);padding:24px 22px}
  .main{margin-left:0;padding:24px 22px 60px}
  .nav{display:grid;grid-template-columns:repeat(2,1fr);border-top:0;margin-top:18px}
  .nav-item{border-left:1px solid var(--line);border-bottom:0;padding:8px 10px}
  .hero h2{font-size:28px}
  .profile-grid{grid-template-columns:repeat(2,1fr)}
  .pipeline{grid-template-columns:repeat(2,1fr)}
  .kpi-grid{grid-template-columns:1fr}
  .kpi{border-left:0!important;padding-left:0!important}
}

/* ===== 视觉改版：蓝色渐变仪表盘 ===== */
body{
  background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
  background-attachment:fixed;
  font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
}
.sidebar{
  background:linear-gradient(180deg,#1e3c72 0%,#2a5298 58%,#182848 100%);
  border-right:0;
  box-shadow:6px 0 22px rgba(24,40,72,.16);
  color:#fff;
}
.brand{color:rgba(255,255,255,.72)}
.side-title{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#fff;font-size:24px;line-height:1.38}
.side-sub{color:rgba(255,255,255,.76)}
.nav{border-top-color:rgba(255,255,255,.18)}
.nav-item{color:rgba(255,255,255,.72);border-bottom-color:rgba(255,255,255,.12)}
.nav-item:hover{color:#fff;border-left-color:rgba(255,255,255,.36)}
.nav-item.active{background:rgba(255,255,255,.12);color:#fff;border-left-color:#7ee8c2}
.nav-index{color:rgba(255,255,255,.52)}
.nav-item.active .nav-index{color:#7ee8c2}
.side-note{color:rgba(255,255,255,.58);border-top-color:rgba(255,255,255,.18)}
.main{padding:30px 38px 64px;max-width:1480px}
.hero{
  background:linear-gradient(90deg,#1e3c72 0%,#2a5298 68%,#4b6cb7 100%);
  border:0;border-radius:15px;padding:30px 34px;
  box-shadow:0 8px 22px rgba(30,60,114,.18);color:#fff;
}
.hero .eyebrow{color:rgba(255,255,255,.72)}
.hero h2{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#fff;font-size:36px;letter-spacing:-.01em}
.hero p{color:rgba(255,255,255,.82)}
.badge{background:rgba(255,255,255,.14);border-color:rgba(255,255,255,.24);color:#fff;border-radius:999px;padding:4px 11px}
.kpi-grid{gap:14px;border-top:0}
.kpi{
  background:#fff;border:1px solid #e3e8ef!important;border-top:3px solid var(--view-accent,#2a5298)!important;
  border-radius:12px;padding:16px 18px;box-shadow:0 4px 12px rgba(30,60,114,.06);
}
.kpi:not(:first-child){padding-left:18px}
.kpi .k{color:#6b7a90}
.kpi .v{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#1e3c72}
.kpi .s{color:#6b7a90}
.section-head{border-top:0;margin:36px 0 16px;padding:0}
.section-head h3{
  font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#1e3c72;
  position:relative;padding-left:14px;
}
.section-head h3:before{
  content:"";position:absolute;left:0;top:5px;bottom:4px;width:4px;border-radius:2px;
  background:linear-gradient(180deg,#4b6cb7,#00b09b);
}
.section-head p{color:#6b7a90}
.card{
  background:#fff;border:1px solid #e3e8ef;border-radius:15px;
  box-shadow:0 6px 18px rgba(30,60,114,.08);padding:22px 24px;overflow:hidden;
}
.card h4{
  font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:15px;
  background:linear-gradient(90deg,#4b6cb7,#182848);color:#fff;
  padding:13px 17px;margin:-22px -24px 16px;border-radius:14px 14px 0 0;
}
.card h4:after{display:none}
.card p,.note{color:#6b7a90}
.callout{border-left:4px solid #cbd5e1;background:#f8fafc;border-radius:10px;padding:12px 15px}
.callout.green{border-left-color:#43a047;background:#eef8ef}
.callout.red{border-left-color:#e53935;background:#ffebee}
.callout.navy{border-left-color:#2a5298;background:#e8f1fb}
.callout.gold{border-left-color:#f0a500;background:#fff8e1}
.insight{background:#f8fafc;border:1px solid #e3e8ef;border-radius:12px;padding:14px 16px}
.insight .tag{color:#2a5298}
.step{background:#f8fafc;border:1px solid #e3e8ef;border-radius:10px;padding:12px 13px}
.control{background:#f8fafc;border:1px solid #e3e8ef;border-radius:10px;padding:12px 14px}
.profile-item{background:#f8fafc;border:1px solid #e3e8ef;border-radius:10px;padding:10px 12px}
.panel,.eval-panel{background:transparent}
.table-wrap{max-height:560px;overflow:auto;border-radius:10px;border:1px solid #e3e8ef;padding-bottom:0}
.data-table{min-width:620px;background:#fff}
.data-table th{
  background:#2c3e50;color:#fff;border:0;position:sticky;top:0;z-index:3;
}
.data-table td{background:#fff;border:1px solid #e9edf3}
.data-table tbody tr:nth-child(even) td{background:#f8f9fa}
.data-table tr:hover td{background:#e9f7fe}
.matrix-table{min-width:0}
.pill{border-radius:999px;background:#eef3fb;border:0;color:#2a5298;padding:3px 9px}
.pill.green{background:#e8f5e9;color:#2e7d32;padding:3px 9px}
.pill.red{background:#ffebee;color:#c62828;padding:3px 9px}
.pill.navy{background:#eef3fb;color:#2a5298;font-family:"Consolas","Cascadia Mono",monospace;padding:3px 9px}
select{
  border-radius:8px!important;border:1px solid #dbe3ee!important;
  background:#fff!important;color:#1e3c72!important;box-shadow:0 2px 8px rgba(30,60,114,.04);
}
.control input[type=number]{border-radius:8px;border-color:#dbe3ee;background:#fff}
.chart-img{border-radius:10px}
.footer{border-top:1px solid #dbe3ee;color:#6b7a90;text-align:center;padding-top:18px}
@media(max-width:820px){
  .main{padding:20px 16px 50px}
  .hero{padding:24px 20px}
  .hero h2{font-size:28px}
  .card{padding:18px 18px}
  .card h4{margin:-18px -18px 14px;padding:12px 14px}
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  .kpi{padding:14px 15px!important}
  .section-head h3{font-size:22px}
}
@media(max-width:560px){
  .kpi-grid{grid-template-columns:1fr}
}

</style>
</head>
<body>
<div class="sidebar">
  <div class="brand">Coupon Targeting · 2026</div>
  <h1 class="side-title">优惠券精准投放<br>与增量效果评估</h1>
  <p class="side-sub">Starbucks Promotions · 策略分析</p>
  <nav class="nav" id="nav">
    <button class="nav-item active" data-view="overview"><span class="nav-index">00</span>项目总览</button>
    <button class="nav-item" data-view="segments"><span class="nav-index">01</span>人群分层</button>
    <button class="nav-item" data-view="offers"><span class="nav-index">02</span>券型矩阵</button>
    <button class="nav-item" data-view="budget"><span class="nav-index">03</span>预算模拟</button>
    <button class="nav-item" data-view="response"><span class="nav-index">04</span>响应排序</button>
    <button class="nav-item" data-view="uplift"><span class="nav-index">05</span>增量评估</button>
    <button class="nav-item" data-view="validation"><span class="nav-index">06</span>验证与落地</button>
  </nav>
  <div class="side-note">30 天活动 · 17,000 用户<br>76,277 次发券 · 306,534 事件<br>数据源：Kaggle Starbucks Offers</div>
</div>

<main class="main">
  <section class="hero">
    <div class="eyebrow">优惠券投放策略 · 增量评估 · 预算决策</div>
    <h2>从“谁会买”到“因为券才买”</h2>
    <p>用信息型券作“同触达准对照”，结合用户固定效应与 T-learner，把券的增量拆到人群 × 券型；再用“增量 GMV − 补贴”而不是账面 ROI，决定预算发给谁。</p>
    <div class="badges">
      <span class="badge">9+1 人群</span>
      <span class="badge">6 种券型</span>
      <span class="badge">72h / 168h 窗口</span>
      <span class="badge">AUC 0.866</span>
      <span class="badge">v2 增量净额 +11.96 万</span>
      <span class="badge">SQL 34 项</span>
      <span class="badge">A/B 设计 5,542 人</span>
    </div>
  </section>

  <div class="kpi-grid" id="kpiGrid"></div>

  <!-- ================= 00 总览 ================= -->
  <section class="view active" id="view-overview">
    <div class="section-head">
      <div><h3>项目总览</h3><p>先把问题讲清楚：为什么“完成率高”不等于“券有效”？</p></div>
      <span class="pill navy">00 / 07</span>
    </div>

    <div class="insight-grid">
      <div class="insight"><div class="tag">问题</div><h5>完成率 ≠ 增量</h5><p>03 中全响应人群完成率 100%，容易被看成券带来了全部购买；但信息型券本身也会提醒购买。没有反事实，就无法判断“谁是因为券才买”。</p></div>
      <div class="insight"><div class="tag">方法</div><h5>信息型券作准对照</h5><p>同一用户既收到过券、也收到过信息型券；比较 72h 购买率/GMV，并用用户固定效应吸收稳定画像。</p></div>
      <div class="insight"><div class="tag">判断</div><h5>账面净额会高估</h5><p>S1 账面净额 +16.8 万，05 同口径 −3.08 万；v2 是回溯上限测算，增量净额 +11.96 万，不是可上线规则。</p></div>
    </div>

    <div class="card" style="margin-top:18px">
      <h4>分析流程：从解析到策略</h4>
      <div class="pipeline">
        <div class="step"><div class="n">00</div><div class="t">解析</div><div class="d">JSONL 三件套 → 事件表，修正字段名/缺失。</div></div>
        <div class="step"><div class="n">01</div><div class="t">特征</div><div class="d">RFM + 券行为，17,000×52；可完成券口径。</div></div>
        <div class="step"><div class="n">02</div><div class="t">分层</div><div class="d">价值 KMeans k=3 + 行为 GMM k=4 → 9+1 人群。</div></div>
        <div class="step"><div class="n">03</div><div class="t">策略 v1</div><div class="d">人群×券型矩阵；S1 只给选择性发 d10。</div></div>
        <div class="step"><div class="n">04</div><div class="t">响应模型</div><div class="d">LightGBM AUC 0.866；只排序，不证明增量。</div></div>
        <div class="step"><div class="n">05</div><div class="t">增量评估</div><div class="d">信息型准对照 + 用户 FE + T-learner；v2 只发 d7。</div></div>
      </div>
    </div>

    <div class="grid-2" style="margin-top:18px">
      <div class="card"><h4>三组关键对比</h4>
        <div class="callout green" style="margin-bottom:10px"><b>① 全响应不是“伪增量”</b><br>高·全响应 +9.6pp、中·全响应 +20.5pp；不发券时仍有 69%/37.5% 会买。</div>
        <div class="callout gold" style="margin-bottom:10px"><b>② 选择性“有转化但不赚钱”</b><br>高·选择性 d7 +7.6pp，但增量 GMV 1.87 元 &lt; 期望补贴 2.52 元，净额 −0.65。</div>
        <div class="callout red"><b>③ 账面净额 ≠ 增量净额</b><br>S1 账面 +16.8 万，05 同口径 −3.08 万；v2 回溯测算 +11.96 万。</div>
      </div>
      <div class="card"><h4>v2 预算对比</h4><img class="chart-img" src="__IMG_policy_compare__" alt="S0/S1/v2 补贴与增量净额对比"><p class="note">补贴：S0 16.5 万 → S1 4.2 万 → v2 7.2 万；增量净额：S0 −6.3 万 → S1 −3.08 万 → v2 +11.96 万。</p></div>
    </div>

    <div class="card" style="margin-top:18px">
      <h4>核心数字速览</h4>
      <div id="overviewTable"></div>
    </div>
  </section>

  <!-- ================= 01 人群分层 ================= -->
  <section class="view" id="view-segments">
    <div class="section-head"><div><h3>人群分层</h3><p>价值维 × 行为维：先回答“谁值得运营”，再回答“谁值得发券”。</p></div><span class="pill navy">01 / 07</span></div>
    <div class="grid-2">
      <div class="card"><h4>3×4 人群矩阵（人数 / 占比）</h4><div class="table-wrap" id="segmentMatrix"></div><p class="note">空组合显示“—”；矩阵中同时给出人数、占比与可完成券完成率。完成率是收到券后的条件概率，不是增量。</p></div>
      <div class="card"><h4>人群画像</h4><select id="segmentSelect" style="width:100%;padding:8px;border:1px solid var(--line);border-radius:2px;background:#fff;margin-bottom:10px"></select><div id="segmentProfile"></div></div>
    </div>
    <div class="grid-1" style="margin-top:18px">
      <div class="card"><h4>行为簇画像</h4><img class="chart-img" src="__IMG_behavior_profile__" alt="行为四簇箱线图与折扣偏好"><p class="note">全响应簇完成率恒为 100%，选择性簇约 52%，零完成簇为 0，低活跃簇均值 7.5%；折扣偏好主要集中在选择性响应型。</p></div>
    </div>
  </section>

  <!-- ================= 02 券型矩阵 ================= -->
  <section class="view" id="view-offers">
    <div class="section-head"><div><h3>券型矩阵</h3><p>同一人群，发哪种券最划算？看完成率，也看每千次净 GMV。</p></div><span class="pill navy">02 / 07</span></div>
    <div class="card">
      <label class="small muted">选择人群：</label>
      <select id="offerSelect" style="width:100%;max-width:420px;padding:8px;border:1px solid var(--line);border-radius:2px;background:#fff;margin:6px 0 12px"></select>
      <div id="offerTable"></div>
    </div>
    <div class="grid-1" style="margin-top:18px">
      <div class="card"><h4>每千次净 GMV 热力图</h4><div id="offerHeatNet" class="heatmap"></div><p class="note">BOGO 按 reward=difficulty 计，净 GMV 为 0；折扣券才有正净额。</p></div>
      <div class="card"><h4>完成率热力图</h4><div id="offerHeatRate" class="heatmap"></div><p class="note">完成率是“收到券后完成”的条件概率，不是增量。</p></div>
    </div>
    <div class="card" style="margin-top:18px"><h4>03 现状审计</h4><div id="auditTable"></div></div>
  </section>

  <!-- ================= 03 预算模拟 ================= -->
  <section class="view" id="view-budget">
    <div class="section-head"><div><h3>预算模拟</h3><p>只给两个全响应人群发 d7；改补贴/触达量，看增量净额怎么变。</p></div><span class="pill navy">03 / 07</span></div>
    <div class="grid-2">
      <div class="card"><h4>可调参数</h4>
        <div class="controls">
          <div class="control"><label>高·全响应触达 <span class="val" id="vHigh">21,685</span></label><input id="highSends" type="range" min="0" max="30000" step="1" value="21685" oninput="budgetCompute()"></div>
          <div class="control"><label>中·全响应触达 <span class="val" id="vMid">2,649</span></label><input id="midSends" type="range" min="0" max="8000" step="1" value="2649" oninput="budgetCompute()"></div>
          <div class="control"><label>d7 补贴/次（元） <span class="val" id="vSub">3.0</span></label><input id="unitSub" type="range" min="1" max="6" step="0.5" value="3" oninput="budgetCompute()"></div>
          <div class="control"><label>增量 GMV/次（元） <span class="val" id="vGmvHigh">8.06</span></label><input id="gmvHigh" type="range" min="0" max="12" step="0.01" value="8.06" oninput="budgetCompute()"><label style="margin-top:6px">中·全响应增量 GMV/次 <span class="val" id="vGmvMid">6.46</span></label><input id="gmvMid" type="range" min="0" max="12" step="0.01" value="6.46" oninput="budgetCompute()"></div>
        </div>
        <p class="note">默认值来自 05 的分人群×券型点估计：高·全响应 d7 +8.059 元/次，中·全响应 d7 +6.461 元/次。测算是外推，不含毛利/固定成本。</p>
      </div>
      <div class="card"><h4>你的方案</h4>
        <div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);margin:0">
          <div class="kpi"><div class="k">补贴</div><div class="v" id="bSub">—</div><div class="s">元</div></div>
          <div class="kpi"><div class="k">增量 GMV</div><div class="v" id="bGmv">—</div><div class="s">元</div></div>
          <div class="kpi"><div class="k">增量净额</div><div class="v" id="bNet">—</div><div class="s">增量 GMV − 补贴</div></div>
          <div class="kpi"><div class="k">增量 ROI</div><div class="v" id="bRoi">—</div><div class="s">增量 GMV / 补贴</div></div>
        </div>
        <div class="callout green" style="margin-top:14px" id="budgetHint">发券次数与补贴在默认值时，接近 05 的 v2：期望核销补贴 7.2 万、增量净额 11.96 万。</div>
      </div>
    </div>
    <div class="card" style="margin-top:18px"><h4>S0 / S1 / v2 对比</h4><div id="budgetChart" class="chart"></div></div>
    <div class="card" style="margin-top:18px"><h4>v2 策略表</h4><div id="policyV2Table"></div><p class="note">“信息型”表示只提醒、不发购买激励；未触达 6 人不计入预算。</p></div>
  </section>

  <!-- ================= 04 响应排序 ================= -->
  <section class="view" id="view-response">
    <div class="section-head"><div><h3>响应排序</h3><p>04 的 LightGBM 能排序“会用券的人”，但不能证明券有增量。</p></div><span class="pill navy">04 / 07</span></div>
    <div class="kpi-grid" id="modelKpi"></div>
    <div class="grid-1">
      <div class="card">
        <h4>模型评估：ROC / PR / 校准</h4>
        <div class="subtabs" id="evalTabs">
          <button class="subtab active" data-eval="roc">ROC</button>
          <button class="subtab" data-eval="pr">PR</button>
          <button class="subtab" data-eval="cal">校准</button>
        </div>
        <div class="eval-panel active" id="eval-roc"><img class="chart-img" src="__IMG_model_eval_roc__" alt="ROC 曲线"></div>
        <div class="eval-panel" id="eval-pr"><img class="chart-img" src="__IMG_model_eval_pr__" alt="PR 曲线"></div>
        <div class="eval-panel" id="eval-cal"><img class="chart-img" src="__IMG_model_eval_cal__" alt="校准曲线"></div>
        <p class="note">按标签切换；单张图已放大，点击可再看大图。</p>
      </div>
      <div class="card"><h4>累计提升曲线</h4><img class="chart-img" src="__IMG_lift_chart__" alt="累计提升曲线"></div>
    </div>
    <div class="grid-1" style="margin-top:18px">
      <div class="card">
        <h4>特征解释：Gain、平均 |SHAP|、SHAP 分布</h4>
        <div class="subtabs" id="impTabs">
          <button class="subtab active" data-imp="gain">Gain 重要性</button>
          <button class="subtab" data-imp="shap">平均 |SHAP|</button>
          <button class="subtab" data-imp="shapdist">SHAP 分布</button>
        </div>
        <div class="eval-panel active" id="imp-gain"><img class="chart-img" src="__IMG_feature_importance_gain__" alt="Gain 特征重要性"></div>
        <div class="eval-panel" id="imp-shap"><img class="chart-img" src="__IMG_feature_importance_shap__" alt="平均 SHAP"></div>
        <div class="eval-panel" id="imp-shapdist"><img class="chart-img" src="__IMG_shap_summary__" alt="SHAP 分布"></div>
        <p class="note">Gain 看分裂贡献，平均 |SHAP| 看幅度，SHAP 分布看特征把完成概率推高还是拉低。</p>
      </div>
    </div>
    <div class="grid-2" style="margin-top:18px">
      <div class="card"><h4>分人群校准</h4><div id="segmentEvalTable"></div><p class="note">全局校准约 2pp，但分人群偏差可达 30pp：零完成人群被高估。</p></div>
      <div class="card"><h4>校准曲线</h4><div id="calibrationChart" class="chart"></div></div>
    </div>
    <div class="callout red" style="margin-top:18px"><b>关键边界：</b>04 的 p_complete 是“收到券后完成”的条件概率；它不能回答“因为券才完成”。05 才用信息型券作准对照估计增量。模型适合在“已经决定发券”的人群里排序，不适合单独决定“发不发”。</div>
  </section>

  <!-- ================= 05 增量评估 ================= -->
  <section class="view" id="view-uplift">
    <div class="section-head"><div><h3>增量评估</h3><p>券相对纯提醒的增量；分人群、分券型、分人群×券型、T-learner 排序。</p></div><span class="pill navy">05 / 07</span></div>
    <div class="kpi-grid" id="upliftKpi"></div>
    <div class="subtabs" id="upliftTabs">
      <button class="subtab active" data-subtab="seg">分人群 CATE</button>
      <button class="subtab" data-subtab="type">分券型</button>
      <button class="subtab" data-subtab="cross">人群 × 券型</button>
      <button class="subtab" data-subtab="tlearner">T-learner</button>
      <button class="subtab" data-subtab="policy">v2 策略</button>
    </div>

    <div class="panel active" id="panel-seg">
      <div class="grid-1">
        <div class="card"><h4>分人群增量：点估计与 95% CI</h4><img class="chart-img" src="__IMG_seg_effects__" alt="分人群 forest plot"></div>
        <div class="card"><h4>门槛 / 客单价 vs 增量</h4><img class="chart-img" src="__IMG_mechanism__" alt="门槛与客单比机制图"></div>
      </div>
      <div class="card" style="margin-top:18px"><h4>分人群明细</h4><div id="segUpliftTable"></div><p class="note">当前券型组合下的平均 CATE；高·选择性平均为负，但拆到 d7 后购买率增量为 +7.6pp。</p></div>
    </div>

    <div class="panel" id="panel-type">
      <div class="grid-2">
        <div class="card"><h4>分券型增量</h4><img class="chart-img" src="__IMG_type_effects__" alt="分券型增量柱状图"></div>
        <div class="card"><h4>读法</h4><div class="callout green"><b>d7 最优：</b>+8.14pp、增量净额 +1.23/次、ROI 1.61，整体净额最高；d10 也为正（+0.63/次、ROI 1.52）。</div><div class="callout red" style="margin-top:10px"><b>d20 显著为负：</b>−8.44pp；门槛 20 元高于多数人一单客单，用户会等/凑/放弃。</div><div class="callout gold" style="margin-top:10px"><b>BOGO 贵：</b>BOGO 10/10 购买率 +6.76pp，但期望补贴 4.55 元，增量净额 −2.23。</div></div>
      </div>
      <div class="card" style="margin-top:18px"><h4>分券型明细</h4><div id="typeUpliftTable"></div></div>
    </div>

    <div class="panel" id="panel-cross">
      <div class="card">
        <label class="small muted">选择人群：</label>
        <select id="crossSelect" style="width:100%;max-width:420px;padding:8px;border:1px solid var(--line);border-radius:2px;background:#fff;margin:6px 0 12px"></select>
        <div class="grid-2">
          <div><div id="crossTable"></div></div>
          <div><div id="crossChart" class="chart"></div></div>
        </div>
        <p class="note">这张表回答“不是发不发，而是发哪种”：高·选择性 d7 +7.6pp，但增量净额 −0.65；d20 −22.6pp。</p>
      </div>
    </div>

    <div class="panel" id="panel-tlearner">
      <div class="grid-2">
        <div class="card"><h4>T-learner 十分组</h4><div id="decileChart" class="chart"></div></div>
        <div class="card"><h4>验收指标</h4><div id="tlearnerMetrics"></div><p class="note">Top 20% gap +14.38pp，是整体 +2.40pp 的 6 倍；但第 3 组有回落，不能当完美单调。</p></div>
      </div>
      <div class="card" style="margin-top:18px"><h4>十分组明细</h4><div id="decileTable"></div></div>
    </div>

    <div class="panel" id="panel-policy">
      <div class="grid-2">
        <div class="card"><h4>v2 策略表</h4><div id="policyV2TableUplift"></div></div>
        <div class="card"><h4>预算汇总</h4><div id="policySummaryCards"></div><p class="note">S1 的 +16.8 万是账面净额；05 同口径 −3.08 万。v2 是回溯上限测算，增量净额 +11.96 万，不是可上线规则。</p></div>
      </div>
      <div class="card" style="margin-top:18px"><h4>S0 / S1 / v2 增量净额</h4><img class="chart-img" src="__IMG_policy_compare__" alt="S0/S1/v2 对比"></div>
    </div>
  </section>

  <!-- ================= 06 验证与落地 ================= -->
  <section class="view" id="view-validation">
    <div class="section-head"><div><h3>验证与落地</h3><p>分析结论之外，补三层可验收证据：SQL 对账、A/B 样本量、离线监控。</p></div><span class="pill navy">06 / 07</span></div>

    <div class="kpi-grid" id="validationKpi"></div>

    <div class="grid-2">
      <div class="card">
        <h4>SQL 对账：34 项全过</h4>
        <div id="sqlSummary"></div>
        <div id="sqlTable"></div>
        <p class="note">9 条查询覆盖事件、用户、分层、券矩阵、增量、质量与策略；任何一项不过，策略数字就不应被引用。</p>
      </div>
      <div class="card">
        <h4>A/B 实验设计：合并 5,542 人</h4>
        <div id="abSummary"></div>
        <div id="abTable"></div>
        <p class="note">MDE 是“能检测到的最小效应”，不是预期收益；合并样本的净额 power 99.5%，高/中价值单看仍不够。</p>
      </div>
    </div>

    <div class="grid-2" style="margin-top:18px">
      <div class="card"><h4>2.5 元净额目标：现有样本 vs 所需样本</h4><div id="abChart" class="chart"></div><p class="note">合并样本缺 3,488 人；高价值缺 5,071 人，中价值缺 280 人。</p></div>
      <div class="card"><h4>监控：模型衰减与券效</h4><div id="monitorTable"></div><p class="note">D7–13 为基准；D0–6 是冷启动，不作为模型衰减结论。</p></div>
    </div>

    <div class="grid-2" style="margin-top:18px">
      <div class="card"><h4>PSI：分数稳定，历史行为是回看伪影</h4><div id="psiTable"></div><p class="note">p_complete PSI 约 0.06，属于 stable；spend_prior 等高 PSI 先核查 30 天回看口径，不直接判线上漂移。</p></div>
      <div class="card"><h4>告警：12 条，全部 watch</h4><div id="alertTable"></div><p class="note">没有模型、券效或日数据 actionable；12 条均为 check_lookback。</p></div>
    </div>

    <div class="callout red" style="margin-top:18px"><b>边界：</b>Spark 未实跑、A/B 不是线上实验、监控是离线回放；这一页是验收证据，不是生产系统。</div>
  </section>

  <div class="footer">优惠券精准投放与增量效果评估 · 策略分析 · 00–09 管线 · 分析 + 验证 · 本页为单文件 HTML，图表与数据已内嵌；点击图表可放大。</div>
</main>

<div class="lightbox" id="lightbox" aria-hidden="true">
  <button class="lightbox-close" type="button" aria-label="关闭">×</button>
  <img id="lightboxImg" alt="放大图表">
</div>

<script>
__ECHARTS__
</script>
<script>
const DATA = __DATA__;
const IMG = __IMAGES__;
</script>
<script>
/* ================= 工具函数 ================= */
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m]));}
function fmtInt(v){return (v==null||Number.isNaN(Number(v)))?"—":Number(v).toLocaleString("zh-CN",{maximumFractionDigits:0});}
function fmt1(v){return (v==null||Number.isNaN(Number(v)))?"—":Number(v).toFixed(1);}
function fmt2(v){return (v==null||Number.isNaN(Number(v)))?"—":Number(v).toFixed(2);}
function pct(v,d=1){return (v==null||Number.isNaN(Number(v)))?"—":(Number(v)*100).toFixed(d)+"%";}
function pp(v,d=1){return (v==null||Number.isNaN(Number(v)))?"—":(Number(v)*100>=0?"+":"")+(Number(v)*100).toFixed(d)+"pp";}
function ppFromPp(v,d=1){return (v==null||Number.isNaN(Number(v)))?"—":(Number(v)>=0?"+":"")+Number(v).toFixed(d)+"pp";}
function money(v,d=0){return (v==null||Number.isNaN(Number(v)))?"—":Number(v).toLocaleString("zh-CN",{maximumFractionDigits:d});}
function wan(v,d=2){return (v==null||Number.isNaN(Number(v)))?"—":(Number(v)/10000).toFixed(d)+" 万";}
function table(headers, rows){
  return `<div class="table-wrap"><table class="data-table"><thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td class="num">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function optionBase(){return {animation:false,textStyle:{fontFamily:'"Microsoft YaHei","PingFang SC",sans-serif'},color:["#2F4A6D","#2F5D50","#B06A3C","#8A3B45","#5B5478"]};}
let charts={};
function initChart(id, option){
  if(!document.getElementById(id)) return;
  if(!charts[id]){
    const dom=document.getElementById(id);
    charts[id]=echarts.init(dom,null,{renderer:"svg"});
    if(window.ResizeObserver){
      new ResizeObserver(()=>{ if(charts[id]) charts[id].resize(); }).observe(dom);
    }
  }
  charts[id].setOption(Object.assign(optionBase(),option),true);
}
function resizeCharts(){Object.values(charts).forEach(c=>c.resize());}
function showSubtab(name){
  document.querySelectorAll("#upliftTabs .subtab").forEach(b=>b.classList.toggle("active",b.dataset.subtab===name));
  document.querySelectorAll(".panel").forEach(p=>p.classList.toggle("active",p.id==="panel-"+name));
  requestAnimationFrame(()=>resizeCharts());
  setTimeout(resizeCharts,200);
}

/* ================= 00 总览 ================= */
function renderOverview(){
  const k=DATA.kpi;
  document.getElementById("kpiGrid").innerHTML = [
    ["用户数",fmtInt(k.users),"30 天活动窗口"],
    ["发券次数",fmtInt(k.sends),"券 61,042 / 信息 15,235"],
    ["整体增量",(k.overallAnyPp>=0?"+":"")+k.overallAnyPp+"pp","券 − 信息型，72h"],
    ["v2 增量净额",wan(k.v2Net),"05 同口径"]
  ].map(([a,b,c])=>`<div class="kpi"><div class="k">${a}</div><div class="v">${b}</div><div class="s">${c}</div></div>`).join("");
  const rows=[
    ["整体增量（72h 购买率）","+2.36pp（SE 0.47）","用户 FE；朴素 +3.03pp"],
    ["整体增量（72h GMV）","+1.478 元/次","信息型本身已带来大部分购买"],
    ["最值得发券人群","高·全响应 / 中·全响应","d7 增量净额 +5.08 / +3.59 元/次"],
    ["最需拆分的人群","高·选择性","当前组合 −4.7pp；d7 +7.6pp 但净额 −0.65"],
    ["v2 补贴 / 增量净额","7.2 万 / +11.96 万","增量 ROI 2.655；回溯上限"],
    ["S1 账面 / 增量","+16.8 万 / −3.08 万","账面净额会骗人；口径不同"]
  ];
  document.getElementById("overviewTable").innerHTML = table(["指标","数值","说明"],rows);
}

/* ================= 01 人群分层 ================= */
const VALUE_ORDER=["高价值","中价值","低价值"];
const BEHAVIOR_ORDER=["全响应型","选择性响应型","零完成型","低活跃型"];
function renderSegments(){
  const smap={}; DATA.segments.forEach(s=>smap[s.value_label+"|"+s.behavior_label]=s);
  let html='<table class="data-table matrix-table"><thead><tr><th>价值 \\ 行为</th>'+BEHAVIOR_ORDER.map(b=>`<th>${b}</th>`).join("")+"</tr></thead><tbody>";
  VALUE_ORDER.forEach(v=>{
    html+=`<tr><td style="font-weight:700;text-align:left">${v}</td>`;
    BEHAVIOR_ORDER.forEach(b=>{
      const s=smap[v+"|"+b];
      html+=s?`<td><div class="matrix-n">${fmtInt(s.n)}</div><div class="matrix-s">占比 ${pct(s.share)}</div><div class="matrix-s">完成 ${pct(s.comp_rate)}</div></td>`:`<td class="matrix-cell empty">—</td>`;
    });
    html+="</tr>";
  });
  html+="</tbody></table>";
  document.getElementById("segmentMatrix").innerHTML=html;
  const sel=document.getElementById("segmentSelect");
  sel.innerHTML=DATA.segments.map(s=>`<option value="${esc(s.value_label+"·"+s.behavior_label)}">${esc(s.value_label)}·${s.behavior_label}（${fmtInt(s.n)} 人）</option>`).join("");
  sel.value="高价值·全响应型";
  sel.onchange=()=>renderSegmentProfile(sel.value);
  renderSegmentProfile(sel.value);
}
function renderSegmentProfile(name){
  const s=DATA.segments.find(x=>x.value_label+"·"+x.behavior_label===name);
  if(!s) return;
  document.getElementById("segmentProfile").innerHTML=`
    <div class="profile-grid">
      <div class="profile-item"><div class="k">人数 / 占比</div><div class="v">${fmtInt(s.n)} <span class="small muted">(${pct(s.share)})</span></div></div>
      <div class="profile-item"><div class="k">客单价中位</div><div class="v">${fmt1(s.monetary_median)} 元</div></div>
      <div class="profile-item"><div class="k">交易笔数中位</div><div class="v">${fmtInt(s.n_tx_median)} 笔</div></div>
      <div class="profile-item"><div class="k">查看率</div><div class="v">${pct(s.view_rate)}</div></div>
      <div class="profile-item"><div class="k">可完成券完成率</div><div class="v">${pct(s.comp_rate)}</div></div>
      <div class="profile-item"><div class="k">折扣偏好</div><div class="v">${fmt2(s.pref_discount)}</div></div>
    </div>
    <p class="note">完成率是“收到可完成券后完成”的条件概率，不是“因为券才买”的增量。</p>`;
}

/* ================= 02 券型矩阵 ================= */
const OFFER_ORDER=["BOGO 5/5","BOGO 10/10","折扣 7-3","折扣 10-2","折扣 20-5","信息型"];
function segNames(){return DATA.audit.filter(r=>r.segment_name!=="中价值·未触达").map(r=>r.segment_name).reverse();}
function renderOffers(){
  const sel=document.getElementById("offerSelect");
  sel.innerHTML=DATA.audit.filter(r=>r.segment_name!=="中价值·未触达").map(r=>`<option value="${esc(r.segment_name)}">${esc(r.segment_name)}</option>`).join("");
  sel.value="高价值·选择性响应型";
  sel.onchange=()=>renderOfferTable(sel.value);
  renderOfferTable(sel.value);
  const segs=["高价值·全响应型","高价值·选择性响应型","高价值·零完成型","中价值·全响应型","中价值·选择性响应型","中价值·零完成型","中价值·低活跃型","低价值·低活跃型","低价值·零完成型"];
  const netData=[], rateData=[];
  DATA.offer.forEach(r=>{
    const xi=OFFER_ORDER.indexOf(r.offer_group), yi=segs.indexOf(r.segment_name);
    if(xi>=0&&yi>=0){netData.push([xi,yi,r.net_per_1k]);rateData.push([xi,yi,r.comp_rate]);}
  });
  const base={tooltip:{position:"top"},grid:{left:142,right:72,top:18,bottom:50},xAxis:{type:"category",data:OFFER_ORDER,axisLabel:{rotate:28,fontSize:11}},yAxis:{type:"category",data:segs,inverse:true,axisLabel:{fontSize:11}},visualMap:{orient:"vertical",right:0,top:36,bottom:36,calculable:true,textStyle:{fontSize:11},itemWidth:12,itemHeight:110}};
  initChart("offerHeatNet",Object.assign({},base,{visualMap:Object.assign({},base.visualMap,{min:0,max:16000,inRange:{color:["#F4F1E9","#E3E8DF","#AFC1B4","#6F8A78","#2F5D50"]}}),series:[{type:"heatmap",data:netData,label:{show:false},emphasis:{itemStyle:{borderColor:"#fff",borderWidth:1}}}]}));
  initChart("offerHeatRate",Object.assign({},base,{visualMap:Object.assign({},base.visualMap,{min:0,max:1,inRange:{color:["#F4F1E9","#E8D6D4","#C99A96","#9B5E60","#8A3B45"]}}),series:[{type:"heatmap",data:rateData,label:{show:true,fontSize:11.5,formatter:p=>p.value[2]==null?"":(p.value[2]*100).toFixed(0)+"%"} ,emphasis:{itemStyle:{borderColor:"#fff",borderWidth:1}}}]}));
  document.getElementById("auditTable").innerHTML=table(
    ["人群","人数","触达","完成","触达→完成","补贴","账面净额","ROI","增量净额*"],
    DATA.audit.map(r=>[esc(r.segment_name),fmtInt(r.users),fmtInt(r.sends),fmtInt(r.completions),pct(r.comp_rate),fmtInt(r.subsidy),fmtInt(r.net),fmt2(r.roi),fmtInt(r.net_incr)])
  );
}
function renderOfferTable(name){
  const rows=DATA.offer.filter(r=>r.segment_name===name).slice().sort((a,b)=>OFFER_ORDER.indexOf(a.offer_group)-OFFER_ORDER.indexOf(b.offer_group));
  document.getElementById("offerTable").innerHTML=table(
    ["券型","发送","完成","完成率","每千次净 GMV","每发送净额"],
    rows.map(r=>[esc(r.offer_group),fmtInt(r.sends),fmtInt(r.comps),pct(r.comp_rate),fmtInt(r.net_per_1k),fmt2(r.net_per_send)])
  );
}

/* ================= 03 预算模拟 ================= */
function budgetCompute(){
  const high=+document.getElementById("highSends").value;
  const mid=+document.getElementById("midSends").value;
  const sub=+document.getElementById("unitSub").value;
  const gHigh=+document.getElementById("gmvHigh").value;
  const gMid=+document.getElementById("gmvMid").value;
  document.getElementById("vHigh").textContent=fmtInt(high);
  document.getElementById("vMid").textContent=fmtInt(mid);
  document.getElementById("vSub").textContent=sub.toFixed(1);
  document.getElementById("vGmvHigh").textContent=gHigh.toFixed(2);
  document.getElementById("vGmvMid").textContent=gMid.toFixed(2);
  const sends=high+mid;
  const subsidy=sends*sub;
  const gmv=high*gHigh+mid*gMid;
  const net=gmv-subsidy;
  const roi=subsidy>0?gmv/subsidy:null;
  document.getElementById("bSub").textContent=wan(subsidy);
  document.getElementById("bGmv").textContent=wan(gmv);
  document.getElementById("bNet").textContent=wan(net);
  document.getElementById("bRoi").textContent=roi==null?"—":roi.toFixed(2);
  const hint=document.getElementById("budgetHint");
  hint.className="callout "+(net>=0?"green":"red");
  hint.innerHTML=net>=0
    ?`<b>可行：</b>补贴 ${wan(subsidy)}，增量 GMV ${wan(gmv)}，增量净额 ${wan(net)}，ROI ${roi==null?"—":roi.toFixed(2)}。`
    :`<b>亏损：</b>补贴 ${wan(subsidy)}，增量 GMV ${wan(gmv)}，增量净额 ${wan(net)}。必须提高增量 GMV/次，或降低补贴。`;
}
function renderBudget(){
  budgetCompute();
  const s=DATA.summary;
  const cats=["S0 现状","S1 03 d10","v2 d7 全响应"];
  const subsidy=[s.current_subsidy/10000,s.s1_subsidy/10000,s.v2_subsidy/10000];
  const net=[s.current_inc_net_fe/10000,s.s1_inc_net_05/10000,s.v2_inc_net/10000];
  initChart("budgetChart",{
    tooltip:{trigger:"axis",valueFormatter:v=>v.toFixed(2)+" 万"},
    legend:{top:0,data:[{name:"补贴",icon:"roundRect",itemStyle:{color:"#B06A3C"}},{name:"增量净额",icon:"roundRect",itemStyle:{color:"#2F5D50"}}]},
    grid:{left:54,right:20,top:36,bottom:28},
    xAxis:{type:"category",data:cats,axisLabel:{fontSize:11}},
    yAxis:{type:"value",name:"万元"},
    series:[
      {name:"补贴",type:"bar",data:subsidy,itemStyle:{color:"#B06A3C"},label:{show:true,position:"top",formatter:p=>p.value.toFixed(1)+" 万",fontSize:10}},
      {name:"增量净额",type:"bar",data:net,itemStyle:{color:p=>p.value>=0?"#2F5D50":"#8A3B45"},label:{show:true,position:"top",formatter:p=>(p.value>=0?"+":"")+p.value.toFixed(2)+" 万",fontSize:10}}
    ]
  });
  renderPolicyV2("policyV2Table");
}
function renderPolicyV2(id){
  document.getElementById(id).innerHTML=table(
    ["人群","03 的 S1","05 的 v2","触达","补贴/次","增量净额/次","v2 增量净额"],
    DATA.policyV2.map(r=>[
      esc(r.segment_name),esc(r.offer_group),`<span class="pill ${r.v2_offer.includes("7-3")?"green":""}">${esc(r.v2_offer)}</span>`,
      fmtInt(r.est_sends),fmt2(r.v2_subsidy_per_send),fmt2(r.v2_inc_net_per_send),fmtInt(r.v2_est_inc_net)
    ])
  );
}

/* ================= 04 响应排序 ================= */
function renderResponse(){
  const m=DATA.metrics;
  const cards=[
    ["AUC",m.auc==null?"—":Number(m.auc).toFixed(3),"测试集券窗口"],
    ["PR-AUC",m.pr_auc==null?"—":Number(m.pr_auc).toFixed(3),"基线 0.545"],
    ["Brier",m.brier==null?"—":Number(m.brier).toFixed(3),"概率校准误差"],
    ["Best F1",m.best_f1==null?"—":Number(m.best_f1).toFixed(3),"阈值 0.419"]
  ];
  document.getElementById("modelKpi").innerHTML=cards.map(([a,b,c])=>`<div class="kpi"><div class="k">${a}</div><div class="v">${b}</div><div class="s">${c}</div></div>`).join("");
  document.getElementById("segmentEvalTable").innerHTML=table(
    ["人群","n","实际完成率","预测完成率","AUC"],
    DATA.segmentEval.map(r=>[esc(r.segment_name),fmtInt(r.n),pct(r.actual_rate),pct(r.pred_rate),r.auc==null?"—":Number(r.auc).toFixed(3)])
  );
  initChart("calibrationChart",{
    tooltip:{trigger:"axis"},
    grid:{left:48,right:18,top:24,bottom:38},
    xAxis:{type:"value",name:"预测",min:0,max:1,nameLocation:"middle",axisLabel:{formatter:v=>(v*100).toFixed(0)+"%"}},
    yAxis:{type:"value",name:"实际",min:0,max:1,nameLocation:"middle",axisLabel:{formatter:v=>(v*100).toFixed(0)+"%"}},
    series:[
      {type:"line",data:DATA.calibration.map(r=>[r.pred,r.actual]),symbolSize:8,lineStyle:{width:2},itemStyle:{color:"#2F4A6D"}},
      {type:"line",data:[[0,0],[1,1]],lineStyle:{type:"dashed",color:"#A8A49A"},symbol:"none",tooltip:{show:false}}
    ]
  });
}

/* ================= 05 增量评估 ================= */
function renderUplift(){
  const k=DATA.kpi;
  document.getElementById("upliftKpi").innerHTML=[
    ["整体增量",(k.overallAnyPp>=0?"+":"")+k.overallAnyPp+"pp","72h，券 − 信息型"],
    ["Top20% gap",(k.top20Gap>=0?"+":"")+k.top20Gap+"pp","T-learner 测试集"],
    ["S1 增量净额",wan(k.s1Net),"03 方案，05 口径"],
    ["v2 增量净额",wan(k.v2Net),"只给全响应发 d7"]
  ].map(([a,b,c])=>`<div class="kpi"><div class="k">${a}</div><div class="v">${b}</div><div class="s">${c}</div></div>`).join("");

  document.getElementById("segUpliftTable").innerHTML=table(
    ["人群","发券 n","券购买率","信息购买率","购买率增量","增量 GMV/次","补贴/次","增量净额/次","增量 ROI"],
    DATA.segUplift.map(r=>[
      esc(r.segment_name),fmtInt(r.n_sends),pct(r.actual_coupon_rate),pct(r.actual_info_rate),
      pp(r.effect_any72),fmt2(r.effect_gmv72),fmt2(r.avg_subsidy_current),fmt2(r.net_gmv72),
      r.roi_incremental==null?"—":Number(r.roi_incremental).toFixed(2)
    ])
  );

  const types=OFFER_ORDER.slice(0,5);
  const anyRows=DATA.typeUplift.filter(r=>r.outcome==="any72");
  const gmvRows=DATA.typeUplift.filter(r=>r.outcome==="gmv72");
  const typeRows=types.map(t=>{
    const a=anyRows.find(x=>x.offer_type===t), g=gmvRows.find(x=>x.offer_type===t);
    return [esc(t),a?pp(a.estimate):"—",g?fmt2(g.estimate):"—",g?fmt2(g.subsidy):"—",g?fmt2(g.net_gmv72):"—",g&&g.roi_incremental!=null?Number(g.roi_incremental).toFixed(2):"—"];
  });
  document.getElementById("typeUpliftTable").innerHTML=table(
    ["券型","购买率增量","增量 GMV/次","补贴/次","增量净额/次","增量 ROI"],typeRows
  );

  const sel=document.getElementById("crossSelect");
  sel.innerHTML=DATA.audit.filter(r=>r.segment_name!=="中价值·未触达").map(r=>`<option value="${esc(r.segment_name)}">${esc(r.segment_name)}</option>`).join("");
  sel.value="高价值·选择性响应型";
  sel.onchange=()=>renderCross(sel.value);
  renderCross(sel.value);

  document.getElementById("decileTable").innerHTML=table(
    ["分组","n","券购买率","信息购买率","gap","预测 uplift"],
    DATA.deciles.map(r=>[fmtInt(r.decile),fmtInt(r.n),pct(r.coupon_rate),pct(r.info_rate),pp(r.gap),fmt2(r.pred_uplift)])
  );
  initChart("decileChart",{
    tooltip:{trigger:"axis",valueFormatter:v=>(v>=0?"+":"")+Number(v).toFixed(1)+"pp"},
    grid:{left:48,right:18,top:24,bottom:38},
    xAxis:{type:"category",data:DATA.deciles.map(r=>"D"+r.decile),axisLabel:{fontSize:11}},
    yAxis:{type:"value",name:"gap（pp）",axisLabel:{formatter:v=>v+"pp"}},
    series:[{type:"bar",data:DATA.deciles.map(r=>+(r.gap*100).toFixed(1)),itemStyle:{color:p=>p.value>=0?"#2F5D50":"#8A3B45"},label:{show:true,position:"top",formatter:p=>(p.value>=0?"+":"")+p.value+"pp",fontSize:10}}]
  });

  const um=DATA.upliftMetrics;
  document.getElementById("tlearnerMetrics").innerHTML=table(
    ["指标","值","说明"],[
      ["券模型 AUC",um.auc_coupon_model==null?"—":Number(um.auc_coupon_model).toFixed(3),"测试集券窗口"],
      ["信息模型 AUC",um.auc_info_model==null?"—":Number(um.auc_info_model).toFixed(3),"测试集信息窗口"],
      ["测试集整体 gap",um.overall_gap_pp==null?"—":(Number(um.overall_gap_pp)>=0?"+":"")+Number(um.overall_gap_pp).toFixed(2)+"pp","与 FE 估计一致"],
      ["Top 20% gap",um.top20_gap_pp==null?"—":(Number(um.top20_gap_pp)>=0?"+":"")+Number(um.top20_gap_pp).toFixed(2)+"pp","n=3,043"],
      ["Bottom 20% gap",um.bottom20_gap_pp==null?"—":Number(um.bottom20_gap_pp).toFixed(2)+"pp","n=3,043"]
    ]
  );
  renderPolicyV2("policyV2TableUplift");
  const s=DATA.summary;
  document.getElementById("policySummaryCards").innerHTML=`
    <div class="profile-grid">
      <div class="profile-item"><div class="k">v2 补贴</div><div class="v">${wan(s.v2_subsidy)}</div></div>
      <div class="profile-item"><div class="k">v2 增量 GMV</div><div class="v">${wan(s.v2_inc_gmv)}</div></div>
      <div class="profile-item"><div class="k">v2 增量净额</div><div class="v">${wan(s.v2_inc_net)}</div></div>
      <div class="profile-item"><div class="k">v2 增量 ROI</div><div class="v">${s.v2_inc_roi==null?"—":Number(s.v2_inc_roi).toFixed(2)}</div></div>
      <div class="profile-item"><div class="k">S1 增量净额</div><div class="v">${wan(s.s1_inc_net_05)}</div></div>
      <div class="profile-item"><div class="k">v2 vs S1</div><div class="v">${wan(s.v2_vs_s1)}</div></div>
    </div>`;
}
/* ================= 06 验证与落地 ================= */
function sqlPassed(r){return r.pass===true || String(r.pass).toLowerCase()==="true";}
function levelPill(level){
  if(level==="stable") return '<span class="pill green">稳定</span>';
  if(level==="warn") return '<span class="pill">关注</span>';
  return '<span class="pill red">严重</span>';
}
function renderValidation(){
  const sql=DATA.sqlValidation||[];
  const ab=DATA.abPower||[];
  const pooled=ab.find(r=>String(r.segment).includes("pooled"))||ab[ab.length-1]||{};
  const pass=sql.filter(sqlPassed).length;
  document.getElementById("validationKpi").innerHTML=[
    ["SQL 对账",pass+" / "+sql.length,"9 条查询 · 全链路"],
    ["A/B 合并样本",fmtInt(pooled.n_users),"每臂 "+fmtInt(pooled.n_per_arm)+" 人"],
    ["净额 MDE",fmt2(pooled.mde_net_gmv_80)+" 元","80% power；观测 power "+pct(pooled.power_net_observed)],
    ["2.5 元缺口",fmtInt(pooled.shortfall_users_net_2_5)+" 人","合并样本还不够"]
  ].map(([a,b,c])=>`<div class="kpi"><div class="k">${a}</div><div class="v">${b}</div><div class="s">${c}</div></div>`).join("");

  document.getElementById("sqlSummary").innerHTML=`
    <div class="profile-grid">
      <div class="profile-item"><div class="k">检查项</div><div class="v">${pass} / ${sql.length}</div></div>
      <div class="profile-item"><div class="k">SQL 查询</div><div class="v">${fmtInt(DATA.sqlQueryCount)} 条</div></div>
      <div class="profile-item"><div class="k">结论</div><div class="v">${pass===sql.length?"全部通过":"存在未过"}</div></div>
    </div>`;
  const keyChecks=["event::total","user_features::rows","offer_matrix::sends","raw_uplift::coupon","raw_uplift::info","policy::subsidy","policy::inc_gmv","policy::inc_net","policy::roi"];
  const sqlRows=keyChecks.map(c=>{
    const r=sql.find(x=>x.check===c);
    if(!r) return [c,"—","—","—"];
    const isRoi=c==="policy::roi";
    return [esc(r.note),isRoi?fmt2(r.actual):fmtInt(r.actual),isRoi?fmt2(r.expected):fmtInt(r.expected),sqlPassed(r)?'<span class="pill green">通过</span>':'<span class="pill red">未过</span>'];
  });
  document.getElementById("sqlTable").innerHTML=table(["检查","实际","期望","结果"],sqlRows);

  document.getElementById("abSummary").innerHTML=`
    <div class="profile-grid">
      <div class="profile-item"><div class="k">目标样本</div><div class="v">${fmtInt(pooled.n_users)}</div></div>
      <div class="profile-item"><div class="k">历史 d7 用户</div><div class="v">${fmtInt(pooled.n_coupon_users)}</div></div>
      <div class="profile-item"><div class="k">净 GMV / 次</div><div class="v">${fmt2(pooled.mu0_info_gmv)} → ${fmt2(pooled.mu1_coupon_net_gmv)}</div></div>
      <div class="profile-item"><div class="k">合并净额增量</div><div class="v">+${fmt2(pooled.delta_net_gmv)}</div></div>
      <div class="profile-item"><div class="k">转化 MDE</div><div class="v">${fmt2(pooled.mde_conv_pp_80)}pp</div></div>
      <div class="profile-item"><div class="k">净额 power</div><div class="v">${pct(pooled.power_net_observed)}</div></div>
    </div>`;
  const abLabel=s=>String(s).includes("pooled")?"合并":String(s).replace("价值·全响应型","·全");
  const abRows=ab.map(r=>[
    esc(abLabel(r.segment)),fmtInt(r.n_users),fmtInt(r.n_per_arm),fmt2(r.mde_conv_pp_80)+"pp",fmt2(r.mde_net_gmv_80)+" 元",pct(r.power_net_observed),fmtInt(r.required_n_per_arm_net_2_5*2),fmtInt(r.shortfall_users_net_2_5)
  ]);
  document.getElementById("abTable").innerHTML=table(["人群","现有 n","每臂","转化 MDE","净额 MDE","净额 power","2.5 元所需总样本","缺口"],abRows);
  initChart("abChart",{
    tooltip:{trigger:"axis"},
    legend:{top:0,data:["现有样本","2.5 元净额所需"]},
    grid:{left:58,right:18,top:38,bottom:40},
    xAxis:{type:"category",data:ab.map(r=>abLabel(r.segment)),axisLabel:{fontSize:11}},
    yAxis:{type:"value",name:"人"},
    series:[
      {name:"现有样本",type:"bar",data:ab.map(r=>r.n_users),itemStyle:{color:"#2F5D50"},label:{show:true,position:"top",fontSize:10}},
      {name:"2.5 元净额所需",type:"bar",data:ab.map(r=>Math.round(r.required_n_per_arm_net_2_5*2)),itemStyle:{color:"#B06A3C"},label:{show:true,position:"top",fontSize:10}}
    ]
  });

  const monitorRows=DATA.modelDecay.map(r=>{
    const c=DATA.couponMonitor.find(x=>x.period===r.period)||{};
    const alert=r.period==="D7-13"?"基准":r.alert==="ok"?"正常":r.alert==="cold_start"?"冷启动":"—";
    const cls=r.period==="D7-13"?"":(r.alert==="ok"?"green":"");
    return [esc(r.period),fmtInt(r.n),pct(r.pos_rate),fmt2(r.auc),fmt2(r.brier),fmt2(c.net_gmv_per_send),"<span class=\"pill "+cls+"\">"+alert+"</span>"];
  });
  document.getElementById("monitorTable").innerHTML=table(["周期","n","正率","AUC","Brier","净 GMV/发送","状态"],monitorRows);

  const psiRows=DATA.psi.filter(r=>r.feature==="p_complete"||r.feature==="spend_prior").map(r=>[
    esc(r.feature_cn),esc(r.period),fmt2(r.psi),levelPill(r.level)
  ]);
  document.getElementById("psiTable").innerHTML=table(["特征","周期","PSI","判断"],psiRows);

  const grouped={};
  DATA.alerts.forEach(r=>{const f=String(r.object).split(" / ")[0];(grouped[f]=grouped[f]||[]).push(r);});
  const alertRows=Object.entries(grouped).map(([f,items])=>[esc(f),fmtInt(items.length),"<span class=\"pill navy\">"+esc(items[0].action)+"</span>"]);
  document.getElementById("alertTable").innerHTML=table(["对象","条数","建议动作"],alertRows);
}

function renderCross(name){
  const rows=DATA.segType.filter(r=>r.segment_name===name);
  const types=OFFER_ORDER.slice(0,5);
  const any=rows.filter(r=>r.outcome==="any72"), gmv=rows.filter(r=>r.outcome==="gmv72");
  const tableRows=types.map(t=>{
    const a=any.find(r=>r.offer_type===t), g=gmv.find(r=>r.offer_type===t);
    return [esc(t),a?pp(a.estimate):"—",g?fmt2(g.estimate):"—",g?fmt2(g.subsidy):"—",g?fmt2(g.net_gmv72):"—",g&&g.roi_incremental!=null?Number(g.roi_incremental).toFixed(2):"—"];
  });
  document.getElementById("crossTable").innerHTML=table(["券型","购买率增量","增量 GMV/次","补贴/次","增量净额/次","ROI"],tableRows);
  initChart("crossChart",{
    tooltip:{trigger:"axis"},
    legend:{top:0,data:["购买率增量（pp）","增量净额/次（元）"]},
    grid:{left:52,right:18,top:36,bottom:40},
    xAxis:{type:"category",data:types,axisLabel:{fontSize:11,rotate:20}},
    yAxis:[{type:"value",name:"pp"},{type:"value",name:"元"}],
    series:[
      {name:"购买率增量（pp）",type:"bar",data:types.map(t=>{const a=any.find(r=>r.offer_type===t);return a?+(a.estimate*100).toFixed(1):null;}),itemStyle:{color:"#2F4A6D"}},
      {name:"增量净额/次（元）",type:"line",yAxisIndex:1,data:types.map(t=>{const g=gmv.find(r=>r.offer_type===t);return g?+Number(g.net_gmv72).toFixed(2):null;}),symbolSize:8,lineStyle:{width:2},itemStyle:{color:"#8A3B45"}}
    ]
  });
}

/* ================= 路由与初始化 ================= */
function showView(name){
  document.querySelectorAll(".view").forEach(v=>v.classList.remove("active"));
  const el=document.getElementById("view-"+name); if(el) el.classList.add("active");
  document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b.dataset.view===name));
  requestAnimationFrame(()=>resizeCharts());
  setTimeout(resizeCharts,200);
}
document.getElementById("nav").addEventListener("click",e=>{
  const b=e.target.closest(".nav-item"); if(!b) return;
  showView(b.dataset.view); history.replaceState(null,"","#"+b.dataset.view);
});
document.getElementById("upliftTabs").addEventListener("click",e=>{
  const b=e.target.closest(".subtab"); if(!b) return;
  showSubtab(b.dataset.subtab);
});
document.addEventListener("DOMContentLoaded",()=>{
  renderOverview(); renderSegments(); renderOffers(); renderBudget(); renderResponse(); renderUplift(); renderValidation();
  const h=(location.hash||"").replace("#","");
  if(["overview","segments","offers","budget","response","uplift","validation"].includes(h)) showView(h);
});
window.addEventListener("resize",()=>{clearTimeout(window.__rz);window.__rz=setTimeout(resizeCharts,120);});
window.addEventListener("hashchange",()=>{
  const h=(location.hash||"").replace("#","");
  if(["overview","segments","offers","budget","response","uplift","validation"].includes(h)) showView(h);
});
</script>

<script>
(function(){
  const box = document.getElementById('lightbox');
  const img = document.getElementById('lightboxImg');
  if(!box || !img){ return; }
  const closeBtn = box.querySelector('.lightbox-close');
  function closeLightbox(){
    box.classList.remove('open');
    box.setAttribute('aria-hidden','true');
    img.removeAttribute('src');
  }
  function openLightbox(src, alt){
    img.src = src;
    img.alt = alt || '放大图表';
    box.classList.add('open');
    box.setAttribute('aria-hidden','false');
  }
  document.querySelectorAll('.chart-img').forEach(el=>{
    el.title = '点击放大';
    el.addEventListener('click', ()=>openLightbox(el.src, el.alt));
  });
  closeBtn.addEventListener('click', closeLightbox);
  box.addEventListener('click', e=>{ if(e.target === box){ closeLightbox(); } });
  document.addEventListener('keydown', e=>{ if(e.key === 'Escape'){ closeLightbox(); } });

  const evalTabs = document.getElementById('evalTabs');
  if(evalTabs){
    evalTabs.addEventListener('click', e=>{
      const b = e.target.closest('.subtab'); if(!b){ return; }
      evalTabs.querySelectorAll('.subtab').forEach(x=>x.classList.toggle('active', x===b));
      ['roc','pr','cal'].forEach(k=>document.getElementById('eval-'+k).classList.toggle('active', k===b.dataset.eval));
    });
  }
  const impTabs = document.getElementById('impTabs');
  if(impTabs){
    impTabs.addEventListener('click', e=>{
      const b = e.target.closest('.subtab'); if(!b){ return; }
      impTabs.querySelectorAll('.subtab').forEach(x=>x.classList.toggle('active', x===b));
      ['gain','shap','shapdist'].forEach(k=>document.getElementById('imp-'+k).classList.toggle('active', k===b.dataset.imp));
    });
  }
})();
</script>
</body>
</html>
"""

html = (HTML
        .replace("__ECHARTS__", ECHARTS)
        .replace("__DATA__", json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__IMAGES__", json.dumps({k: v for k, v in IMAGES.items()}, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__IMG_policy_compare__", IMAGES["policy_compare"])
        .replace("__IMG_behavior_profile__", IMAGES["behavior_profile"])
        .replace("__IMG_model_eval_roc__", IMAGES["model_eval_roc"])
        .replace("__IMG_model_eval_pr__", IMAGES["model_eval_pr"])
        .replace("__IMG_model_eval_cal__", IMAGES["model_eval_cal"])
        .replace("__IMG_lift_chart__", IMAGES["lift_chart"])
        .replace("__IMG_feature_importance_gain__", IMAGES["feature_importance_gain"])
        .replace("__IMG_feature_importance_shap__", IMAGES["feature_importance_shap"])
        .replace("__IMG_shap_summary__", IMAGES["shap_summary"])
        .replace("__IMG_seg_effects__", IMAGES["seg_effects"])
        .replace("__IMG_mechanism__", IMAGES["mechanism"])
        .replace("__IMG_type_effects__", IMAGES["type_effects"])
        .replace("__IMG_uplift_deciles__", IMAGES["uplift_deciles"])
        )

out = DASH / "06_dashboard.html"
out.write_text(html, encoding="utf-8")
print(f"OK: {out}  {out.stat().st_size/1024/1024:.2f} MiB")
