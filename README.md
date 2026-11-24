# 优惠券精准投放与增量效果评估

基于 Starbucks Promotions 公开数据的复现项目：从用户分层、响应模型，到增量评估与策略 v2。主线结论是 **完成率不等于增量净额**。

## 目录

```text
优惠券投放复现/
├─ data/
│  ├─ raw/                 3 个原始 JSONL
│  └─ clean/               清洗后的 CSV（含用户特征与人群标签）
├─ src/
│  ├─ 00_parse_json.py     原始 JSONL → CSV
│  ├─ 01_profile_rfm.py    RFM / 行为特征
│  ├─ 02_cluster.py        价值 × 行为双聚类
│  ├─ 03_strategy.py       现状审计与策略矩阵
│  ├─ 04_response_model.py 响应模型与 SHAP
│  ├─ 05_uplift.py         增量评估与策略 v2
│  ├─ 06_build_dashboard.py 七视图看板（分析 + 验证）
│  ├─ 07_sql_pipeline.py   DuckDB 执行 SQL 并与 Pandas 结果对账
│  ├─ 08_ab_power.py      A/B 样本量与功效计算
│  ├─ 09_monitoring.py     PSI、模型衰减、券效与数据量监控
│  ├─ run_all.py           一键运行 00–09，统一写运行日志
│  ├─ utils_viz.py         图表风格
│  └─ assets/              ECharts 构建依赖
├─ sql/                    SQL / Spark SQL 对账查询（01–09）
├─ tests/                  关键结果回归测试（unittest）
│  └─ test_project_contracts.py
├─ outputs/
│  ├─ figures/             全部分析图 PNG；dashboard/ 为看板用单面板裁图
│  ├─ tables/              全部 CSV 结果表
│  ├─ 报告/                文献综述、分析文档、演示脚本
│  ├─ 看板/                06_dashboard.html 与看板说明
│  └─ logs/                模型文件与运行日志
├─ 汇报PPT/                41 页幻灯片、页面 XML、图片与历史版本
└─ _archive/               调试截图等历史文件
```

## 工程化入口

```text
.github/workflows/ci.yml   GitHub Actions：push / PR 后跑 00–09 + 9 项回归
Makefile                   Linux / GitHub Actions 入口
scripts/check.ps1          Windows 无 make 时的等价入口
```

常用命令：

```powershell
# Windows：只跑 9 项回归
pwsh -File .\scripts\check.ps1

# Windows：完整 00–09 + 9 项回归（约 4–5 分钟）
pwsh -File .\scripts\check.ps1 -Full

# Linux / CI
make test
make pipeline
make ci
```

`data/clean/` 和 `outputs/` 下的大部分文件是生成物，默认不提交；`outputs/报告/*.md` 与 `outputs/看板/看板说明.md` 是交付文档，已纳入 Git。新环境先跑一次全量流水线，再执行回归测试。

## 一键复现

在项目根目录执行：

```powershell
python src\run_all.py                 # 从原始 JSON 跑到 09 监控，最后构建看板
python src\run_all.py --from-step 7   # 只跑 SQL / A/B / 监控，并自动刷新看板
python src\run_all.py --dry-run        # 只检查输入和命令，不执行
```

每次运行会生成 `outputs\logs\run_all_时间戳.log`，同时打印到控制台。

06 看板依赖 07–09 的验证产物，`run_all` 会自动把 06 排在 09 之后；单独运行 `python src\06_build_dashboard.py` 前，必须先有 07–09 的输出。

跑完做关键结果回归：

```powershell
python -m unittest discover -s tests -v
```

当前干净目录完整 00→09 实测 **263 秒（约 4 分 23 秒）**；07→09 约 16 秒。

## 分步运行

在项目根目录执行：

```powershell
python src\00_parse_json.py
python src\01_profile_rfm.py
python src\02_cluster.py
python src\03_strategy.py
python src\04_response_model.py
python src\05_uplift.py
python src\06_build_dashboard.py
python src\07_sql_pipeline.py
python src\08_ab_power.py
python src\09_monitoring.py
```

`07_sql_pipeline.py` 会执行 `sql/` 下的 9 条查询，输出到 `outputs\tables\sql\`，并用 34 项检查与现有 Pandas 结果对账。详细口径见 `sql\README.md`。

每一步的输入输出都已经在 `src/config.py` 中统一配置，不需要改脚本里的路径。

## 关键交付

- 看板：`outputs\看板\06_dashboard.html`
- 一键运行：`src\run_all.py`；运行日志在 `outputs\logs\`
- 回归测试：`tests\test_project_contracts.py`（9 项，当前全过）
- SQL 对账结果：`outputs\tables\sql\`
- SQL 对账报告：`outputs\报告\07_SQL对账.md`、`outputs\logs\07_sql_validation.txt`
- A/B 实验设计：`outputs\报告\08_AB实验设计.md`、`outputs\tables\08_ab_power.csv`
- 监控与稳定性：`outputs\报告\09_监控与稳定性.md`、`outputs\tables\09_psi.csv`、`outputs\figures\09_psi_heatmap.png`
- 一页纸总结 / 简历描述：`outputs\报告\10_一页纸项目总结.md`、`outputs\报告\10_简历项目描述.md`
- 分析报告：`outputs\报告\`
- 当前 PDF：`汇报PPT\deai_final_rev220.pdf`（revision 220，41 页，由线上幻灯片导出）
- 旧版 PDF：`汇报PPT\deai_final6.pdf`（仅保留历史版本，不作为当前交付）
- 在线幻灯片：<https://my.feishu.cn/slides/ZsH8sInL9lVbqId2HA7czDZs6hg>

## 口径边界

- 03 的 S1 是历史方案，账面净额 +16.8 万；05 用信息型券准对照 + 用户固定效应重算后为 −5.5 万。
- 05 是准实验，不是严格 RCT；可以说“准对照/准实验”，不要说“证明因果”。
- 04 的 `p_complete` 只做人群内排序，不能单独回答“发不发券”。
- `sql\05_segment_uplift_raw.sql` 是原始分组差异，不是 05 的用户固定效应估计；汇报和面试都以 05 的 FE 结果为准。
- `sql\06_uplift_deciles.sql` 用全量窗口做描述性 `NTILE`，与 05 的测试集 `qcut` 不是同一口径，不能逐行对账。
- `03_offer_matrix.csv` 的 `comps` 是 33,579 条完成事件；`04_offer_dataset.csv` 的 `y_completed` 回溯到唯一发券实例后是 33,101，二者差 478，不能混用。
- `08_AB实验设计.md` 是实验方案与功效计算，不是线上 A/B；5,542 人对观测差 5.16 元够，但 2.5 元净 GMV MDE 需要约 9,030 人。
- `09_监控与稳定性.md` 是 30 天历史离线回放，不是生产实时监控；行为历史特征 PSI 高主要是回看窗口左截断，先查口径再判漂移。
