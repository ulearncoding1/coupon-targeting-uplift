# 07 SQL 对账层

## 目标与边界

这一层回答两个问题：

1. 项目里的关键数字，能不能不依赖 Pandas，换一套 SQL 引擎重新算出来？
2. 这些 SQL 能不能平移到 Spark SQL，而不是只能在本地跑？

边界：SQL 只做**特征聚合、描述统计、数据质量和对账**，不替代 04 的 LightGBM 响应模型，也不替代 05 的用户固定效应 / T-learner 估计。模型结论仍以 Python 输出为准。

## 查询与结果

| 查询 | 结果 | 对账 |
|---|---|---|
| `01_event_funnel` | 306,534 条事件；transaction 138,953；offer received 76,277；viewed 57,725；completed 33,579 | 与原始事件表一致 |
| `02_user_features_rebuild` | 17,000 用户；`n_tx` 合计 138,953；`monetary` 合计 1,775,451.97；`recency_h` 均值 91.43；422 人无交易 | 与 `user_features.csv` 一致 |
| `03_segment_profile` | 9 个人群；覆盖 16,994 人 | 与 `02_segment_profile.csv` 一致；6 个“中价值·未触达”不计入 |
| `04_offer_matrix` | 54 个人群×券型格；发送 76,277；完成 33,579 | 与 `03_offer_matrix.csv` 一致 |
| `05_segment_uplift_raw` | 9 个人群的券/信息型原始差异 | 只做描述，不与 FE 估计逐行对比 |
| `06_uplift_deciles` | 全量 76,277 条窗口；10 组，每组 7,627–7,628 条 | 与 Python 测试集 `qcut` 不是同一口径 |
| `07_purchase_sequence` | 122,375 次相邻交易；16,177 个至少两单用户；平均间隔 66.52 小时，最短 6 小时，最长 678 小时 | 交易数减去有交易用户数 |
| `08_data_quality` | 8 项检查，失败数全部为 0 | 主键、孤儿键、空值、事件白名单 |
| `09_policy_summary` | 补贴 72,258（期望核销）；增量 GMV 191,880；增量净额 119,622；ROI 2.655 | 与 `05_policy_v2_summary.csv` 一致 |

运行方式：

```powershell
python src\07_sql_pipeline.py
```

输出到 `outputs\tables\sql\`，对账报告在 `outputs\logs\07_sql_validation.txt`。

## 三个关键口径坑

### 1. 完成事件 ≠ 完成标签

- `03_offer_matrix.csv` 的 `comps` 是 `offer completed` **事件数**：33,579。
- `04_offer_dataset.csv` 的 `y_completed` 是把完成事件回溯到**唯一发券实例**后的标签：33,101 个正样本。
- 差额 478 条，来自多个完成事件落到同一个发券实例。运营报表说“完成 33,579 次”，模型训练说“33,101 个正样本”，两者都对，但不能混用。
- 因此 `04_offer_matrix.sql` 用原始事件口径重算，才能和 03 表对上。

### 2. 原始差异 ≠ 固定效应估计

- `05_segment_uplift.csv` 是 OLS + 用户固定效应，控制了用户不随时间变化的差异。
- `05_segment_uplift_raw.sql` 是券 vs 信息型的原始分组差异，会被“谁更可能收到券”干扰。
- 例：中价值·全响应型 raw +18.78pp，FE 口径 +20.5pp；低价值·零完成型 raw −19.34pp，但只有 79 条记录，不能单独下结论。
- 汇报和面试都以 05 的 FE/T-learner 结果为准，SQL 只证明“描述统计能复现”。

### 3. 十分组不是同一批人

- `05_uplift_deciles.csv` 是 Python 在**测试集**上做 `qcut`，合计 15,215 行。
- SQL 的 `06_uplift_deciles.sql` 是**全量** `send_windows` 做 `NTILE(10)`，合计 76,277 行，每组约 7,628 条。
- 两者都呈现“从低到高，券−信息差距单调上升”的形态，但行数、分组边界不同，不能逐行对账。
- 全量十分组的 gap 会被券/信息型分配差异放大，不能当因果效应。

## 面试怎么讲

> 我在项目里加了一层 SQL 对账：用 9 条查询重算事件漏斗、用户特征、人群×券型矩阵、券/信息型原始差异、十分组、购买间隔和 v2 策略汇总。DuckDB 本地执行，34 项检查和 Pandas 结果全部对上。过程中发现一个很容易混的口径：03 表的 `comps` 是 33,579 条完成事件，04 表的 `y_completed` 回溯到唯一发券实例后是 33,101，差 478；所以报表和模型标签必须分开。SQL 用的是 Spark SQL 兼容写法，`NTILE`、`RANK`、`LAG`、`ROW_NUMBER` 都能直接平移；本地没装 Java/PySpark，所以 Spark 集群没实跑，我不会把它说成已上集群。

这个讲法的重点不是“我会写 select”，而是：**能发现口径差、能解释差在哪、能判断哪张表该信**。数据分析岗和业务型数据科学岗，通常更看重这一点。
