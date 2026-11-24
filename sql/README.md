# SQL / Spark SQL 对账层

这一层不是替代 Python 建模，而是把“数据 → 人群 → 券型 → 增量 → 策略”链路中的关键聚合，用 SQL 重新算一遍，证明口径能对上、代码能落地。

## 运行

```powershell
python src\07_sql_pipeline.py
```

DuckDB 在本地执行，输出：

- `outputs\tables\sql\*.csv`：每条 SQL 的结果
- `outputs\tables\sql\_manifest.csv`：查询清单、行数、列数
- `outputs\tables\sql\_validation.csv`：逐项对账结果
- `outputs\logs\07_sql_validation.txt`：可读版对账报告

## 查询清单

| 文件 | 作用 | 关键口径 |
|---|---|---|
| `01_event_funnel.sql` | 4 类事件量与用户数 | 306,534 条事件；4 类事件 |
| `02_user_features_rebuild.sql` | 从原始事件重算 RFM 与券行为 | 与 `user_features.csv` 的 `n_tx`、`monetary` 对账 |
| `03_segment_profile.sql` | 9 个人群的人数、占比、均值 | 排除 6 个“中价值·未触达”，与 `02_segment_profile` 对齐 |
| `04_offer_matrix.sql` | 9 人群 × 6 券型的发送/完成率 | 用原始 `offer completed` 事件口径，与 `03_offer_matrix` 对齐 |
| `05_segment_uplift_raw.sql` | 券 vs 信息型的原始差异 | **描述性 raw difference，不是 05 的 FE 估计** |
| `06_uplift_deciles.sql` | `NTILE(10)` 十分组 | 全量 `send_windows`，不是 `05_uplift_deciles` 的测试集口径 |
| `07_purchase_sequence.sql` | `ROW_NUMBER + LAG` 购买间隔 | 只统计至少有两次交易用户 |
| `08_data_quality.sql` | 主键、孤儿键、空值、事件白名单 | 8 项质量检查 |
| `09_policy_summary.sql` | v2 补贴、增量 GMV、净额、ROI | 与 `05_policy_v2_summary` 对齐 |

## 两个必须说清楚的口径差

1. **33,579 vs 33,101**
   - `03_offer_matrix.csv` 的 `comps` 是 `offer completed` 原始事件数：33,579。
   - `04_offer_dataset.csv` 的 `y_completed` 是把完成事件回溯到“唯一发券实例”后的标签，多个完成事件可能落到同一个实例，正样本实例是 33,101。
   - 所以 `04_offer_matrix.sql` 用原始事件口径，才能和 03 表对上；这也解释了为什么不能拿 `y_completed` 直接当“完成次数”做运营报表。

2. **FE 估计 vs 原始差异**
   - `05_segment_uplift.csv` 是 OLS + 用户固定效应输出，不是简单分组均值。
   - `05_segment_uplift_raw.sql` 是描述性 raw difference，只用来展示 SQL 能算、能排序，不能替换 05 的因果/准实验结论。
   - 典型差异：中价值·全响应型 raw +18.78pp，FE 口径 +20.5pp；低价值·零完成型 raw −19.34pp，样本只有 79 条，不能单独下结论。

3. **`05_uplift_deciles.csv` 是测试集**
   - Python 用 `GroupShuffleSplit(test_size=0.2, random_state=42)` 取测试集，再做 `qcut` 十分组，合计 15,215 行。
   - SQL 的 `06_uplift_deciles.sql` 用全量 `send_windows` 做 `NTILE(10)`，合计 76,277 行。
   - 两者只适合看“分组形态”，不能逐行对账；SQL 里已注明。

## 平移到 Spark SQL

SQL 文件只引用视图名，不写 `read_csv_auto`；在 Spark 里注册同名临时视图即可：

```python
from pathlib import Path
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
base = Path("C:/Users/kk159/Desktop/优惠券投放复现/优惠券投放复现")

views = {
    "raw_transcript": base / "data/clean/transcript.csv",
    "raw_profile": base / "data/clean/profile.csv",
    "raw_portfolio": base / "data/clean/portfolio.csv",
    "user_features": base / "data/clean/user_features.csv",
    "user_segments": base / "data/clean/user_segments.csv",
    "offer_dataset": base / "outputs/tables/04_offer_dataset.csv",
    "send_windows": base / "outputs/tables/05_send_windows.csv",
    "policy_v2": base / "outputs/tables/05_policy_v2.csv",
}

for name, path in views.items():
    (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
        .createOrReplaceTempView(name))

for sql_file in sorted((base / "sql").glob("*.sql")):
    df = spark.sql(sql_file.read_text(encoding="utf-8"))
    df.toPandas().to_csv(
        base / "outputs/tables/sql" / f"{sql_file.stem}.csv",
        index=False,
        encoding="utf-8-sig",
    )
```

注意：

- SQL 里的 `CAST(... AS STRING)` 是 DuckDB 和 Spark 都支持的写法；没有用 `VARCHAR`。
- `NTILE`、`RANK`、`LAG`、`ROW_NUMBER` 都是 Spark SQL 原生窗口函数。
- 本地没有装 Java / PySpark，所以 Spark 版本没有实跑；DuckDB 版本已跑通 34 项对账。面试时可以说“SQL 兼容 Spark，DuckDB 本地验证”，不要说“已在 Spark 集群跑过”。
