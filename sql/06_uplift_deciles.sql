-- 06 T-learner 分的描述性十分组：用 NTILE 窗口函数切分，展示窗口函数用法
-- 注意：05_uplift_deciles.csv 是 Python 在测试集上按 qcut 生成的；本 SQL 用全量 send_windows 做描述性分组，
-- 行数不会与 05_uplift_deciles.csv 完全一致，不能直接当模型验收。
WITH ranked AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY uplift_score) AS decile
    FROM send_windows
)
SELECT
    decile,
    COUNT(*) AS n,
    SUM(CASE WHEN coupon = 1 THEN 1 ELSE 0 END) AS n_coupon,
    SUM(CASE WHEN coupon = 0 THEN 1 ELSE 0 END) AS n_info,
    ROUND(AVG(CASE WHEN coupon = 1 THEN any72 END), 4) AS coupon_rate,
    ROUND(AVG(CASE WHEN coupon = 0 THEN any72 END), 4) AS info_rate,
    ROUND(
        AVG(CASE WHEN coupon = 1 THEN any72 END)
        - AVG(CASE WHEN coupon = 0 THEN any72 END),
        4
    ) AS gap,
    ROUND(AVG(uplift_score), 4) AS avg_pred_uplift
FROM ranked
GROUP BY decile
ORDER BY decile;
