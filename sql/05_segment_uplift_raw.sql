-- 05 分人群增量：券 vs 信息型的原始差异
-- 注意：这里是描述性 raw difference，不是 05 的用户固定效应估计；FE 结果仍以 Python 输出为准。
WITH seg AS (
    SELECT
        segment_name,
        COUNT(*) AS n_sends,
        COUNT(DISTINCT person) AS n_users,
        SUM(CASE WHEN coupon = 1 THEN 1 ELSE 0 END) AS n_coupon,
        SUM(CASE WHEN coupon = 0 THEN 1 ELSE 0 END) AS n_info,
        ROUND(AVG(CASE WHEN coupon = 1 THEN any72 END), 4) AS coupon_rate,
        ROUND(AVG(CASE WHEN coupon = 0 THEN any72 END), 4) AS info_rate,
        ROUND(
            AVG(CASE WHEN coupon = 1 THEN any72 END)
            - AVG(CASE WHEN coupon = 0 THEN any72 END),
            4
        ) AS raw_any72_diff,
        ROUND(
            AVG(CASE WHEN coupon = 1 THEN gmv72 END)
            - AVG(CASE WHEN coupon = 0 THEN gmv72 END),
            2
        ) AS raw_gmv72_diff
    FROM send_windows
    GROUP BY segment_name
),
ranked AS (
    SELECT
        *,
        RANK() OVER (ORDER BY raw_any72_diff DESC) AS raw_rank
    FROM seg
)
SELECT *
FROM ranked
ORDER BY raw_rank;
