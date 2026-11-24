-- 03 分群画像：人数、占比与核心行为均值
-- 说明：报告里用中位数；SQL 层用均值做快速对账，避免 DuckDB / Spark 的中位数语法差异。
WITH seg AS (
    SELECT
        s.segment_name,
        s.value_label,
        s.behavior_label,
        u.n_tx,
        u.monetary,
        u.view_rate,
        u.comp_rate
    FROM user_segments s
    INNER JOIN user_features u
        ON CAST(s.user_id AS STRING) = CAST(u.user_id AS STRING)
    WHERE s.segment_name <> '中价值·未触达'
),
total AS (
    SELECT COUNT(*) AS total_users
    FROM seg
)
SELECT
    s.value_label,
    s.behavior_label,
    COUNT(*) AS n,
    ROUND(COUNT(*) * 1.0 / t.total_users, 4) AS share,
    ROUND(AVG(s.monetary), 2) AS avg_monetary,
    ROUND(AVG(s.n_tx), 2) AS avg_n_tx,
    ROUND(AVG(s.view_rate), 4) AS avg_view_rate,
    ROUND(AVG(s.comp_rate), 4) AS avg_comp_rate
FROM seg s
CROSS JOIN total t
GROUP BY s.value_label, s.behavior_label, t.total_users
ORDER BY n DESC;
