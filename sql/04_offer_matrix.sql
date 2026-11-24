-- 04 券型矩阵：从原始事件重算分人群 × 券型的发送/完成率
-- 注意：03_offer_matrix.csv 的 comps 是「offer completed 事件数」；
-- 04_offer_dataset.csv 的 y_completed 是回溯到唯一发券实例的标签，两者相差 478 条完成事件，
-- 因此 SQL 层用原始事件口径重算，才能和 03 表对上。
WITH events AS (
    SELECT
        s.segment_name,
        t.person,
        t.offer_id,
        t.event,
        pf.difficulty,
        pf.reward,
        pf.offer_type
    FROM raw_transcript t
    INNER JOIN raw_portfolio pf
        ON CAST(t.offer_id AS STRING) = CAST(pf.id AS STRING)
    INNER JOIN user_segments s
        ON CAST(t.person AS STRING) = CAST(s.user_id AS STRING)
    WHERE t.event IN ('offer received', 'offer completed')
),
labeled AS (
    SELECT
        segment_name,
        person,
        offer_id,
        event,
        CASE
            WHEN offer_type = 'bogo' AND difficulty = 5 AND reward = 5 THEN 'BOGO 5/5'
            WHEN offer_type = 'bogo' AND difficulty = 10 AND reward = 10 THEN 'BOGO 10/10'
            WHEN offer_type = 'discount' AND difficulty = 7 AND reward = 3 THEN '折扣 7-3'
            WHEN offer_type = 'discount' AND difficulty = 10 AND reward = 2 THEN '折扣 10-2'
            WHEN offer_type = 'discount' AND difficulty = 20 AND reward = 5 THEN '折扣 20-5'
            ELSE '信息型'
        END AS offer_group
    FROM events
)
SELECT
    segment_name,
    offer_group,
    SUM(CASE WHEN event = 'offer received' THEN 1 ELSE 0 END) AS sends,
    SUM(CASE WHEN event = 'offer completed' THEN 1 ELSE 0 END) AS comps,
    ROUND(
        SUM(CASE WHEN event = 'offer completed' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN event = 'offer received' THEN 1 ELSE 0 END), 0),
        4
    ) AS comp_rate,
    COUNT(DISTINCT person) AS users
FROM labeled
GROUP BY segment_name, offer_group
ORDER BY segment_name, offer_group;
