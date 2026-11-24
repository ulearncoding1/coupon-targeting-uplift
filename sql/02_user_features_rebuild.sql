-- 02 核心用户特征：从原始事件重算 RFM 与券行为，用于和 user_features.csv 对账
WITH tx AS (
    SELECT
        person,
        COUNT(*) AS n_tx,
        SUM(amount) AS monetary,
        MAX(time) AS last_tx_time
    FROM raw_transcript
    WHERE event = 'transaction'
    GROUP BY person
),
off AS (
    SELECT
        person,
        SUM(CASE WHEN event = 'offer received' THEN 1 ELSE 0 END) AS n_offer_recv,
        SUM(CASE WHEN event = 'offer viewed' THEN 1 ELSE 0 END) AS n_offer_view,
        SUM(CASE WHEN event = 'offer completed' THEN 1 ELSE 0 END) AS n_offer_comp
    FROM raw_transcript
    WHERE event IN ('offer received', 'offer viewed', 'offer completed')
    GROUP BY person
)
SELECT
    CAST(p.id AS STRING) AS user_id,
    COALESCE(t.n_tx, 0) AS n_tx,
    ROUND(COALESCE(t.monetary, 0), 2) AS monetary,
    ROUND(
        CASE
            WHEN t.last_tx_time IS NULL THEN NULL
            ELSE 720.0 - t.last_tx_time
        END,
        2
    ) AS recency_h,
    COALESCE(o.n_offer_recv, 0) AS n_offer_recv,
    COALESCE(o.n_offer_view, 0) AS n_offer_view,
    COALESCE(o.n_offer_comp, 0) AS n_offer_comp,
    ROUND(
        CASE
            WHEN COALESCE(o.n_offer_recv, 0) = 0 THEN 0
            ELSE COALESCE(o.n_offer_view, 0) * 1.0 / o.n_offer_recv
        END,
        4
    ) AS view_rate,
    ROUND(
        CASE
            WHEN COALESCE(o.n_offer_recv, 0) = 0 THEN 0
            ELSE COALESCE(o.n_offer_comp, 0) * 1.0 / o.n_offer_recv
        END,
        4
    ) AS comp_rate
FROM raw_profile p
LEFT JOIN tx t
    ON CAST(p.id AS STRING) = CAST(t.person AS STRING)
LEFT JOIN off o
    ON CAST(p.id AS STRING) = CAST(o.person AS STRING)
ORDER BY user_id;
