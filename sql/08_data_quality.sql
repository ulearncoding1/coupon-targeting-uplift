-- 08 数据质量检查：主键、孤儿键、空值与事件白名单
SELECT 'transcript_rows' AS check_name, COUNT(*) AS total, 0 AS failures
FROM raw_transcript
UNION ALL
SELECT 'unknown_event_types', COUNT(*),
       SUM(CASE WHEN event NOT IN ('transaction', 'offer received', 'offer viewed', 'offer completed') THEN 1 ELSE 0 END)
FROM raw_transcript
UNION ALL
SELECT 'duplicate_user_id', COUNT(*),
       SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END)
FROM (
    SELECT user_id, COUNT(*) AS n
    FROM user_features
    GROUP BY user_id
) u
UNION ALL
SELECT 'duplicate_offer_send_key', COUNT(*),
       SUM(CASE WHEN n > 1 THEN n - 1 ELSE 0 END)
FROM (
    SELECT person, offer_id, send_time, COUNT(*) AS n
    FROM offer_dataset
    GROUP BY person, offer_id, send_time
) o
UNION ALL
SELECT 'orphan_transcript_users', COUNT(*),
       SUM(CASE WHEN p.id IS NULL THEN 1 ELSE 0 END)
FROM raw_transcript t
LEFT JOIN raw_profile p
    ON CAST(t.person AS STRING) = CAST(p.id AS STRING)
UNION ALL
SELECT 'orphan_transcript_offers', COUNT(*),
       SUM(CASE WHEN pf.id IS NULL THEN 1 ELSE 0 END)
FROM raw_transcript t
LEFT JOIN raw_portfolio pf
    ON CAST(t.offer_id AS STRING) = CAST(pf.id AS STRING)
WHERE t.offer_id IS NOT NULL
UNION ALL
SELECT 'missing_user_features', COUNT(*),
       SUM(CASE WHEN uf.user_id IS NULL THEN 1 ELSE 0 END)
FROM raw_profile p
LEFT JOIN user_features uf
    ON CAST(p.id AS STRING) = CAST(uf.user_id AS STRING)
UNION ALL
SELECT 'missing_offer_split', COUNT(*),
       SUM(CASE WHEN split IS NULL THEN 1 ELSE 0 END)
FROM offer_dataset;
