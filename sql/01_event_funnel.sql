-- 01 事件漏斗：看 306,534 条事件由什么构成
WITH event_counts AS (
    SELECT
        event,
        COUNT(*) AS n_events,
        COUNT(DISTINCT person) AS n_users
    FROM raw_transcript
    GROUP BY event
),
total AS (
    SELECT COUNT(*) AS total_events
    FROM raw_transcript
)
SELECT
    e.event,
    e.n_events,
    e.n_users,
    ROUND(100.0 * e.n_events / t.total_events, 2) AS share_pct
FROM event_counts e
CROSS JOIN total t
ORDER BY e.n_events DESC;
