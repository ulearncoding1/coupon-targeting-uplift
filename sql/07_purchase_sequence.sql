-- 07 用户购买序列：ROW_NUMBER + LAG 计算相邻两单间隔
-- 原始 time 的单位是小时（活动窗口 0~714 小时），不要再除以 3600。
WITH tx AS (
    SELECT
        person,
        time,
        ROW_NUMBER() OVER (PARTITION BY person ORDER BY time) AS rn,
        LAG(time) OVER (PARTITION BY person ORDER BY time) AS prev_time
    FROM raw_transcript
    WHERE event = 'transaction'
),
gaps AS (
    SELECT
        person,
        time - prev_time AS gap_hours
    FROM tx
    WHERE prev_time IS NOT NULL
)
SELECT
    COUNT(*) AS n_transitions,
    COUNT(DISTINCT person) AS n_users,
    ROUND(AVG(gap_hours), 2) AS avg_gap_hours,
    ROUND(MIN(gap_hours), 2) AS min_gap_hours,
    ROUND(MAX(gap_hours), 2) AS max_gap_hours
FROM gaps;
