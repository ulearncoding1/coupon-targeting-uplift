-- 09 v2 策略汇总：从 05_policy_v2.csv 汇总补贴、增量 GMV、增量净额与 ROI
SELECT
    'v2' AS scenario,
    ROUND(SUM(v2_est_subsidy), 0) AS subsidy,
    ROUND(SUM(v2_est_inc_gmv), 0) AS inc_gmv,
    ROUND(SUM(v2_est_inc_net), 0) AS inc_net,
    ROUND(
        SUM(v2_est_inc_gmv) / NULLIF(SUM(v2_est_subsidy), 0),
        3
    ) AS incremental_roi
FROM policy_v2;
