-- Analytics layer consumed by the Streamlit dashboard (portable SQL).

CREATE VIEW v_portfolio_kpis AS
SELECT
    COUNT(DISTINCT e.customer_id)                                                    AS n_customers,
    COUNT(*)                                                                         AS n_loans,
    SUM(e.ead)                                                                       AS total_ead,
    SUM(e.ecl)                                                                       AS total_ecl,
    SUM(e.ecl) / SUM(e.ead)                                                          AS coverage_ratio,
    SUM(CASE WHEN e.stage = 3 THEN e.ead ELSE 0 END) / SUM(e.ead)                    AS gnpa_ratio,
    SUM(CASE WHEN e.stage = 2 THEN e.ead ELSE 0 END) / SUM(e.ead)                    AS stage2_ratio,
    SUM(CASE WHEN e.stage < 3 THEN e.pd_12m * e.ead ELSE 0 END)
      / NULLIF(SUM(CASE WHEN e.stage < 3 THEN e.ead ELSE 0 END), 0)                  AS wavg_pd_performing,
    SUM(CASE WHEN e.stage = 3 THEN e.ecl ELSE 0 END)
      / NULLIF(SUM(CASE WHEN e.stage = 3 THEN e.ead ELSE 0 END), 0)                  AS stage3_coverage,
    (SELECT COUNT(*) FROM customer_risk WHERE ews_level = 'Red')                     AS n_ews_red,
    (SELECT COUNT(*) FROM customer_risk WHERE ews_level = 'Amber')                   AS n_ews_amber,
    (SELECT COUNT(*) FROM anomaly_scores WHERE is_flagged = 1)                       AS n_aml_flagged
FROM ecl_results e;

CREATE VIEW v_exposure_by_product AS
SELECT
    product,
    COUNT(*)                                                         AS n_loans,
    SUM(ead)                                                         AS ead,
    SUM(ecl)                                                         AS ecl,
    SUM(ecl) / SUM(ead)                                              AS coverage,
    SUM(pd_12m * ead) / SUM(ead)                                     AS wavg_pd,
    SUM(lgd * ead) / SUM(ead)                                        AS wavg_lgd,
    SUM(CASE WHEN stage = 1 THEN ead ELSE 0 END)                     AS stage1_ead,
    SUM(CASE WHEN stage = 2 THEN ead ELSE 0 END)                     AS stage2_ead,
    SUM(CASE WHEN stage = 3 THEN ead ELSE 0 END)                     AS stage3_ead,
    SUM(CASE WHEN stage = 3 THEN ead ELSE 0 END) / SUM(ead)          AS npa_ratio
FROM ecl_results
GROUP BY product;

CREATE VIEW v_stage_summary AS
SELECT
    stage,
    COUNT(*)                     AS n_loans,
    SUM(ead)                     AS ead,
    SUM(ecl)                     AS ecl,
    SUM(ecl) / SUM(ead)          AS coverage,
    SUM(pd_12m * ead) / SUM(ead) AS wavg_pd
FROM ecl_results
GROUP BY stage;

CREATE VIEW v_exposure_by_segment AS
SELECT 'State' AS dimension, c.state AS segment, COUNT(*) AS n_loans, SUM(e.ead) AS ead, SUM(e.ecl) AS ecl,
       SUM(e.pd_12m * e.ead) / SUM(e.ead) AS wavg_pd, SUM(e.ecl) / SUM(e.ead) AS coverage
FROM ecl_results e JOIN customers c ON c.customer_id = e.customer_id
GROUP BY c.state
UNION ALL
SELECT 'Employment', c.employment_type, COUNT(*), SUM(e.ead), SUM(e.ecl),
       SUM(e.pd_12m * e.ead) / SUM(e.ead), SUM(e.ecl) / SUM(e.ead)
FROM ecl_results e JOIN customers c ON c.customer_id = e.customer_id
GROUP BY c.employment_type
UNION ALL
SELECT 'City tier', 'Tier ' || CAST(c.city_tier AS VARCHAR(2)), COUNT(*), SUM(e.ead), SUM(e.ecl),
       SUM(e.pd_12m * e.ead) / SUM(e.ead), SUM(e.ecl) / SUM(e.ead)
FROM ecl_results e JOIN customers c ON c.customer_id = e.customer_id
GROUP BY c.city_tier;

CREATE VIEW v_ticket_size AS
SELECT
    bucket, bucket_order, COUNT(*) AS n_loans, SUM(ead) AS ead, SUM(ecl) AS ecl
FROM (
    SELECT ead, ecl,
           CASE WHEN ead < 100000 THEN '< 1L' WHEN ead < 500000 THEN '1L - 5L'
                WHEN ead < 2500000 THEN '5L - 25L' WHEN ead < 10000000 THEN '25L - 1Cr'
                ELSE '> 1Cr' END AS bucket,
           CASE WHEN ead < 100000 THEN 1 WHEN ead < 500000 THEN 2 WHEN ead < 2500000 THEN 3
                WHEN ead < 10000000 THEN 4 ELSE 5 END AS bucket_order
    FROM ecl_results
) t
GROUP BY bucket, bucket_order;

CREATE VIEW v_top_exposures AS
SELECT customer_id, employment_type, state, total_ead, total_ecl, pd_12m, risk_band,
       ews_level, risk_severity, max_stage
FROM customer_risk
ORDER BY total_ead DESC
LIMIT 25;

CREATE VIEW v_risk_band_summary AS
SELECT
    risk_band,
    COUNT(*)              AS n_customers,
    SUM(total_ead)        AS ead,
    SUM(total_ecl)        AS ecl,
    AVG(pd_12m)           AS avg_pd,
    MIN(risk_score)       AS min_score,
    MAX(risk_score)       AS max_score
FROM customer_risk
GROUP BY risk_band;

CREATE VIEW v_severity_matrix AS
SELECT
    r.risk_severity, e.product,
    COUNT(*)     AS n_loans,
    SUM(e.ead)   AS ead,
    SUM(e.ecl)   AS ecl
FROM ecl_results e JOIN customer_risk r ON r.customer_id = e.customer_id
GROUP BY r.risk_severity, e.product;

CREATE VIEW v_sma_summary AS
SELECT sma_class, COUNT(*) AS n_customers, SUM(total_ead) AS ead, AVG(pd_12m) AS avg_pd
FROM customer_risk
GROUP BY sma_class;

CREATE VIEW v_ews_signal_frequency AS
SELECT
    s.signal_code, s.signal_name, s.category, s.weight,
    COUNT(*)                     AS n_customers,
    SUM(r.total_ead)             AS ead_flagged,
    AVG(r.pd_12m)                AS avg_pd
FROM ews_signals s JOIN customer_risk r ON r.customer_id = s.customer_id
GROUP BY s.signal_code, s.signal_name, s.category, s.weight;

CREATE VIEW v_ews_watchlist AS
SELECT
    r.customer_id, r.employment_type, r.state, r.ews_level, r.ews_score, r.n_signals, r.sma_class,
    r.current_dpd, r.pd_12m, r.risk_band, r.total_ead, r.total_ecl, r.max_stage,
    r.risk_severity, r.top_risk_driver
FROM customer_risk r
WHERE r.ews_level IN ('Red', 'Amber');

CREATE VIEW v_anomaly_alerts AS
SELECT
    a.customer_id, c.employment_type, c.monthly_income, c.kyc_risk,
    a.alert_score, a.alert_priority, a.isolation_percentile, a.rule_points,
    a.inflow_to_income, a.pass_through_ratio, a.n_near_threshold_cash, a.n_odd_hour_high_value,
    (SELECT COUNT(*) FROM txn_alerts t WHERE t.customer_id = a.customer_id) AS n_txn_alerts
FROM anomaly_scores a JOIN customers c ON c.customer_id = a.customer_id
WHERE a.is_flagged = 1;

CREATE VIEW v_aml_rule_summary AS
SELECT rule_code, rule_name, weight, COUNT(*) AS n_customers
FROM aml_rule_hits
GROUP BY rule_code, rule_name, weight;
