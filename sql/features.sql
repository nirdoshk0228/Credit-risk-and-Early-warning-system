-- Feature engineering queries, executed in-database.
-- Portable SQL (PostgreSQL / SQLite): month_idx windows instead of date maths.

-- name: loan_features
SELECT
    customer_id,
    COUNT(*)                                                             AS n_loans,
    SUM(outstanding)                                                     AS total_outstanding,
    SUM(emi)                                                             AS total_emi,
    SUM(CASE WHEN secured = 1 THEN outstanding ELSE 0 END)               AS secured_outstanding,
    SUM(CASE WHEN product IN ('Personal Loan', 'Credit Card')
             THEN outstanding ELSE 0 END)                                AS unsecured_outstanding,
    SUM(interest_rate * outstanding) / NULLIF(SUM(outstanding), 0)       AS wavg_interest_rate,
    MAX(CASE WHEN collateral_value > 0
             THEN outstanding / collateral_value END)                    AS max_ltv,
    MAX(CASE WHEN product = 'Credit Card'
             THEN outstanding / NULLIF(credit_limit, 0) END)             AS cc_utilization,
    MIN(months_on_book)                                                  AS min_months_on_book,
    AVG(1.0 * months_on_book)                                            AS avg_months_on_book,
    MAX(CASE WHEN product = 'Personal Loan' THEN 1 ELSE 0 END)           AS has_personal_loan,
    MAX(CASE WHEN product = 'Credit Card'   THEN 1 ELSE 0 END)           AS has_credit_card,
    MAX(CASE WHEN product = 'Home Loan'     THEN 1 ELSE 0 END)           AS has_home_loan,
    MAX(CASE WHEN product = 'Business Loan' THEN 1 ELSE 0 END)           AS has_business_loan
FROM loans
GROUP BY customer_id;

-- name: payment_features
SELECT
    l.customer_id,
    MAX(CASE WHEN p.month_idx = 1 THEN p.dpd ELSE 0 END)                 AS current_dpd,
    MAX(CASE WHEN p.month_idx <= 3 THEN p.dpd ELSE 0 END)                AS max_dpd_3m,
    MAX(CASE WHEN p.month_idx BETWEEN 4 AND 6 THEN p.dpd ELSE 0 END)     AS max_dpd_prev_3m,
    MAX(CASE WHEN p.month_idx <= 6 THEN p.dpd ELSE 0 END)                AS max_dpd_6m,
    MAX(p.dpd)                                                           AS max_dpd_12m,
    AVG(CASE WHEN p.month_idx <= 6 THEN 1.0 * p.dpd END)                 AS avg_dpd_6m,
    COUNT(DISTINCT CASE WHEN p.dpd > 0 THEN p.month_idx END)             AS n_delinquent_months_12m,
    COUNT(DISTINCT CASE WHEN p.dpd > 0 AND p.month_idx <= 6
                        THEN p.month_idx END)                            AS n_delinquent_months_6m,
    SUM(CASE WHEN p.month_idx <= 6 AND p.paid_amount = 0 THEN 1 ELSE 0 END) AS n_missed_payments_6m,
    SUM(CASE WHEN p.month_idx <= 6 AND p.paid_amount > 0
              AND p.paid_amount < 0.8 * p.due_amount THEN 1 ELSE 0 END)  AS n_partial_payments_6m,
    SUM(CASE WHEN p.month_idx <= 3 THEN p.paid_amount ELSE 0 END)
      / NULLIF(SUM(CASE WHEN p.month_idx <= 3 THEN p.due_amount ELSE 0 END), 0)   AS payment_ratio_3m,
    SUM(CASE WHEN p.month_idx BETWEEN 4 AND 12 THEN p.paid_amount ELSE 0 END)
      / NULLIF(SUM(CASE WHEN p.month_idx BETWEEN 4 AND 12 THEN p.due_amount ELSE 0 END), 0) AS payment_ratio_prev,
    AVG(CASE WHEN p.month_idx <= 6 THEN p.days_late END)                 AS avg_days_late_6m,
    MIN(CASE WHEN p.dpd > 0 THEN p.month_idx END)                        AS months_since_delinquency
FROM payments p
JOIN loans l ON l.loan_id = p.loan_id
GROUP BY l.customer_id;

-- name: transaction_features
SELECT
    customer_id,
    SUM(CASE WHEN direction = 'CR' AND status = 'SUCCESS' AND month_idx <= 3
             THEN amount ELSE 0 END) / 3.0                               AS avg_credit_3m,
    SUM(CASE WHEN direction = 'CR' AND status = 'SUCCESS' AND month_idx BETWEEN 4 AND 6
             THEN amount ELSE 0 END) / 3.0                               AS avg_credit_prev_3m,
    SUM(CASE WHEN direction = 'DR' AND status = 'SUCCESS' AND month_idx <= 3
             THEN amount ELSE 0 END) / 3.0                               AS avg_debit_3m,
    SUM(CASE WHEN category = 'CASH_WITHDRAWAL' AND month_idx <= 3
             THEN amount ELSE 0 END) / 3.0                               AS avg_cash_withdrawal_3m,
    SUM(CASE WHEN category = 'CASH_WITHDRAWAL' AND month_idx BETWEEN 4 AND 6
             THEN amount ELSE 0 END) / 3.0                               AS avg_cash_withdrawal_prev_3m,
    SUM(CASE WHEN status = 'RETURNED' AND month_idx <= 3 THEN 1 ELSE 0 END) AS n_bounces_3m,
    SUM(CASE WHEN status = 'RETURNED' THEN 1 ELSE 0 END)                 AS n_bounces_6m,
    COUNT(DISTINCT CASE WHEN category = 'SALARY' THEN month_idx END)     AS n_salary_months_6m,
    COUNT(DISTINCT CASE WHEN category = 'SALARY' AND month_idx <= 3
                        THEN month_idx END)                              AS n_salary_months_3m,
    SUM(CASE WHEN month_idx <= 3 THEN 1 ELSE 0 END)                      AS txn_count_3m,
    SUM(CASE WHEN month_idx BETWEEN 4 AND 6 THEN 1 ELSE 0 END)           AS txn_count_prev_3m,
    -- AML-oriented behaviour
    SUM(CASE WHEN category = 'CASH_DEPOSIT' THEN amount ELSE 0 END)      AS cash_deposit_6m,
    SUM(CASE WHEN category = 'CASH_DEPOSIT' AND amount >= 40000
              AND amount < 50000 THEN 1 ELSE 0 END)                      AS n_near_threshold_cash,
    SUM(CASE WHEN category = 'INTERNATIONAL' THEN 1 ELSE 0 END)          AS n_international,
    SUM(CASE WHEN category = 'INTERNATIONAL' THEN amount ELSE 0 END)     AS international_amount,
    SUM(CASE WHEN txn_hour < 5 AND amount >= 50000 THEN 1 ELSE 0 END)    AS n_odd_hour_high_value,
    SUM(CASE WHEN amount >= 10000
              AND CAST(amount AS INTEGER) % 10000 = 0 THEN 1 ELSE 0 END) AS n_round_amount,
    SUM(CASE WHEN category = 'TRANSFER_IN' THEN amount ELSE 0 END)       AS transfer_in_6m,
    SUM(CASE WHEN category = 'TRANSFER_OUT' THEN amount ELSE 0 END)      AS transfer_out_6m,
    COUNT(DISTINCT CASE WHEN category IN ('TRANSFER_IN', 'TRANSFER_OUT')
                        THEN counterparty_id END)                        AS n_transfer_counterparties,
    MAX(amount)                                                          AS max_txn_amount
FROM transactions
GROUP BY customer_id;

-- name: monthly_flows
SELECT
    customer_id,
    month_idx,
    SUM(CASE WHEN direction = 'CR' AND status = 'SUCCESS' THEN amount ELSE 0 END) AS credits,
    SUM(CASE WHEN direction = 'DR' AND status = 'SUCCESS' THEN amount ELSE 0 END) AS debits,
    COUNT(*)                                                                    AS n_txn
FROM transactions
GROUP BY customer_id, month_idx;
