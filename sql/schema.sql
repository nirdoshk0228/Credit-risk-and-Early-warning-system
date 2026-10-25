-- Core banking schema (portable: PostgreSQL primary, SQLite fallback)
-- Analytical output tables (features, scores, ECL, alerts) are written by the pipeline.

DROP TABLE IF EXISTS aml_ground_truth;
DROP TABLE IF EXISTS default_outcomes;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS macro_history;

CREATE TABLE customers (
    customer_id          VARCHAR(12) PRIMARY KEY,
    age                  INTEGER NOT NULL,
    employment_type      VARCHAR(20) NOT NULL,
    city_tier            INTEGER NOT NULL,
    state                VARCHAR(30) NOT NULL,
    monthly_income       DOUBLE PRECISION NOT NULL,
    years_with_bank      DOUBLE PRECISION NOT NULL,
    bureau_score         INTEGER NOT NULL,
    bureau_score_6m_ago  INTEGER NOT NULL,
    kyc_risk             VARCHAR(10) NOT NULL
);

CREATE TABLE loans (
    loan_id              VARCHAR(12) PRIMARY KEY,
    customer_id          VARCHAR(12) NOT NULL REFERENCES customers(customer_id),
    product              VARCHAR(20) NOT NULL,
    sanctioned_amount    DOUBLE PRECISION NOT NULL,
    credit_limit         DOUBLE PRECISION,
    outstanding          DOUBLE PRECISION NOT NULL,
    interest_rate        DOUBLE PRECISION NOT NULL,
    tenure_months        INTEGER NOT NULL,
    months_on_book       INTEGER NOT NULL,
    emi                  DOUBLE PRECISION NOT NULL,
    collateral_value     DOUBLE PRECISION NOT NULL,
    secured              INTEGER NOT NULL,
    origination_date     DATE NOT NULL
);

-- month_idx: 1 = snapshot month (Dec-2025) ... 12 = Jan-2025
CREATE TABLE payments (
    payment_id           INTEGER PRIMARY KEY,
    loan_id              VARCHAR(12) NOT NULL REFERENCES loans(loan_id),
    month_idx            INTEGER NOT NULL,
    due_date             DATE NOT NULL,
    due_amount           DOUBLE PRECISION NOT NULL,
    paid_amount          DOUBLE PRECISION NOT NULL,
    days_late            DOUBLE PRECISION,
    dpd                  INTEGER NOT NULL
);

CREATE TABLE transactions (
    txn_id               INTEGER PRIMARY KEY,
    customer_id          VARCHAR(12) NOT NULL REFERENCES customers(customer_id),
    txn_date             DATE NOT NULL,
    month_idx            INTEGER NOT NULL,
    txn_hour             INTEGER NOT NULL,
    amount               DOUBLE PRECISION NOT NULL,
    direction            VARCHAR(2) NOT NULL,
    channel              VARCHAR(12) NOT NULL,
    category             VARCHAR(20) NOT NULL,
    status               VARCHAR(10) NOT NULL,
    counterparty_id      VARCHAR(20)
);

CREATE TABLE macro_history (
    quarter                 VARCHAR(8) PRIMARY KEY,
    gdp_growth              DOUBLE PRECISION,
    unemployment_rate       DOUBLE PRECISION,
    repo_rate               DOUBLE PRECISION,
    hpi_growth              DOUBLE PRECISION,
    portfolio_default_rate  DOUBLE PRECISION
);

-- Labels: kept apart from features; used only for model training / evaluation
CREATE TABLE default_outcomes (
    customer_id          VARCHAR(12) PRIMARY KEY REFERENCES customers(customer_id),
    default_12m          INTEGER NOT NULL,
    already_npa          INTEGER NOT NULL
);

CREATE TABLE aml_ground_truth (
    customer_id          VARCHAR(12) PRIMARY KEY REFERENCES customers(customer_id),
    pattern              VARCHAR(20) NOT NULL
);

CREATE INDEX ix_loans_customer   ON loans(customer_id);
CREATE INDEX ix_payments_loan    ON payments(loan_id, month_idx);
CREATE INDEX ix_txn_customer     ON transactions(customer_id, month_idx);
CREATE INDEX ix_txn_category     ON transactions(category, status);
