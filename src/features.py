"""Customer-level feature store built from in-database SQL aggregates."""
import numpy as np
import pandas as pd

from src.db import load_named_queries, read_sql

# Features fed to the PD models (behavioural + bureau + exposure)
MODEL_FEATURES = [
    # profile & bureau
    "age", "log_income", "years_with_bank", "city_tier", "bureau_score", "bureau_score_change_6m",
    "emp_self_employed", "emp_business_owner", "emp_gig", "emp_retired",
    # exposure / loan
    "n_loans", "log_total_outstanding", "dti", "unsecured_share", "wavg_interest_rate",
    "max_ltv", "cc_utilization", "min_months_on_book",
    "has_personal_loan", "has_credit_card", "has_business_loan",
    # repayment behaviour
    "current_dpd", "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "dpd_trend",
    "n_delinquent_months_12m", "n_missed_payments_6m", "n_partial_payments_6m",
    "payment_ratio_3m", "payment_ratio_change", "avg_days_late_6m", "months_since_delinquency",
    # account / transaction behaviour
    "credit_inflow_trend", "inflow_to_income", "debit_to_credit_ratio", "cash_withdrawal_share",
    "cash_withdrawal_trend", "n_bounces_3m", "n_bounces_6m", "salary_regularity", "txn_activity_trend",
]


def _safe_ratio(a, b, default=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.where(b > 0, a / np.where(b > 0, b, 1), default)
    return out


def build_features(engine) -> pd.DataFrame:
    q = load_named_queries("features.sql")
    cust = read_sql("SELECT * FROM customers ORDER BY customer_id", engine)
    lf = read_sql(q["loan_features"], engine)
    pf = read_sql(q["payment_features"], engine)
    tf = read_sql(q["transaction_features"], engine)

    df = cust.merge(lf, on="customer_id", how="left") \
             .merge(pf, on="customer_id", how="left") \
             .merge(tf, on="customer_id", how="left")

    inc = df["monthly_income"]
    df["log_income"] = np.log(inc)
    df["bureau_score_change_6m"] = df["bureau_score"] - df["bureau_score_6m_ago"]
    for col, label in [("emp_self_employed", "Self-Employed"), ("emp_business_owner", "Business Owner"),
                       ("emp_gig", "Gig / Contract"), ("emp_retired", "Retired")]:
        df[col] = (df["employment_type"] == label).astype(int)

    df["log_total_outstanding"] = np.log1p(df["total_outstanding"])
    df["dti"] = df["total_emi"] / inc
    df["unsecured_share"] = _safe_ratio(df["unsecured_outstanding"], df["total_outstanding"], 0.0)
    df["max_ltv"] = df["max_ltv"].fillna(0.0)
    df["cc_utilization"] = df["cc_utilization"].fillna(0.0)

    df["dpd_trend"] = df["max_dpd_3m"] - df["max_dpd_prev_3m"]
    df["payment_ratio_3m"] = df["payment_ratio_3m"].fillna(1.0).clip(0, 3)
    df["payment_ratio_change"] = df["payment_ratio_3m"] - df["payment_ratio_prev"].fillna(1.0).clip(0, 3)
    df["avg_days_late_6m"] = df["avg_days_late_6m"].fillna(0.0)
    df["months_since_delinquency"] = df["months_since_delinquency"].fillna(13)

    df["credit_inflow_trend"] = _safe_ratio(df["avg_credit_3m"], df["avg_credit_prev_3m"]).clip(0, 5)
    df["inflow_to_income"] = (df["avg_credit_3m"] / inc).clip(0, 50)
    df["debit_to_credit_ratio"] = _safe_ratio(df["avg_debit_3m"], df["avg_credit_3m"], 2.0).clip(0, 5)
    df["cash_withdrawal_share"] = _safe_ratio(df["avg_cash_withdrawal_3m"], df["avg_debit_3m"], 0.0).clip(0, 1)
    df["cash_withdrawal_trend"] = _safe_ratio(df["avg_cash_withdrawal_3m"],
                                              df["avg_cash_withdrawal_prev_3m"]).clip(0, 10)
    salaried = df["employment_type"].isin(["Salaried", "Retired"])
    df["salary_regularity"] = np.where(salaried, df["n_salary_months_6m"] / 6.0, 1.0)
    df["txn_activity_trend"] = _safe_ratio(df["txn_count_3m"], df["txn_count_prev_3m"]).clip(0, 10)

    count_cols = ["n_bounces_3m", "n_bounces_6m", "n_near_threshold_cash", "n_international",
                  "n_odd_hour_high_value", "n_round_amount", "n_transfer_counterparties"]
    df[count_cols] = df[count_cols].fillna(0).astype(int)
    return df.sort_values("customer_id").reset_index(drop=True)   # deterministic across DB backends
