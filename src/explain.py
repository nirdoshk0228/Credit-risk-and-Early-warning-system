"""
Explainable risk engine: SHAP (TreeExplainer) on the XGBoost PD model.

For every account we keep the top risk drivers (features pushing PD up) and
translate them into adverse-action style reason codes a credit officer can act on.
"""
import numpy as np
import pandas as pd
import shap

from src.features import MODEL_FEATURES

# feature -> (reason code, human-readable template using the feature value)
REASONS = {
    "current_dpd":              ("RC01", "Currently {v:.0f} days past due"),
    "max_dpd_3m":               ("RC02", "Delinquency in last 3 months (max {v:.0f} DPD)"),
    "max_dpd_6m":               ("RC02", "Delinquency in last 6 months (max {v:.0f} DPD)"),
    "max_dpd_12m":              ("RC03", "Past delinquency in last 12 months (max {v:.0f} DPD)"),
    "dpd_trend":                ("RC04", "Days past due rising vs previous quarter (+{v:.0f})"),
    "n_delinquent_months_12m":  ("RC05", "Delinquent in {v:.0f} of last 12 months"),
    "n_missed_payments_6m":     ("RC06", "{v:.0f} missed instalment(s) in last 6 months"),
    "n_partial_payments_6m":    ("RC07", "{v:.0f} partial payment(s) in last 6 months"),
    "payment_ratio_3m":         ("RC08", "Paid only {v:.0%} of amount due in last 3 months"),
    "payment_ratio_change":     ("RC08", "Repayment ratio fell by {v:+.0%} vs earlier months"),
    "avg_days_late_6m":         ("RC09", "Pays {v:.0f} days late on average"),
    "months_since_delinquency": ("RC10", "Recent delinquency ({v:.0f} month(s) ago)"),
    "n_bounces_3m":             ("RC11", "{v:.0f} NACH/ECS bounce(s) in last 3 months"),
    "n_bounces_6m":             ("RC11", "{v:.0f} NACH/ECS bounce(s) in last 6 months"),
    "credit_inflow_trend":      ("RC12", "Account inflows at {v:.0%} of previous quarter"),
    "inflow_to_income":         ("RC12", "Inflows only {v:.0%} of declared income"),
    "salary_regularity":        ("RC13", "Salary credited in only {v:.0%} of last 6 months"),
    "cash_withdrawal_share":    ("RC14", "Cash withdrawals are {v:.0%} of spend"),
    "cash_withdrawal_trend":    ("RC14", "Cash withdrawals up {v:.1f}x vs previous quarter"),
    "debit_to_credit_ratio":    ("RC15", "Spending is {v:.0%} of inflows"),
    "txn_activity_trend":       ("RC16", "Account activity at {v:.0%} of previous quarter"),
    "cc_utilization":           ("RC17", "Credit card utilisation at {v:.0%}"),
    "dti":                      ("RC18", "Debt-to-income ratio of {v:.0%}"),
    "unsecured_share":          ("RC19", "{v:.0%} of exposure is unsecured"),
    "max_ltv":                  ("RC20", "Loan-to-value of {v:.0%}"),
    "bureau_score":             ("RC21", "Bureau score of {v:.0f}"),
    "bureau_score_change_6m":   ("RC22", "Bureau score changed {v:+.0f} points in 6 months"),
    "log_income":               ("RC23", "Lower income band"),
    "log_total_outstanding":    ("RC24", "High total outstanding exposure"),
    "wavg_interest_rate":       ("RC25", "High-cost borrowing ({v:.1f}% weighted rate)"),
    "n_loans":                  ("RC26", "{v:.0f} active credit facilities"),
    "min_months_on_book":       ("RC27", "New relationship ({v:.0f} months on book)"),
    "years_with_bank":          ("RC27", "Short banking relationship ({v:.1f} years)"),
    "emp_gig":                  ("RC28", "Gig / contract employment (income volatility)"),
    "emp_self_employed":        ("RC28", "Self-employed (income volatility)"),
    "emp_business_owner":       ("RC28", "Business owner (cash-flow volatility)"),
    "city_tier":                ("RC29", "Tier-{v:.0f} city location"),
}


def reason_text(feature, value):
    code, tmpl = REASONS.get(feature, ("RC99", feature.replace("_", " ") + " = {v:.2f}"))
    try:
        return code, tmpl.format(v=value)
    except (ValueError, TypeError):
        return code, tmpl


def explain(model, perf: pd.DataFrame, top_n: int = 8):
    X = perf[MODEL_FEATURES]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    base = float(np.ravel(explainer.expected_value)[0])

    importance = pd.DataFrame({"feature": MODEL_FEATURES,
                               "mean_abs_shap": np.abs(sv).mean(axis=0),
                               "mean_shap": sv.mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    order = np.argsort(-sv, axis=1)[:, :top_n]
    rows = []
    vals = X.to_numpy()
    ids = perf["customer_id"].to_numpy()
    for r in range(len(perf)):
        for rank, j in enumerate(order[r], start=1):
            s = sv[r, j]
            if s <= 0 and rank > 3:
                break
            code, text = reason_text(MODEL_FEATURES[j], vals[r, j])
            rows.append((ids[r], rank, MODEL_FEATURES[j], float(vals[r, j]), float(s), code, text))
    drivers = pd.DataFrame(rows, columns=["customer_id", "driver_rank", "feature", "feature_value",
                                          "shap_value", "reason_code", "reason_text"])
    drivers["direction"] = np.where(drivers["shap_value"] > 0, "increases risk", "reduces risk")
    log_odds = pd.DataFrame({"customer_id": ids, "shap_base_value": base,
                             "shap_log_odds": base + sv.sum(axis=1)})
    return drivers, importance, log_odds
