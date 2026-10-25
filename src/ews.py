"""
Rule-based Early Warning Signals (EWS).

Modelled on the RBI framework for early recognition of stress (SMA
classification, Red-Flagged Account indicators) and typical bank EWS
playbooks. Each signal has a category and a weight; the weighted score maps
to a Green / Amber / Red watch-list level.
"""
import numpy as np
import pandas as pd

# (code, name, category, weight, condition)
SIGNALS = [
    ("EWS01", "DPD rising vs previous quarter", "Repayment", 3,
     lambda d: (d["dpd_trend"] > 0) & (d["max_dpd_3m"] > 0)),
    ("EWS02", "Repeated delinquency (>=3 of last 12 months)", "Repayment", 2,
     lambda d: d["n_delinquent_months_12m"] >= 3),
    ("EWS03", "Missed instalment in last 6 months", "Repayment", 2,
     lambda d: d["n_missed_payments_6m"] >= 1),
    ("EWS04", "Partial payments (<80% of due)", "Repayment", 1,
     lambda d: d["n_partial_payments_6m"] >= 2),
    ("EWS05", "NACH / ECS bounces in last 3 months", "Account conduct", 3,
     lambda d: d["n_bounces_3m"] >= 2),
    ("EWS06", "Account inflows down >30% QoQ", "Account conduct", 2,
     lambda d: d["credit_inflow_trend"] < 0.70),
    ("EWS07", "Salary credit irregular / stopped", "Account conduct", 3,
     lambda d: d["salary_regularity"] < 0.84),
    ("EWS08", "Cash withdrawals up >50% QoQ", "Account conduct", 1,
     lambda d: (d["cash_withdrawal_trend"] > 1.5) & (d["cash_withdrawal_share"] > 0.15)),
    ("EWS09", "Debits exceed credits by >20%", "Account conduct", 1,
     lambda d: d["debit_to_credit_ratio"] > 1.2),
    ("EWS10", "Credit card utilisation > 90%", "Leverage", 2,
     lambda d: d["cc_utilization"] > 0.90),
    ("EWS11", "Debt-to-income above 60%", "Leverage", 2,
     lambda d: d["dti"] > 0.60),
    ("EWS12", "Bureau score dropped >40 points in 6m", "Bureau", 3,
     lambda d: d["bureau_score_change_6m"] < -40),
    ("EWS13", "Bureau score below 650", "Bureau", 1,
     lambda d: d["bureau_score"] < 650),
]


def sma_class(dpd: pd.Series) -> pd.Series:
    """RBI Special Mention Account classification."""
    return pd.Series(np.select(
        [dpd == 0, dpd <= 30, dpd <= 60, dpd <= 90],
        ["Standard", "SMA-0", "SMA-1", "SMA-2"], "NPA"), index=dpd.index)


def run_ews(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits = []
    score = np.zeros(len(feat))
    n = np.zeros(len(feat), dtype=int)
    for code, name, cat, w, cond in SIGNALS:
        mask = cond(feat).fillna(False).to_numpy()
        score += w * mask
        n += mask
        if mask.any():
            hits.append(pd.DataFrame({"customer_id": feat.loc[mask, "customer_id"].to_numpy(),
                                      "signal_code": code, "signal_name": name,
                                      "category": cat, "weight": w}))
    sma = sma_class(feat["current_dpd"].fillna(0))
    level = np.where((score >= 8) | sma.isin(["SMA-2", "NPA"]), "Red",
                     np.where((score >= 4) | (sma == "SMA-1"), "Amber", "Green"))
    summary = pd.DataFrame({"customer_id": feat["customer_id"], "ews_score": score.astype(int),
                            "n_signals": n, "sma_class": sma.to_numpy(), "ews_level": level})
    signals = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(
        columns=["customer_id", "signal_code", "signal_name", "category", "weight"])
    return summary, signals
