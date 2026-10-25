"""
Transaction anomaly detection: unsupervised Isolation Forest on account
behaviour + rule-based AML typologies (structuring, pass-through, odd-hour
international transfers, dormant reactivation, income mismatch).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.db import load_named_queries, read_sql

BEHAVIOUR_FEATURES = [
    "inflow_to_income", "max_month_credit_to_income", "pass_through_ratio", "cash_deposit_to_income",
    "n_near_threshold_cash", "n_international", "intl_to_income", "n_odd_hour_high_value",
    "n_round_amount", "n_transfer_counterparties", "activity_burst", "max_txn_to_income",
    "transfer_volume_to_income",
]

AML_RULES = [
    ("AML01", "Structuring: repeated cash deposits just below Rs 50,000", 3,
     lambda d: d["n_near_threshold_cash"] >= 4),
    ("AML02", "Rapid pass-through: large inflow moved out within days", 3,
     lambda d: d["pass_through_ratio"] >= 0.9),
    ("AML03", "High-value transfers at odd hours (00:00-05:00)", 3,
     lambda d: d["n_odd_hour_high_value"] >= 2),
    ("AML04", "Dormant account reactivated with burst of activity", 2,
     lambda d: (d["txn_count_prev_3m"] <= 6) & (d["txn_count_3m"] >= 30)),
    ("AML05", "Inflows inconsistent with declared income (>4x)", 1,
     lambda d: d["inflow_to_income"] > 4),
    ("AML06", "Frequent round-figure high-value transactions", 1,
     lambda d: d["n_round_amount"] >= 5),
]


def behaviour_features(feat: pd.DataFrame, engine) -> pd.DataFrame:
    flows = read_sql(load_named_queries("features.sql")["monthly_flows"], engine)
    flows = flows.merge(feat[["customer_id", "monthly_income"]], on="customer_id")
    big = flows["credits"] > 3 * flows["monthly_income"]
    flows["pt"] = np.where(big, np.minimum(flows["debits"] / flows["credits"].clip(lower=1), 1.0), 0.0)
    agg = flows.groupby("customer_id").agg(
        max_month_credit=("credits", "max"), pass_through_ratio=("pt", "max")).reset_index()
    d = feat.merge(agg, on="customer_id", how="left").sort_values("customer_id").reset_index(drop=True)
    inc = d["monthly_income"]
    d["max_month_credit_to_income"] = d["max_month_credit"] / inc
    d["cash_deposit_to_income"] = d["cash_deposit_6m"] / (6 * inc)
    d["intl_to_income"] = d["international_amount"] / (6 * inc)
    d["activity_burst"] = d["txn_count_3m"] / (d["txn_count_prev_3m"] + 1)
    d["max_txn_to_income"] = d["max_txn_amount"] / inc
    d["transfer_volume_to_income"] = (d["transfer_in_6m"] + d["transfer_out_6m"]) / (6 * inc)
    d[BEHAVIOUR_FEATURES] = d[BEHAVIOUR_FEATURES].fillna(0)
    return d


def run_anomaly(feat: pd.DataFrame, engine, seed: int = 42):
    d = behaviour_features(feat, engine)
    X = np.log1p(d[BEHAVIOUR_FEATURES].clip(lower=0).to_numpy())
    iso = IsolationForest(n_estimators=400, contamination="auto", random_state=seed, n_jobs=-1)
    iso.fit(X)
    raw = -iso.score_samples(X)
    if_pct = pd.Series(raw).rank(pct=True).to_numpy() * 100

    rule_points = np.zeros(len(d))
    hits = []
    for code, name, w, cond in AML_RULES:
        mask = cond(d).fillna(False).to_numpy()
        rule_points += w * mask
        if mask.any():
            hits.append(pd.DataFrame({"customer_id": d.loc[mask, "customer_id"].to_numpy(),
                                      "rule_code": code, "rule_name": name, "weight": w}))
    alert_score = 0.5 * if_pct + 50 * np.minimum(rule_points / 3, 1)
    # alert if strong typology hit, or moderate hit corroborated by the model, or extreme outlier
    flagged = (rule_points >= 3) | ((rule_points >= 2) & (if_pct >= 97)) | (if_pct >= 99.5)
    priority = np.where(~flagged, "None",
                        np.where(alert_score >= 85, "High", np.where(alert_score >= 65, "Medium", "Low")))
    scores = pd.DataFrame({
        "customer_id": d["customer_id"], "isolation_score": raw.round(5),
        "isolation_percentile": if_pct.round(2), "rule_points": rule_points.astype(int),
        "alert_score": alert_score.round(2), "is_flagged": flagged.astype(int), "alert_priority": priority,
    })
    for c in BEHAVIOUR_FEATURES:
        scores[c] = d[c].round(4).to_numpy()
    rule_hits = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(
        columns=["customer_id", "rule_code", "rule_name", "weight"])
    txn_alerts = _transaction_alerts(scores, engine)
    metrics = _evaluate(scores, engine)
    return scores, rule_hits, txn_alerts, metrics


def _transaction_alerts(scores, engine) -> pd.DataFrame:
    """Pull the specific suspicious transactions behind each flagged account."""
    ids = scores.loc[scores["is_flagged"] == 1, "customer_id"].tolist()
    if not ids:
        return pd.DataFrame()
    id_list = ",".join(f"'{i}'" for i in ids)
    t = read_sql(f"""SELECT t.*, c.monthly_income FROM transactions t
                     JOIN customers c ON c.customer_id = t.customer_id
                     WHERE t.customer_id IN ({id_list})""", engine)
    # per-customer robust z-score of amount
    g = t.groupby("customer_id")["amount"]
    med = g.transform("median")
    mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
    t["amount_zscore"] = ((t["amount"] - med) / (1.4826 * mad)).fillna(0)
    reasons = [
        ((t["category"] == "CASH_DEPOSIT") & t["amount"].between(40_000, 49_999.99), "Near-threshold cash deposit"),
        ((t["txn_hour"] < 5) & (t["amount"] >= 50_000), "High-value transaction at odd hour"),
        (t["channel"] == "SWIFT", "Outward international wire"),
        (t["category"].isin(["TRANSFER_IN", "TRANSFER_OUT"]) & (t["amount"] > 2 * t["monthly_income"]),
         "Transfer > 2x monthly income"),
        ((t["amount_zscore"] > 6) & (t["amount"] >= 50_000), "Amount far outside customer's normal range"),
    ]
    t["alert_reason"] = None
    for mask, label in reversed(reasons):
        t.loc[mask, "alert_reason"] = label
    out = t[t["alert_reason"].notna()].drop(columns="monthly_income")
    out["amount_zscore"] = out["amount_zscore"].round(2)
    return out.reset_index(drop=True)


def _evaluate(scores, engine) -> dict:
    truth = read_sql("SELECT customer_id, pattern FROM aml_ground_truth", engine)
    y = scores["customer_id"].isin(truth["customer_id"]).astype(int).to_numpy()
    k = int(y.sum())
    top_k = scores["alert_score"].rank(ascending=False, method="first") <= k
    flagged = scores["is_flagged"].to_numpy() == 1
    tp = int((flagged & (y == 1)).sum())
    by_pattern = truth.merge(scores[["customer_id", "is_flagged"]], on="customer_id") \
                      .groupby("pattern")["is_flagged"].mean().round(3).to_dict()
    return {
        "n_true_suspicious": k,
        "n_flagged": int(flagged.sum()),
        "precision": round(tp / max(flagged.sum(), 1), 4),
        "recall": round(tp / max(k, 1), 4),
        "precision_at_k": round(float(y[top_k.to_numpy()].mean()), 4),
        "isolation_forest_auc": round(roc_auc_score(y, scores["isolation_score"]), 4),
        "isolation_forest_pr_auc": round(average_precision_score(y, scores["isolation_score"]), 4),
        "combined_auc": round(roc_auc_score(y, scores["alert_score"]), 4),
        "recall_by_pattern": by_pattern,
    }
