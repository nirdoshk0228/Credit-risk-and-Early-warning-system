"""
Probability of Default (12-month) modelling.

Population : performing borrowers at the snapshot (current DPD < 90).
             Accounts already NPA are Stage 3 with PD = 100% by definition.
Target     : default (90+ DPD) within the next 12 months.
Models     : Logistic Regression (benchmark / regulatory-style), Random Forest,
             XGBoost (champion). Champion PDs are Platt-calibrated on a held-out
             validation set so they can be used directly in ECL.
Metrics    : ROC-AUC, Gini, PR-AUC, KS, F1 (threshold tuned on validation),
             Brier score, log-loss, ECE, Hosmer-Lemeshow.
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (MODELS_DIR, NPA_DPD, RISK_BANDS, SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO)
from src.features import MODEL_FEATURES


# ----------------------------------------------------------------------------- metrics
def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def hosmer_lemeshow(y, p, bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["g"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("g").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    stat = (((g["obs"] - g["exp"]) ** 2) / (g["exp"] * (1 - g["exp"] / g["n"])).clip(lower=1e-9)).sum()
    return float(stat), float(1 - chi2.cdf(stat, bins - 2))


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def evaluate(y, p, threshold):
    hl_stat, hl_p = hosmer_lemeshow(y, p)
    pred = (p >= threshold).astype(int)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y, p),
        "ks": ks_statistic(y, p), "f1": f1_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "threshold": threshold, "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "ece": expected_calibration_error(y, p),
        "hl_stat": hl_stat, "hl_pvalue": hl_p, "mean_pd": float(p.mean()), "default_rate": float(y.mean()),
    }


def curves(y, p, n_points=120):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    pick = lambda a: np.unique(np.linspace(0, len(a) - 1, min(n_points, len(a))).astype(int))
    i, j = pick(fpr), pick(prec)
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = 10 - pd.qcut(df["p"].rank(method="first"), 10, labels=False)  # 1 = riskiest
    dec = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"),
                                   mean_pd=("p", "mean"), min_pd=("p", "min"), max_pd=("p", "max"))
    dec["default_rate"] = dec["defaults"] / dec["n"]
    dec["cum_capture"] = dec["defaults"].cumsum() / dec["defaults"].sum()
    dec["lift"] = dec["default_rate"] / y.mean()
    dec["cum_good"] = (dec["n"] - dec["defaults"]).cumsum() / (len(y) - y.sum())
    dec["ks"] = dec["cum_capture"] - dec["cum_good"]
    df["cal_bin"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
    cal = df.groupby("cal_bin").agg(mean_pd=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return {
        "roc": {"fpr": fpr[i].round(4).tolist(), "tpr": tpr[i].round(4).tolist()},
        "pr": {"precision": prec[j].round(4).tolist(), "recall": rec[j].round(4).tolist()},
        "deciles": dec.reset_index().round(4).to_dict(orient="list"),
        "calibration": cal.reset_index(drop=True).round(5).to_dict(orient="list"),
    }


# ----------------------------------------------------------------------------- helpers
def pd_to_score(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    factor = SCORE_PDO / np.log(2)
    offset = SCORE_BASE - factor * np.log(SCORE_BASE_ODDS)
    return np.clip(np.round(offset + factor * np.log((1 - p) / p)), 300, 900).astype(int)


def risk_band(p):
    out = np.empty(len(p), dtype=object)
    for lo, hi, name in RISK_BANDS:
        out[(p >= lo) & (p < hi)] = name
    return out


class PlattCalibrator:
    """Logistic recalibration on the model's log-odds (rank-preserving)."""

    def fit(self, p, y):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e6).fit(z, y)
        return self

    def transform(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


def build_models(y_train, seed):
    spw = 1.0  # no re-weighting: keeps PDs on the true base rate for ECL use
    return {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.3, max_iter=3000)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=400, min_samples_leaf=15,
                                                              max_features=0.4, n_jobs=-1,
                                                              random_state=seed)),
        "XGBoost": XGBClassifier(n_estimators=2000, learning_rate=0.02, max_depth=4, min_child_weight=10,
                                 subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, gamma=1.0,
                                 scale_pos_weight=spw, eval_metric="logloss", early_stopping_rounds=200,
                                 random_state=seed, n_jobs=-1),
    }


# ----------------------------------------------------------------------------- train
def train_pd_models(feat: pd.DataFrame, outcomes: pd.DataFrame, seed: int = 42):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = feat.merge(outcomes[["customer_id", "default_12m"]], on="customer_id") \
             .sort_values("customer_id").reset_index(drop=True)
    perf = df[df["current_dpd"] < NPA_DPD].reset_index(drop=True)
    X, y = perf[MODEL_FEATURES], perf["default_12m"].to_numpy()

    idx = np.arange(len(perf))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)

    models = build_models(y[tr], seed)
    results, curve_data, raw_preds = {}, {}, {}
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        else:
            model.fit(X.iloc[tr], y[tr])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        p_te = model.predict_proba(X.iloc[te])[:, 1]
        thr = best_f1_threshold(y[va], p_va)
        results[name] = {"validation": evaluate(y[va], p_va, thr), "test": evaluate(y[te], p_te, thr)}
        curve_data[name] = curves(y[te], p_te)
        raw_preds[name] = model.predict_proba(X)[:, 1]
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # champion: XGBoost. Decide on Platt recalibration with a 2-fold cross-fit on the
    # validation set (fit on one half, score the other) so the decision is out-of-sample.
    champ = models["XGBoost"]
    p_va_raw = champ.predict_proba(X.iloc[va])[:, 1]
    halves = np.array_split(np.random.default_rng(seed).permutation(len(va)), 2)
    ece_raw, ece_cal = [], []
    for a, b in [(halves[0], halves[1]), (halves[1], halves[0])]:
        c = PlattCalibrator().fit(p_va_raw[a], y[va][a])
        ece_raw.append(expected_calibration_error(y[va][b], p_va_raw[b]))
        ece_cal.append(expected_calibration_error(y[va][b], c.transform(p_va_raw[b])))
    use_cal = np.mean(ece_cal) < np.mean(ece_raw)
    cal = PlattCalibrator().fit(p_va_raw, y[va])
    joblib.dump({"calibrator": cal, "applied": bool(use_cal)}, MODELS_DIR / "xgboost_calibrator.joblib")
    p_cal_va = cal.transform(p_va_raw)
    p_cal_te = cal.transform(champ.predict_proba(X.iloc[te])[:, 1])
    thr = best_f1_threshold(y[va], p_cal_va)
    results["XGBoost (Platt)"] = {"validation": evaluate(y[va], p_cal_va, thr),
                                  "test": evaluate(y[te], p_cal_te, thr)}
    curve_data["XGBoost (Platt)"] = curves(y[te], p_cal_te)
    pd_champion = cal.transform(raw_preds["XGBoost"]) if use_cal else raw_preds["XGBoost"]
    calibration_decision = {"cv_ece_raw": float(np.mean(ece_raw)), "cv_ece_platt": float(np.mean(ece_cal)),
                            "platt_applied": bool(use_cal)}

    split = np.empty(len(perf), dtype=object)
    split[tr], split[va], split[te] = "train", "validation", "test"
    scores = pd.DataFrame({
        "customer_id": perf["customer_id"], "split": split,
        "pd_logistic": raw_preds["Logistic Regression"], "pd_random_forest": raw_preds["Random Forest"],
        "pd_xgboost_raw": raw_preds["XGBoost"], "pd_12m": pd_champion,
    })
    npa = df[df["current_dpd"] >= NPA_DPD]
    scores = pd.concat([scores, pd.DataFrame({
        "customer_id": npa["customer_id"], "split": "stage3_npa", "pd_logistic": 1.0,
        "pd_random_forest": 1.0, "pd_xgboost_raw": 1.0, "pd_12m": 1.0})], ignore_index=True)
    scores["risk_score"] = pd_to_score(scores["pd_12m"].to_numpy())
    scores["risk_band"] = np.where(scores["split"] == "stage3_npa", "F - Default",
                                   risk_band(scores["pd_12m"].to_numpy()))

    report = {
        "population": {"n_performing": int(len(perf)), "n_npa_excluded": int(len(npa)),
                       "train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
                       "default_rate": float(y.mean()),
                       "xgb_best_iteration": int(champ.best_iteration)},
        "champion": "XGBoost", "calibration": calibration_decision,
        "features": MODEL_FEATURES,
        "metrics": results,
        "curves": curve_data,
    }
    (MODELS_DIR.parent / "model_report.json").write_text(json.dumps(report, indent=1, default=float))
    metrics_rows = [{"model": m, "dataset": ds, **{k: round(float(v), 5) for k, v in vals.items()}}
                    for m, r in results.items() for ds, vals in r.items()]
    test_idx = perf.index[te]
    return scores, pd.DataFrame(metrics_rows), champ, perf, test_idx
