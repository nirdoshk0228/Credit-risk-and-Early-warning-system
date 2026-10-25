"""
End-to-end pipeline:
  generate data -> load DB -> SQL features -> EWS -> anomaly detection ->
  PD models -> SHAP reason codes -> ECL & staging -> stress tests -> SQL views

Usage:
  python run_pipeline.py                         # uses DATABASE_URL or local SQLite
  python run_pipeline.py --db-url postgresql+psycopg2://risk:risk@localhost:5432/credit_risk
  python run_pipeline.py --n-customers 20000 --seed 7
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import config
from src.anomaly import run_anomaly
from src.data_generator import PortfolioGenerator
from src.db import create_views, drop_views, get_engine, read_sql, run_sql_file, write_df
from src.ecl import compute_ecl, ead_lgd, ecl_results_table, load_loan_base, severity
from src.ews import run_ews
from src.explain import explain
from src.features import MODEL_FEATURES, build_features
from src.models import train_pd_models
from src.stress import fit_satellite, run_stress_tests, scenario_table

RAW_TABLES = ["customers", "loans", "payments", "transactions", "macro_history",
              "default_outcomes", "aml_ground_truth"]


class Step:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"-> {self.name} ...", flush=True)

    def __exit__(self, *a):
        print(f"   done in {time.time() - self.t:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=config.DATABASE_URL)
    ap.add_argument("--n-customers", type=int, default=config.N_CUSTOMERS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    engine = get_engine(args.db_url)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Database: {engine.dialect.name}  |  customers: {args.n_customers}  |  seed: {args.seed}")

    with Step("Generating synthetic portfolio"):
        data = PortfolioGenerator(args.n_customers, args.seed,
                                  config.PAYMENT_HISTORY_MONTHS, config.TXN_HISTORY_MONTHS).generate()
        for k, v in data.items():
            print(f"   {k:<18} {len(v):>9,} rows")

    with Step("Creating schema and loading raw tables"):
        drop_views(engine)
        run_sql_file(engine, "schema.sql")
        for t in RAW_TABLES:
            write_df(data[t], t, engine, if_exists="append")

    with Step("Engineering features in SQL"):
        feat = build_features(engine)
        write_df(feat[["customer_id"] + MODEL_FEATURES + ["monthly_income", "employment_type", "state"]],
                 "customer_features", engine)

    with Step("Running Early Warning Signals"):
        ews_summary, ews_signals = run_ews(feat)
        write_df(ews_summary, "ews_summary", engine)
        write_df(ews_signals, "ews_signals", engine)
        print("   EWS levels:", ews_summary["ews_level"].value_counts().to_dict())

    with Step("Detecting transaction anomalies"):
        anom, rule_hits, txn_alerts, anom_metrics = run_anomaly(feat, engine, args.seed)
        write_df(anom, "anomaly_scores", engine)
        write_df(rule_hits, "aml_rule_hits", engine)
        write_df(txn_alerts, "txn_alerts", engine)
        print(f"   flagged {anom_metrics['n_flagged']} accounts | precision {anom_metrics['precision']:.2f}"
              f" | recall {anom_metrics['recall']:.2f} | IF AUC {anom_metrics['isolation_forest_auc']:.3f}")

    with Step("Training PD models (LR, RF, XGBoost)"):
        outcomes = read_sql("SELECT * FROM default_outcomes", engine)
        scores, metrics, champ, perf, _ = train_pd_models(feat, outcomes, args.seed)
        test = metrics[metrics["dataset"] == "test"].set_index("model")
        print(test[["roc_auc", "gini", "pr_auc", "ks", "f1", "brier", "ece"]].round(3).to_string())

    with Step("Explaining risk with SHAP"):
        drivers, importance, log_odds = explain(champ, perf)
        scores = scores.merge(log_odds, on="customer_id", how="left")
        write_df(scores, "model_scores", engine)
        write_df(metrics, "model_metrics", engine)
        write_df(drivers, "risk_drivers", engine)
        write_df(importance, "shap_importance", engine)

    with Step("Computing ECL, staging and risk severity"):
        cust = feat[["customer_id", "current_dpd", "monthly_income", "employment_type", "city_tier",
                     "state", "bureau_score"]] \
            .merge(scores[["customer_id", "pd_12m", "risk_score", "risk_band"]], on="customer_id") \
            .merge(ews_summary, on="customer_id") \
            .merge(anom[["customer_id", "alert_score", "is_flagged", "alert_priority"]], on="customer_id")
        base = load_loan_base(engine, cust)
        loan_ecl = compute_ecl(ead_lgd(base))
        write_df(ecl_results_table(loan_ecl), "ecl_results", engine)

        agg = loan_ecl.groupby("customer_id").agg(total_ead=("ead", "sum"), total_ecl=("ecl", "sum"),
                                                  max_stage=("stage", "max"), n_loans=("loan_id", "size"))
        cust = cust.merge(agg, on="customer_id", how="left")
        top = drivers[drivers["driver_rank"] == 1][["customer_id", "reason_text"]] \
            .rename(columns={"reason_text": "top_risk_driver"})
        cust = cust.merge(top, on="customer_id", how="left")
        cust["top_risk_driver"] = cust["top_risk_driver"].fillna("Account already NPA (90+ DPD)")
        cust["risk_severity"] = severity(cust["pd_12m"], cust["ews_level"], cust["max_stage"], cust["is_flagged"])
        write_df(cust, "customer_risk", engine)
        print("   severity:", cust["risk_severity"].value_counts().to_dict())

    with Step("Fitting macro satellite model and stress testing"):
        macro = read_sql("SELECT * FROM macro_history ORDER BY quarter", engine)
        coefs = fit_satellite(macro)
        write_df(pd.DataFrame([coefs]), "macro_coefficients", engine)
        write_df(scenario_table(), "stress_scenarios", engine)
        stress = run_stress_tests(base, loan_ecl["ecl"], coefs)
        write_df(stress, "stress_results", engine)
        tot = stress.groupby("scenario")[["ecl_stressed", "ead"]].sum()
        print("   ECL by scenario (Rs cr):", (tot["ecl_stressed"] / 1e7).round(2).to_dict())

    with Step("Creating analytics views"):
        create_views(engine)

    total_ead = loan_ecl["ead"].sum()
    summary = {
        "database": engine.dialect.name,
        "customers": int(len(cust)), "loans": int(len(loan_ecl)),
        "transactions": int(len(data["transactions"])), "payments": int(len(data["payments"])),
        "total_ead_cr": round(total_ead / 1e7, 2),
        "total_ecl_cr": round(loan_ecl["ecl"].sum() / 1e7, 2),
        "coverage_pct": round(100 * loan_ecl["ecl"].sum() / total_ead, 2),
        "gnpa_pct": round(100 * loan_ecl.loc[loan_ecl["stage"] == 3, "ead"].sum() / total_ead, 2),
        "stage_mix_ead_pct": (100 * loan_ecl.groupby("stage")["ead"].sum() / total_ead).round(2).to_dict(),
        "pd_test_metrics": test.round(4).to_dict(orient="index"),
        "anomaly_metrics": anom_metrics,
        "satellite_model": coefs,
        "stress_ecl_cr": (tot["ecl_stressed"] / 1e7).round(2).to_dict(),
        "ews_red_default_rate": float(cust.merge(outcomes, on="customer_id")
                                      .query("current_dpd < 90").groupby("ews_level")["default_12m"]
                                      .mean().round(4).to_dict().get("Red", np.nan)),
        "ews_default_rate_by_level": cust.merge(outcomes, on="customer_id").query("current_dpd < 90")
                                         .groupby("ews_level")["default_12m"].mean().round(4).to_dict(),
    }
    (config.ARTIFACTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nPipeline complete. Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
