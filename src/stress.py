"""
Macroeconomic stress testing.

1. Satellite model: logit(portfolio default rate) regressed on GDP growth,
   unemployment, repo rate and house-price growth (quarterly history).
2. Scenario shocks shift every account's PD on the log-odds scale:
       logit(PD_s) = logit(PD) + sensitivity_product * sum_k beta_k * shock_k
3. Collateral values are haircut (property, vehicles, gold), LGD is re-derived,
   accounts crossing the SICR threshold migrate to Stage 2, and ECL is recomputed.
"""
import numpy as np
import pandas as pd

from config import MACRO_SENSITIVITY
from src.ecl import compute_ecl, ead_lgd

MACRO_VARS = ["gdp_growth", "unemployment_rate", "repo_rate", "hpi_growth"]

SCENARIOS = {
    "Baseline": {"gdp_growth": 0.0, "unemployment_rate": 0.0, "repo_rate": 0.0, "hpi_growth": 0.0,
                 "property": 0.00, "vehicle": 0.00, "gold": 0.00,
                 "narrative": "Current macro conditions persist."},
    "Adverse": {"gdp_growth": -3.5, "unemployment_rate": 2.0, "repo_rate": 1.0, "hpi_growth": -8.0,
                "property": -0.10, "vehicle": -0.10, "gold": -0.05,
                "narrative": "Growth slowdown with sticky inflation; RBI hikes 100bp, job losses in services."},
    "Severe": {"gdp_growth": -7.0, "unemployment_rate": 4.5, "repo_rate": 2.0, "hpi_growth": -18.0,
               "property": -0.25, "vehicle": -0.20, "gold": -0.15,
               "narrative": "Deep recession (COVID-scale shock): unemployment spike, 200bp tightening, "
                            "real-estate correction."},
}


def fit_satellite(macro: pd.DataFrame) -> dict:
    y = macro["portfolio_default_rate"].clip(1e-5, 1 - 1e-5)
    z = np.log(y / (1 - y)).to_numpy()
    X = np.column_stack([np.ones(len(macro)), macro[MACRO_VARS].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ beta
    r2 = 1 - resid.var() / z.var()
    out = {"intercept": float(beta[0]), "r_squared": float(r2)}
    out.update({v: float(b) for v, b in zip(MACRO_VARS, beta[1:])})
    return out


def collateral_shocks(sc: dict) -> dict:
    return {"Home Loan": sc["property"], "Business Loan": sc["property"] * 0.8,
            "Auto Loan": sc["vehicle"], "Gold Loan": sc["gold"]}


def apply_scenario(base: pd.DataFrame, coefs: dict, sc: dict) -> pd.DataFrame:
    """base = loan-level frame with pd_12m, ews_level, current_dpd, loan_dpd and loan terms."""
    shift = sum(coefs[v] * sc.get(v, 0.0) for v in MACRO_VARS)
    sens = base["product"].map(MACRO_SENSITIVITY).fillna(1.0).to_numpy()
    p = base["pd_12m"].clip(1e-6, 1 - 1e-6).to_numpy()
    logit = np.log(p / (1 - p)) + sens * shift
    p_s = np.where(base["pd_12m"] >= 1.0, 1.0, 1 / (1 + np.exp(-logit)))
    df = base.copy()
    df["pd_base"] = base["pd_12m"]
    df["pd_12m"] = p_s
    df = ead_lgd(df, collateral_shocks(sc))
    return compute_ecl(df, "pd_12m")


def summarise(df: pd.DataFrame, scenario: str, base_ecl: pd.Series) -> pd.DataFrame:
    df = df.assign(ecl_base=base_ecl.to_numpy())
    g = df.groupby("product").apply(lambda x: pd.Series({
        "ead": x["ead"].sum(),
        "pd_base": np.average(x["pd_base"], weights=x["ead"]),
        "pd_stressed": np.average(x["pd_12m"], weights=x["ead"]),
        "lgd_stressed": np.average(x["lgd"], weights=x["ead"]),
        "ecl_base": x["ecl_base"].sum(), "ecl_stressed": x["ecl"].sum(),
        "stage2_ead_share": x.loc[x["stage"] == 2, "ead"].sum() / x["ead"].sum(),
        "stage3_ead_share": x.loc[x["stage"] == 3, "ead"].sum() / x["ead"].sum(),
    }), include_groups=False).reset_index()
    g.insert(0, "scenario", scenario)
    g["ecl_increase"] = g["ecl_stressed"] - g["ecl_base"]
    return g


def run_stress_tests(base_loans: pd.DataFrame, base_ecl: pd.Series, coefs: dict) -> pd.DataFrame:
    out = [summarise(apply_scenario(base_loans, coefs, sc), name, base_ecl) for name, sc in SCENARIOS.items()]
    return pd.concat(out, ignore_index=True)


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame([{"scenario": k, **v} for k, v in SCENARIOS.items()])
