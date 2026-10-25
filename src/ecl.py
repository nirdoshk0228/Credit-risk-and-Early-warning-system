"""
Expected Credit Loss (IFRS 9 / Ind AS 109 style): ECL = PD x LGD x EAD.

Stage 1 : performing                      -> 12-month ECL
Stage 2 : significant increase in credit  -> lifetime ECL
          risk (DPD > 30, EWS Red, or PD above SICR threshold)
Stage 3 : credit-impaired (DPD >= 90 or   -> PD = 100%, ECL = LGD x EAD
          cross-default on the borrower)
"""
import numpy as np
import pandas as pd

from config import (CARD_BEHAVIOURAL_LIFE_M, CCF_REVOLVING, DISCOUNT_RATE, LGD_PARAMS, NPA_DPD,
                    SICR_DPD, SICR_PD_THRESHOLD)
from src.db import read_sql

LOAN_BASE_SQL = """
SELECT l.*, COALESCE(p.dpd, 0) AS loan_dpd
FROM loans l
LEFT JOIN payments p ON p.loan_id = l.loan_id AND p.month_idx = 1
ORDER BY l.loan_id
"""


def load_loan_base(engine, customer_risk: pd.DataFrame) -> pd.DataFrame:
    loans = read_sql(LOAN_BASE_SQL, engine)
    cols = ["customer_id", "pd_12m", "ews_level", "current_dpd"]
    return loans.merge(customer_risk[cols], on="customer_id", how="left")


def ead_lgd(df: pd.DataFrame, collateral_shock: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    card = df["product"] == "Credit Card"
    undrawn = (df["credit_limit"].fillna(0) - df["outstanding"]).clip(lower=0)
    accrued = df["outstanding"] * df["interest_rate"] / 1200          # one month accrued interest
    df["ead"] = np.where(card, df["outstanding"] + CCF_REVOLVING * undrawn, df["outstanding"] + accrued)

    shock = 1.0
    if collateral_shock:
        shock = df["product"].map(collateral_shock).fillna(0.0).to_numpy() + 1.0
    collateral = df["collateral_value"] * shock
    haircut = df["product"].map(lambda p: LGD_PARAMS[p]["haircut"])
    floor = df["product"].map(lambda p: LGD_PARAMS[p]["floor"])
    unsec = df["product"].map(lambda p: LGD_PARAMS[p]["unsecured_lgd"])
    recovery = collateral * (1 - haircut)
    lgd_secured = (1 - recovery / df["ead"].clip(lower=1)).clip(lower=floor, upper=1.0)
    df["lgd"] = np.where(df["secured"] == 1, lgd_secured, unsec)
    df["collateral_stressed"] = collateral
    return df


def assign_stage(df: pd.DataFrame, pd_col: str = "pd_12m") -> np.ndarray:
    s3 = (df["loan_dpd"] >= NPA_DPD) | (df["current_dpd"] >= NPA_DPD)
    s2 = (df["loan_dpd"] > SICR_DPD) | (df["ews_level"] == "Red") | (df[pd_col] >= SICR_PD_THRESHOLD)
    return np.select([s3, s2], [3, 2], 1)


def compute_ecl(df: pd.DataFrame, pd_col: str = "pd_12m") -> pd.DataFrame:
    df = df.copy()
    df["stage"] = assign_stage(df, pd_col)
    pd12 = df[pd_col].clip(0, 1).to_numpy()
    card = (df["product"] == "Credit Card").to_numpy()
    remaining_m = np.where(card, CARD_BEHAVIOURAL_LIFE_M,
                           (df["tenure_months"] - df["months_on_book"]).clip(lower=1))
    T = remaining_m / 12.0
    ead, lgd = df["ead"].to_numpy(), df["lgd"].to_numpy()

    # lifetime ECL: yearly marginal PDs on an amortising EAD, discounted at EIR proxy
    years = np.arange(1, 21)[None, :]
    frac = np.clip(T[:, None] - (years - 1), 0, 1)                # portion of each year still alive
    surv_prev = (1 - pd12[:, None]) ** (years - 1)
    marginal = surv_prev * (1 - (1 - pd12[:, None]) ** frac)
    amort = np.where(card[:, None], 1.0, np.clip(1 - (years - 1) / np.maximum(T[:, None], 1e-9), 0, 1))
    df_y = 1 / (1 + DISCOUNT_RATE) ** (years - 0.5)
    lifetime_ecl = (marginal * amort * df_y).sum(axis=1) * ead * lgd
    lifetime_pd = 1 - (1 - pd12) ** T

    horizon_pd = 1 - (1 - pd12) ** np.minimum(T, 1.0)
    ecl_12m = horizon_pd * lgd * ead / (1 + DISCOUNT_RATE) ** 0.5
    stage = df["stage"].to_numpy()
    df["lifetime_pd"] = np.where(stage == 3, 1.0, lifetime_pd)
    df["ecl_12m"] = ecl_12m
    df["ecl_lifetime"] = lifetime_ecl
    df["ecl"] = np.select([stage == 3, stage == 2], [lgd * ead, lifetime_ecl], ecl_12m)
    df["coverage"] = df["ecl"] / df["ead"].clip(lower=1)
    return df


def severity(pd12, ews_level, max_stage, flagged):
    crit = (max_stage == 3) | (pd12 >= 0.30) | ((ews_level == "Red") & (pd12 >= 0.15))
    high = (pd12 >= 0.10) | (ews_level == "Red") | (max_stage == 2)
    med = (pd12 >= 0.03) | (ews_level == "Amber") | (flagged == 1)
    return np.select([crit, high, med], ["Critical", "High", "Medium"], "Low")


def ecl_results_table(loan_ecl: pd.DataFrame) -> pd.DataFrame:
    keep = ["loan_id", "customer_id", "product", "secured", "outstanding", "credit_limit", "collateral_value",
            "interest_rate", "tenure_months", "months_on_book", "loan_dpd", "pd_12m", "lifetime_pd",
            "lgd", "ead", "stage", "ecl_12m", "ecl_lifetime", "ecl", "coverage"]
    return loan_ecl[keep].round(6)
