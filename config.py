"""Central configuration for the Credit Risk & Early Warning System."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# PostgreSQL is the primary target. Falls back to a local SQLite file so the
# project runs with zero setup. Example:
#   DATABASE_URL=postgresql+psycopg2://risk:risk@localhost:5432/credit_risk
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'credit_risk.db'}")

SEED = int(os.getenv("SEED", "42"))
N_CUSTOMERS = int(os.getenv("N_CUSTOMERS", "15000"))
SNAPSHOT_DATE = "2025-12-31"          # observation point; target = default in next 12m
PAYMENT_HISTORY_MONTHS = 12
TXN_HISTORY_MONTHS = 6

# ---- Credit policy parameters -------------------------------------------
NPA_DPD = 90                  # Stage 3 / default definition (RBI 90+ DPD)
SICR_DPD = 30                 # Stage 2 backstop
SICR_PD_THRESHOLD = 0.20      # absolute 12m PD threshold for SICR
CCF_REVOLVING = 0.75          # credit conversion factor on undrawn card limits
DISCOUNT_RATE = 0.10          # EIR proxy for discounting lifetime losses
CARD_BEHAVIOURAL_LIFE_M = 36  # behavioural life for revolving products

# Collateral haircut (forced-sale discount + recovery costs) and LGD floor
LGD_PARAMS = {
    "Home Loan":      {"haircut": 0.25, "floor": 0.10, "unsecured_lgd": 0.45},
    "Auto Loan":      {"haircut": 0.40, "floor": 0.20, "unsecured_lgd": 0.60},
    "Gold Loan":      {"haircut": 0.15, "floor": 0.05, "unsecured_lgd": 0.40},
    "Business Loan":  {"haircut": 0.40, "floor": 0.25, "unsecured_lgd": 0.70},
    "Personal Loan":  {"haircut": 0.00, "floor": 0.65, "unsecured_lgd": 0.65},
    "Credit Card":    {"haircut": 0.00, "floor": 0.75, "unsecured_lgd": 0.75},
}

# Product sensitivity of PD to macro shocks (multiplier on satellite-model shift)
MACRO_SENSITIVITY = {
    "Personal Loan": 1.30, "Credit Card": 1.40, "Business Loan": 1.50,
    "Auto Loan": 1.10, "Home Loan": 0.80, "Gold Loan": 0.70,
}

# PD risk bands (12-month PD)
RISK_BANDS = [
    (0.00, 0.02, "A - Low"),
    (0.02, 0.05, "B - Moderate"),
    (0.05, 0.15, "C - Elevated"),
    (0.15, 0.30, "D - High"),
    (0.30, 1.01, "E - Very High"),
]

# Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double odds
SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO = 600, 50, 20
