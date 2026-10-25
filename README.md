# Credit Risk & Early Warning System

End-to-end banking risk analytics platform for a retail lending book: **Early Warning Signals (EWS)**, **Probability of Default (PD) modelling**, **transaction anomaly detection**, **SHAP-based reason codes**, **IFRS 9 / Ind AS 109 Expected Credit Loss**, and **macroeconomic stress testing** — served through a SQL-backed Streamlit dashboard.

`Python` · `SQL` · `PostgreSQL` · `XGBoost` · `scikit-learn` · `SHAP` · `Streamlit` · `Plotly`

![Portfolio overview](docs/screenshots/portfolio.png)

---

## Headline results

Synthetic Indian retail portfolio: **15,000 borrowers · 24,864 facilities · 266k repayment records · 1.24M account transactions** (seed 42).

| Area | Result |
|---|---|
| **PD model (test set)** | XGBoost ROC-AUC **0.887**, Gini **0.77**, KS **0.63**, PR-AUC **0.62**, Brier **0.039**, ECE **0.008** |
| Benchmarks | Logistic Regression AUC 0.882 · Random Forest AUC 0.885 (RF fails Hosmer-Lemeshow, p < 0.001) |
| Rank ordering | Riskiest score decile captures **67%** of defaults at **6.7× lift** |
| **Early warning** | 12-month default rate **2.1%** Green → **9.7%** Amber → **43.5%** Red |
| **Anomaly detection** | 279 alerts · precision **65%** · recall **81%** · Isolation Forest AUC **0.99** vs injected AML typologies |
| **ECL** | EAD ₹2,185 cr · ECL ₹63.5 cr (**2.9%** coverage) · GNPA **4.6%** · Stage 2 **13.2%** |
| **Stress testing** | ECL **+30%** under Adverse, **+89%** under Severe scenario |

All figures are reproducible with `python run_pipeline.py` (identical on PostgreSQL and SQLite up to floating-point summation order).

---

## Architecture

```mermaid
flowchart LR
    G[Synthetic data generator<br/>customers · loans · payments<br/>transactions · macro] --> DB[(PostgreSQL<br/>core banking schema)]
    DB -->|in-database SQL<br/>aggregations| F[Feature store<br/>42 features]
    F --> EWS[EWS rule engine<br/>13 signals · RBI SMA]
    F --> PD[PD models<br/>LR · RF · XGBoost]
    DB --> AN[Anomaly detection<br/>Isolation Forest + AML rules]
    PD --> SH[SHAP explainer<br/>reason codes]
    PD --> ECL[ECL engine<br/>staging · PD×LGD×EAD]
    EWS --> ECL
    ECL --> ST[Stress testing<br/>macro satellite model]
    EWS & SH & ECL & AN & ST --> OUT[(Output tables<br/>+ SQL views)]
    OUT --> UI[Streamlit dashboard]
```

## What each component does

### 1. Data (`src/data_generator.py`, `sql/schema.sql`)
A realistic, internally consistent book: 6 products (home, auto, personal, credit card, gold, business loans), amortisation schedules, collateral (LTV, depreciation), risk-based pricing, and bank-account transactions (salary, UPI/POS spends, ATM cash, NACH EMI debits). Each borrower has a latent risk factor; a subset **deteriorates** over time, which surfaces gradually as falling inflows, more cash usage, NACH bounces, rising DPD and a falling bureau score — exactly what an EWS should catch *before* the account turns NPA. Default outcomes include **non-linear effects** (bureau-score cliff, over-leverage threshold, revolving stress, income volatility × shock) and **unforeseeable shocks**, so no model can be perfect. Labels live in separate tables and are used only for training/evaluation.

### 2. Feature engineering (`sql/features.sql`, `src/features.py`)
Aggregations run **inside the database** with portable SQL (window logic via `month_idx`, no dialect-specific date maths): loan/exposure features (DTI, unsecured share, LTV, card utilisation), repayment features (max DPD 3/6/12m, DPD trend, missed/partial payments, payment ratio, days late), and account-conduct features (inflow trend, salary regularity, cash-withdrawal share, bounces, debit/credit ratio).

### 3. Early Warning Signals (`src/ews.py`)
13 weighted signals across *Repayment*, *Account conduct*, *Leverage* and *Bureau* categories, plus **RBI SMA classification** (SMA-0/1/2, NPA). Weighted score → Green / Amber / Red watch-list.

### 4. PD modelling (`src/models.py`)
- Population: performing borrowers (DPD < 90); target: 90+ DPD within 12 months.
- Stratified 60/20/20 train / validation / test; XGBoost early-stopped on validation.
- **Logistic Regression** (regulatory-style benchmark), **Random Forest**, **XGBoost** (champion).
- Metrics: ROC-AUC, Gini, **PR-AUC**, **KS**, **F1** (threshold tuned on validation only), precision/recall, **Brier**, log-loss, **ECE**, **Hosmer-Lemeshow**, decile lift/gains.
- **Calibration**: Platt recalibration is applied only if a 2-fold cross-fit on the validation set shows it reduces ECE out-of-sample (with no class re-weighting, XGBoost was already well calibrated, so it was *not* applied — an evidence-based decision rather than a reflex).
- PDs map to a points-based **risk score** (600 at 50:1 odds, 20 PDO) and **risk bands A–F**.

### 5. Explainable risk engine (`src/explain.py`)
SHAP TreeExplainer on the champion. Every account gets its top risk drivers (log-odds contributions) translated into **adverse-action style reason codes**, e.g. `RC22 Bureau score changed −122 points in 6 months`, `RC18 Debt-to-income ratio of 66%`. Global importance = mean |SHAP|.

### 6. Expected Credit Loss (`src/ecl.py`)
- **Staging**: Stage 3 = 90+ DPD or borrower cross-default; Stage 2 (SICR) = 30+ DPD, EWS Red, or PD ≥ 20%; else Stage 1.
- **EAD**: outstanding + accrued interest; revolving = drawn + 75% CCF × undrawn.
- **LGD**: collateral-based with product haircuts and floors (home 25% haircut / 10% floor, auto 40% / 20%, gold 15% / 5%); unsecured 65–75%.
- **ECL = PD × LGD × EAD**: 12-month for Stage 1, lifetime (yearly marginal PDs on an amortising EAD, discounted) for Stage 2, LGD × EAD for Stage 3.
- **Risk severity** segmentation (Low / Medium / High / Critical) combines PD, EWS level, stage and AML flags.

### 7. Stress testing (`src/stress.py`)
A **macro satellite model** — logit(portfolio default rate) on GDP growth, unemployment, repo rate and house-price growth (R² 0.89) — converts scenario shocks into PD shifts on the log-odds scale, scaled by product sensitivity. Collateral is haircut (property, vehicles, gold), LGD re-derived, accounts crossing the SICR threshold migrate to Stage 2, and ECL is recomputed. Baseline / Adverse / Severe are pre-computed; the dashboard has sliders for **custom scenarios** computed live.

### 8. Transaction anomaly detection (`src/anomaly.py`)
Unsupervised **Isolation Forest** on 13 behavioural features + six **AML typology rules** (structuring below the ₹50k cash threshold, rapid pass-through, odd-hour high-value wires, dormant-account reactivation, income mismatch, round-figure transactions). Alerts require a strong typology hit, a moderate hit corroborated by the model, or an extreme outlier. Transaction-level alerts explain *which* transactions triggered. Legitimate look-alikes (FD maturities, cash-heavy businesses, overseas education remittances) are deliberately present so precision is realistic.

### 9. Dashboard (`app.py`, `sql/views.sql`)
Every panel reads from SQL views/tables: portfolio KPIs, exposure by product × stage, concentration (state / employment / tier, HHI, ticket size), PD distribution, risk bands, severity × product heatmap, SMA mix, EWS signal frequency and watch-list, **account drill-down** (SHAP drivers, reason codes, DPD history, cash-flows, facility-level ECL), AML alert queue, stress scenarios, and model diagnostics (ROC, PR, calibration, gains/KS, SHAP importance).

| Account drill-down | Model performance |
|---|---|
| ![](docs/screenshots/account_drilldown.png) | ![](docs/screenshots/model_performance.png) |

| Stress testing | Anomaly detection |
|---|---|
| ![](docs/screenshots/stress_testing.png) | ![](docs/screenshots/anomaly_detection.png) |

---

## Run it

```bash
pip install -r requirements.txt

# Option A — zero setup (SQLite file)
python run_pipeline.py
streamlit run app.py

# Option B — PostgreSQL
docker compose up -d
export DATABASE_URL=postgresql+psycopg2://risk:risk@localhost:5432/credit_risk   # Windows: set DATABASE_URL=...
python run_pipeline.py
streamlit run app.py

# Tests
pytest -q
```

Options: `python run_pipeline.py --n-customers 30000 --seed 7 --db-url <url>`. The pipeline takes ~1–2 minutes.

## Project structure

```
├── app.py                  # Streamlit dashboard (SQL-backed)
├── run_pipeline.py         # End-to-end orchestration
├── config.py               # Policy parameters: SICR, LGD haircuts, CCF, risk bands, scorecard scaling
├── sql/
│   ├── schema.sql          # Core banking schema + indexes
│   ├── features.sql        # In-database feature engineering
│   └── views.sql           # Analytics layer for the dashboard
├── src/
│   ├── data_generator.py   # Synthetic portfolio, transactions, macro history, AML typologies
│   ├── db.py               # PostgreSQL/SQLite engine, COPY bulk loads
│   ├── features.py         # Feature store
│   ├── ews.py              # Early Warning Signals + RBI SMA
│   ├── models.py           # PD models, metrics, calibration, scorecard
│   ├── explain.py          # SHAP risk drivers and reason codes
│   ├── ecl.py              # Staging, EAD, LGD, 12m/lifetime ECL, severity
│   ├── stress.py           # Satellite model and scenarios
│   └── anomaly.py          # Isolation Forest + AML rules
├── tests/test_core.py      # Unit tests for ECL, staging, CCF, KS/ECE, scorecard, stress
└── docs/                   # Screenshots and interview notes
```

## Limitations

- Data is synthetic; absolute metric levels depend on the generator. The pipeline, methodology and code paths are what carry over to real data.
- Single snapshot with a random split — a production model would add an **out-of-time** validation window and PSI/CSI stability monitoring.
- Lifetime PD assumes a constant hazard; a production IFRS 9 model would use PD term structures and forward-looking scenario weighting.
- The macro series are stylised approximations of India 2015–2025.
