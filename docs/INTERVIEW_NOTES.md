# Interview notes — Credit Risk & Early Warning System

## 30-second pitch
"I built an end-to-end credit risk platform for a retail lending book. Data sits in PostgreSQL and features are engineered in SQL. An early-warning rule engine flags deteriorating borrowers, and three PD models — logistic regression, random forest and XGBoost — estimate 12-month default risk. SHAP turns each prediction into reason codes. PDs feed an IFRS 9-style ECL engine, which a macro satellite model stresses under adverse and severe scenarios. Isolation Forest plus AML typology rules catch suspicious transactions. Everything surfaces in a SQL-backed Streamlit dashboard."

## Numbers to remember
- XGBoost: AUC 0.887 · Gini 0.77 · KS 0.63 · PR-AUC 0.62 (default rate 6.8%, so PR-AUC is ~9× the random baseline).
- Top decile captures 67% of defaults (6.7× lift).
- EWS default rates: Green 2.1% → Amber 9.7% → Red 43.5%.
- ECL ₹63.5 cr on ₹2,185 cr EAD (2.9% coverage); GNPA 4.6%; Stage 2 13.2%.
- Stress: Adverse +30% ECL, Severe +89%.
- AML: 65% precision, 81% recall; Isolation Forest AUC 0.99.

## Likely questions

**Why did XGBoost only beat logistic regression by ~0.5 AUC?**
Most of the signal is monotonic (DPD, bureau score, DTI), and logistic regression captures that well. XGBoost adds value through threshold and interaction effects: the bureau-score cliff, over-leverage beyond ~55% DTI, and stress hitting volatile-income borrowers harder. In practice many banks keep a logistic scorecard for regulatory transparency and use GBMs as challengers. That's why SHAP matters: it makes the GBM explainable enough to use.

**Why not use class weights / SMOTE for the imbalance?**
Re-weighting distorts probabilities, and ECL needs calibrated PDs, not just rankings. I trained on the true base rate, tuned the F1 threshold on validation only, and judged imbalance-sensitive quality with PR-AUC and KS.

**How did you handle calibration?**
I measured ECE and Hosmer-Lemeshow. I tested Platt recalibration with a 2-fold cross-fit on the validation set, so the decision itself was made out-of-sample. Raw XGBoost was already well calibrated (ECE 0.008), so I didn't apply it. Random Forest fails Hosmer-Lemeshow (p < 0.001), which is typical because averaging trees pushes probabilities toward the middle.

**What's KS and why do banks use it?**
KS is the maximum gap between the cumulative score distributions of defaulters and non-defaulters. It measures separation at the best cut-off and is a traditional scorecard acceptance metric. Gini = 2·AUC − 1 is the other standard one.

**Walk me through ECL.**
ECL = PD × LGD × EAD.
- Staging: Stage 3 is 90+ DPD, with cross-default at borrower level. Stage 2 is a significant increase in credit risk: 30+ DPD backstop, EWS Red, or PD ≥ 20%.
- Horizon: Stage 1 uses 12-month ECL. Stage 2 uses lifetime ECL — yearly marginal PDs on an amortising EAD, discounted. Stage 3 is LGD × EAD.
- EAD: for cards I applied a 75% CCF on the undrawn limit.
- LGD: comes from collateral after a forced-sale haircut, with product floors.

**How does the stress test work?**
I regressed logit(portfolio default rate) on GDP, unemployment, repo and house-price growth. Scenario shocks × coefficients give a log-odds shift. That shift is scaled by product sensitivity (unsecured > secured) and applied to every account's PD. Collateral is haircut, LGD is recomputed, and accounts crossing the SICR threshold migrate to Stage 2. That migration is why ECL rises more than proportionally under Severe.

**How does the EWS differ from the PD model?**
EWS is transparent rules that credit officers can act on, aligned with RBI SMA. The PD model is statistical. I validated they agree: average model PD rises monotonically Green → Amber → Red, and realised default rates do too.

**How would you take this to production?**
- Validate on an out-of-time window and monitor drift with PSI/CSI.
- Move to PD term structures with probability-weighted forward-looking scenarios.
- Add a champion–challenger framework and periodic recalibration.
- Run the SQL features as scheduled jobs.
- Set up an AML feedback loop so investigator outcomes retrain alert thresholds.

**Is the data real?**
No, it's synthetic but deliberately realistic:
- Non-linear default drivers.
- Unforeseeable shocks, so models can't be perfect.
- Legitimate look-alikes for AML (FD maturities, cash-heavy businesses, overseas remittances), so precision isn't inflated.

The pipeline is data-agnostic. Swapping in a real dataset only changes the loader.

## Links to the BlackRock / Aladdin context
The platform mirrors the portfolio risk view Aladdin provides, applied to a loan book:
- Exposure and concentration analytics.
- Model-driven risk measures with explainability.
- Scenario-based stress testing.
- A single SQL-backed source of truth feeding the dashboard.
