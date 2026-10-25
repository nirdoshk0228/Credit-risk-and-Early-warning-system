"""
Synthetic Indian retail-banking portfolio.

Generates a realistic, internally consistent dataset:
  customers -> loans -> monthly payments (DPD state machine) -> bank-account
  transactions (salary, spends, cash, NACH EMI debits incl. bounces)
plus a quarterly macro history and injected AML typologies.

Every borrower has a latent credit-risk factor. A subset are "deteriorating":
from some month onward their stress rises, which shows up gradually as lower
inflows, more cash usage, NACH bounces, rising DPD and a falling bureau score.
These are exactly the patterns an Early Warning System should pick up *before*
the account turns NPA. Outcome labels (default in the next 12 months, AML
ground truth) are kept in separate tables and are only used for training and
evaluation.
"""
import numpy as np
import pandas as pd

EMPLOYMENT = ["Salaried", "Self-Employed", "Business Owner", "Retired", "Gig / Contract"]
EMP_P = [0.55, 0.18, 0.14, 0.06, 0.07]
EMP_INCOME = {"Salaried": 55_000, "Self-Employed": 60_000, "Business Owner": 95_000,
              "Retired": 38_000, "Gig / Contract": 28_000}
EMP_RISK = {"Salaried": -0.20, "Self-Employed": 0.25, "Business Owner": 0.15,
            "Retired": 0.00, "Gig / Contract": 0.55}
STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi NCR", "Gujarat", "Telangana",
          "Uttar Pradesh", "West Bengal", "Madhya Pradesh", "Rajasthan", "Kerala", "Punjab"]
STATE_P = [0.17, 0.11, 0.10, 0.11, 0.08, 0.07, 0.09, 0.06, 0.06, 0.06, 0.05, 0.04]

PRODUCTS = ["Home Loan", "Auto Loan", "Personal Loan", "Credit Card", "Gold Loan", "Business Loan"]
PRODUCT_RISK_TILT = {"Home Loan": -0.45, "Auto Loan": -0.10, "Personal Loan": 0.25,
                     "Credit Card": 0.20, "Gold Loan": -0.30, "Business Loan": 0.30}
SNAPSHOT_MONTH = pd.Timestamp("2025-12-01")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def month_start(m: int) -> pd.Timestamp:
    """month_idx 1 = snapshot month (Dec-2025), 12 = Jan-2025."""
    return SNAPSHOT_MONTH - pd.DateOffset(months=int(m) - 1)


class PortfolioGenerator:
    def __init__(self, n_customers: int = 8000, seed: int = 42,
                 pay_months: int = 12, txn_months: int = 6):
        self.n = n_customers
        self.rng = np.random.default_rng(seed)
        self.pay_months = pay_months
        self.txn_months = txn_months

    # ------------------------------------------------------------------ customers
    def _customers(self):
        rng, n = self.rng, self.n
        q = rng.normal(0, 1, n)                                     # credit quality
        emp = rng.choice(EMPLOYMENT, n, p=EMP_P)
        tier = rng.choice([1, 2, 3], n, p=[0.40, 0.35, 0.25])
        age = np.clip(rng.normal(37, 9, n), 22, 62).astype(int)
        age = np.where(emp == "Retired", rng.integers(58, 72, n), age)
        med = np.array([EMP_INCOME[e] for e in emp])
        tier_mult = np.select([tier == 1, tier == 2], [1.25, 1.0], 0.8)
        income = np.round(med * tier_mult * np.exp(rng.normal(0, 0.5, n) + 0.15 * q), -2)
        years = np.clip(rng.exponential(5, n), 0.5, 25).round(1)
        bureau_prev = np.clip(725 + 55 * q + rng.normal(0, 22, n), 300, 900)

        latent = (-0.85 * q + np.array([EMP_RISK[e] for e in emp]) + 0.20 * (tier == 3)
                  - 0.25 * np.log(income / 50_000) - 0.04 * years + rng.normal(0, 0.55, n))

        deteriorating = rng.random(n) < sigmoid(1.1 * latent - 2.3)
        det_start = rng.integers(2, 10, n)                 # month_idx when stress began
        det_slope = rng.uniform(0.12, 0.35, n)

        # stress[c, m] for m = 1..12 (index 0 unused)
        m = np.arange(self.pay_months + 1)
        stress = np.clip(det_start[:, None] - m[None, :] + 1, 0, None) * det_slope[:, None]
        stress = np.where(deteriorating[:, None], stress, 0.0)
        stress[:, 0] = 0

        drop = np.where(deteriorating, stress[:, 1] * rng.uniform(40, 90, n), rng.normal(0, 10, n))
        bureau_now = np.clip(bureau_prev - drop, 300, 900).astype(int)

        cust = pd.DataFrame({
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": age,
            "employment_type": emp,
            "city_tier": tier,
            "state": rng.choice(STATES, n, p=STATE_P),
            "monthly_income": income,
            "years_with_bank": years,
            "bureau_score": bureau_now,
            "bureau_score_6m_ago": bureau_prev.astype(int),
            "kyc_risk": rng.choice(["Low", "Medium", "High"], n, p=[0.80, 0.17, 0.03]),
        })
        self._q, self._latent, self._stress = q, latent, stress
        self._deteriorating = deteriorating
        return cust

    # ------------------------------------------------------------------ loans
    def _loans(self, cust):
        rng = self.rng
        rows = []
        n_loans = rng.choice([1, 2, 3], self.n, p=[0.50, 0.35, 0.15])
        for i, c in enumerate(cust.itertuples(index=False)):
            w = np.array([0.18, 0.15, 0.22, 0.30, 0.08, 0.07])
            if c.employment_type in ("Salaried", "Retired"):
                w[5] = 0.0
            elif c.employment_type == "Gig / Contract":
                w[5] = 0.02
            else:
                w[5] = 0.35
            if c.city_tier == 3:
                w[4] = 0.20
            if c.monthly_income < 35_000:
                w[0] = 0.03
            w = w / w.sum()
            for prod in rng.choice(PRODUCTS, n_loans[i], replace=False, p=w):
                rows.append(self._make_loan(i, c, prod))
        loans = pd.DataFrame(rows)
        loans.insert(0, "loan_id", [f"L{500000 + k}" for k in range(len(loans))])
        return loans

    def _make_loan(self, i, c, prod):
        rng = self.rng
        inc, bureau = c.monthly_income, c.bureau_score
        risk_premium = max(0, 760 - bureau) / 100.0
        spec = {
            "Home Loan":     ((50, 90), [180, 240], (8.4, 1.0, 0.5)),
            "Auto Loan":     ((8, 16), [48, 60, 84], (9.0, 1.5, 0.8)),
            "Personal Loan": ((3, 10), [24, 36, 48, 60], (11.0, 4.0, 1.5)),
            "Credit Card":   ((1.5, 4), [0], (36.0, 6.0, 0.0)),
            "Gold Loan":     ((1, 4), [12], (9.0, 2.5, 0.5)),
            "Business Loan": ((10, 30), [36, 60, 84], (11.5, 3.0, 1.2)),
        }[prod]
        (lo, hi), tenures, (r0, rspread, rrisk) = spec
        sanction = float(np.round(inc * rng.uniform(lo, hi), -3))
        tenure = int(rng.choice(tenures))
        rate = round(r0 + rng.uniform(0, rspread) + rrisk * risk_premium, 2)
        if prod == "Credit Card":
            mob = int(rng.integers(6, 97))
        elif prod == "Gold Loan":
            mob = int(rng.integers(1, 12))
        else:
            mob = int(rng.integers(3, max(4, tenure - 2)))
            if prod == "Home Loan":
                mob = min(mob, 120)
        r = rate / 1200.0
        limit, collateral, secured = np.nan, 0.0, 0
        if prod == "Credit Card":
            limit = sanction
            util = float(np.clip(sigmoid(0.9 * self._latent[i] + 0.8 * self._stress[i, 1]
                                         - 0.6 + rng.normal(0, 0.6)), 0.03, 1.0))
            outstanding = round(limit * util, 2)
            emi = round(max(0.05 * outstanding, 500.0), 2)        # minimum amount due
        elif prod == "Gold Loan":
            outstanding = sanction
            emi = round(sanction * r, 2)                          # interest-only bullet
            collateral = sanction / rng.uniform(0.65, 0.75)       # RBI LTV cap 75%
            secured = 1
        else:
            emi = sanction * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1)
            outstanding = sanction * (1 + r) ** mob - emi * ((1 + r) ** mob - 1) / r
            outstanding, emi = round(max(outstanding, 0), 2), round(emi, 2)
            if prod == "Home Loan":
                collateral = sanction / rng.uniform(0.60, 0.80) * 1.06 ** (mob / 12)
                secured = 1
            elif prod == "Auto Loan":
                collateral = sanction / rng.uniform(0.80, 0.90) * 0.85 ** (mob / 12)
                secured = 1
            elif prod == "Business Loan" and rng.random() < 0.55:
                collateral = sanction * rng.uniform(0.8, 1.3)
                secured = 1
        orig = month_start(1) - pd.DateOffset(months=mob)
        return {
            "customer_id": c.customer_id, "product": prod, "sanctioned_amount": sanction,
            "credit_limit": limit, "outstanding": outstanding, "interest_rate": rate,
            "tenure_months": tenure, "months_on_book": mob, "emi": emi,
            "collateral_value": round(collateral, 2), "secured": secured,
            "origination_date": orig.date().isoformat(), "_cidx": i,
        }

    # ------------------------------------------------------------------ payments
    def _payments(self, loans):
        """Monthly DPD state machine (bucket = installments overdue)."""
        rng = self.rng
        L = len(loans)
        cidx = loans["_cidx"].to_numpy()
        risk = self._cust_risk[cidx] + loans["product"].map(PRODUCT_RISK_TILT).to_numpy()
        mob = loans["months_on_book"].to_numpy()
        due = loans["emi"].to_numpy()
        bucket = np.zeros(L, dtype=int)
        recs, outcomes = [], {}
        for m in range(self.pay_months, 0, -1):
            active = mob >= m
            s = self._stress[cidx, m]
            shock = rng.random(L) < 0.02
            p_miss = sigmoid(-5.2 + 1.0 * risk + 2.2 * s + 1.6 * (bucket > 0) + 0.35 * bucket + 1.8 * shock)
            p_cure = sigmoid(0.9 - 0.8 * risk - 1.5 * s - 0.5 * bucket)
            p_part = sigmoid(-4.0 + 0.8 * risk + 1.5 * s)
            u1, u2, u3 = rng.random(L), rng.random(L), rng.random(L)
            miss = u1 < p_miss
            cure = ~miss & (bucket > 0) & (u2 < p_cure)
            part = ~miss & ~cure & (u3 < p_part)
            paid = np.where(miss, 0.0,
                            np.where(cure, due * (bucket + 1),
                                     np.where(part, due * rng.uniform(0.4, 0.85, L), due)))
            bucket = np.where(miss, np.minimum(bucket + 1, 7), np.where(cure, 0, bucket))
            dpd = np.where(bucket > 0, 30 * (bucket - 1) + rng.integers(1, 31, L), 0)
            dpd = np.where(part & (bucket == 0), rng.integers(3, 25, L), dpd)
            late = np.where(miss, np.nan,
                            np.clip(rng.exponential(1.5 + 4 * np.clip(risk, 0, None) + 6 * s), 0, 28).round())
            idx = np.where(active)[0]
            due_date = (month_start(m) + pd.Timedelta(days=4))
            recs.append(pd.DataFrame({
                "loan_id": loans["loan_id"].to_numpy()[idx], "month_idx": m,
                "due_date": due_date.date().isoformat(),
                "due_amount": due[idx], "paid_amount": np.round(paid[idx], 2),
                "days_late": late[idx], "dpd": dpd[idx],
            }))
            if m <= self.txn_months:
                outcomes[m] = np.select([miss, part], ["MISS", "PART"], "OK"), paid.copy(), active.copy()
        pay = pd.concat(recs, ignore_index=True)
        pay.insert(0, "payment_id", np.arange(1, len(pay) + 1))
        self._final_bucket = bucket
        self._pay_outcomes = outcomes
        return pay

    # ------------------------------------------------------------------ transactions
    def _transactions(self, cust, loans):
        rng = self.rng
        T = self.txn_months
        loan_by_c = loans.groupby("_cidx").indices
        due = loans["emi"].to_numpy()
        lid = loans["loan_id"].to_numpy()
        cols = {k: [] for k in ["customer_id", "month_idx", "day", "txn_hour", "amount",
                                "direction", "channel", "category", "status", "counterparty_id"]}

        def add(cid, m, day, hour, amt, d, ch, cat, st, cp):
            k = len(amt)
            cols["customer_id"].append(np.repeat(cid, k)); cols["month_idx"].append(np.repeat(m, k))
            cols["day"].append(day); cols["txn_hour"].append(hour); cols["amount"].append(amt)
            cols["direction"].append(np.repeat(d, k)); cols["channel"].append(np.broadcast_to(ch, k))
            cols["category"].append(np.repeat(cat, k)); cols["status"].append(np.broadcast_to(st, k))
            cols["counterparty_id"].append(cp)

        def hours(k):
            return np.where(rng.random(k) < 0.01, rng.integers(0, 5, k), rng.integers(7, 23, k))

        n_aml = int(round(self.n * 0.015))
        aml_idx = set(rng.choice(self.n, n_aml, replace=False).tolist())
        patterns = ["STRUCTURING", "PASS_THROUGH", "INTL_ODD_HOUR", "DORMANT_BURST"]
        aml_truth = {}

        for i, c in enumerate(cust.itertuples(index=False)):
            cid, inc = c.customer_id, c.monthly_income
            s_all = self._stress[i]
            pattern = None
            if i in aml_idx:
                pattern = patterns[len(aml_truth) % 4]
                aml_truth[cid] = pattern
            clients = rng.integers(10_000, 99_999, 8)
            # legitimate behaviour that *looks* unusual -> realistic false positives
            biz = c.employment_type in ("Self-Employed", "Business Owner")
            cash_biz = biz and rng.random() < 0.30
            windfall_m = int(rng.integers(1, T + 1)) if rng.random() < 0.03 else 0   # FD maturity / asset sale
            remitter = inc > 80_000 and rng.random() < 0.02                          # fees / family abroad
            returner = pattern is None and rng.random() < 0.01                       # back from abroad
            for m in range(1, T + 1):
                if returner and m >= 4:
                    add(cid, m, rng.integers(1, 28, 2), hours(2), np.round(rng.uniform(100, 2000, 2), 2),
                        "DR", "UPI", "SPEND", "SUCCESS", rng.integers(100_000, 105_000, 2).astype(str))
                    continue
                if cash_biz:
                    k = int(rng.integers(0, 4))
                    if k:
                        add(cid, m, rng.integers(1, 29, k), rng.integers(10, 17, k),
                            np.round(rng.uniform(8_000, 120_000, k), -2), "CR", "BRANCH", "CASH_DEPOSIT",
                            "SUCCESS", np.repeat("CASH", k))
                if windfall_m == m:
                    big = inc * rng.uniform(3, 10)
                    d0 = int(rng.integers(1, 20))
                    add(cid, m, np.array([d0]), rng.integers(9, 18, 1), np.array([round(big, 2)]), "CR",
                        "NEFT", "TRANSFER_IN", "SUCCESS", np.array(["FD_MATURITY"]))
                    j = int(rng.integers(1, 4))
                    out = big * rng.uniform(0.3, 0.97) * rng.dirichlet(np.ones(j))
                    add(cid, m, d0 + rng.integers(0, 8, j), rng.integers(8, 23, j), np.round(out, 2), "DR",
                        "NETBANKING", "TRANSFER_OUT", "SUCCESS", rng.integers(800_000, 899_999, j).astype(str))
                if remitter and rng.random() < 0.35:
                    add(cid, m, rng.integers(1, 29, 1),
                        np.where(rng.random(1) < 0.2, rng.integers(0, 5, 1), rng.integers(9, 23, 1)),
                        np.round(rng.uniform(1e5, 6e5, 1), -4), "DR", "SWIFT", "INTERNATIONAL", "SUCCESS",
                        np.array(["REMIT_EDU"]))
                if pattern == "DORMANT_BURST" and m >= 4:
                    add(cid, m, rng.integers(1, 28, 1), hours(1), np.round(rng.uniform(200, 900, 1), 2),
                        "DR", "UPI", "SPEND", "SUCCESS", rng.integers(100_000, 105_000, 1).astype(str))
                    continue
                s = s_all[m]
                factor = float(np.clip(1 - 0.35 * s + rng.normal(0, 0.08), 0.2, 1.4))
                # inflows
                if c.employment_type in ("Salaried", "Retired"):
                    if not (self._deteriorating[i] and s > 1.2 and rng.random() < 0.5):
                        add(cid, m, rng.integers(1, 4, 1), rng.integers(9, 13, 1),
                            np.array([round(inc * factor, 2)]), "CR", "NEFT", "SALARY", "SUCCESS",
                            np.array(["EMPLOYER"]))
                else:
                    k = rng.poisson(4) + 2
                    amt = inc * factor * rng.uniform(0.85, 1.15) * rng.dirichlet(np.ones(k))
                    add(cid, m, rng.integers(1, 29, k), hours(k), np.round(amt, 2), "CR",
                        rng.choice(["UPI", "IMPS", "NEFT"], k), "BUSINESS_INFLOW", "SUCCESS",
                        rng.choice(clients, k).astype(str))
                # spends
                k = rng.poisson(6) + 2
                amt = inc * rng.uniform(0.35, 0.60) * rng.dirichlet(np.ones(k) * 0.8)
                add(cid, m, rng.integers(1, 29, k), hours(k), np.round(np.maximum(amt, 20), 2), "DR",
                    rng.choice(["UPI", "POS", "NETBANKING"], k, p=[0.6, 0.3, 0.1]), "SPEND", "SUCCESS",
                    rng.integers(100_000, 105_000, k).astype(str))
                # cash withdrawals (share rises under stress)
                k = rng.poisson(1.0 + 1.2 * s)
                if k:
                    amt = inc * rng.uniform(0.05, 0.15) * (1 + 0.8 * s) * rng.dirichlet(np.ones(k))
                    add(cid, m, rng.integers(1, 29, k), hours(k), np.round(np.maximum(amt, 500), -2), "DR",
                        "ATM", "CASH_WITHDRAWAL", "SUCCESS", np.repeat("ATM", k))
                # EMI auto-debits; a missed instalment shows up as a NACH return (bounce)
                for j in loan_by_c.get(i, []):
                    outcome, paid, active = self._pay_outcomes[m]
                    if not active[j]:
                        continue
                    st = "RETURNED" if outcome[j] == "MISS" else "SUCCESS"
                    amt = due[j] if outcome[j] == "MISS" else paid[j]
                    add(cid, m, np.array([5]), np.array([6]), np.array([round(float(amt), 2)]),
                        "DR", "NACH", "EMI", st, np.array([lid[j]]))
                if inc > 150_000 and rng.random() < 0.15:
                    add(cid, m, rng.integers(1, 29, 1), hours(1), np.round(rng.uniform(5e3, 6e4, 1), 2),
                        "DR", "POS_INTL", "INTERNATIONAL", "SUCCESS", np.array(["MERCHANT_INTL"]))
                if pattern and m <= 3:
                    self._inject(pattern, cid, m, inc, add)
        txn = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
        day = np.clip(txn["day"].to_numpy(), 1, 28)
        base = txn["month_idx"].map(lambda m: month_start(m))
        txn["txn_date"] = (base + pd.to_timedelta(day - 1, unit="D")).dt.date.astype(str)
        txn = txn.drop(columns="day").sort_values(["customer_id", "txn_date", "txn_hour"]).reset_index(drop=True)
        txn.insert(0, "txn_id", np.arange(1, len(txn) + 1))
        txn = txn[["txn_id", "customer_id", "txn_date", "month_idx", "txn_hour", "amount",
                   "direction", "channel", "category", "status", "counterparty_id"]]
        truth = pd.DataFrame({"customer_id": list(aml_truth), "pattern": list(aml_truth.values())})
        return txn, truth

    def _inject(self, pattern, cid, m, inc, add):
        rng = self.rng
        if pattern == "STRUCTURING" and rng.random() < 0.70:
            k = int(rng.integers(2, 6))
            amt = np.round(rng.uniform(36_000, 49_900, k), -2)        # just under the Rs 50k PAN threshold
            add(cid, m, rng.integers(1, 29, k), rng.integers(10, 16, k), amt, "CR", "BRANCH",
                "CASH_DEPOSIT", "SUCCESS", np.repeat("CASH", k))
            j = int(rng.integers(1, 3))
            out = amt.sum() * rng.uniform(0.75, 0.9) * rng.dirichlet(np.ones(j))
            add(cid, m, rng.integers(20, 29, j), rng.integers(9, 22, j), np.round(out, 2), "DR", "IMPS",
                "TRANSFER_OUT", "SUCCESS", rng.integers(900_000, 999_999, j).astype(str))
        elif pattern == "PASS_THROUGH" and rng.random() < 0.70:
            big = inc * rng.uniform(2.5, 8)
            day = int(rng.integers(1, 22))
            add(cid, m, np.array([day]), rng.integers(9, 18, 1), np.array([round(big, 2)]), "CR",
                "RTGS", "TRANSFER_IN", "SUCCESS", rng.integers(900_000, 999_999, 1).astype(str))
            j = int(rng.integers(2, 6))
            out = big * rng.uniform(0.80, 0.98) * rng.dirichlet(np.ones(j))
            add(cid, m, day + rng.integers(0, 5, j), rng.integers(9, 23, j), np.round(out, 2),
                "DR", "IMPS", "TRANSFER_OUT", "SUCCESS", rng.integers(900_000, 999_999, j).astype(str))
        elif pattern == "INTL_ODD_HOUR" and rng.random() < 0.75:
            k = int(rng.integers(1, 3))
            amt = np.round(rng.uniform(40_000, 250_000, k), -3)
            add(cid, m, rng.integers(1, 29, k), rng.integers(0, 6, k), amt, "DR", "SWIFT",
                "INTERNATIONAL", "SUCCESS", rng.integers(900_000, 999_999, k).astype(str))
        elif pattern == "DORMANT_BURST":
            k = int(rng.integers(8, 18))
            amt = np.round(rng.uniform(10_000, 90_000, k), -3)
            d = np.where(rng.random(k) < 0.5, "CR", "DR")
            cat = np.where(d == "CR", "TRANSFER_IN", "TRANSFER_OUT")
            for dd in ("CR", "DR"):
                mask = d == dd
                if mask.any():
                    add(cid, m, rng.integers(1, 29, mask.sum()), rng.integers(8, 23, mask.sum()), amt[mask],
                        dd, "IMPS", cat[mask][0], "SUCCESS",
                        rng.integers(900_000, 999_999, mask.sum()).astype(str))

    # ------------------------------------------------------------------ outcomes
    def _outcomes(self, cust):
        rng = self.rng
        npa_now = pd.Series(self._final_bucket >= 4).groupby(self._loans_cidx).max()
        npa_now = npa_now.reindex(range(self.n), fill_value=False).to_numpy()
        max_b = pd.Series(self._final_bucket).groupby(self._loans_cidx).max()
        max_b = max_b.reindex(range(self.n), fill_value=0).to_numpy()
        future_shock = rng.random(self.n) < 0.04                    # unforeseeable events
        s_now = self._stress[:, 1]
        volatile = cust["employment_type"].isin(["Self-Employed", "Gig / Contract", "Business Owner"]).to_numpy()
        bureau = cust["bureau_score"].to_numpy()
        # non-linear / interaction effects seen in real credit data:
        #   bureau cliff below ~650, over-leverage threshold, revolving stress,
        #   cash-flow shocks hurt volatile-income borrowers more
        nonlinear = (1.0 * (bureau < 650) + 0.6 * (bureau < 600)
                     + 1.1 * (self._dti > 0.55) * (1 + s_now)
                     + 0.9 * (self._cc_util > 0.85)
                     + 1.0 * s_now * volatile
                     - 0.5 * (cust["years_with_bank"].to_numpy() > 8) * (s_now == 0))
        p = sigmoid(-4.75 + 0.70 * self._cust_risk + 1.1 * s_now + 0.5 * max_b + nonlinear
                    + 2.5 * future_shock + rng.normal(0, 0.60, self.n))
        default = np.where(npa_now, 1, (rng.random(self.n) < p).astype(int))
        return pd.DataFrame({"customer_id": cust["customer_id"], "default_12m": default,
                             "already_npa": npa_now.astype(int)})

    # ------------------------------------------------------------------ macro
    def _macro(self):
        rng = self.rng
        q = pd.period_range("2015Q1", "2025Q4", freq="Q")
        gdp = np.array([7.5] * 4 + [8.0] * 4 + [6.8] * 4 + [6.5] * 4 + [4.5] * 4
                       + [3.0, -8.0, -4.0, 0.7, 2.5, 9.0, 8.5, 5.4]
                       + [7.0] * 4 + [7.8] * 4 + [6.5] * 4 + [7.0] * 4) + rng.normal(0, 0.4, len(q))
        unemp = np.array([5.5] * 12 + [6.5] * 4 + [7.5] * 4
                         + [8.0, 12.0, 9.0, 7.5, 7.0, 9.5, 7.5, 7.5]
                         + [7.8] * 4 + [7.6] * 4 + [7.4] * 4 + [7.2] * 4) + rng.normal(0, 0.2, len(q))
        repo = np.array([7.5, 7.25, 6.75, 6.75, 6.5, 6.5, 6.5, 6.25, 6.25, 6.25, 6.0, 6.0,
                         6.0, 6.25, 6.5, 6.5, 6.25, 5.75, 5.4, 5.15, 4.4, 4.0, 4.0, 4.0,
                         4.0, 4.0, 4.0, 4.0, 4.0, 4.9, 5.9, 6.25, 6.5, 6.5, 6.5, 6.5,
                         6.5, 6.5, 6.5, 6.5, 6.25, 6.0, 5.5, 5.5])
        hpi = np.array([6.0] * 8 + [4.5] * 4 + [4.0] * 4 + [3.0] * 4
                       + [1.0, -2.0, -1.0, 1.0, 2.5, 3.5, 4.0, 4.5]
                       + [5.5] * 4 + [6.5] * 4 + [6.0] * 4 + [5.0] * 4) + rng.normal(0, 0.5, len(q))
        dr = sigmoid(-3.05 - 0.035 * gdp + 0.06 * unemp + 0.10 * (repo - 6) - 0.03 * hpi
                     + rng.normal(0, 0.05, len(q)))
        return pd.DataFrame({"quarter": q.astype(str), "gdp_growth": gdp.round(2),
                             "unemployment_rate": unemp.round(2), "repo_rate": repo,
                             "hpi_growth": hpi.round(2), "portfolio_default_rate": dr.round(5)})

    # ------------------------------------------------------------------ main
    def generate(self) -> dict[str, pd.DataFrame]:
        cust = self._customers()
        loans = self._loans(cust)
        self._loans_cidx = loans["_cidx"].to_numpy()
        due_c = loans.groupby("_cidx")["emi"].sum().reindex(range(self.n), fill_value=0).to_numpy()
        dti = due_c / cust["monthly_income"].to_numpy()
        self._dti = dti
        cc = loans[loans["product"] == "Credit Card"]
        self._cc_util = (cc["outstanding"] / cc["credit_limit"]).groupby(cc["_cidx"]).max() \
            .reindex(range(self.n), fill_value=0.0).to_numpy()
        self._cust_risk = self._latent + 1.5 * np.clip(dti - 0.45, 0, None)
        pay = self._payments(loans)
        txn, aml = self._transactions(cust, loans)
        outcomes = self._outcomes(cust)
        return {
            "customers": cust,
            "loans": loans.drop(columns="_cidx"),
            "payments": pay,
            "transactions": txn,
            "macro_history": self._macro(),
            "default_outcomes": outcomes,
            "aml_ground_truth": aml,
        }
