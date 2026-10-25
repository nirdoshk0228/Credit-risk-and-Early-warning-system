"""Unit tests for the risk maths (run: pytest -q)."""
import numpy as np
import pandas as pd
import pytest

from src.ecl import compute_ecl, ead_lgd
from src.ews import sma_class
from src.models import expected_calibration_error, ks_statistic, pd_to_score
from src.stress import SCENARIOS, apply_scenario, fit_satellite


def loan(**kw):
    base = dict(loan_id="L1", customer_id="C1", product="Personal Loan", sanctioned_amount=500_000,
                credit_limit=np.nan, outstanding=400_000.0, interest_rate=12.0, tenure_months=48,
                months_on_book=12, emi=13_000.0, collateral_value=0.0, secured=0, loan_dpd=0,
                pd_12m=0.05, ews_level="Green", current_dpd=0)
    base.update(kw)
    return pd.DataFrame([base])


def test_stage1_ecl_is_pd_lgd_ead():
    out = compute_ecl(ead_lgd(loan()))
    r = out.iloc[0]
    assert r.stage == 1
    expected = 0.05 * 0.65 * r.ead / 1.10 ** 0.5
    assert r.ecl == pytest.approx(expected, rel=1e-9)


def test_stage3_full_lgd_and_stage2_lifetime_exceeds_12m():
    s3 = compute_ecl(ead_lgd(loan(loan_dpd=120))).iloc[0]
    assert s3.stage == 3 and s3.ecl == pytest.approx(s3.lgd * s3.ead)
    s2 = compute_ecl(ead_lgd(loan(loan_dpd=45))).iloc[0]
    assert s2.stage == 2 and s2.ecl == pytest.approx(s2.ecl_lifetime) and s2.ecl > s2.ecl_12m


def test_secured_lgd_respects_collateral_and_floor():
    over = ead_lgd(loan(product="Home Loan", secured=1, collateral_value=10_000_000)).iloc[0]
    assert over.lgd == pytest.approx(0.10)                        # floor
    under = ead_lgd(loan(product="Home Loan", secured=1, collateral_value=300_000)).iloc[0]
    assert 0.10 < under.lgd < 1.0


def test_credit_card_ead_uses_ccf():
    r = ead_lgd(loan(product="Credit Card", credit_limit=200_000, outstanding=50_000)).iloc[0]
    assert r.ead == pytest.approx(50_000 + 0.75 * 150_000)


def test_sma_buckets():
    s = sma_class(pd.Series([0, 15, 45, 75, 120]))
    assert s.tolist() == ["Standard", "SMA-0", "SMA-1", "SMA-2", "NPA"]


def test_ks_and_calibration_metrics():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    p = np.full(1000, 0.1)
    yy = (np.arange(1000) < 100).astype(int)
    assert expected_calibration_error(yy, p) == pytest.approx(0.0, abs=1e-9)


def test_score_scaling_monotonic():
    s = pd_to_score(np.array([0.01, 0.05, 0.2]))
    assert s[0] > s[1] > s[2]
    assert pd_to_score(np.array([1 / 51]))[0] == 600              # 50:1 odds -> 600


def test_stress_increases_pd_and_ecl():
    macro = pd.DataFrame({"gdp_growth": np.linspace(-5, 8, 30), "unemployment_rate": np.linspace(10, 5, 30),
                          "repo_rate": np.linspace(7, 4, 30), "hpi_growth": np.linspace(-2, 6, 30)})
    macro["portfolio_default_rate"] = 1 / (1 + np.exp(3 + 0.04 * macro["gdp_growth"]
                                                      - 0.05 * macro["unemployment_rate"]))
    coefs = fit_satellite(macro)
    base = pd.concat([loan(), loan(loan_id="L2", product="Home Loan", secured=1, collateral_value=600_000)])
    calm = apply_scenario(base, coefs, SCENARIOS["Baseline"])
    severe = apply_scenario(base, coefs, SCENARIOS["Severe"])
    assert (severe["pd_12m"].to_numpy() > calm["pd_12m"].to_numpy()).all()
    assert severe["ecl"].sum() > calm["ecl"].sum()
