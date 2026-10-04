"""Rupee value, cleaning verdict and emissions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trueyield.economics import economics, emissions

ECFG = {"days_per_month": 30.44}
DAYS = 365          # about 12 months


def _loss(moderate, moderate_ci, outage=0.0, deep=0.0):
    """A loss result shaped like detect.estimate_loss returns it."""
    lost = moderate + outage + deep
    spread = (moderate_ci[1] - moderate_ci[0]) / 2
    return {"lost_kwh": lost, "lost_kwh_ci": (lost - spread, lost + spread),
            "sustained_kwh": moderate + deep, "sustained_kwh_ci": (moderate_ci[0] + deep, moderate_ci[1] + deep),
            "moderate_shortfall_kwh": moderate, "moderate_shortfall_kwh_ci": moderate_ci,
            "deep_shortfall_kwh": deep, "outage_kwh": outage}


def _site(tariff=8.0, cost=1000.0):
    return pd.Series({"tariff_inr_per_kwh": tariff, "cleaning_cost_inr": cost})


def test_rupees_follow_kwh_and_carry_the_range():
    out = economics(_loss(1200.0, (900.0, 1500.0)), _site(), DAYS, ECFG)
    months = DAYS / 30.44
    assert out["rupees_lost"] == round(1200.0 * 8.0)
    assert out["rupees_lost_ci"] == [round(900.0 * 8.0), round(1500.0 * 8.0)]
    assert out["rupees_lost_per_month"] == round(1200.0 * 8.0 / months)
    assert out["rupees_lost_per_month_ci"][0] < out["rupees_lost_per_month"] < out["rupees_lost_per_month_ci"][1]


@pytest.mark.parametrize("cost, verdict", [
    (300.0, "fix likely pays"),       # even the low end (about Rs 600/month) beats the cost
    (2000.0, "fix does not pay"),     # even the high end (about Rs 1000/month) is below the cost
    (800.0, "uncertain"),             # the cost lies inside the range
])
def test_verdict_respects_the_range(cost, verdict):
    out = economics(_loss(1200.0, (900.0, 1500.0)), _site(cost=cost), DAYS, ECFG)
    lo, hi = out["sustained_rupees_per_month_ci"]
    assert out["verdict"] == verdict
    assert (lo > cost) == (verdict == "fix likely pays")
    assert (hi < cost) == (verdict == "fix does not pay")


def test_verdict_ignores_outages_and_deep_drops():
    """Cleaning cannot fix an outage, so a huge outage must not make cleaning 'pay'."""
    small = economics(_loss(100.0, (50.0, 150.0)), _site(cost=500.0), DAYS, ECFG)
    with_outage = economics(_loss(100.0, (50.0, 150.0), outage=50_000.0, deep=20_000.0), _site(cost=500.0), DAYS, ECFG)
    assert small["verdict"] == with_outage["verdict"] == "fix does not pay"
    assert with_outage["rupees_lost"] > small["rupees_lost"]         # but the total value still counts it


@pytest.mark.parametrize("tariff", [np.nan, None, ""])
def test_missing_tariff_is_not_guessed(tariff):
    out = economics(_loss(1200.0, (900.0, 1500.0)), _site(tariff=tariff), DAYS, ECFG)
    assert "rupees_lost" not in out
    assert out["verdict"].startswith("[not available")


def test_missing_cleaning_cost_still_gives_rupees():
    out = economics(_loss(1200.0, (900.0, 1500.0)), _site(cost=np.nan), DAYS, ECFG)
    assert out["rupees_lost"] == 9600
    assert out["verdict"].startswith("[not available")


def test_no_interval_means_uncertain():
    loss = {"lost_kwh": 1200.0, "sustained_kwh": 1200.0, "moderate_shortfall_kwh": 1200.0}
    out = economics(loss, _site(), DAYS, ECFG)
    assert out["verdict"].startswith("uncertain")


@pytest.mark.parametrize("factor", [None, "", np.nan])
def test_emissions_skipped_without_a_factor(factor):
    out = emissions(_loss(1200.0, (900.0, 1500.0)), {"emission_factor_kg_per_kwh": factor})
    assert "kg_co2" not in out and out["note"].startswith("[not available")


def test_emissions_with_a_factor():
    out = emissions(_loss(1200.0, (900.0, 1500.0)),
                    {"emission_factor_kg_per_kwh": 0.5, "emission_factor_source": "test source",
                     "emission_factor_year": 2024})
    assert out["kg_co2"] == 600 and out["kg_co2_ci"] == [450, 750]
    assert out["source"] == "test source" and out["year"] == 2024
