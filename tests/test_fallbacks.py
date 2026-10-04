"""The built-in soiling and ageing estimates used when rdtools is not installed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trueyield import causes
from trueyield.expected import add_performance_index
from trueyield.pipeline import analyse_frame

NO_EVENTS = {"events": [], "event_source": "natural rain event, not a maintenance intervention"}


@pytest.fixture()
def without_rdtools(monkeypatch):
    """Make the module behave as if rdtools could not be imported."""
    monkeypatch.setattr(causes, "HAVE_RDTOOLS", False)


def _soiled(frame, cfg, rate_per_day=0.0015, every_days=60):
    """Add a known sawtooth: steady soiling, washed off by heavy rain every `every_days`."""
    days_since_rain = np.arange(len(frame)) % every_days
    frame["energy_kwh"] *= 1.0 - rate_per_day * days_since_rain
    frame.loc[frame.index[days_since_rain == 0], "precip_mm"] = 20.0
    true_loss_pct = 100 * rate_per_day * (every_days - 1) / 2
    return add_performance_index(frame, cfg["expected"]), true_loss_pct


def test_fallback_soiling_finds_a_known_sawtooth(cfg, synthetic_frame, without_rdtools):
    frame, true_loss = _soiled(synthetic_frame, cfg)            # true mean loss about 4.4%
    out = analyse_frame(frame, cfg)
    res = causes.soiling_cause(frame, out["outage"], NO_EVENTS, [], cfg)
    n = res["numbers"]
    assert "fallback" in n["method"] and res["status"] == "indicated"
    assert n["soiling_loss_pct_ci"][0] <= n["soiling_loss_pct"] <= n["soiling_loss_pct_ci"][1]
    # The reference is "own best", which already sits a little below perfectly
    # clean, so the estimate may be modestly low; it must be the right size.
    assert 0.5 * true_loss <= n["soiling_loss_pct"] <= 1.3 * true_loss


def test_fallback_soiling_reports_nothing_on_a_clean_system(cfg, synthetic_frame, without_rdtools):
    synthetic_frame.loc[synthetic_frame.index[::60], "precip_mm"] = 20.0
    out = analyse_frame(synthetic_frame, cfg)
    res = causes.soiling_cause(synthetic_frame, out["outage"], NO_EVENTS, [], cfg)
    assert res["status"] == "not indicated"
    assert res["numbers"]["soiling_loss_pct"] < 1.0


def test_fallback_degradation_finds_a_known_trend(cfg, synthetic_frame, without_rdtools):
    years = (synthetic_frame.index - synthetic_frame.index[0]).days / 365.25
    synthetic_frame["energy_kwh"] *= 1.0 - 0.02 * years          # 2% of the starting level per year
    frame = add_performance_index(synthetic_frame, cfg["expected"])
    out = analyse_frame(frame, cfg)
    res = causes.degradation_cause(frame, out["outage"], cfg)
    n = res["numbers"]
    assert "fallback" in n["method"] and res["status"] == "indicated"
    assert n["rate_pct_per_year_ci"][0] <= -2.0 <= n["rate_pct_per_year_ci"][1] + 0.3
    assert -2.6 <= n["rate_pct_per_year"] <= -1.6


def test_fallback_degradation_is_silent_on_a_stable_system(cfg, synthetic_frame, without_rdtools):
    out = analyse_frame(synthetic_frame, cfg)
    res = causes.degradation_cause(synthetic_frame, out["outage"], cfg)
    assert res["status"] == "not indicated"
    assert abs(res["numbers"]["rate_pct_per_year"]) < 0.5


def test_degradation_needs_more_than_two_years(cfg, synthetic_frame, without_rdtools):
    short = synthetic_frame.loc[:"2022-06-30"].copy()
    out = analyse_frame(short, cfg)
    res = causes.degradation_cause(short, out["outage"], cfg)
    assert res["status"] == "insufficient data"
