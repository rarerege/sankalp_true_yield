"""Bootstrap interval, outage detection and the injected-loss recovery check."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trueyield import detect, validation, verify
from trueyield.expected import add_performance_index
from trueyield.pipeline import analyse_frame


def test_block_indices_keep_blocks_together():
    idx = detect._block_indices(100, 7, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() <= 99
    assert (np.diff(idx[:, :7], axis=1) == 1).all()           # each block is consecutive days


def test_percentile_interval_covers_known_mean():
    """A 95% bootstrap interval for a mean should contain the truth about 95% of the time."""
    rng = np.random.default_rng(42)
    hits = 0
    for _ in range(200):
        sample = rng.normal(10.0, 2.0, 60)
        means = sample[rng.integers(0, 60, (500, 60))].mean(axis=1)
        lo, hi = detect.percentile_interval(means, 0.95)
        hits += lo <= 10.0 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def test_healthy_system_shows_almost_no_loss(cfg, synthetic_frame):
    """The high-quantile reference must not invent a loss out of pure noise."""
    out = analyse_frame(synthetic_frame, cfg)
    assert not out["outage"].any()
    assert out["loss"]["loss_pct"] < 0.5
    assert out["loss"]["loss_pct_ci"][0] <= out["loss"]["loss_pct"] <= out["loss"]["loss_pct_ci"][1]


def test_outage_detection(cfg, synthetic_frame):
    f = synthetic_frame
    sunny = f.index[(f["clearness"] > 0.9).to_numpy()][100:104]
    cloudy = f.index[(f["clearness"] < 0.45).to_numpy()][5]
    f.loc[sunny, "energy_kwh"] = 0.0
    f.loc[cloudy, "energy_kwh"] = 0.0                         # zero on a dark day: not provable as an outage
    f = add_performance_index(f, cfg["expected"])
    outage = detect.find_outages(f, cfg["detect"])
    assert set(f.index[outage]) == set(sunny)
    out = analyse_frame(f, cfg)
    expected_lost = float(f.loc[sunny, "expected_kwh"].sum())
    assert abs(out["loss"]["outage_kwh"] - expected_lost) < 1e-6
    assert out["loss"]["outage_kwh_ci"][0] <= expected_lost <= out["loss"]["outage_kwh_ci"][1]


def test_detection_limit_shrinks_with_less_noise(cfg, synthetic_frame):
    noisy = analyse_frame(synthetic_frame.copy(), cfg)["noise"]
    quiet_frame = synthetic_frame.copy()
    quiet_frame["energy_kwh"] = quiet_frame["model_kwh"] * 0.8 * (
        1 + 0.25 * (quiet_frame["energy_kwh"] / (quiet_frame["model_kwh"] * 0.8) - 1))
    quiet = analyse_frame(add_performance_index(quiet_frame, cfg["expected"]), cfg)["noise"]
    assert quiet["day_to_day_sigma_fraction"] < noisy["day_to_day_sigma_fraction"]
    assert quiet["windows"][14]["se_fraction"] < noisy["windows"][14]["se_fraction"]
    assert noisy["windows"][14]["limit_fraction"] >= cfg["detect"]["min_shortfall_fraction"]


def test_injected_loss_is_recovered_within_range(cfg, synthetic_frame):
    """Inject a known soiling ramp and outage; the pipeline must find them."""
    base = analyse_frame(synthetic_frame, cfg)
    stretch = (pd.Timestamp("2022-02-01"), pd.Timestamp("2022-09-28"))
    baseline = detect.estimate_loss(synthetic_frame, base["outage"], base["noise"], base["ref_info"], cfg, period=stretch)
    modified, info = validation.inject(synthetic_frame, stretch, cfg["validation"], cfg["detect"]["outage_min_clearness"])
    modified = add_performance_index(modified, cfg["expected"])
    after = analyse_frame(modified, cfg, period=stretch)
    lo, hi = after["loss"]["lost_kwh_ci"]
    truth = baseline["lost_kwh"] + info["injected_kwh"]
    assert len(info["outage_days"]) == cfg["validation"]["outage_days"]
    assert all(bool(after["outage"][pd.Timestamp(d)]) for d in info["outage_days"])
    assert lo <= truth <= hi
    recovered = after["loss"]["lost_kwh"] - baseline["lost_kwh"]
    assert 0.6 * info["injected_kwh"] <= recovered <= 1.3 * info["injected_kwh"]


def test_recovery_test_finds_a_known_step(cfg, synthetic_frame):
    f = synthetic_frame
    event_day = pd.Timestamp("2022-06-15")
    f.loc[event_day - pd.Timedelta(days=40):event_day - pd.Timedelta(days=1), "energy_kwh"] *= 0.92
    f = add_performance_index(f, cfg["expected"])
    out = analyse_frame(f, cfg)
    events = pd.DataFrame([{"system_id": "T", "date": event_day, "type": "cleaning", "note": "test"}])
    rec = verify.verify_events(f, out["outage"], events, "T", cfg)
    ev = rec["events"][0]
    assert rec["event_source"] == "dated maintenance events" and ev["testable"]
    true_change = 100 * (1 / 0.92 - 1)                         # +8.7%
    assert ev["change_pct_ci"][0] <= true_change <= ev["change_pct_ci"][1]
    assert ev["distinguishable_from_zero"]
