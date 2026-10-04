"""Counters that reset every day ("energy today")."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trueyield import ingest


def _systems(capacity=5.0):
    return pd.DataFrame([{"system_id": "A1", "name": "t", "capacity_kwp": capacity, "latitude": 19.0,
                          "longitude": 73.0, "tilt_deg": 15, "azimuth_deg": 180, "commissioning_date": "2020-01-01",
                          "tariff_inr_per_kwh": 7.0, "cleaning_cost_inr": 500.0, "utc_offset_hours": 5.0,
                          "location": ""}]).set_index("system_id", drop=False)


def _energy_today(days, hold_overnight: bool, capacity=5.0):
    """Hourly 'energy today' readings and the true energy of each day.

    hold_overnight=False: the counter drops to zero at midnight.
    hold_overnight=True:  it keeps yesterday's total until the inverter wakes at 05:00.
    """
    ts = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")
    # Energy produced in the hour ending at each stamp, with a different size each day.
    size = 0.5 + 0.05 * (ts.normalize() - days[0]).days.to_numpy()
    hourly = capacity * size * np.clip(np.sin(np.pi * (ts.hour.to_numpy() - 6) / 12.0), 0, None)
    produced = pd.Series(hourly, index=ts)
    today = produced.groupby(ts.normalize()).cumsum()
    if hold_overnight:
        total = produced.groupby(ts.normalize()).sum()
        early = ts.hour < 5
        yesterday = (ts.normalize() - pd.Timedelta(days=1))[early]
        today[early] = total.reindex(yesterday).fillna(0.0).to_numpy()
    truth = produced.groupby(ts.normalize()).sum()
    return pd.DataFrame({"Time": ts, "E-Today (kWh)": today.to_numpy()}), truth


def test_counter_to_interval_with_daily_reset():
    ts = pd.date_range("2024-03-01 15:00", periods=7, freq="h")
    out = ingest.counter_to_interval(pd.Series(ts), pd.Series([10.0, 15.0, 18.0, 18.0, 0.0, 0.5, 6.0]),
                                     resets_daily=True)
    assert np.isnan(out["energy"].iloc[0])
    assert out["energy"].iloc[1:].tolist() == [5.0, 3.0, 0.0, 0.0, 0.5, 5.5]
    assert out["reset"].tolist() == [False, False, False, False, True, False, False]


def test_lifetime_counter_reset_still_unknown():
    """The default (lifetime) behaviour must not change: a drop stays unknown."""
    ts = pd.date_range("2024-03-01 15:00", periods=4, freq="h")
    out = ingest.counter_to_interval(pd.Series(ts), pd.Series([500.0, 502.0, 3.0, 5.0]))
    assert np.isnan(out["energy"].iloc[2])


@pytest.mark.parametrize("hold_overnight", [False, True])
def test_daily_reset_counter_gives_true_daily_energy(cfg, hold_overnight):
    days = pd.date_range("2024-03-01", "2024-03-12", freq="D")
    df, truth = _energy_today(days, hold_overnight)
    systems = _systems()
    layout, _ = ingest.infer_layout(df, "A1_today.csv", systems, cfg["ingest"])
    assert layout.kind == "daily_reset_counter" and layout.unit == "kWh"
    sd = ingest.clean_system("A1", [(layout, df)], systems.loc["A1"], cfg["ingest"])
    got = sd.daily["energy_kwh"].dropna()
    assert len(got) >= len(days) - 1                      # the first day has no earlier reading to step from
    np.testing.assert_allclose(got.to_numpy(), truth.reindex(got.index).to_numpy(), rtol=1e-9, atol=1e-9)
    assert sd.completeness["counter_resets"] == 0         # an expected daily reset is not an anomaly


def test_daily_reset_counter_with_a_missing_day(cfg):
    days = pd.date_range("2024-03-01", "2024-03-12", freq="D")
    df, truth = _energy_today(days, hold_overnight=False)
    df = df[df["Time"].dt.normalize() != pd.Timestamp("2024-03-06")]
    systems = _systems()
    layout, _ = ingest.infer_layout(df, "A1_today.csv", systems, cfg["ingest"])
    assert layout.kind == "daily_reset_counter"
    sd = ingest.clean_system("A1", [(layout, df)], systems.loc["A1"], cfg["ingest"])
    d = sd.daily
    assert not bool(d.loc["2024-03-06", "valid"]) and np.isnan(d.loc["2024-03-06", "energy_kwh"])
    for day in ("2024-03-05", "2024-03-08"):              # neighbours of the gap keep their true energy
        assert d.loc[day, "energy_kwh"] == pytest.approx(truth.loc[day])
    # The day after the gap must never be inflated by energy from the missing day.
    after = d.loc["2024-03-07", "energy_kwh"]
    assert np.isnan(after) or after == pytest.approx(truth.loc["2024-03-07"])


def test_other_layouts_are_not_mistaken_for_daily_reset(cfg):
    icfg = cfg["ingest"]
    ts = pd.Series(pd.date_range("2024-03-01", periods=24 * 10, freq="h"))
    bell = np.clip(np.sin(np.pi * (ts.dt.hour.to_numpy() - 6) / 12.0), 0, None)
    assert not ingest._resets_daily(ts, pd.Series(3.0 * bell), icfg)                 # interval energy or power
    assert not ingest._resets_daily(ts, pd.Series(1000 + np.cumsum(3.0 * bell)), icfg)  # lifetime counter
    systems = _systems()
    lifetime = pd.DataFrame({"Time": ts, "Total Energy (kWh)": 1000 + np.cumsum(3.0 * bell)})
    assert ingest.infer_layout(lifetime, "A1.csv", systems, icfg)[0].kind == "cumulative_counter"
    daily_yield = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=40, freq="D"),
                                "Energy (kWh)": 20 + 5 * np.sin(np.arange(40))})
    assert ingest.infer_layout(daily_yield, "A1.csv", systems, icfg)[0].kind == "interval_energy"
