"""Counter-to-interval conversion, layout inference and gap handling."""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from trueyield import ingest


def test_counter_to_interval_basic():
    ts = pd.date_range("2024-03-01 06:00", periods=5, freq="h")
    out = ingest.counter_to_interval(pd.Series(ts), pd.Series([100.0, 101.5, 104.0, 104.0, 107.0]))
    assert np.isnan(out["energy"].iloc[0])                    # nothing known before the first reading
    assert out["energy"].iloc[1:].tolist() == [1.5, 2.5, 0.0, 3.0]
    assert out["energy"].sum() == 7.0                         # equals last minus first


def test_counter_reset_is_not_guessed():
    ts = pd.date_range("2024-03-01 06:00", periods=4, freq="h")
    out = ingest.counter_to_interval(pd.Series(ts), pd.Series([500.0, 502.0, 3.0, 5.0]))
    assert out["reset"].tolist() == [False, False, True, False]
    assert np.isnan(out["energy"].iloc[2])                    # the step across a reset is unknown
    assert out["energy"].iloc[3] == 2.0


def test_counter_long_gap_is_unassignable():
    ts = pd.to_datetime(["2024-03-01 12:00", "2024-03-01 13:00", "2024-03-04 13:00", "2024-03-04 14:00"])
    out = ingest.counter_to_interval(pd.Series(ts), pd.Series([10.0, 12.0, 150.0, 153.0]), max_gap_hours=18)
    assert bool(out["long_gap"].iloc[2])
    assert np.isnan(out["energy"].iloc[2])                    # 138 kWh over 3 days: cannot be dated
    assert out["energy"].iloc[3] == 3.0


def test_counter_skips_missing_readings():
    ts = pd.date_range("2024-03-01 06:00", periods=4, freq="h")
    out = ingest.counter_to_interval(pd.Series(ts), pd.Series([10.0, np.nan, 14.0, 15.0]))
    assert out["energy"].dropna().tolist() == [4.0, 1.0]      # step spans the missing reading


def _systems(capacity=5.0):
    return pd.DataFrame([{"system_id": "A1", "name": "t", "capacity_kwp": capacity, "latitude": 19.0,
                          "longitude": 73.0, "tilt_deg": 15, "azimuth_deg": 180, "commissioning_date": "2020-01-01",
                          "tariff_inr_per_kwh": 7.0, "cleaning_cost_inr": 500.0, "utc_offset_hours": 5.0,
                          "location": ""}]).set_index("system_id", drop=False)


def _hourly_power(days, capacity=5.0):
    """A plain bell-shaped day, sampled hourly, in watts."""
    ts = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")
    shape = np.clip(np.sin(np.pi * (ts.hour.to_numpy() - 6) / 12.0), 0, None)
    return pd.DataFrame({"Timestamp": ts, "AC Power (W)": capacity * 1000 * 0.7 * shape})


def test_gap_and_partial_days_are_flagged_not_filled(cfg):
    days = pd.date_range("2024-03-01", "2024-03-10", freq="D")
    df = _hourly_power(days)
    df = df[df["Timestamp"].dt.normalize() != pd.Timestamp("2024-03-04")]                 # a whole day missing
    partial = (df["Timestamp"].dt.normalize() == pd.Timestamp("2024-03-07")) & (df["Timestamp"].dt.hour >= 11)
    df = df[~partial]                                                                     # afternoon missing
    systems = _systems()
    layout, _ = ingest.infer_layout(df, "A1_export.csv", systems, cfg["ingest"])
    assert layout.kind == "power" and layout.unit == "W" and layout.system_id == "A1"
    sd = ingest.clean_system("A1", [(layout, df)], systems.loc["A1"], cfg["ingest"])
    d = sd.daily
    assert bool(d.loc["2024-03-04", "flag_gap"]) and not bool(d.loc["2024-03-04", "valid"])
    assert bool(d.loc["2024-03-07", "flag_partial"]) and not bool(d.loc["2024-03-07", "valid"])
    assert np.isnan(d.loc["2024-03-04", "energy_kwh"]) and np.isnan(d.loc["2024-03-07", "energy_kwh"])
    assert bool(d.loc["2024-03-05", "valid"]) and d.loc["2024-03-05", "energy_kwh"] > 0
    assert sd.completeness["gap_days"] == 1 and sd.completeness["partial_days"] == 1
    assert list(sd.interval.columns) == ["system_id", "timestamp", "energy_kwh"]


def test_impossible_values_are_rejected(cfg):
    days = pd.date_range("2024-03-01", "2024-03-05", freq="D")
    df = _hourly_power(days)
    df.loc[(df["Timestamp"] == pd.Timestamp("2024-03-03 12:00")), "AC Power (W)"] = 50_000.0   # 10x nameplate
    systems = _systems()
    layout, _ = ingest.infer_layout(df, "A1.csv", systems, cfg["ingest"])
    sd = ingest.clean_system("A1", [(layout, df)], systems.loc["A1"], cfg["ingest"])
    assert bool(sd.daily.loc["2024-03-03", "flag_impossible"]) and not bool(sd.daily.loc["2024-03-03", "valid"])


def test_daily_counter_layout(cfg):
    days = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Total Energy (kWh)": 1000 + np.arange(40) * 20.0})
    systems = _systems()
    layout, _ = ingest.infer_layout(df, "A1_daily.csv", systems, cfg["ingest"])
    assert layout.kind == "cumulative_counter" and layout.unit == "kWh"
    sd = ingest.clean_system("A1", [(layout, df)], systems.loc["A1"], cfg["ingest"])
    assert sd.resolution == "daily"
    assert sd.daily["energy_kwh"].dropna().eq(20.0).all()


def _write_systems(tmp_path, **overrides):
    row = {"system_id": "A1", "name": "t", "capacity_kwp": 5, "latitude": 19.0, "longitude": 73.0, "tilt_deg": 15,
           "azimuth_deg": 180, "commissioning_date": "2020-01-01", "tariff_inr_per_kwh": 7.0, "cleaning_cost_inr": 500}
    row.update(overrides)
    path = tmp_path / "systems.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_orientation_and_commissioning_may_be_blank(tmp_path):
    """A source that does not publish tilt, azimuth or start date is accepted, never guessed."""
    systems, warnings = ingest.load_systems(_write_systems(tmp_path, tilt_deg="", azimuth_deg="", commissioning_date=""))
    assert np.isnan(systems.loc["A1", "tilt_deg"]) and np.isnan(systems.loc["A1", "azimuth_deg"])
    assert warnings == []


@pytest.mark.parametrize("field", ["capacity_kwp", "latitude", "longitude"])
def test_capacity_and_coordinates_are_still_required(tmp_path, field):
    with pytest.raises(ingest.SystemsFileError):
        ingest.load_systems(_write_systems(tmp_path, **{field: ""}))


def test_data_label_with_several_public_sources():
    from trueyield.config import data_label
    sources = [{"name": "Set A", "system_ids": ["1", "2"]}, {"name": "Set B", "system_ids": ["J1"]}]
    assert data_label(sources, "2") == "Public dataset: Set A"
    assert data_label(sources, "J1") == "Public dataset: Set B"
    assert data_label(sources, "mine") == "Real system data"
    assert data_label([], "mine") == data_label(None, "mine") == "Real system data"
