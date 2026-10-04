"""Step 2: expected yield.

Two stages, kept separate on purpose:

1. A *physical model* turns weather into the energy an ideal array of this size
   would produce (daily mode: irradiation x capacity x temperature factor;
   hourly mode: pvlib transposition plus a PVWatts-style DC model).
2. A *self-calibrated reference* rescales that model to what this particular
   system achieves when it is at its best, month by month.

Stage 2 is why the absolute accuracy of satellite irradiance does not matter:
any constant bias in the weather data, the nameplate or the wiring losses is
absorbed by the reference. Only day-to-day *changes* relative to the system's
own best are interpreted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pvlib
from scipy.stats import norm


def robust_sigma(values: np.ndarray | pd.Series) -> float:
    """Standard deviation estimated from the median absolute deviation.

    Unlike the ordinary standard deviation it is not inflated by a few real
    events (an outage week, a dust storm), so it measures noise, not signal.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def day_length_hours(dates: pd.DatetimeIndex, latitude: float) -> np.ndarray:
    """Astronomical day length from latitude and solar declination."""
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + dates.dayofyear.to_numpy()) / 365.0)
    cos_h = np.clip(-np.tan(np.radians(latitude)) * np.tan(decl), -1, 1)
    return 24.0 / np.pi * np.arccos(cos_h)


def daily_model(weather: pd.DataFrame, site: pd.Series, ecfg: dict[str, Any]) -> pd.Series:
    """Daily mode: energy of an ideal array from daily irradiation.

    model = H x P_stc x [1 + gamma x (T_cell - 25)]
    H is irradiation on the horizontal in kWh/m2 (equivalent to hours at
    1 kW/m2), so H x kWp is the energy at standard conditions. Cell temperature
    uses the NOCT rule: air temperature plus a rise proportional to the mean
    daytime irradiance. The tilt gain is not modelled here; it varies smoothly
    with season and is absorbed by the monthly reference.
    """
    hours = day_length_hours(weather.index, float(site["latitude"]))
    mean_irradiance = weather["ghi"] * 1000.0 / hours                     # W/m2 while the sun is up
    t_cell = weather["t2m_c"] + (ecfg["noct_c"] - 20.0) / 800.0 * mean_irradiance
    return weather["ghi"] * float(site["capacity_kwp"]) * (1.0 + ecfg["temp_coeff_per_c"] * (t_cell - 25.0))


def hourly_model(weather_hourly: pd.DataFrame, site: pd.Series, ecfg: dict[str, Any]) -> pd.Series:
    """Hourly mode: pvlib plane-of-array irradiance and a PVWatts DC model.

    Steps: sun position at the middle of each hour; Erbs split of global
    irradiance into beam and diffuse; Hay-Davies transposition onto the array
    plane; NOCT cell temperature; PVWatts DC power with a linear temperature
    coefficient. Returns kWh per hour indexed by local-standard-time hour start.
    Inverter efficiency, wiring and reflection losses are left to the reference.
    """
    lat, lon = float(site["latitude"]), float(site["longitude"])
    mid = (weather_hourly.index + pd.Timedelta(minutes=30)).tz_localize("UTC")
    sun = pvlib.solarposition.get_solarposition(mid, lat, lon)
    ghi = weather_hourly["ghi"].to_numpy()
    split = pvlib.irradiance.erbs(ghi, sun["zenith"].to_numpy(), mid.dayofyear.to_numpy())
    extra = pvlib.irradiance.get_extra_radiation(mid).to_numpy()
    poa = pvlib.irradiance.get_total_irradiance(
        float(site["tilt_deg"]), float(site["azimuth_deg"]),
        sun["apparent_zenith"].to_numpy(), sun["azimuth"].to_numpy(),
        dni=np.asarray(split["dni"]), ghi=ghi, dhi=np.asarray(split["dhi"]),
        dni_extra=extra, model="haydavies")["poa_global"]
    poa = np.where(np.isfinite(ghi), np.nan_to_num(np.asarray(poa), nan=0.0), np.nan)
    t_cell = weather_hourly["t2m_c"].to_numpy() + (ecfg["noct_c"] - 20.0) / 800.0 * poa
    p_dc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, float(site["capacity_kwp"]), ecfg["temp_coeff_per_c"])
    local = weather_hourly.index + pd.Timedelta(hours=float(site["utc_offset_hours"]))
    return pd.DataFrame({"model_kwh": np.clip(p_dc, 0, None),
                         "elevation": sun["apparent_elevation"].to_numpy()}, index=local)


def build_frame(daily: pd.DataFrame, weather: pd.DataFrame, site: pd.Series, ecfg: dict[str, Any],
                model_hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join measured energy, weather and the physical model on one daily table."""
    f = daily[["energy_kwh", "valid"]].join(weather[["ghi", "clear_ghi", "clearness", "t2m_c", "precip_mm"]])
    if model_hourly is not None:
        per_day = model_hourly["model_kwh"].groupby(model_hourly.index.normalize())
        # A day's model is only used when all 24 hours of weather exist.
        f["model_kwh"] = per_day.sum(min_count=1).where(per_day.count() == 24).reindex(f.index)
    else:
        f["model_kwh"] = daily_model(f, site, ecfg)
    return add_performance_index(f, ecfg)


def add_performance_index(f: pd.DataFrame, ecfg: dict[str, Any]) -> pd.DataFrame:
    """Performance index = measured / modelled energy, and the clear-day flag.

    Days that are invalid, too dark to normalise, or lack weather get no index.
    """
    usable = (f["valid"] & f["energy_kwh"].notna() & (f["ghi"] >= ecfg["min_daily_irradiation"])
              & (f["model_kwh"] > 0)).fillna(False)
    f["pi"] = (f["energy_kwh"] / f["model_kwh"]).where(usable)
    f["clear"] = usable & (f["clearness"] >= ecfg["clear_day_min_clearness"]).fillna(False)
    return f


def sustained(series: pd.Series, n: int, min_n: int, max_span_days: int) -> pd.Series:
    """Median over `n` consecutive clear days ("sustained" performance).

    Works on the clear days only, so a cloudy week does not break a window, but
    a window that would stretch over more than `max_span_days` is discarded.
    """
    s = series.dropna()
    if s.empty:
        return s
    med = s.rolling(n, center=True, min_periods=min_n).median()
    ordinal = pd.Series(s.index.map(pd.Timestamp.toordinal), index=s.index, dtype=float)
    span = ordinal.rolling(n, center=True, min_periods=min_n).max() - ordinal.rolling(
        n, center=True, min_periods=min_n).min()
    return med.where(span <= max_span_days)


def relative_residuals(pi_clear: pd.Series, window_days: int, min_clear_days: int) -> pd.Series:
    """Clear-day performance relative to its own running median (two months by default).

    The running median follows season and slow soiling, so what is left is the
    fast scatter of the measurement itself: satellite error, haze, logging.
    """
    daily = pi_clear.asfreq("D")
    trend = daily.rolling(window_days, center=True, min_periods=min_clear_days).median()
    return (daily / trend - 1.0).dropna()


def calibrate_reference(frame: pd.DataFrame, ecfg: dict[str, Any], outage: pd.Series) -> dict[str, Any]:
    """Calibrate the system's own seasonal reference and add it to `frame`.

    For each calendar month the reference is the `reference_quantile` (default
    90th percentile) of sustained clear-day performance, pooled over all years.
    A high quantile of a noisy series sits above the true clean level purely by
    luck, so the expected luck (quantile z-score x measured noise of a sustained
    median) is subtracted. With no real losses this lands on the typical clean
    level; with real losses it errs low, which understates loss rather than
    inventing it.

    Adds columns: reference, expected_kwh, ratio (= performance / reference).
    """
    pi_clear = frame["pi"].where(frame["clear"] & ~outage)
    window = (ecfg["sustained_clear_days"], ecfg["sustained_min_clear_days"], ecfg["sustained_max_span_days"])
    sus = sustained(pi_clear, *window)
    resid = relative_residuals(pi_clear, ecfg["noise_trend_window_days"], ecfg["noise_trend_min_clear_days"])
    sus_resid = sustained(resid, *window)
    se_sustained = robust_sigma(sus_resid)
    q = ecfg["reference_quantile"]
    luck = float(norm.ppf(q) * se_sustained) if np.isfinite(se_sustained) else 0.0

    monthly_raw = pd.Series(np.nan, index=range(1, 13))
    counts = pd.Series(0, index=range(1, 13))
    for month, vals in sus.dropna().groupby(sus.dropna().index.month):
        counts[month] = len(vals)
        if len(vals) >= ecfg["reference_min_days_per_month"]:
            monthly_raw[month] = vals.quantile(q)
    if monthly_raw.notna().sum() == 0:
        raise ValueError("too few clear days to calibrate a reference in any month")
    monthly = monthly_raw * (1.0 - luck)

    frame["reference"] = interpolate_monthly(monthly, frame.index)
    frame["expected_kwh"] = frame["model_kwh"] * frame["reference"]
    frame["ratio"] = frame["pi"] / frame["reference"]
    return {
        "monthly_reference": {int(m): (None if pd.isna(v) else round(float(v), 4)) for m, v in monthly.items()},
        "monthly_clear_day_counts": {int(m): int(c) for m, c in counts.items()},
        "months_without_own_reference": [int(m) for m in monthly.index[monthly.isna()]],
        "sustained_noise_fraction": None if not np.isfinite(se_sustained) else round(se_sustained, 5),
        "luck_allowance_fraction": round(luck, 5),
        "sustained_values": sus,
    }


def interpolate_monthly(monthly: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Spread 12 monthly values over the year without steps at month ends.

    Each monthly value is placed at mid-month and joined by straight lines,
    wrapping from December to January. Months with too few clear days borrow
    from their neighbours through the same interpolation.
    """
    known = monthly.dropna()
    mid_doy = np.array([pd.Timestamp(2001, int(m), 15).dayofyear for m in known.index], dtype=float)
    x = np.concatenate([mid_doy - 365.0, mid_doy, mid_doy + 365.0])
    y = np.tile(known.to_numpy(dtype=float), 3)
    return np.interp(np.minimum(dates.dayofyear.to_numpy(), 365).astype(float), x, y)
