"""Step 5: likely causes, as rules.

Every entry keeps two things apart:
  facts       what the data show (counts, dates, measured rates);
  hypothesis  what that pattern is consistent with.
A hypothesis is never proof of cause. Site inspection is needed to confirm.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .detect import _block_indices, percentile_interval

try:                                   # rdtools is optional
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import rdtools
        from rdtools import soiling as rd_soiling
    HAVE_RDTOOLS = True
except Exception:                      # pragma: no cover - depends on the environment
    HAVE_RDTOOLS = False


def _runs(days: pd.DatetimeIndex) -> list[str]:
    """Collapse dates into 'start to end (n days)' strings."""
    out, start, prev = [], None, None
    for d in days:
        if start is None:
            start = prev = d
        elif (d - prev).days == 1:
            prev = d
        else:
            out.append((start, prev))
            start = prev = d
    if start is not None:
        out.append((start, prev))
    return [f"{a.date()}" if a == b else f"{a.date()} to {b.date()} ({(b - a).days + 1} days)" for a, b in out]


def outage_cause(frame: pd.DataFrame, outage: pd.Series, daily: pd.DataFrame, loss: dict[str, Any],
                 cfg: dict[str, Any]) -> dict[str, Any]:
    """Outage or fault: near-zero output on days with good irradiance, or a
    sustained drop so deep that part of the array must have been off."""
    deep_cut = cfg["detect"]["partial_outage_min_fraction"]
    days = frame.index[outage]
    deep = frame.index[(frame["flagged"] & (frame["loss_fraction"] > deep_cut) & ~outage).to_numpy()]
    gap_days = int(daily["flag_gap"].sum())
    facts = []
    if len(days):
        facts.append(f"Output was near zero on {len(days)} day(s) with good irradiance: "
                     + "; ".join(_runs(days)[:12]) + ("; ..." if len(_runs(days)) > 12 else "") + ".")
    else:
        facts.append("No day with good irradiance had near-zero output.")
    if gap_days:
        facts.append(f"{gap_days} day(s) have no usable generation data (nothing logged, or a reading "
                     "missing from at least one inverter); whether the system produced on those days is "
                     "unknown and they are not counted as loss.")
    if len(deep):
        level = float(frame.loc[deep, "smooth_ratio"].median())
        facts.append(f"Sustained output was more than {100 * deep_cut:.0f}% below the reference on {len(deep)} "
                     f"day(s) (median level {100 * level:.0f}% of reference): " + "; ".join(_runs(deep)[:8])
                     + ("; ..." if len(_runs(deep)) > 8 else "") + f". This accounts for "
                     f"{loss['deep_shortfall_kwh']:,.0f} kWh of the sustained shortfall.")
    cold = int((frame.loc[days, "t2m_c"] < cfg["causes"]["snow_possible_below_c"]).sum()) if len(days) else 0
    parts = []
    if len(days):
        parts.append("Near-zero days are consistent with an inverter trip, grid outage or a monitoring fault "
                     "that recorded zeros. The data cannot tell these apart.")
        if cold:
            parts.append(f"Near-freezing temperatures on {cold} of these days mean snow cover is also possible.")
    if len(deep):
        parts.append("The deep sustained drop is consistent with part of the system being off (one inverter, "
                     "string or meter channel), which is too large and too abrupt for soiling.")
    return {"cause": "outage or fault", "status": "indicated" if (len(days) or len(deep)) else "not indicated",
            "facts": facts, "hypothesis": " ".join(parts) or None, "outage_days": int(len(days)),
            "deep_shortfall_days": int(len(deep))}


def _simple_soiling(ratio: pd.Series, insolation: pd.Series, precip: pd.Series, clean_dates: list[pd.Timestamp],
                    rain_mm: float, reps: int, confidence: float, rng: np.random.Generator, ccfg: dict[str, Any]) -> dict[str, Any]:
    """Fallback when rdtools is unavailable: a plain rate-and-recovery estimate.

    The clear-day ratio is cut into intervals at every heavy-rain day and every
    recorded cleaning. In each interval a Theil-Sen (median-of-slopes) line
    gives the soiling rate. The soiling ratio is 1 at the start of an interval
    and falls along that line; rising or flat intervals count as clean. The
    interval comes from re-drawing each slope within its own confidence band.
    """
    cuts = sorted(set(precip.index[precip >= rain_mm]) | set(clean_dates))
    edges = [ratio.index[0]] + cuts + [ratio.index[-1] + pd.Timedelta(days=1)]
    intervals = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = ratio.loc[a:b - pd.Timedelta(days=1)].dropna()
        if (len(seg) >= ccfg["fallback_soiling_min_points"]
                and (seg.index[-1] - seg.index[0]).days >= ccfg["fallback_soiling_min_days"]):
            x = (seg.index - seg.index[0]).days.to_numpy(dtype=float)
            slope, _, lo, hi = theilslopes(seg.to_numpy(), x, 0.68)
            intervals.append((a, b, slope / np.median(seg), lo / np.median(seg), hi / np.median(seg)))
    if not intervals:
        return {"available": False, "reason": "no interval between cleaning events was long enough"}
    w = insolation.reindex(ratio.index).fillna(0.0)

    def profile(slopes: np.ndarray) -> float:
        sr = pd.Series(1.0, index=ratio.index)
        for (a, b, *_), s in zip(intervals, slopes):
            idx = sr.loc[a:b - pd.Timedelta(days=1)].index
            sr.loc[idx] = np.clip(1.0 + min(s, 0.0) * (idx - a).days.to_numpy(), 0.0, 1.0)
        return float((sr * w).sum() / w.sum())

    point = profile(np.array([i[2] for i in intervals]))
    sd = np.array([max((i[4] - i[3]) / 2.0, 1e-9) for i in intervals])
    draws = [profile(np.array([i[2] for i in intervals]) + rng.normal(0, sd)) for _ in range(reps)]
    rates = np.array([i[2] for i in intervals])
    return {"available": True, "method": "simple rate-and-recovery fallback (Theil-Sen slopes between heavy rain "
                                         "or recorded cleanings; cruder than rdtools SRR and usually lower)",
            "soiling_ratio": point, "soiling_ratio_ci": percentile_interval(np.array(draws), confidence),
            "intervals": len(intervals), "declining_intervals": int((rates < 0).sum()),
            "median_rate_pct_per_day": float(100 * np.median(rates[rates < 0])) if (rates < 0).any() else 0.0}


def _srr_cached(ratio: pd.Series, ghi: pd.Series, precip: pd.Series, reps: int, confidence: float, seed: int,
                cache_dir: Path | None) -> dict[str, Any]:
    """Run rdtools SRR, caching the result because its Monte Carlo is slow.

    The cache key is a hash of the exact inputs, so any change to the data or
    settings recomputes; an unchanged rerun reads the stored numbers.
    """
    key = hashlib.sha1(b"|".join([ratio.to_numpy().tobytes(), ghi.to_numpy().tobytes(), precip.to_numpy().tobytes(),
                                  f"{reps},{confidence},{seed},{rdtools.__version__}".encode()])).hexdigest()[:16]
    path = cache_dir / f"rdtools_srr_{key}.json" if cache_dir else None
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    np.random.seed(seed)                              # rdtools draws from numpy's global generator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, ci, info = rd_soiling.soiling_srr(ratio, ghi, reps=reps, precipitation_daily=precip,
                                              confidence_level=100 * confidence)
    summary = info["soiling_interval_summary"]
    valid = summary[summary["valid"]]
    declining = valid[valid["soiling_rate"] < 0]
    est = {"available": True, "method": f"rdtools {rdtools.__version__} stochastic rate and recovery (SRR)",
           "soiling_ratio": float(sr), "soiling_ratio_ci": [float(ci[0]), float(ci[1])],
           "intervals": int(len(valid)), "declining_intervals": int(len(declining)),
           "median_rate_pct_per_day": float(100 * declining["soiling_rate"].median()) if len(declining) else 0.0}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(est), encoding="utf-8")
    return est


def soiling_cause(frame: pd.DataFrame, outage: pd.Series, recovery: dict[str, Any], clean_dates: list[pd.Timestamp],
                  cfg: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    """Soiling: gradual decline with step recoveries after rain or cleaning.

    Uses rdtools' stochastic rate-and-recovery (SRR) method when available. SRR
    finds cleaning events as upward steps, fits a decline between them, and
    Monte-Carlo samples the fits to give a soiling ratio with a confidence
    interval (1.0 = no soiling loss).

    SRR looks for upward steps, and day-to-day scatter alone produces steps. So
    the estimate is compared with a shuffle check: the same real clear-day
    values are put in random time order, which destroys any soiling pattern but
    keeps the scatter, and SRR is run again. Soiling is called "indicated" only
    if the low end of the real estimate is above every shuffled result.
    """
    ccfg, bcfg = cfg["causes"], cfg["bootstrap"]
    ratio = frame["ratio"].where(frame["clear"] & ~outage)
    facts: list[str] = []
    numbers: dict[str, Any] = {}
    est: dict[str, Any] = {"available": False, "reason": "not run"}
    if ratio.notna().sum() < ccfg["soiling_min_clear_days"]:
        return {"cause": "soiling", "status": "insufficient data", "hypothesis": None,
                "facts": [f"Fewer than {ccfg['soiling_min_clear_days']} clear days: a soiling pattern cannot be "
                          "assessed."]}
    if HAVE_RDTOOLS:
        try:
            est = _srr_cached(ratio.asfreq("D"), frame["ghi"].asfreq("D"),
                              frame["precip_mm"].asfreq("D").fillna(0.0), ccfg["soiling_reps"],
                              bcfg["confidence"], bcfg["seed"], cache_dir)
        except Exception as exc:
            facts.append(f"rdtools SRR could not be run on this series ({type(exc).__name__}: {exc}); "
                         "the simple fallback was used.")
    shuffled: list[float] = []
    if est["available"]:
        clear_pos = np.flatnonzero(ratio.notna().to_numpy())
        rng = np.random.default_rng(bcfg["seed"])
        for _ in range(ccfg["soiling_shuffle_runs"]):
            mixed = ratio.copy()
            mixed.iloc[clear_pos] = rng.permutation(ratio.to_numpy()[clear_pos])
            try:
                null = _srr_cached(mixed.asfreq("D"), frame["ghi"].asfreq("D"),
                                   frame["precip_mm"].asfreq("D").fillna(0.0), ccfg["soiling_shuffle_reps"],
                                   bcfg["confidence"], bcfg["seed"], cache_dir)
                shuffled.append(100 * (1 - null["soiling_ratio"]))
            except Exception:
                continue
    if not est["available"]:
        est = _simple_soiling(ratio.dropna(), frame["ghi"], frame["precip_mm"].fillna(0.0), clean_dates,
                              cfg["verify"]["heavy_rain_mm"], ccfg["fallback_soiling_reps"], bcfg["confidence"],
                              np.random.default_rng(bcfg["seed"]), ccfg)
    if not est["available"]:
        return {"cause": "soiling", "status": "insufficient data", "hypothesis": None,
                "facts": facts + [f"Soiling could not be estimated: {est['reason']}."]}

    lo, hi = est["soiling_ratio_ci"]
    numbers = {"method": est["method"], "soiling_ratio": round(est["soiling_ratio"], 4),
               "soiling_ratio_ci": [round(lo, 4), round(hi, 4)],
               "soiling_loss_pct": round(100 * (1 - est["soiling_ratio"]), 2),
               "soiling_loss_pct_ci": [round(100 * (1 - hi), 2), round(100 * (1 - lo), 2)],
               "confidence": bcfg["confidence"]}
    facts.append(f"Between cleaning events, performance declined in {est['declining_intervals']} of "
                 f"{est['intervals']} usable intervals, at a median of "
                 f"{abs(est['median_rate_pct_per_day']):.3f}% per day in the declining ones.")
    tested = [e for e in recovery["events"] if e.get("testable")]
    rises = [e for e in tested if e["distinguishable_from_zero"] and e["change_pct"] > 0]
    if tested:
        facts.append(f"{len(rises)} of {len(tested)} testable {'natural rain' if 'rain' in recovery['event_source'] else 'dated maintenance'} "
                     "event(s) were followed by a rise in normalised performance that is "
                     "distinguishable from zero.")
    estimate = (f"{numbers['soiling_loss_pct']:.1f}% (range {numbers['soiling_loss_pct_ci'][0]:.1f}–"
                f"{numbers['soiling_loss_pct_ci'][1]:.1f}%, {est['method']})")
    floor = ccfg["soiling_min_loss_pct"]
    if shuffled:
        numbers["shuffled_order_loss_pct"] = [round(v, 2) for v in shuffled]
        floor = max(floor, max(shuffled))
        facts.append(f"rdtools estimates a soiling loss of {estimate}. Shuffle check: with the same clear-day values "
                     f"in random time order, which contain no soiling pattern, it reports "
                     f"{min(shuffled):.1f}–{max(shuffled):.1f}%.")
    indicated = numbers["soiling_loss_pct_ci"][0] > floor
    if indicated:
        hyp = f"Pattern consistent with soiling: estimated average soiling loss {estimate}."
        if shuffled:
            hyp += (f" The low end of that range is above the shuffle check ({max(shuffled):.1f}%), so the pattern is "
                    f"more than scatter; roughly {max(numbers['soiling_loss_pct'] - float(np.median(shuffled)), 0):.1f} "
                    "percentage points exceed what scatter alone produces.")
    elif shuffled:
        hyp = (f"No soiling distinguishable from scatter. The estimate of {estimate} is not clearly above what the "
               f"same method reports on shuffled data (up to {max(shuffled):.1f}%), so most or all of it can be "
               "produced by day-to-day scatter alone.")
    else:
        hyp = ("No soiling loss distinguishable from noise: the lower end of the estimated range is "
               f"below {ccfg['soiling_min_loss_pct']:g}%.")
    return {"cause": "soiling", "status": "indicated" if indicated else "not indicated",
            "facts": facts, "hypothesis": hyp, "numbers": numbers}


def degradation_cause(frame: pd.DataFrame, outage: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Ageing: a slow multi-year trend, measured year on year.

    Each clear day is compared with the same calendar day one year later. This
    cancels the seasons and largely cancels soiling that repeats every year,
    leaving the change that neither rain nor cleaning reversed. The median of
    all such pairs is the rate; its interval is a bootstrap.
    """
    ccfg, bcfg = cfg["causes"], cfg["bootstrap"]
    pi = frame["pi"].where(frame["clear"] & ~outage)
    span_years = (frame.index[-1] - frame.index[0]).days / 365.25
    if span_years <= ccfg["degradation_min_years"]:
        return {"cause": "ageing or degradation", "status": "insufficient data", "hypothesis": None,
                "facts": [f"Only {span_years:.1f} years of data; more than {ccfg['degradation_min_years']:g} "
                          "years are needed to separate ageing from seasonal effects."]}
    rate = ci = None
    method = ""
    if HAVE_RDTOOLS:
        try:
            np.random.seed(bcfg["seed"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rd, rd_ci, _ = rdtools.degradation_year_on_year(
                    pi.dropna().tz_localize("UTC"), confidence_level=100 * bcfg["confidence"])
            rate, ci, method = float(rd), (float(rd_ci[0]), float(rd_ci[1])), f"rdtools {rdtools.__version__} year-on-year"
        except Exception:
            rate = None
    if rate is None:
        daily = pi.asfreq("D")
        yoy = (100 * (daily.shift(-365) / daily - 1.0)).dropna().to_numpy()
        if len(yoy) < ccfg["degradation_min_pairs"]:
            return {"cause": "ageing or degradation", "status": "insufficient data", "hypothesis": None,
                    "facts": ["Too few clear-day pairs one year apart to measure a trend."]}
        rng = np.random.default_rng(bcfg["seed"])
        idx = _block_indices(len(yoy), ccfg["degradation_block_days"], int(bcfg["samples"]), rng)
        rate, ci = float(np.median(yoy)), percentile_interval(np.median(yoy[idx], axis=1), bcfg["confidence"])
        method = "year-on-year median (built-in fallback)"
    facts = [f"Year-on-year change of clear-day performance over {span_years:.1f} years: "
             f"{rate:+.2f}% per year (range {ci[0]:+.2f} to {ci[1]:+.2f}% per year, {method})."]
    declining = ci[1] < 0
    hyp = ("Pattern consistent with gradual ageing or another slow loss not reversed by rain: "
           "the whole range is below zero." if declining else
           "No ageing trend distinguishable from zero: the range includes zero or is positive.")
    return {"cause": "ageing or degradation", "status": "indicated" if declining else "not indicated",
            "facts": facts, "hypothesis": hyp,
            "numbers": {"rate_pct_per_year": round(rate, 2), "rate_pct_per_year_ci": [round(ci[0], 2), round(ci[1], 2)],
                        "method": method, "confidence": bcfg["confidence"]}}


def shading_cause(frame: pd.DataFrame, outage: pd.Series, hourly: pd.Series | None,
                  model_hourly: pd.DataFrame | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Shading: recurring dips at the same hours of the day (hourly data only).

    On clear days each hour's measured/modelled ratio is divided by the same
    day's midday ratio, which removes soiling and anything else that scales the
    whole day. What remains is the *shape* of the day. An hour that sits well
    below midday on most clear days of a month is a recurring dip.
    """
    ccfg = cfg["causes"]
    if hourly is None or model_hourly is None:
        return {"cause": "shading", "status": "cannot be assessed", "hypothesis": None,
                "facts": ["Cannot be assessed from daily data: shading shows as dips at fixed hours."]}
    clear_days = frame.index[(frame["clear"] & ~outage).to_numpy()]
    h = pd.DataFrame({"actual": hourly, "model": model_hourly["model_kwh"],
                      "elevation": model_hourly["elevation"]}).dropna()
    h = h[h.index.normalize().isin(clear_days) & (h["elevation"] >= ccfg["shading_min_sun_elevation_deg"])
          & (h["model"] > 0)]
    if h.empty:
        return {"cause": "shading", "status": "insufficient data", "hypothesis": None,
                "facts": ["No complete clear-day hours with the sun high enough to test."]}
    h["r"] = h["actual"] / h["model"]
    day = h.index.normalize()
    # Midday level: the few hours with the highest modelled output each day.
    rank = h.groupby(day)["model"].rank(ascending=False, method="first")
    midday = h["r"].where(rank <= ccfg["shading_midday_hours"]).groupby(day).transform("median")
    h["shape"] = h["r"] / midday
    h["month"], h["hour"] = h.index.month, h.index.hour

    # Clock check: do measured and modelled days peak at the same time?
    hours = h.index.hour + 0.5
    centroid = lambda col: (h[col] * hours).groupby(day).sum() / h[col].groupby(day).sum()
    daily_offset = 60 * (centroid("actual") - centroid("model"))
    offset_min = float(daily_offset.median())
    summer = float(daily_offset[daily_offset.index.month.isin(ccfg["clock_summer_months"])].median())
    winter = float(daily_offset[daily_offset.index.month.isin(ccfg["clock_winter_months"])].median())
    facts = [f"Clock check: on clear days the measured daily profile is centred {offset_min:+.0f} minutes "
             f"from the modelled one ({summer:+.0f} min in May-August, {winter:+.0f} min in November-February)."]
    seasonal_max = ccfg["clock_max_seasonal_difference_minutes"]
    clock_ok = abs(offset_min) <= ccfg["clock_max_offset_minutes"] and abs(summer - winter) <= seasonal_max
    if not clock_ok:
        facts.append("The timestamps do not line up with solar time well enough for an hour-by-hour test"
                     + (" (the summer-winter difference suggests the logger follows daylight-saving time)"
                        if abs(summer - winter) > seasonal_max else "") + ".")
        return {"cause": "shading", "status": "cannot be assessed", "facts": facts,
                "numbers": {"clock_offset_minutes": round(offset_min), "summer_offset_minutes": round(summer),
                            "winter_offset_minutes": round(winter)},
                "hypothesis": None}

    g = h.groupby(["month", "hour"])["shape"]
    table = pd.DataFrame({"median_shape": g.median(), "days": g.count(),
                          "share_low": g.apply(lambda s: float((s < 1 - ccfg["shading_dip_fraction"]).mean()))})
    dips = table[(table["days"] >= ccfg["shading_min_clear_days"])
                 & (table["share_low"] >= ccfg["shading_min_share_of_days"])].reset_index()
    numbers = {"clock_offset_minutes": round(offset_min), "recurring_dips": [
        {"month": int(r.month), "hour": f"{int(r.hour):02d}:00-{int(r.hour) + 1:02d}:00",
         "median_dip_pct": round(100 * (1 - r.median_shape), 1), "share_of_clear_days_pct": round(100 * r.share_low),
         "clear_days": int(r.days)} for r in dips.itertuples()]}
    if dips.empty:
        facts.append(f"No hour of the day (sun above {ccfg['shading_min_sun_elevation_deg']}°) was at least "
                     f"{100 * ccfg['shading_dip_fraction']:.0f}% below the midday level on "
                     f"{100 * ccfg['shading_min_share_of_days']:.0f}% of clear days in any month.")
        return {"cause": "shading", "status": "not indicated", "facts": facts, "numbers": numbers,
                "hypothesis": "No recurring hour-of-day dip found. Shading at very low sun is not tested."}
    by_hour = dips.groupby("hour").agg(months=("month", lambda m: sorted(m)), dip=("median_shape", "median"))
    for hour, r in by_hour.iterrows():
        facts.append(f"{int(hour):02d}:00-{int(hour) + 1:02d}:00: output was a median {100 * (1 - r.dip):.0f}% "
                     f"below the midday level on most clear days in month(s) {r.months}.")
    side = "morning" if by_hour.index.max() < 12 else "afternoon" if by_hour.index.min() >= 12 else "both ends of the day"

    # Symmetry check. The model's weaknesses at low sun (reflection, the
    # beam/diffuse split) depend on sun height, not on the time of day, so they
    # lower morning and afternoon alike. A real obstruction sits on one side.
    peak = pd.DatetimeIndex(pd.Series(day).map(h["model"].groupby(day).idxmax()))
    low_sun = (h["elevation"] < ccfg["shading_min_sun_elevation_deg"] + ccfg["shading_symmetry_band_deg"]).to_numpy()
    am = float(h["shape"][low_sun & (h.index < peak)].median())
    pm = float(h["shape"][low_sun & (h.index > peak)].median())
    numbers["low_sun_level_morning"], numbers["low_sun_level_afternoon"] = round(am, 3), round(pm, 3)
    band = (f"{ccfg['shading_min_sun_elevation_deg']}–"
            f"{ccfg['shading_min_sun_elevation_deg'] + ccfg['shading_symmetry_band_deg']}°")
    facts.append(f"Symmetry check: with the sun {band} up, output is at {100 * am:.0f}% of the midday level in the "
                 f"morning and {100 * pm:.0f}% in the afternoon.")
    if np.isfinite(am) and np.isfinite(pm) and abs(am - pm) >= ccfg["shading_symmetry_min_difference"]:
        lower = "afternoon" if pm < am else "morning"
        hyp = (f"Pattern consistent with shading or a horizon obstruction in the {lower}. A modelling error at low sun "
               f"would lower morning and afternoon alike; here the {lower} is clearly lower at the same sun height, "
               "which points to the site: an obstruction on that side, or an array facing slightly away from the "
               "stated azimuth. It needs a site check.")
    else:
        hyp = (f"Recurring dips in the {side}, but morning and afternoon are equally low at the same sun height. "
               "That is what a modelling error at low sun looks like (reflection losses, the beam/diffuse split), so "
               "shading is not established.")
        return {"cause": "shading", "status": "not indicated", "facts": facts, "hypothesis": hyp, "numbers": numbers}
    return {"cause": "shading", "status": "indicated", "facts": facts, "hypothesis": hyp, "numbers": numbers}
