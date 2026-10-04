"""Step 6: recovery verification around dated events.

For each event the weather-normalised performance (`ratio`) on clear days in
the window before is compared with the window after. Because `ratio` is already
corrected for irradiance, temperature and season, a change between the windows
is a change in the system, not in the weather. The uncertainty comes from a
bootstrap over the clear days on each side.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from .config import NATURAL_RAIN_LABEL
from .detect import percentile_interval


def natural_rain_events(frame: pd.DataFrame, system_id: str, vcfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Heavy-rain spells, used as natural cleaning events when no dated events exist.

    Consecutive heavy-rain days form one spell; the before window ends the day
    before the spell and the after window starts the day after it.
    """
    wet = frame.index[frame["precip_mm"] >= vcfg["heavy_rain_mm"]]
    events: list[dict[str, Any]] = []
    for day in wet:
        if events and (day - events[-1]["end"]).days <= 1:
            events[-1]["end"] = day
            events[-1]["rain_mm"] += float(frame.loc[day, "precip_mm"])
        else:
            events.append({"system_id": system_id, "start": day, "end": day, "type": "natural rain",
                           "rain_mm": float(frame.loc[day, "precip_mm"]), "label": NATURAL_RAIN_LABEL,
                           "note": ""})
    return events


def dated_events(events: pd.DataFrame, system_id: str) -> list[dict[str, Any]]:
    """Maintenance events for this system from data/events.csv."""
    rows = events[events["system_id"].astype(str) == str(system_id)] if len(events) else events
    return [{"system_id": system_id, "start": r["date"], "end": r["date"], "type": r["type"],
             "label": "dated maintenance event", "note": str(r.get("note", ""))} for _, r in rows.iterrows()]


def _resampled_means(values: np.ndarray, reps: int, confidence: float, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap means with a small-sample correction.

    With only a handful of clear days a plain bootstrap is too narrow: it
    underestimates the variance by (n-1)/n and uses normal instead of Student-t
    tails. Simulation at n = 4 to 12 gave 85-92% real coverage for a nominal
    95% interval. Stretching each resampled mean's distance from the sample
    mean by sqrt(n/(n-1)) x t/z restores 95-97%.
    """
    n = len(values)
    means = values[rng.integers(0, n, (reps, n))].mean(axis=1)
    tail = 0.5 + confidence / 2.0
    stretch = np.sqrt(n / (n - 1.0)) * student_t.ppf(tail, n - 1) / norm.ppf(tail)
    return values.mean() + (means - values.mean()) * stretch


def check_event(frame: pd.DataFrame, outage: pd.Series, event: dict[str, Any], vcfg: dict[str, Any],
               bcfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Before/after comparison for one event, with a bootstrap interval."""
    ratio = frame["ratio"].where(frame["clear"] & ~outage)
    b0, b1 = event["start"] - pd.Timedelta(days=vcfg["window_days_before"]), event["start"] - pd.Timedelta(days=1)
    a0, a1 = event["end"] + pd.Timedelta(days=1), event["end"] + pd.Timedelta(days=vcfg["window_days_after"])
    before, after = ratio.loc[b0:b1].dropna(), ratio.loc[a0:a1].dropna()
    out: dict[str, Any] = {
        "system_id": event["system_id"], "type": event["type"], "label": event["label"],
        "date": str(event["end"].date()),
        "start": str(event["start"].date()), "end": str(event["end"].date()),
        "note": event.get("note", ""), "clear_days_before": int(len(before)), "clear_days_after": int(len(after)),
        "window_days": [vcfg["window_days_before"], vcfg["window_days_after"]],
    }
    if "rain_mm" in event:
        out["rain_mm"] = round(event["rain_mm"], 1)
        others = frame["precip_mm"].loc[b0:a1].drop(pd.date_range(event["start"], event["end"]), errors="ignore")
        out["other_heavy_rain_in_windows"] = int((others >= vcfg["heavy_rain_mm"]).sum())
    need = vcfg["min_clear_days_each_side"]
    if len(before) < need or len(after) < need:
        out.update(testable=False, reason=f"fewer than {need} clear, fault-free days on one side "
                                          f"({len(before)} before, {len(after)} after)")
        return out

    scale_kwh = float(frame["expected_kwh"].loc[b0:a1].mean())      # typical expected kWh/day here
    mb, ma = float(before.mean()), float(after.mean())
    reps = int(bcfg["samples"])
    bb = _resampled_means(before.to_numpy(), reps, bcfg["confidence"], rng)
    aa = _resampled_means(after.to_numpy(), reps, bcfg["confidence"], rng)
    pct = 100.0 * (aa / bb - 1.0)
    lo, hi = percentile_interval(pct, bcfg["confidence"])
    klo, khi = percentile_interval((aa - bb) * scale_kwh, bcfg["confidence"])
    out.update(
        testable=True, ratio_before=round(mb, 4), ratio_after=round(ma, 4),
        change_pct=round(100.0 * (ma / mb - 1.0), 2), change_pct_ci=[round(lo, 2), round(hi, 2)],
        change_kwh_per_day=round((ma - mb) * scale_kwh, 2), change_kwh_per_day_ci=[round(klo, 2), round(khi, 2)],
        confidence=bcfg["confidence"],
        distinguishable_from_zero=bool(lo > 0 or hi < 0),
        before_days=[str(d.date()) for d in before.index], after_days=[str(d.date()) for d in after.index],
    )
    return out


def verify_events(frame: pd.DataFrame, outage: pd.Series, events: pd.DataFrame, system_id: str,
                  cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the recovery test for dated events, or for natural rain if none exist."""
    vcfg, bcfg = cfg["verify"], cfg["bootstrap"]
    rng = np.random.default_rng(bcfg["seed"] + 1)
    chosen = dated_events(events, system_id)
    kind = "dated maintenance events"
    if not chosen:
        chosen = natural_rain_events(frame, system_id, vcfg)
        kind = NATURAL_RAIN_LABEL
    results = [check_event(frame, outage, ev, vcfg, bcfg, rng) for ev in chosen]
    return {"event_source": kind, "events": results, "clearest": clearest_event(results)}


def clearest_event(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The best-observed event: most clear days on its weaker side, then the
    narrowest interval. Chosen on data quality only, never on effect size, so
    that the headline event is not cherry-picked for a large recovery."""
    ok = [r for r in results if r.get("testable")]
    if not ok:
        return None
    return max(ok, key=lambda r: (min(r["clear_days_before"], r["clear_days_after"]),
                                  -(r["change_pct_ci"][1] - r["change_pct_ci"][0])))
