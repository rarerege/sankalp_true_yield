"""Step 9: method validation with an injected, known loss.

This is the ONLY place outside the unit tests where data are altered. A real,
relatively clean stretch is copied, a known soiling ramp and a short outage are
subtracted from the copy, and the unchanged pipeline is run on it. Because the
injected loss is known exactly, the pipeline's answer can be checked.

Every output of this module carries VALIDATION_LABEL. None of it is a field
result and none of it enters the loss, rupee or recovery figures.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import VALIDATION_LABEL
from .expected import add_performance_index

BASE_COLUMNS = ["energy_kwh", "valid", "ghi", "clear_ghi", "clearness", "t2m_c", "precip_mm", "model_kwh"]


def choose_clean_stretch(frame: pd.DataFrame, outage: pd.Series,
                         vcfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Pick the cleanest real stretch: no outages, few gaps, least flagged loss,
    and performance closest to the reference (a stretch sitting well above the
    reference would hide part of any loss injected into it)."""
    best, best_key = None, None
    days = vcfg["stretch_days"]
    for start in range(0, len(frame) - days + 1, vcfg["stretch_step_days"]):
        win = frame.iloc[start:start + days]
        if (outage.iloc[start:start + days].any() or win["valid"].mean() < vcfg["stretch_min_valid_share"]
                or win["clear"].sum() < days * vcfg["stretch_min_clear_share"]):
            continue
        key = (float(win["flagged"].mean()), float((win["smooth_ratio"] - 1.0).abs().mean()))
        if best_key is None or key < best_key:
            best, best_key = (win.index[0], win.index[-1]), key
    return best


def inject(frame: pd.DataFrame, stretch: tuple[pd.Timestamp, pd.Timestamp], vcfg: dict[str, Any],
           min_clearness: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Subtract a known soiling ramp and a short outage from a copy of the data."""
    f = frame[BASE_COLUMNS].copy()
    original = f["energy_kwh"].copy()
    ramp_start = stretch[0] + pd.Timedelta(days=vcfg["ramp_start_offset_days"])
    ramp_days = pd.date_range(ramp_start, periods=vcfg["soiling_ramp_days"], freq="D")
    factor = pd.Series(1.0, index=f.index)
    # Soiling builds linearly to the full fraction, then is "cleaned" in one step.
    factor.loc[ramp_days] = 1.0 - vcfg["soiling_ramp_fraction"] * np.arange(1, len(ramp_days) + 1) / len(ramp_days)
    after = f.loc[ramp_days[-1] + pd.Timedelta(days=vcfg["outage_gap_after_ramp_days"]):stretch[1]]
    sunny = after.index[(after["valid"] & (after["clearness"] >= min_clearness)).to_numpy()]
    outage_days = pd.DatetimeIndex([])
    for i in range(len(sunny) - vcfg["outage_days"] + 1):       # first run of consecutive sunny days
        run = sunny[i:i + vcfg["outage_days"]]
        if (run[-1] - run[0]).days == vcfg["outage_days"] - 1:
            outage_days = run
            break
    factor.loc[outage_days] = 0.0
    f["energy_kwh"] = original * factor
    ramp_kwh = float((original - f["energy_kwh"]).loc[ramp_days].sum())
    outage_kwh = float((original - f["energy_kwh"]).loc[outage_days].sum())
    return f, {"ramp_start": str(ramp_days[0].date()), "ramp_end": str(ramp_days[-1].date()),
               "ramp_peak_pct": 100 * vcfg["soiling_ramp_fraction"],
               "outage_days": [str(d.date()) for d in outage_days],
               "injected_ramp_kwh": round(ramp_kwh, 1), "injected_outage_kwh": round(outage_kwh, 1),
               "injected_kwh": round(ramp_kwh + outage_kwh, 1), "_factor": factor}


def detection_check(frame: pd.DataFrame, stretch: tuple[pd.Timestamp, pd.Timestamp], noise: dict[str, Any],
                    analyse: Callable, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Empirical check of the stated detection limit.

    Step losses of several sizes (multiples of the stated limit) are injected
    at random positions in the clean stretch, for the window length the limit
    refers to. The share of trials in which the pipeline raises a new flag
    inside the injected window is the measured detection rate.
    """
    vcfg = cfg["validation"]
    rng = np.random.default_rng(vcfg["seed"])
    quick = copy.deepcopy(cfg)
    quick["bootstrap"]["samples"] = 0
    base_flag = frame["flagged"].to_numpy()
    out = []
    for w, v in noise["windows"].items():
        if v.get("limit_fraction") is None:
            continue
        span = (stretch[1] - stretch[0]).days - w
        if span <= 0:
            continue
        starts = [stretch[0] + pd.Timedelta(days=int(d)) for d in rng.integers(0, span, vcfg["detection_check_trials"])]
        for multiple in vcfg["detection_check_multiples"]:
            size = multiple * v["limit_fraction"]
            hits = 0
            for s in starts:
                window = pd.date_range(s, periods=w, freq="D")
                f = frame[BASE_COLUMNS].copy()
                f.loc[window, "energy_kwh"] *= (1.0 - size)
                f = add_performance_index(f, cfg["expected"])
                analyse(f, quick)
                new = f["flagged"].to_numpy() & ~base_flag
                hits += bool(new[f.index.isin(window)].any())
            out.append({"window_days": int(w), "injected_step_pct": round(100 * size, 2),
                        "multiple_of_stated_limit": multiple, "trials": len(starts),
                        "detected_share_pct": round(100 * hits / len(starts))})
    return out


def reliable_detection(check: list[dict[str, Any]], power: float) -> dict[str, Any]:
    """Smallest injected step that was detected in at least `power` of trials."""
    out: dict[str, Any] = {}
    for w in sorted({c["window_days"] for c in check}):
        rows = sorted((c for c in check if c["window_days"] == w), key=lambda c: c["injected_step_pct"])
        hit = next((c for c in rows if c["detected_share_pct"] >= 100 * power
                    and all(d["detected_share_pct"] >= 100 * power for d in rows[rows.index(c):])), None)
        out[str(w)] = ({"step_pct": hit["injected_step_pct"], "detected_share_pct": hit["detected_share_pct"]}
                       if hit else {"step_pct": None, "note": f"not reached within {rows[-1]['injected_step_pct']}%"})
    return out


def validate(frame: pd.DataFrame, analysis: dict[str, Any], analyse: Callable, cfg: dict[str, Any],
             system_id: str) -> dict[str, Any]:
    """Inject a known loss into a clean real stretch and check it is recovered."""
    vcfg = cfg["validation"]
    stretch = choose_clean_stretch(frame, analysis["outage"], vcfg)
    if stretch is None:
        return {"label": VALIDATION_LABEL, "system_id": system_id, "available": False,
                "reason": f"no {vcfg['stretch_days']}-day stretch without outages and with enough valid, clear days"}
    from .detect import estimate_loss
    baseline = estimate_loss(frame, analysis["outage"], analysis["noise"], analysis["ref_info"], cfg, period=stretch)
    modified, info = inject(frame, stretch, vcfg, cfg["detect"]["outage_min_clearness"])
    modified = add_performance_index(modified, cfg["expected"])
    after = analyse(modified, cfg, period=stretch)
    est = after["loss"]
    truth = baseline["lost_kwh"] + info["injected_kwh"]
    lo, hi = est.get("lost_kwh_ci", (np.nan, np.nan))
    found = [d for d in info["outage_days"] if bool(after["outage"].get(pd.Timestamp(d), False))]
    recovered = est["lost_kwh"] - baseline["lost_kwh"]
    result = {
        "label": VALIDATION_LABEL, "system_id": system_id, "available": True,
        "stretch": [str(stretch[0].date()), str(stretch[1].date())],
        "baseline_before_injection": {"lost_kwh": round(baseline["lost_kwh"], 1),
                                      "lost_kwh_ci": [round(x, 1) for x in baseline.get("lost_kwh_ci", (np.nan, np.nan))],
                                      "loss_pct": round(baseline["loss_pct"], 2)},
        "injected": {k: v for k, v in info.items() if not k.startswith("_")},
        "estimated_after_injection": {"lost_kwh": round(est["lost_kwh"], 1), "lost_kwh_ci": [round(lo, 1), round(hi, 1)],
                                      "sustained_kwh": round(est["sustained_kwh"], 1),
                                      "outage_kwh": round(est["outage_kwh"], 1)},
        "recovered_kwh": round(recovered, 1),
        "recovered_share_of_injected_pct": round(100 * recovered / info["injected_kwh"], 1) if info["injected_kwh"] else None,
        "injected_within_stated_range": bool(lo <= truth <= hi),
        "outage_days_detected": f"{len(found)} of {len(info['outage_days'])}",
        "detection_limit_check": (check := detection_check(frame, stretch, analysis["noise"], analyse, cfg)),
        "reliable_detection": reliable_detection(check, cfg["detect"]["power"]),
        "_frames": {"original": frame, "modified": modified, "factor": info["_factor"], "stretch": stretch},
    }
    return result
