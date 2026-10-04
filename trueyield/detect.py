"""Steps 3 and 4: detect sustained underperformance, size it, and state the limit.

Everything here works on `ratio` = weather-normalised performance divided by
the system's own seasonal reference (1.0 = performing at its sustained best).
Only clear days are used to judge the state of the system, because that is
where satellite irradiance is most trustworthy. The state found on clear days
is then applied to the expected energy of all days in between.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .expected import interpolate_monthly, relative_residuals, robust_sigma


def odd(n: int) -> int:
    """Centred windows need an odd length."""
    return int(n) if int(n) % 2 else int(n) + 1


def find_outages(frame: pd.DataFrame, dcfg: dict[str, Any]) -> pd.Series:
    """Days with near-zero output although the sky delivered good irradiance.

    A fact, not a hypothesis: the data are valid, the sun shone, the meter did
    not move. (Why it did not move is a hypothesis; see causes.py.)
    """
    typical = np.nanmedian(frame["pi"].where(frame["clear"]))
    return (frame["valid"] & (frame["clearness"] >= dcfg["outage_min_clearness"])
            & (frame["pi"] < dcfg["outage_max_fraction_of_expected"] * typical)).fillna(False)


def noise_profile(frame: pd.DataFrame, outage: pd.Series, dcfg: dict[str, Any],
                  ecfg: dict[str, Any]) -> dict[str, Any]:
    """Measure the noise floor and derive the detection limit (step 4).

    For each window length W we ask: when nothing slow is happening, how much
    does a W-day median of clear-day performance wobble? This is measured
    directly on the data (median of every W-day window, after removing the slow
    trend with a running median at least three times longer), so day-to-day
    correlation and the real number of clear days are included automatically.

      threshold_W = max(floor, z(1 - false_alarm_rate) x SE_W)
      limit_W     = threshold_W + z(power) x SE_W

    A true loss equal to limit_W is flagged with probability `power`; anything
    smaller cannot be reliably told apart from noise.
    """
    pi_clear = frame["pi"].where(frame["clear"] & ~outage)
    z_alpha = float(norm.ppf(1.0 - dcfg["false_alarm_rate"]))
    z_beta = float(norm.ppf(dcfg["power"]))
    base, min_clear = ecfg["noise_trend_window_days"], ecfg["noise_trend_min_clear_days"]
    day_sigma = robust_sigma(relative_residuals(pi_clear, base, min_clear))
    windows = {}
    for w in dcfg["detection_windows_days"]:
        trend_days = max(base, odd(dcfg["noise_trend_window_multiple"] * w))
        resid = relative_residuals(pi_clear, trend_days, min_clear).asfreq("D")
        medians = resid.rolling(odd(w), center=True, min_periods=dcfg["smoothing_min_clear_days"]).median()
        se = robust_sigma(medians)
        if not np.isfinite(se):
            windows[int(w)] = {"se_fraction": None, "threshold_fraction": None, "limit_fraction": None,
                               "note": "too few clear days to measure noise at this window"}
            continue
        threshold = max(dcfg["min_shortfall_fraction"], z_alpha * se)
        windows[int(w)] = {"se_fraction": round(se, 5), "threshold_fraction": round(threshold, 5),
                           "limit_fraction": round(threshold + z_beta * se, 5)}
    clear_share = float(frame["clear"].sum() / max(frame["valid"].sum(), 1))
    return {"day_to_day_sigma_fraction": None if not np.isfinite(day_sigma) else round(day_sigma, 5),
            "clear_day_share": round(clear_share, 3), "false_alarm_rate": dcfg["false_alarm_rate"],
            "power": dcfg["power"], "windows": windows}


def detection_limit_text(noise: dict[str, Any]) -> str:
    """Plain-language statement of the detection limit."""
    parts = [f"about {100 * v['limit_fraction']:.1f}% sustained over {w} days"
             for w, v in noise["windows"].items() if v.get("limit_fraction") is not None]
    if not parts:
        return "[not available: too few clear days to measure the noise floor]"
    return "losses below " + ", or ".join(parts) + ", cannot be distinguished from noise"


def _smooth(ratio: pd.DataFrame | pd.Series, w: int, dcfg: dict[str, Any]):
    """Centred running median of clear-day ratio, bridged across short gaps."""
    sm = ratio.rolling(odd(w), center=True, min_periods=dcfg["smoothing_min_clear_days"]).median()
    return sm.interpolate(limit=dcfg["max_interpolation_gap_days"], limit_area="inside")


def _long_runs(flag: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only flags belonging to a run of at least `min_len` days (per column)."""
    f = flag.astype(np.int32)
    if f.ndim == 1:
        return _long_runs(f[:, None], min_len)[:, 0]
    n = f.shape[0]
    if n < min_len:
        return np.zeros_like(flag, dtype=bool)
    c = np.vstack([np.zeros((1, f.shape[1]), np.int32), np.cumsum(f, axis=0)])
    full = (c[min_len:] - c[:-min_len]) == min_len          # run of min_len starting at row i
    keep = np.zeros_like(f, dtype=bool)
    for k in range(min_len):                                 # spread each start over its run
        keep[k:k + full.shape[0]] |= full
    return keep


def _extend(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Grow each seeded run forwards and backwards while `allowed` holds."""
    out = seed.copy()
    for i in range(1, len(out)):
        out[i] |= out[i - 1] & allowed[i]
    for i in range(len(out) - 2, -1, -1):
        out[i] |= out[i + 1] & allowed[i]
    return out


def _shortfall_fraction(ratio, noise: dict[str, Any], dcfg: dict[str, Any]):
    """Loss fraction per day (0 where no sustained shortfall is flagged).

    An episode is triggered when the running median at any tested window is
    below 1 - threshold for that window for at least min_episode_days. It then
    extends in both directions for as long as the running median stays below
    1 - release_ratio x threshold.
    """
    usable = [(w, v) for w, v in noise["windows"].items() if v.get("threshold_fraction") is not None]
    if not usable:
        return None, None, None
    flagged = None
    finest = None
    for w, v in sorted(usable):
        sm = _smooth(ratio, w, dcfg)
        below = (sm < 1.0 - v["threshold_fraction"]).to_numpy()
        flagged = below if flagged is None else (flagged | below)
        # Loss is sized with the shortest window; longer ones only fill its gaps.
        finest = sm if finest is None else finest.fillna(sm)
    keep = _long_runs(flagged, dcfg["min_episode_days"])
    # Hysteresis: a confirmed episode is counted from where performance left
    # the reference to where it returned, not only while it is below the
    # (stricter) trigger. Otherwise the start of every slow ramp is missed.
    release = min(v["threshold_fraction"] for _, v in usable) * dcfg["release_ratio"]
    keep = _extend(keep, (finest < 1.0 - release).to_numpy())
    frac = np.clip(1.0 - finest.to_numpy(), 0.0, 1.0)
    frac = np.where(keep & np.isfinite(frac), frac, 0.0)
    keep = keep & (frac > 0)            # a day with no measurable shortfall is not part of an episode
    return frac, finest, keep


def find_shortfalls(frame: pd.DataFrame, outage: pd.Series, noise: dict[str, Any],
                    dcfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag sustained shortfalls and list them as episodes (step 3).

    Adds `smooth_ratio`, `loss_fraction` and `flagged` to `frame`.
    """
    ratio = frame["ratio"].where(frame["clear"] & ~outage)
    frac, smooth, keep = _shortfall_fraction(ratio, noise, dcfg)
    if frac is None:
        frame["smooth_ratio"] = np.nan
        frame["loss_fraction"] = 0.0
        frame["flagged"] = False
        return []
    frame["smooth_ratio"] = smooth
    frame["loss_fraction"] = frac
    frame["flagged"] = keep
    episodes = []
    active = frame["flagged"]
    group = (active != active.shift()).cumsum()
    for _, days in frame[active].groupby(group[active]):
        episodes.append({
            "start": str(days.index[0].date()), "end": str(days.index[-1].date()),
            "days": int(len(days)),
            "mean_shortfall_pct": round(100 * float(days["loss_fraction"].mean()), 1),
            "max_shortfall_pct": round(100 * float(days["loss_fraction"].max()), 1),
        })
    return episodes


def _block_indices(m: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices: `reps` rows, each a resample of range(m).

    Whole blocks of consecutive days are drawn so that the day-to-day
    correlation of the noise survives in every resample.
    """
    block = max(1, min(block, m))
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(reps, n_blocks))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :m]


def percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """Percentile bootstrap interval."""
    tail = 100.0 * (1.0 - confidence) / 2.0
    lo, hi = np.nanpercentile(samples, [tail, 100.0 - tail])
    return float(lo), float(hi)


def estimate_loss(frame: pd.DataFrame, outage: pd.Series, noise: dict[str, Any], ref_info: dict[str, Any],
                  cfg: dict[str, Any], period: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> dict[str, Any]:
    """Lost energy with a bootstrap confidence interval.

    Point estimate
      sustained loss = sum over flagged days of expected_kwh x loss_fraction
                       (split into "deep" days, more than partial_outage_min_fraction
                       below the reference, and "moderate" days)
      outage loss    = sum over outage days of (expected_kwh - measured_kwh)
    Days without valid generation data contribute nothing (unknown, not zero).

    Uncertainty: each bootstrap replicate redoes the analysis with
      (a) a re-drawn reference (block resample of the sustained clear-day values
          behind each monthly quantile), and
      (b) re-drawn noise (block resample of the clear-day residuals around the
          smoothed ratio), which also lets borderline episodes appear or vanish.
    The interval is the central `confidence` share of the replicates.
    """
    bcfg, dcfg, ecfg = cfg["bootstrap"], cfg["detect"], cfg["expected"]
    rng = np.random.default_rng(bcfg["seed"])
    reps = int(bcfg["samples"])
    in_period = np.ones(len(frame), bool) if period is None else (
        (frame.index >= period[0]) & (frame.index <= period[1]))
    expected = frame["expected_kwh"].to_numpy()
    countable = (frame["valid"] & frame["expected_kwh"].notna()).to_numpy() & in_period
    out = outage.to_numpy() & countable
    sus_days = countable & ~outage.to_numpy()

    frac0 = frame["loss_fraction"].to_numpy()
    deep_cut = dcfg["partial_outage_min_fraction"]
    sustained_kwh = float(np.nansum(expected[sus_days] * frac0[sus_days]))
    # Days far below the reference behave like part of the array being off,
    # not like dirt. They are sized separately so the cleaning verdict can
    # leave them out.
    deep_kwh = float(np.nansum(np.where(sus_days & (frac0 > deep_cut), expected * frac0, 0.0)))
    outage_kwh = float(np.nansum(np.clip(expected[out] - frame["energy_kwh"].to_numpy()[out], 0, None)))
    expected_total = float(np.nansum(expected[countable]))
    result: dict[str, Any] = {
        "expected_kwh": expected_total, "sustained_kwh": sustained_kwh, "outage_kwh": outage_kwh,
        "deep_shortfall_kwh": deep_kwh, "moderate_shortfall_kwh": sustained_kwh - deep_kwh,
        "deep_shortfall_days": int((sus_days & (frac0 > deep_cut)).sum()),
        "lost_kwh": sustained_kwh + outage_kwh, "days_assessed": int(countable.sum()),
        "confidence": bcfg["confidence"],
    }
    result["loss_pct"] = 100 * result["lost_kwh"] / expected_total if expected_total > 0 else float("nan")

    ratio = frame["ratio"].where(frame["clear"] & ~outage)
    clear_pos = np.flatnonzero(ratio.notna().to_numpy())
    if reps <= 0:                       # point estimate only (used inside validation trials)
        return result
    if len(clear_pos) < dcfg["bootstrap_min_clear_days"] or expected_total <= 0 or frame["smooth_ratio"].notna().sum() == 0:
        result["interval_note"] = "too few clear days for a bootstrap interval"
        return result

    # (a) reference uncertainty: re-draw each month's quantile.
    sus = ref_info["sustained_values"].dropna()
    q = ecfg["reference_quantile"]
    monthly_b = np.full((reps, 12), np.nan)
    monthly_0 = np.full(12, np.nan)
    for month in range(1, 13):
        vals = sus[sus.index.month == month].to_numpy()
        if len(vals) >= ecfg["reference_min_days_per_month"]:
            monthly_0[month - 1] = np.quantile(vals, q)
            idx = _block_indices(len(vals), ecfg["sustained_clear_days"], reps, rng)
            monthly_b[:, month - 1] = np.quantile(vals[idx], q, axis=1)
    months = pd.Series(monthly_0, index=range(1, 13))
    base_curve = interpolate_monthly(months, frame.index)
    scale = np.empty((len(frame), reps))                    # reference_b / reference
    for b in range(reps):
        scale[:, b] = interpolate_monthly(pd.Series(monthly_b[b], index=range(1, 13)), frame.index) / base_curve

    # (b) noise: residual block bootstrap around the smoothed ratio.
    smooth0 = frame["smooth_ratio"].to_numpy()[clear_pos]
    values0 = ratio.to_numpy()[clear_pos]
    centre = np.where(np.isfinite(smooth0), smooth0, values0)
    resid = values0 - centre
    idx = _block_indices(len(clear_pos), bcfg["block_days"], reps, rng)
    mat = np.full((len(frame), reps), np.nan)
    mat[clear_pos, :] = (centre[None, :] + resid[idx]).T / scale[clear_pos, :]
    frac_b, _, _ = _shortfall_fraction(pd.DataFrame(mat, index=frame.index), noise, dcfg)

    expected_b = expected[:, None] * scale
    sustained_b = np.nansum(np.where(sus_days[:, None], expected_b * frac_b, 0.0), axis=0)
    deep_b = np.nansum(np.where(sus_days[:, None] & (frac_b > deep_cut), expected_b * frac_b, 0.0), axis=0)
    energy = frame["energy_kwh"].to_numpy()
    if out.any():
        jitter = 1.0 + resid[rng.integers(0, len(resid), size=(int(out.sum()), reps))]
        outage_b = np.nansum(np.clip(expected_b[out] * jitter - energy[out][:, None], 0, None), axis=0)
    else:
        outage_b = np.zeros(reps)
    total_b = sustained_b + outage_b
    expected_total_b = np.nansum(np.where(countable[:, None], expected_b, 0.0), axis=0)

    conf = bcfg["confidence"]
    result.update({
        "lost_kwh_ci": percentile_interval(total_b, conf),
        "loss_pct_ci": percentile_interval(100 * total_b / expected_total_b, conf),
        "sustained_kwh_ci": percentile_interval(sustained_b, conf),
        "outage_kwh_ci": percentile_interval(outage_b, conf),
        "deep_shortfall_kwh_ci": percentile_interval(deep_b, conf),
        "moderate_shortfall_kwh_ci": percentile_interval(sustained_b - deep_b, conf),
        "_replicates": {"lost": total_b, "sustained": sustained_b, "expected": expected_total_b},
    })
    # A point estimate can fall just outside a percentile interval when
    # episodes sit on the detection threshold; widen so the range contains it.
    for key in ("lost_kwh", "sustained_kwh", "outage_kwh", "loss_pct", "deep_shortfall_kwh",
                "moderate_shortfall_kwh"):
        lo, hi = result[f"{key}_ci"]
        result[f"{key}_ci"] = (min(lo, result[key]), max(hi, result[key]))
    return result
