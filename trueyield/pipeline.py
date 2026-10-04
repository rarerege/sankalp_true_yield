"""Orchestration: run every step for every system and collect the results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import causes, detect, economics, expected, ingest, validation, verify, weather
from .config import data_label, path_of


def analyse_frame(frame: pd.DataFrame, cfg: dict[str, Any],
                  period: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> dict[str, Any]:
    """Steps 2b to 4 on one daily table: reference, noise, shortfalls, loss.

    Kept as one function so that the validation step runs exactly the same code
    on the data with an injected loss.
    """
    outage = detect.find_outages(frame, cfg["detect"])
    ref_info = expected.calibrate_reference(frame, cfg["expected"], outage)
    noise = detect.noise_profile(frame, outage, cfg["detect"], cfg["expected"])
    episodes = detect.find_shortfalls(frame, outage, noise, cfg["detect"])
    loss = detect.estimate_loss(frame, outage, noise, ref_info, cfg, period=period)
    return {"outage": outage, "ref_info": ref_info, "noise": noise, "episodes": episodes, "loss": loss}


def run_system(sd: ingest.SystemData, site: pd.Series, events: pd.DataFrame, cfg: dict[str, Any],
               sources: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Run the whole pipeline for one system. Gaps are recorded, not hidden."""
    notes: list[str] = []
    res: dict[str, Any] = {"system_id": sd.system_id, "name": site["name"], "site": site, "sd": sd,
                           "data_label": data_label(sources, sd.system_id), "notes": notes, "status": "ok"}
    has_orientation = bool(pd.notna(site["tilt_deg"]) and pd.notna(site["azimuth_deg"]))
    if not has_orientation:
        notes.append("Tilt and azimuth are not provided by the data source. Daily mode does not use them (the "
                     "seasonal reference absorbs the orientation), so no value was assumed.")
    if pd.isna(site["commissioning_date"]) or str(site["commissioning_date"]).strip() == "":
        notes.append("Commissioning date is not provided by the data source; it is not used in any calculation.")
    start, end = sd.daily.index[0], sd.daily.index[-1]
    cache = path_of(cfg, "cache_dir")
    try:
        wx = weather.fetch_daily(float(site["latitude"]), float(site["longitude"]), start, end, cache, cfg["weather"])
    except weather.WeatherUnavailable as exc:
        res["status"] = "weather unavailable"
        notes.append(f"NASA POWER daily weather could not be fetched ({exc}); expected yield, loss, "
                     "causes and recovery were not computed for this system.")
        return res

    mode = cfg["expected"]["mode"]
    use_hourly = sd.resolution == "hourly" and mode in ("auto", "hourly") and has_orientation
    model_hourly = None
    if sd.resolution == "hourly" and mode in ("auto", "hourly") and not has_orientation:
        notes.append("The hourly model needs tilt and azimuth, which are missing; daily mode was used and the "
                     "shading test could not be run.")
    if mode == "hourly" and sd.resolution != "hourly":
        notes.append("Hourly mode was requested but the generation data are daily; daily mode was used.")
    if use_hourly:
        try:
            wx_h = weather.fetch_hourly(float(site["latitude"]), float(site["longitude"]),
                                        start - pd.Timedelta(days=1), end + pd.Timedelta(days=1), cache, cfg["weather"])
            model_hourly = expected.hourly_model(wx_h, site, cfg["expected"])
        except weather.WeatherUnavailable as exc:
            notes.append(f"Hourly weather could not be fetched ({exc}); daily mode was used instead.")
    res["mode"] = "hourly (pvlib)" if model_hourly is not None else "daily"
    res["model_hourly"] = model_hourly

    frame = expected.build_frame(sd.daily, wx, site, cfg["expected"], model_hourly)
    missing_wx = int(frame["ghi"].isna().sum())
    if missing_wx:
        notes.append(f"{missing_wx} day(s) have no satellite irradiance (NASA POWER fill value); "
                     "they are not normalised and not counted.")
    try:
        analysis = analyse_frame(frame, cfg)
    except ValueError as exc:
        res["status"] = "too few clear days"
        notes.append(f"Reference could not be calibrated ({exc}); loss was not estimated.")
        res["frame"] = frame
        return res
    res.update(frame=frame, **analysis)
    if analysis["ref_info"]["months_without_own_reference"]:
        notes.append("Months with too few clear days for their own reference (borrowed from neighbouring "
                     f"months): {analysis['ref_info']['months_without_own_reference']}.")
    not_assessable = int((frame["valid"] & frame["smooth_ratio"].isna() & ~analysis["outage"]).sum())
    if not_assessable:
        notes.append(f"{not_assessable} valid day(s) lie in stretches with too few clear days to judge "
                     "the system's state; no loss is counted there.")

    # Cross-check: noise of the simpler daily-mode normalisation on the same data.
    if model_hourly is not None:
        alt = expected.build_frame(sd.daily, wx, site, cfg["expected"], None)
        alt_noise = detect.noise_profile(alt, detect.find_outages(alt, cfg["detect"]), cfg["detect"],
                                         cfg["expected"])
        res["daily_mode_cross_check"] = {"day_to_day_sigma_fraction": alt_noise["day_to_day_sigma_fraction"],
                                         "windows": alt_noise["windows"]}

    # Days when part of the system was off say nothing about dirt, ageing or
    # shade, and a rain day that happens to fall next to a fault would look like
    # a huge "recovery" or "loss". Such days are left out of the tests below.
    deep = frame["flagged"] & (frame["loss_fraction"] > cfg["detect"]["partial_outage_min_fraction"])
    fault = analysis["outage"] | deep
    res["fault_days"] = fault
    if deep.any():
        notes.append(f"{int(deep.sum())} day(s) in deep-drop episodes (part of the system off) are excluded from "
                     "the recovery, soiling, ageing and shading tests, so those tests describe the system while "
                     "it was otherwise working.")
    res["recovery"] = verify.verify_events(frame, fault, events, sd.system_id, cfg)
    cleanings = [pd.Timestamp(e["end"]) for e in res["recovery"]["events"] if e["type"] == "cleaning"]
    res["causes"] = [
        causes.outage_cause(frame, analysis["outage"], sd.daily, analysis["loss"], cfg),
        causes.soiling_cause(frame, fault, res["recovery"], cleanings, cfg, cache),
        causes.degradation_cause(frame, fault, cfg),
        causes.shading_cause(frame, fault, sd.hourly, model_hourly, cfg),
    ]
    period_days = len(frame)
    res["economics"] = economics.economics(analysis["loss"], site, period_days, cfg["economics"])
    res["emissions"] = economics.emissions(analysis["loss"], cfg["emissions"])
    return res


def portfolio(results: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    """Add up the systems. Replicates are summed so the range is carried through."""
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return {"systems": 0}
    lost = sum(r["loss"]["lost_kwh"] for r in ok)
    exp = sum(r["loss"]["expected_kwh"] for r in ok)
    out: dict[str, Any] = {"systems": len(ok), "lost_kwh": lost, "expected_kwh": exp,
                           "loss_pct": 100 * lost / exp if exp else float("nan")}
    if all("_replicates" in r["loss"] for r in ok):
        lost_b = sum(r["loss"]["_replicates"]["lost"] for r in ok)
        exp_b = sum(r["loss"]["_replicates"]["expected"] for r in ok)
        lo, hi = detect.percentile_interval(lost_b, confidence)
        plo, phi = detect.percentile_interval(100 * lost_b / exp_b, confidence)
        out["lost_kwh_ci"] = (min(lo, lost), max(hi, lost))
        out["loss_pct_ci"] = (min(plo, out["loss_pct"]), max(phi, out["loss_pct"]))
    rupees = [r["economics"].get("rupees_lost") for r in ok]
    if all(v is not None for v in rupees):
        out["rupees_lost"] = sum(rupees)
        if all("rupees_lost_ci" in r["economics"] for r in ok) and "lost_kwh_ci" in out:
            # Each system has its own tariff, so scale each system's replicates.
            rb = sum(r["loss"]["_replicates"]["lost"] * r["economics"]["tariff_inr_per_kwh"] for r in ok)
            lo, hi = detect.percentile_interval(rb, confidence)
            out["rupees_lost_ci"] = (min(lo, out["rupees_lost"]), max(hi, out["rupees_lost"]))
    return out


def run_all(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ingest, analyse every system, validate the method, and return everything."""
    raw_dir = path_of(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    created = ingest.ensure_templates(path_of(cfg, "systems_csv"), path_of(cfg, "events_csv"))
    has_raw = any(p.suffix.lower() in (".csv", ".xlsx", ".xls") for p in raw_dir.iterdir())
    if not has_raw:
        from .fetch_public import fetch_public
        print("data/raw/ is empty: fetching the configured public dataset (real measurements).")
        if created and Path(created[0]).name == "systems.csv":
            path_of(cfg, "systems_csv").unlink()            # let the fetcher fill it from dataset metadata
        fetch_public(cfg)

    systems, econ_warnings = ingest.load_systems(path_of(cfg, "systems_csv"))
    events = ingest.load_events(path_of(cfg, "events_csv"))
    # One SOURCE*.yaml per public dataset; each lists the systems it covers.
    sources = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("SOURCE*.yaml"))]

    data = ingest.ingest_all(raw_dir, systems, cfg["ingest"])
    results = []
    for sid, sd in data.items():
        print(f"[{sid}] " + " | ".join(l.describe() for l in sd.layouts))
        print(f"[{sid}] completeness {sd.completeness['completeness_pct']}% "
              f"({sd.completeness['valid_days']} of {sd.completeness['calendar_days']} days valid)")
        results.append(run_system(sd, systems.loc[sid], events, cfg, sources))

    # Systems are grouped by dataset. Totals are never added across datasets
    # (different places, periods and data quality), and the method is validated
    # separately on each, on its most complete system.
    groups = []
    for label in dict.fromkeys(r["data_label"] for r in results):
        members = [r for r in results if r["data_label"] == label]
        ok = [r for r in members if r["status"] == "ok"]
        val = None
        if ok:
            best = max(ok, key=lambda r: r["sd"].completeness["completeness_pct"])
            val = validation.validate(best["frame"], best, analyse_frame, cfg, best["system_id"])
            val["data_label"] = label
        groups.append({"data_label": label, "system_ids": [r["system_id"] for r in members],
                       "portfolio": portfolio(members, cfg["bootstrap"]["confidence"]), "validation": val})
    return {"results": results, "groups": groups, "sources": sources, "systems": systems, "events": events,
            "economics_warnings": econ_warnings, "cfg": cfg}
