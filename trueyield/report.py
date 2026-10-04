"""Write every output under results/.

Rules enforced here rather than left to the reader's goodwill:
  * every output states its data label;
  * every loss or recovery figure is printed with its range;
  * detected facts and hypotheses are printed under separate headings;
  * anything that could not be computed is written as "[not available: reason]".
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import charts
from .config import NATURAL_RAIN_LABEL, VALIDATION_LABEL, path_of
from .detect import detection_limit_text


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------
def rng(pair: Any, fmt: str = "{:,.0f}", unit: str = "") -> str:
    """Format a (low, high) range, or say that none is available."""
    if pair is None:
        return "range not available"
    return f"{fmt.format(pair[0])}–{fmt.format(pair[1])}{unit}"


def jsonable(obj: Any) -> Any:
    """Convert numpy/pandas objects to plain JSON types; drop private keys."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, dt.date)):
        return str(obj)[:10]
    if isinstance(obj, (pd.Series, pd.DataFrame, np.ndarray)):
        return None
    return obj


def group_of(run: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    """The dataset group (same data label) a system belongs to."""
    return next(g for g in run["groups"] if res["system_id"] in g["system_ids"])


def months_of(res: dict[str, Any]) -> float:
    c = res["sd"].completeness
    return (pd.Timestamp(c["period_end"]) - pd.Timestamp(c["period_start"])).days / 30.44


def reliable_text(val: dict[str, Any] | None) -> str:
    """What the injected-loss trials say about the stated detection limit."""
    if not val or not val.get("available"):
        return "No injected-loss check of this limit was possible."
    parts = [f"about {v['step_pct']:.0f}% over {w} days" if v.get("step_pct") is not None
             else f"more than the largest step tried over {w} days" for w, v in val["reliable_detection"].items()]
    if all(v.get("step_pct") is None for v in val["reliable_detection"].values()):
        return (f"Injected-loss trials on real data from system {val['system_id']} did not reach reliable detection "
                "(80% of trials) at any size tried, so this limit is optimistic for this dataset.")
    parts = [f"at about {v['step_pct']:.0f}% over {w} days" if v.get("step_pct") is not None
             else f"not at any size tried over {w} days" for w, v in val["reliable_detection"].items()]
    return (f"Checked with injected-loss trials on real data from system {val['system_id']}: reliable detection "
            "(at least 80% of trials) was reached " + ", and ".join(parts) + ".")


def method_text(res: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Describe the chosen expected-yield method and its assumptions."""
    e = cfg["expected"]
    if res.get("mode", "").startswith("hourly"):
        model = ("Hourly mode. NASA POWER hourly global irradiance is split into beam and diffuse (Erbs), "
                 "transposed to the array plane (Hay-Davies, pvlib) and converted to DC energy with a "
                 f"PVWatts model (temperature coefficient {100 * e['temp_coeff_per_c']:.1f}%/°C, NOCT {e['noct_c']} °C). "
                 "Hourly values are summed to days.")
    else:
        model = ("Daily mode. Daily horizontal irradiation × nameplate capacity × a temperature factor "
                 f"({100 * e['temp_coeff_per_c']:.1f}%/°C, NOCT {e['noct_c']} °C cell-temperature rule).")
    return (model + " Performance index = measured ÷ modelled energy. The reference is the system's own "
            f"{100 * e['reference_quantile']:.0f}th-percentile sustained ({e['sustained_clear_days']}-clear-day median) "
            "performance per calendar month, pooled over all years and reduced by the upward bias expected from noise. "
            f"Clear day = all-sky ÷ clear-sky irradiation ≥ {e['clear_day_min_clearness']}. Assumptions: constant "
            "inverter and wiring losses are absorbed by the reference; timestamps are local standard time; "
            "the cleanest state seen in each month is representative of a clean array.")


# --------------------------------------------------------------------------
# summary.json
# --------------------------------------------------------------------------
def system_summary(res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    c = res["sd"].completeness
    out: dict[str, Any] = {
        "system_id": res["system_id"], "name": res["name"], "data_label": res["data_label"],
        "status": res["status"],
        "period": {"start": c["period_start"], "end": c["period_end"], "months": round(months_of(res), 1)},
        "input_resolution": res["sd"].resolution, "completeness": c,
        "inferred_layout": [l.describe() for l in res["sd"].layouts],
        "honest_gaps": list(res["notes"]),
    }
    if res["status"] != "ok":
        return out
    loss = res["loss"]
    out["reference_method"] = {"mode": res["mode"], "description": method_text(res, cfg),
                               **{k: v for k, v in res["ref_info"].items() if k != "sustained_values"}}
    out["estimated_loss"] = {
        "loss_pct": loss["loss_pct"], "loss_pct_range": loss.get("loss_pct_ci"),
        "lost_kwh": loss["lost_kwh"], "lost_kwh_range": loss.get("lost_kwh_ci"),
        "of_which_sustained_shortfall_kwh": loss["sustained_kwh"],
        "sustained_shortfall_kwh_range": loss.get("sustained_kwh_ci"),
        "sustained_part_deep_drop_kwh": loss["deep_shortfall_kwh"],
        "sustained_part_deep_drop_kwh_range": loss.get("deep_shortfall_kwh_ci"),
        "sustained_part_moderate_kwh": loss["moderate_shortfall_kwh"],
        "sustained_part_moderate_kwh_range": loss.get("moderate_shortfall_kwh_ci"),
        "of_which_outage_days_kwh": loss["outage_kwh"], "outage_days_kwh_range": loss.get("outage_kwh_ci"),
        "expected_kwh_over_valid_days": loss["expected_kwh"], "valid_days_assessed": loss["days_assessed"],
        "confidence_level": loss["confidence"], "interval_method": "block bootstrap of clear-day residuals and of the reference",
        "interval_note": loss.get("interval_note"),
        "shortfall_episodes": res["episodes"],
    }
    out["detection_limit"] = {"statement": detection_limit_text(res["noise"]), **res["noise"]}
    if "daily_mode_cross_check" in res:
        out["detection_limit"]["daily_mode_cross_check"] = res["daily_mode_cross_check"]
    out["cause_hypotheses"] = res["causes"]
    out["recovery"] = res["recovery"]
    out["rupee_value"] = res["economics"]
    out["verdict"] = res["economics"].get("verdict")
    out["emissions"] = res["emissions"]
    return out


def write_summary(run: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    cfg = run["cfg"]
    summary = {
        "generated_on": dt.date.today().isoformat(),
        "tool": "TrueYield prototype (physics and statistics only; no machine learning, no LLM calls)",
        "data_labels": sorted({r["data_label"] for r in run["results"]}),
        "datasets": [{"data_label": g["data_label"], "system_ids": g["system_ids"], "totals": g["portfolio"],
                      "method_validation": g["validation"]} for g in run["groups"]],
        "systems": [system_summary(r, cfg) for r in run["results"]],
        "notes": run["economics_warnings"],
    }
    clean = jsonable(summary)
    (out_dir / "summary.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return clean


# --------------------------------------------------------------------------
# slide6_values.md
# --------------------------------------------------------------------------
def headline_event(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best-observed event across the given systems (same data-quality rule as per system)."""
    from .verify import clearest_event
    best = [r["recovery"]["clearest"] for r in results if r["status"] == "ok" and r["recovery"]["clearest"]]
    return clearest_event(best)


def slide_lines(run: dict[str, Any], group: dict[str, Any], planned_test: str | None = None) -> list[str]:
    """The five slide lines for one dataset.

    With `planned_test`, line 3 uses the template's "Recovery test planned"
    wording instead of reporting a tested event.
    """
    members = [r for r in run["results"] if r["system_id"] in group["system_ids"]]
    ok = [r for r in members if r["status"] == "ok"]
    lines: list[str] = []
    if not ok:
        reason = "; ".join(n for r in members for n in r["notes"]) or "no system could be analysed"
        return [f"[not available: {reason}]"] * 3 + ["Data: " + group["data_label"], f"[not available: {reason}]"]
    locations = sorted({str(r["site"].get("location", "")).strip() for r in ok} - {""})
    where = "; ".join(locations) if locations else "; ".join(
        sorted({f"{float(r['site']['latitude']):.2f}°, {float(r['site']['longitude']):.2f}°" for r in ok}))
    months = round(max(months_of(r) for r in ok))
    res_kind = "hourly" if all(r["sd"].resolution == "hourly" for r in ok) else "daily"
    lines.append(f"Analysed {len(ok)} system(s) in {where}, {months} months of {res_kind} data.")

    p = group["portfolio"]
    start = min(r["sd"].completeness["period_start"] for r in ok)
    end = max(r["sd"].completeness["period_end"] for r in ok)
    period = f"{pd.Timestamp(start):%b %Y}–{pd.Timestamp(end):%b %Y}"
    if "loss_pct_ci" in p:
        # Template: "Detected loss: X% (range A–B%), about K kWh and ₹R over <period>."
        # The kWh and rupee ranges are in summary.json; the slide carries the range once, on the percentage.
        rupees = (f"₹{p['rupees_lost']:,.0f}" if "rupees_lost" in p
                  else "₹[not available: tariff_inr_per_kwh not provided in data/systems.csv]")
        lines.append(f"Detected loss: {p['loss_pct']:.1f}% (range {rng(p['loss_pct_ci'], '{:.1f}', '%')}), about "
                     f"{p['lost_kwh']:,.0f} kWh and {rupees} over {period}.")
    else:
        lines.append("[not available: too few clear days to put a range on the loss, and a loss is never "
                     "reported without one]")

    ev = None if planned_test else headline_event(ok)
    if planned_test:
        lines.append(f"Recovery test planned: {planned_test}.")
    elif ev is not None:
        # Template: "Recovery after <event type on date>: +Y% (range C–D%)."
        kind = (f"natural rain event ({NATURAL_RAIN_LABEL.split(', ')[1]})" if "rain" in ev["type"] else ev["type"])
        where = f" at system {ev['system_id']}" if len(ok) > 1 else ""
        lo, hi = ev["change_pct_ci"]
        # A dash between a negative and a positive number is unreadable, so a
        # range that crosses zero is written with signs and "to".
        span = f"{lo:.1f}–{hi:.1f}%" if lo >= 0 else f"{lo:+.1f} to {hi:+.1f}%"
        lines.append(f"Recovery after {kind}{where} on {ev['date']}: {ev['change_pct']:+.1f}% (range {span}).")
    else:
        target = max(ok, key=lambda r: r["loss"]["sustained_kwh"])
        lines.append(f"Recovery test planned: one dated panel cleaning on system {target['system_id']} "
                     f"({target['name']}), comparing clear-day normalised output for 14 days before and after.")
    lines.append("Data: " + "; ".join(sorted({r["data_label"].replace("Public dataset: ", "Public dataset (") + ")"
                                             if r["data_label"].startswith("Public") else r["data_label"] for r in ok})))
    limits = [(r["system_id"], w, v["limit_fraction"]) for r in ok for w, v in r["noise"]["windows"].items()
              if v.get("limit_fraction") is not None]
    if limits:
        w0 = min(w for _, w, _ in limits)
        at = [100 * l for _, w, l in limits if w == w0]
        span = f"{min(at):.1f}%" if max(at) - min(at) < 0.05 else f"{min(at):.1f}–{max(at):.1f}% depending on the system"
        text = (f"Detection limit: losses sustained over {w0} days that are below about {span} cannot be "
                "distinguished from noise")
        val = group["validation"]
        hit = (val or {}).get("reliable_detection", {}).get(str(w0), {}) if val and val.get("available") else {}
        if hit.get("step_pct") is not None:
            text += (f"; injected-loss trials on real data (system {val['system_id']}) reached reliable "
                     f"detection at about {hit['step_pct']:.0f}%")
        elif hit:
            text += (f"; injected-loss trials on real data (system {val['system_id']}) did not reach reliable "
                     "detection at any size tried, so treat this limit as optimistic")
        lines.append(text + ".")
    else:
        lines.append("[not available: too few clear days to measure the noise floor]")
    return lines


def _find_group(run: dict[str, Any], text: str | None) -> dict[str, Any] | None:
    return next((g for g in run["groups"] if text and text.lower() in g["data_label"].lower()), None)


def _paren(label: str) -> str:
    """'Public dataset: X' -> 'Public dataset (X)', as the slide template writes it."""
    return label.replace("Public dataset: ", "Public dataset (") + ")" if label.startswith("Public") else label


def lead_slide_lines(run: dict[str, Any], lead: dict[str, Any], check: dict[str, Any] | None) -> list[str]:
    """Five lines for the lead dataset, citing another dataset as the method check."""
    scfg = run["cfg"]["slide"]
    planned = scfg.get("planned_recovery_test") if scfg.get("recovery_line") == "planned" else None
    lines = slide_lines(run, lead, planned_test=planned)
    val = (check or {}).get("validation")
    if check and check is not lead and val and val.get("available"):
        members = [r for r in run["results"] if r["system_id"] in check["system_ids"] and r["status"] == "ok"]
        where = "; ".join(sorted({str(r["site"].get("location", "")).strip() for r in members} - {""}))
        months = round(max(months_of(r) for r in members))
        verdict = "inside" if val["injected_within_stated_range"] else "outside"
        lines[3] += f"; method validated on {_paren(check['data_label'])}"
        lines[4] += (f" Method check on {len(members)} system(s) with {months} months of data"
                     + (f" in {where}" if where else "")
                     + f": {val['recovered_share_of_injected_pct']:.0f}% of a known injected loss was recovered, "
                     f"with the true value {verdict} the stated range.")
    return lines


def write_slide(run: dict[str, Any], out_dir: Path) -> list[tuple[str, list[str]]]:
    """Write slide6_values.md.

    With a lead dataset configured, the file is exactly five lines about that
    dataset, and the full five lines of every dataset go to
    slide6_values_by_dataset.md. Otherwise slide6_values.md holds one block of
    five lines per dataset (exactly five lines when there is only one).
    """
    def numbered(lines: list[str]) -> str:
        return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))

    blocks = [(g["data_label"], slide_lines(run, g)) for g in run["groups"]]
    per_dataset = "\n\n".join(numbered(lines) if len(blocks) == 1 else f"## {label}\n\n{numbered(lines)}"
                               for label, lines in blocks) + "\n"
    scfg = run["cfg"].get("slide", {})
    lead = _find_group(run, scfg.get("lead_dataset"))
    if lead is None:
        (out_dir / "slide6_values_by_dataset.md").unlink(missing_ok=True)
        (out_dir / "slide6_values.md").write_text(per_dataset, encoding="utf-8")
        return blocks
    lines = lead_slide_lines(run, lead, _find_group(run, scfg.get("validation_dataset")))
    (out_dir / "slide6_values.md").write_text(numbered(lines) + "\n", encoding="utf-8")
    (out_dir / "slide6_values_by_dataset.md").write_text(
        "Reference only: the five lines for each dataset on its own, with the best-observed event on line 3.\n\n"
        + per_dataset, encoding="utf-8")
    return [("slide6_values.md (lead: " + lead["data_label"] + ")", lines)]


# --------------------------------------------------------------------------
# data_provenance.md and LIMITATIONS.md
# --------------------------------------------------------------------------
def write_provenance(run: dict[str, Any], out_dir: Path) -> None:
    cfg = run["cfg"]
    lines = ["# Data provenance", "", f"Generated {dt.date.today().isoformat()}.", ""]
    public_ids: set[str] = set()
    for src in run["sources"]:
        public_ids |= {str(s) for s in src["system_ids"]}
        lines += [
            f"## Generation data: public dataset — {src['name']}", "",
            f"- **Name:** {src['name']}",
            f"- **Exact source URL:** {src['url']}" + (f" ({src['url_note']})" if src.get("url_note") else ""),
            f"- **Landing page:** {src.get('landing_page', '[not recorded]')}",
            f"- **Licence:** {src.get('licence') or '[not recorded]'}",
            f"- **Citation:** {src.get('citation') or '[not recorded]'}",
            f"- **Systems used:** {', '.join(str(s) for s in src['system_ids'])}",
            f"- **Period:** {src['period'][0]} to {src['period'][1]}",
            f"- **Retrieved on:** {src['retrieved_on']}",
            f"- **Site details:** {src.get('site_details', '[not recorded]')}",
        ]
        if src.get("preparation"):
            lines.append(f"- **Preparation:** {src['preparation']}")
        lines += [
            f"- **Permission notes:** {src.get('permission_notes', 'Cite the dataset as above.')} "
            "These are measurements from real systems. They are **not** the user's own systems.",
            "- **Label on all outputs:** `Public dataset: " + src["name"] + "`", "",
        ]
    own = [r for r in run["results"] if r["system_id"] not in public_ids]
    if own:
        lines += ["## Generation data: real system data", "",
                  "Files placed in `data/raw/` by the user for systems: " + ", ".join(r["system_id"] for r in own) + ".",
                  "- **Source:** the user's own inverter exports.",
                  "- **Licence / permission:** [to be stated by the user: owner's consent to use and publish]",
                  "- **Label on all outputs:** `Real system data`", ""]
    lines += [
        "## Weather data", "",
        "- **Name:** NASA POWER (Prediction Of Worldwide Energy Resources), daily and hourly point API",
        f"- **URL:** {cfg['weather']['nasa_power_base']}/daily/point and /hourly/point",
        "- **Parameters:** ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, PRECTOTCORR "
        "(sources: CERES SYN1deg for irradiance, MERRA-2 for temperature and precipitation)",
        "- **Licence:** NASA open data, free to use without restriction; acknowledgement requested: "
        "\"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through "
        "the NASA Earth Science/Applied Science Program.\"",
        "- **Cached copies:** `data/cache/nasa_power_*.json` (exact responses used in this run)",
        "- **Resolution caveat:** grid cells are about 1° (irradiance) and 0.5° × 0.625° (meteorology); "
        "values are area averages, not site measurements.", "",
        "## Synthetic data", "",
        "None in any field result. The only altered data are in the step labelled "
        f"\"{VALIDATION_LABEL}\" (`results/validation/`), where a known loss is subtracted from a copy of real "
        "data to test the method, and in the unit tests.", "",
    ]
    (out_dir / "data_provenance.md").write_text("\n".join(lines), encoding="utf-8")


def write_limitations(run: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Limitations", "",
        "Plain-language list of what this prototype cannot yet do or prove.", "",
        "## What the method cannot prove", "",
        "- **It never proves a cause.** Soiling, ageing, shading and faults are reported as hypotheses that fit a "
        "pattern. Only a site visit can confirm them.",
        "- **\"Loss\" means loss against the system's own best.** If a system was never clean, or has a permanent "
        "fault from day one, its \"best\" already includes that loss and the method cannot see it.",
        "- **Seasonal self-calibration can hide a seasonal problem.** The reference is set per calendar month. A loss "
        "that returns in the same month every year (for example heavy dust every May) lowers that month's reference "
        "and is partly hidden. More years with at least one clean spell per month reduce this.",
        "- **Small losses are invisible.** Anything below the stated detection limit cannot be told apart from noise, "
        "and is counted as zero. The true loss can therefore be larger than reported.",
        "- **The stated detection limit assumes the system starts at its reference.** When a system is running a few "
        "percent above its reference (a better-than-usual month), a loss of that size can go unseen. The validation "
        "stretch is chosen close to the reference for this reason; on a stretch running above it, less of an injected "
        "loss is recovered.",
        "- **The rdtools soiling estimate is inflated by scatter.** rdtools looks for upward steps, and day-to-day "
        "scatter alone produces steps. On a test input with no soiling and 3% daily scatter it reported about 3% "
        "loss. Each estimate is therefore shown next to a shuffle check (the same values in random time order), and "
        "soiling is only called indicated when the estimate is clearly above it. The detected loss in the headline "
        "does not use rdtools and is not affected.",
        "- **A slow unexplained drift cannot be attributed.** Month-to-month wander of a few percent may be soiling, "
        "haze the satellite misjudged, or the simple model. It is flagged only above the trigger and never proven.",
        "- **A natural rain event is not a cleaning.** Rain may clean poorly or coincide with other changes; a "
        "before/after rise after rain shows the method can see a recovery, not that maintenance works.",
        "- **The cleaning verdict is a screening rule.** It assumes one cleaning a month recovers the whole sustained "
        "shortfall. If part of that shortfall is ageing, cleaning recovers less.",
        "", "## Data limits", "",
        "- **Satellite irradiance, not a site sensor.** NASA POWER is an area average over tens of kilometres. Local "
        "cloud, haze or dust storms add day-to-day scatter; this is the main reason for the detection limit.",
        "- **Clear days only.** The state of the system is judged on clear days and assumed to hold on the days in "
        "between. Long cloudy spells (for example a monsoon) leave stretches that cannot be assessed.",
        "- **Missing days are unknown, not zero.** Days without valid data are left out of both loss and expected "
        "energy. A real outage during a data gap is not counted.",
        "- **An outage and a logging fault look the same** when the logger records zeros.",
        "- **Timestamps are assumed to be local standard time** with a fixed offset from longitude unless "
        "`utc_offset_hours` is given. Daylight-saving shifts are not corrected; a clock check detects them and the "
        "shading test is then reported as not assessable. Daily totals are not affected.",
        "", "## Loader limits", "",
        "- One system per file (or a `system_id` column with one value). Wide files with one column per system are "
        "not supported.",
        "- Lifetime counters and counters that reset every day (\"energy today\") are both supported. For a "
        "daily-reset counter, energy produced between the last reading before a reset and the reset itself is not "
        "seen; this is zero when the reset happens at night.",
        "- Units are inferred from magnitude against the nameplate. A wrong capacity in `systems.csv` gives wrong "
        "units; check the inferred layout printed at the start of each run.",
        "", "## Modelling limits", "",
        "- Fixed-tilt arrays with a single orientation only. No trackers, no split east-west arrays.",
        "- No inverter clipping model, no spectral or reflection (incidence-angle) model, no snow model. Constant "
        "effects are absorbed by the reference; effects that vary within a month are not.",
        "- Shading is tested only above 15° sun elevation and only as a recurring hourly dip; partial or seasonal "
        "shading at low sun is not assessed.",
        "- Ageing needs more than two years of data and is a single linear rate.",
        "- **Without rdtools the soiling and ageing estimates are cruder.** The built-in soiling fallback only treats "
        "heavy rain and recorded cleanings as cleaning events. On the public data used here it gave 0.8-1.8% against "
        "4.3-5.9% from rdtools; lowering its rain threshold did not close the gap, and most of the difference is "
        "the scatter-driven inflation of rdtools described above. The built-in ageing fallback gives a similar rate "
        "with a wider range.",
        "- The bootstrap range covers day-to-day noise and the uncertainty of the reference. It does not cover "
        "systematic errors such as a wrong tilt in `systems.csv` or a drifting satellite product.",
        "", "## Not yet built", "",
        "- No live data connection; files are dropped in by hand.",
        "- No cost model beyond one cleaning price per system.",
        "- Emissions are only reported if a grid emission factor is entered in `config.yaml`.",
        "- **One year of data is the bare minimum, and it is weak.** Each month's reference then comes from that same "
        "month, so a loss lasting most of a month lowers its own reference and is partly hidden. On the one-year "
        "Jaipur dataset the injected-loss validation does not pass; see `results/validation/`.",
        "- Validated on one stretch of one system per dataset (see the method-validation section of the report), not across "
        "climates. The validation recovers most, not all, of an injected slow ramp: the part of a ramp that stays "
        "inside the noise is not counted.", "",
    ]
    gaps = [(r["system_id"], n) for r in run["results"] for n in r["notes"]]
    if gaps or run["economics_warnings"]:
        lines += ["## Gaps in this particular run", ""]
        lines += [f"- System {sid}: {note}" for sid, note in gaps]
        lines += [f"- Economics: {w}; rupee figures are not available for it." for w in run["economics_warnings"]]
        lines.append("")
    (out_dir / "LIMITATIONS.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------
def _fonts() -> tuple[str, str]:
    """Register DejaVu Sans (shipped with matplotlib) so that ₹ and – render."""
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    return "DejaVu", "DejaVu-Bold"


def write_pdf(res: dict[str, Any], run: dict[str, Any], chart_paths: dict[str, str], out_dir: Path) -> str:
    """One-to-two page auditable report for one system."""
    cfg = run["cfg"]
    font, bold = _fonts()
    navy = colors.HexColor(cfg["charts"]["navy"])
    body = ParagraphStyle("body", fontName=font, fontSize=8.2, leading=10.6, textColor=colors.HexColor("#1B1B1B"))
    small = ParagraphStyle("small", parent=body, fontSize=7.2, leading=9.2, textColor=colors.HexColor("#444444"))
    h1 = ParagraphStyle("h1", fontName=bold, fontSize=15, leading=18, textColor=navy)
    h2 = ParagraphStyle("h2", fontName=bold, fontSize=9.6, leading=12, textColor=navy, spaceBefore=5, spaceAfter=1.5)
    path = out_dir / f"report_{res['system_id']}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=12 * mm, bottomMargin=13 * mm,
                            title=f"TrueYield report {res['system_id']}", author="TrueYield prototype")
    c = res["sd"].completeness
    story: list[Any] = [
        Paragraph(f"TrueYield recovery report — {res['name']} (system {res['system_id']})", h1),
        Paragraph(f"<b>Data label: {res['data_label']}</b> &nbsp;|&nbsp; Period {c['period_start']} to "
                  f"{c['period_end']} &nbsp;|&nbsp; {float(res['site']['capacity_kwp']):g} kWp, "
                  + (f"tilt {float(res['site']['tilt_deg']):g}°, azimuth {float(res['site']['azimuth_deg']):g}°"
                     if pd.notna(res["site"]["tilt_deg"]) and pd.notna(res["site"]["azimuth_deg"])
                     else "tilt and azimuth not provided by the source")
                  + f" &nbsp;|&nbsp; generated {dt.date.today().isoformat()}", small),
        Spacer(1, 3),
    ]

    def bullets(items: list[str], style=body) -> None:
        for item in items:
            story.append(Paragraph("• " + item, style))

    story.append(Paragraph("1. What is measured (facts)", h2))
    bullets([
        f"Generation: {c['valid_days']} valid days of {c['calendar_days']} ({c['completeness_pct']}% complete); "
        f"{c['gap_days']} days with no data, {c['partial_days']} partial days and {c['impossible_days']} days with "
        "impossible values were excluded, not filled in.",
        "Loader inferred: " + " ".join(l.describe() for l in res["sd"].layouts),
        "Weather: NASA POWER satellite irradiance (all-sky and clear-sky), 2 m temperature and rainfall for the "
        "site coordinates. No on-site sensor is used.",
    ])
    if res["status"] != "ok":
        story.append(Paragraph("2. What could not be done", h2))
        bullets(res["notes"])
        doc.build(story)
        return str(path)

    loss, econ, noise = res["loss"], res["economics"], res["noise"]
    conf = f"{100 * loss['confidence']:.0f}%"
    story.append(Paragraph("2. What is estimated, and how sure", h2))
    if "loss_pct_ci" in loss:
        loss_line = (f"<b>Estimated loss: {loss['loss_pct']:.1f}% of expected energy (range "
                     f"{rng(loss['loss_pct_ci'], '{:.1f}', '%')}), {loss['lost_kwh']:,.0f} kWh (range "
                     f"{rng(loss['lost_kwh_ci'])} kWh)</b> over {loss['days_assessed']} valid days. Of this, "
                     f"{loss['sustained_kwh']:,.0f} kWh (range {rng(loss['sustained_kwh_ci'])}) is sustained shortfall in "
                     f"{len(res['episodes'])} flagged episode(s)"
                     + (f", including {loss['deep_shortfall_kwh']:,.0f} kWh (range {rng(loss['deep_shortfall_kwh_ci'])}) "
                        f"on days more than {100 * cfg['detect']['partial_outage_min_fraction']:.0f}% below the reference"
                        if loss["deep_shortfall_kwh"] > 0 else "")
                     + f", and {loss['outage_kwh']:,.0f} kWh (range {rng(loss['outage_kwh_ci'])}) is from days with "
                     f"near-zero output. Ranges are {conf} bootstrap intervals covering day-to-day noise and the "
                     "uncertainty of the reference. Shortfalls below the trigger are counted as zero, so this is a "
                     "conservative figure.")
    else:
        loss_line = f"[not available: {loss.get('interval_note', 'no interval could be computed')}]"
    bullets([
        loss_line,
        f"<b>Detection limit:</b> {detection_limit_text(noise)} (from clear-day scatter of "
        f"{100 * (noise['day_to_day_sigma_fraction'] or float('nan')):.1f}% per day; {100 * noise['false_alarm_rate']:.0f}% "
        f"false-alarm rate per window, {100 * noise['power']:.0f}% power, starting from a system at its reference). "
        + reliable_text(group_of(run, res)["validation"]) + " Smaller losses are counted as zero.",
        "<b>Method:</b> " + method_text(res, cfg),
    ])
    w = doc.width / 2 - 2
    story.append(Spacer(1, 3))
    story.append(Table([[Image(chart_paths["chart1"], width=w, height=w * 9 / 16),
                         Image(chart_paths["chart2"], width=w, height=w * 9 / 16)]],
                       style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                                         ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)])))

    story.append(Paragraph("3. Likely causes: detected facts vs hypotheses", h2))
    rows = [[Paragraph("<b>Cause</b>", small), Paragraph("<b>Detected facts</b>", small),
             Paragraph("<b>Hypothesis (not proven)</b>", small)]]
    for cause in res["causes"]:
        rows.append([Paragraph(f"{cause['cause']}<br/><i>{cause['status']}</i>", small),
                     Paragraph("<br/>".join(cause["facts"]), small),
                     Paragraph(cause.get("hypothesis") or "—", small)])
    story.append(Table(rows, colWidths=[doc.width * 0.14, doc.width * 0.47, doc.width * 0.39], style=TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEAE0")), ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2)])))

    story.append(Paragraph("4. Recovery verification", h2))
    rec = res["recovery"]
    tested = [e for e in rec["events"] if e.get("testable")]
    lines = [f"Events tested: {rec['event_source']}. {len(tested)} of {len(rec['events'])} event(s) had enough clear "
             f"days on both sides ({cfg['verify']['window_days_before']} days before, "
             f"{cfg['verify']['window_days_after']} after)."]
    if rec["clearest"]:
        e = rec["clearest"]
        lines.append(f"<b>Best-observed event, {e['date']} ({e['label']}): {e['change_pct']:+.1f}% (range "
                     f"{e['change_pct_ci'][0]:+.1f} to {e['change_pct_ci'][1]:+.1f}%), {e['change_kwh_per_day']:+,.1f} kWh/day "
                     f"(range {e['change_kwh_per_day_ci'][0]:+,.1f} to {e['change_kwh_per_day_ci'][1]:+,.1f})</b>, from "
                     f"{e['clear_days_before']} clear days before and {e['clear_days_after']} after. It was chosen for "
                     "having the most clear days, not for its size. "
                     + ("The range excludes zero." if e["distinguishable_from_zero"] else
                        "The range includes zero: no recovery can be claimed for this event."))
        up = sum(1 for t in tested if t["distinguishable_from_zero"] and t["change_pct"] > 0)
        down = sum(1 for t in tested if t["distinguishable_from_zero"] and t["change_pct"] < 0)
        lines.append(f"Across all testable events: {up} clear rise(s), {down} clear fall(s), {len(tested) - up - down} "
                     "not distinguishable from zero. Full list in summary.json.")
    else:
        lines.append("[not available: no event had enough clear days on both sides]")
    bullets(lines)

    story.append(Paragraph("5. Value and verdict", h2))
    money = []
    if "rupees_lost" in econ:
        money.append(f"Loss value at ₹{econ['tariff_inr_per_kwh']:g}/kWh: ₹{econ['rupees_lost']:,.0f} (range ₹"
                     f"{rng(econ.get('rupees_lost_ci'))}) over the period, ₹{econ['rupees_lost_per_month']:,.0f} per month "
                     f"(range ₹{rng(econ.get('rupees_lost_per_month_ci'))}).")
        money.append(f"<b>Verdict: {econ['verdict']}</b>" + (f" — {econ['verdict_basis']}." if "verdict_basis" in econ else ""))
    else:
        money.append(econ["note"])
    em = res["emissions"]
    money.append(f"Emissions: {em['kg_co2']:,.0f} kg CO₂ (range {rng(em.get('kg_co2_ci'))}) at {em['emission_factor_kg_per_kwh']} "
                 f"kg/kWh ({em.get('source')}, {em.get('year')})." if "kg_co2" in em else "Emissions: " + em["note"])
    bullets(money)

    story.append(Paragraph("6. Limits of this report", h2))
    limits = ["Causes are hypotheses; none is proven without a site inspection.",
              "Loss is measured against this system's own best month by month; a loss present from the start, or one "
              "that recurs in the same month every year, is partly or wholly invisible.",
              "Irradiance is a satellite area average; losses below the detection limit are not counted."]
    bullets(limits + res["notes"], small)

    val = group_of(run, res)["validation"]
    if val and not val.get("available"):
        story.append(Paragraph(f"7. {VALIDATION_LABEL}", h2))
        bullets([f"[not available: {val.get('reason')}]"], small)
    if val and val.get("available"):
        story.append(Paragraph(f"7. {VALIDATION_LABEL}", h2))
        same = "this system" if val["system_id"] == res["system_id"] else f"system {val['system_id']}"
        inj, est = val["injected"], val["estimated_after_injection"]
        checks = "; ".join(f"over {w} days: " + ", ".join(
            f"{d['injected_step_pct']:.0f}% step found in {d['detected_share_pct']}%" for d in val["detection_limit_check"]
            if d["window_days"] == int(w)) for w in val["reliable_detection"])
        checks += f" of {val['detection_limit_check'][0]['trials']} trials each"
        bullets([f"On a clean real stretch of {same} ({val['stretch'][0]} to {val['stretch'][1]}), a "
                 f"{inj['ramp_peak_pct']:.0f}% soiling ramp and a {len(inj['outage_days'])}-day outage were injected "
                 f"({inj['injected_kwh']:,.0f} kWh). The unchanged pipeline recovered {val['recovered_kwh']:,.0f} kWh "
                 f"({val['recovered_share_of_injected_pct']:.0f}% of the injected loss; estimate {est['lost_kwh']:,.0f} kWh, "
                 f"range {rng(est['lost_kwh_ci'])}); the known truth is "
                 f"{'inside' if val['injected_within_stated_range'] else 'OUTSIDE'} the stated range. Outage days found: "
                 f"{val['outage_days_detected']}. Detection-limit check: {checks}. These are test results on altered "
                 "data, not field results."
                 + ("" if val["injected_within_stated_range"] else " <b>The method did not pass this check on this "
                    "dataset: treat its sustained-shortfall figures as rough, and its stated range as too narrow.</b>")],
                small)
    doc.build(story)
    return str(path)


# --------------------------------------------------------------------------
# Everything
# --------------------------------------------------------------------------
def write_all(run: dict[str, Any]) -> dict[str, Any]:
    cfg = run["cfg"]
    out_dir = path_of(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = Path(cfg["_root"]) / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {"charts": {}, "reports": {}}
    for res in run["results"]:
        sid = res["system_id"]
        res["sd"].interval.to_csv(processed / f"normalised_{sid}.csv", index=False)
        if res["status"] != "ok":
            written["reports"][sid] = write_pdf(res, run, {}, out_dir)
            continue
        keep = ["energy_kwh", "valid", "ghi", "clear_ghi", "clearness", "t2m_c", "precip_mm", "model_kwh", "pi",
                "clear", "reference", "expected_kwh", "ratio", "smooth_ratio", "flagged", "loss_fraction"]
        table = res["frame"][keep].copy()
        table["outage"] = res["outage"]
        table.insert(0, "data_label", res["data_label"])
        table.round(5).to_csv(out_dir / f"daily_{sid}.csv")
        paths = {
            "chart1": charts.chart_expected_vs_actual(res, cfg["charts"], out_dir / f"chart1_expected_vs_actual_{sid}.png"),
            "chart2": charts.chart_normalised(res, cfg["charts"], cfg["verify"], out_dir / f"chart2_normalised_performance_{sid}.png"),
            "chart3": charts.chart_recovery(res, cfg["charts"], out_dir / f"chart3_recovery_{sid}.png"),
        }
        written["charts"][sid] = paths
        written["reports"][sid] = write_pdf(res, run, paths, out_dir)
    vdir = out_dir / "validation"
    vdir.mkdir(exist_ok=True)
    for stale in ("validation.json", "validation_injected_loss.png"):       # names used before datasets were grouped
        (vdir / stale).unlink(missing_ok=True)
    written["headline_systems"] = []
    for group in run["groups"]:
        val = group["validation"]
        if val and val.get("available"):
            sid = val["system_id"]
            charts.chart_validation(val, cfg["charts"], vdir / f"validation_injected_loss_{sid}.png")
            (vdir / f"validation_{sid}.json").write_text(json.dumps(jsonable(val), indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
        ok = [r for r in run["results"] if r["system_id"] in group["system_ids"] and r["status"] == "ok"]
        ev = headline_event(ok)
        if ok:
            written["headline_systems"].append(ev["system_id"] if ev else ok[0]["system_id"])
    write_summary(run, out_dir)
    written["slide"] = write_slide(run, out_dir)
    write_provenance(run, out_dir)
    write_limitations(run, out_dir)
    return written
