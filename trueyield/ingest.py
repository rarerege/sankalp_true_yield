"""Step 1: ingest and clean inverter exports.

Whatever layout arrives in data/raw/ is normalised to one long table,
`system_id, timestamp, energy_kwh`, where each row is the energy produced in
the interval *ending* at `timestamp`. Daily totals and quality flags are then
derived from that table. Nothing is filled in: a missing or untrustworthy day
is flagged and excluded, never estimated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib

SYSTEM_COLUMNS = [
    "system_id", "name", "capacity_kwp", "latitude", "longitude", "tilt_deg",
    "azimuth_deg", "commissioning_date", "tariff_inr_per_kwh", "cleaning_cost_inr",
]
# Needed by every analysis. Tilt, azimuth and commissioning date may be blank
# when the source does not publish them: daily mode does not use them, and the
# pipeline says so in its output instead of guessing.
PHYSICAL_COLUMNS = SYSTEM_COLUMNS[:5]
OPTIONAL_SITE_COLUMNS = SYSTEM_COLUMNS[5:8]
ECONOMIC_COLUMNS = SYSTEM_COLUMNS[8:]
EVENT_COLUMNS = ["system_id", "date", "type", "note"]


class SystemsFileError(Exception):
    """data/systems.csv is missing or lacks site details. Never guess these."""


# --------------------------------------------------------------------------
# Site and event tables
# --------------------------------------------------------------------------
def ensure_templates(systems_csv: Path, events_csv: Path) -> list[str]:
    """Create empty templates for missing input tables; return what was created."""
    created = []
    if not systems_csv.exists():
        systems_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SYSTEM_COLUMNS).to_csv(systems_csv, index=False)
        created.append(str(systems_csv))
    if not events_csv.exists():
        events_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=EVENT_COLUMNS).to_csv(events_csv, index=False)
        created.append(str(events_csv))
    return created


def load_systems(systems_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read systems.csv.

    Missing capacity or coordinates raise SystemsFileError (the pipeline must
    stop and the user must supply them). Tilt, azimuth and commissioning date
    may be blank; the hourly model then cannot run and daily mode is used.
    Missing tariff or cleaning cost is returned as a list of warnings: the
    physics still runs, the rupee lines do not.
    """
    if not systems_csv.exists():
        raise SystemsFileError(f"{systems_csv} does not exist")
    df = pd.read_csv(systems_csv, dtype={"system_id": str})
    missing_cols = [c for c in SYSTEM_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemsFileError(f"{systems_csv} lacks columns: {missing_cols}")
    if df.empty:
        raise SystemsFileError(f"{systems_csv} has no rows; add one row per system")
    problems = []
    for _, row in df.iterrows():
        blank = [c for c in PHYSICAL_COLUMNS if pd.isna(row[c]) or str(row[c]).strip() == ""]
        if blank:
            problems.append(f"system {row['system_id']}: missing {blank}")
    if problems:
        raise SystemsFileError("; ".join(problems))
    warnings = []
    for _, row in df.iterrows():
        blank = [c for c in ECONOMIC_COLUMNS if pd.isna(row[c]) or str(row[c]).strip() == ""]
        if blank:
            warnings.append(f"system {row['system_id']}: {', '.join(blank)} not provided")
    df["system_id"] = df["system_id"].astype(str).str.strip()
    for c in ["capacity_kwp", "latitude", "longitude", "tilt_deg", "azimuth_deg"] + ECONOMIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "utc_offset_hours" not in df.columns:
        df["utc_offset_hours"] = np.nan
    # Standard-time offset from longitude when the user gives none (15 deg = 1 h).
    df["utc_offset_hours"] = pd.to_numeric(df["utc_offset_hours"], errors="coerce").fillna(
        (df["longitude"] / 15.0).round())
    if "location" not in df.columns:
        df["location"] = ""
    df["location"] = df["location"].fillna("").astype(str)
    return df.set_index("system_id", drop=False), warnings


def load_events(events_csv: Path) -> pd.DataFrame:
    """Read the optional table of dated interventions."""
    if not events_csv.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev = pd.read_csv(events_csv, dtype={"system_id": str})
    if ev.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["type"] = ev["type"].fillna("other").str.strip().str.lower()
    ev["note"] = ev.get("note", "").fillna("")
    return ev


# --------------------------------------------------------------------------
# Layout inference
# --------------------------------------------------------------------------
@dataclass
class Layout:
    """What the loader inferred about one raw file (reported to the user)."""
    file: str
    system_id: str
    timestamp_col: str
    kind: str                      # cumulative_counter | daily_reset_counter | interval_energy | power
    value_cols: list[str]
    unit: str
    scale_to_kwh_or_kw: float
    interval_minutes: float
    notes: list[str] = field(default_factory=list)
    row_scale: pd.Series | None = None      # per-row scale when the unit changes over time

    def describe(self) -> str:
        return (f"{self.file}: system {self.system_id}; time column '{self.timestamp_col}'; "
                f"{self.kind} from {self.value_cols} in {self.unit}; "
                f"{self.interval_minutes:g}-minute steps. " + " ".join(self.notes)).strip()


def _base(col: str) -> str:
    """Lower-case column name without PVDAQ's trailing '__<sensor id>'."""
    return re.sub(r"__\d+$", "", str(col).strip().lower())


def _find_timestamp(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Locate the time column by name first, then by parseability."""
    names = sorted(df.columns, key=lambda c: 0 if re.search(
        r"measured_on|timestamp|datetime|date|time", _base(c)) else 1)
    for col in names:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.9:
            return col, parsed
    raise ValueError("no parseable timestamp column found")


def _is_counter(values: pd.Series, min_share: float) -> bool:
    """A lifetime counter (almost) never decreases and ends above its start.

    Interval energy and power go up and down all the time, so they fail this
    test. Counters that reset every day are recognised by `_resets_daily`.
    """
    v = values.dropna()
    if len(v) < 10:
        return False
    steps = v.diff().dropna()
    return bool((steps >= 0).mean() > min_share and v.iloc[-1] > v.iloc[0])


def _resets_daily(ts: pd.Series, values: pd.Series, icfg: dict[str, Any]) -> bool:
    """Recognise an "energy today" counter: a sawtooth that restarts each day.

    Three things must hold: the value mostly rises; it drops about once per
    day; and after a drop it restarts near zero. Power and interval energy
    fail the first test, a lifetime counter fails the second.
    """
    s = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=pd.DatetimeIndex(ts)).dropna().sort_index()
    n_days = s.index.normalize().nunique()
    if len(s) < 20 or n_days < 3:
        return False
    steps = s.diff().dropna()
    drops = steps < 0
    if not drops.any() or (steps >= 0).mean() < icfg["daily_reset_min_nondecreasing_share"]:
        return False
    lo, hi = icfg["daily_reset_drops_per_day_range"]
    if not lo <= drops.sum() / n_days <= hi:
        return False
    before = s.shift(1).reindex(steps.index)[drops]
    restart = (s.reindex(steps.index)[drops] / before.where(before > 0)).median()
    return bool(restart <= icfg["daily_reset_max_restart_fraction"])


def _pick_scale(magnitude: float, capacity: float, target: tuple[float, float],
                options: dict[str, float]) -> tuple[str, float, bool]:
    """Choose the unit whose scaled magnitude is physically plausible.

    Column names lie (PVDAQ has a 'kwac' column holding watts), so units are
    inferred from size relative to the nameplate capacity instead.
    """
    centre = np.sqrt(target[0] * target[1])
    best = min(options.items(), key=lambda kv: abs(np.log(max(magnitude * kv[1], 1e-12) / capacity / centre)))
    ratio = magnitude * best[1] / capacity
    return best[0], best[1], bool(target[0] <= ratio <= target[1])


def _scale_over_time(power: pd.Series, ts: pd.Series, capacity: float, options: dict[str, float],
                     icfg: dict[str, Any]) -> tuple[pd.Series, list[str]]:
    """Detect a unit change inside one column (seen in real logger exports).

    Each calendar month gets the unit that makes its peak power plausible for
    the nameplate. If more than one unit appears, every day is then assigned
    the largest of those units that keeps its own peak inside the plausible
    range, which places the change on the right day. Returns the scale per
    row and a description of each change.
    """
    s = pd.Series(power.to_numpy(), index=pd.DatetimeIndex(ts)).dropna()
    plausible = tuple(icfg["plausible_peak_power_fraction"])
    monthly_peak = s.groupby(s.index.to_period("M")).quantile(icfg["unit_peak_quantile"])
    names = {}
    for month, peak in monthly_peak.items():
        if peak > 0:
            name, _, ok = _pick_scale(float(peak), capacity, plausible, options)
            if ok:
                names[month] = name
    used = sorted(set(names.values()), key=lambda n: -options[n])
    if len(used) <= 1:
        return pd.Series(dtype=float), []
    month_unit = pd.Series(names).reindex(monthly_peak.index).ffill().bfill()
    day_peak = s.groupby(s.index.normalize()).max()
    day_month = day_peak.index.to_period("M")
    day_unit = pd.Series(day_month.map(month_unit).to_numpy(), index=day_peak.index, dtype=object)
    # Only the two months around a change are re-examined day by day, so a
    # single dark day elsewhere can never be mistaken for a unit change.
    for i in range(1, len(month_unit)):
        a, b = month_unit.iloc[i - 1], month_unit.iloc[i]
        if a == b:
            continue
        around = day_month.isin([month_unit.index[i - 1], month_unit.index[i]])
        big, small = sorted((a, b), key=lambda n: -options[n])
        fits_big = day_peak * options[big] <= plausible[1] * capacity
        day_unit[around] = np.where(fits_big[around], big, small)
        # Dark days fit either unit; give them the unit of the nearest bright day.
        ambiguous = around & (day_peak * options[big] < icfg["dark_day_peak_fraction"] * capacity)
        day_unit[ambiguous] = np.nan
        day_unit = day_unit.ffill().bfill()
    changes = [f"{day_unit.iloc[i - 1]} until {day_unit.index[i - 1].date()}, {day_unit.iloc[i]} from "
               f"{day_unit.index[i].date()}" for i in range(1, len(day_unit)) if day_unit.iloc[i] != day_unit.iloc[i - 1]]
    row_days = pd.DatetimeIndex(ts).normalize()
    row_scale = pd.Series(row_days.map(day_unit.map(options)).to_numpy(dtype=float), index=power.index)
    return row_scale, changes


def infer_layout(df: pd.DataFrame, file_name: str, systems: pd.DataFrame,
                 icfg: dict[str, Any]) -> tuple[Layout, pd.Series]:
    """Infer which columns hold time and energy, and in which units."""
    yield_range = tuple(icfg["plausible_daily_yield_kwh_per_kwp"])
    power_range = tuple(icfg["plausible_peak_power_fraction"])
    ts_col, ts = _find_timestamp(df)
    notes: list[str] = []

    sid_col = next((c for c in df.columns if _base(c) in ("system_id", "systemid", "plant_id", "site_id")), None)
    if sid_col is not None and df[sid_col].nunique() == 1:
        system_id = str(df[sid_col].dropna().iloc[0]).split(".")[0]
    else:
        # A system_id from systems.csv that appears as a whole word in the file name.
        stem = Path(file_name).stem
        match = [s for s in systems.index
                 if re.search(rf"(?<![A-Za-z0-9]){re.escape(str(s))}(?![A-Za-z0-9])", stem, re.IGNORECASE)]
        if match:
            system_id = max(match, key=len)
            notes.append("system_id taken from the file name.")
        elif len(systems) == 1:
            system_id = str(systems.index[0])
            notes.append("no system_id in the file; assigned to the only system in systems.csv.")
        else:
            system_id = stem
    if system_id not in systems.index:
        raise SystemsFileError(f"{file_name}: system '{system_id}' is not in systems.csv")
    capacity = float(systems.loc[system_id, "capacity_kwp"])

    numeric = [c for c in df.columns if c not in (ts_col, sid_col) and pd.api.types.is_numeric_dtype(df[c])]
    is_dc = lambda c: bool(re.search(r"(^|_)dc(_|$)", _base(c)))
    energy_like = [c for c in numeric if re.search(r"kwh|energy|yield|generation|e_total|etotal|e_day", _base(c))
                   and not is_dc(c)]
    power_like = [c for c in numeric if re.search(r"power|(^|_)p_?ac|(^|_)kw($|_|ac)", _base(c))
                  and not re.search(r"factor|kwh", _base(c)) and not is_dc(c)]

    step = ts.sort_values().diff().dropna()
    interval_min = float(step[step > pd.Timedelta(0)].median() / pd.Timedelta(minutes=1)) if len(step) else 1440.0

    counters = [c for c in energy_like if _is_counter(df[c], icfg["counter_min_nondecreasing_share"])]
    if counters:
        span_days = max((ts.max() - ts.min()) / pd.Timedelta(days=1), 1.0)
        per_day = sum(float(df[c].dropna().iloc[-1] - df[c].dropna().iloc[0]) for c in counters) / span_days
        unit, scale, ok = _pick_scale(per_day, capacity, yield_range, {"kWh": 1.0, "Wh": 1e-3, "MWh": 1e3})
        if not ok:
            notes.append("counter magnitude is unusual for this capacity; unit choice is uncertain.")
        if len(counters) > 1:
            notes.append(f"{len(counters)} counters (one per inverter) are summed.")
        return Layout(file_name, system_id, ts_col, "cumulative_counter", counters, unit, scale,
                      interval_min, notes), ts

    resetting = ([c for c in energy_like if _resets_daily(ts, df[c], icfg)] if interval_min < 1440 - 1 else [])
    if resetting:
        # Each day's energy is the highest value the counter reached that day.
        day = pd.DatetimeIndex(ts).normalize()
        per_day = sum(float(pd.to_numeric(df[c], errors="coerce").groupby(day.to_numpy()).max().mean())
                      for c in resetting)
        unit, scale, ok = _pick_scale(per_day, capacity, yield_range, {"kWh": 1.0, "Wh": 1e-3, "MWh": 1e3})
        notes.append("this counter resets every day; energy is rebuilt from its steps, and the first reading "
                     "after each reset counts as energy since the reset.")
        if not ok:
            notes.append("counter magnitude is unusual for this capacity; unit choice is uncertain.")
        if len(resetting) > 1:
            notes.append(f"{len(resetting)} counters (one per inverter) are summed.")
        return Layout(file_name, system_id, ts_col, "daily_reset_counter", resetting, unit, scale,
                      interval_min, notes), ts

    if energy_like:
        col = energy_like[0]
        per_day = float(df[col].sum()) / max((ts.max() - ts.min()) / pd.Timedelta(days=1), 1.0)
        unit, scale, ok = _pick_scale(per_day, capacity, yield_range, {"kWh": 1.0, "Wh": 1e-3, "MWh": 1e3})
        if not ok:
            notes.append("energy magnitude is unusual for this capacity; unit choice is uncertain.")
        return Layout(file_name, system_id, ts_col, "interval_energy", [col], unit, scale, interval_min, notes), ts

    if power_like:
        total = [c for c in power_like if _base(c) in ("ac_power", "power", "pac", "p_ac", "ac_power_kw", "ac_power_hw")]
        cols = total[:1] if total else power_like
        if not total and len(cols) > 1:
            notes.append(f"{len(cols)} power columns are summed.")
        power = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(cols))
        options = {"kW": 1.0, "hW (x100 W)": 0.1, "W": 1e-3}
        peak = float(power.quantile(icfg["unit_peak_quantile"]))
        unit, scale, ok = _pick_scale(peak, capacity, power_range, options)
        notes.append(f"unit inferred from peak power vs {capacity:g} kWp nameplate, not from the column name.")
        layout = Layout(file_name, system_id, ts_col, "power", cols, unit, scale, interval_min, notes)
        row_scale, changes = _scale_over_time(power, ts, capacity, options, icfg)
        if changes:
            layout.row_scale = row_scale
            layout.unit = "mixed units"
            notes.append("the unit of this column changes over time: " + "; ".join(changes) + ".")
        elif not ok:
            notes.append("peak power is unusual for this capacity; unit choice is uncertain.")
        notes.append("energy = power x sampling interval; missing samples stay missing.")
        return layout, ts

    raise ValueError(f"{file_name}: no energy or AC power column recognised among {list(df.columns)}")


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------
def counter_to_interval(timestamps: pd.Series, counter: pd.Series, max_gap_hours: float = 18.0,
                        resets_daily: bool = False) -> pd.DataFrame:
    """Convert a cumulative energy counter to energy per interval.

    Each row gets the energy accumulated since the previous *valid* reading.
    Two cases are deliberately left as NaN rather than guessed:
      * the counter went down (reset or replacement): the step is unknown;
      * the previous reading is more than `max_gap_hours` old: the energy is
        real but cannot be assigned to a particular day.
    With `resets_daily=True` (an "energy today" counter) a drop is expected:
    the first reading after it is taken as the energy since the reset.
    Returns a frame with `timestamp, energy, reset, long_gap`.
    """
    s = pd.DataFrame({"timestamp": pd.to_datetime(timestamps).to_numpy(),
                      "value": pd.to_numeric(counter, errors="coerce").to_numpy()})
    s = s.dropna().drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    step = s["value"].diff()
    elapsed_h = s["timestamp"].diff() / pd.Timedelta(hours=1)
    reset = step < 0
    long_gap = elapsed_h > max_gap_hours
    if resets_daily:
        # An "energy today" counter restarts from zero, so the first reading
        # after a drop is itself the energy produced since the reset.
        energy = step.where(~reset, s["value"]).mask(long_gap)
    else:
        energy = step.mask(reset | long_gap)
    return pd.DataFrame({"timestamp": s["timestamp"], "energy": energy,
                         "reset": reset.fillna(False), "long_gap": long_gap.fillna(False),
                         "elapsed_h": elapsed_h})


@dataclass
class SystemData:
    """Cleaned data for one system."""
    system_id: str
    layouts: list[Layout]
    interval: pd.DataFrame          # system_id, timestamp, energy_kwh (normalised output)
    daily: pd.DataFrame             # one row per calendar day, with quality flags
    hourly: pd.Series | None        # kWh per local-standard-time hour, NaN if incomplete
    completeness: dict[str, Any]
    resolution: str                 # "hourly" (sub-daily input) or "daily"


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _sun_window(days: pd.DatetimeIndex, lat: float, lon: float, utc_offset: float,
                min_elevation: float) -> pd.DataFrame:
    """Start and end of usable daylight (sun above `min_elevation` degrees) for each day."""
    grid = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq="5min", inclusive="left")
    utc = (grid - pd.Timedelta(hours=utc_offset)).tz_localize("UTC")
    up = pvlib.solarposition.get_solarposition(utc, lat, lon)["apparent_elevation"].to_numpy() > min_elevation
    lit = pd.Series(grid[up], index=grid[up].normalize())
    return pd.DataFrame({"rise": lit.groupby(level=0).min(),
                         "set": lit.groupby(level=0).max() + pd.Timedelta(minutes=5)}).reindex(days)


def _sample_hours(index: pd.DatetimeIndex, gap_factor: float) -> tuple[pd.Series, pd.Series]:
    """Hours of production each reading stands for, and each day's sampling step.

    Loggers change their sampling rate over the years, so the step is measured
    per day. A reading stands for the time since the previous reading, unless
    that is longer than `gap_factor` steps (a logging gap), in which case it stands for
    one step only and the gap stays uncovered.
    """
    stamps = index.to_series()
    elapsed = stamps.diff() / pd.Timedelta(hours=1)
    day_step = elapsed.groupby(index.normalize()).transform("median").fillna(elapsed.median())
    hours = elapsed.where(elapsed <= gap_factor * day_step, day_step).fillna(day_step)
    return hours, day_step


def _step_note(day_step: pd.Series) -> str | None:
    """Describe a sampling interval that varies between days."""
    per_day = (day_step.groupby(day_step.index.normalize()).first() * 60).round().value_counts()
    per_day = per_day[per_day >= 5]
    if len(per_day) <= 1:
        return None
    return "sampling interval varies: " + ", ".join(f"{m:g} min on {n} days" for m, n in per_day.items()) + "."


def clean_system(system_id: str, pieces: list[tuple[Layout, pd.DataFrame]], site: pd.Series,
                 icfg: dict[str, Any]) -> SystemData:
    """Turn inferred raw columns into interval energy, daily totals and flags."""
    capacity = float(site["capacity_kwp"])
    frames = []
    reset_n = long_gap_n = 0
    for layout, df in pieces:
        ts = pd.to_datetime(df[layout.timestamp_col], errors="coerce")
        if layout.kind in ("cumulative_counter", "daily_reset_counter"):
            daily_reset = layout.kind == "daily_reset_counter"
            parts = []
            for col in layout.value_cols:
                c = counter_to_interval(ts, df[col] * layout.scale_to_kwh_or_kw, icfg["max_counter_gap_hours"],
                                        resets_daily=daily_reset)
                if not daily_reset:                  # a daily reset is expected, not an anomaly
                    reset_n += int(c["reset"].sum())
                long_gap_n += int(c["long_gap"].sum())
                parts.append(c.set_index("timestamp")[["energy", "elapsed_h"]])
            energy = sum(p["energy"] for p in parts) if len(parts) > 1 else parts[0]["energy"]
            hours = parts[0]["elapsed_h"].reindex(energy.index)
            _, day_step = _sample_hours(pd.DatetimeIndex(energy.index), icfg["logging_gap_step_factor"])
        else:
            raw = df[layout.value_cols].apply(pd.to_numeric, errors="coerce")
            scale = layout.row_scale if layout.row_scale is not None else layout.scale_to_kwh_or_kw
            val = raw.sum(axis=1, min_count=len(layout.value_cols)) * scale
            energy = pd.Series(val.to_numpy(), index=ts.to_numpy())
            # Rows without a reading (e.g. a second sensor logged a moment later)
            # are dropped first so they cannot displace a real reading.
            energy = energy[~energy.index.isna()].dropna()
            energy = energy[~energy.index.duplicated(keep="last")].sort_index()
            hours, day_step = _sample_hours(pd.DatetimeIndex(energy.index), icfg["logging_gap_step_factor"])
            if layout.kind == "power":
                energy = energy * hours
        if len(day_step.dropna()):
            layout.interval_minutes = float(round(day_step.groupby(day_step.index.normalize()).first().median() * 60, 3))
            note = _step_note(day_step) if layout.interval_minutes < 1440 - 1 else None
            if note and note not in layout.notes:
                layout.notes.append(note)
        frames.append(pd.DataFrame({"energy_kwh": energy, "hours": hours}))
    iv = pd.concat(frames).sort_index()
    iv = iv[~iv.index.duplicated(keep="last")]
    interval_min = float(np.median([l.interval_minutes for l, _ in pieces]))
    subdaily = interval_min < 1440 - 1
    iv["hours"] = iv["hours"].fillna(interval_min / 60.0)
    if not subdaily:
        # One row per day stands for that day only, even when the day before is missing.
        iv["hours"] = interval_min / 60.0

    # Impossible values: more energy than nameplate allows, or large negatives.
    limit = capacity * iv["hours"] * icfg["max_capacity_factor"]
    small_neg = (iv["energy_kwh"] < 0) & (iv["energy_kwh"] >= -icfg["negative_clip_fraction"] * limit)
    iv.loc[small_neg, "energy_kwh"] = 0.0
    impossible = (iv["energy_kwh"] < 0) | (iv["energy_kwh"] > limit)
    iv["impossible"] = impossible
    iv.loc[impossible, "energy_kwh"] = np.nan

    # Each stamp marks the end of its interval: it covers (t - hours, t].
    t1 = pd.DatetimeIndex(iv.index)
    t0 = t1 - pd.to_timedelta(iv["hours"].to_numpy(), unit="h")
    mid = t0 + (t1 - t0) / 2 if subdaily else t1
    iv["date"] = mid.normalize()
    days = pd.date_range(iv["date"].min(), iv["date"].max(), freq="D")

    hourly = None
    straddle_n = 0
    if subdaily:
        all_days = pd.date_range(days[0] - pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=1), freq="D")
        sun = _sun_window(all_days, float(site["latitude"]), float(site["longitude"]),
                          float(site["utc_offset_hours"]), icfg["daylight_min_sun_elevation_deg"])

        def lit_hours(day: pd.DatetimeIndex) -> np.ndarray:
            """Hours of each reading's interval that fall inside that day's daylight."""
            rise, sunset = sun["rise"].reindex(day).to_numpy(), sun["set"].reindex(day).to_numpy()
            span = np.minimum(t1.to_numpy(), sunset) - np.maximum(t0.to_numpy(), rise)
            return np.clip(span / np.timedelta64(1, "h"), 0.0, None)

        d0, d1 = t0.normalize(), t1.normalize()
        on_end_day, on_start_day = lit_hours(d1), np.where(d0 != d1, lit_hours(d0), 0.0)
        # A reading whose interval holds daylight of two different days cannot be
        # split between them: its energy is real but undatable.
        both = icfg["straddle_min_daylight_hours"]
        straddle = (on_end_day > both) & (on_start_day > both)
        straddle_n = int((straddle & iv["energy_kwh"].notna().to_numpy()).sum())
        iv.loc[straddle, "energy_kwh"] = np.nan
        ok = iv["energy_kwh"].notna().to_numpy()
        covered = (pd.Series(np.where(ok, on_end_day, 0.0)).groupby(d1.to_numpy()).sum()
                   .add(pd.Series(np.where(ok, on_start_day, 0.0)).groupby(d0.to_numpy()).sum(), fill_value=0.0))
        daylight = (sun["set"] - sun["rise"]) / pd.Timedelta(hours=1)
        coverage = (covered.reindex(days).fillna(0.0) / daylight.reindex(days)).clip(upper=1.0)

        # Hourly energy is trusted only when the hour is fully covered by
        # readings that each stand for no more than that hour.
        by_hour = iv.assign(lit=iv["hours"].where(iv["energy_kwh"].notna(), 0.0)).groupby(mid.floor("h"))
        full = by_hour["lit"].sum().between(*icfg["hourly_coverage_range"]) & (by_hour["hours"].max() <= 1.0)
        hourly = by_hour["energy_kwh"].sum(min_count=1).where(full)
        hourly = hourly.reindex(pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h"))

    daily = pd.DataFrame(index=days)
    daily.index.name = "date"
    grouped = iv.groupby("date")
    daily["energy_kwh_raw"] = grouped["energy_kwh"].sum(min_count=1).reindex(days)
    daily["n_samples"] = grouped["energy_kwh"].count().reindex(days).fillna(0).astype(int)
    daily["n_impossible"] = grouped["impossible"].sum().reindex(days).fillna(0).astype(int)
    daily["coverage"] = coverage if subdaily else np.where(daily["n_samples"] > 0, 1.0, np.nan)

    if subdaily:
        # Drop edge days that hold only a stray midnight sample of the next/previous day.
        while len(daily) > 2 and daily["n_samples"].iloc[0] <= 2:
            daily = daily.iloc[1:]
        while len(daily) > 2 and daily["n_samples"].iloc[-1] <= 2:
            daily = daily.iloc[:-1]
        days = daily.index

    # Gap = nothing recorded while the sun was up (a lone midnight sample does not count).
    daily["flag_gap"] = (daily["n_samples"] == 0) | (subdaily & (daily["coverage"].fillna(0) == 0))
    daily["flag_partial"] = (~daily["flag_gap"]) & (daily["coverage"].fillna(0) < icfg["min_daylight_coverage"])
    daily["flag_impossible"] = (daily["n_impossible"] > 0) | (
        daily["energy_kwh_raw"] / capacity > icfg["max_daily_specific_yield"])
    daily["valid"] = ~(daily["flag_gap"] | daily["flag_partial"] | daily["flag_impossible"])
    daily["energy_kwh"] = daily["energy_kwh_raw"].where(daily["valid"])

    n = len(daily)
    completeness = {
        "period_start": str(days[0].date()), "period_end": str(days[-1].date()),
        "calendar_days": int(n),
        "valid_days": int(daily["valid"].sum()),
        "gap_days": int(daily["flag_gap"].sum()),
        "partial_days": int(daily["flag_partial"].sum()),
        "impossible_days": int(daily["flag_impossible"].sum()),
        "completeness_pct": round(100.0 * float(daily["valid"].mean()), 1),
        "counter_resets": reset_n, "counter_unassignable_gaps": long_gap_n + straddle_n,
        "impossible_interval_values": int(impossible.sum()),
    }
    out = iv[["energy_kwh"]].reset_index(names="timestamp")
    out.insert(0, "system_id", system_id)
    return SystemData(system_id, [l for l, _ in pieces], out, daily, hourly, completeness,
                      "hourly" if subdaily else "daily")


def ingest_all(raw_dir: Path, systems: pd.DataFrame, icfg: dict[str, Any]) -> dict[str, SystemData]:
    """Read every CSV/XLSX in data/raw/, infer its layout and clean it."""
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".xls"))
    by_system: dict[str, list[tuple[Layout, pd.DataFrame]]] = {}
    for path in files:
        df = _read_any(path)
        layout, _ = infer_layout(df, path.name, systems, icfg)
        by_system.setdefault(layout.system_id, []).append((layout, df))
    return {sid: clean_system(sid, pieces, systems.loc[sid], icfg) for sid, pieces in by_system.items()}
