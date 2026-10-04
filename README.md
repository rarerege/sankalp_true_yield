# TrueYield

A small Python prototype that independently verifies rooftop solar yield. From inverter exports and free satellite weather it:

1. estimates how much energy a system is losing, with a confidence range;
2. states the smallest loss it could have detected;
3. suggests likely causes, kept apart from the facts that support them;
4. checks whether a fix pays;
5. measures recovery after a dated event (or after heavy rain, if there are no dated events).

Physics and statistics only. There is no machine learning and no LLM call anywhere in the pipeline.

## Honesty rules built into the code

- Generation data are never simulated. Missing or untrustworthy days are flagged and left out, not filled in.
- Every output carries a data label: `Real system data` or `Public dataset: <name>`.
- Every loss or recovery figure is printed with a range.
- Facts and hypotheses appear under separate headings.
- Anything that cannot be computed is written as `[not available: <reason>]`.
- The only altered data are in the unit tests and in the step labelled "Method validation with injected loss — not a field result".

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

On macOS or Linux use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Run

```bash
.venv\Scripts\python -m trueyield run
```

The run prints what the loader inferred about each file, then writes everything to `results/`. A first run fetches weather from NASA POWER and caches it in `data/cache/`; later runs work offline. Expect about two minutes for three systems with four years of data (most of it is the rdtools soiling estimate, which is cached after the first run).

Other commands:

```bash
.venv\Scripts\python -m trueyield templates
```

creates empty `data/systems.csv` and `data/events.csv` templates.

```bash
.venv\Scripts\python -m trueyield fetch-public
```

downloads the public dataset named in `config.yaml` (only needed when `data/raw/` is empty; `run` does this itself).

```bash
.venv\Scripts\python -m pytest tests -q
```

runs the tests.

```bash
.venv\Scripts\python -m streamlit run app.py
```

opens the optional one-page viewer of the results.

If `pip install` fails on Windows with a "No such file or directory" error inside `site-packages`, the project folder's path is too long for Windows. Move the project to a shorter path (for example `C:\trueyield`) or enable Windows long-path support.

## Using your own inverter files

1. **Remove the public data**: delete the `pvdaq_system_*.csv` files and `SOURCE.yaml` from `data/raw/`. `SOURCE.yaml` is what makes outputs say "Public dataset"; without it your files are labelled "Real system data".
2. **Drop your exports into `data/raw/`**: CSV or XLSX, one system per file. The loader accepts:
   - a time column (any of `Timestamp`, `Date`, `Time`, `measured_on`, ...);
   - one of: a lifetime cumulative energy counter, a counter that resets every day ("energy today"), energy per interval, or AC power;
   - daily or sub-daily rows. Units (W, kW, Wh, kWh, MWh) are inferred from the size of the numbers against the nameplate, not from the column name.
   - The system is identified by a `system_id` column, or by the system id appearing in the file name (for example `A1_2024.csv`).
3. **Fill in `data/systems.csv`**, one row per system: `system_id, name, capacity_kwp, latitude, longitude, tilt_deg, azimuth_deg, commissioning_date, tariff_inr_per_kwh, cleaning_cost_inr`. Azimuth is degrees from north (180 = south-facing). Two optional columns: `location` (used on the slide) and `utc_offset_hours` (5.5 for India; otherwise taken from longitude). The run stops if capacity or coordinates are missing; it does not guess them. Tilt, azimuth and commissioning date may be left blank when a source does not publish them: daily mode does not use them, the hourly model and the shading test are then skipped, and the outputs say so. Without tariff or cleaning cost the physics still runs and the rupee lines say "not available".
4. **Optionally fill in `data/events.csv`**: `system_id, date, type, note` with type `cleaning`, `repair` or `other`. If a system has no dated events, heavy-rain days are tested instead and labelled "natural rain event, not a maintenance intervention".
5. **Optionally set the emission factor** in `config.yaml` (`emissions:` block, with source and year). Left empty, the emissions line is skipped with a note.
6. Run, and **read the first lines of the output**: check that the loader inferred the right column, unit and interval.

Not supported yet: one file holding several systems side by side, trackers, east-west arrays. See `results/LIMITATIONS.md`.

## How the method works

| Step | Module | What it does |
|---|---|---|
| Ingest | `ingest.py` | Infers the layout, converts counters or power to interval energy, sums to days, flags gaps, partial days and impossible values. |
| Weather | `weather.py` | NASA POWER irradiance (all-sky, clear-sky), temperature, rain; cached; missing flags become "no data". |
| Expected yield | `expected.py` | A physical model (daily, or hourly with pvlib when sub-daily data exist) gives the performance index = measured ÷ modelled. The reference is the system's own best sustained clear-day performance per calendar month, so a constant bias in satellite irradiance or nameplate cancels out. |
| Detect | `detect.py` | Measures the noise floor, states the detection limit, flags sustained shortfalls, and sizes the lost energy with a block-bootstrap interval. |
| Causes | `causes.py` | Rules for outage, soiling (rdtools SRR), ageing (year-on-year) and shading (hourly dips), each with facts and a hypothesis. |
| Verify | `verify.py` | Clear-day performance before vs after each event, with a bootstrap interval. |
| Economics | `economics.py` | kWh × tariff, cleaning verdict that respects the range, optional emissions. |
| Validation | `validation.py` | Injects a known loss into a clean real stretch and checks the pipeline recovers it. |
| Outputs | `report.py`, `charts.py` | Everything in `results/`. |
| Orchestration | `pipeline.py`, `cli.py` | Runs the steps in order. |

All thresholds are in `config.yaml`, each with a comment. Random seeds are fixed, so a rerun on the same data gives the same numbers.

## How to read each output

All under `results/`:

- **`slide6_values.md`**: five ready-to-paste lines for the pitch deck. The `slide:` block in `config.yaml` chooses which dataset leads, which dataset's validation is cited, and whether line 3 reports a tested event or a planned test. `slide6_values_by_dataset.md` keeps the plain five lines for every dataset for reference.
- **`summary.json`**: everything per system: data label, period, completeness, what the loader inferred, reference method, loss in % and kWh with ranges, detection limit, cause hypotheses with their supporting facts, every recovery test, rupee value and verdict. `portfolio` adds the systems up.
- **`report_<system_id>.pdf`**: a two-page auditable report: what is measured, what is estimated, how sure, and the limits.
- **`chart1_expected_vs_actual_<id>.png`**: navy = expected (the system's own best, adjusted for the weather), amber = actual. Thin lines are the daily values, bold lines their 7-day mean. The shaded gap is energy below expectation.
- **`chart2_normalised_performance_<id>.png`**: 1.00 is the system at its seasonal best. Dots are clear days, the line their running median. Shaded bands are flagged shortfalls, green lines are tested events, small ticks at the bottom are rain days, crosses are days with near-zero output in good sun.
- **`chart3_recovery_<id>.png`**: clear days before (amber) and after (green) the best-observed event, with the change and its range in the subtitle. "Best-observed" means most clear days on both sides, not the biggest effect.
- **`daily_<id>.csv`**: the day-by-day table behind the charts (energy, weather, model, reference, ratio, flags).
- **`validation/`**: the injected-loss test and the detection-limit check. Labelled "Method validation with injected loss — not a field result"; nothing here enters the field results.
- **`data_provenance.md`**: source, licence and permission notes for every dataset.
- **`LIMITATIONS.md`**: what the prototype cannot yet do or prove, plus the gaps of the current run.

Two terms worth knowing before a jury asks:

- **Range**: a 95% bootstrap interval. For the loss it covers day-to-day noise and the uncertainty of the reference; it does not cover a wrong tilt or capacity in `systems.csv`.
- **Detection limit**: the smallest sustained loss the method would flag 80% of the time with a 1% false-alarm rate per window. Smaller losses are counted as zero, so the reported loss is conservative.

## Data in this repository

`data/raw/` currently holds two public datasets, not the author's own systems. See `results/data_provenance.md` for citations.

| Dataset | What it is | Role |
|---|---|---|
| NREL PVDAQ (CC BY 4.0) | Three systems in the Las Vegas area, 5- and 15-minute data, 2016 to 2019 | Method validation: four years, hourly mode, the injected-loss check passes |
| Solar Panel Data, Mendeley Data (CC BY 4.0) | One 120 kW rooftop plant in Jaipur, daily energy for 2022 | Indian case study: daily mode, one year, so results are coarser and the injected-loss check does not pass |

The Jaipur file is prepared by `tools/prepare_jaipur.py`, which downloads the original spreadsheet to `data/source/` and copies each day's date and plant total into `data/raw/JAIPUR1_daily.csv` without changing any value.

Each public dataset has its own `SOURCE*.yaml` in `data/raw/` listing the systems it covers. Systems are grouped by dataset in the outputs: totals are never added across datasets, `slide6_values.md` has one block of five lines per dataset, and the method is validated separately on each (`results/validation/`).
