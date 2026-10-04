# Limitations

Plain-language list of what this prototype cannot yet do or prove.

## What the method cannot prove

- **It never proves a cause.** Soiling, ageing, shading and faults are reported as hypotheses that fit a pattern. Only a site visit can confirm them.
- **"Loss" means loss against the system's own best.** If a system was never clean, or has a permanent fault from day one, its "best" already includes that loss and the method cannot see it.
- **Seasonal self-calibration can hide a seasonal problem.** The reference is set per calendar month. A loss that returns in the same month every year (for example heavy dust every May) lowers that month's reference and is partly hidden. More years with at least one clean spell per month reduce this.
- **Small losses are invisible.** Anything below the stated detection limit cannot be told apart from noise, and is counted as zero. The true loss can therefore be larger than reported.
- **The stated detection limit assumes the system starts at its reference.** When a system is running a few percent above its reference (a better-than-usual month), a loss of that size can go unseen. The validation stretch is chosen close to the reference for this reason; on a stretch running above it, less of an injected loss is recovered.
- **The rdtools soiling estimate is inflated by scatter.** rdtools looks for upward steps, and day-to-day scatter alone produces steps. On a test input with no soiling and 3% daily scatter it reported about 3% loss. Each estimate is therefore shown next to a shuffle check (the same values in random time order), and soiling is only called indicated when the estimate is clearly above it. The detected loss in the headline does not use rdtools and is not affected.
- **A slow unexplained drift cannot be attributed.** Month-to-month wander of a few percent may be soiling, haze the satellite misjudged, or the simple model. It is flagged only above the trigger and never proven.
- **A natural rain event is not a cleaning.** Rain may clean poorly or coincide with other changes; a before/after rise after rain shows the method can see a recovery, not that maintenance works.
- **The cleaning verdict is a screening rule.** It assumes one cleaning a month recovers the whole sustained shortfall. If part of that shortfall is ageing, cleaning recovers less.

## Data limits

- **Satellite irradiance, not a site sensor.** NASA POWER is an area average over tens of kilometres. Local cloud, haze or dust storms add day-to-day scatter; this is the main reason for the detection limit.
- **Clear days only.** The state of the system is judged on clear days and assumed to hold on the days in between. Long cloudy spells (for example a monsoon) leave stretches that cannot be assessed.
- **Missing days are unknown, not zero.** Days without valid data are left out of both loss and expected energy. A real outage during a data gap is not counted.
- **An outage and a logging fault look the same** when the logger records zeros.
- **Timestamps are assumed to be local standard time** with a fixed offset from longitude unless `utc_offset_hours` is given. Daylight-saving shifts are not corrected; a clock check detects them and the shading test is then reported as not assessable. Daily totals are not affected.

## Loader limits

- One system per file (or a `system_id` column with one value). Wide files with one column per system are not supported.
- Lifetime counters and counters that reset every day ("energy today") are both supported. For a daily-reset counter, energy produced between the last reading before a reset and the reset itself is not seen; this is zero when the reset happens at night.
- Units are inferred from magnitude against the nameplate. A wrong capacity in `systems.csv` gives wrong units; check the inferred layout printed at the start of each run.

## Modelling limits

- Fixed-tilt arrays with a single orientation only. No trackers, no split east-west arrays.
- No inverter clipping model, no spectral or reflection (incidence-angle) model, no snow model. Constant effects are absorbed by the reference; effects that vary within a month are not.
- Shading is tested only above 15° sun elevation and only as a recurring hourly dip; partial or seasonal shading at low sun is not assessed.
- Ageing needs more than two years of data and is a single linear rate.
- **Without rdtools the soiling and ageing estimates are cruder.** The built-in soiling fallback only treats heavy rain and recorded cleanings as cleaning events. On the public data used here it gave 0.8-1.8% against 4.3-5.9% from rdtools; lowering its rain threshold did not close the gap, and most of the difference is the scatter-driven inflation of rdtools described above. The built-in ageing fallback gives a similar rate with a wider range.
- The bootstrap range covers day-to-day noise and the uncertainty of the reference. It does not cover systematic errors such as a wrong tilt in `systems.csv` or a drifting satellite product.

## Not yet built

- No live data connection; files are dropped in by hand.
- No cost model beyond one cleaning price per system.
- Emissions are only reported if a grid emission factor is entered in `config.yaml`.
- **One year of data is the bare minimum, and it is weak.** Each month's reference then comes from that same month, so a loss lasting most of a month lowers its own reference and is partly hidden. On the one-year Jaipur dataset the injected-loss validation does not pass; see `results/validation/`.
- Validated on one stretch of one system per dataset (see the method-validation section of the report), not across climates. The validation recovers most, not all, of an injected slow ramp: the part of a ramp that stays inside the noise is not counted.

## Gaps in this particular run

- System JAIPUR1: Tilt and azimuth are not provided by the data source. Daily mode does not use them (the seasonal reference absorbs the orientation), so no value was assumed.
- System JAIPUR1: Commissioning date is not provided by the data source; it is not used in any calculation.
- System JAIPUR1: Months with too few clear days for their own reference (borrowed from neighbouring months): [7, 8, 12].
- System 1419: 188 day(s) in deep-drop episodes (part of the system off) are excluded from the recovery, soiling, ageing and shading tests, so those tests describe the system while it was otherwise working.
- Economics: system 34: tariff_inr_per_kwh, cleaning_cost_inr not provided; rupee figures are not available for it.
- Economics: system 1419: tariff_inr_per_kwh, cleaning_cost_inr not provided; rupee figures are not available for it.
- Economics: system 1420: tariff_inr_per_kwh, cleaning_cost_inr not provided; rupee figures are not available for it.
- Economics: system JAIPUR1: tariff_inr_per_kwh, cleaning_cost_inr not provided; rupee figures are not available for it.
