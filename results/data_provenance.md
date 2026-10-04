# Data provenance

Generated 2026-10-04.

## Generation data: public dataset — NREL PVDAQ (Photovoltaic Data Acquisition) via OEDI data lake

- **Name:** NREL PVDAQ (Photovoltaic Data Acquisition) via OEDI data lake
- **Exact source URL:** https://oedi-data-lake.s3.us-west-2.amazonaws.com/pvdaq/csv/ (objects under pvdata/system_id=<id>/year=<y>/month=<m>/day=<d>/)
- **Landing page:** https://data.openei.org/submissions/4568
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **Citation:** Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D. (2021). Photovoltaic Data Acquisition (PVDAQ) Public Datasets. [Data set]. Open Energy Data Initiative (OEDI). NREL. https://doi.org/10.25984/1846021
- **Systems used:** 34, 1419, 1420
- **Period:** 2016-01-01 to 2019-12-31
- **Retrieved on:** 2026-10-03
- **Site details:** capacity, coordinates, tilt, azimuth and start date in data/systems.csv are copied from PVDAQ's own system_metadata/<id>_system_metadata.json. Nothing was guessed.
- **Permission notes:** CC BY 4.0 allows reuse with attribution; cite the dataset as above. Used unmodified. These are measurements from real systems. They are **not** the user's own systems.
- **Label on all outputs:** `Public dataset: NREL PVDAQ (Photovoltaic Data Acquisition) via OEDI data lake`

## Generation data: public dataset — Solar Panel Data, Jaipur rooftop plant (Mendeley Data)

- **Name:** Solar Panel Data, Jaipur rooftop plant (Mendeley Data)
- **Exact source URL:** https://data.mendeley.com/public-files/datasets/82jswfyz5p/files/18ed9e38-4dbc-473e-b599-8ba235943761/file_downloaded
- **Landing page:** https://data.mendeley.com/datasets/82jswfyz5p/1
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Citation:** Parashar, Bhupender; Verma, Richa (2024), "Solar Panel Data", Mendeley Data, V1, doi: 10.17632/82jswfyz5p.1
- **Systems used:** JAIPUR1
- **Period:** 2022-01-01 to 2022-12-31
- **Retrieved on:** 2026-10-04
- **Site details:** Plant name (Fitpack Textiles, Jaipur), coordinates and the two inverter capacities (20 kW + 100 kW) are taken from the dataset. Capacity in data/systems.csv is the 120 kW inverter total; the DC array size is not published. Tilt, azimuth and commissioning date are not published and are left blank, not guessed.
- **Preparation:** tools/prepare_jaipur.py copies each day's date and plant total ('Main', kWh) from the original sheet (kept unmodified in data/source/) into data/raw/JAIPUR1_daily.csv. The year 2022 comes from the dataset description. No value is altered. Zero-energy days (27 in December, which the authors describe as a plant breakdown, and 3 in September) are kept as recorded.
- **Permission notes:** CC BY 4.0 allows reuse with attribution; cite the dataset as above. These are measurements from real systems. They are **not** the user's own systems.
- **Label on all outputs:** `Public dataset: Solar Panel Data, Jaipur rooftop plant (Mendeley Data)`

## Weather data

- **Name:** NASA POWER (Prediction Of Worldwide Energy Resources), daily and hourly point API
- **URL:** https://power.larc.nasa.gov/api/temporal/daily/point and /hourly/point
- **Parameters:** ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, PRECTOTCORR (sources: CERES SYN1deg for irradiance, MERRA-2 for temperature and precipitation)
- **Licence:** NASA open data, free to use without restriction; acknowledgement requested: "These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program."
- **Cached copies:** `data/cache/nasa_power_*.json` (exact responses used in this run)
- **Resolution caveat:** grid cells are about 1° (irradiance) and 0.5° × 0.625° (meteorology); values are area averages, not site measurements.

## Synthetic data

None in any field result. The only altered data are in the step labelled "Method validation with injected loss — not a field result" (`results/validation/`), where a known loss is subtracted from a copy of real data to test the method, and in the unit tests.
