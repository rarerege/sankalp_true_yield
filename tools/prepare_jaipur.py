"""Prepare the public Jaipur rooftop dataset for TrueYield.

Source: Parashar, B. & Verma, R. (2024), "Solar Panel Data", Mendeley Data, V1,
doi:10.17632/82jswfyz5p.1, licence CC BY 4.0.

What this script does, and nothing more:
  1. downloads the original spreadsheet to data/source/ (checked against the
     SHA-256 hash published by Mendeley Data) unless it is already there;
  2. copies the date and the plant's daily total ("Main", kWh) into
     data/raw/JAIPUR1_daily.csv, because the original sheet has six title rows
     and separate Month / Day columns that the generic loader cannot read;
  3. writes data/raw/SOURCE_jaipur.yaml so every output is labelled as a public dataset;
  4. adds the plant to data/systems.csv with only the details the source publishes.

No value is changed, filled in or invented. Run from the project folder:
    python tools/prepare_jaipur.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://data.mendeley.com/public-files/datasets/82jswfyz5p/files/"
       "18ed9e38-4dbc-473e-b599-8ba235943761/file_downloaded")
SHA256 = "0d5d2b8a3c9c4af56e7e78d57cda35a3b0afa16a8f2a164cceff024ac7341a94"
SYSTEM_ID = "JAIPUR1"
YEAR = 2022                      # stated in the dataset description; the sheet holds month and day only


def main() -> None:
    original = ROOT / "data" / "source" / "jaipur_mendeley_82jswfyz5p_v1.xlsx"
    original.parent.mkdir(parents=True, exist_ok=True)
    if not original.exists():
        original.write_bytes(requests.get(URL, timeout=120).content)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{original} does not match the published hash; refusing to use it.")

    sheet = pd.read_excel(original, header=None)
    body = sheet.iloc[7:, :5].copy()                       # row 7 on: Month, Day, Inv-1, Inv-2, Main
    body.columns = ["month", "day", "inv1_kwh", "inv2_kwh", "main_kwh"]
    body = body[pd.to_numeric(body["month"], errors="coerce").notna() & pd.to_numeric(body["day"], errors="coerce").notna()]
    date = pd.to_datetime(dict(year=YEAR, month=body["month"].astype(int), day=body["day"].astype(int)))
    tidy = pd.DataFrame({"Date": date.dt.strftime("%Y-%m-%d"), "Energy (kWh)": pd.to_numeric(body["main_kwh"], errors="coerce")})
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(raw / f"{SYSTEM_ID}_daily.csv", index=False)

    (raw / "SOURCE_jaipur.yaml").write_text(yaml.safe_dump({
        "kind": "public_dataset",
        "name": "Solar Panel Data, Jaipur rooftop plant (Mendeley Data)",
        "url": URL,
        "landing_page": "https://data.mendeley.com/datasets/82jswfyz5p/1",
        "licence": "Creative Commons Attribution 4.0 (CC BY 4.0)",
        "citation": ("Parashar, Bhupender; Verma, Richa (2024), \"Solar Panel Data\", Mendeley Data, V1, "
                     "doi: 10.17632/82jswfyz5p.1"),
        "system_ids": [SYSTEM_ID],
        "period": [str(date.min().date()), str(date.max().date())],
        "retrieved_on": dt.date.today().isoformat(),
        "site_details": ("Plant name (Fitpack Textiles, Jaipur), coordinates and the two inverter capacities "
                         "(20 kW + 100 kW) are taken from the dataset. Capacity in data/systems.csv is the 120 kW "
                         "inverter total; the DC array size is not published. Tilt, azimuth and commissioning date "
                         "are not published and are left blank, not guessed."),
        "preparation": ("tools/prepare_jaipur.py copies each day's date and plant total ('Main', kWh) from the original "
                        "sheet (kept unmodified in data/source/) into data/raw/JAIPUR1_daily.csv. The year 2022 comes "
                        "from the dataset description. No value is altered. Zero-energy days (27 in December, which "
                        "the authors describe as a plant breakdown, and 3 in September) are kept as recorded."),
        "permission_notes": "CC BY 4.0 allows reuse with attribution; cite the dataset as above.",
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")

    systems_csv = ROOT / "data" / "systems.csv"
    systems = pd.read_csv(systems_csv, dtype=str, keep_default_na=False)
    if SYSTEM_ID not in set(systems["system_id"]):
        row = {c: "" for c in systems.columns}
        row.update({"system_id": SYSTEM_ID, "name": "Fitpack Textiles rooftop, Jaipur", "capacity_kwp": "120",
                    "latitude": "26.7664", "longitude": "75.8326", "location": "Jaipur, Rajasthan, India",
                    "utc_offset_hours": "5.5"})
        pd.concat([systems, pd.DataFrame([row])], ignore_index=True).to_csv(systems_csv, index=False)
    print(f"{len(tidy)} daily rows written to {raw / (SYSTEM_ID + '_daily.csv')}")


if __name__ == "__main__":
    main()
