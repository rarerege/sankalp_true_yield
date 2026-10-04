"""Weather from the NASA POWER point API (satellite and reanalysis, free, global).

Daily: all-sky and clear-sky surface irradiation, 2 m temperature, rainfall.
Hourly: the same parameters, used only when sub-daily generation data exists.
Responses are cached under data/cache/ so that a run is reproducible offline.
POWER marks missing values with a fill flag (-999); these become NaN and the
affected days are treated as "cannot be normalised", never interpolated.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

RENAME = {"ALLSKY_SFC_SW_DWN": "ghi", "CLRSKY_SFC_SW_DWN": "clear_ghi",
          "T2M": "t2m_c", "PRECTOTCORR": "precip_mm"}


class WeatherUnavailable(Exception):
    """NASA POWER could not be reached and no cached copy exists."""


def _cached_get(url: str, params: dict[str, Any], cache_dir: Path, wcfg: dict[str, Any]) -> dict[str, Any]:
    key = hashlib.sha1(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:16]
    temporal = url.rstrip("/").split("/")[-2]
    path = cache_dir / f"nasa_power_{temporal}_{params['latitude']}_{params['longitude']}_{params['start']}_{params['end']}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    last = ""
    for attempt in range(wcfg["retries"]):
        try:
            r = requests.get(url, params=params, timeout=wcfg["timeout_s"])
            if r.status_code == 200:
                payload = r.json()
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.RequestException, ValueError) as exc:
            last = repr(exc)
        time.sleep(1.5 * (attempt + 1))
    raise WeatherUnavailable(f"NASA POWER request failed ({last})")


def _to_frame(payload: dict[str, Any], fmt: str, fill: float) -> pd.DataFrame:
    params = payload["properties"]["parameter"]
    df = pd.DataFrame({RENAME.get(k, k): pd.Series(v, dtype=float) for k, v in params.items()})
    df.index = pd.to_datetime(df.index, format=fmt)
    fill = float(payload.get("header", {}).get("fill_value", fill))
    return df.mask(np.isclose(df, fill)).sort_index()


def fetch_daily(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp,
                cache_dir: Path, wcfg: dict[str, Any]) -> pd.DataFrame:
    """Daily weather. `ghi` and `clear_ghi` are in kWh/m2/day, rain in mm/day."""
    payload = _cached_get(f"{wcfg['nasa_power_base']}/daily/point", {
        "parameters": ",".join(wcfg["daily_parameters"]), "community": wcfg["community"],
        "longitude": round(lon, 4), "latitude": round(lat, 4),
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON",
    }, cache_dir, wcfg)
    df = _to_frame(payload, "%Y%m%d", wcfg["missing_flag"])
    df["clearness"] = df["ghi"] / df["clear_ghi"].where(df["clear_ghi"] > 0)
    return df


def fetch_hourly(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp,
                 cache_dir: Path, wcfg: dict[str, Any]) -> pd.DataFrame:
    """Hourly weather indexed by UTC hour start. `ghi` is the hour's mean W/m2.

    Requested one calendar year at a time to keep responses small.
    """
    frames = []
    for year in range(start.year, end.year + 1):
        s = max(start, pd.Timestamp(year, 1, 1))
        e = min(end, pd.Timestamp(year, 12, 31))
        payload = _cached_get(f"{wcfg['nasa_power_base']}/hourly/point", {
            "parameters": ",".join(wcfg["hourly_parameters"]), "community": wcfg["community"],
            "longitude": round(lon, 4), "latitude": round(lat, 4),
            "start": s.strftime("%Y%m%d"), "end": e.strftime("%Y%m%d"), "format": "JSON",
            "time-standard": "UTC",
        }, cache_dir, wcfg)
        frames.append(_to_frame(payload, "%Y%m%d%H", wcfg["missing_flag"]))
    return pd.concat(frames).sort_index()
