"""Fetch a real, public PV generation dataset when data/raw/ is empty.

Source: NREL PVDAQ, served from the OEDI public data lake (no credentials).
One CSV per system per day is downloaded, cached, and concatenated *unmodified*
into data/raw/pvdaq_system_<id>.csv. Nothing is simulated or filled in: days
that do not exist in PVDAQ simply stay missing and are reported as gaps later.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from .config import path_of


def _get(url: str, tries: int = 25, timeout: tuple[int, int] = (6, 10)) -> bytes | None:
    """GET with short timeouts and retries; returns None for a real 404.

    The length check guards against truncated bodies on a flaky connection.
    """
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                expected = r.headers.get("Content-Length")
                if expected is None or int(expected) == len(r.content):
                    return r.content
        except requests.RequestException:
            continue
    raise ConnectionError(f"Could not fetch after {tries} tries: {url}")


def _list_keys(base: str, prefix: str) -> list[str]:
    """List every object key under an S3 prefix (handles pagination)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{base}?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + requests.utils.quote(token, safe="")
        body = _get(url)
        text = body.decode() if body else ""
        keys += re.findall(r"<Key>([^<]+)</Key>", text)
        m = re.search(r"<NextContinuationToken>([^<]+)<", text)
        if not m:
            return keys
        token = m.group(1)


def _date_of(key: str) -> dt.date | None:
    m = re.search(r"date_(\d{4})_(\d{2})_(\d{2})\.csv$", key)
    return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def fetch_public(cfg: dict[str, Any]) -> dict[str, Any]:
    """Download the configured PVDAQ systems and write provenance files."""
    pub = cfg["public_dataset"]
    base, prefix = pub["base_url"], pub["prefix"]
    start = dt.date.fromisoformat(str(pub["start"]))
    end = dt.date.fromisoformat(str(pub["end"]))
    raw_dir = path_of(cfg, "raw_dir")
    cache = path_of(cfg, "cache_dir") / "pvdaq"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {}
    for sid in pub["system_ids"]:
        meta_path = cache / f"{sid}_system_metadata.json"
        if not meta_path.exists():
            body = _get(f"{base}{prefix}system_metadata/{sid}_system_metadata.json")
            if body is None:
                raise FileNotFoundError(f"PVDAQ has no metadata for system {sid}")
            meta_path.write_bytes(body)
        metadata[str(sid)] = json.loads(meta_path.read_text(encoding="utf-8"))

        keys: list[str] = []
        for year in range(start.year, end.year + 1):
            keys += _list_keys(base, f"{prefix}pvdata/system_id={sid}/year={year}/")
        wanted = sorted(k for k in keys if (d := _date_of(k)) and start <= d <= end)
        sys_cache = cache / f"system_{sid}"
        sys_cache.mkdir(exist_ok=True)

        def grab(key: str) -> None:
            target = sys_cache / f"{_date_of(key).isoformat()}.csv"
            if target.exists():
                return
            body = _get(base + key)
            target.write_bytes(body if body is not None else b"")

        with cf.ThreadPoolExecutor(pub.get("workers", 16)) as pool:
            list(pool.map(grab, wanted))

        frames = []
        for f in sorted(sys_cache.glob("*.csv")):
            if f.stat().st_size == 0:      # PVDAQ holds a few empty day files
                continue
            frames.append(pd.read_csv(io.BytesIO(f.read_bytes())))
        if not frames:
            raise RuntimeError(f"No data downloaded for PVDAQ system {sid}")
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(raw_dir / f"pvdaq_system_{sid}.csv", index=False)
        print(f"PVDAQ system {sid}: {len(frames)} day files, {len(out)} rows")

    source = {
        "kind": "public_dataset",
        "name": pub["name"],
        "url": base + prefix,
        "url_note": "objects under pvdata/system_id=<id>/year=<y>/month=<m>/day=<d>/",
        "site_details": ("capacity, coordinates, tilt, azimuth and start date in data/systems.csv are copied from "
                         "PVDAQ's own system_metadata/<id>_system_metadata.json. Nothing was guessed."),
        "permission_notes": "CC BY 4.0 allows reuse with attribution; cite the dataset as above. Used unmodified.",
        "landing_page": pub["landing_page"],
        "licence": pub.get("licence", ""),
        "citation": pub.get("citation", ""),
        "system_ids": [str(s) for s in pub["system_ids"]],
        "period": [str(start), str(end)],
        "retrieved_on": dt.date.today().isoformat(),
    }
    (raw_dir / "SOURCE.yaml").write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    _write_systems_template(cfg, metadata)
    return source


def _write_systems_template(cfg: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Create data/systems.csv from PVDAQ's own metadata if it does not exist.

    Only fields that PVDAQ publishes are filled. Tariff and cleaning cost are
    left blank on purpose: they are the user's numbers, not the dataset's.
    """
    path = path_of(cfg, "systems_csv")
    if path.exists() and len(pd.read_csv(path)) > 0:      # never overwrite the user's rows
        return
    rows = []
    for sid, meta in metadata.items():
        mount = next(iter(meta.get("Mount", {}).values()), {})
        lon = float(meta["Site"]["longitude"])
        rows.append({
            "system_id": sid,
            "name": re.sub(r"^\[\d+\]\s*", "", meta["System"]["public_name"]),
            "capacity_kwp": meta["System"].get("power", ""),
            "latitude": meta["Site"]["latitude"],
            "longitude": meta["Site"]["longitude"],
            "tilt_deg": mount.get("tilt", ""),
            "azimuth_deg": mount.get("azimuth", ""),
            "commissioning_date": str(meta["System"].get("started_on", ""))[:10],
            "tariff_inr_per_kwh": "",
            "cleaning_cost_inr": "",
            "location": cfg["public_dataset"].get("location_label", ""),
            # PVDAQ timestamps are local standard time; offset from longitude.
            "utc_offset_hours": round(lon / 15.0),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
