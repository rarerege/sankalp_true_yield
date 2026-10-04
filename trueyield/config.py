"""Configuration loading and small shared helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VALIDATION_LABEL = "Method validation with injected loss — not a field result"
NATURAL_RAIN_LABEL = "natural rain event, not a maintenance intervention"


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read config.yaml. All thresholds come from here, never from code."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(Path(path).resolve().parent)
    return cfg


def path_of(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the folder holding config.yaml."""
    p = Path(cfg["_root"]) / cfg["paths"][key]
    return p


def data_label(sources: list[dict[str, Any]] | None, system_id: str) -> str:
    """Return the mandatory data label for one system.

    The label is "Public dataset: <name>" only when one of the SOURCE*.yaml
    files in data/raw/ lists this system. Everything else that the user dropped
    into data/raw/ is "Real system data".
    """
    for source in sources or []:
        if str(system_id) in {str(s) for s in source.get("system_ids", [])}:
            return f"Public dataset: {source['name']}"
    return "Real system data"
