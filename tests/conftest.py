"""Shared fixtures. Synthetic data appears ONLY here and in the tests."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trueyield.config import load_config  # noqa: E402
from trueyield.expected import add_performance_index  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = copy.deepcopy(load_config(ROOT / "config.yaml"))
    c["bootstrap"]["samples"] = 400          # enough for a test, quicker than 2000
    return c


@pytest.fixture()
def synthetic_frame(cfg):
    """Three years of a healthy 10 kWp system: seasonal sun, 2% daily noise."""
    rng = np.random.default_rng(1)
    days = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    season = 5.5 + 2.0 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 80) / 365.0)
    clear = rng.random(len(days)) < 0.7
    clearness = np.where(clear, rng.uniform(0.9, 1.0, len(days)), rng.uniform(0.3, 0.8, len(days)))
    ghi = season * clearness
    model = ghi * 10.0
    frame = pd.DataFrame({
        "energy_kwh": model * 0.80 * (1.0 + rng.normal(0, 0.02, len(days))),
        "valid": True, "ghi": ghi, "clear_ghi": season, "clearness": clearness,
        "t2m_c": 25.0, "precip_mm": 0.0, "model_kwh": model,
    }, index=days)
    return add_performance_index(frame, cfg["expected"])
