"""Steps 7 and 8: rupee value of the loss, a cleaning verdict, and emissions.

All figures inherit the confidence interval of the lost energy. Nothing here
adds information; it only converts units with numbers the user supplied.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def economics(loss: dict[str, Any], site: pd.Series, period_days: int, ecfg: dict[str, Any]) -> dict[str, Any]:
    """Rupees lost per month and a simple verdict on whether cleaning pays.

    Verdict rule: the monthly value of the *moderate sustained* shortfall is
    compared with the cost of one cleaning. Outage days and deep drops (part of
    the system off) are excluded, since cleaning does not fix those. If even the low end of the range exceeds the cost the fix
    likely pays; if even the high end is below it, it does not; otherwise the
    data cannot decide. It assumes a monthly cleaning would recover the whole
    sustained shortfall, which is optimistic if part of it is ageing.
    """
    months = period_days / ecfg["days_per_month"]
    tariff, cost = site.get("tariff_inr_per_kwh"), site.get("cleaning_cost_inr")
    out: dict[str, Any] = {"months": round(months, 1)}
    if _blank(tariff):
        out["note"] = "[not available: tariff_inr_per_kwh not provided in data/systems.csv]"
        out["verdict"] = "[not available: tariff_inr_per_kwh not provided]"
        return out
    tariff = float(tariff)
    out["tariff_inr_per_kwh"] = tariff
    out["rupees_lost"] = round(loss["lost_kwh"] * tariff)
    out["rupees_lost_per_month"] = round(loss["lost_kwh"] * tariff / months)
    if "lost_kwh_ci" in loss:
        out["rupees_lost_ci"] = [round(v * tariff) for v in loss["lost_kwh_ci"]]
        out["rupees_lost_per_month_ci"] = [round(v * tariff / months) for v in loss["lost_kwh_ci"]]
    if _blank(cost):
        out["verdict"] = "[not available: cleaning_cost_inr not provided]"
        return out
    cost = float(cost)
    out["cleaning_cost_inr"] = cost
    out["sustained_rupees_per_month"] = round(loss["moderate_shortfall_kwh"] * tariff / months)
    if "moderate_shortfall_kwh_ci" not in loss:
        out["verdict"] = "uncertain (no confidence interval available)"
        return out
    lo, hi = (v * tariff / months for v in loss["moderate_shortfall_kwh_ci"])
    out["sustained_rupees_per_month_ci"] = [round(lo), round(hi)]
    if lo > cost:
        out["verdict"] = "fix likely pays"
    elif hi < cost:
        out["verdict"] = "fix does not pay"
    else:
        out["verdict"] = "uncertain"
    out["verdict_basis"] = (f"moderate sustained shortfall (outages and deep drops excluded) worth ₹{lo:,.0f}–₹{hi:,.0f} per month "
                            f"vs ₹{cost:,.0f} per cleaning, assuming one cleaning per month")
    return out


def emissions(loss: dict[str, Any], emcfg: dict[str, Any]) -> dict[str, Any]:
    """Avoided-emissions equivalent of the lost energy, if a factor is configured."""
    factor = emcfg.get("emission_factor_kg_per_kwh")
    if _blank(factor):
        return {"note": "[not available: emission_factor_kg_per_kwh is empty in config.yaml; "
                        "no factor is assumed]"}
    factor = float(factor)
    out = {"emission_factor_kg_per_kwh": factor, "source": emcfg.get("emission_factor_source"),
           "year": emcfg.get("emission_factor_year"), "kg_co2": round(loss["lost_kwh"] * factor)}
    if "lost_kwh_ci" in loss:
        out["kg_co2_ci"] = [round(v * factor) for v in loss["lost_kwh_ci"]]
    return out
