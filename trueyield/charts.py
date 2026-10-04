"""Charts. One visual language throughout:

  navy   expected / reference
  amber  actual / measured
  green  recovery events

Every figure carries the data label in its corner. Text is always dark ink,
never a series colour, and each series is labelled directly as well as in the
legend so that identity never depends on colour alone.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import VALIDATION_LABEL

INK, MUTED, GRID = "#1B1B1B", "#5A5A5A", "#DDD8CC"


def _figure(ccfg: dict[str, Any], title: str, subtitle: str, label: str):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=ccfg["dpi"])
    fig.patch.set_facecolor(ccfg["background"])
    ax.set_facecolor(ccfg["background"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.80, bottom=0.235)
    fig.text(0.085, 0.955, title, fontsize=21, fontweight="bold", va="top")
    fig.text(0.085, 0.893, textwrap.fill(subtitle, 118), fontsize=13, color=MUTED, va="top", linespacing=1.35)
    # The data label sits alone in the bottom corner, below the legend rows.
    fig.text(0.965, 0.022, "\n".join(textwrap.fill(part, 110) for part in label.split("\n")), fontsize=11.5, color=INK, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=MUTED, linewidth=0.8))
    return fig, ax


def _legend(ax, ncol: int = 3, fontsize: float = 12.5) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.085), ncol=ncol, frameon=False, fontsize=fontsize,
              columnspacing=2.2, handletextpad=0.6, borderaxespad=0)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _years(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def chart_expected_vs_actual(res: dict[str, Any], ccfg: dict[str, Any], path: Path) -> str:
    """Chart 1: expected vs actual daily energy, gap shaded."""
    f = res["frame"]
    n = int(ccfg.get("energy_smoothing_days", 7))
    both = f["valid"] & f["expected_kwh"].notna() & f["energy_kwh"].notna()
    exp = f["expected_kwh"].where(both).rolling(n, center=True, min_periods=max(1, n // 2)).mean()
    act = f["energy_kwh"].where(both).rolling(n, center=True, min_periods=max(1, n // 2)).mean()
    loss = res["loss"]
    rng = (f"range {loss['loss_pct_ci'][0]:.1f}–{loss['loss_pct_ci'][1]:.1f}%" if "loss_pct_ci" in loss
           else "range not available")
    smooth = f"Thin lines are daily energy; bold lines are its {n}-day running mean" if n > 1 else "Daily energy"
    fig, ax = _figure(ccfg, "Expected vs actual daily energy",
                      f"{res['name']} (system {res['system_id']}). {smooth}. Estimated loss "
                      f"{loss['loss_pct']:.1f}% ({rng}) of expected energy over the period.", res["data_label"])
    ax.fill_between(exp.index, act, exp, where=(act < exp).to_numpy(), color=ccfg["amber"], alpha=0.30,
                    linewidth=0, label="Gap (actual below expected)")
    if n > 1:                                   # the daily values themselves, behind the running means
        ax.plot(f.index, f["expected_kwh"].where(both), color=ccfg["navy"], linewidth=0.6, alpha=0.35)
        ax.plot(f.index, f["energy_kwh"].where(both), color=ccfg["amber"], linewidth=0.6, alpha=0.55)
    ax.plot(exp.index, exp, color=ccfg["navy"], linewidth=2.0, label="Expected (own best, weather-adjusted)")
    ax.plot(act.index, act, color=ccfg["amber"], linewidth=2.4, label="Actual (measured)")
    ax.set_ylabel("Energy per day (kWh)")
    ax.set_ylim(bottom=0)
    _years(ax)
    _legend(ax)
    return _save(fig, path)


def chart_normalised(res: dict[str, Any], ccfg: dict[str, Any], vcfg: dict[str, Any], path: Path) -> str:
    """Chart 2: normalised performance over time with events and rain marked."""
    f = res["frame"]
    clear = f["ratio"].where(f["clear"] & ~res["outage"])
    floor = float(np.nanmin(f["smooth_ratio"])) if f["smooth_ratio"].notna().any() else 0.7
    lo, hi = (0.70 if floor >= 0.74 else max(0.0, np.floor(floor * 10) / 10 - 0.05)), 1.12
    step = (hi - lo) / 0.42                      # marker rows scale with the axis range
    fig, ax = _figure(ccfg, "Weather-normalised performance",
                      f"{res['name']} (system {res['system_id']}). 1.00 = the system's own sustained best for the "
                      "season. Dots are clear days; the line is their running median.", res["data_label"])
    flagged = f["flagged"].to_numpy()
    ax.fill_between(f.index, lo, hi, where=flagged, color=ccfg["amber"], alpha=0.16, linewidth=0,
                    label="Sustained shortfall flagged")
    ax.axhline(1.0, color=ccfg["navy"], linewidth=2.0, label="Reference (own best)")
    ax.scatter(clear.index, clear.clip(lo, hi), s=9, color=ccfg["amber"], alpha=0.55, linewidths=0,
               label="Clear-day performance")
    ax.plot(f.index, f["smooth_ratio"], color=ccfg["amber"], linewidth=2.4)
    ax.plot(f.index, f["smooth_ratio"], color=INK, linewidth=0.7, alpha=0.6)
    events = [e for e in res["recovery"]["events"]]
    for i, e in enumerate(events):
        ax.axvline(pd.Timestamp(e["end"]), color=ccfg["green"], linewidth=1.6, alpha=0.9,
                   label=("Heavy rain (natural cleaning)" if "rain" in e["type"] else "Maintenance event") if i == 0 else None)
    rain = f.index[(f["precip_mm"] >= ccfg["rain_marker_mm"]).to_numpy()]
    ax.plot(rain, np.full(len(rain), lo + 0.008 * step), linestyle="none", marker="|", markersize=9, color=ccfg["navy"],
            alpha=0.55, label=f"Rain day (≥ {ccfg['rain_marker_mm']:g} mm)")
    out_days = f.index[res["outage"].to_numpy()]
    if len(out_days):
        ax.plot(out_days, np.full(len(out_days), lo + 0.03 * step), linestyle="none", marker="x", markersize=8,
                color=INK, label="Near-zero output, good sun")
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Performance ÷ reference")
    _years(ax)
    _legend(ax)
    below = int((clear < lo).sum())
    if below:
        ax.text(0.995, 0.985, f"{below} clear day(s) below {lo:.2f} are drawn at the bottom edge",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=MUTED)
    return _save(fig, path)


def chart_recovery(res: dict[str, Any], ccfg: dict[str, Any], path: Path) -> str:
    """Chart 3: before/after plot for the clearest event."""
    ev = res["recovery"]["clearest"]
    if ev is None:
        fig, ax = _figure(ccfg, "Recovery test", f"{res['name']} (system {res['system_id']}). No event could be "
                          "tested.", res["data_label"])
        reasons = sorted({e.get("reason", "") for e in res["recovery"]["events"]}) or ["no dated or heavy-rain events in the data"]
        ax.axis("off")
        ax.text(0.5, 0.5, "No before/after test was possible:\n" + "\n".join(reasons[:4]), ha="center", va="center",
                fontsize=15, color=MUTED, transform=ax.transAxes)
        return _save(fig, path)
    f = res["frame"]
    ratio = f["ratio"].where(f["clear"] & ~res["fault_days"])
    start, end = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    before = ratio.loc[[pd.Timestamp(d) for d in ev["before_days"]]]
    after = ratio.loc[[pd.Timestamp(d) for d in ev["after_days"]]]
    kind = "natural rain event" if "rain" in ev["type"] else f"{ev['type']} event"
    verdict = "" if ev["distinguishable_from_zero"] else " The range includes zero."
    sub = (f"{res['name']} (system {res['system_id']}). Clear days in the {ev['window_days'][0]} days before and "
           f"{ev['window_days'][1]} days after. Change {ev['change_pct']:+.1f}% (range {ev['change_pct_ci'][0]:+.1f} "
           f"to {ev['change_pct_ci'][1]:+.1f}%).{verdict}")
    fig, ax = _figure(ccfg, f"Before and after the {kind} of {ev['date']}", sub, res["data_label"])
    x = lambda idx: [(d - end).days if d > end else (d - start).days for d in idx]
    ax.axvspan(-0.5, 0.5, color=ccfg["green"], alpha=0.22, linewidth=0)
    ax.axvline(0, color=ccfg["green"], linewidth=2.2)
    ax.scatter(x(before.index), before, s=90, color=ccfg["amber"], edgecolor=INK, linewidth=0.8, zorder=3,
               label=f"Clear days before (mean {before.mean():.3f})")
    ax.scatter(x(after.index), after, s=90, color=ccfg["green"], edgecolor=INK, linewidth=0.8, zorder=3, marker="s",
               label=f"Clear days after (mean {after.mean():.3f})")
    wb, wa = ev["window_days"]
    ax.hlines(before.mean(), -wb, -0.5, color=ccfg["amber"], linewidth=2.6)
    ax.hlines(before.mean(), -wb, -0.5, color=INK, linewidth=0.7, alpha=0.6)
    ax.hlines(after.mean(), 0.5, wa, color=ccfg["green"], linewidth=2.6)
    ax.axhline(1.0, color=ccfg["navy"], linewidth=1.6, linestyle=(0, (5, 4)), label="Reference (own best)")
    tag = ev["label"] if "rain" in ev["type"] else "dated maintenance event"
    extra = f"{ev['rain_mm']:.0f} mm rain; " if "rain_mm" in ev else ""
    ax.text(0.5, 0.97, f"{extra}{tag}", transform=ax.transAxes, ha="center", va="top", fontsize=12, color=MUTED)
    span = max(float(pd.concat([before, after]).max()), 1.0) - float(pd.concat([before, after]).min())
    ax.set_ylim(float(pd.concat([before, after]).min()) - 0.25 * span - 0.01,
                max(float(pd.concat([before, after]).max()), 1.0) + 0.3 * span + 0.01)
    ax.set_xlim(-wb - 0.8, wa + 0.8)
    ax.set_xlabel("Days relative to the event", labelpad=2)
    ax.set_ylabel("Performance ÷ reference")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.13), ncol=3, frameon=False, fontsize=12.5,
              columnspacing=2.2, borderaxespad=0)
    return _save(fig, path)


def chart_validation(val: dict[str, Any], ccfg: dict[str, Any], path: Path) -> str:
    """Method-validation figure. Labelled as such in title and corner."""
    fr = val["_frames"]
    s0, s1 = fr["stretch"]
    pad = pd.Timedelta(days=20)
    orig = fr["original"].loc[s0 - pad:s1 + pad]
    mod = fr["modified"].loc[s0 - pad:s1 + pad]
    est, inj = val["estimated_after_injection"], val["injected"]
    verdict = "inside" if val["injected_within_stated_range"] else "OUTSIDE"
    fig, ax = _figure(ccfg, "Method validation with injected loss — not a field result",
                      f"System {val['system_id']}: {inj['injected_kwh']:,.0f} kWh of known loss was injected into a "
                      f"clean real stretch; the pipeline recovered {val['recovered_kwh']:,.0f} kWh "
                      f"({val['recovered_share_of_injected_pct']:.0f}%). The known truth is {verdict} the stated range.",
                      f"{VALIDATION_LABEL}\nBase data: {val['data_label']}")
    injected = (1.0 - fr["factor"].loc[orig.index]).clip(lower=0)
    truth = (1.0 - injected).where(injected > 0)
    ax.axhline(1.0, color=ccfg["navy"], linewidth=2.0, label="Reference (own best)")
    ax.plot(truth.index, truth.clip(lower=0.7), color=ccfg["navy"], linewidth=2.0, linestyle=(0, (5, 4)),
            label="Injected loss (known truth)")
    clear = mod["ratio"].where(mod["clear"])
    ax.scatter(clear.index, clear.clip(0.7, 1.12), s=14, color=ccfg["amber"], alpha=0.6, linewidths=0,
               label="Clear days after injection")
    ax.plot(mod.index, 1.0 - mod["loss_fraction"].where(mod["loss_fraction"] > 0), color=ccfg["amber"], linewidth=2.6,
            label="Loss recovered by the pipeline")
    for d in inj["outage_days"]:
        ax.axvline(pd.Timestamp(d), color=INK, linewidth=1.0, alpha=0.6)
    if inj["outage_days"]:
        ax.text(pd.Timestamp(inj["outage_days"][-1]) + pd.Timedelta(days=2), 0.72,
                f"injected outage\n({val['outage_days_detected']} days detected)", fontsize=11.5, color=MUTED, va="bottom")
    ax.set_ylim(0.7, 1.12)
    ax.set_ylabel("Performance ÷ reference")
    _years(ax)
    ax.set_xlim(s0 - pad, s1 + pad)
    _legend(ax, ncol=2)
    return _save(fig, path)
