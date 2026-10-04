"""Optional single-page viewer: streamlit run app.py

Shows what `python -m trueyield run` already wrote to results/. It computes
nothing itself, so it can never disagree with the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RESULTS = Path(__file__).parent / "results"

st.set_page_config(page_title="TrueYield", layout="wide")
st.title("TrueYield: independent verification of rooftop solar yield")

summary_path = RESULTS / "summary.json"
if not summary_path.exists():
    st.error("No results yet. Run `python -m trueyield run` first.")
    st.stop()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
systems = {f"{s['system_id']}: {s['name']}": s for s in summary["systems"]}
s = systems[st.selectbox("System", list(systems))]

st.info(f"Data label: **{s['data_label']}**")
if s["status"] != "ok":
    st.warning("This system could not be analysed: " + "; ".join(s["honest_gaps"]))
    st.stop()

loss, limit = s["estimated_loss"], s["detection_limit"]
lo, hi = loss["loss_pct_range"] or (None, None)
klo, khi = loss["lost_kwh_range"] or (None, None)
a, b, c = st.columns(3)
a.metric("Estimated loss", f"{loss['loss_pct']:.1f}%", f"range {lo:.1f}–{hi:.1f}%" if lo is not None else "no range",
         delta_color="off")
b.metric("Lost energy", f"{loss['lost_kwh']:,.0f} kWh", f"range {klo:,.0f}–{khi:,.0f} kWh" if klo is not None else "no range",
         delta_color="off")
c.metric("Data completeness", f"{s['completeness']['completeness_pct']}%",
         f"{s['period']['start']} to {s['period']['end']}", delta_color="off")
st.caption("Detection limit: " + limit["statement"] + ".")
st.caption("Verdict: " + str(s["verdict"]))

sid = s["system_id"]
for name, caption in [("chart1_expected_vs_actual", "Expected vs actual daily energy"),
                      ("chart2_normalised_performance", "Normalised performance, events and rain"),
                      ("chart3_recovery", "Before and after the best-observed event")]:
    path = RESULTS / f"{name}_{sid}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")

st.subheader("Likely causes")
for cause in s["cause_hypotheses"]:
    with st.expander(f"{cause['cause']} — {cause['status']}"):
        st.markdown("**Detected facts**")
        for fact in cause["facts"]:
            st.markdown(f"- {fact}")
        st.markdown("**Hypothesis (not proven)**")
        st.markdown(cause.get("hypothesis") or "—")

if s["honest_gaps"]:
    st.subheader("Gaps in this run")
    for gap in s["honest_gaps"]:
        st.markdown(f"- {gap}")

with st.expander("Full summary for this system (JSON)"):
    st.json(s)
