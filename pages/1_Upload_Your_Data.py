"""Upload page: CSV/Excel -> automatic column matching -> cleaning pipeline -> analysis."""
import calendar
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import upload_pipeline as up

st.set_page_config(page_title="Upload Your Data", layout="wide")

PLOT_CONFIG = {"displayModeBar": False}
ACCENT = "#6366f1"
SAMPLE_PATH = ROOT / "data" / "processed" / "flights_sample.csv"
CATEGORY_COLORS = {"Early / on time": "#10b981", "Minor": "#fbbf24",
                   "Moderate": "#f97316", "Severe": "#ef4444"}


@st.cache_data(show_spinner="Reading file...")
def load_raw(data: bytes, filename: str, sheet):
    return up.read_table(data, filename, sheet)


@st.cache_data(show_spinner="Reading sample...")
def load_sample():
    return up.read_table(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)


@st.cache_data(show_spinner="Preparing download...")
def to_csv_bytes(_df: pd.DataFrame, cache_key: str) -> bytes:
    return _df.to_csv(index=False).encode("utf-8")


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0))
    return fig


st.title("Upload Your Data")
st.caption("Upload any flight-delay file (CSV or Excel). The pipeline detects your columns, "
           "cleans the data, logs every step, and builds the analysis automatically.")

with st.expander("What data do I need? (read this first)", expanded=True):
    st.markdown(
        "This tool analyses **flight-delay data**: one row per flight, with at least the "
        "**airline**, the **origin airport** and the **arrival delay in minutes**. "
        "Everything else is optional, and every extra column you include unlocks more analysis."
    )
    st.dataframe(up.requirements_table(), hide_index=True)
    g1, g2 = st.columns(2)
    g1.download_button("Download a template file (CSV)", up.template_csv(),
                       file_name="flight_delay_template.csv", mime="text/csv")
    g2.markdown(
        "**No data yet?** Choose *Use the built-in 2015 US sample* below, or download real "
        "files from the US Bureau of Transportation Statistics (On-Time Performance) or "
        "Kaggle (2015 Flight Delays and Cancellations)."
    )
    st.caption("Accepted: .csv or .xlsx, one flight per row, column names in the first row. "
               "Names don't have to match exactly; common variants are detected automatically. "
               "This app does not save your file.")

# ----------------------------- Step 1: load ------------------------------
st.header("1. Choose your data")
source = st.radio("Data source", ["Upload a file", "Use the built-in 2015 US sample"],
                  horizontal=True)
raw, name = None, None
if source == "Upload a file":
    uploaded = st.file_uploader("CSV or Excel (.xlsx) file", type=["csv", "xlsx"])
    if uploaded is not None:
        data = uploaded.getvalue()
        sheet = None
        if uploaded.name.lower().endswith(".xlsx"):
            sheets = up.excel_sheets(data)
            sheet = st.selectbox("Excel sheet", sheets) if len(sheets) > 1 else sheets[0]
        try:
            raw, name = load_raw(data, uploaded.name, sheet), uploaded.name
        except Exception as exc:  # unreadable or corrupt file
            st.error(f"Could not read this file: {exc}")
else:
    raw, name = load_sample(), "flights_sample.csv (built-in)"

if raw is None:
    st.info("Upload a file to begin, or switch to the built-in sample to see how it works.")
    st.stop()

st.caption(f"{name}: {len(raw):,} rows x {raw.shape[1]} columns")
with st.expander("Preview the first rows", expanded=False):
    st.dataframe(raw.head(10))

# ----------------------------- Column matching (automatic) ---------------
mapping = up.detect_columns(list(raw.columns))
missing_keys = [k for k, (_, req, _) in up.FIELDS.items() if req and not mapping[k]]
if missing_keys:
    # Only if auto-detection failed: ask for the missing required columns.
    st.warning("We couldn't find: " + ", ".join(up.FIELDS[k][0] for k in missing_keys) +
               ". Please choose the matching column from your file. "
               "Open 'What data do I need?' above to see the expected format.")
    options = ["(none)"] + list(raw.columns)
    pick_cols = st.columns(len(missing_keys))
    for col, key in zip(pick_cols, missing_keys):
        choice = col.selectbox(up.FIELDS[key][0] + " *", options, key=f"map_{key}_{name}")
        mapping[key] = None if choice == "(none)" else choice
    if any(not mapping[k] for k in missing_keys):
        st.stop()
else:
    st.caption(f"{sum(1 for v in mapping.values() if v)} columns were matched automatically.")

# ----------------------------- Step 2: cleaning options ------------------
st.header("2. Cleaning options")
c1, c2, c3, c4 = st.columns(4)
threshold = c1.slider("Flight is 'delayed' if arrival delay is above (min)", 5, 60, 15)
drop_cancelled = c2.checkbox("Remove cancelled flights", True,
                             disabled=mapping["cancelled"] is None)
drop_dups = c3.checkbox("Remove exact duplicate rows", True)
cap = c4.number_input("Exclude delays beyond +/- (min). 0 = keep all", 0, 10000, 0, step=60)

if st.button("Run pipeline", type="primary"):
    try:
        with st.spinner("Cleaning and analysing..."):
            clean, report = up.clean_data(raw, mapping, threshold=threshold,
                                          drop_cancelled=drop_cancelled,
                                          drop_duplicates=drop_dups,
                                          max_abs_delay=cap or None)
        st.session_state["upload_result"] = {
            "clean": clean, "report": report, "threshold": threshold, "name": name}
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get("upload_result")
if result is None:
    st.stop()
if result["name"] != name:
    st.info("A different file is selected. Press **Run pipeline** to analyse it.")
    st.stop()

clean, report, thr = result["clean"], result["report"], result["threshold"]
if clean.empty:
    st.error("No rows left after cleaning. Check that your file has the columns "
             "listed in 'What data do I need?'.")
    st.stop()

# ----------------------------- Results -----------------------------------
st.header("3. Results")
k = up.kpis(clean)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Flights analysed", f"{k['flights']:,}")
m2.metric(f"Delay rate (>{thr} min)", f"{k['delay_rate']:.1f}%")
m3.metric("Mean delay", f"{k['mean_delay']:+.1f} min")
m4.metric("Median delay", f"{k['median_delay']:+.1f} min")
m5.metric("90th percentile", f"{k['p90_delay']:.0f} min")

min_n = up.adaptive_min_flights(len(clean))
tab_clean, tab_over, tab_rank, tab_time, tab_sim = st.tabs(
    ["Cleaning report", "Overview & insights", "Rankings", "Time patterns", "What-if & export"])

with tab_clean:
    a, b, c = st.columns(3)
    a.metric("Rows uploaded", f"{report.rows_in:,}")
    b.metric("Rows after cleaning", f"{report.rows_out:,}")
    c.metric("Rows kept", f"{report.rows_out / report.rows_in * 100:.1f}%")
    st.subheader("What the pipeline did")
    st.dataframe(pd.DataFrame(report.steps), hide_index=True)
    if report.notes:
        st.subheader("Data-quality notes")
        for note in report.notes:
            st.warning(note)
    else:
        st.success("No data-quality problems detected.")
    st.subheader("Column mapping used")
    st.dataframe(pd.DataFrame({"Field": [up.FIELDS[f][0] for f in report.mapping],
                               "Your column": list(report.mapping.values())}),
                 hide_index=True)

with tab_over:
    st.subheader("Key findings")
    for line in up.make_insights(clean):
        st.markdown("- " + line)
    left, right = st.columns(2)
    with left:
        cat = clean["DELAY_CATEGORY"].value_counts().reindex(up.CATEGORY_ORDER).dropna()
        fig = px.pie(values=cat.values, names=cat.index, hole=0.55,
                     color=cat.index, color_discrete_map=CATEGORY_COLORS,
                     title="Delay severity")
        st.plotly_chart(style(fig), config=PLOT_CONFIG)
    with right:
        hist = clean[clean["ARRIVAL_DELAY"].between(-30, 180)]
        fig = px.histogram(hist, x="ARRIVAL_DELAY", nbins=70,
                           color_discrete_sequence=[ACCENT],
                           title="Arrival delay distribution (-30 to 180 min)")
        fig.update_layout(xaxis_title="Minutes vs. schedule (negative = early)",
                          yaxis_title="Flights")
        st.plotly_chart(style(fig), config=PLOT_CONFIG)
    causes = up.cause_breakdown(clean)
    if causes is not None:
        fig = px.bar(causes.sort_values("Minutes"), x="Minutes", y="Cause", orientation="h",
                     color_discrete_sequence=[ACCENT], title="Root causes (total delay minutes on late flights)")
        st.plotly_chart(style(fig, 320), config=PLOT_CONFIG)
    else:
        st.info("No delay-cause columns were found, so root-cause analysis is skipped.")

with tab_rank:
    st.caption(f"Groups with fewer than {min_n} flights are hidden so tiny samples don't dominate.")
    left, right = st.columns(2)
    with left:
        g = up.group_stats(clean, "AIRLINE", min_n).sort_values("Delay rate (%)")
        fig = px.bar(g, x="Delay rate (%)", y="AIRLINE", orientation="h",
                     color_discrete_sequence=[ACCENT], title="Delay rate by airline")
        st.plotly_chart(style(fig, max(320, 26 * len(g) + 80)), config=PLOT_CONFIG)
    with right:
        top_n = st.slider("Airports to show", 5, 30, 15)
        g = up.group_stats(clean, "ORIGIN", min_n).sort_values("Delay rate (%)").tail(top_n)
        fig = px.bar(g, x="Delay rate (%)", y="ORIGIN", orientation="h",
                     color_discrete_sequence=["#ef4444"], title="Highest delay rate by origin airport")
        st.plotly_chart(style(fig, max(320, 26 * len(g) + 80)), config=PLOT_CONFIG)
    routes = up.top_routes(clean, min_n)
    if routes is not None:
        st.subheader("Routes with the highest delay rate")
        st.dataframe(routes.round(1), hide_index=True)

with tab_time:
    shown = False
    if "TIME_OF_DAY" in clean and clean["TIME_OF_DAY"].notna().any():
        g = up.group_stats(clean, "TIME_OF_DAY")
        g["TIME_OF_DAY"] = pd.Categorical(g["TIME_OF_DAY"], up.TIME_ORDER, ordered=True)
        fig = px.bar(g.sort_values("TIME_OF_DAY"), x="TIME_OF_DAY", y="Delay rate (%)",
                     color_discrete_sequence=[ACCENT], title="Delay rate by departure window")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(style(fig), config=PLOT_CONFIG)
        shown = True
    left, right = st.columns(2)
    if "DAY_NAME" in clean and clean["DAY_NAME"].notna().any():
        g = up.group_stats(clean, "DAY_NAME")
        g["DAY_NAME"] = pd.Categorical(g["DAY_NAME"], up.DAY_ORDER, ordered=True)
        fig = px.bar(g.sort_values("DAY_NAME"), x="DAY_NAME", y="Delay rate (%)",
                     color_discrete_sequence=[ACCENT], title="Delay rate by weekday")
        fig.update_layout(xaxis_title="")
        left.plotly_chart(style(fig), config=PLOT_CONFIG)
        shown = True
    if "MONTH" in clean and clean["MONTH"].notna().any():
        g = up.group_stats(clean, "MONTH").sort_values("MONTH")
        g["Month"] = g["MONTH"].astype(int).map(lambda v: calendar.month_abbr[v])
        fig = px.line(g, x="Month", y="Delay rate (%)", markers=True,
                      color_discrete_sequence=[ACCENT], title="Delay rate by month")
        right.plotly_chart(style(fig), config=PLOT_CONFIG)
        shown = True
    if not shown:
        st.info("Your file has no date, month, weekday or departure-time column, so time analysis is skipped.")

with tab_sim:
    st.subheader("What-if simulator")
    st.caption("Pick any airline, airport or time window and cut its delays by a percentage. "
               "This is a scenario based on your assumptions, not a forecast.")
    dims = {"Airline": "AIRLINE", "Origin airport": "ORIGIN"}
    if "TIME_OF_DAY" in clean and clean["TIME_OF_DAY"].notna().any():
        dims["Departure window"] = "TIME_OF_DAY"
    s1, s2, s3 = st.columns(3)
    dim_label = s1.selectbox("Apply to", list(dims))
    column = dims[dim_label]
    values = sorted(clean[column].dropna().astype(str).unique())
    value = s2.selectbox("Value", values)
    pct = s3.slider("Reduce its delays by (%)", 0, 100, 25)
    p1, p2 = st.columns(2)
    pax = p1.number_input("Passengers per flight", 1, 600, 100)
    scale = p2.number_input("Scale-up factor", 1.0, 1000.0, 1.0, step=1.0,
                            help="Use 10 if your file is a 10% sample of all flights.")
    res = up.simulate(clean, column, value, pct, thr, pax, scale)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Flights affected", f"{res['affected_flights']:,}")
    r2.metric("Delay rate before", f"{res['rate_before']:.2f}%")
    r3.metric("Delay rate after", f"{res['rate_after']:.2f}%",
              f"{res['rate_after'] - res['rate_before']:.2f} pts")
    r4.metric("Passenger hours saved", f"{res['passenger_hours_saved']:,.0f}")

    st.subheader("Export")
    key = f"{name}-{len(clean)}-{thr}"
    st.download_button("Download cleaned data (CSV)", to_csv_bytes(clean, key),
                       file_name="cleaned_flights.csv", mime="text/csv")
    log = pd.DataFrame(report.steps).to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaning log (CSV)", log,
                       file_name="cleaning_log.csv", mime="text/csv")