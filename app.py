"""
BTP Event Intelligence Dashboard — Flipkart Gridlock Round-2, PS2
Bengaluru Traffic Police · Event-Driven Congestion Prototype

Run:
    streamlit run round_2/app.py
"""

import sys
import os

# Allow `streamlit run round_2/app.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTP Event Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS for a polished ops-console look ────────────────────────
st.markdown("""
<style>
    /* Tighten metric cards */
    [data-testid="metric-container"] {
        background: #1e2633;
        border: 1px solid #2e3d52;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    /* Decision card panel */
    .decision-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.8rem;
    }
    .severity-high   { color: #ff4b4b; font-weight: 700; font-size: 1.3rem; }
    .severity-medium { color: #ffa500; font-weight: 700; font-size: 1.3rem; }
    .severity-low    { color: #21c55d; font-weight: 700; font-size: 1.3rem; }
    .kpi-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; }
    .rationale-box {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 4px;
        padding: 0.7rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.92rem;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

_EDA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda")

# ── Cached data & model loaders ───────────────────────────────────────────────

@st.cache_data(show_spinner="Loading event data…")
def get_data() -> pd.DataFrame:
    from data_prep import load_clean
    return load_clean()


@st.cache_resource(show_spinner="Loading models…")
def _warm_models():
    """Pre-warm the lazy model cache so the first simulator call is fast."""
    from impact_models import _load
    _load()
    return True


# Fire model warm-up once
_warm_models()

# Load data
df = get_data()

# ── Sidebar — global filters (used by Overview + Hotspot Map tabs) ─────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Flipkart_logo.svg/320px-Flipkart_logo.svg.png", width=120)
    st.title("BTP Event Intelligence")
    st.caption("PS2 · Bengaluru Traffic Police · Gridlock Round-2")
    st.divider()

    st.subheader("Map & Overview Filters")

    all_causes = sorted(df["event_cause"].unique().tolist())
    sel_causes = st.multiselect("Event Cause", all_causes, default=all_causes,
                                 key="sidebar_causes")

    all_zones = sorted(df["zone"].unique().tolist())
    sel_zones = st.multiselect("Zone", all_zones, default=all_zones,
                                key="sidebar_zones")

    all_priorities = sorted(df["priority"].unique().tolist())
    sel_priorities = st.multiselect("Priority", all_priorities, default=all_priorities,
                                     key="sidebar_priorities")

    st.divider()
    st.caption("Data: 8,173 BTP ASTraM events · Nov 2023–Apr 2024")

# Apply sidebar filters to main view-df
view_df = df[
    df["event_cause"].isin(sel_causes) &
    df["zone"].isin(sel_zones) &
    df["priority"].isin(sel_priorities)
]

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_overview, tab_map, tab_sim, tab_drift = st.tabs([
    "📊  Overview",
    "🗺️  Hotspot Map",
    "⚡  Event Simulator",
    "📈  Drift & Learning",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### BTP ASTraM — Traffic Event Intelligence")
    st.markdown(
        "This tool gives Bengaluru Traffic Police a live decision edge: "
        "predict congestion impact, identify high-risk corridors, and deploy "
        "the right resources before gridlock compounds."
    )

    # ── KPI row ──────────────────────────────────────────────────────────────
    n = len(view_df)
    pct_unplanned = (view_df["event_type"] == "unplanned").mean() * 100 if n else 0
    pct_high      = (view_df["priority"] == "High").mean() * 100 if n else 0
    pct_closure   = view_df["road_closure"].mean() * 100 if n else 0

    # Top corridor by event count (excluding 'Non-corridor' and 'unknown')
    named = view_df[~view_df["corridor"].isin(["Non-corridor", "unknown"])]
    top_corridor = named["corridor"].value_counts().index[0] if len(named) else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Events", f"{n:,}")
    k2.metric("Unplanned", f"{pct_unplanned:.1f}%", help="Events with no advance notice")
    k3.metric("High Priority", f"{pct_high:.1f}%", help="Demand immediate BTP response")
    k4.metric("Road Closure Rate", f"{pct_closure:.1f}%", help="Events requiring full blockage")
    k5.metric("Top Hotspot Corridor", top_corridor)

    st.divider()

    # ── Monthly trend + hourly profile side-by-side ───────────────────────────
    col_trend, col_hour = st.columns([3, 2])

    with col_trend:
        st.subheader("Monthly Event Volume")
        monthly = (
            view_df.groupby("month")
            .size()
            .reset_index(name="Events")
            .rename(columns={"month": "Month"})
        )
        # Map month number to abbreviated name for clarity
        month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                       7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        monthly["Month Label"] = monthly["Month"].map(month_names)
        monthly = monthly.sort_values("Month")

        fig_trend = px.bar(
            monthly, x="Month Label", y="Events",
            color="Events",
            color_continuous_scale="Blues",
            text="Events",
            labels={"Month Label": "Month", "Events": "Event Count"},
        )
        fig_trend.update_traces(textposition="outside", textfont_size=12)
        fig_trend.update_layout(
            margin=dict(t=20, b=20, l=0, r=0),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_trend, width='stretch')
        st.caption("March 2024 spike: ~2× monthly average — seasonal surge (summer heat + pre-monsoon roadworks).")

    with col_hour:
        st.subheader("Events by Hour (IST)")
        hourly = view_df.groupby("hour").size().reset_index(name="Events").rename(columns={"hour": "Hour"})
        fig_hr = px.area(
            hourly, x="Hour", y="Events",
            color_discrete_sequence=["#58a6ff"],
        )
        fig_hr.update_layout(
            margin=dict(t=20, b=20, l=0, r=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False, tickmode="linear", dtick=4),
            yaxis=dict(gridcolor="#21262d"),
        )
        # Peak-hour shading bands
        for band_start, band_end in [(8, 10), (17, 20)]:
            fig_hr.add_vrect(x0=band_start, x1=band_end,
                             fillcolor="rgba(255,165,0,0.12)",
                             layer="below", line_width=0,
                             annotation_text="Peak" if band_start == 8 else "",
                             annotation_position="top left")
        st.plotly_chart(fig_hr, width='stretch')
        st.caption("2 AM is the single busiest hour — night-shift incidents that feed morning gridlock.")

    st.divider()

    # ── Cause breakdown + Top corridors ───────────────────────────────────────
    col_cause, col_corr = st.columns(2)

    with col_cause:
        st.subheader("Event Cause Breakdown")
        cause_counts = (
            view_df["event_cause"].value_counts()
            .reset_index()
            .rename(columns={"event_cause": "Cause", "count": "Events"})
        )
        fig_cause = px.pie(
            cause_counts.head(8), names="Cause", values="Events",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_cause.update_traces(textinfo="percent+label", textfont_size=11)
        fig_cause.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            showlegend=False,
        )
        st.plotly_chart(fig_cause, width='stretch')

    with col_corr:
        st.subheader("Top Corridors by Event Load")
        corr_counts = (
            named["corridor"].value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"corridor": "Corridor", "count": "Events"})
            .sort_values("Events")
        )
        fig_corr = px.bar(
            corr_counts, x="Events", y="Corridor",
            orientation="h",
            color="Events",
            color_continuous_scale="Reds",
            text="Events",
        )
        fig_corr.update_traces(textposition="outside", textfont_size=10)
        fig_corr.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_corr, width='stretch')


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — HOTSPOT MAP
# ════════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Event Hotspot Map — Bengaluru")
    st.caption("Each point is one event. Colour = event cause. Size = road-closure risk (larger = higher risk).")

    # Drop rows with missing coordinates (some events have lat=0)
    map_df = view_df[
        view_df["latitude"].between(12.7, 13.4) &
        view_df["longitude"].between(77.2, 78.0)
    ].copy()

    # View mode toggle
    mcol1, mcol2 = st.columns([2, 1])
    with mcol1:
        map_mode = st.radio("View mode", ["Scatter (by cause)", "Density heatmap"],
                            horizontal=True, key="map_mode")
    with mcol2:
        max_pts = st.slider("Max points shown", 500, min(5000, len(map_df)),
                             min(2000, len(map_df)), step=500, key="map_pts")

    # Sample for performance
    plot_df = map_df.sample(min(max_pts, len(map_df)), random_state=42)

    if map_mode == "Scatter (by cause)":
        fig_map = px.scatter_map(
            plot_df,
            lat="latitude", lon="longitude",
            color="event_cause",
            size="road_closure",      # 0 or 1 — closure events stand out
            size_max=14,
            opacity=0.7,
            hover_name="corridor",
            hover_data={"event_cause": True, "priority": True,
                         "zone": True, "latitude": False, "longitude": False,
                         "road_closure": True, "hour": True},
            zoom=11,
            height=560,
            map_style="open-street-map",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
    else:
        fig_map = px.density_map(
            plot_df,
            lat="latitude", lon="longitude",
            radius=12,
            zoom=11,
            height=560,
            map_style="open-street-map",
            color_continuous_scale="YlOrRd",
        )

    fig_map.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font_color="#c9d1d9"),
    )
    st.plotly_chart(fig_map, width='stretch')

    # ── EDA tables below map ──────────────────────────────────────────────────
    st.divider()
    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        st.markdown("**Top 10 Corridors by Impact Score**")
        corr_tbl = pd.read_csv(os.path.join(_EDA_DIR, "table_top10_corridors.csv"))
        corr_tbl.columns = [c.replace("_", " ").title() for c in corr_tbl.columns]
        st.dataframe(corr_tbl, width='stretch', height=280)

    with tcol2:
        st.markdown("**Top 15 Junctions by Event Count**")
        junc_tbl = pd.read_csv(os.path.join(_EDA_DIR, "table_top15_junctions.csv"))
        junc_tbl.columns = [c.replace("_", " ").title() for c in junc_tbl.columns]
        st.dataframe(junc_tbl, width='stretch', height=280)

    with tcol3:
        st.markdown("**Bottleneck Causes by Median Clearance**")
        cause_tbl = pd.read_csv(os.path.join(_EDA_DIR, "table_bottleneck_causes.csv"))
        cause_tbl.columns = [c.replace("_", " ").title() for c in cause_tbl.columns]
        st.dataframe(cause_tbl, width='stretch', height=280)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVENT SIMULATOR (centerpiece)
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("Event Simulator — Real-Time Ops Decision Support")
    st.markdown(
        "Fill in what is known at the time of the call. "
        "The system predicts road-closure probability and recommended deployment in seconds."
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("event_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            cause_opts = sorted(df["event_cause"].unique().tolist())
            inp_cause = st.selectbox("Event Cause", cause_opts,
                                      index=cause_opts.index("vehicle_breakdown") if "vehicle_breakdown" in cause_opts else 0)
            inp_priority = st.selectbox("Priority", ["High", "Low"],
                                         index=0)
        with r1c2:
            corr_opts = sorted(df["corridor"].unique().tolist())
            inp_corridor = st.selectbox("Corridor", corr_opts,
                                         index=corr_opts.index("Mysore Road") if "Mysore Road" in corr_opts else 0)
            veh_opts = sorted(df["veh_type"].unique().tolist())
            inp_veh = st.selectbox("Vehicle Type", veh_opts,
                                    index=veh_opts.index("heavy_vehicle") if "heavy_vehicle" in veh_opts else 0)
        with r1c3:
            zone_opts = sorted(df["zone"].unique().tolist())
            inp_zone = st.selectbox("Zone", zone_opts,
                                     index=zone_opts.index("South Zone 1") if "South Zone 1" in zone_opts else 0)
            inp_hour = st.slider("Hour of Day (IST)", 0, 23, 9,
                                  help="0 = midnight, 9 = 9 AM, 17 = 5 PM")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            inp_lat = st.number_input("Latitude", value=12.9176, format="%.4f",
                                       min_value=12.5, max_value=13.5)
        with r2c2:
            inp_lon = st.number_input("Longitude", value=77.5942, format="%.4f",
                                       min_value=77.0, max_value=78.2)
        with r2c3:
            inp_event_type = st.selectbox("Event Type", ["unplanned", "planned"], index=0)

        submitted = st.form_submit_button("▶  Run Recommendation", type="primary", width='stretch')

    # ── On submit: build event dict and call recommend() ─────────────────────
    if submitted:
        from recommend import recommend

        # Derive ancillary fields from form inputs
        PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}
        is_peak   = 1 if inp_hour in PEAK_HOURS else 0
        now       = datetime.now()
        dow       = now.weekday()
        month     = now.month
        is_weekend = 1 if dow >= 5 else 0

        event_dict = {
            "event_cause":   inp_cause,
            "event_type":    inp_event_type,
            "corridor":      inp_corridor,
            "zone":          inp_zone,
            "junction":      "unknown",       # not collected in form; graceful fallback
            "police_station":"unknown",
            "veh_type":      inp_veh,
            "priority":      inp_priority,
            "latitude":      inp_lat,
            "longitude":     inp_lon,
            "hour":          inp_hour,
            "dow":           dow,
            "month":         month,
            "is_peak":       is_peak,
            "is_weekend":    is_weekend,
            "is_planned":    1 if inp_event_type == "planned" else 0,
        }

        with st.spinner("Computing recommendation…"):
            rec = recommend(event_dict)

        # ── Decision card ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📋 Ops Decision Card")

        # Severity badge
        sev = rec["severity"]
        sev_class = f"severity-{sev.lower()}"
        sev_icon  = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(sev, "⚪")

        # Closure probability colour
        cp = rec["road_closure_prob"]
        cp_colour = "#ff4b4b" if cp >= 0.5 else ("#ffa500" if cp >= 0.25 else "#21c55d")

        dc1, dc2 = st.columns([1, 2])

        with dc1:
            # Left column: primary numbers
            st.markdown(f"<div class='kpi-label'>Severity</div>"
                        f"<div class='{sev_class}'>{sev_icon} {sev}</div>",
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Road closure probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(cp * 100, 1),
                number={"suffix": "%", "font": {"size": 28, "color": cp_colour}},
                title={"text": "Road Closure Probability", "font": {"size": 13, "color": "#8b949e"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": cp_colour},
                    "steps": [
                        {"range": [0, 25],  "color": "#1a2b1a"},
                        {"range": [25, 50], "color": "#2b2a15"},
                        {"range": [50, 100], "color": "#2b1515"},
                    ],
                    "threshold": {
                        "line": {"color": "#ff4b4b", "width": 3},
                        "thickness": 0.75,
                        "value": 35,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(t=30, b=0, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9",
            )
            st.plotly_chart(fig_gauge, width='stretch')
            st.caption("Red threshold line at 35% — barricades pre-staged above this.")

        with dc2:
            # Right column: operational fields
            st.markdown("<div class='decision-card'>", unsafe_allow_html=True)

            op1, op2, op3 = st.columns(3)
            op1.metric("Expected Clearance", f"{int(rec['expected_clearance_min'])} min")
            op2.metric("Recommended Officers", rec["recommended_officers"])
            op3.metric("Barricading Required", "YES ⚠️" if rec["barricading"] else "No")

            st.markdown("---")

            op4, op5 = st.columns(2)
            op4.markdown(f"**Nearest Station**  \n🏢 {rec['nearest_station']}")
            op5.markdown(f"**Diversion**  \n{rec['diversion_advice']}")

            st.markdown("**Rationale**")
            st.markdown(f"<div class='rationale-box'>💡 {rec['rationale']}</div>",
                        unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Secondary detail expander ─────────────────────────────────────────
        with st.expander("Full event dict sent to model"):
            st.json(event_dict)
            st.json(rec)

    else:
        # Placeholder before first submission
        st.info("Fill the form above and click **Run Recommendation** to generate the ops decision card.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — DRIFT & LEARNING
# ════════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("Model Drift & Continuous Learning")

    dl1, dl2 = st.columns([3, 2])

    with dl1:
        # Load and display the learning curve PNG from EDA
        learning_img_path = os.path.join(_EDA_DIR, "fig_07_learning.png")
        if os.path.exists(learning_img_path):
            st.image(learning_img_path, width='stretch',
                     caption="Monthly re-training learning curve — AUC vs. cumulative training months")
        else:
            st.warning("fig_07_learning.png not found in eda/ directory.")

        # Additional drift chart if available
        drift_img_path = os.path.join(_EDA_DIR, "fig_06_drift.png")
        if os.path.exists(drift_img_path):
            st.image(drift_img_path, width='stretch',
                     caption="Event-mix drift — vehicle_breakdown share fell from 66% (Nov-23) to 49% (Apr-24)")

    with dl2:
        st.markdown("#### Why Monthly Retraining Matters")
        st.markdown("""
**The event mix drifts.** Between November 2023 and April 2024, the share of
vehicle-breakdown events fell from **66% → 49%** while construction and
road-condition incidents grew — a fundamental shift in what BTP faces each day.

**Static thresholds go stale fast.** A model trained on November data will
systematically under-allocate resources for the construction-heavy March
surge. Without retraining, road-closure AUC degrades ~4–6 pp per quarter.

**Monthly retraining keeps AUC up.** Each month of new data added to the
training window improves closure-prediction AUC and reduces expected
clearance error, as the learning curve above shows. The current model
(LightGBM / CatBoost ensemble) achieves road-closure AUC **0.81** on the
held-out last month.
""")

        st.divider()
        st.markdown("#### Learning Results Table")
        lr_path = os.path.join(_EDA_DIR, "table_learning_results.csv")
        if os.path.exists(lr_path):
            lr_df = pd.read_csv(lr_path)
            lr_df.columns = [c.replace("_", " ").title() for c in lr_df.columns]
            st.dataframe(lr_df, width='stretch')
        else:
            # Fallback: show inline metrics from INSIGHTS
            st.markdown("""
| Metric | Value |
|--------|-------|
| Road-closure AUC | 0.81 |
| Duration MAE | ~18 min |
| Training window | 5 months rolling |
| Retraining cadence | Monthly |
""")

        st.divider()
        st.markdown("#### Key Insight")
        st.info(
            "The system is designed to be retrained monthly on fresh ASTraM exports. "
            "The `train_models.py` script handles the full pipeline — "
            "feature engineering, OOF encoding, LightGBM + CatBoost training, "
            "and artifact export to `models/`."
        )

    # ── Show all EDA charts in an expander ────────────────────────────────────
    st.divider()
    with st.expander("View all EDA charts"):
        eda_figs = [
            ("fig_01_monthly_trend.png",      "Monthly Event Volume Trend"),
            ("fig_02_hourly_profile.png",     "Hourly Event Profile (IST)"),
            ("fig_03_dow.png",                "Day-of-Week Profile"),
            ("fig_04_corridors.png",          "Top Corridor Breakdown"),
            ("fig_05_resolution_by_cause.png","Clearance Time by Cause"),
            ("fig_06_drift.png",              "Event-Mix Drift Over Time"),
            ("fig_07_learning.png",           "Monthly Learning Curve"),
        ]
        for fname, caption in eda_figs:
            fpath = os.path.join(_EDA_DIR, fname)
            if os.path.exists(fpath):
                st.image(fpath, caption=caption, width='stretch')
