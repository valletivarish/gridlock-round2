"""
BTP Event Intelligence Dashboard
Bengaluru Traffic Police - PS2: Event-Driven Congestion
Gridlock Hackathon 2.0, Round 2

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
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BTP Event Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS: ensure dark text on light background, clean metric cards
st.markdown("""
<style>
[data-testid="metric-container"] {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    background: #f6f8fa;
}
.rec-block {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    background: #f6f8fa;
    margin-top: 0.5rem;
    color: #1f2328;
}
.rationale-box {
    border-left: 4px solid #2a5db0;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    background: #edf2fb;
    color: #1f2328;
    font-size: 0.94rem;
    margin-top: 0.4rem;
}
.reliability-box {
    border-left: 4px solid #6e7781;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    background: #f6f8fa;
    color: #1f2328;
    font-size: 0.93rem;
}
</style>
""", unsafe_allow_html=True)

_EDA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading event data...")
def get_data() -> pd.DataFrame:
    from data_prep import load_clean
    return load_clean()


@st.cache_resource(show_spinner="Loading models...")
def _warm_models():
    from impact_models import _load
    _load()
    return True


# Pre-warm model cache
_warm_models()
df = get_data()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("BTP Event Intelligence")
st.caption(
    "Bengaluru Traffic Police — PS2: Event-Driven Congestion"
    " | Data: 8,173 ASTraM events, Nov 2023 - Apr 2024"
)
st.divider()

# ---------------------------------------------------------------------------
# Two tabs only
# ---------------------------------------------------------------------------
tab_response, tab_insights = st.tabs(["Event Response", "City Insights"])


# ============================================================================
# TAB 1 — EVENT RESPONSE (the decision tool)
# ============================================================================
with tab_response:
    st.subheader("Event Response Planner")
    st.markdown(
        "Enter the details reported at the time of the call. "
        "The system predicts road-closure probability and recommends the deployment response."
    )

    # Derive dropdown options from real data
    cause_opts    = sorted(df["event_cause"].unique().tolist())
    corridor_opts = sorted(df["corridor"].unique().tolist())
    zone_opts     = sorted(df["zone"].unique().tolist())
    veh_opts      = sorted(df["veh_type"].unique().tolist())

    def _idx(lst, val):
        return lst.index(val) if val in lst else 0

    with st.form("event_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            inp_cause    = st.selectbox("Event Cause", cause_opts,
                                        index=_idx(cause_opts, "vehicle_breakdown"))
            inp_priority = st.selectbox("Priority", ["High", "Low"], index=0)

        with col_b:
            inp_corridor = st.selectbox("Corridor", corridor_opts,
                                        index=_idx(corridor_opts, "Mysore Road"))
            inp_zone     = st.selectbox("Zone", zone_opts,
                                        index=_idx(zone_opts, "South Zone 1"))

        with col_c:
            inp_veh      = st.selectbox("Vehicle Type", veh_opts,
                                        index=_idx(veh_opts, "heavy_vehicle"))
            inp_hour     = st.slider("Hour of Day (IST)", 0, 23, 9,
                                     help="0 = midnight, 9 = 9 AM, 17 = 5 PM")

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            inp_lat = st.number_input("Latitude", value=12.9176, format="%.4f",
                                      min_value=12.5, max_value=13.5)
        with col_e:
            inp_lon = st.number_input("Longitude", value=77.5942, format="%.4f",
                                      min_value=77.0, max_value=78.2)
        with col_f:
            inp_event_type = st.selectbox("Event Type", ["unplanned", "planned"], index=0)

        submitted = st.form_submit_button(
            "Get Recommendation", type="primary", use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Result block
    # -------------------------------------------------------------------------
    if submitted:
        from recommend import recommend

        PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}
        now        = datetime.now()

        event_dict = {
            "event_cause":    inp_cause,
            "event_type":     inp_event_type,
            "corridor":       inp_corridor,
            "zone":           inp_zone,
            "junction":       "unknown",
            "police_station": "unknown",
            "veh_type":       inp_veh,
            "priority":       inp_priority,
            "latitude":       inp_lat,
            "longitude":      inp_lon,
            "hour":           inp_hour,
            "dow":            now.weekday(),
            "month":          now.month,
            "is_peak":        1 if inp_hour in PEAK_HOURS else 0,
            "is_weekend":     1 if now.weekday() >= 5 else 0,
            "is_planned":     1 if inp_event_type == "planned" else 0,
        }

        with st.spinner("Computing recommendation..."):
            rec = recommend(event_dict)

        st.divider()
        st.subheader("Recommended Response")

        # Top row: three key metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Road-Closure Probability",
            f"{rec['road_closure_prob']:.0%}",
        )
        m2.metric("Severity", rec["severity"])
        m3.metric("Expected Clearance", f"{int(rec['expected_clearance_min'])} min")

        st.write("")  # vertical spacing

        # Recommendation detail block
        st.markdown("<div class='rec-block'>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"**Officers to deploy:** {rec['recommended_officers']}")
            st.markdown(
                f"**Barricading required:** {'Yes' if rec['barricading'] else 'No'}"
            )
            st.markdown(f"**Nearest station:** {rec['nearest_station']}")
        with r2:
            st.markdown(f"**Diversion advice:**")
            st.markdown(rec["diversion_advice"])

        st.write("")
        st.markdown("**Rationale**")
        st.markdown(
            f"<div class='rationale-box'>{rec['rationale']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Debug: full event input and model output"):
            st.json(event_dict)
            st.json(rec)

    else:
        st.info(
            "Fill in the event details above and click Get Recommendation "
            "to generate the response plan."
        )


# ============================================================================
# TAB 2 — CITY INSIGHTS (supporting evidence)
# ============================================================================
with tab_insights:
    st.subheader("City-Wide Traffic Insights")
    st.markdown(
        "Summary statistics and hotspot analysis from 8,173 BTP ASTraM events "
        "(November 2023 to April 2024)."
    )

    # -------------------------------------------------------------------------
    # KPI row
    # -------------------------------------------------------------------------
    pct_unplanned = (df["event_type"] == "unplanned").mean() * 100
    pct_high      = (df["priority"] == "High").mean() * 100
    pct_closure   = df["road_closure"].mean() * 100

    named = df[~df["corridor"].isin(["Non-corridor", "unknown"])]
    top_corridor = named["corridor"].value_counts().index[0] if len(named) else "—"
    top_corridor_count = int(named["corridor"].value_counts().iloc[0]) if len(named) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Events", "8,173")
    k2.metric("Unplanned", f"{pct_unplanned:.1f}%",
              help="Events with no advance notice")
    k3.metric("High Priority", f"{pct_high:.1f}%",
              help="Events requiring immediate BTP response")
    k4.metric("Road Closure Rate", f"{pct_closure:.1f}%",
              help="Events requiring full road blockage")

    # Top corridor shown as text (avoids truncation)
    st.caption(
        f"Top hotspot corridor: {top_corridor} ({top_corridor_count} events)"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Hotspot map + top corridors bar chart
    # -------------------------------------------------------------------------
    col_map, col_bar = st.columns([3, 2])

    with col_map:
        st.markdown("**Event Hotspot Map**")
        st.caption("Each point is one event. Colour = event cause.")

        map_df = df[
            df["latitude"].between(12.7, 13.4) &
            df["longitude"].between(77.2, 78.0)
        ].copy()

        # Sample for performance
        plot_df = map_df.sample(min(2000, len(map_df)), random_state=42)

        fig_map = px.scatter_map(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="event_cause",
            opacity=0.65,
            hover_name="corridor",
            hover_data={
                "event_cause": True,
                "priority": True,
                "zone": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=11,
            height=480,
            map_style="open-street-map",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_map.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_map, width='stretch')

    with col_bar:
        st.markdown("**Top Hotspot Corridors by Event Load**")
        corr_counts = (
            named["corridor"]
            .value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"corridor": "Corridor", "count": "Events"})
            .sort_values("Events")
        )
        fig_bar = px.bar(
            corr_counts,
            x="Events",
            y="Corridor",
            orientation="h",
            color="Events",
            color_continuous_scale="Blues",
            text="Events",
        )
        fig_bar.update_traces(textposition="outside", textfont_size=10)
        fig_bar.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            coloraxis_showscale=False,
            height=480,
            xaxis=dict(showgrid=True, gridcolor="#e0e4e8"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bar, width='stretch')

    st.divider()

    # -------------------------------------------------------------------------
    # EDA tables
    # -------------------------------------------------------------------------
    st.markdown("**Key Reference Tables**")
    t1, t2, t3 = st.columns(3)

    with t1:
        st.caption("Top 10 Corridors by Impact Score")
        tbl = pd.read_csv(os.path.join(_EDA_DIR, "table_top10_corridors.csv"))
        tbl.columns = [c.replace("_", " ").title() for c in tbl.columns]
        st.dataframe(tbl, width='stretch', height=270)

    with t2:
        st.caption("Top 15 Junctions by Event Count")
        tbl2 = pd.read_csv(os.path.join(_EDA_DIR, "table_top15_junctions.csv"))
        tbl2.columns = [c.replace("_", " ").title() for c in tbl2.columns]
        st.dataframe(tbl2, width='stretch', height=270)

    with t3:
        st.caption("Bottleneck Causes by Median Clearance")
        tbl3 = pd.read_csv(os.path.join(_EDA_DIR, "table_bottleneck_causes.csv"))
        tbl3.columns = [c.replace("_", " ").title() for c in tbl3.columns]
        st.dataframe(tbl3, width='stretch', height=270)

    st.divider()

    # -------------------------------------------------------------------------
    # Model reliability note
    # -------------------------------------------------------------------------
    st.markdown("**Model Reliability**")
    st.markdown(
        "<div class='reliability-box'>"
        "Road-closure classifier AUC: <strong>0.816</strong> on held-out future months; "
        "approximately <strong>0.70</strong> on unseen corridors. "
        "Event duration is not reliably predictable from available features, "
        "so the Expected Clearance figure uses the cause-specific median from historical data "
        "rather than a model prediction. "
        "Monthly retraining on fresh ASTraM exports is recommended to counter event-mix drift "
        "(vehicle-breakdown share fell from 66% to 49% across the six-month window)."
        "</div>",
        unsafe_allow_html=True,
    )
