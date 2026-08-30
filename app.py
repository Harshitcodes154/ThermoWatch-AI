import os
import pandas as pd
import streamlit as st

# ============================================================
# THERMO WATCH - LIVE AI DASHBOARD
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI",
    page_icon="🔥",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "live_final_predictions.csv"
)

# ============================================================
# HEADER
# ============================================================

st.title("🔥 ThermoWatch AI")
st.subheader("Live Satellite Fire Detection & Industrial Risk Intelligence")

st.caption(
    "Live FIRMS observations → OSM facility attribution → V5 AI source classification → risk scoring"
)

# ============================================================
# LOAD LIVE DATA
# ============================================================

if not os.path.exists(PREDICTION_FILE):

    st.error(
        "No live ThermoWatch prediction data found."
    )

    st.info(
        "Run the LIVE FIRMS → Facility Attribution → V5 AI pipeline first."
    )

    st.code(
        """
python .\\scripts\\live_firms_fetch.py
python .\\scripts\\live_facility_attribution.py
python .\\scripts\\live_v5_inference.py
        """
    )

    st.stop()


try:
    df = pd.read_csv(PREDICTION_FILE)

except Exception as e:

    st.error(f"Unable to load live prediction file: {e}")

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "hotspot_id",
    "latitude",
    "longitude",
    "frp",
    "predicted_source",
    "confidence",
    "live_risk_score",
    "live_risk_category"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        f"Missing columns in live prediction file: {missing}"
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df["frp"] = pd.to_numeric(
    df["frp"],
    errors="coerce"
)

df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce"
)

df["live_risk_score"] = pd.to_numeric(
    df["live_risk_score"],
    errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "live_risk_score"
    ]
).copy()


# ============================================================
# DASHBOARD METRICS
# ============================================================

total_hotspots = len(df)

high_risk = (
    df["live_risk_category"]
    .isin(["HIGH", "CRITICAL"])
    .sum()
)

medium_risk = (
    df["live_risk_category"]
    == "MEDIUM"
).sum()

low_risk = (
    df["live_risk_category"]
    == "LOW"
).sum()

avg_risk = df["live_risk_score"].mean()

max_risk = df["live_risk_score"].max()

avg_confidence = df["confidence"].mean()


# ============================================================
# KPI CARDS
# ============================================================

st.markdown("## 📡 Live Monitoring")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🔥 Live Hotspots",
        total_hotspots
    )

with col2:
    st.metric(
        "🚨 High Risk",
        high_risk
    )

with col3:
    st.metric(
        "⚠️ Medium Risk",
        medium_risk
    )

with col4:
    st.metric(
        "📊 Avg Risk",
        f"{avg_risk:.2f}"
    )

with col5:
    st.metric(
        "🤖 AI Confidence",
        f"{avg_confidence:.1f}%"
    )


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

st.markdown("## 🤖 AI Source Classification")

source_counts = (
    df["predicted_source"]
    .value_counts()
)

st.bar_chart(source_counts)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.markdown("## 🚨 Risk Distribution")

risk_counts = (
    df["live_risk_category"]
    .value_counts()
)

st.bar_chart(risk_counts)


# ============================================================
# LIVE HOTSPOT MAP
# ============================================================

st.markdown("## 🗺️ Live Fire Detection Map")

map_df = df[
    [
        "latitude",
        "longitude"
    ]
].dropna()

st.map(
    map_df,
    latitude="latitude",
    longitude="longitude",
    zoom=4
)


# ============================================================
# TOP HIGH-RISK DETECTIONS
# ============================================================

st.markdown("## 🔥 Highest-Risk Live Detections")

top_columns = [
    "hotspot_id",
    "facility_name",
    "latitude",
    "longitude",
    "frp",
    "predicted_source",
    "confidence",
    "live_risk_score",
    "live_risk_category"
]

top_columns = [
    col
    for col in top_columns
    if col in df.columns
]

top_risk = (
    df
    .sort_values(
        "live_risk_score",
        ascending=False
    )
    [top_columns]
    .head(20)
)

st.dataframe(
    top_risk,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FACILITY INFORMATION
# ============================================================

if "facility_name" in df.columns:

    st.markdown("## 🏭 Facility Attribution")

    facility_df = df[
        [
            "hotspot_id",
            "facility_name",
            "facility_type",
            "facility_power",
            "facility_industrial",
            "distance_to_facility_km",
            "proximity_category",
            "facility_match_quality"
        ]
    ].copy()

    available_cols = [
        col
        for col in facility_df.columns
        if col in df.columns
    ]

    facility_df = facility_df[
        available_cols
    ]

    st.dataframe(
        facility_df
        .sort_values(
            "distance_to_facility_km"
        )
        .head(30),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIVE DATA TABLE
# ============================================================

st.markdown("## 📡 All Live Detections")

st.dataframe(
    df.sort_values(
        "live_risk_score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# REFRESH
# ============================================================

st.markdown("---")

st.caption(
    f"Data source: live_final_predictions.csv | "
    f"Records: {len(df)}"
)

if st.button("🔄 Refresh Live Data"):

    st.cache_data.clear()

    st.rerun()
