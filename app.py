import io
import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.neighbors import BallTree

# ============================================================
# THERMO WATCH - LIVE STREAMLIT APPLICATION
# FIRMS -> OSM FACILITY -> V5 AI -> RISK
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI",
    page_icon="🔥",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_URL = (
    "https://huggingface.co/harshitcodes1544/sihharshit154/resolve/main/source_classifier_v5.joblib"
)

HF_OSM_URL = (
    "https://huggingface.co/datasets/harshitcodes1544/thermowatch-osm-data/resolve/main/osm_facilities.csv"
)

# Secure key retrieval without exposing input to the UI
FIRMS_API_KEY = st.secrets.get("FIRMS_API_KEY", os.environ.get("FIRMS_API_KEY", ""))

# NASA FIRMS VIIRS source
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"

# India bounding box: west, south, east, north
INDIA_BBOX = "68.0,6.0,97.5,37.5"

# ============================================================
# CACHE MODEL
# ============================================================

@st.cache_resource
def load_model():
    response = requests.get(HF_MODEL_URL, timeout=120, allow_redirects=True)
    response.raise_for_status()
    model = joblib.load(io.BytesIO(response.content))
    return model

# ============================================================
# CACHE OSM FACILITIES
# ============================================================

@st.cache_data(ttl=86400)
def load_osm_facilities():
    response = requests.get(HF_OSM_URL, timeout=180, allow_redirects=True)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))
    return df

# ============================================================
# FETCH LIVE FIRMS (WITH SATELLITE & DAY RANGE FALLBACK)
# ============================================================

@st.cache_data(ttl=1800)
def fetch_live_firms():
    if not FIRMS_API_KEY:
        raise RuntimeError("FIRMS_API_KEY is missing from Streamlit Secrets.")

    # List of VIIRS satellite products to check in order of priority
    sources = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA21_NRT"]
    day_ranges = [1, 2, 3]  # If today has no fires or satellite lag, look back up to 3 days

    df = pd.DataFrame()

    for days in day_ranges:
        for src in sources:
            url = f"{FIRMS_URL}{FIRMS_API_KEY}/{src}/{INDIA_BBOX}/{days}"
            try:
                response = requests.get(url, timeout=120)
                if response.status_code == 200 and "latitude" in response.text:
                    temp_df = pd.read_csv(io.StringIO(response.text))
                    if not temp_df.empty:
                        df = temp_df
                        break
            except Exception:
                continue
        if not df.empty:
            break

    if df.empty:
        raise RuntimeError("NASA FIRMS returned no live observations across VIIRS sensors.")

    rename_map = {
        "latitude": "latitude",
        "longitude": "longitude",
        "bright_ti4": "bright_ti4",
        "scan": "scan",
        "track": "track",
        "acq_date": "acq_date",
        "acq_time": "acq_time",
        "satellite": "satellite",
        "instrument": "instrument",
        "confidence": "confidence",
        "version": "version",
        "bright_ti5": "bright_ti5",
        "frp": "frp",
        "daynight": "daynight"
    }
    df = df.rename(columns=rename_map)

    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
        df["acquisition_datetime"] = pd.to_datetime(
            df["acq_date"].astype(str)
            + " "
            + df["acq_time"].str[:2]
            + ":"
            + df["acq_time"].str[2:],
            errors="coerce"
        )

    numeric_cols = [
        "latitude", "longitude", "bright_ti4", "bright_ti5", "frp", "scan", "track"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df.reset_index(drop=True)
    df["hotspot_id"] = [f"LIVE_{i:06d}" for i in range(1, len(df) + 1)]
    df["fetched_at"] = pd.Timestamp.utcnow().isoformat()

    return df

# ============================================================
# FACILITY ATTRIBUTION
# ============================================================

def attribute_facilities(live, facilities):
    live = live.copy()
    facilities = facilities.copy()

    live["latitude"] = pd.to_numeric(live["latitude"], errors="coerce")
    live["longitude"] = pd.to_numeric(live["longitude"], errors="coerce")
    facilities["latitude"] = pd.to_numeric(facilities["latitude"], errors="coerce")
    facilities["longitude"] = pd.to_numeric(facilities["longitude"], errors="coerce")

    live = live.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    facilities = facilities.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    if facilities.empty:
        raise RuntimeError("OSM facility dataset contains no valid coordinates.")

    facility_coords = np.radians(facilities[["latitude", "longitude"]].values)
    live_coords = np.radians(live[["latitude", "longitude"]].values)

    tree = BallTree(facility_coords, metric="haversine")
    distances, indices = tree.query(live_coords, k=1)

    earth_radius_km = 6371.0088
    distances_km = distances[:, 0] * earth_radius_km
    nearest = facilities.iloc[indices[:, 0]].reset_index(drop=True)

    result = live.copy()

    field_mapping = {
        "facility_osm_id": "osm_id",
        "facility_type": "feature_type",
        "facility_name": "name",
        "facility_power": "power",
        "facility_industrial": "industrial",
        "facility_landuse": "landuse",
        "facility_operator": "operator",
        "facility_plant_source": "plant_source",
        "facility_plant_method": "plant_method"
    }

    for output_col, input_col in field_mapping.items():
        if input_col in nearest.columns:
            result[output_col] = nearest[input_col].values
        else:
            result[output_col] = np.nan

    result["facility_latitude"] = nearest["latitude"].values
    result["facility_longitude"] = nearest["longitude"].values
    result["distance_to_facility_km"] = distances_km

    def proximity(distance):
        if distance <= 1:
            return "VERY_CLOSE"
        elif distance <= 2:
            return "CLOSE"
        elif distance <= 5:
            return "NEAR"
        elif distance <= 10:
            return "DISTANT"
        return "FAR"

    result["proximity_category"] = result["distance_to_facility_km"].apply(proximity)

    def quality(distance):
        if distance <= 1:
            return "HIGH"
        elif distance <= 5:
            return "MEDIUM"
        elif distance <= 10:
            return "LOW"
        return "VERY_LOW"

    result["facility_match_quality"] = result["distance_to_facility_km"].apply(quality)
    return result

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    df["mean_frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df["max_frp"] = df["mean_frp"]
    df["frp_ratio"] = df["max_frp"] / (df["mean_frp"] + 1e-6)

    df["mean_brightness"] = pd.to_numeric(df.get("bright_ti4", 0), errors="coerce").fillna(0)
    df["max_brightness"] = df["mean_brightness"]
    df["brightness_range"] = 0.0

    df["observation_count"] = 1
    df["persistence_days"] = 1
    df["observations_per_day"] = 1.0
    df["activity_density"] = 1.0
    df["satellite_count"] = 1

    if "distance_to_facility_km" not in df:
        df["distance_to_facility_km"] = 999.0

    df["distance_to_facility_km"] = pd.to_numeric(
        df["distance_to_facility_km"], errors="coerce"
    ).fillna(999.0)

    df["facility_within_1km"] = (df["distance_to_facility_km"] <= 1).astype(int)
    df["facility_within_5km"] = (df["distance_to_facility_km"] <= 5).astype(int)
    df["persistent_long_term"] = 0
    df["persistent_180_days"] = 0

    categorical = [
        "facility_type", "facility_power", "facility_industrial", "facility_landuse"
    ]
    for col in categorical:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    df["frp_ratio_v5"] = df["max_frp"] / (df["mean_frp"] + 1e-6)
    df["frp_excess_v5"] = np.maximum(df["max_frp"] - 5.0, 0)
    df["frp_log_v5"] = np.log1p(df["mean_frp"])
    df["max_frp_log_v5"] = np.log1p(df["max_frp"])
    df["brightness_range_v5"] = 0.0
    df["brightness_ratio_v5"] = 1.0
    df["brightness_excess_v5"] = 0.0
    df["persistence_log_v5"] = np.log1p(df["persistence_days"])
    df["persistence_months_v5"] = df["persistence_days"] / 30.0
    df["persistent_30d_v5"] = (df["persistence_days"] >= 30).astype(int)
    df["persistent_90d_v5"] = (df["persistence_days"] >= 90).astype(int)
    df["persistent_180d_v5"] = (df["persistence_days"] >= 180).astype(int)
    df["persistent_270d_v5"] = (df["persistence_days"] >= 270).astype(int)
    df["observation_log_v5"] = np.log1p(df["observation_count"])
    df["obs_per_persistence_v5"] = df["observation_count"] / (df["persistence_days"] + 1e-6)
    df["activity_persistence_v5"] = df["activity_density"] * df["persistence_days"]
    df["distance_log_v5"] = np.log1p(df["distance_to_facility_km"])
    df["very_close_v5"] = (df["distance_to_facility_km"] <= 1).astype(int)
    df["within_2km_v5"] = (df["distance_to_facility_km"] <= 2).astype(int)
    df["within_5km_v5"] = (df["distance_to_facility_km"] <= 5).astype(int)
    df["within_10km_v5"] = (df["distance_to_facility_km"] <= 10).astype(int)
    df["frp_persistence_v5"] = df["mean_frp"] * df["persistence_days"]
    df["frp_distance_signal_v5"] = df["mean_frp"] / (df["distance_to_facility_km"] + 1)
    df["activity_distance_signal_v5"] = df["activity_density"] / (df["distance_to_facility_km"] + 1)

    return df

# ============================================================
# EXACT V5 FEATURES
# ============================================================

FEATURES = [
    "mean_frp", "max_frp", "frp_ratio",
    "mean_brightness", "max_brightness", "brightness_range",
    "observation_count", "persistence_days", "observations_per_day", "activity_density", "satellite_count",
    "distance_to_facility_km", "facility_within_1km", "facility_within_5km",
    "persistent_long_term", "persistent_180_days",
    "facility_type", "facility_power", "facility_industrial", "facility_landuse",
    "frp_ratio_v5", "frp_excess_v5", "frp_log_v5", "max_frp_log_v5",
    "brightness_range_v5", "brightness_ratio_v5", "brightness_excess_v5",
    "persistence_log_v5", "persistence_months_v5",
    "persistent_30d_v5", "persistent_90d_v5", "persistent_180d_v5", "persistent_270d_v5",
    "observation_log_v5", "obs_per_persistence_v5", "activity_persistence_v5",
    "distance_log_v5",
    "very_close_v5", "within_2km_v5", "within_5km_v5", "within_10km_v5",
    "frp_persistence_v5", "frp_distance_signal_v5", "activity_distance_signal_v5"
]

# ============================================================
# V5 PREDICTION
# ============================================================

def run_v5_prediction(df, model):
    df = engineer_features(df)

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise RuntimeError("Missing V5 features: " + str(missing))

    X = df[FEATURES].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns
    X[numeric_features] = (
        X[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    categorical_features = [
        "facility_type",
        "facility_power",
        "facility_industrial",
        "facility_landuse"
    ]
    for col in categorical_features:
        X[col] = X[col].fillna("unknown").astype(str)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    confidence = np.max(probabilities, axis=1) * 100

    df["predicted_source"] = predictions
    df["confidence"] = confidence.round(1)

    frp_component = np.clip(df["mean_frp"] * 3, 0, 35)
    max_frp_component = np.clip(df["max_frp"] * 0.5, 0, 20)
    confidence_component = confidence * 0.20

    risk_score = frp_component + max_frp_component + confidence_component
    risk_score = np.clip(risk_score, 0, 100)
    df["live_risk_score"] = risk_score.round(2)

    def risk_category(score):
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        return "LOW"

    df["live_risk_category"] = df["live_risk_score"].apply(risk_category)
    return df

# ============================================================
# HEADER & SIDEBAR
# ============================================================

st.title("🔥 ThermoWatch AI")
st.subheader("LIVE Satellite Fire Detection & Industrial Risk Intelligence")
st.caption("NASA FIRMS → OSM Facilities → V5 AI → Risk Assessment")

st.sidebar.header("ThermoWatch Controls")

if st.sidebar.button("🔄 Fetch Latest LIVE Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("Live observations are fetched directly from NASA FIRMS VIIRS sensors over India.")

# ============================================================
# MAIN PIPELINE
# ============================================================

try:
    with st.spinner("Loading V5 AI model..."):
        model = load_model()

    with st.spinner("Loading OSM facility database..."):
        facilities = load_osm_facilities()

    with st.spinner("Fetching LIVE NASA FIRMS observations..."):
        live = fetch_live_firms()

    with st.spinner("Matching hotspots with nearest facilities..."):
        attributed = attribute_facilities(live, facilities)

    with st.spinner("Running V5 AI inference..."):
        predictions = run_v5_prediction(attributed, model)

except Exception as e:
    st.error("ThermoWatch LIVE pipeline failed.")
    st.exception(e)
    st.stop()

# ============================================================
# DASHBOARD VISUALIZATIONS
# ============================================================

st.success(f"LIVE pipeline completed — {len(predictions)} hotspots analyzed.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LIVE Hotspots", len(predictions))
with col2:
    st.metric("High Risk", int((predictions["live_risk_category"] == "HIGH").sum()))
with col3:
    st.metric("Critical", int((predictions["live_risk_category"] == "CRITICAL").sum()))
with col4:
    st.metric("Avg Risk", round(predictions["live_risk_score"].mean(), 2))

col_left, col_right = st.columns(2)
with col_left:
    st.header("Risk Distribution")
    st.bar_chart(predictions["live_risk_category"].value_counts())

with col_right:
    st.header("AI Source Classification")
    st.bar_chart(predictions["predicted_source"].value_counts())

st.header("LIVE Fire Hotspots")
map_df = predictions[["latitude", "longitude"]].dropna()
st.map(map_df)

st.header("🚨 Highest Risk LIVE Detections")
display_cols = [
    "hotspot_id",
    "facility_name",
    "latitude",
    "longitude",
    "frp",
    "predicted_source",
    "confidence",
    "distance_to_facility_km",
    "facility_match_quality",
    "live_risk_score",
    "live_risk_category"
]
display_cols = [c for c in display_cols if c in predictions.columns]

top_risk = (
    predictions
    .sort_values("live_risk_score", ascending=False)[display_cols]
    .head(20)
)
st.dataframe(top_risk, use_container_width=True, hide_index=True)

st.header("Facility Proximity")
st.bar_chart(predictions["proximity_category"].value_counts())

st.header("Export LIVE Results")
csv_data = predictions.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download LIVE Predictions CSV",
    data=csv_data,
    file_name="thermowatch_live_predictions.csv",
    mime="text/csv"
)

st.divider()
st.caption("ThermoWatch AI | LIVE FIRMS + OSM + V5 Machine Learning")
