import io
import math
import html
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium

from sklearn.neighbors import BallTree
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster


# ============================================================
# THERMOWATCH AI
# LIVE FIRMS -> TIME WINDOW -> OSM -> V5 AI -> RISK
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_URL = (
    "https://huggingface.co/"
    "harshitcodes1544/sihharshit154/"
    "resolve/main/source_classifier_v5.joblib"
)

HF_OSM_URL = (
    "https://huggingface.co/datasets/"
    "harshitcodes1544/thermowatch-osm-data/"
    "resolve/main/osm_facilities.csv"
)

FIRMS_BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/area/csv/"
)

# India:
# west, south, east, north
INDIA_BBOX = "68.0,6.0,97.5,37.5"

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 50% -20%,
            rgba(18,42,68,.40),
            transparent 45%
        ),
        #05080d;
    color: #e7edf5;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.tw-header {
    border: 1px solid #1b2a3a;
    background: rgba(8,14,23,.96);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 15px;
}

.tw-title {
    font-size: 28px;
    font-weight: 900;
    letter-spacing: .4px;
}

.tw-subtitle {
    color: #8fa2b8;
    font-size: 13px;
    margin-top: 5px;
    font-family: monospace;
}

.live-badge {
    display: inline-block;
    color: #29e3a2;
    border: 1px solid #155e49;
    background: #071b16;
    border-radius: 6px;
    padding: 4px 9px;
    font-family: monospace;
    font-size: 11px;
    margin-top: 10px;
}

.metric-box {
    border: 1px solid #172536;
    background: linear-gradient(
        180deg,
        rgba(12,22,34,.95),
        rgba(7,13,21,.95)
    );
    border-radius: 12px;
    padding: 13px;
}

.section-title {
    font-size: 18px;
    font-weight: 850;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #718096;
    font-size: 11px;
    font-family: monospace;
    margin-bottom: 10px;
}

.notice {
    border: 1px solid #26374a;
    background: #0a111b;
    border-radius: 9px;
    padding: 11px 14px;
    color: #9eb0c4;
    font-size: 12px;
    margin-bottom: 12px;
}

.no-fire {
    border: 1px solid #28445b;
    background:
        linear-gradient(
            135deg,
            rgba(8,25,39,.95),
            rgba(5,12,20,.95)
        );
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin: 15px 0;
}

.no-fire-title {
    font-size: 22px;
    font-weight: 850;
    color: #d9e7f5;
}

.no-fire-text {
    color: #8295aa;
    font-size: 13px;
    margin-top: 8px;
}

.alert-card {
    border: 1px solid #202f40;
    background: #09111b;
    border-radius: 9px;
    padding: 10px 12px;
    margin-bottom: 8px;
}

.alert-title {
    font-weight: 800;
    margin-top: 3px;
}

.alert-meta {
    color: #718096;
    font-family: monospace;
    font-size: 10px;
    margin-top: 4px;
}

.footer {
    color: #5f7083;
    font-family: monospace;
    font-size: 10px;
    border-top: 1px solid #172536;
    padding-top: 12px;
    margin-top: 20px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="tw-header">
    <div class="tw-title">🔥 ThermoWatch AI</div>
    <div class="tw-subtitle">
        LIVE SATELLITE FIRE DETECTION & INDUSTRIAL RISK INTELLIGENCE
    </div>
    <div class="tw-subtitle">
        NASA FIRMS → OSM FACILITIES → V5 AI → RISK ASSESSMENT
    </div>
    <div class="live-badge">
        ● LIVE SATELLITE THERMAL MONITORING
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value, default="UNKNOWN"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return default

    return text


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def format_observed(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True
    )

    if pd.isna(dt):
        return "Unavailable"

    return dt.strftime(
        "%d %b %Y · %H:%M UTC"
    )


def data_age(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True
    )

    if pd.isna(dt):
        return "Unknown"

    now = pd.Timestamp.now(
        tz="UTC"
    )

    hours = max(
        0,
        (now - dt).total_seconds() / 3600
    )

    if hours < 1:
        return f"{int(hours * 60)} min ago"

    if hours < 24:
        return f"{hours:.1f} hours ago"

    return f"{hours / 24:.1f} days ago"


def risk_color(risk):

    risk = clean_text(
        risk,
        "LOW"
    ).upper()

    if risk == "CRITICAL":
        return "#ff3154"

    if risk == "HIGH":
        return "#ff7547"

    if risk == "MEDIUM":
        return "#f5b942"

    return "#35d6a1"


def source_badge(source):

    source = clean_text(
        source,
        "UNKNOWN"
    ).upper()

    return source.replace(
        "_",
        " "
    )


# ============================================================
# FEATURE LIST
# EXACT V5 FEATURES
# ============================================================

FEATURES = [

    "mean_frp",
    "max_frp",
    "frp_ratio",

    "mean_brightness",
    "max_brightness",
    "brightness_range",

    "observation_count",
    "persistence_days",
    "observations_per_day",
    "activity_density",
    "satellite_count",

    "distance_to_facility_km",
    "facility_within_1km",
    "facility_within_5km",
    "persistent_long_term",
    "persistent_180_days",

    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse",

    "frp_ratio_v5",
    "frp_excess_v5",
    "frp_log_v5",
    "max_frp_log_v5",

    "brightness_range_v5",
    "brightness_ratio_v5",
    "brightness_excess_v5",

    "persistence_log_v5",
    "persistence_months_v5",

    "persistent_30d_v5",
    "persistent_90d_v5",
    "persistent_180d_v5",
    "persistent_270d_v5",

    "observation_log_v5",
    "obs_per_persistence_v5",
    "activity_persistence_v5",

    "distance_log_v5",

    "very_close_v5",
    "within_2km_v5",
    "within_5km_v5",
    "within_10km_v5",

    "frp_persistence_v5",
    "frp_distance_signal_v5",
    "activity_distance_signal_v5",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    response = requests.get(
        HF_MODEL_URL,
        timeout=180,
        allow_redirects=True
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Could not download V5 model. "
            f"HTTP {response.status_code}"
        )

    return joblib.load(
        io.BytesIO(response.content)
    )


# ============================================================
# LOAD OSM FACILITIES
# ============================================================

@st.cache_data(ttl=86400)
def load_osm_facilities():

    response = requests.get(
        HF_OSM_URL,
        timeout=300,
        allow_redirects=True
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Could not download OSM facilities. "
            f"HTTP {response.status_code}"
        )

    df = pd.read_csv(
        io.BytesIO(response.content)
    )

    required = [
        "latitude",
        "longitude"
    ]

    for col in required:

        if col not in df.columns:

            raise RuntimeError(
                f"OSM dataset missing column: {col}"
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
            "longitude"
        ]
    ).copy()

    return df


# ============================================================
# FETCH FIRMS
# ============================================================

def fetch_firms(hours):

    # ============================================================
    # THERMOWATCH - FIRMS LIVE DATA FETCH
    # FIRMS API maximum = 5 days per request
    # ============================================================

    if "FIRMS_API_KEY" not in st.secrets:
        raise RuntimeError(
            "FIRMS_API_KEY is missing from Streamlit Secrets."
        )

    api_key = st.secrets["FIRMS_API_KEY"]

    # ------------------------------------------------------------
    # Convert requested hours into days
    # ------------------------------------------------------------

    requested_hours = max(1, int(hours))
    requested_days = int(math.ceil(requested_hours / 24))

    # FIRMS allows maximum 5 days per request.
    # Therefore:
    #
    # 1-5 days  -> one request
    # >5 days   -> multiple requests
    #
    # Example:
    # 7 days -> 5 days + 2 days
    # ------------------------------------------------------------

    all_frames = []

    remaining_days = requested_days

    while remaining_days > 0:

        chunk_days = min(5, remaining_days)

        print(
            f"Fetching FIRMS data: "
            f"{chunk_days} day chunk..."
        )

        url = (
            FIRMS_BASE_URL
            + api_key
            + "/VIIRS_NOAA20_NRT/"
            + INDIA_BBOX
            + f"/{chunk_days}"
        )

        try:

            response = requests.get(
                url,
                timeout=180
            )

            response.raise_for_status()

        except requests.RequestException as e:

            # Do not immediately crash the complete dashboard.
            st.warning(
                f"FIRMS request failed for "
                f"{chunk_days}-day window: {e}"
            )

            remaining_days -= chunk_days
            continue

        # --------------------------------------------------------
        # Empty response
        # --------------------------------------------------------

        if not response.text.strip():

            remaining_days -= chunk_days
            continue

        try:

            chunk_df = pd.read_csv(
                io.StringIO(response.text)
            )

        except Exception as e:

            st.warning(
                f"Could not parse FIRMS response: {e}"
            )

            remaining_days -= chunk_days
            continue

        # --------------------------------------------------------
        # Add valid data
        # --------------------------------------------------------

        if not chunk_df.empty:

            all_frames.append(chunk_df)

        remaining_days -= chunk_days

    # ============================================================
    # NO DATA
    # ============================================================

    if not all_frames:

        return pd.DataFrame()

    # ============================================================
    # MERGE ALL FIRMS CHUNKS
    # ============================================================

    df = pd.concat(
        all_frames,
        ignore_index=True
    )

    # ============================================================
    # REMOVE DUPLICATES
    # ============================================================

    duplicate_columns = [
        col
        for col in [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument"
        ]
        if col in df.columns
    ]

    if duplicate_columns:

        df = df.drop_duplicates(
            subset=duplicate_columns
        )

    # ============================================================
    # CLEAN COORDINATES
    # ============================================================

    for col in [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track"
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    # ============================================================
    # INDIA BOUNDARY SAFETY FILTER
    # ============================================================

    # Prevent accidental points outside the intended region.
    # Keep this broad enough for FIRMS India bounding box.
    
    if "latitude" in df.columns and "longitude" in df.columns:

        df = df[
            (df["latitude"] >= 6.0)
            & (df["latitude"] <= 38.0)
            & (df["longitude"] >= 68.0)
            & (df["longitude"] <= 98.0)
        ].copy()

    # ============================================================
    # CREATE ACQUISITION DATETIME
    # ============================================================

    if (
        "acq_date" in df.columns
        and "acq_time" in df.columns
    ):

        def make_datetime(row):

            try:

                date_value = str(
                    row["acq_date"]
                )

                time_value = str(
                    row["acq_time"]
                ).split(".")[0]

                # FIRMS time can be HHMM
                time_value = time_value.zfill(4)

                return pd.to_datetime(
                    date_value
                    + " "
                    + time_value[:2]
                    + ":"
                    + time_value[2:4],
                    errors="coerce"
                )

            except Exception:

                return pd.NaT

        df["acquisition_datetime"] = (
            df.apply(
                make_datetime,
                axis=1
            )
        )

    # ============================================================
    # SORT BY MOST RECENT OBSERVATION
    # ============================================================

    if "acquisition_datetime" in df.columns:

        df = df.sort_values(
            "acquisition_datetime",
            ascending=False
        )

    # ============================================================
    # FINAL RESET
    # ============================================================

    df = df.reset_index(
        drop=True
    )

    print(
        f"FIRMS observations fetched: {len(df)}"
    )

    return df
    # --------------------------------------------------------
    # Build exact observation timestamp
    # --------------------------------------------------------

    if "acquisition_datetime" in df.columns:

        df["observed_at"] = pd.to_datetime(
            df["acquisition_datetime"],
            errors="coerce",
            utc=True
        )

    elif {
        "acq_date",
        "acq_time"
    }.issubset(df.columns):

        date_part = (
            df["acq_date"]
            .astype(str)
            .str.strip()
        )

        time_part = (
            pd.to_numeric(
                df["acq_time"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )

        df["observed_at"] = pd.to_datetime(
            date_part + " " + time_part,
            errors="coerce",
            utc=True
        )

    else:

        df["observed_at"] = pd.NaT

    # --------------------------------------------------------
    # Exact requested time window
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    cutoff = now - pd.Timedelta(
        hours=hours
    )

    if df["observed_at"].notna().any():

        df = df[
            (df["observed_at"] >= cutoff)
            &
            (df["observed_at"] <= now)
        ].copy()

    # --------------------------------------------------------
    # Numeric normalization
    # --------------------------------------------------------

    numeric_cols = [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
        "confidence"
    ]

    for col in numeric_cols:

        if col not in df.columns:

            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    # India coordinate sanity filter

    df = df[
        df["latitude"].between(
            6,
            38
        )
        &
        df["longitude"].between(
            68,
            98
        )
    ].copy()

    # Stable IDs

    df["hotspot_id"] = [
        f"LIVE_{i:06d}"
        for i in range(
            1,
            len(df) + 1
        )
    ]

    return df.reset_index(
        drop=True
    )


# ============================================================
# FACILITY ATTRIBUTION
# ============================================================

def attribute_facilities(
    live,
    facilities
):

    if live.empty:

        return live.copy()

    result = live.copy()

    facility_coords = np.radians(
        facilities[
            [
                "latitude",
                "longitude"
            ]
        ].values
    )

    live_coords = np.radians(
        live[
            [
                "latitude",
                "longitude"
            ]
        ].values
    )

    tree = BallTree(
        facility_coords,
        metric="haversine"
    )

    distances, indices = tree.query(
        live_coords,
        k=1
    )

    distances_km = (
        distances[:, 0]
        * EARTH_RADIUS_KM
    )

    nearest = facilities.iloc[
        indices[:, 0]
    ].reset_index(
        drop=True
    )

    result[
        "distance_to_facility_km"
    ] = distances_km

    # --------------------------------------------------------
    # Facility fields
    # --------------------------------------------------------

    mapping = {

        "osm_id":
            "facility_osm_id",

        "feature_type":
            "facility_type",

        "name":
            "facility_name",

        "power":
            "facility_power",

        "industrial":
            "facility_industrial",

        "landuse":
            "facility_landuse",

        "operator":
            "facility_operator",

        "plant_source":
            "facility_plant_source",

        "plant_method":
            "facility_plant_method",
    }

    for source, target in mapping.items():

        if source in nearest.columns:

            result[target] = (
                nearest[source].values
            )

        else:

            result[target] = np.nan

    result[
        "facility_latitude"
    ] = nearest[
        "latitude"
    ].values

    result[
        "facility_longitude"
    ] = nearest[
        "longitude"
    ].values

    # --------------------------------------------------------
    # Proximity
    # --------------------------------------------------------

    def proximity(distance):

        if distance <= 1:
            return "VERY_CLOSE"

        if distance <= 2:
            return "CLOSE"

        if distance <= 5:
            return "NEAR"

        if distance <= 10:
            return "DISTANT"

        return "FAR"

    result[
        "proximity_category"
    ] = result[
        "distance_to_facility_km"
    ].apply(proximity)

    # --------------------------------------------------------
    # Attribution quality
    # --------------------------------------------------------

    def quality(distance):

        if distance <= 1:
            return "HIGH"

        if distance <= 5:
            return "MEDIUM"

        if distance <= 10:
            return "LOW"

        return "VERY_LOW"

    result[
        "facility_match_quality"
    ] = result[
        "distance_to_facility_km"
    ].apply(quality)

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Thermal values
    # --------------------------------------------------------

    df["mean_frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce"
    ).fillna(0)

    df["max_frp"] = df[
        "mean_frp"
    ]

    df["frp_ratio"] = (
        df["max_frp"]
        /
        (df["mean_frp"] + 1e-6)
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    if "bright_ti4" in df.columns:

        brightness = pd.to_numeric(
            df["bright_ti4"],
            errors="coerce"
        )

        df["mean_brightness"] = (
            brightness
            .fillna(0)
        )

        df["max_brightness"] = (
            brightness
            .fillna(0)
        )

    else:

        df["mean_brightness"] = 0.0
        df["max_brightness"] = 0.0

    df["brightness_range"] = (
        df["max_brightness"]
        -
        df["mean_brightness"]
    )

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    df["observation_count"] = 1

    df["persistence_days"] = 1

    df["observations_per_day"] = 1.0

    df["activity_density"] = 1.0

    df["satellite_count"] = 1

    # --------------------------------------------------------
    # Facility context
    # --------------------------------------------------------

    if "distance_to_facility_km" not in df.columns:

        df[
            "distance_to_facility_km"
        ] = 999.0

    df[
        "distance_to_facility_km"
    ] = pd.to_numeric(
        df[
            "distance_to_facility_km"
        ],
        errors="coerce"
    ).fillna(999.0)

    df["facility_within_1km"] = (
        df["distance_to_facility_km"] <= 1
    ).astype(int)

    df["facility_within_5km"] = (
        df["distance_to_facility_km"] <= 5
    ).astype(int)

    df["persistent_long_term"] = 0

    df["persistent_180_days"] = 0

    categorical_defaults = [
        "facility_type",
        "facility_power",
        "facility_industrial",
        "facility_landuse",
    ]

    for col in categorical_defaults:

        if col not in df.columns:

            df[col] = "unknown"

        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # V5 features
    # --------------------------------------------------------

    df["frp_ratio_v5"] = (
        df["max_frp"]
        /
        (df["mean_frp"] + 1e-6)
    )

    df["frp_excess_v5"] = np.maximum(
        df["max_frp"] - 5.0,
        0
    )

    df["frp_log_v5"] = np.log1p(
        df["mean_frp"]
    )

    df["max_frp_log_v5"] = np.log1p(
        df["max_frp"]
    )

    df["brightness_range_v5"] = (
        df["brightness_range"]
    )

    df["brightness_ratio_v5"] = 1.0

    df["brightness_excess_v5"] = 0.0

    df["persistence_log_v5"] = np.log1p(
        df["persistence_days"]
    )

    df["persistence_months_v5"] = (
        df["persistence_days"] / 30.0
    )

    df["persistent_30d_v5"] = (
        df["persistence_days"] >= 30
    ).astype(int)

    df["persistent_90d_v5"] = (
        df["persistence_days"] >= 90
    ).astype(int)

    df["persistent_180d_v5"] = (
        df["persistence_days"] >= 180
    ).astype(int)

    df["persistent_270d_v5"] = (
        df["persistence_days"] >= 270
    ).astype(int)

    df["observation_log_v5"] = np.log1p(
        df["observation_count"]
    )

    df["obs_per_persistence_v5"] = (
        df["observation_count"]
        /
        (df["persistence_days"] + 1e-6)
    )

    df["activity_persistence_v5"] = (
        df["activity_density"]
        *
        df["persistence_days"]
    )

    df["distance_log_v5"] = np.log1p(
        df["distance_to_facility_km"]
    )

    df["very_close_v5"] = (
        df["distance_to_facility_km"] <= 1
    ).astype(int)

    df["within_2km_v5"] = (
        df["distance_to_facility_km"] <= 2
    ).astype(int)

    df["within_5km_v5"] = (
        df["distance_to_facility_km"] <= 5
    ).astype(int)

    df["within_10km_v5"] = (
        df["distance_to_facility_km"] <= 10
    ).astype(int)

    df["frp_persistence_v5"] = (
        df["mean_frp"]
        *
        df["persistence_days"]
    )

    df["frp_distance_signal_v5"] = (
        df["mean_frp"]
        /
        (
            df[
                "distance_to_facility_km"
            ]
            + 1
        )
    )

    df["activity_distance_signal_v5"] = (
        df["activity_density"]
        /
        (
            df[
                "distance_to_facility_km"
            ]
            + 1
        )
    )

    return df


# ============================================================
# V5 PREDICTION
# ============================================================

def run_v5_prediction(
    df,
    model
):

    if df.empty:

        return df.copy()

    df = engineer_features(
        df
    )

    missing = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing V5 features: "
            + str(missing)
        )

    X = df[
        FEATURES
    ].copy()

    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns

    X[numeric_features] = (
        X[numeric_features]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Categorical cleanup
    # --------------------------------------------------------

    categorical_features = [
        "facility_type",
        "facility_power",
        "facility_industrial",
        "facility_landuse"
    ]

    for col in categorical_features:

        X[col] = (
            X[col]
            .fillna("unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    confidence = (
        np.max(
            probabilities,
            axis=1
        )
        * 100
    )

    df["predicted_source"] = (
        predictions
    )

    df["confidence"] = (
        confidence.round(1)
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    frp_component = np.clip(
        df["mean_frp"] * 3,
        0,
        35
    )

    max_frp_component = np.clip(
        df["max_frp"] * 0.5,
        0,
        20
    )

    confidence_component = (
        confidence * 0.20
    )

    risk_score = (
        frp_component
        +
        max_frp_component
        +
        confidence_component
    )

    df["live_risk_score"] = np.clip(
        risk_score,
        0,
        100
    ).round(2)

    def risk_category(score):

        if score >= 75:
            return "CRITICAL"

        if score >= 50:
            return "HIGH"

        if score >= 25:
            return "MEDIUM"

        return "LOW"

    df["live_risk_category"] = (
        df[
            "live_risk_score"
        ].apply(
            risk_category
        )
    )

    return df


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown(
    "## ⚙️ THERMOWATCH CONTROLS"
)

time_options = {
    "24 Hours": 24,
    "48 Hours": 48,
    "72 Hours": 72,
    "96 Hours": 96,
    "7 Days": 168,
    "Custom": None,
}

selected_window = st.sidebar.selectbox(
    "Satellite observation window",
    list(time_options.keys()),
    index=0,
)

if selected_window == "Custom":

    hours = st.sidebar.number_input(
        "Hours",
        min_value=1,
        max_value=720,
        value=24,
        step=1,
    )

else:

    hours = time_options[
        selected_window
    ]

st.sidebar.markdown("---")

refresh = st.sidebar.button(
    "🔄 FETCH LATEST LIVE DATA",
    use_container_width=True,
)

st.sidebar.caption(
    "NASA FIRMS observations are filtered "
    "to the selected UTC time window."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
**CURRENT WINDOW**

`{hours} HOURS`

**DATA SOURCE**

NASA FIRMS VIIRS NOAA-20 NRT

**GEOSPATIAL**

OpenStreetMap

**AI**

V5 ExtraTrees Classifier
"""
)


# ============================================================
# SESSION CACHE INVALIDATION
# ============================================================

if refresh:

    st.cache_data.clear()

    st.rerun()


# ============================================================
# MAIN PIPELINE
# ============================================================

try:

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with st.spinner(
        "Loading V5 AI model..."
    ):

        model = load_model()

    # --------------------------------------------------------
    # OSM
    # --------------------------------------------------------

    with st.spinner(
        "Loading industrial facility database..."
    ):

        facilities = load_osm_facilities()

    # --------------------------------------------------------
    # FIRMS
    # --------------------------------------------------------

    with st.spinner(
        f"Fetching NASA FIRMS observations "
        f"for last {hours} hours..."
    ):

        live = fetch_firms(
            hours
        )

except Exception as e:

    st.error(
        "ThermoWatch LIVE pipeline failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# NO FIRE / NO OBSERVATION STATE
# ============================================================

if live.empty:

    st.markdown(
        f"""
<div class="no-fire">

<div class="no-fire-title">
🛰️ NO THERMAL OBSERVATIONS DETECTED
</div>

<div class="no-fire-text">
NASA FIRMS returned no satellite thermal observations
inside the selected <b>{hours}-hour</b> window over India.
</div>

<div class="no-fire-text">
This does <b>not</b> mean that fires can never occur;
it means that no FIRMS observations passed the current
time-window and India filters.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.info(
        "Use the sidebar to try 48 Hours, 72 Hours, "
        "96 Hours or a custom window."
    )

    st.markdown(
        f"""
<div class="notice">
<b>Monitoring status:</b> ONLINE &nbsp;·&nbsp;
<b>Window:</b> {hours} hours &nbsp;·&nbsp;
<b>Satellite source:</b> VIIRS NOAA-20 NRT &nbsp;·&nbsp;
<b>Observations:</b> 0
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
### 🔍 What ThermoWatch is doing

The application remains operational even when the
selected window has no detected hotspots.

The same dashboard can be used with a wider time window
to inspect recent satellite observations.
""",
    )

    st.stop()


# ============================================================
# FACILITY ATTRIBUTION
# ============================================================

with st.spinner(
    "Matching thermal hotspots with nearest OSM facilities..."
):

    attributed = attribute_facilities(
        live,
        facilities
    )


# ============================================================
# AI
# ============================================================

with st.spinner(
    "Running V5 AI source classification and risk assessment..."
):

    predictions = run_v5_prediction(
        attributed,
        model
    )


# ============================================================
# DISPLAY FACILITY NAME
# ============================================================

if "facility_name" not in predictions.columns:

    predictions[
        "facility_name"
    ] = ""

predictions[
    "display_facility_name"
] = predictions.apply(
    lambda row:
        clean_text(
            row.get(
                "facility_name",
                ""
            ),
            ""
        )
        or
        clean_text(
            row.get(
                "facility_operator",
                ""
            ),
            ""
        )
        or
        "Unidentified Facility",
    axis=1,
)


# ============================================================
# PIPELINE STATUS
# ============================================================

latest_observation = None

if "observed_at" in predictions.columns:

    parsed = pd.to_datetime(
        predictions["observed_at"],
        errors="coerce",
        utc=True
    )

    if parsed.notna().any():

        latest_observation = parsed.max()


latest_text = (
    format_observed(
        latest_observation
    )
    if latest_observation is not None
    else "Unavailable"
)


st.success(
    f"LIVE PIPELINE COMPLETE · "
    f"{len(predictions):,} satellite observations analyzed"
)


st.markdown(
    f"""
<div class="notice">
<b>Observation window:</b> Last {hours} hours
&nbsp; · &nbsp;
<b>Latest satellite observation:</b> {latest_text}
&nbsp; · &nbsp;
<b>Source:</b> NASA FIRMS VIIRS NOAA-20 NRT
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# METRICS
# ============================================================

total_hotspots = len(
    predictions
)

high_count = int(
    (
        predictions[
            "live_risk_category"
        ]
        == "HIGH"
    ).sum()
)

critical_count = int(
    (
        predictions[
            "live_risk_category"
        ]
        == "CRITICAL"
    ).sum()
)

medium_count = int(
    (
        predictions[
            "live_risk_category"
        ]
        == "MEDIUM"
    ).sum()
)

avg_risk = safe_float(
    predictions[
        "live_risk_score"
    ].mean()
)

max_risk = safe_float(
    predictions[
        "live_risk_score"
    ].max()
)


c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "🔥 LIVE HOTSPOTS",
        f"{total_hotspots:,}"
    )

with c2:

    st.metric(
        "🚨 HIGH RISK",
        high_count
    )

with c3:

    st.metric(
        "🔴 CRITICAL",
        critical_count
    )

with c4:

    st.metric(
        "⚠️ MEDIUM",
        medium_count
    )

with c5:

    st.metric(
        "RISK AVG",
        f"{avg_risk:.1f}"
    )


# ============================================================
# FILTERS
# ============================================================

st.markdown(
    '<div class="section-title">LIVE FILTERS</div>',
    unsafe_allow_html=True
)

f1, f2, f3, f4 = st.columns(4)

with f1:

    risk_filter = st.multiselect(
        "Risk",
        [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        default=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
    )

with f2:

    source_values = sorted(
        predictions[
            "predicted_source"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .unique()
        .tolist()
    )

    source_filter = st.multiselect(
        "AI Source",
        source_values,
        default=source_values,
    )

with f3:

    proximity_values = sorted(
        predictions[
            "proximity_category"
        ]
        .fillna("FAR")
        .astype(str)
        .unique()
        .tolist()
    )

    proximity_filter = st.multiselect(
        "Facility Proximity",
        proximity_values,
        default=proximity_values,
    )

with f4:

    min_frp = st.number_input(
        "Minimum FRP (MW)",
        min_value=0.0,
        max_value=float(
            max(
                1,
                predictions[
                    "frp"
                ].max()
            )
        ),
        value=0.0,
        step=0.5,
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = predictions.copy()

filtered = filtered[
    filtered[
        "live_risk_category"
    ].isin(
        risk_filter
    )
]

filtered = filtered[
    filtered[
        "predicted_source"
    ]
    .fillna("UNKNOWN")
    .isin(
        source_filter
    )
]

filtered = filtered[
    filtered[
        "proximity_category"
    ]
    .fillna("FAR")
    .isin(
        proximity_filter
    )
]

filtered = filtered[
    filtered[
        "frp"
    ].fillna(0)
    >= min_frp
].copy()


# ============================================================
# LAYOUT
# ============================================================

left, center, right = st.columns(
    [1.0, 2.2, 1.0],
    gap="small"
)


# ============================================================
# ALERT FEED
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">⚡ ALERT FEED</div>'
        '<div class="section-subtitle">'
        'RANKED BY LIVE RISK SCORE'
        '</div>',
        unsafe_allow_html=True,
    )

    alert_df = (
        filtered
        .sort_values(
            "live_risk_score",
            ascending=False
        )
        .head(12)
    )

    if alert_df.empty:

        st.info(
            "No hotspots match current filters."
        )

    else:

        for _, row in alert_df.iterrows():

            risk = clean_text(
                row[
                    "live_risk_category"
                ],
                "LOW"
            ).upper()

            color = risk_color(
                risk
            )

            facility = html.escape(
                clean_text(
                    row[
                        "display_facility_name"
                    ],
                    "Unidentified Facility"
                )
            )

            source = html.escape(
                source_badge(
                    row[
                        "predicted_source"
                    ]
                )
            )

            hotspot = html.escape(
                clean_text(
                    row[
                        "hotspot_id"
                    ]
                )
            )

            score = safe_float(
                row[
                    "live_risk_score"
                ]
            )

            frp = safe_float(
                row[
                    "frp"
                ]
            )

            st.markdown(
                f"""
<div class="alert-card">

<div style="
color:{color};
font-family:monospace;
font-weight:900;
font-size:11px;
">
{risk} · {score:.1f}
</div>

<div class="alert-title">
{facility}
</div>

<div style="
color:#596b7e;
font-family:monospace;
font-size:9px;
">
{hotspot}
</div>

<div class="alert-meta">
FRP {frp:.2f} MW · {source}
</div>

</div>
""",
                unsafe_allow_html=True,
            )


# ============================================================
# INDIA THERMAL MAP
# ============================================================

with center:

    st.markdown(
        '<div class="section-title">'
        '🇮🇳 INDIA THERMAL ACTIVITY MAP'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="section-subtitle">
{len(filtered):,} HOTSPOTS · THERMAL INTENSITY · AI RISK · FACILITY PROXIMITY
</div>
""",
        unsafe_allow_html=True,
    )

    m = folium.Map(
        location=[
            22.5,
            79.0
        ],
        zoom_start=5,
        min_zoom=4,
        max_zoom=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heat_data = []

    for row in filtered.itertuples():

        lat = getattr(
            row,
            "latitude",
            np.nan
        )

        lon = getattr(
            row,
            "longitude",
            np.nan
        )

        score = getattr(
            row,
            "live_risk_score",
            0
        )

        if (
            pd.notna(lat)
            and pd.notna(lon)
        ):

            heat_data.append(
                [
                    float(lat),
                    float(lon),
                    max(
                        float(score) / 100,
                        0.05
                    )
                ]
            )

    if heat_data:

        HeatMap(
            heat_data,
            name="🔥 Thermal Risk Zones",
            radius=25,
            blur=20,
            min_opacity=0.30,
            max_zoom=9,
        ).add_to(m)

    # --------------------------------------------------------
    # Marker cluster
    # --------------------------------------------------------

    cluster = MarkerCluster(
        name="Satellite Hotspots",
        options={
            "maxClusterRadius": 35,
            "disableClusteringAtZoom": 8,
        }
    ).add_to(m)

    # --------------------------------------------------------
    # Markers
    # --------------------------------------------------------

    for _, row in filtered.iterrows():

        lat = row[
            "latitude"
        ]

        lon = row[
            "longitude"
        ]

        if (
            pd.isna(lat)
            or pd.isna(lon)
        ):

            continue

        risk = clean_text(
            row[
                "live_risk_category"
            ],
            "LOW"
        ).upper()

        color = risk_color(
            risk
        )

        facility = html.escape(
            clean_text(
                row[
                    "display_facility_name"
                ],
                "Unidentified Facility"
            )
        )

        source = html.escape(
            clean_text(
                row[
                    "predicted_source"
                ],
                "UNKNOWN"
            )
        )

        hotspot = html.escape(
            clean_text(
                row[
                    "hotspot_id"
                ]
            )
        )

        frp = safe_float(
            row[
                "frp"
            ]
        )

        score = safe_float(
            row[
                "live_risk_score"
            ]
        )

        confidence = safe_float(
            row[
                "confidence"
            ]
        )

        distance = safe_float(
            row[
                "distance_to_facility_km"
            ],
            999
        )

        observed = html.escape(
            format_observed(
                row[
                    "observed_at"
                ]
            )
        )

        popup_html = f"""
<div style="
font-family:Arial;
min-width:270px;
">

<div style="
font-size:16px;
font-weight:800;
margin-bottom:7px;
">
{facility}
</div>

<hr>

<b>🔥 Hotspot</b><br>
{hotspot}<br><br>

<b>Risk</b><br>
<span style="
color:{color};
font-weight:800;
">
{risk} · {score:.1f}/100
</span>

<br><br>

<b>AI Source</b><br>
{source}

<br><br>

<b>AI Confidence</b><br>
{confidence:.1f}%

<br><br>

<b>FRP</b><br>
{frp:.2f} MW

<br><br>

<b>Facility Distance</b><br>
{distance:.3f} km

<br><br>

<b>Observed</b><br>
{observed}

</div>
"""

        folium.CircleMarker(
            location=[
                float(lat),
                float(lon)
            ],
            radius=6
            if risk == "LOW"
            else 8
            if risk == "MEDIUM"
            else 10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                popup_html,
                max_width=350
            ),
            tooltip=(
                f"{risk} · "
                f"Risk {score:.1f} · "
                f"FRP {frp:.2f} MW"
            ),
        ).add_to(cluster)

    # --------------------------------------------------------
    # Map controls
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    st_folium(
        m,
        width=None,
        height=650,
        returned_objects=[],
    )


# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">'
        '🚨 HIGH PRIORITY'
        '</div>',
        unsafe_allow_html=True,
    )

    high_priority = (
        filtered
        .sort_values(
            "live_risk_score",
            ascending=False
        )
        .head(10)
    )

    if high_priority.empty:

        st.info(
            "No high-priority observations."
        )

    else:

        for _, row in high_priority.iterrows():

            risk = clean_text(
                row[
                    "live_risk_category"
                ],
                "LOW"
            )

            color = risk_color(
                risk
            )

            facility = clean_text(
                row[
                    "display_facility_name"
                ],
                "Unidentified Facility"
            )

            score = safe_float(
                row[
                    "live_risk_score"
                ]
            )

            frp = safe_float(
                row[
                    "frp"
                ]
            )

            source = clean_text(
                row[
                    "predicted_source"
                ],
                "UNKNOWN"
            )

            st.markdown(
                f"""
<div class="alert-card">

<div style="
color:{color};
font-family:monospace;
font-weight:900;
">
{risk} · {score:.1f}
</div>

<div style="
font-weight:800;
margin-top:4px;
">
{html.escape(facility)}
</div>

<div class="alert-meta">
FRP {frp:.2f} MW ·
{html.escape(source)}
</div>

</div>
""",
                unsafe_allow_html=True
            )


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 AI SOURCE CLASSIFICATION'
    '</div>',
    unsafe_allow_html=True,
)

source_counts = (
    filtered[
        "predicted_source"
    ]
    .fillna("UNKNOWN")
    .value_counts()
)

if not source_counts.empty:

    st.bar_chart(
        source_counts
    )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⚠️ RISK DISTRIBUTION'
    '</div>',
    unsafe_allow_html=True,
)

risk_counts = (
    filtered[
        "live_risk_category"
    ]
    .value_counts()
)

if not risk_counts.empty:

    st.bar_chart(
        risk_counts
    )


# ============================================================
# TOP DETECTIONS TABLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📡 LIVE DETECTION TABLE'
    '</div>',
    unsafe_allow_html=True,
)

display_cols = [

    "hotspot_id",

    "display_facility_name",

    "latitude",
    "longitude",

    "observed_at",

    "frp",

    "predicted_source",

    "confidence",

    "distance_to_facility_km",

    "facility_match_quality",

    "live_risk_score",

    "live_risk_category",
]

display_cols = [
    c
    for c in display_cols
    if c in filtered.columns
]

table = (
    filtered
    .sort_values(
        "live_risk_score",
        ascending=False
    )
    [display_cols]
    .head(100)
    .copy()
)

table = table.rename(
    columns={

        "hotspot_id":
            "Hotspot ID",

        "display_facility_name":
            "Facility",

        "latitude":
            "Latitude",

        "longitude":
            "Longitude",

        "observed_at":
            "Observed At",

        "frp":
            "FRP (MW)",

        "predicted_source":
            "AI Source",

        "confidence":
            "Confidence %",

        "distance_to_facility_km":
            "Facility Distance (km)",

        "facility_match_quality":
            "Attribution",

        "live_risk_score":
            "Risk Score",

        "live_risk_category":
            "Risk",
    }
)

if "Observed At" in table.columns:

    table["Observed At"] = (
        table[
            "Observed At"
        ].apply(
            format_observed
        )
    )

if "Latitude" in table.columns:

    table["Latitude"] = (
        table["Latitude"].round(5)
    )

if "Longitude" in table.columns:

    table["Longitude"] = (
        table["Longitude"].round(5)
    )

if "FRP (MW)" in table.columns:

    table["FRP (MW)"] = (
        table["FRP (MW)"].round(2)
    )

if "Confidence %" in table.columns:

    table["Confidence %"] = (
        table["Confidence %"].round(1)
    )

if "Facility Distance (km)" in table.columns:

    table[
        "Facility Distance (km)"
    ] = (
        table[
            "Facility Distance (km)"
        ].round(3)
    )

if "Risk Score" in table.columns:

    table["Risk Score"] = (
        table["Risk Score"].round(2)
    )


st.dataframe(
    table,
    use_container_width=True,
    height=450,
    hide_index=True,
)


# ============================================================
# EXPORT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⬇️ EXPORT LIVE RESULTS'
    '</div>',
    unsafe_allow_html=True,
)

csv_data = (
    filtered
    .to_csv(
        index=False
    )
    .encode("utf-8")
)

st.download_button(
    label="⬇️ Download LIVE ThermoWatch CSV",
    data=csv_data,
    file_name=(
        f"thermowatch_live_"
        f"{hours}h.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# DATA PROVENANCE
# ============================================================

st.markdown(
    f"""
<div class="footer">

THERMOWATCH AI · V5 MODEL · LIVE SATELLITE MONITORING

&nbsp; | &nbsp;

OBSERVATION WINDOW: {hours} HOURS

&nbsp; | &nbsp;

OBSERVATIONS: {len(predictions):,}

&nbsp; | &nbsp;

VISIBLE: {len(filtered):,}

&nbsp; | &nbsp;

LATEST OBSERVATION: {latest_text}

&nbsp; | &nbsp;

SOURCE: NASA FIRMS VIIRS NOAA-20 NRT

&nbsp; | &nbsp;

FACILITY CONTEXT: OPENSTREETMAP

</div>
""",
    unsafe_allow_html=True,
)
