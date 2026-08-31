import os
import io
from datetime import timedelta

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.neighbors import BallTree


# ============================================================
# THERMO WATCH AI
# NASA FIRMS -> OSM FACILITIES -> V5 AI -> RISK
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
    "https://huggingface.co/"
    "harshitcodes1544/sihharshit154/"
    "resolve/main/source_classifier_v5.joblib"
)

HF_OSM_URL = (
    "https://huggingface.co/datasets/"
    "harshitcodes1544/thermowatch-osm-data/"
    "resolve/main/osm_facilities.csv"
)

# Previous local result - useful as fallback/reference
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LOCAL_PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "live_final_predictions.csv"
)

LOCAL_FACILITY_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "osm_facilities.csv"
)


# ============================================================
# FIRMS CONFIG
# ============================================================

FIRMS_API_KEY = st.secrets.get(
    "FIRMS_API_KEY",
    ""
)

FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/area/csv/"
)

# India:
# west, south, east, north
INDIA_BBOX = "68.0,6.0,97.5,37.5"


# ============================================================
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
    "activity_distance_signal_v5"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    response = requests.get(
        HF_MODEL_URL,
        timeout=120
    )

    response.raise_for_status()

    model = joblib.load(
        io.BytesIO(response.content)
    )

    return model


# ============================================================
# LOAD OSM FACILITIES
# ============================================================

@st.cache_data(ttl=86400)
def load_osm_facilities():

    # --------------------------------------------------------
    # First try Hugging Face
    # --------------------------------------------------------

    try:

        response = requests.get(
            HF_OSM_URL,
            timeout=180
        )

        response.raise_for_status()

        df = pd.read_csv(
            io.BytesIO(
                response.content
            )
        )

        if not df.empty:
            return df

    except Exception as hf_error:

        # ----------------------------------------------------
        # Fallback to local file
        # ----------------------------------------------------

        if os.path.exists(
            LOCAL_FACILITY_FILE
        ):

            return pd.read_csv(
                LOCAL_FACILITY_FILE
            )

        raise RuntimeError(
            "Unable to load OSM facility data.\n\n"
            f"Hugging Face error: {hf_error}"
        )

    raise RuntimeError(
        "OSM facility dataset is empty."
    )


# ============================================================
# FETCH LIVE FIRMS
# ============================================================

@st.cache_data(ttl=900)
def fetch_live_firms():

    if not FIRMS_API_KEY:

        raise RuntimeError(
            "FIRMS_API_KEY is missing from "
            "Streamlit Secrets."
        )

    # --------------------------------------------------------
    # VIIRS NOAA-20 NRT
    # --------------------------------------------------------

    url = (
        FIRMS_URL
        + FIRMS_API_KEY
        + "/VIIRS_NOAA20_NRT/"
        + INDIA_BBOX
        + "/1"
    )

    try:

        response = requests.get(
            url,
            timeout=120
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"NASA FIRMS request failed: {e}"
        )

    # --------------------------------------------------------
    # Parse CSV
    # --------------------------------------------------------

    if not response.text.strip():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            io.StringIO(
                response.text
            )
        )

    except Exception as e:

        raise RuntimeError(
            f"Unable to parse NASA FIRMS response: {e}"
        )

    # --------------------------------------------------------
    # Empty FIRMS response is NOT an error
    # --------------------------------------------------------

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "latitude",
        "longitude"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "NASA FIRMS response is missing "
            "required columns: "
            + ", ".join(missing)
        )

    # ========================================================
    # CLEAN DATA
    # ========================================================

    numeric_columns = [

        "latitude",
        "longitude",

        "bright_ti4",
        "bright_ti5",

        "scan",
        "track",

        "frp"
    ]

    for col in numeric_columns:

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

    if df.empty:

        return pd.DataFrame()

    # ========================================================
    # CREATE ACQUISITION DATETIME
    # ========================================================

    if (
        "acq_date" in df.columns
        and
        "acq_time" in df.columns
    ):

        df["acq_time"] = (
            df["acq_time"]
            .astype(str)
            .str.replace(
                ".0",
                "",
                regex=False
            )
            .str.zfill(4)
        )

        df["acquisition_datetime"] = pd.to_datetime(
            df["acq_date"].astype(str)
            + " "
            + df["acq_time"].str[:2]
            + ":"
            + df["acq_time"].str[2:],
            errors="coerce",
            utc=True
        )

    else:

        df["acquisition_datetime"] = pd.NaT

    # ========================================================
    # LAST 24 HOURS FILTER
    # ========================================================

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    cutoff = (
        now_utc
        - pd.Timedelta(hours=24)
    )

    if (
        "acquisition_datetime" in df.columns
        and
        df["acquisition_datetime"].notna().any()
    ):

        df = df[
            (
                df["acquisition_datetime"]
                >= cutoff
            )
            &
            (
                df["acquisition_datetime"]
                <= now_utc
            )
        ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Empty last-24-hour result is NORMAL.
    # Do NOT raise an exception.
    # --------------------------------------------------------

    if df.empty:

        return pd.DataFrame()

    # ========================================================
    # HOTSPOT IDS
    # ========================================================

    df = df.reset_index(
        drop=True
    )

    df["hotspot_id"] = [
        f"LIVE_{i:06d}"
        for i in range(
            1,
            len(df) + 1
        )
    ]

    df["fetched_at"] = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    return df


# ============================================================
# FACILITY ATTRIBUTION
# ============================================================

def attribute_facilities(
    live,
    facilities
):

    live = live.copy()
    facilities = facilities.copy()

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    for col in [
        "latitude",
        "longitude"
    ]:

        live[col] = pd.to_numeric(
            live[col],
            errors="coerce"
        )

        facilities[col] = pd.to_numeric(
            facilities[col],
            errors="coerce"
        )

    live = live.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).reset_index(
        drop=True
    )

    facilities = facilities.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).reset_index(
        drop=True
    )

    if live.empty:

        return pd.DataFrame()

    if facilities.empty:

        raise RuntimeError(
            "OSM facility dataset contains "
            "no valid coordinates."
        )

    # ========================================================
    # BALL TREE
    # ========================================================

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

    earth_radius_km = 6371.0088

    distances_km = (
        distances[:, 0]
        * earth_radius_km
    )

    nearest = facilities.iloc[
        indices[:, 0]
    ].reset_index(
        drop=True
    )

    result = live.copy()

    # ========================================================
    # FACILITY FIELDS
    # ========================================================

    field_mapping = {

        "facility_osm_id":
            "osm_id",

        "facility_type":
            "feature_type",

        "facility_name":
            "name",

        "facility_power":
            "power",

        "facility_industrial":
            "industrial",

        "facility_landuse":
            "landuse",

        "facility_operator":
            "operator",

        "facility_plant_source":
            "plant_source",

        "facility_plant_method":
            "plant_method"
    }

    for output_col, input_col in field_mapping.items():

        if input_col in nearest.columns:

            result[output_col] = (
                nearest[
                    input_col
                ].values
            )

        else:

            result[output_col] = np.nan

    result["facility_latitude"] = (
        nearest[
            "latitude"
        ].values
    )

    result["facility_longitude"] = (
        nearest[
            "longitude"
        ].values
    )

    result["distance_to_facility_km"] = (
        distances_km
    )

    # ========================================================
    # PROXIMITY
    # ========================================================

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
    ] = (
        result[
            "distance_to_facility_km"
        ]
        .apply(proximity)
    )

    # ========================================================
    # MATCH QUALITY
    # ========================================================

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
    ] = (
        result[
            "distance_to_facility_km"
        ]
        .apply(quality)
    )

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

    # ========================================================
    # FRP
    # ========================================================

    df["mean_frp"] = (
        pd.to_numeric(
            df.get(
                "frp",
                0
            ),
            errors="coerce"
        )
        .fillna(0)
    )

    df["max_frp"] = (
        df["mean_frp"]
    )

    df["frp_ratio"] = (
        df["max_frp"]
        /
        (
            df["mean_frp"]
            + 1e-6
        )
    )

    # ========================================================
    # BRIGHTNESS
    # ========================================================

    if "bright_ti4" in df.columns:

        brightness = pd.to_numeric(
            df["bright_ti4"],
            errors="coerce"
        ).fillna(0)

    else:

        brightness = pd.Series(
            0.0,
            index=df.index
        )

    df["mean_brightness"] = (
        brightness
    )

    df["max_brightness"] = (
        brightness
    )

    df["brightness_range"] = 0.0

    # ========================================================
    # OBSERVATION FEATURES
    # ========================================================

    df["observation_count"] = 1

    df["persistence_days"] = 1

    df["observations_per_day"] = 1.0

    df["activity_density"] = 1.0

    df["satellite_count"] = 1

    # ========================================================
    # FACILITY FEATURES
    # ========================================================

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
    ).fillna(999)

    df["facility_within_1km"] = (
        df[
            "distance_to_facility_km"
        ] <= 1
    ).astype(int)

    df["facility_within_5km"] = (
        df[
            "distance_to_facility_km"
        ] <= 5
    ).astype(int)

    df["persistent_long_term"] = 0

    df["persistent_180_days"] = 0

    # ========================================================
    # CATEGORICAL
    # ========================================================

    categorical = [

        "facility_type",
        "facility_power",
        "facility_industrial",
        "facility_landuse"
    ]

    for col in categorical:

        if col not in df.columns:

            df[col] = "unknown"

        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
        )

    # ========================================================
    # V5 FEATURES
    # ========================================================

    df["frp_ratio_v5"] = (
        df["max_frp"]
        /
        (
            df["mean_frp"]
            + 1e-6
        )
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

    df["brightness_range_v5"] = 0.0

    df["brightness_ratio_v5"] = 1.0

    df["brightness_excess_v5"] = 0.0

    df["persistence_log_v5"] = np.log1p(
        df["persistence_days"]
    )

    df["persistence_months_v5"] = (
        df["persistence_days"]
        / 30.0
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
        (
            df["persistence_days"]
            + 1e-6
        )
    )

    df["activity_persistence_v5"] = (
        df["activity_density"]
        *
        df["persistence_days"]
    )

    df["distance_log_v5"] = np.log1p(
        df[
            "distance_to_facility_km"
        ]
    )

    df["very_close_v5"] = (
        df[
            "distance_to_facility_km"
        ] <= 1
    ).astype(int)

    df["within_2km_v5"] = (
        df[
            "distance_to_facility_km"
        ] <= 2
    ).astype(int)

    df["within_5km_v5"] = (
        df[
            "distance_to_facility_km"
        ] <= 5
    ).astype(int)

    df["within_10km_v5"] = (
        df[
            "distance_to_facility_km"
        ] <= 10
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
# RUN V5 AI
# ============================================================

def run_v5_prediction(
    df,
    model
):

    if df.empty:

        return pd.DataFrame()

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

    # ========================================================
    # NUMERIC CLEANUP
    # ========================================================

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        ).columns
    )

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

    # ========================================================
    # CATEGORICAL CLEANUP
    # ========================================================

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

    # ========================================================
    # PREDICTION
    # ========================================================

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

    df[
        "predicted_source"
    ] = predictions

    df[
        "confidence"
    ] = confidence.round(1)

    # ========================================================
    # RISK SCORE
    # ========================================================

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

    risk_score = np.clip(
        risk_score,
        0,
        100
    )

    df[
        "live_risk_score"
    ] = risk_score.round(2)

    # ========================================================
    # RISK CATEGORY
    # ========================================================

    def risk_category(score):

        if score >= 75:
            return "CRITICAL"

        if score >= 50:
            return "HIGH"

        if score >= 25:
            return "MEDIUM"

        return "LOW"

    df[
        "live_risk_category"
    ] = (
        df[
            "live_risk_score"
        ]
        .apply(risk_category)
    )

    return df


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_previous_results():

    if not os.path.exists(
        LOCAL_PREDICTIONS_FILE
    ):

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            LOCAL_PREDICTIONS_FILE
        )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔥 ThermoWatch AI"
)

st.subheader(
    "LIVE Satellite Fire Detection & "
    "Industrial Risk Intelligence"
)

st.caption(
    "NASA FIRMS → OSM Facilities → "
    "V5 AI → Risk Assessment"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "ThermoWatch Controls"
)

refresh = st.sidebar.button(
    "🔄 Fetch Latest LIVE Data"
)

if refresh:

    st.cache_data.clear()

    st.rerun()

st.sidebar.info(
    "NASA FIRMS observations are filtered "
    "to the latest 24-hour window."
)


# ============================================================
# STEP 1 - MODEL
# ============================================================

try:

    with st.spinner(
        "Loading V5 AI model..."
    ):

        model = load_model()

except Exception as e:

    st.error(
        "❌ Unable to load V5 AI model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# STEP 2 - OSM
# ============================================================

try:

    with st.spinner(
        "Loading OSM facility database..."
    ):

        facilities = (
            load_osm_facilities()
        )

    st.success(
        f"OSM facilities loaded: "
        f"{len(facilities):,}"
    )

except Exception as e:

    st.error(
        "❌ Unable to load OSM facilities."
    )

    st.exception(e)

    st.stop()


# ============================================================
# STEP 3 - LIVE FIRMS
# ============================================================

try:

    with st.spinner(
        "Fetching latest NASA FIRMS observations..."
    ):

        live = fetch_live_firms()

except Exception as e:

    st.error(
        "❌ NASA FIRMS request failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMPORTANT:
# NO LIVE FIRE IS NOT AN ERROR
# ============================================================

if live.empty:

    st.warning(
        "🔥 No fire observations were detected "
        "in the last 24 hours."
    )

    st.info(
        "This is a valid live-data state, "
        "not an application failure."
    )

    st.metric(
        "LIVE Fires — Last 24 Hours",
        "0"
    )

    st.divider()

    st.subheader(
        "📊 ThermoWatch Status"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            "✓ V5 AI Model Ready"
        )

    with col2:

        st.success(
            f"✓ OSM Facilities Ready "
            f"({len(facilities):,})"
        )

    with col3:

        st.success(
            "✓ FIRMS Connected"
        )

    # --------------------------------------------------------
    # Previous results
    # --------------------------------------------------------

    previous = load_previous_results()

    if not previous.empty:

        st.divider()

        st.subheader(
            "🕘 Previous ThermoWatch Results"
        )

        st.caption(
            "No new fire was detected in the "
            "last 24 hours. The table below is "
            "historical/reference data and is "
            "NOT being presented as current."
        )

        previous_display = previous.copy()

        # Clearly label historical data
        previous_display[
            "data_status"
        ] = "HISTORICAL / REFERENCE"

        st.dataframe(
            previous_display.head(20),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No previous ThermoWatch prediction "
            "file is available."
        )

    st.divider()

    st.caption(
        "ThermoWatch will automatically analyze "
        "new FIRMS observations when they become "
        "available."
    )

    st.stop()


# ============================================================
# LIVE DATA AVAILABLE
# ============================================================

st.success(
    f"🔥 {len(live):,} LIVE fire observations "
    "detected in the last 24 hours."
)


# ============================================================
# STEP 4 - FACILITY ATTRIBUTION
# ============================================================

try:

    with st.spinner(
        "Matching live hotspots with nearest OSM facilities..."
    ):

        attributed = attribute_facilities(
            live,
            facilities
        )

except Exception as e:

    st.error(
        "❌ Facility attribution failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# STEP 5 - V5 AI
# ============================================================

try:

    with st.spinner(
        "Running V5 AI source classification and risk assessment..."
    ):

        predictions = run_v5_prediction(
            attributed,
            model
        )

except Exception as e:

    st.error(
        "❌ V5 AI inference failed."
    )

    st.exception(e)

    st.stop()


if predictions.empty:

    st.warning(
        "No predictions were generated."
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

st.success(
    "✓ LIVE FIRMS → OSM → V5 AI pipeline completed."
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "LIVE Hotspots",
        f"{len(predictions):,}"
    )


with col2:

    high_count = int(
        (
            predictions[
                "live_risk_category"
            ]
            == "HIGH"
        ).sum()
    )

    st.metric(
        "HIGH Risk",
        high_count
    )


with col3:

    critical_count = int(
        (
            predictions[
                "live_risk_category"
            ]
            == "CRITICAL"
        ).sum()
    )

    st.metric(
        "CRITICAL",
        critical_count
    )


with col4:

    st.metric(
        "Average Risk",
        round(
            predictions[
                "live_risk_score"
            ].mean(),
            2
        )
    )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.header(
    "📊 LIVE Risk Distribution"
)

risk_counts = (
    predictions[
        "live_risk_category"
    ]
    .value_counts()
)

st.bar_chart(
    risk_counts
)


# ============================================================
# AI SOURCE DISTRIBUTION
# ============================================================

st.header(
    "🤖 AI Source Classification"
)

source_counts = (
    predictions[
        "predicted_source"
    ]
    .value_counts()
)

st.bar_chart(
    source_counts
)


# ============================================================
# LIVE MAP
# ============================================================

st.header(
    "🗺️ LIVE Fire Hotspots"
)

map_df = predictions[
    [
        "latitude",
        "longitude"
    ]
].dropna()

if not map_df.empty:

    st.map(
        map_df
    )

else:

    st.info(
        "No valid coordinates available."
    )


# ============================================================
# HIGH RISK DETECTIONS
# ============================================================

st.header(
    "🚨 Highest Risk LIVE Detections"
)

display_cols = [

    "hotspot_id",

    "facility_name",

    "latitude",
    "longitude",

    "acquisition_datetime",

    "frp",

    "predicted_source",

    "confidence",

    "distance_to_facility_km",

    "facility_match_quality",

    "live_risk_score",

    "live_risk_category"
]

display_cols = [
    c
    for c in display_cols
    if c in predictions.columns
]

top_risk = (
    predictions
    .sort_values(
        "live_risk_score",
        ascending=False
    )
    [
        display_cols
    ]
    .head(20)
)

st.dataframe(
    top_risk,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FACILITY PROXIMITY
# ============================================================

st.header(
    "🏭 Facility Proximity"
)

proximity_counts = (
    predictions[
        "proximity_category"
    ]
    .value_counts()
)

st.bar_chart(
    proximity_counts
)


# ============================================================
# LIVE DATA TABLE
# ============================================================

st.header(
    "📡 LIVE FIRMS Observations"
)

live_display_cols = [

    "hotspot_id",

    "latitude",
    "longitude",

    "acquisition_datetime",

    "frp",

    "bright_ti4",

    "satellite",

    "confidence",

    "facility_name",

    "facility_type",

    "distance_to_facility_km",

    "predicted_source",

    "live_risk_score",

    "live_risk_category"
]

live_display_cols = [
    c
    for c in live_display_cols
    if c in predictions.columns
]

st.dataframe(
    predictions[
        live_display_cols
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

st.header(
    "⬇️ Export LIVE Results"
)

csv_data = predictions.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download LIVE Predictions CSV",
    data=csv_data,
    file_name="thermowatch_live_predictions.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ThermoWatch AI | "
    "NASA FIRMS + OSM + V5 Machine Learning | "
    "LIVE 24-Hour Fire Intelligence"
)
