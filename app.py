import io
import math
import html
from pathlib import Path

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
# LIVE FIRMS -> OSM FACILITIES -> V5 AI -> RISK
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
# west,south,east,north
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

.metric-card {
    border: 1px solid #172536;
    background:
        linear-gradient(
            180deg,
            rgba(12,22,34,.96),
            rgba(7,13,21,.96)
        );
    border-radius: 12px;
    padding: 14px;
}

.metric-label {
    color: #718096;
    font-family: monospace;
    font-size: 10px;
    text-transform: uppercase;
}

.metric-value {
    font-size: 25px;
    font-weight: 900;
    margin-top: 4px;
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
    padding: 30px;
    text-align: center;
    margin: 18px 0;
}

.no-fire-title {
    font-size: 23px;
    font-weight: 900;
    color: #d9e7f5;
}

.no-fire-text {
    color: #8295aa;
    font-size: 13px;
    margin-top: 9px;
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
    margin-top: 4px;
}

.alert-id {
    color: #596b7e;
    font-family: monospace;
    font-size: 9px;
    margin-top: 3px;
}

.alert-meta {
    color: #718096;
    font-family: monospace;
    font-size: 10px;
    margin-top: 5px;
}

.footer {
    color: #5f7083;
    font-family: monospace;
    font-size: 10px;
    border-top: 1px solid #172536;
    padding-top: 12px;
    margin-top: 20px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1a2b3d;
    border-radius: 8px;
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

    <div class="tw-title">
        🔥 ThermoWatch <span style="color:#f5b42c">AI</span>
    </div>

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

    if text.lower() in {
        "",
        "nan",
        "none",
        "null"
    }:
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


def risk_color(category):

    category = clean_text(
        category,
        "LOW"
    ).upper()

    return {
        "CRITICAL": "#ff405d",
        "HIGH": "#ff9f32",
        "MEDIUM": "#ffd23f",
        "LOW": "#29a9ff",
    }.get(
        category,
        "#29a9ff"
    )


def source_badge(source):

    source = clean_text(
        source,
        "UNKNOWN"
    ).upper()

    if source == "INDUSTRIAL":

        return (
            '<span style="'
            'color:#ffb347;'
            'border:1px solid #6d4816;'
            'background:#251a08;'
            'padding:2px 6px;'
            'border-radius:4px;'
            'font-size:9px;'
            'font-family:monospace;'
            '">'
            'INDUSTRIAL'
            '</span>'
        )

    if source == "POWER_GENERATION":

        return (
            '<span style="'
            'color:#5bbcff;'
            'border:1px solid #1d5275;'
            'background:#071824;'
            'padding:2px 6px;'
            'border-radius:4px;'
            'font-size:9px;'
            'font-family:monospace;'
            '">'
            'POWER'
            '</span>'
        )

    return (
        '<span style="'
        'color:#aab7c5;'
        'border:1px solid #344454;'
        'background:#101820;'
        'padding:2px 6px;'
        'border-radius:4px;'
        'font-size:9px;'
        'font-family:monospace;'
        '">'
        + html.escape(source)
        + '</span>'
    )


def facility_label(row):

    name = clean_text(
        row.get("facility_name"),
        ""
    )

    if name:
        return name

    ftype = clean_text(
        row.get("facility_type"),
        ""
    ).lower()

    power = clean_text(
        row.get("facility_power"),
        ""
    ).lower()

    industrial = clean_text(
        row.get("facility_industrial"),
        ""
    ).lower()

    if (
        "substation" in ftype
        or "substation" in power
    ):
        return "Unnamed Power Substation"

    if (
        "generator" in ftype
        or "generator" in power
        or "plant" in power
    ):
        return "Unnamed Power Facility"

    if (
        industrial
        or "industrial" in ftype
    ):
        return "Unnamed Industrial Facility"

    return "Unidentified Facility"


# ============================================================
# V5 FEATURE LIST
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
# LOAD V5 MODEL
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
            "Could not download V5 model. "
            f"HTTP {response.status_code}"
        )

    try:

        model = joblib.load(
            io.BytesIO(
                response.content
            )
        )

    except Exception as e:

        raise RuntimeError(
            "Downloaded V5 model could not be loaded: "
            + str(e)
        )

    return model


# ============================================================
# LOAD OSM FACILITIES
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_osm_facilities():

    response = requests.get(
        HF_OSM_URL,
        timeout=300,
        allow_redirects=True
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Could not download OSM facilities. "
            f"HTTP {response.status_code}"
        )

    try:

        df = pd.read_csv(
            io.BytesIO(
                response.content
            )
        )

    except Exception as e:

        raise RuntimeError(
            "Could not read OSM facilities CSV: "
            + str(e)
        )

    required = [
        "latitude",
        "longitude"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "OSM dataset missing columns: "
            + str(missing)
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
# FETCH LIVE NASA FIRMS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def fetch_firms(days):

    if "FIRMS_API_KEY" not in st.secrets:

        raise RuntimeError(
            "FIRMS_API_KEY is missing from "
            "Streamlit Secrets."
        )

    api_key = st.secrets[
        "FIRMS_API_KEY"
    ]

    days = int(days)

    # --------------------------------------------------------
    # NASA FIRMS AREA API maximum request window = 5 days
    # --------------------------------------------------------

    if days < 1:
        days = 1

    if days > 5:

        raise RuntimeError(
            "NASA FIRMS NRT area endpoint supports "
            "a maximum of 5 days per request. "
            "Please select 5 Days or less."
        )

    url = (
        FIRMS_BASE_URL
        + api_key
        + "/VIIRS_NOAA20_NRT/"
        + INDIA_BBOX
        + f"/{days}"
    )

    try:

        response = requests.get(
            url,
            timeout=180
        )

        response.raise_for_status()

    except requests.RequestException as e:

        raise RuntimeError(
            "NASA FIRMS request failed: "
            + str(e)
        )

    # --------------------------------------------------------
    # Empty API response
    # --------------------------------------------------------

    if not response.text.strip():

        return pd.DataFrame()

    # --------------------------------------------------------
    # Parse CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            io.StringIO(
                response.text
            )
        )

    except Exception as e:

        raise RuntimeError(
            "NASA FIRMS returned an unreadable response: "
            + str(e)
        )

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    expected_columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "bright_ti5",
        "frp",
        "daynight",
    ]

    for col in expected_columns:

        if col not in df.columns:

            df[col] = np.nan

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [

        "latitude",
        "longitude",

        "bright_ti4",
        "bright_ti5",

        "scan",
        "track",

        "frp",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Acquisition time
    # --------------------------------------------------------

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
        + df["acq_time"].str[2:4],
        errors="coerce",
        utc=True
    )

    # --------------------------------------------------------
    # India coordinate validation
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    df = df[
        (df["latitude"] >= 6.0)
        & (df["latitude"] <= 38.0)
        & (df["longitude"] >= 68.0)
        & (df["longitude"] <= 98.0)
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # FIRMS feed may contain observations from the requested
    # API window. We keep the real acquisition timestamp.
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    # Stable hotspot IDs
    df["hotspot_id"] = [
        f"LIVE_{i:06d}"
        for i in range(
            1,
            len(df) + 1
        )
    ]

    return df


# ============================================================
# FACILITY ATTRIBUTION
# ============================================================

def attribute_facilities(
    live,
    facilities
):

    if live.empty:

        return live.copy()

    if facilities.empty:

        result = live.copy()

        for col in [
            "facility_osm_id",
            "facility_type",
            "facility_name",
            "facility_power",
            "facility_industrial",
            "facility_landuse",
            "facility_operator",
            "facility_plant_source",
            "facility_plant_method",
            "facility_latitude",
            "facility_longitude",
            "distance_to_facility_km",
            "proximity_category",
            "facility_match_quality",
        ]:

            result[col] = np.nan

        return result

    # --------------------------------------------------------
    # Facility coordinates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Spatial index
    # --------------------------------------------------------

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

    nearest = (
        facilities.iloc[
            indices[:, 0]
        ]
        .reset_index(drop=True)
    )

    result = (
        live
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Attach OSM fields
    # --------------------------------------------------------

    def attach(
        output_col,
        source_col
    ):

        if source_col in nearest.columns:

            result[output_col] = (
                nearest[source_col]
                .values
            )

        else:

            result[output_col] = np.nan

    attach(
        "facility_osm_id",
        "osm_id"
    )

    attach(
        "facility_type",
        "feature_type"
    )

    attach(
        "facility_name",
        "name"
    )

    attach(
        "facility_power",
        "power"
    )

    attach(
        "facility_industrial",
        "industrial"
    )

    attach(
        "facility_landuse",
        "landuse"
    )

    attach(
        "facility_operator",
        "operator"
    )

    attach(
        "facility_plant_source",
        "plant_source"
    )

    attach(
        "facility_plant_method",
        "plant_method"
    )

    result["facility_latitude"] = (
        nearest["latitude"]
        .values
    )

    result["facility_longitude"] = (
        nearest["longitude"]
        .values
    )

    result["distance_to_facility_km"] = (
        distances_km
    )

    # --------------------------------------------------------
    # Proximity
    # --------------------------------------------------------

    def proximity_category(
        distance
    ):

        if distance <= 1:
            return "VERY_CLOSE"

        if distance <= 2:
            return "CLOSE"

        if distance <= 5:
            return "NEAR"

        if distance <= 10:
            return "DISTANT"

        return "FAR"

    result["proximity_category"] = (
        result[
            "distance_to_facility_km"
        ].apply(
            proximity_category
        )
    )

    # --------------------------------------------------------
    # Match quality
    # --------------------------------------------------------

    def match_quality(
        distance
    ):

        if distance <= 1:
            return "HIGH"

        if distance <= 5:
            return "MEDIUM"

        if distance <= 10:
            return "LOW"

        return "VERY_LOW"

    result["facility_match_quality"] = (
        result[
            "distance_to_facility_km"
        ].apply(
            match_quality
        )
    )

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(
    df
):

    df = df.copy()

    # --------------------------------------------------------
    # Thermal features
    # --------------------------------------------------------

    df["mean_frp"] = pd.to_numeric(
        df.get(
            "frp",
            0
        ),
        errors="coerce"
    ).fillna(0)

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

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    df["mean_brightness"] = (
        pd.to_numeric(
            df.get(
                "bright_ti4",
                0
            ),
            errors="coerce"
        )
        .fillna(0)
    )

    df["max_brightness"] = (
        df["mean_brightness"]
    )

    df["brightness_range"] = 0.0

    # --------------------------------------------------------
    # Observation features
    # --------------------------------------------------------

    df["observation_count"] = 1

    df["persistence_days"] = 1

    df["observations_per_day"] = 1.0

    df["activity_density"] = 1.0

    df["satellite_count"] = 1

    # --------------------------------------------------------
    # Facility defaults
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

    for col in [
        "facility_within_1km",
        "facility_within_5km",
        "persistent_long_term",
        "persistent_180_days",
    ]:

        if col not in df.columns:

            df[col] = 0

    categorical = [
        "facility_type",
        "facility_power",
        "facility_industrial",
        "facility_landuse",
    ]

    for col in categorical:

        if col not in df.columns:

            df[col] = "unknown"

        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # Facility proximity flags
    # --------------------------------------------------------

    df[
        "facility_within_1km"
    ] = (
        df[
            "distance_to_facility_km"
        ] <= 1
    ).astype(int)

    df[
        "facility_within_5km"
    ] = (
        df[
            "distance_to_facility_km"
        ] <= 5
    ).astype(int)

    # --------------------------------------------------------
    # V5 engineered features
    # --------------------------------------------------------

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
            df["distance_to_facility_km"]
            + 1
        )
    )

    df["activity_distance_signal_v5"] = (
        df["activity_density"]
        /
        (
            df["distance_to_facility_km"]
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
    # Model prediction
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

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

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
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚙️ THERMOWATCH CONTROLS"
)

# IMPORTANT:
# FIRMS endpoint is limited to 5 days.
# Therefore do not expose 7 days here.

time_options = {
    "24 Hours": 1,
    "48 Hours": 2,
    "72 Hours": 3,
    "96 Hours": 4,
    "5 Days": 5,
}

selected_window = st.sidebar.selectbox(
    "Satellite observation window",
    list(
        time_options.keys()
    ),
    index=0,
)

selected_days = time_options[
    selected_window
]

refresh = st.sidebar.button(
    "🔄 FETCH LATEST LIVE DATA",
    use_container_width=True,
)

st.sidebar.caption(
    "FIRMS observation window. "
    "The selected period is sent to NASA FIRMS."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
**CURRENT WINDOW**

`{selected_window}`

**DATA SOURCE**

NASA FIRMS VIIRS NOAA-20 NRT

**GEOSPATIAL**

OpenStreetMap

**AI**

V5 ExtraTrees Classifier
"""
)

# ------------------------------------------------------------
# Clear cache on manual refresh
# ------------------------------------------------------------

if refresh:

    st.cache_data.clear()

    st.rerun()


# ============================================================
# MAIN PIPELINE
# ============================================================

try:

    with st.spinner(
        "Loading V5 AI model..."
    ):

        model = load_model()

    with st.spinner(
        "Loading industrial facility database..."
    ):

        facilities = (
            load_osm_facilities()
        )

    with st.spinner(
        f"Fetching NASA FIRMS data for {selected_window}..."
    ):

        live = fetch_firms(
            selected_days
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
        🛰️ NO SATELLITE THERMAL OBSERVATIONS DETECTED
    </div>

    <div class="no-fire-text">
        NASA FIRMS returned no thermal observations
        inside the selected
        <b>{html.escape(selected_window)}</b>
        window over the India monitoring region.
    </div>

    <div class="no-fire-text">
        ThermoWatch is still
        <b>ONLINE</b>.
        This does not mean that no fire exists anywhere;
        it means that no FIRMS observation was returned
        for the selected monitoring window.
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.success(
        "THERMOWATCH MONITORING SYSTEM ONLINE"
    )

    n1, n2, n3 = st.columns(3)

    with n1:

        st.metric(
            "Monitoring Window",
            selected_window
        )

    with n2:

        st.metric(
            "Satellite Observations",
            0
        )

    with n3:

        st.metric(
            "Pipeline Status",
            "ONLINE"
        )

    st.markdown(
        f"""
<div class="notice">

<b>STATUS:</b> ONLINE

&nbsp; · &nbsp;

<b>WINDOW:</b> {html.escape(selected_window)}

&nbsp; · &nbsp;

<b>SOURCE:</b> NASA FIRMS VIIRS NOAA-20 NRT

&nbsp; · &nbsp;

<b>OBSERVATIONS:</b> 0

</div>
""",
        unsafe_allow_html=True,
    )

    st.info(
        "Try a wider available window such as "
        "48 Hours, 72 Hours, 96 Hours or 5 Days."
    )

    st.markdown(
        """
### 🔍 ThermoWatch status

The dashboard remains operational even when
the selected satellite window contains zero
returned observations.

No AI prediction or risk score is generated
when there is no satellite observation to analyze.
"""
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
# V5 AI
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

predictions[
    "display_facility_name"
] = predictions.apply(
    facility_label,
    axis=1
)


# ============================================================
# LATEST OBSERVATION
# ============================================================

latest_observation = None

if "acquisition_datetime" in predictions.columns:

    parsed = pd.to_datetime(
        predictions[
            "acquisition_datetime"
        ],
        errors="coerce",
        utc=True
    )

    if parsed.notna().any():

        latest_observation = (
            parsed.max()
        )


latest_text = (
    format_observed(
        latest_observation
    )
    if latest_observation is not None
    else "Unavailable"
)


# ============================================================
# PIPELINE STATUS
# ============================================================

st.success(
    f"LIVE PIPELINE COMPLETE · "
    f"{len(predictions):,} SATELLITE OBSERVATIONS ANALYZED"
)

st.markdown(
    f"""
<div class="notice">

<b>Observation window:</b>
{html.escape(selected_window)}

&nbsp; · &nbsp;

<b>Latest satellite observation:</b>
{html.escape(latest_text)}

&nbsp; · &nbsp;

<b>Source:</b>
NASA FIRMS VIIRS NOAA-20 NRT

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SUMMARY METRICS
# ============================================================

total = len(predictions)

critical_count = int(
    (
        predictions[
            "live_risk_category"
        ]
        == "CRITICAL"
    ).sum()
)

high_count = int(
    (
        predictions[
            "live_risk_category"
        ]
        == "HIGH"
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

m1, m2, m3, m4, m5 = st.columns(5)

with m1:

    st.metric(
        "LIVE HOTSPOTS",
        f"{total:,}"
    )

with m2:

    st.metric(
        "CRITICAL",
        critical_count
    )

with m3:

    st.metric(
        "HIGH RISK",
        high_count
    )

with m4:

    st.metric(
        "AVG RISK",
        f"{avg_risk:.1f}"
    )

with m5:

    st.metric(
        "MAX RISK",
        f"{max_risk:.1f}"
    )


# ============================================================
# FILTERS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 LIVE FILTERS'
    '</div>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4, f5 = st.columns(
    [1, 1, 1.2, 1.2, 2]
)

with f1:

    risk_filter = st.selectbox(
        "Risk",
        [
            "All",
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ]
    )

with f2:

    conf_filter = st.selectbox(
        "Confidence",
        [
            "All",
            "≥80%",
            "≥60%",
            "≥40%"
        ]
    )

with f3:

    source_options = [
        "All"
    ] + sorted(
        predictions[
            "predicted_source"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    source_filter = st.selectbox(
        "AI Source",
        source_options
    )

with f4:

    proximity_options = [
        "All",
        "VERY_CLOSE",
        "CLOSE",
        "NEAR",
        "DISTANT",
        "FAR",
    ]

    proximity_filter = st.selectbox(
        "Facility Proximity",
        proximity_options
    )

with f5:

    search = st.text_input(
        "Search",
        placeholder=(
            "Hotspot, facility, source, "
            "latitude, longitude..."
        )
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = predictions.copy()

if risk_filter != "All":

    filtered = filtered[
        filtered[
            "live_risk_category"
        ] == risk_filter
    ]

if conf_filter != "All":

    threshold = int(
        conf_filter
        .replace(
            "≥",
            ""
        )
        .replace(
            "%",
            ""
        )
    )

    filtered = filtered[
        filtered[
            "confidence"
        ] >= threshold
    ]

if source_filter != "All":

    filtered = filtered[
        filtered[
            "predicted_source"
        ] == source_filter
    ]

if proximity_filter != "All":

    filtered = filtered[
        filtered[
            "proximity_category"
        ] == proximity_filter
    ]

if search.strip():

    q = search.strip().lower()

    searchable = (

        filtered[
            "hotspot_id"
        ]
        .astype(str)

        + " "

        + filtered[
            "display_facility_name"
        ]
        .astype(str)

        + " "

        + filtered[
            "predicted_source"
        ]
        .astype(str)

        + " "

        + filtered[
            "latitude"
        ]
        .astype(str)

        + " "

        + filtered[
            "longitude"
        ]
        .astype(str)

    ).str.lower()

    filtered = filtered[
        searchable.str.contains(
            q,
            na=False
        )
    ]


# ============================================================
# VISIBLE STATUS
# ============================================================

st.markdown(
    f"""
<div class="notice">

Showing
<b>{len(filtered):,}</b>
of
<b>{len(predictions):,}</b>
satellite-detected hotspots.

&nbsp; · &nbsp;

Latest satellite observation:
<b>{html.escape(latest_text)}</b>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, center, right = st.columns(
    [1.0, 2.05, 1.0],
    gap="small"
)


# ============================================================
# ALERT FEED
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">'
        '⚡ ALERT FEED'
        '</div>'
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

            hotspot = html.escape(
                clean_text(
                    row[
                        "hotspot_id"
                    ],
                    "UNKNOWN"
                )
            )

            source = source_badge(
                row[
                    "predicted_source"
                ]
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

<div class="alert-id">
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
{len(filtered):,}
SATELLITE-DETECTED HOTSPOTS ·
THERMAL INTENSITY ·
AI RISK ·
FACILITY PROXIMITY
</div>
""",
        unsafe_allow_html=True,
    )

    if filtered.empty:

        st.warning(
            "No hotspots match the selected filters."
        )

    else:

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

        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

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
                            float(score)
                            / 100,
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

        # ----------------------------------------------------
        # Marker cluster
        # ----------------------------------------------------

        cluster = MarkerCluster(
            name="Satellite Hotspots",
            options={
                "maxClusterRadius": 35,
                "disableClusteringAtZoom": 8,
            }
        ).add_to(m)

        # ----------------------------------------------------
        # Markers
        # ----------------------------------------------------

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

            hotspot = html.escape(
                clean_text(
                    row[
                        "hotspot_id"
                    ],
                    "UNKNOWN"
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
                row.get(
                    "distance_to_facility_km",
                    999
                ),
                999
            )

            observed = html.escape(
                format_observed(
                    row.get(
                        "acquisition_datetime"
                    )
                )
            )

            popup_html = f"""
<div style="
font-family:Arial;
min-width:280px;
">

<b style="font-size:15px;">
{facility}
</b>

<hr>

<b>Hotspot:</b>
{hotspot}
<br>

<b>Risk:</b>
{score:.1f}/100 · {risk}
<br>

<b>FRP:</b>
{frp:.2f} MW
<br>

<b>AI Source:</b>
{source}
<br>

<b>Confidence:</b>
{confidence:.1f}%
<br>

<b>Facility distance:</b>
{distance:.3f} km
<br>

<b>Observed by satellite:</b>
{observed}

<br><br>

<span style="
font-size:10px;
color:#666;
">
ThermoWatch provides
decision-support signals based on
satellite observations and model inference.
</span>

</div>
"""

            folium.CircleMarker(
                location=[
                    float(lat),
                    float(lon)
                ],
                radius=7,
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
                    f"{score:.1f}"
                ),
            ).add_to(
                cluster
            )

        folium.LayerControl(
            collapsed=False
        ).add_to(m)

        st_folium(
            m,
            use_container_width=True,
            height=650,
            returned_objects=[],
        )


# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">'
        '📡 SYSTEM INTELLIGENCE'
        '</div>',
        unsafe_allow_html=True,
    )

    source_counts = (
        filtered[
            "predicted_source"
        ]
        .value_counts()
    )

    proximity_counts = (
        filtered[
            "proximity_category"
        ]
        .value_counts()
    )

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
Satellite observations
</div>

<div class="metric-value">
{len(filtered):,}
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
Average risk
</div>

<div class="metric-value">
{safe_float(
    filtered["live_risk_score"].mean()
):.1f}
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        "### AI Source Distribution"
    )

    if source_counts.empty:

        st.info(
            "No source data."
        )

    else:

        st.bar_chart(
            source_counts
        )

    st.markdown(
        "### Facility Proximity"
    )

    if proximity_counts.empty:

        st.info(
            "No proximity data."
        )

    else:

        st.bar_chart(
            proximity_counts
        )


# ============================================================
# TOP RISK EVENTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 HIGHEST RISK LIVE DETECTIONS'
    '</div>',
    unsafe_allow_html=True,
)

top_columns = [

    "hotspot_id",
    "display_facility_name",

    "latitude",
    "longitude",

    "acquisition_datetime",

    "frp",

    "predicted_source",

    "confidence",

    "distance_to_facility_km",

    "facility_match_quality",

    "live_risk_score",

    "live_risk_category",
]

top_columns = [
    c
    for c in top_columns
    if c in filtered.columns
]

top_risk = (
    filtered
    .sort_values(
        "live_risk_score",
        ascending=False
    )
    [top_columns]
    .head(20)
    .copy()
)


if not top_risk.empty:

    top_risk[
        "acquisition_datetime"
    ] = (
        top_risk[
            "acquisition_datetime"
        ]
        .apply(
            format_observed
        )
    )

    top_risk = top_risk.rename(
        columns={
            "hotspot_id":
                "Hotspot ID",

            "display_facility_name":
                "Facility",

            "latitude":
                "Latitude",

            "longitude":
                "Longitude",

            "acquisition_datetime":
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

    st.dataframe(
        top_risk,
        use_container_width=True,
        height=450,
        hide_index=True,
    )

else:

    st.info(
        "No events available."
    )


# ============================================================
# FULL LIVE DATA
# ============================================================

with st.expander(
    "📋 View all filtered LIVE detections"
):

    table_columns = [

        "hotspot_id",
        "display_facility_name",

        "latitude",
        "longitude",

        "acquisition_datetime",

        "frp",

        "predicted_source",

        "confidence",

        "distance_to_facility_km",

        "proximity_category",

        "live_risk_score",

        "live_risk_category",
    ]

    table_columns = [
        c
        for c in table_columns
        if c in filtered.columns
    ]

    table = filtered[
        table_columns
    ].copy()

    if "acquisition_datetime" in table.columns:

        table[
            "acquisition_datetime"
        ] = table[
            "acquisition_datetime"
        ].apply(
            format_observed
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

            "acquisition_datetime":
                "Observed At",

            "frp":
                "FRP (MW)",

            "predicted_source":
                "AI Source",

            "confidence":
                "Confidence %",

            "distance_to_facility_km":
                "Facility Distance (km)",

            "proximity_category":
                "Proximity",

            "live_risk_score":
                "Risk Score",

            "live_risk_category":
                "Risk",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        height=500,
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
    .encode(
        "utf-8"
    )
)

st.download_button(
    label="⬇️ Download LIVE ThermoWatch CSV",
    data=csv_data,
    file_name=(
        "thermowatch_"
        f"{selected_days}day_live.csv"
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

WINDOW: {html.escape(selected_window)}

&nbsp; | &nbsp;

OBSERVATIONS: {len(predictions):,}

&nbsp; | &nbsp;

VISIBLE: {len(filtered):,}

&nbsp; | &nbsp;

LATEST OBSERVATION: {html.escape(latest_text)}

&nbsp; | &nbsp;

SOURCE: NASA FIRMS VIIRS NOAA-20 NRT

&nbsp; | &nbsp;

FACILITY CONTEXT: OPENSTREETMAP

</div>
""",
    unsafe_allow_html=True,
)
