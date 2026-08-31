import io
import math
from datetime import datetime, timezone, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.neighbors import BallTree

# ============================================================
# THERMOWATCH AI
# NASA FIRMS -> OSM FACILITIES -> V5 AI -> RISK ASSESSMENT
# Streamlit-native UI: NO custom HTML/CSS
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI",
    page_icon="🔥",
    layout="wide",
)

# ============================================================
# CONFIG
# ============================================================

MODEL_URL = (
    "https://huggingface.co/harshitcodes1544/sihharshit/"
    "resolve/main/source_classifier_v5.joblib"
)

OSM_URL = (
    "https://huggingface.co/datasets/harshitcodes1544/"
    "thermowatch-osm-data/resolve/main/osm_facilities.csv"
)

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
INDIA_BBOX = "68.0,6.0,97.5,37.5"

EARTH_RADIUS_KM = 6371.0088

# Exact feature names expected by the V5 pipeline.
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

CATEGORICAL_FEATURES = [
    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse",
]

# ============================================================
# HEADER
# ============================================================

st.title("🔥 ThermoWatch AI")
st.subheader("LIVE Satellite Fire Detection & Industrial Risk Intelligence")
st.caption("NASA FIRMS → OSM Facilities → V5 AI → Risk Assessment")

st.success("● LIVE SATELLITE THERMAL MONITORING")

# ============================================================
# HELPERS
# ============================================================

def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model():
    response = requests.get(MODEL_URL, timeout=180)
    response.raise_for_status()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(response.content)
        path = f.name

    return joblib.load(path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_osm_facilities():
    response = requests.get(OSM_URL, timeout=180)
    response.raise_for_status()

    facilities = pd.read_csv(io.BytesIO(response.content))

    required = {"latitude", "longitude"}
    missing = required - set(facilities.columns)

    if missing:
        raise RuntimeError(
            "OSM facility file is missing columns: "
            + ", ".join(sorted(missing))
        )

    facilities["latitude"] = pd.to_numeric(
        facilities["latitude"], errors="coerce"
    )
    facilities["longitude"] = pd.to_numeric(
        facilities["longitude"], errors="coerce"
    )

    facilities = facilities.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    return facilities


def parse_firms_datetime(df):
    # FIRMS commonly supplies acq_date + acq_time.
    if "acq_date" in df.columns:
        dates = df["acq_date"].astype(str)

        if "acq_time" in df.columns:
            times = (
                pd.to_numeric(df["acq_time"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                .str.zfill(4)
            )
            stamp = dates + " " + times.str[:2] + ":" + times.str[2:] + ":00"
            return pd.to_datetime(stamp, errors="coerce", utc=True)

        return pd.to_datetime(dates, errors="coerce", utc=True)

    if "observed_at" in df.columns:
        return pd.to_datetime(
            df["observed_at"], errors="coerce", utc=True
        )

    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def fetch_firms(hours):
    api_key = get_secret("FIRMS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "FIRMS_API_KEY is missing from Streamlit Secrets."
        )

    # FIRMS area endpoint accepts whole-day windows.
    # We request up to 5 days and then apply the exact selected
    # hour window locally.
    api_days = max(1, min(5, int(math.ceil(hours / 24))))

    url = (
        FIRMS_BASE_URL
        + str(api_key)
        + "/VIIRS_NOAA20_NRT/"
        + INDIA_BBOX
        + f"/{api_days}"
    )

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    if not response.text.strip():
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(response.text))

    if df.empty:
        return df

    # Normalize coordinates and FRP.
    for col in ["latitude", "longitude", "frp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["latitude", "longitude", "frp"]
    ).copy()

    # Exact selected-hour filtering.
    df["observed_at"] = parse_firms_datetime(df)

    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff = now_utc - pd.Timedelta(hours=hours)

    df = df[
        df["observed_at"].notna()
        & (df["observed_at"] >= cutoff)
        & (df["observed_at"] <= now_utc + pd.Timedelta(hours=1))
    ].copy()

    if df.empty:
        return df

    # Stable IDs for this live run.
    df = df.reset_index(drop=True)
    df["hotspot_id"] = [
        f"LIVE_{i:06d}" for i in range(1, len(df) + 1)
    ]

    return df


def nearest_facilities(live, facilities):
    if live.empty:
        return live.copy()

    facility_coords = np.radians(
        facilities[["latitude", "longitude"]].values
    )
    live_coords = np.radians(
        live[["latitude", "longitude"]].values
    )

    tree = BallTree(
        facility_coords,
        metric="haversine"
    )

    distances, indices = tree.query(
        live_coords,
        k=1
    )

    distances_km = distances[:, 0] * EARTH_RADIUS_KM
    nearest_indices = indices[:, 0]

    nearest = facilities.iloc[
        nearest_indices
    ].reset_index(drop=True)

    result = live.reset_index(drop=True).copy()

    def facility_col(name):
        if name in nearest.columns:
            return nearest[name].values
        return np.full(len(nearest), np.nan, dtype=object)

    result["facility_osm_id"] = facility_col("osm_id")
    result["facility_type"] = facility_col("feature_type")
    result["facility_name"] = facility_col("name")
    result["facility_power"] = facility_col("power")
    result["facility_industrial"] = facility_col("industrial")
    result["facility_landuse"] = facility_col("landuse")
    result["facility_operator"] = facility_col("operator")
    result["facility_plant_source"] = facility_col("plant_source")
    result["facility_plant_method"] = facility_col("plant_method")

    result["facility_latitude"] = nearest["latitude"].values
    result["facility_longitude"] = nearest["longitude"].values
    result["distance_to_facility_km"] = distances_km

    result["proximity_category"] = np.select(
        [
            result["distance_to_facility_km"] <= 1,
            result["distance_to_facility_km"] <= 2,
            result["distance_to_facility_km"] <= 5,
            result["distance_to_facility_km"] <= 10,
        ],
        [
            "VERY_CLOSE",
            "CLOSE",
            "NEAR",
            "DISTANT",
        ],
        default="FAR",
    )

    result["facility_match_quality"] = np.select(
        [
            result["distance_to_facility_km"] <= 1,
            result["distance_to_facility_km"] <= 5,
            result["distance_to_facility_km"] <= 10,
        ],
        [
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        default="VERY_LOW",
    )

    return result


def engineer_features(df):
    out = df.copy()

    out["mean_frp"] = out["frp"].fillna(0)
    out["max_frp"] = out["frp"].fillna(0)

    out["frp_ratio"] = (
        out["max_frp"] / (out["mean_frp"] + 1e-6)
    )

    # Live FIRMS records may not contain brightness.
    out["mean_brightness"] = pd.to_numeric(
        out.get("bright_ti4", 0), errors="coerce"
    ).fillna(0) if isinstance(
        out.get("bright_ti4", 0), pd.Series
    ) else 0.0

    out["max_brightness"] = out["mean_brightness"]
    out["brightness_range"] = 0.0

    # Per-hotspot live observation approximation.
    out["observation_count"] = 1
    out["persistence_days"] = 1
    out["observations_per_day"] = 1.0
    out["activity_density"] = 1.0
    out["satellite_count"] = 1

    # Facility information is already attached.
    out["distance_to_facility_km"] = pd.to_numeric(
        out["distance_to_facility_km"], errors="coerce"
    ).fillna(999.0)

    out["facility_within_1km"] = (
        out["distance_to_facility_km"] <= 1
    ).astype(int)

    out["facility_within_5km"] = (
        out["distance_to_facility_km"] <= 5
    ).astype(int)

    out["persistent_long_term"] = 0
    out["persistent_180_days"] = 0

    for col in CATEGORICAL_FEATURES:
        out[col] = (
            out[col]
            .fillna("unknown")
            .astype(str)
        )

    # V5 engineered features.
    out["frp_ratio_v5"] = (
        out["max_frp"] / (out["mean_frp"] + 1e-6)
    )

    out["frp_excess_v5"] = np.maximum(
        out["max_frp"] - 5.0, 0
    )

    out["frp_log_v5"] = np.log1p(out["mean_frp"])
    out["max_frp_log_v5"] = np.log1p(out["max_frp"])

    out["brightness_range_v5"] = 0.0
    out["brightness_ratio_v5"] = 1.0
    out["brightness_excess_v5"] = 0.0

    out["persistence_log_v5"] = np.log1p(
        out["persistence_days"]
    )

    out["persistent_180d_v5"] = (
        out["persistence_days"] >= 180
    ).astype(int)

    out["persistent_270d_v5"] = (
        out["persistence_days"] >= 270
    ).astype(int)

    out["observation_log_v5"] = np.log1p(
        out["observation_count"]
    )

    out["obs_per_persistence_v5"] = (
        out["observation_count"]
        / (out["persistence_days"] + 1e-6)
    )

    out["activity_persistence_v5"] = (
        out["activity_density"]
        * out["persistence_days"]
    )

    out["distance_log_v5"] = np.log1p(
        out["distance_to_facility_km"]
    )

    out["very_close_v5"] = (
        out["distance_to_facility_km"] <= 1
    ).astype(int)

    out["within_2km_v5"] = (
        out["distance_to_facility_km"] <= 2
    ).astype(int)

    out["within_5km_v5"] = (
        out["distance_to_facility_km"] <= 5
    ).astype(int)

    out["within_10km_v5"] = (
        out["distance_to_facility_km"] <= 10
    ).astype(int)

    out["frp_persistence_v5"] = (
        out["mean_frp"]
        * out["persistence_days"]
    )

    out["frp_distance_signal_v5"] = (
        out["mean_frp"]
        / (out["distance_to_facility_km"] + 1)
    )

    out["activity_distance_signal_v5"] = (
        out["activity_density"]
        / (out["distance_to_facility_km"] + 1)
    )

    # Guarantee every model column exists.
    for col in FEATURES:
        if col not in out.columns:
            if col in CATEGORICAL_FEATURES:
                out[col] = "unknown"
            else:
                out[col] = 0.0

    X = out[FEATURES].copy()

    numeric = [
        c for c in FEATURES
        if c not in CATEGORICAL_FEATURES
    ]

    for col in numeric:
        X[col] = pd.to_numeric(
            X[col], errors="coerce"
        )

    X[numeric] = X[numeric].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    for col in CATEGORICAL_FEATURES:
        X[col] = (
            X[col]
            .fillna("unknown")
            .astype(str)
        )

    return out, X


def add_predictions(df, model, X):
    result = df.copy()

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result["predicted_source"] = predictions
    result["confidence"] = (
        np.max(probabilities, axis=1) * 100
    ).round(1)

    # Live risk model.
    frp_component = np.clip(
        result["mean_frp"].fillna(0) * 3,
        0,
        35
    )

    max_frp_component = np.clip(
        result["max_frp"].fillna(0) * 0.5,
        0,
        20
    )

    confidence_component = (
        result["confidence"] * 0.20
    )

    risk = (
        frp_component
        + max_frp_component
        + confidence_component
    )

    result["live_risk_score"] = np.clip(
        risk,
        0,
        100
    ).round(2)

    result["live_risk_category"] = pd.cut(
        result["live_risk_score"],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right=False,
    ).astype(str)

    return result


def risk_icon(category):
    return {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }.get(str(category), "⚪")


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Live Controls")

hours = st.sidebar.selectbox(
    "Monitoring window",
    [24, 48, 72, 96, 120],
    index=0,
    format_func=lambda x: (
        f"{x} Hours" if x < 120 else "5 Days (120 Hours)"
    ),
)

run_pipeline = st.sidebar.button(
    "🚀 FETCH LIVE FIRE DATA",
    type="primary",
    use_container_width=True,
)

st.sidebar.info(
    "FIRMS supplies whole-day data. "
    "ThermoWatch requests up to 5 days and then "
    "filters observations to the exact selected hour window."
)

# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "last_hours" not in st.session_state:
    st.session_state.last_hours = hours

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None

# ============================================================
# PIPELINE
# ============================================================

if run_pipeline:
    st.session_state.pipeline_error = None
    st.session_state.results = None
    st.session_state.last_hours = hours

    progress = st.progress(0, text="Starting LIVE pipeline...")

    try:
        progress.progress(10, text="Loading V5 AI model...")
        model = load_model()

        progress.progress(25, text="Fetching LIVE NASA FIRMS observations...")
        live = fetch_firms(hours)

        # IMPORTANT:
        # Zero observations is a valid state, NOT a pipeline failure.
        if live.empty:
            st.session_state.results = pd.DataFrame()
            progress.progress(100, text="Pipeline ONLINE — no observations found.")
            st.rerun()

        progress.progress(
            45,
            text=f"Loading OSM facilities for {len(live)} hotspots..."
        )
        facilities = load_osm_facilities()

        progress.progress(
            60,
            text="Finding nearest industrial/facility locations..."
        )
        enriched = nearest_facilities(
            live,
            facilities
        )

        progress.progress(
            75,
            text="Engineering V5 AI features..."
        )
        engineered, X = engineer_features(enriched)

        progress.progress(
            90,
            text="Running V5 source classification and risk assessment..."
        )
        final = add_predictions(
            engineered,
            model,
            X
        )

        st.session_state.results = final

        progress.progress(
            100,
            text="LIVE ThermoWatch pipeline complete."
        )

    except Exception as exc:
        st.session_state.pipeline_error = str(exc)
        progress.empty()

# ============================================================
# ERROR
# ============================================================

if st.session_state.pipeline_error:
    st.error("ThermoWatch LIVE pipeline failed.")
    st.exception(RuntimeError(st.session_state.pipeline_error))

# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results

if results is None:
    st.info(
        "Select a monitoring window and click "
        "'FETCH LIVE FIRE DATA' to start LIVE satellite analysis."
    )

else:
    # --------------------------------------------------------
    # ZERO-FIRE STATE
    # --------------------------------------------------------

    if results.empty:
        st.warning(
            "🛰️ NO SATELLITE THERMAL OBSERVATIONS DETECTED"
        )

        st.write(
            f"NASA FIRMS returned no thermal observations "
            f"inside the selected {st.session_state.last_hours} Hours "
            "window over the India monitoring region."
        )

        st.success(
            "THERMOWATCH MONITORING SYSTEM ONLINE"
        )

        a, b, c = st.columns(3)
        a.metric(
            "Monitoring Window",
            f"{st.session_state.last_hours} Hours"
        )
        b.metric(
            "Satellite Observations",
            0
        )
        c.metric(
            "Pipeline Status",
            "ONLINE"
        )

        st.caption(
            "This does not prove that no fire exists anywhere. "
            "It means NASA FIRMS returned no observations for "
            "the selected monitoring window."
        )

    # --------------------------------------------------------
    # LIVE FIRE RESULTS
    # --------------------------------------------------------

    else:
        st.success(
            f"THERMOWATCH MONITORING SYSTEM ONLINE — "
            f"{len(results)} LIVE SATELLITE OBSERVATIONS"
        )

        # Metrics.
        total = len(results)
        high = int(
            results["live_risk_category"]
            .isin(["HIGH", "CRITICAL"])
            .sum()
        )
        medium = int(
            (results["live_risk_category"] == "MEDIUM").sum()
        )
        avg_risk = float(
            results["live_risk_score"].mean()
        )
        max_risk = float(
            results["live_risk_score"].max()
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("LIVE Hotspots", total)
        c2.metric("HIGH / CRITICAL", high)
        c3.metric("MEDIUM", medium)
        c4.metric("Average Risk", f"{avg_risk:.1f}")
        c5.metric("Maximum Risk", f"{max_risk:.1f}")

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        st.subheader("🔎 LIVE Detection Filters")

        f1, f2, f3 = st.columns(3)

        risk_options = [
            "ALL",
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
        ]

        source_values = ["ALL"] + sorted(
            results["predicted_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        risk_filter = f1.selectbox(
            "Risk category",
            risk_options,
        )

        source_filter = f2.selectbox(
            "AI predicted source",
            source_values,
        )

        max_distance = f3.slider(
            "Maximum facility distance (km)",
            min_value=1,
            max_value=1000,
            value=100,
        )

        filtered = results.copy()

        if risk_filter != "ALL":
            filtered = filtered[
                filtered["live_risk_category"]
                == risk_filter
            ]

        if source_filter != "ALL":
            filtered = filtered[
                filtered["predicted_source"].astype(str)
                == source_filter
            ]

        filtered = filtered[
            filtered["distance_to_facility_km"]
            <= max_distance
        ].copy()

        st.caption(
            f"Showing {len(filtered)} of {len(results)} live observations."
        )

        # ----------------------------------------------------
        # INDIA MAP
        # ----------------------------------------------------

        st.subheader("🗺️ LIVE India Thermal Zones")

        map_df = filtered[
            ["latitude", "longitude", "live_risk_score"]
        ].copy()

        if not map_df.empty:
            # Native Streamlit map. No HTML/Folium required.
            st.map(
                map_df,
                latitude="latitude",
                longitude="longitude",
                size="live_risk_score",
                zoom=4,
                height=620,
            )

        else:
            st.info(
                "No hotspots match the current filters."
            )

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("📊 LIVE Risk Distribution")

        risk_counts = (
            results["live_risk_category"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                fill_value=0
            )
        )

        st.bar_chart(risk_counts)

        # ----------------------------------------------------
        # SOURCE DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("🤖 AI Source Distribution")

        source_counts = (
            results["predicted_source"]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(source_counts)

        # ----------------------------------------------------
        # TOP DETECTIONS
        # ----------------------------------------------------

        st.subheader("🔥 Highest-Risk LIVE Detections")

        top = (
            filtered
            .sort_values(
                "live_risk_score",
                ascending=False
            )
            .head(20)
            .copy()
        )

        display_cols = [
            "hotspot_id",
            "facility_name",
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
            c for c in display_cols
            if c in top.columns
        ]

        st.dataframe(
            top[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # FACILITY SUMMARY
        # ----------------------------------------------------

        st.subheader("🏭 Nearest Facility Intelligence")

        fc1, fc2, fc3 = st.columns(3)

        fc1.metric(
            "Very Close Matches",
            int(
                (
                    results["proximity_category"]
                    == "VERY_CLOSE"
                ).sum()
            ),
        )

        fc2.metric(
            "High-Quality Matches",
            int(
                (
                    results["facility_match_quality"]
                    == "HIGH"
                ).sum()
            ),
        )

        fc3.metric(
            "Facilities ≤ 5 km",
            int(
                (
                    results["distance_to_facility_km"]
                    <= 5
                ).sum()
            ),
        )

        facility_types = (
            results["facility_type"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .head(15)
        )

        st.write("Nearest facility types")
        st.dataframe(
            facility_types.rename("detections"),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # SELECTED DETECTION DETAILS
        # ----------------------------------------------------

        st.subheader("📍 Detection Details")

        if not filtered.empty:
            selected_id = st.selectbox(
                "Select a hotspot",
                filtered["hotspot_id"].tolist(),
            )

            selected = filtered[
                filtered["hotspot_id"] == selected_id
            ].iloc[0]

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "Risk",
                f"{selected['live_risk_score']:.2f}"
            )

            d2.metric(
                "Category",
                str(selected["live_risk_category"])
            )

            d3.metric(
                "AI Confidence",
                f"{selected['confidence']:.1f}%"
            )

            d4.metric(
                "FRP",
                f"{selected['frp']:.2f}"
            )

            st.write(
                f"**AI Source:** {selected['predicted_source']}"
            )
            st.write(
                f"**Facility:** "
                f"{selected.get('facility_name', 'Unknown')}"
            )
            st.write(
                f"**Facility Type:** "
                f"{selected.get('facility_type', 'Unknown')}"
            )
            st.write(
                f"**Facility Distance:** "
                f"{selected['distance_to_facility_km']:.3f} km"
            )
            st.write(
                f"**Observed At:** "
                f"{selected['observed_at']}"
            )

        # ----------------------------------------------------
        # COMPLETE DATA
        # ----------------------------------------------------

        st.subheader("📄 Complete LIVE Dataset")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download LIVE Predictions CSV",
            data=csv_bytes,
            file_name=(
                "thermowatch_live_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # PIPELINE STATUS
        # ----------------------------------------------------

        st.divider()

        st.success(
            "FIRMS HOTSPOTS → READY | "
            "OSM FACILITIES → READY | "
            "V5 AI → READY | "
            "RISK ASSESSMENT → READY"
        )

        st.caption(
            "ThermoWatch uses satellite thermal observations as "
            "detections/signals. A detection is not by itself "
            "proof of a confirmed ground fire."
        )
