import os
import joblib
import numpy as np
import pandas as pd


# ======================================================================
# THERMO WATCH - LIVE FIRMS -> V5 AI INFERENCE
# ======================================================================

print("=" * 72)
print("THERMO WATCH - LIVE FIRMS -> V5 AI INFERENCE")
print("=" * 72)


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "source_classifier_v5.joblib"
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "live_firms_hotspots.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "live_v5_predictions.csv"
)


# ======================================================================
# LOAD MODEL
# ======================================================================

print("\nLoading V5 model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: V5 model not found:")
    print(MODEL_PATH)
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

print("V5 model loaded successfully.")


# ======================================================================
# LOAD LIVE FIRMS DATA
# ======================================================================

print("\nLoading live FIRMS data...")

if not os.path.exists(INPUT_PATH):
    print("ERROR: Live FIRMS file not found:")
    print(INPUT_PATH)
    raise SystemExit(1)

df = pd.read_csv(INPUT_PATH)

print("Live hotspots loaded:", len(df))


if len(df) == 0:
    print("\nNo live hotspots available.")
    raise SystemExit(0)


# ======================================================================
# SAFE NUMERIC CONVERSION
# ======================================================================

numeric_columns = [
    "latitude",
    "longitude",
    "frp",
    "scan",
    "track"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

print("\nEngineering live features...")


# ----------------------------------------------------------------------
# FRP FEATURES
# ----------------------------------------------------------------------

df["mean_frp"] = df["frp"].fillna(0)

df["max_frp"] = df["frp"].fillna(0)

df["frp_ratio"] = (
    df["max_frp"] /
    (df["mean_frp"] + 1e-6)
)

df["mean_brightness"] = 0.0

df["max_brightness"] = 0.0

df["brightness_range"] = 0.0


# ----------------------------------------------------------------------
# OBSERVATION FEATURES
# ----------------------------------------------------------------------

df["observation_count"] = 1

df["persistence_days"] = 1

df["observations_per_day"] = 1.0

df["activity_density"] = 1.0

df["satellite_count"] = 1


# ----------------------------------------------------------------------
# FACILITY FEATURES
# ----------------------------------------------------------------------

# Live FIRMS data alone does not contain facility information.
# Use neutral defaults for the first live inference stage.

df["distance_to_facility_km"] = 999.0

df["facility_within_1km"] = 0

df["facility_within_5km"] = 0

df["persistent_long_term"] = 0

df["persistent_180_days"] = 0

df["facility_type"] = "unknown"

df["facility_power"] = "unknown"

df["facility_industrial"] = "unknown"

df["facility_landuse"] = "unknown"


# ======================================================================
# V5 ENGINEERED FEATURES
# ======================================================================

df["frp_ratio_v5"] = (
    df["max_frp"] /
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

df["brightness_range_v5"] = 0.0

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
    df["observation_count"] /
    (df["persistence_days"] + 1e-6)
)

df["activity_persistence_v5"] = (
    df["activity_density"] *
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
    df["mean_frp"] *
    df["persistence_days"]
)

df["frp_distance_signal_v5"] = (
    df["mean_frp"] /
    (df["distance_to_facility_km"] + 1)
)

df["activity_distance_signal_v5"] = (
    df["activity_density"] /
    (df["distance_to_facility_km"] + 1)
)


# ======================================================================
# EXACT V5 FEATURE LIST
# ======================================================================

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


# ======================================================================
# CHECK FEATURES
# ======================================================================

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    raise SystemExit(1)


X = df[FEATURES].copy()


# ======================================================================
# CLEAN NUMERIC FEATURES
# ======================================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns

X[numeric_features] = X[numeric_features].replace(
    [np.inf, -np.inf],
    np.nan
)

X[numeric_features] = X[numeric_features].fillna(0)


# ======================================================================
# CLEAN CATEGORICAL FEATURES
# ======================================================================

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


# ======================================================================
# PREDICTION
# ======================================================================

print("\nRunning V5 AI predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = (
    np.max(probabilities, axis=1) * 100
)


# ======================================================================
# RISK SCORE
# ======================================================================

print("\nCalculating live risk...")


frp_component = np.clip(
    df["mean_frp"].fillna(0) * 3,
    0,
    35
)

max_frp_component = np.clip(
    df["max_frp"].fillna(0) * 0.5,
    0,
    20
)

confidence_component = (
    confidence * 0.20
)

risk_score = (
    frp_component +
    max_frp_component +
    confidence_component
)

risk_score = np.clip(
    risk_score,
    0,
    100
)

df["live_risk_score"] = (
    risk_score.round(2)
)


# ======================================================================
# RISK CATEGORY
# ======================================================================

def get_risk_category(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


df["live_risk_category"] = (
    df["live_risk_score"]
    .apply(get_risk_category)
)


# ======================================================================
# AI OUTPUT
# ======================================================================

df["predicted_source"] = predictions

df["confidence"] = confidence.round(2)


# ======================================================================
# SOURCE EXPLANATION
# ======================================================================

def explain_source(row):

    source = row["predicted_source"]

    frp = row["mean_frp"]

    if source == "POWER_GENERATION":
        return (
            "AI associates hotspot with "
            "power-generation characteristics."
        )

    if source == "POWER_SUBSTATION":
        return (
            "AI associates hotspot with "
            "power-substation characteristics."
        )

    if source == "INDUSTRIAL":
        return (
            "AI associates hotspot with "
            "industrial characteristics."
        )

    return "Source uncertain."


df["source_explanation"] = df.apply(
    explain_source,
    axis=1
)


# ======================================================================
# FINAL COLUMNS
# ======================================================================

preferred_columns = [
    "hotspot_id",
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "acquisition_datetime",
    "satellite",
    "instrument",
    "confidence",
    "frp",
    "mean_frp",
    "max_frp",
    "predicted_source",
    "confidence",
    "live_risk_score",
    "live_risk_category",
    "source_explanation",
    "fetched_at"
]


# Remove duplicate column names while preserving order
final_columns = []

for col in preferred_columns:

    if col in df.columns and col not in final_columns:
        final_columns.append(col)


result = df[final_columns].copy()


# ======================================================================
# SAVE
# ======================================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ======================================================================
# DISPLAY
# ======================================================================

print("\n" + "=" * 72)
print("LIVE V5 INFERENCE RESULTS")
print("=" * 72)

print(
    result[
        [
            "hotspot_id",
            "latitude",
            "longitude",
            "frp",
            "predicted_source",
            "confidence",
            "live_risk_score",
            "live_risk_category"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LIVE INFERENCE SUMMARY")
print("=" * 72)

print("\nTotal hotspots:", len(result))

print("\nPredicted source distribution:")
print(
    result["predicted_source"]
    .value_counts()
    .to_string()
)

print("\nRisk distribution:")
print(
    result["live_risk_category"]
    .value_counts()
    .to_string()
)

print(
    "\nAverage live risk:",
    round(result["live_risk_score"].mean(), 2)
)

print(
    "Maximum live risk:",
    round(result["live_risk_score"].max(), 2)
)


# ======================================================================
# OUTPUT
# ======================================================================

print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 72)
print("LIVE V5 AI INFERENCE COMPLETE")
print("=" * 72)