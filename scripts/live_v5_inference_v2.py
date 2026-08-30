import os
import sys
import joblib
import numpy as np
import pandas as pd

# ============================================================
# THERMO WATCH - LIVE FACILITY -> V5 AI INFERENCE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_FILE = os.path.join(
    BASE_DIR, "processed", "live_firms_hotspots.csv"
)

FACILITY_FILE = os.path.join(
    BASE_DIR, "processed", "live_facility_matches.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "source_classifier_v5.joblib"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "processed", "live_final_predictions.csv"
)

print("=" * 75)
print("THERMO WATCH - LIVE FACILITY -> V5 AI INFERENCE")
print("=" * 75)


# ============================================================
# 1. CHECK FILES
# ============================================================

for file_path, name in [
    (LIVE_FILE, "live FIRMS data"),
    (FACILITY_FILE, "facility matches"),
    (MODEL_FILE, "V5 model")
]:
    if not os.path.exists(file_path):
        print(f"\nERROR: {name} not found:")
        print(file_path)
        sys.exit(1)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading V5 model...")

model = joblib.load(MODEL_FILE)

print("V5 model loaded successfully.")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading live FIRMS data...")
live = pd.read_csv(LIVE_FILE)

print(f"Live hotspots: {len(live)}")

print("\nLoading facility matches...")
fac = pd.read_csv(FACILITY_FILE)

print(f"Facility matches: {len(fac)}")


# ============================================================
# 4. MERGE
# ============================================================

print("\nMerging live hotspot + facility information...")

# Keep facility columns that are not already in live data
facility_columns = [
    "hotspot_id",
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
    "facility_match_quality"
]

facility_columns = [
    c for c in facility_columns
    if c in fac.columns
]

fac_small = fac[facility_columns].copy()

# Remove duplicate hotspot IDs if any
fac_small = fac_small.drop_duplicates(
    subset=["hotspot_id"]
)

df = live.merge(
    fac_small,
    on="hotspot_id",
    how="left"
)

print(f"Merged rows: {len(df)}")


# ============================================================
# 5. NUMERIC CLEANING
# ============================================================

numeric_columns = [
    "frp",
    "bright_ti4",
    "bright_ti5",
    "latitude",
    "longitude",
    "distance_to_facility_km"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 6. LIVE FIRE FEATURES
# ============================================================

print("\nEngineering live fire features...")

# FIRMS current observation
df["mean_frp"] = df["frp"].fillna(0)
df["max_frp"] = df["frp"].fillna(0)

# Avoid division by zero
df["frp_ratio"] = (
    df["max_frp"] /
    df["mean_frp"].replace(0, np.nan)
).fillna(1.0)

df["mean_brightness"] = (
    df["bright_ti4"]
    if "bright_ti4" in df.columns
    else 0
)

df["max_brightness"] = df["mean_brightness"]

df["brightness_range"] = 0.0

# A LIVE FIRMS observation is only one observation
df["observation_count"] = 1

df["persistence_days"] = 0

df["observations_per_day"] = 1

df["activity_density"] = 1

df["satellite_count"] = 1

# No historical persistence claim
df["persistent_long_term"] = 0
df["persistent_180_days"] = 0


# ============================================================
# 7. FACILITY FEATURES
# ============================================================

df["distance_to_facility_km"] = (
    pd.to_numeric(
        df["distance_to_facility_km"],
        errors="coerce"
    )
)

# Do NOT use fake 999 km distance.
# If matching failed, use a conservative 20 km distance
# and mark proximity as FAR.

df["distance_to_facility_km"] = (
    df["distance_to_facility_km"]
    .fillna(20.0)
)

df["facility_within_1km"] = (
    df["distance_to_facility_km"] <= 1
).astype(int)

df["facility_within_5km"] = (
    df["distance_to_facility_km"] <= 5
).astype(int)


# ============================================================
# 8. V5 FEATURE ENGINEERING
# ============================================================

df["frp_ratio_v5"] = (
    df["max_frp"] /
    df["mean_frp"].replace(0, np.nan)
).fillna(1.0)

df["frp_excess_v5"] = np.maximum(
    df["max_frp"] - 3.0,
    0
)

df["frp_log_v5"] = np.log1p(
    df["mean_frp"].clip(lower=0)
)

df["max_frp_log_v5"] = np.log1p(
    df["max_frp"].clip(lower=0)
)

df["brightness_range_v5"] = (
    df["brightness_range"]
)

df["brightness_ratio_v5"] = (
    df["max_brightness"] /
    df["mean_brightness"].replace(0, np.nan)
).fillna(1.0)

df["brightness_excess_v5"] = np.maximum(
    df["max_brightness"] - 300,
    0
)

df["persistence_log_v5"] = np.log1p(
    df["persistence_days"].clip(lower=0)
)

df["persistence_months_v5"] = (
    df["persistence_days"] / 30
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
    (df["persistence_days"] + 1)
)

df["activity_persistence_v5"] = (
    df["observations_per_day"] *
    (1 + df["persistence_days"])
)

df["distance_log_v5"] = np.log1p(
    df["distance_to_facility_km"].clip(lower=0)
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
    (1 + df["persistence_days"])
)

df["frp_distance_signal_v5"] = (
    df["mean_frp"] /
    (1 + df["distance_to_facility_km"])
)

df["activity_distance_signal_v5"] = (
    df["activity_density"] /
    (1 + df["distance_to_facility_km"])
)


# ============================================================
# 9. EXACT V5 FEATURES
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
# 10. PREPARE MODEL INPUT
# ============================================================

print("\nPreparing V5 model input...")

X = df[FEATURES].copy()

categorical_features = [
    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse"
]

for col in categorical_features:
    X[col] = (
        X[col]
        .fillna("UNKNOWN")
        .astype(str)
    )

numeric_features = [
    c for c in FEATURES
    if c not in categorical_features
]

for col in numeric_features:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

X[numeric_features] = (
    X[numeric_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# 11. RUN V5 MODEL
# ============================================================

print("\nRunning V5 AI predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = (
    np.max(probabilities, axis=1) * 100
)

df["predicted_source"] = predictions

df["confidence"] = np.round(
    confidence,
    2
)


# ============================================================
# 12. CONFIDENCE QUALITY
# ============================================================

def confidence_level(value):

    if value >= 85:
        return "HIGH"

    elif value >= 65:
        return "MEDIUM"

    else:
        return "LOW"


df["confidence_level"] = (
    df["confidence"]
    .apply(confidence_level)
)


# ============================================================
# 13. LIVE RISK SCORE
# ============================================================

print("\nCalculating live risk...")


def calculate_risk(row):

    frp = float(row["frp"])

    distance = float(
        row["distance_to_facility_km"]
    )

    confidence = float(
        row["confidence"]
    )

    # FRP component
    frp_score = min(
        frp * 5,
        35
    )

    # Proximity component
    if distance <= 1:
        proximity_score = 30
    elif distance <= 2:
        proximity_score = 25
    elif distance <= 5:
        proximity_score = 18
    elif distance <= 10:
        proximity_score = 10
    else:
        proximity_score = 3

    # AI confidence component
    confidence_score = (
        confidence * 0.20
    )

    risk = (
        frp_score +
        proximity_score +
        confidence_score
    )

    return min(
        round(risk, 2),
        100
    )


df["live_risk_score"] = (
    df.apply(
        calculate_risk,
        axis=1
    )
)


def risk_category(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 55:
        return "HIGH"

    elif score >= 30:
        return "MEDIUM"

    else:
        return "LOW"


df["live_risk_category"] = (
    df["live_risk_score"]
    .apply(risk_category)
)


# ============================================================
# 14. ATTRIBUTION QUALITY
# ============================================================

def attribution_quality(row):

    distance = row["distance_to_facility_km"]
    confidence = row["confidence"]

    if distance <= 1 and confidence >= 80:
        return "STRONG"

    elif distance <= 5 and confidence >= 65:
        return "MODERATE"

    elif distance <= 10:
        return "WEAK"

    else:
        return "UNCONFIRMED"


df["attribution_quality"] = (
    df.apply(
        attribution_quality,
        axis=1
    )
)


# ============================================================
# 15. IMPORTANT INTERPRETATION FLAG
# ============================================================

df["source_statement"] = np.where(
    df["attribution_quality"] == "STRONG",
    "Likely source - high confidence",
    np.where(
        df["attribution_quality"] == "MODERATE",
        "Possible source - moderate confidence",
        np.where(
            df["attribution_quality"] == "WEAK",
            "Weak association - investigate",
            "Source unconfirmed"
        )
    )
)


# ============================================================
# 16. SAVE FINAL RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("LIVE V5 INFERENCE RESULTS")
print("=" * 75)

display_columns = [
    "hotspot_id",
    "latitude",
    "longitude",
    "frp",
    "facility_name",
    "facility_type",
    "distance_to_facility_km",
    "predicted_source",
    "confidence",
    "confidence_level",
    "attribution_quality",
    "live_risk_score",
    "live_risk_category",
    "source_statement"
]

display_columns = [
    c for c in display_columns
    if c in df.columns
]

print(
    df[display_columns]
    .sort_values(
        "live_risk_score",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("LIVE INFERENCE SUMMARY")
print("=" * 75)

print(f"\nTotal hotspots: {len(df)}")

print("\nPredicted source distribution:")
print(
    df["predicted_source"]
    .value_counts()
    .to_string()
)

print("\nConfidence levels:")
print(
    df["confidence_level"]
    .value_counts()
    .to_string()
)

print("\nAttribution quality:")
print(
    df["attribution_quality"]
    .value_counts()
    .to_string()
)

print("\nRisk distribution:")
print(
    df["live_risk_category"]
    .value_counts()
    .to_string()
)

print(
    f"\nAverage live risk: "
    f"{df['live_risk_score'].mean():.2f}"
)

print(
    f"Maximum live risk: "
    f"{df['live_risk_score'].max():.2f}"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 75)
print("LIVE FACILITY -> V5 INFERENCE COMPLETE")
print("=" * 75)