# ================================================================
# THERMO WATCH - LIVE HOTSPOT INFERENCE
# ================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd

# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "source_classifier_v5.joblib"
)

TRAINING_DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "live_predictions.csv"
)

# ================================================================
# LOAD MODEL
# ================================================================

print("=" * 70)
print("THERMO WATCH - LIVE HOTSPOT INFERENCE")
print("=" * 70)

print("\nLoading V5 model...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: V5 model not found!")
    print(MODEL_PATH)
    sys.exit(1)

model = joblib.load(MODEL_PATH)

print("V5 model loaded successfully.")

# ================================================================
# FEATURE ENGINEERING
# ================================================================

def create_features(df):

    df = df.copy()

    # ------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------

    numeric_columns = [
        "mean_frp",
        "max_frp",
        "mean_brightness",
        "max_brightness",
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
        "frp_ratio",
        "brightness_range"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ------------------------------------------------------------
    # FRP FEATURES
    # ------------------------------------------------------------

    if {
        "mean_frp",
        "max_frp"
    }.issubset(df.columns):

        df["frp_ratio_v5"] = (
            df["max_frp"] /
            (df["mean_frp"] + 0.01)
        )

        df["frp_excess_v5"] = (
            df["max_frp"] -
            df["mean_frp"]
        )

        df["frp_log_v5"] = np.log1p(
            df["mean_frp"].clip(lower=0)
        )

        df["max_frp_log_v5"] = np.log1p(
            df["max_frp"].clip(lower=0)
        )

    # ------------------------------------------------------------
    # BRIGHTNESS FEATURES
    # ------------------------------------------------------------

    if {
        "mean_brightness",
        "max_brightness"
    }.issubset(df.columns):

        df["brightness_range_v5"] = (
            df["max_brightness"] -
            df["mean_brightness"]
        )

        df["brightness_ratio_v5"] = (
            df["max_brightness"] /
            (df["mean_brightness"] + 0.01)
        )

        df["brightness_excess_v5"] = (
            df["max_brightness"] - 300
        )

    # ------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------

    if "persistence_days" in df.columns:

        persistence = df[
            "persistence_days"
        ].clip(lower=0)

        df["persistence_log_v5"] = np.log1p(
            persistence
        )

        df["persistence_months_v5"] = (
            persistence / 30
        )

        df["persistent_30d_v5"] = (
            persistence >= 30
        ).astype(int)

        df["persistent_90d_v5"] = (
            persistence >= 90
        ).astype(int)

        df["persistent_180d_v5"] = (
            persistence >= 180
        ).astype(int)

        df["persistent_270d_v5"] = (
            persistence >= 270
        ).astype(int)

    # ------------------------------------------------------------
    # OBSERVATIONS
    # ------------------------------------------------------------

    if "observation_count" in df.columns:

        observations = df[
            "observation_count"
        ].clip(lower=0)

        df["observation_log_v5"] = np.log1p(
            observations
        )

    if {
        "observation_count",
        "persistence_days"
    }.issubset(df.columns):

        df["obs_per_persistence_v5"] = (
            df["observation_count"] /
            (
                df["persistence_days"] + 1
            )
        )

    # ------------------------------------------------------------
    # ACTIVITY
    # ------------------------------------------------------------

    if {
        "observations_per_day",
        "persistence_days"
    }.issubset(df.columns):

        df["activity_persistence_v5"] = (
            df["observations_per_day"] *
            np.log1p(
                df["persistence_days"].clip(
                    lower=0
                )
            )
        )

    # ------------------------------------------------------------
    # DISTANCE
    # ------------------------------------------------------------

    if "distance_to_facility_km" in df.columns:

        distance = df[
            "distance_to_facility_km"
        ].clip(lower=0)

        df["distance_log_v5"] = np.log1p(
            distance
        )

        df["very_close_v5"] = (
            distance <= 1
        ).astype(int)

        df["within_2km_v5"] = (
            distance <= 2
        ).astype(int)

        df["within_5km_v5"] = (
            distance <= 5
        ).astype(int)

        df["within_10km_v5"] = (
            distance <= 10
        ).astype(int)

    # ------------------------------------------------------------
    # COMBINED SIGNALS
    # ------------------------------------------------------------

    if {
        "mean_frp",
        "persistence_days"
    }.issubset(df.columns):

        df["frp_persistence_v5"] = (
            df["mean_frp"] *
            np.log1p(
                df["persistence_days"].clip(
                    lower=0
                )
            )
        )

    if {
        "mean_frp",
        "distance_to_facility_km"
    }.issubset(df.columns):

        df["frp_distance_signal_v5"] = (
            df["mean_frp"] /
            (
                df["distance_to_facility_km"] + 1
            )
        )

    if {
        "observation_count",
        "distance_to_facility_km"
    }.issubset(df.columns):

        df["activity_distance_signal_v5"] = (
            np.log1p(
                df["observation_count"].clip(
                    lower=0
                )
            )
            /
            (
                df["distance_to_facility_km"] + 1
            )
        )

    return df


# ================================================================
# RISK CALCULATION
# ================================================================

def calculate_live_risk(row, confidence):

    score = 0.0

    # ------------------------------------------------------------
    # FRP contribution
    # ------------------------------------------------------------

    mean_frp = float(
        row.get("mean_frp", 0) or 0
    )

    max_frp = float(
        row.get("max_frp", 0) or 0
    )

    if mean_frp >= 10:
        score += 25
    elif mean_frp >= 5:
        score += 18
    elif mean_frp >= 2:
        score += 10
    elif mean_frp > 0:
        score += 5

    if max_frp >= 50:
        score += 15
    elif max_frp >= 25:
        score += 10
    elif max_frp >= 10:
        score += 5

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    persistence = float(
        row.get("persistence_days", 0) or 0
    )

    if persistence >= 180:
        score += 20
    elif persistence >= 90:
        score += 15
    elif persistence >= 30:
        score += 10
    elif persistence >= 14:
        score += 5

    # ------------------------------------------------------------
    # Facility proximity
    # ------------------------------------------------------------

    distance = float(
        row.get(
            "distance_to_facility_km",
            999
        ) or 999
    )

    if distance <= 1:
        score += 20
    elif distance <= 2:
        score += 15
    elif distance <= 5:
        score += 10
    elif distance <= 10:
        score += 5

    # ------------------------------------------------------------
    # Model confidence
    # ------------------------------------------------------------

    if confidence >= 0.90:
        score += 10
    elif confidence >= 0.75:
        score += 7
    elif confidence >= 0.60:
        score += 4

    score = min(
        round(score, 2),
        100
    )

    # ------------------------------------------------------------
    # Category
    # ------------------------------------------------------------

    if score >= 70:
        category = "CRITICAL"

    elif score >= 50:
        category = "HIGH"

    elif score >= 30:
        category = "MEDIUM"

    else:
        category = "LOW"

    return score, category


# ================================================================
# TEST WITH EXISTING HOTSPOTS
# ================================================================

print("\nLoading test hotspot...")

if not os.path.exists(TRAINING_DATA_PATH):

    print(
        "\nERROR: training_dataset.csv not found!"
    )

    sys.exit(1)

training_df = pd.read_csv(
    TRAINING_DATA_PATH
)

print(
    f"Available hotspots: "
    f"{len(training_df)}"
)

# ---------------------------------------------------------------
# Take top 10 existing hotspots for pipeline test
# ---------------------------------------------------------------

if "risk_score" in training_df.columns:

    test_df = training_df.sort_values(
        "risk_score",
        ascending=False
    ).head(10).copy()

else:

    test_df = training_df.head(
        min(10, len(training_df))
    ).copy()

print(
    f"Testing hotspots: "
    f"{len(test_df)}"
)

# ================================================================
# CREATE MODEL FEATURES
# ================================================================

print("\nPreparing live features...")

feature_df = create_features(
    test_df
)

# ================================================================
# REMOVE COLUMNS NOT USED BY V5
# ================================================================

DROP_COLUMNS = [
    "ai_label",
    "event_id",
    "osm_id",
    "facility_osm_id",

    "source_type",
    "source_confidence",
    "source_confidence_score",
    "risk_category",
    "risk_score",

    "facility_name",
    "facility_operator",

    "facility_latitude",
    "facility_longitude",

    "latitude",
    "longitude",

    "first_seen",
    "last_seen"
]

X_live = feature_df.drop(
    columns=[
        c for c in DROP_COLUMNS
        if c in feature_df.columns
    ],
    errors="ignore"
)

# ================================================================
# PREDICTION
# ================================================================

print("\nRunning V5 AI predictions...")

predictions = model.predict(
    X_live
)

probabilities = model.predict_proba(
    X_live
)

classes = model.named_steps[
    "model"
].classes_

# ================================================================
# BUILD RESULTS
# ================================================================

results = []

for i in range(
    len(test_df)
):

    row = test_df.iloc[i]

    prediction = predictions[i]

    confidence = float(
        np.max(
            probabilities[i]
        )
    )

    risk_score, risk_category = (
        calculate_live_risk(
            row,
            confidence
        )
    )

    results.append(
        {
            "event_id":
                row.get(
                    "event_id",
                    f"LIVE_{i+1}"
                ),

            "latitude":
                row.get(
                    "latitude",
                    np.nan
                ),

            "longitude":
                row.get(
                    "longitude",
                    np.nan
                ),

            "mean_frp":
                row.get(
                    "mean_frp",
                    np.nan
                ),

            "max_frp":
                row.get(
                    "max_frp",
                    np.nan
                ),

            "persistence_days":
                row.get(
                    "persistence_days",
                    np.nan
                ),

            "distance_to_facility_km":
                row.get(
                    "distance_to_facility_km",
                    np.nan
                ),

            "facility_name":
                row.get(
                    "facility_name",
                    np.nan
                ),

            "predicted_source":
                prediction,

            "confidence":
                round(
                    confidence * 100,
                    2
                ),

            "live_risk_score":
                risk_score,

            "live_risk_category":
                risk_category
        }
    )

results_df = pd.DataFrame(
    results
)

# ================================================================
# DISPLAY
# ================================================================

print("\n" + "=" * 70)
print("LIVE INFERENCE RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

# ================================================================
# SAVE
# ================================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("LIVE HOTSPOT INFERENCE COMPLETE")
print("=" * 70)

print(
    f"\nResults saved:"
)

print(
    OUTPUT_PATH
)

print(
    "\nV5 AI source prediction: READY"
)

print(
    "Next step: connect this pipeline "
    "to satellite hotspot data."
)

print(
    "=" * 70
)