import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("THERMO WATCH - RISK FEATURE ENGINEERING")
print("=" * 60)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "processed" / "hotspot_source_attribution.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "risk_features.csv"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading source attribution data...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")

# ---------------------------------------------------------
# DATE FEATURES
# ---------------------------------------------------------

print("Creating temporal features...")

df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")

df["active_days"] = (
    df["last_seen"] - df["first_seen"]
).dt.days + 1

df["active_days"] = df["active_days"].clip(lower=1)

# ---------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------

numeric_columns = [
    "mean_frp",
    "max_frp",
    "mean_brightness",
    "max_brightness",
    "observation_count",
    "persistence_days",
    "observations_per_day",
    "satellite_count",
    "distance_to_facility_km",
    "source_confidence_score",
    "latitude",
    "longitude",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

print("Engineering risk features...")

# FRP intensity
df["frp_intensity"] = (
    df["mean_frp"] * 0.6 +
    df["max_frp"] * 0.4
)

# Persistence score
df["persistence_ratio"] = (
    df["persistence_days"] /
    df["active_days"]
)

df["persistence_ratio"] = df["persistence_ratio"].clip(0, 1)

# Observation density
df["observation_density"] = (
    df["observation_count"] /
    df["active_days"]
)

# Thermal intensity
df["thermal_intensity"] = (
    df["mean_brightness"] * 0.5 +
    df["max_brightness"] * 0.5
)

# Facility proximity score
df["proximity_score"] = 1 / (
    1 + df["distance_to_facility_km"]
)

# Source confidence normalized
df["source_confidence_normalized"] = (
    df["source_confidence_score"] / 100
)

# Satellite support
df["satellite_support"] = df["satellite_count"].clip(0, 10)

# ---------------------------------------------------------
# RISK SCORE
# ---------------------------------------------------------

print("Calculating preliminary risk score...")

def normalize(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


df["frp_norm"] = normalize(df["frp_intensity"])

df["persistence_norm"] = normalize(
    df["persistence_days"]
)

df["observation_norm"] = normalize(
    df["observation_density"]
)

df["brightness_norm"] = normalize(
    df["thermal_intensity"]
)

df["proximity_norm"] = normalize(
    df["proximity_score"]
)

# Weighted risk score
df["risk_score"] = (
    0.25 * df["frp_norm"] +
    0.20 * df["persistence_norm"] +
    0.15 * df["observation_norm"] +
    0.15 * df["brightness_norm"] +
    0.15 * df["proximity_norm"] +
    0.10 * df["source_confidence_normalized"]
) * 100

df["risk_score"] = df["risk_score"].clip(0, 100)

# ---------------------------------------------------------
# RISK CATEGORY
# ---------------------------------------------------------

def risk_category(score):

    if score >= 80:
        return "CRITICAL"

    elif score >= 60:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_category"] = df["risk_score"].apply(
    risk_category
)

# ---------------------------------------------------------
# SELECT FINAL FEATURES
# ---------------------------------------------------------

feature_columns = [
    "event_id",
    "latitude",
    "longitude",
    "first_seen",
    "last_seen",

    "observation_count",
    "mean_frp",
    "max_frp",
    "mean_brightness",
    "max_brightness",

    "persistence_days",
    "observations_per_day",
    "satellite_count",

    "distance_to_facility_km",
    "source_type",
    "source_confidence_score",
    "source_confidence",

    "active_days",
    "frp_intensity",
    "persistence_ratio",
    "observation_density",
    "thermal_intensity",
    "proximity_score",
    "source_confidence_normalized",
    "satellite_support",

    "risk_score",
    "risk_category",
]

feature_columns = [
    col for col in feature_columns
    if col in df.columns
]

result = df[feature_columns].copy()

# ---------------------------------------------------------
# SORT BY RISK
# ---------------------------------------------------------

result = result.sort_values(
    "risk_score",
    ascending=False
).reset_index(drop=True)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RISK FEATURES CREATED")
print("=" * 60)

print(f"Total hotspots : {len(result)}")
print(f"Features       : {len(result.columns)}")

print("\nRISK DISTRIBUTION:")
print(
    result["risk_category"]
    .value_counts()
    .sort_index()
)

print("\nTOP 10 HIGH-RISK HOTSPOTS:")

display_columns = [
    "event_id",
    "latitude",
    "longitude",
    "mean_frp",
    "max_frp",
    "persistence_days",
    "distance_to_facility_km",
    "source_type",
    "source_confidence",
    "risk_score",
    "risk_category",
]

display_columns = [
    c for c in display_columns
    if c in result.columns
]

print(
    result[display_columns]
    .head(10)
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)