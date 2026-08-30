import pandas as pd
import os

print("=" * 60)
print("THERMO WATCH - AI TRAINING DATASET")
print("=" * 60)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RISK_FILE = os.path.join(
    BASE_DIR, "processed", "risk_features.csv"
)

ATTRIBUTION_FILE = os.path.join(
    BASE_DIR, "processed", "hotspot_source_attribution.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "processed", "training_dataset.csv"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading risk features...")
risk = pd.read_csv(RISK_FILE)

print("Risk rows:", len(risk))

print("\nLoading source attribution...")
attr = pd.read_csv(ATTRIBUTION_FILE)

print("Attribution rows:", len(attr))

# ---------------------------------------------------------
# SELECT IMPORTANT COLUMNS
# ---------------------------------------------------------

risk_cols = [
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
    "proximity",
    "source_type",
    "source_confidence_score",
    "source_confidence",
    "risk_score",
    "risk_category"
]

risk_cols = [
    c for c in risk_cols
    if c in risk.columns
]

df = risk[risk_cols].copy()

# ---------------------------------------------------------
# ADD FACILITY INFORMATION
# ---------------------------------------------------------

facility_cols = [
    "event_id",
    "facility_osm_id",
    "facility_name",
    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse",
    "facility_operator",
    "facility_latitude",
    "facility_longitude"
]

facility_cols = [
    c for c in facility_cols
    if c in attr.columns
]

facility = attr[facility_cols].copy()

# Remove duplicate event IDs if any
facility = facility.drop_duplicates(
    subset=["event_id"]
)

# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

print("\nMerging datasets...")

df = df.merge(
    facility,
    on="event_id",
    how="left"
)

# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

print("\nCleaning data...")

# Numeric columns
numeric_cols = [
    "latitude",
    "longitude",
    "observation_count",
    "mean_frp",
    "max_frp",
    "mean_brightness",
    "max_brightness",
    "persistence_days",
    "observations_per_day",
    "satellite_count",
    "distance_to_facility_km",
    "source_confidence_score",
    "risk_score",
    "facility_latitude",
    "facility_longitude"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ---------------------------------------------------------
# TEMPORAL FEATURES
# ---------------------------------------------------------

print("Creating temporal features...")

df["first_seen"] = pd.to_datetime(
    df["first_seen"],
    errors="coerce"
)

df["last_seen"] = pd.to_datetime(
    df["last_seen"],
    errors="coerce"
)

df["start_month"] = df["first_seen"].dt.month
df["start_dayofyear"] = df["first_seen"].dt.dayofyear

# ---------------------------------------------------------
# FIRE INTENSITY FEATURES
# ---------------------------------------------------------

print("Creating fire intensity features...")

df["frp_ratio"] = (
    df["max_frp"] /
    df["mean_frp"].replace(0, pd.NA)
)

df["brightness_range"] = (
    df["max_brightness"] -
    df["mean_brightness"]
)

df["activity_density"] = (
    df["observation_count"] /
    df["persistence_days"].replace(0, pd.NA)
)

# ---------------------------------------------------------
# PROXIMITY FEATURES
# ---------------------------------------------------------

print("Creating proximity features...")

df["facility_within_1km"] = (
    df["distance_to_facility_km"] <= 1
).astype(int)

df["facility_within_5km"] = (
    df["distance_to_facility_km"] <= 5
).astype(int)

# ---------------------------------------------------------
# PERSISTENCE FEATURES
# ---------------------------------------------------------

df["persistent_long_term"] = (
    df["persistence_days"] >= 90
).astype(int)

df["persistent_180_days"] = (
    df["persistence_days"] >= 180
).astype(int)

# ---------------------------------------------------------
# LABEL
# ---------------------------------------------------------

print("\nCreating initial AI label...")

df["ai_label"] = df["source_type"]

# ---------------------------------------------------------
# NORMALIZE LABELS
# ---------------------------------------------------------

label_map = {
    "POWER_GENERATION": "POWER_GENERATION",
    "POWER_SUBSTATION": "POWER_SUBSTATION",
    "INDUSTRIAL": "INDUSTRIAL"
}

df["ai_label"] = df["ai_label"].map(
    lambda x: label_map.get(x, "UNKNOWN")
)

# ---------------------------------------------------------
# REMOVE UNKNOWN
# ---------------------------------------------------------

df = df[
    df["ai_label"] != "UNKNOWN"
].copy()

# ---------------------------------------------------------
# FINAL COLUMN ORDER
# ---------------------------------------------------------

preferred_order = [
    "event_id",
    "latitude",
    "longitude",

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

    "facility_name",
    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse",
    "facility_operator",

    "source_type",
    "source_confidence_score",
    "source_confidence",

    "risk_score",
    "risk_category",

    "ai_label"
]

final_cols = [
    c for c in preferred_order
    if c in df.columns
]

df = df[final_cols]

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING DATASET CREATED")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nAI LABEL DISTRIBUTION:")
print(
    df["ai_label"].value_counts()
)

print("\nRISK DISTRIBUTION:")
print(
    df["risk_category"].value_counts()
)

print("\nMissing values:")
print(
    df.isna().sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFirst 10 rows:")
print(
    df.head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)