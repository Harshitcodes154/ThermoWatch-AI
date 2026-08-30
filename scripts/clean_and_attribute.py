import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

HOTSPOT_FILE = PROCESSED / "persistent_sources.csv"
FACILITY_FILE = PROCESSED / "osm_facilities.csv"
OUTPUT_FILE = PROCESSED / "hotspot_source_attribution.csv"

EARTH_RADIUS_KM = 6371.0088

print("=" * 60)
print("THERMO WATCH - SOURCE ATTRIBUTION")
print("=" * 60)

# --------------------------------------------------
# LOAD
# --------------------------------------------------

print("\nLoading hotspots...")
hotspots = pd.read_csv(HOTSPOT_FILE)

print("Loading OSM facilities...")
facilities = pd.read_csv(FACILITY_FILE)

print(f"Hotspots loaded  : {len(hotspots)}")
print(f"Facilities loaded: {len(facilities)}")

# --------------------------------------------------
# CLEAN
# --------------------------------------------------

facilities["feature_type"] = facilities["feature_type"].fillna("").astype(str)
facilities["power"] = facilities["power"].fillna("").astype(str)
facilities["industrial"] = facilities["industrial"].fillna("").astype(str)
facilities["landuse"] = facilities["landuse"].fillna("").astype(str)

# Actual useful facility indicators
relevant = (
    facilities["power"].str.lower().isin([
        "plant",
        "generator",
        "substation",
        "transformer",
        "tower"
    ])
    |
    facilities["industrial"].str.lower().ne("")
    |
    facilities["landuse"].str.lower().isin([
        "industrial",
        "depot",
        "factory"
    ])
)

facilities_clean = facilities[relevant].copy()

print("\nRelevant facilities:", len(facilities_clean))

# Remove invalid coordinates
facilities_clean["latitude"] = pd.to_numeric(
    facilities_clean["latitude"], errors="coerce"
)
facilities_clean["longitude"] = pd.to_numeric(
    facilities_clean["longitude"], errors="coerce"
)

hotspots["latitude"] = pd.to_numeric(
    hotspots["latitude"], errors="coerce"
)
hotspots["longitude"] = pd.to_numeric(
    hotspots["longitude"], errors="coerce"
)

facilities_clean = facilities_clean.dropna(
    subset=["latitude", "longitude"]
)

hotspots = hotspots.dropna(
    subset=["latitude", "longitude"]
).reset_index(drop=True)

facilities_clean = facilities_clean.reset_index(drop=True)

# --------------------------------------------------
# SPATIAL INDEX
# --------------------------------------------------

print("\nBuilding spatial index...")

facility_coords = np.radians(
    facilities_clean[["latitude", "longitude"]].values
)

hotspot_coords = np.radians(
    hotspots[["latitude", "longitude"]].values
)

tree = BallTree(
    facility_coords,
    metric="haversine"
)

# Find nearest 3 facilities
distances, indices = tree.query(
    hotspot_coords,
    k=3
)

distances_km = distances * EARTH_RADIUS_KM

print("Spatial index ready.")

# --------------------------------------------------
# SELECT BEST FACILITY
# --------------------------------------------------

# nearest facility
nearest_idx = indices[:, 0]

nearest = facilities_clean.iloc[
    nearest_idx
].reset_index(drop=True)

result = hotspots.copy()

result["facility_osm_id"] = nearest["osm_id"].values
result["facility_name"] = nearest["name"].values
result["facility_type"] = nearest["feature_type"].values
result["facility_power"] = nearest["power"].values
result["facility_industrial"] = nearest["industrial"].values
result["facility_landuse"] = nearest["landuse"].values
result["facility_operator"] = nearest["operator"].values
result["facility_latitude"] = nearest["latitude"].values
result["facility_longitude"] = nearest["longitude"].values

result["distance_to_facility_km"] = distances_km[:, 0]

# --------------------------------------------------
# PROXIMITY
# --------------------------------------------------

def proximity(d):
    if d <= 1:
        return "VERY_CLOSE"
    elif d <= 5:
        return "CLOSE"
    elif d <= 10:
        return "NEAR"
    elif d <= 25:
        return "MODERATE"
    else:
        return "FAR"

result["proximity"] = result[
    "distance_to_facility_km"
].apply(proximity)

# --------------------------------------------------
# SOURCE TYPE
# --------------------------------------------------

def source_type(row):

    power = str(row["facility_power"]).lower()
    industrial = str(row["facility_industrial"]).lower()
    landuse = str(row["facility_landuse"]).lower()

    if power in ["plant", "generator"]:
        return "POWER_GENERATION"

    if power == "substation":
        return "POWER_SUBSTATION"

    if industrial:
        return "INDUSTRIAL"

    if landuse in ["industrial", "factory"]:
        return "INDUSTRIAL"

    return "OTHER"

result["source_type"] = result.apply(
    source_type,
    axis=1
)

# --------------------------------------------------
# SOURCE CONFIDENCE
# --------------------------------------------------

def confidence(row):

    distance = row["distance_to_facility_km"]
    persistence = row["persistence_days"]
    obs = row["observation_count"]
    frp = row["mean_frp"]

    score = 0

    # Distance
    if distance <= 1:
        score += 40
    elif distance <= 5:
        score += 30
    elif distance <= 10:
        score += 20
    elif distance <= 25:
        score += 10

    # Persistence
    if persistence >= 180:
        score += 25
    elif persistence >= 90:
        score += 20
    elif persistence >= 30:
        score += 10

    # Observation count
    if obs >= 1000:
        score += 20
    elif obs >= 100:
        score += 10

    # FRP
    if frp >= 10:
        score += 15
    elif frp >= 5:
        score += 10
    elif frp >= 2:
        score += 5

    return min(score, 100)

result["source_confidence_score"] = result.apply(
    confidence,
    axis=1
)

# --------------------------------------------------
# CONFIDENCE LABEL
# --------------------------------------------------

def confidence_label(score):

    if score >= 75:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    else:
        return "LOW"

result["source_confidence"] = result[
    "source_confidence_score"
].apply(confidence_label)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SOURCE ATTRIBUTION COMPLETE")
print("=" * 60)

print("Hotspots:", len(result))
print("Relevant facilities:", len(facilities_clean))

print("\nSOURCE TYPES:")
print(result["source_type"].value_counts())

print("\nCONFIDENCE:")
print(result["source_confidence"].value_counts())

print("\nDISTANCE:")
print(result["distance_to_facility_km"].describe())

print("\nTOP 10 HIGH-CONFIDENCE SOURCES:")

cols = [
    "event_id",
    "latitude",
    "longitude",
    "persistence_days",
    "observation_count",
    "mean_frp",
    "facility_name",
    "source_type",
    "distance_to_facility_km",
    "source_confidence_score",
    "source_confidence"
]

print(
    result.sort_values(
        "source_confidence_score",
        ascending=False
    )[cols].head(10).to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)