import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path

# ============================================================
# THERMO WATCH
# Persistent Hotspot -> Nearest OSM Facility Matching
# ============================================================

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

HOTSPOT_FILE = PROCESSED / "persistent_sources.csv"
FACILITY_FILE = PROCESSED / "osm_facilities.csv"
OUTPUT_FILE = PROCESSED / "hotspot_facility_matches.csv"

print("=" * 60)
print("THERMO WATCH - HOTSPOT FACILITY MATCHING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\nLoading persistent hotspots...")
hotspots = pd.read_csv(HOTSPOT_FILE)

print(f"Hotspots: {len(hotspots)}")

print("\nLoading OSM facilities...")
facilities = pd.read_csv(FACILITY_FILE)

print(f"Facilities: {len(facilities)}")

# ------------------------------------------------------------
# 2. Clean coordinates
# ------------------------------------------------------------

hotspots = hotspots.dropna(subset=["latitude", "longitude"]).copy()
facilities = facilities.dropna(subset=["latitude", "longitude"]).copy()

# Make sure coordinates are numeric
for col in ["latitude", "longitude"]:
    hotspots[col] = pd.to_numeric(hotspots[col], errors="coerce")
    facilities[col] = pd.to_numeric(facilities[col], errors="coerce")

hotspots = hotspots.dropna(subset=["latitude", "longitude"])
facilities = facilities.dropna(subset=["latitude", "longitude"])

print(f"\nValid hotspots: {len(hotspots)}")
print(f"Valid facilities: {len(facilities)}")

# ------------------------------------------------------------
# 3. Build BallTree
# ------------------------------------------------------------

print("\nBuilding spatial index...")

# Earth radius in km
EARTH_RADIUS_KM = 6371.0088

# Convert latitude/longitude to radians
facility_coords = np.radians(
    facilities[["latitude", "longitude"]].to_numpy()
)

hotspot_coords = np.radians(
    hotspots[["latitude", "longitude"]].to_numpy()
)

tree = BallTree(
    facility_coords,
    metric="haversine"
)

print("Spatial index ready.")

# ------------------------------------------------------------
# 4. Find nearest facility
# ------------------------------------------------------------

print("\nFinding nearest facilities...")

distances, indices = tree.query(
    hotspot_coords,
    k=1
)

# Convert radians to kilometres
distances_km = distances[:, 0] * EARTH_RADIUS_KM

nearest_indices = indices[:, 0]

nearest = facilities.iloc[nearest_indices].reset_index(drop=True)

# ------------------------------------------------------------
# 5. Create output
# ------------------------------------------------------------

result = hotspots.reset_index(drop=True).copy()

result["facility_osm_id"] = nearest["osm_id"].values
result["facility_type"] = nearest["feature_type"].values
result["facility_name"] = nearest["name"].values
result["facility_power"] = nearest["power"].values
result["facility_industrial"] = nearest["industrial"].values
result["facility_landuse"] = nearest["landuse"].values
result["facility_operator"] = nearest["operator"].values
result["facility_plant_source"] = nearest["plant_source"].values
result["facility_plant_method"] = nearest["plant_method"].values

result["facility_latitude"] = nearest["latitude"].values
result["facility_longitude"] = nearest["longitude"].values

result["distance_to_facility_km"] = distances_km

# ------------------------------------------------------------
# 6. Add useful proximity categories
# ------------------------------------------------------------

def proximity_category(distance):
    if distance <= 1:
        return "VERY_CLOSE"
    elif distance <= 5:
        return "CLOSE"
    elif distance <= 10:
        return "NEAR"
    elif distance <= 25:
        return "MODERATE"
    else:
        return "FAR"

result["proximity_category"] = result[
    "distance_to_facility_km"
].apply(proximity_category)

# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("MATCHING COMPLETE")
print("=" * 60)

print(f"Hotspots matched : {len(result)}")
print(f"Output file      : {OUTPUT_FILE}")

print("\nDistance statistics:")
print(
    result["distance_to_facility_km"].describe()
)

print("\nProximity categories:")
print(
    result["proximity_category"].value_counts()
)

print("\nTOP 10 CLOSEST HOTSPOTS:")

cols = [
    "event_id",
    "latitude",
    "longitude",
    "persistence_days",
    "mean_frp",
    "max_frp",
    "facility_name",
    "facility_type",
    "distance_to_facility_km",
    "proximity_category"
]

print(
    result.sort_values("distance_to_facility_km")
    [cols]
    .head(10)
    .to_string(index=False)
)

print("\nSaved successfully.")