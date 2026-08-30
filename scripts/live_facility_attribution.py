import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ============================================================
# THERMO WATCH - LIVE FACILITY ATTRIBUTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_FILE = os.path.join(
    BASE_DIR, "processed", "live_firms_hotspots.csv"
)

FACILITY_FILE = os.path.join(
    BASE_DIR, "processed", "osm_facilities.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "processed", "live_facility_matches.csv"
)

print("=" * 70)
print("THERMO WATCH - LIVE FACILITY ATTRIBUTION")
print("=" * 70)


# ============================================================
# 1. LOAD LIVE HOTSPOTS
# ============================================================

print("\nLoading live FIRMS hotspots...")

if not os.path.exists(LIVE_FILE):
    print("ERROR: live_firms_hotspots.csv not found")
    sys.exit(1)

live = pd.read_csv(LIVE_FILE)

print(f"Live hotspots: {len(live)}")


# ============================================================
# 2. LOAD OSM FACILITIES
# ============================================================

print("\nLoading OSM facilities...")

if not os.path.exists(FACILITY_FILE):
    print("ERROR: osm_facilities.csv not found")
    sys.exit(1)

facilities = pd.read_csv(FACILITY_FILE)

print(f"Facilities loaded: {len(facilities)}")


# ============================================================
# 3. VALIDATE COORDINATES
# ============================================================

required_live = ["latitude", "longitude"]

required_facility = [
    "latitude",
    "longitude"
]

for col in required_live:
    if col not in live.columns:
        print(f"ERROR: Missing live column: {col}")
        sys.exit(1)

for col in required_facility:
    if col not in facilities.columns:
        print(f"ERROR: Missing facility column: {col}")
        sys.exit(1)


live = live.dropna(
    subset=["latitude", "longitude"]
).copy()

facilities = facilities.dropna(
    subset=["latitude", "longitude"]
).copy()

print(f"Valid live hotspots: {len(live)}")
print(f"Valid facilities: {len(facilities)}")


# ============================================================
# 4. PREPARE FACILITY COORDINATES
# ============================================================

print("\nBuilding spatial index...")

facility_coords = np.radians(
    facilities[["latitude", "longitude"]].values
)

live_coords = np.radians(
    live[["latitude", "longitude"]].values
)

# Earth radius in km
EARTH_RADIUS_KM = 6371.0088

tree = BallTree(
    facility_coords,
    metric="haversine"
)


# ============================================================
# 5. FIND NEAREST FACILITY
# ============================================================

print("Finding nearest facility for each hotspot...")

distances, indices = tree.query(
    live_coords,
    k=1
)

distances_km = distances[:, 0] * EARTH_RADIUS_KM
nearest_indices = indices[:, 0]

nearest = facilities.iloc[
    nearest_indices
].reset_index(drop=True)


# ============================================================
# 6. ATTACH FACILITY INFORMATION
# ============================================================

result = live.reset_index(drop=True).copy()

result["facility_osm_id"] = nearest.get(
    "osm_id",
    np.nan
)

result["facility_type"] = nearest.get(
    "feature_type",
    np.nan
)

result["facility_name"] = nearest.get(
    "name",
    np.nan
)

result["facility_power"] = nearest.get(
    "power",
    np.nan
)

result["facility_industrial"] = nearest.get(
    "industrial",
    np.nan
)

result["facility_landuse"] = nearest.get(
    "landuse",
    np.nan
)

result["facility_operator"] = nearest.get(
    "operator",
    np.nan
)

result["facility_plant_source"] = nearest.get(
    "plant_source",
    np.nan
)

result["facility_plant_method"] = nearest.get(
    "plant_method",
    np.nan
)

result["facility_latitude"] = nearest[
    "latitude"
].values

result["facility_longitude"] = nearest[
    "longitude"
].values

result["distance_to_facility_km"] = distances_km


# ============================================================
# 7. PROXIMITY CATEGORY
# ============================================================

def proximity_category(distance):
    if distance <= 1:
        return "VERY_CLOSE"
    elif distance <= 2:
        return "CLOSE"
    elif distance <= 5:
        return "NEAR"
    elif distance <= 10:
        return "DISTANT"
    else:
        return "FAR"


result["proximity_category"] = (
    result["distance_to_facility_km"]
    .apply(proximity_category)
)


# ============================================================
# 8. FACILITY MATCH QUALITY
# ============================================================

def match_quality(distance):
    if distance <= 1:
        return "HIGH"
    elif distance <= 5:
        return "MEDIUM"
    elif distance <= 10:
        return "LOW"
    else:
        return "VERY_LOW"


result["facility_match_quality"] = (
    result["distance_to_facility_km"]
    .apply(match_quality)
)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FACILITY ATTRIBUTION RESULTS")
print("=" * 70)

print(f"\nTotal hotspots processed: {len(result)}")

print("\nProximity distribution:")
print(
    result["proximity_category"]
    .value_counts()
    .to_string()
)

print("\nMatch quality:")
print(
    result["facility_match_quality"]
    .value_counts()
    .to_string()
)


# ============================================================
# 10. SHOW TOP MATCHES
# ============================================================

print("\n" + "-" * 70)
print("TOP 20 CLOSEST HOTSPOT-FACILITY MATCHES")
print("-" * 70)

display_cols = [
    "hotspot_id",
    "latitude",
    "longitude",
    "frp",
    "facility_name",
    "facility_type",
    "facility_power",
    "facility_industrial",
    "distance_to_facility_km",
    "proximity_category",
    "facility_match_quality"
]

display_cols = [
    c for c in display_cols
    if c in result.columns
]

print(
    result
    .sort_values("distance_to_facility_km")
    [display_cols]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 11. FACILITY TYPE SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("NEAREST FACILITY TYPE DISTRIBUTION")
print("-" * 70)

print(
    result["facility_type"]
    .fillna("UNKNOWN")
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# 12. SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("LIVE FACILITY ATTRIBUTION COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Connect facility-enriched live data to V5 AI inference.")

print("=" * 70)