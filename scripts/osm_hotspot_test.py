import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import time

# ============================================================
# CONFIGURATION
# ============================================================

INPUT = r"processed\persistent_sources.csv"
OUTPUT = r"processed\osm_hotspot_test.csv"

# Alternative Overpass server
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"

# Don't request huge areas
BUFFER_METERS = 2000

# Test only 10 hotspots first
TEST_LIMIT = 10


# ============================================================
# OSM SETTINGS
# ============================================================

ox.settings.requests_timeout = 120
ox.settings.use_cache = True
ox.settings.log_console = False


# ============================================================
# LOAD DATA
# ============================================================

print("Loading persistent sources...")

df = pd.read_csv(INPUT)

print("Total persistent sources:", len(df))

# Only first 10 for testing
df = df.head(TEST_LIMIT).copy()

print("Test hotspots:", len(df))


# ============================================================
# OSM TAGS
# ============================================================

tags = {
    "power": ["plant", "generator"],
    "industrial": True,
    "landuse": ["industrial"],
    "man_made": ["works"]
}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS HOTSPOTS
# ============================================================

for index, row in df.iterrows():

    event_id = row["event_id"]
    lat = float(row["latitude"])
    lon = float(row["longitude"])

    print()
    print("=" * 60)
    print(f"[{len(results) + 1}/{len(df)}] HOTSPOT")
    print("Event:", event_id)
    print("Location:", lat, lon)

    try:

        # ----------------------------------------------------
        # Create hotspot point
        # ----------------------------------------------------

        point = Point(lon, lat)

        hotspot = gpd.GeoDataFrame(
            {"event_id": [event_id]},
            geometry=[point],
            crs="EPSG:4326"
        )

        # ----------------------------------------------------
        # Project to meters
        # ----------------------------------------------------

        hotspot_projected = hotspot.to_crs("EPSG:3857")

        # ----------------------------------------------------
        # Create 2 km buffer
        # ----------------------------------------------------

        buffer = hotspot_projected.geometry.iloc[0].buffer(
            BUFFER_METERS
        )

        buffer_gdf = gpd.GeoDataFrame(
            geometry=[buffer],
            crs="EPSG:3857"
        )

        # Convert back to WGS84 for OSM
        buffer_gdf = buffer_gdf.to_crs("EPSG:4326")

        polygon = buffer_gdf.geometry.iloc[0]

        print("Searching OSM within 2 km...")

        # ----------------------------------------------------
        # Query OSM
        # ----------------------------------------------------

        facilities = ox.features_from_polygon(
            polygon,
            tags
        )

        print("Facilities found:", len(facilities))

        # ----------------------------------------------------
        # No facilities
        # ----------------------------------------------------

        if len(facilities) == 0:

            results.append({
                "event_id": event_id,
                "latitude": lat,
                "longitude": lon,
                "facility_count": 0,
                "nearest_facility": None,
                "distance_to_facility_km": None
            })

            continue

        # ----------------------------------------------------
        # Convert facilities to meters
        # ----------------------------------------------------

        facilities_projected = facilities.to_crs("EPSG:3857")

        hotspot_point = hotspot_projected.geometry.iloc[0]

        # ----------------------------------------------------
        # Calculate distance
        # ----------------------------------------------------

        facilities_projected["distance_m"] = (
            facilities_projected.geometry
            .distance(hotspot_point)
        )

        # ----------------------------------------------------
        # Find nearest facility
        # ----------------------------------------------------

        nearest = facilities_projected.sort_values(
            "distance_m"
        ).iloc[0]

        distance_km = nearest["distance_m"] / 1000

        # ----------------------------------------------------
        # Identify facility type
        # ----------------------------------------------------

        facility_type = "unknown"

        power_value = nearest.get("power")
        industrial_value = nearest.get("industrial")
        landuse_value = nearest.get("landuse")
        man_made_value = nearest.get("man_made")

        if power_value == "plant":
            facility_type = "power_plant"

        elif power_value == "generator":
            facility_type = "generator"

        elif industrial_value is not None:
            facility_type = "industrial"

        elif landuse_value == "industrial":
            facility_type = "industrial_area"

        elif man_made_value == "works":
            facility_type = "industrial_works"

        # ----------------------------------------------------
        # Facility name
        # ----------------------------------------------------

        facility_name = None

        if "name" in nearest.index:

            value = nearest["name"]

            if pd.notna(value):
                facility_name = str(value)

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({
            "event_id": event_id,
            "latitude": lat,
            "longitude": lon,
            "facility_count": len(facilities),
            "nearest_facility": facility_type,
            "facility_name": facility_name,
            "distance_to_facility_km": round(
                distance_km,
                3
            )
        })

        print("Nearest facility:", facility_type)
        print("Facility name:", facility_name)
        print("Distance:", round(distance_km, 3), "km")

        # Small pause so we don't hammer server
        time.sleep(2)

    except Exception as e:

        print("ERROR:", str(e))

        results.append({
            "event_id": event_id,
            "latitude": lat,
            "longitude": lon,
            "facility_count": None,
            "nearest_facility": None,
            "facility_name": None,
            "distance_to_facility_km": None
        })

        # Wait before next request
        time.sleep(5)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("OSM HOTSPOT TEST COMPLETE")
print("=" * 60)

print("Hotspots tested:", len(df))
print("Results created:", len(result_df))

print()
print(result_df.to_string(index=False))

print()
print("Saved to:")
print(OUTPUT)