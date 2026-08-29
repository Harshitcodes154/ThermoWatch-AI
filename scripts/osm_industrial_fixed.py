import os
import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ==========================================
# CONFIG
# ==========================================

INPUT_FILE = r"processed\persistent_sources.csv"
OUTPUT_FILE = r"processed\osm_industrial_india.gpkg"

# Multiple Overpass servers
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# India approximate bounding box
SOUTH = 6.0
WEST = 68.0
NORTH = 36.0
EAST = 98.0


# ==========================================
# OSM QUERY
# ==========================================

QUERY = f"""
[out:json][timeout:300];

(
  node["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});
  way["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});

  node["industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  way["industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["industrial"]({SOUTH},{WEST},{NORTH},{EAST});

  node["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});
  way["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});

  node["man_made"="refinery"]({SOUTH},{WEST},{NORTH},{EAST});
  way["man_made"="refinery"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["man_made"="refinery"]({SOUTH},{WEST},{NORTH},{EAST});

  node["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  way["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});
);

out center tags;
"""


# ==========================================
# DOWNLOAD
# ==========================================

def download_osm():

    headers = {
        "User-Agent": "THERMO-WATCH/1.0"
    }

    for url in OVERPASS_URLS:

        print("\nTrying:")
        print(url)

        try:

            response = requests.post(
                url,
                data=QUERY,
                headers=headers,
                timeout=600
            )

            print("HTTP:", response.status_code)

            if response.status_code == 200:

                print("OSM DOWNLOAD SUCCESS")

                return response.json()

            else:

                print("Server returned:", response.status_code)

        except Exception as e:

            print("ERROR:", e)

        time.sleep(5)

    raise RuntimeError(
        "All Overpass servers failed. "
        "Try again later or use an OSM extract."
    )


# ==========================================
# PARSE OSM JSON
# ==========================================

def osm_to_gdf(data):

    elements = data.get("elements", [])

    print("OSM elements:", len(elements))

    records = []

    for el in elements:

        tags = el.get("tags", {})

        lat = el.get("lat")
        lon = el.get("lon")

        # For ways/relations
        if lat is None or lon is None:

            center = el.get("center", {})

            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue

        records.append({
            "osm_id": el.get("id"),
            "osm_type": el.get("type"),
            "name": tags.get("name"),
            "industrial": tags.get("industrial"),
            "landuse": tags.get("landuse"),
            "power": tags.get("power"),
            "plant_source": tags.get("plant:source"),
            "plant_type": tags.get("plant:type"),
            "man_made": tags.get("man_made"),
            "latitude": lat,
            "longitude": lon,
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No OSM facilities found.")

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df["longitude"], df["latitude"])
    ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326"
    )

    return gdf


# ==========================================
# MAIN
# ==========================================

print("======================================")
print("THERMO WATCH - OSM INDUSTRIAL DATA")
print("======================================")

print("\nDownloading India OSM facilities...")

data = download_osm()

gdf = osm_to_gdf(data)

print("\n======================================")
print("OSM DATA CREATED")
print("======================================")

print("Rows:", len(gdf))
print("CRS:", gdf.crs)

print("\nFacility types:")

print(
    gdf[
        ["industrial", "power", "plant_source",
         "plant_type", "man_made", "landuse"]
    ].notna().sum()
)

os.makedirs("processed", exist_ok=True)

gdf.to_file(
    OUTPUT_FILE,
    driver="GPKG"
)

print("\nSaved:")
print(OUTPUT_FILE)