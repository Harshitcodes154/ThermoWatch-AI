import osmnx as ox
from pathlib import Path

ox.settings.requests_timeout = 180

place = "Surat, Gujarat, India"

print("Downloading OSM industrial works...")

tags = {
    "man_made": "works"
}

gdf = ox.features_from_place(place, tags)

print("\n==============================")
print("OSM INDUSTRIAL WORKS")
print("==============================")
print("Rows:", len(gdf))
print("CRS:", gdf.crs)

Path("processed").mkdir(exist_ok=True)

gdf.to_file(
    "processed/osm_surat_works.gpkg",
    driver="GPKG"
)

print("Saved: processed/osm_surat_works.gpkg")