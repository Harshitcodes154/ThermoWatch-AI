import osmnx as ox
from pathlib import Path

ox.settings.requests_timeout = 180

place = "Surat, Gujarat, India"

print("Downloading OSM power plants...")

tags = {
    "power": "plant"
}

gdf = ox.features_from_place(place, tags)

print("\n==============================")
print("OSM TEST SUCCESS")
print("==============================")
print("Rows:", len(gdf))
print("CRS:", gdf.crs)
print("Columns:", gdf.columns.tolist())

Path("processed").mkdir(exist_ok=True)

gdf.to_file(
    "processed/osm_surat_powerplants.gpkg",
    driver="GPKG"
)

print("Saved: processed/osm_surat_powerplants.gpkg")