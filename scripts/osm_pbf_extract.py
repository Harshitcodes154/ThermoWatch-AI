from pathlib import Path
import geopandas as gpd
import pandas as pd
import subprocess
import sys
import shutil

# ==============================
# PATHS
# ==============================

BASE = Path(__file__).resolve().parent.parent
OSM_DIR = BASE / "osm_data"
OUT_DIR = BASE / "processed"
OUT_DIR.mkdir(exist_ok=True)

# ==============================
# PBF FILES
# ==============================

PBF_FILES = [
    "central-zone-260828.osm.pbf",
    "eastern-zone-260828.osm.pbf",
    "north-eastern-zone-260828.osm.pbf",
    "northern-zone-260828.osm.pbf",
    "southern-zone-260828.osm.pbf",
    "western-zone-260828.osm.pbf",
]

# ==============================
# CHECK FILES
# ==============================

print("Checking OSM PBF files...")

missing = []

for name in PBF_FILES:
    path = OSM_DIR / name

    if not path.exists():
        missing.append(name)
    else:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"OK: {name} ({size_mb:.1f} MB)")

if missing:
    print("\nMISSING FILES:")
    for x in missing:
        print(" -", x)

    print("\nPut all PBF files inside:")
    print(OSM_DIR)

    sys.exit(1)

print("\nAll PBF files found.")