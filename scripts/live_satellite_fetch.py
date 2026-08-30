import os
import io
import requests
import pandas as pd

# ============================================================
# THERMO WATCH - LIVE SATELLITE HOTSPOT FETCHER
# ============================================================

print("=" * 70)
print("THERMO WATCH - LIVE SATELLITE HOTSPOT FETCH")
print("=" * 70)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# IMPORTANT:
# Yahan apni NASA FIRMS MAP_KEY daalna.
# Example:
# FIRMS_MAP_KEY = "xxxxxxxxxxxxxxxxxxxxxxxx"
#
# Ya environment variable se bhi le sakte ho.

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "").strip()

if not FIRMS_MAP_KEY:
    FIRMS_MAP_KEY = "PASTE_YOUR_FIRMS_MAP_KEY_HERE"

# India bounding box
# west, south, east, north
BBOX = "68,6,98,38"

# MODIS NRT fire data
SOURCE = "MODIS_NRT"

# Last N days
DAYS = 1

OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "live_satellite_hotspots.csv"
)

# ------------------------------------------------------------
# CHECK API KEY
# ------------------------------------------------------------

if FIRMS_MAP_KEY == "b2c4b393bbb4537eba197bc4aeefbfe9":
    print()
    print("ERROR: FIRMS API KEY NOT FOUND")
    print()
    print("Open this file and replace:")
    print("PASTE_YOUR_FIRMS_MAP_KEY_HERE")
    print("with your actual NASA FIRMS MAP_KEY.")
    print()
    raise SystemExit(1)

# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# FIRMS API
# ------------------------------------------------------------

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{FIRMS_MAP_KEY}/{SOURCE}/{BBOX}/{DAYS}"
)

print()
print("Satellite source :", SOURCE)
print("Region           : India")
print("Days             :", DAYS)
print()
print("Fetching latest satellite fire detections...")
print("URL:", url.replace(FIRMS_MAP_KEY, "********"))

# ------------------------------------------------------------
# REQUEST DATA
# ------------------------------------------------------------

try:
    response = requests.get(
        url,
        timeout=60
    )

except requests.RequestException as e:
    print()
    print("ERROR: Network request failed.")
    print(e)
    raise SystemExit(1)

# ------------------------------------------------------------
# STATUS CHECK
# ------------------------------------------------------------

print()
print("HTTP Status:", response.status_code)

if response.status_code != 200:
    print()
    print("ERROR: FIRMS API request failed.")
    print()
    print("Response:")
    print(response.text[:1000])
    raise SystemExit(1)

# ------------------------------------------------------------
# READ CSV
# ------------------------------------------------------------

try:
    df = pd.read_csv(io.StringIO(response.text))

except Exception as e:
    print()
    print("ERROR: Could not read FIRMS response as CSV.")
    print(e)
    raise SystemExit(1)

# ------------------------------------------------------------
# EMPTY DATA CHECK
# ------------------------------------------------------------

if df.empty:
    print()
    print("NO ACTIVE FIRE DETECTIONS FOUND.")
    print()
    print("The API worked, but no fire detections were returned")
    print("for the selected region/time window.")
    raise SystemExit(0)

# ------------------------------------------------------------
# CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = [
    str(col).strip().lower()
    for col in df.columns
]

# ------------------------------------------------------------
# SHOW RAW INFORMATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("LIVE SATELLITE DATA RECEIVED")
print("=" * 70)

print("Rows:", len(df))

print()
print("Columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# STANDARDIZE IMPORTANT COLUMNS
# ------------------------------------------------------------

rename_map = {}

if "latitude" in df.columns:
    rename_map["latitude"] = "latitude"

if "longitude" in df.columns:
    rename_map["longitude"] = "longitude"

if "frp" in df.columns:
    rename_map["frp"] = "frp"

if "brightness" in df.columns:
    rename_map["brightness"] = "brightness"

df = df.rename(columns=rename_map)

# ------------------------------------------------------------
# VALIDATE LOCATION
# ------------------------------------------------------------

required_columns = [
    "latitude",
    "longitude"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print()
    print("ERROR: Required columns missing:")
    print(missing)
    raise SystemExit(1)

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

if "frp" in df.columns:
    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce"
    )

if "brightness" in df.columns:
    df["brightness"] = pd.to_numeric(
        df["brightness"],
        errors="coerce"
    )

# ------------------------------------------------------------
# REMOVE INVALID LOCATIONS
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
).copy()

# ------------------------------------------------------------
# ADD THERMO WATCH COLUMNS
# ------------------------------------------------------------

df["detection_source"] = SOURCE

df["live_detection"] = 1

df["fetched_at"] = pd.Timestamp.utcnow()

# ------------------------------------------------------------
# SORT BY FRP
# ------------------------------------------------------------

if "frp" in df.columns:
    df = df.sort_values(
        by="frp",
        ascending=False
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("LIVE SATELLITE FETCH COMPLETE")
print("=" * 70)

print("Total detections :", len(df))

if "frp" in df.columns:
    print(
        "Maximum FRP      :",
        round(df["frp"].max(), 2)
    )

    print(
        "Average FRP      :",
        round(df["frp"].mean(), 2)
    )

print()
print("Output file:")
print(os.path.abspath(OUTPUT_FILE))

print()
print("=" * 70)
print("TOP 10 LIVE FIRE DETECTIONS")
print("=" * 70)

display_columns = [
    col
    for col in [
        "latitude",
        "longitude",
        "frp",
        "brightness",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence"
    ]
    if col in df.columns
]

print(
    df[display_columns]
    .head(10)
    .to_string(index=False)
)

print()
print("=" * 70)
print("NEXT STEP: LIVE HOTSPOT -> FACILITY MATCHING")
print("=" * 70)