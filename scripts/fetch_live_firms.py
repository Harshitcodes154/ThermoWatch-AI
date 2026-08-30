import os
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timezone


# ======================================================================
# THERMO WATCH - LIVE FIRMS HOTSPOT FETCH
# ======================================================================

print("=" * 70)
print("THERMO WATCH - LIVE FIRMS HOTSPOT FETCH")
print("=" * 70)


# ======================================================================
# CONFIGURATION
# ======================================================================

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

# India bounding box
# west, south, east, north
INDIA_BBOX = "68,6,98,38"

# NASA FIRMS source
SOURCE = "VIIRS_SNPP_NRT"

# Fetch last 5 days
DAYS = 5

# Output
OUTPUT_FILE = "processed/live_firms_hotspots.csv"


# ======================================================================
# CHECK API KEY
# ======================================================================

print("\nChecking MAP_KEY...")

if not MAP_KEY:
    print("\nERROR: FIRMS_MAP_KEY not found.")
    print("\nRun:")
    print('$env:FIRMS_MAP_KEY="YOUR_MAP_KEY"')
    raise SystemExit(1)

print("MAP_KEY found: YES")


# ======================================================================
# BUILD API URL
# ======================================================================

url = (
    f"https://firms.modaps.eosdis.nasa.gov/"
    f"api/area/csv/"
    f"{MAP_KEY}/"
    f"{SOURCE}/"
    f"{INDIA_BBOX}/"
    f"{DAYS}"
)

print("\nRequest configuration")
print("-" * 50)
print("Source :", SOURCE)
print("Region :", INDIA_BBOX)
print("Days   :", DAYS)


# ======================================================================
# REQUEST FIRMS DATA
# ======================================================================

print("\nFetching FIRMS satellite data...")

try:

    response = requests.get(
        url,
        timeout=60
    )

except requests.RequestException as e:

    print("\nERROR connecting to FIRMS:")
    print(e)

    raise SystemExit(1)


print("HTTP Status:", response.status_code)


# ======================================================================
# RESPONSE VALIDATION
# ======================================================================

if response.status_code != 200:

    print("\nFIRMS API ERROR")
    print("-" * 50)
    print(response.text[:2000])

    raise SystemExit(1)


# ======================================================================
# READ CSV
# ======================================================================

try:

    df = pd.read_csv(
        StringIO(response.text)
    )

except Exception as e:

    print("\nERROR reading FIRMS response:")
    print(e)

    print("\nRaw response:")
    print(response.text[:2000])

    raise SystemExit(1)


print("\nRaw hotspots received:", len(df))


# ======================================================================
# EMPTY DATA HANDLING
# ======================================================================

if len(df) == 0:

    print("\nWARNING")
    print("-" * 50)
    print("FIRMS returned ZERO hotspots.")

    print("\nAPI request was successful.")
    print("HTTP 200 confirms the API request worked.")

    print("\nPossible reasons:")
    print("1. No available observations for this query.")
    print("2. FIRMS data availability delay.")
    print("3. VIIRS source currently has no records.")
    print("4. Satellite coverage/data ingestion delay.")

    os.makedirs(
        "processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved empty result:")
    print(os.path.abspath(OUTPUT_FILE))

    print("\n" + "=" * 70)
    print("LIVE FIRMS FETCH COMPLETE - ZERO HOTSPOTS")
    print("=" * 70)

    raise SystemExit(0)


# ======================================================================
# REQUIRED COLUMNS
# ======================================================================

required_columns = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "frp"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Required columns missing")
    print(missing_columns)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise SystemExit(1)


# ======================================================================
# CLEAN NUMERIC DATA
# ======================================================================

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df["frp"] = pd.to_numeric(
    df["frp"],
    errors="coerce"
)


# ======================================================================
# REMOVE INVALID ROWS
# ======================================================================

before = len(df)

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "frp"
    ]
).copy()

after = len(df)

print("\nInvalid rows removed:", before - after)


# ======================================================================
# INDIA FILTER
# ======================================================================

df = df[
    (df["latitude"] >= 6) &
    (df["latitude"] <= 38) &
    (df["longitude"] >= 68) &
    (df["longitude"] <= 98)
].copy()

print(
    "Hotspots after India filter:",
    len(df)
)


# ======================================================================
# ACQUISITION DATETIME
# ======================================================================

df["acquisition_datetime"] = pd.to_datetime(
    df["acq_date"].astype(str)
    + " "
    + df["acq_time"].astype(str).str.zfill(4),
    format="%Y-%m-%d %H%M",
    errors="coerce"
)


# ======================================================================
# SORT LATEST FIRST
# ======================================================================

df = df.sort_values(
    by="acquisition_datetime",
    ascending=False
).reset_index(drop=True)


# ======================================================================
# CREATE THERMO WATCH HOTSPOT ID
# ======================================================================

df["hotspot_id"] = [
    f"LIVE_{i + 1:06d}"
    for i in range(len(df))
]


# ======================================================================
# FETCH TIMESTAMP
# ======================================================================

df["fetched_at"] = datetime.now(
    timezone.utc
).isoformat()


# ======================================================================
# SAVE DATA
# ======================================================================

os.makedirs(
    "processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("LIVE FIRMS FETCH COMPLETE")
print("=" * 70)

print(
    "\nTotal live hotspots:",
    len(df)
)


# ======================================================================
# DATE INFORMATION
# ======================================================================

valid_dates = df[
    "acquisition_datetime"
].dropna()

if len(valid_dates) > 0:

    print(
        "\nOldest observation:",
        valid_dates.min()
    )

    print(
        "Latest observation:",
        valid_dates.max()
    )


# ======================================================================
# FRP SUMMARY
# ======================================================================

print("\nFRP SUMMARY")
print("-" * 50)

print(
    "Average FRP:",
    round(df["frp"].mean(), 2)
)

print(
    "Maximum FRP:",
    round(df["frp"].max(), 2)
)


# ======================================================================
# SATELLITE DISTRIBUTION
# ======================================================================

print("\nSATELLITE DISTRIBUTION")
print("-" * 50)

print(
    df["satellite"]
    .value_counts()
    .to_string()
)


# ======================================================================
# CONFIDENCE DISTRIBUTION
# ======================================================================

print("\nCONFIDENCE DISTRIBUTION")
print("-" * 50)

print(
    df["confidence"]
    .value_counts()
    .to_string()
)


# ======================================================================
# TOP 10 HOTSPOTS
# ======================================================================

print("\nTOP 10 LATEST HOTSPOTS")
print("-" * 50)

display_columns = [
    "hotspot_id",
    "latitude",
    "longitude",
    "frp",
    "confidence",
    "satellite",
    "instrument",
    "acquisition_datetime"
]

print(
    df[display_columns]
    .head(10)
    .to_string(index=False)
)


# ======================================================================
# OUTPUT
# ======================================================================

print("\nSaved:")
print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)