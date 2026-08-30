import os
import time
import pandas as pd
import requests

# ============================================================
# THERMO WATCH - FULL SATELLITE FEATURE COLLECTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "processed", "training_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "processed", "satellite_features.csv"
)

TEMP_FILE = os.path.join(
    BASE_DIR, "processed", "satellite_features_progress.csv"
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

BATCH_SIZE = 25
SLEEP_SECONDS = 1

# Sentinel-2 STAC API
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

# Search window
START_DATE = "2025-08-01T00:00:00Z"
END_DATE = "2026-06-01T00:00:00Z"

# Search radius approximately 10 km
SEARCH_DISTANCE_DEG = 0.10


# ============================================================
# FUNCTIONS
# ============================================================

def search_sentinel(lat, lon):
    """
    Search Sentinel-2 scenes around hotspot.
    """

    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{START_DATE}/{END_DATE}",
        "intersects": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)]
        },
        "limit": 5
    }

    try:

        response = requests.post(
            STAC_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        features = data.get("features", [])

        if not features:
            return {
                "satellite_available": 0,
                "sentinel_scene_count": 0,
                "cloud_cover": None,
                "satellite_scene_date": None
            }

        # Sort newest first
        features.sort(
            key=lambda x: x.get("properties", {}).get(
                "datetime", ""
            ),
            reverse=True
        )

        latest = features[0]

        props = latest.get("properties", {})

        cloud_cover = props.get(
            "eo:cloud_cover"
        )

        scene_date = props.get(
            "datetime"
        )

        return {
            "satellite_available": 1,
            "sentinel_scene_count": len(features),
            "cloud_cover": cloud_cover,
            "satellite_scene_date": scene_date
        }

    except Exception as e:

        print(
            f"ERROR ({lat}, {lon}): {str(e)[:100]}"
        )

        return {
            "satellite_available": 0,
            "sentinel_scene_count": 0,
            "cloud_cover": None,
            "satellite_scene_date": None
        }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THERMO WATCH - FULL SATELLITE FEATURE COLLECTION")
print("=" * 70)

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total hotspots: {len(df)}")


# ============================================================
# RESUME PREVIOUS PROGRESS
# ============================================================

if os.path.exists(TEMP_FILE):

    print("\nPrevious progress found.")
    print("Loading checkpoint...")

    progress = pd.read_csv(TEMP_FILE)

    processed_ids = set(
        progress["event_id"].astype(str)
    )

    print(
        f"Already processed: {len(processed_ids)}"
    )

else:

    progress = pd.DataFrame()

    processed_ids = set()

    print("\nNo previous checkpoint found.")


# ============================================================
# PREPARE HOTSPOTS
# ============================================================

required_columns = [
    "event_id",
    "latitude",
    "longitude"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Missing required column: {col}"
        )


df["event_id"] = df["event_id"].astype(str)

remaining = df[
    ~df["event_id"].isin(processed_ids)
].copy()

print(
    f"Remaining hotspots: {len(remaining)}"
)


# ============================================================
# PROCESS
# ============================================================

new_results = []

total = len(remaining)

start_time = time.time()

for index, row in enumerate(
    remaining.itertuples(index=False),
    start=1
):

    event_id = str(row.event_id)

    lat = float(row.latitude)

    lon = float(row.longitude)

    print()
    print("-" * 70)

    print(
        f"[{index}/{total}] "
        f"{event_id}"
    )

    print(
        f"Location: {lat:.6f}, {lon:.6f}"
    )

    result = search_sentinel(
        lat,
        lon
    )

    result["event_id"] = event_id

    result["latitude"] = lat

    result["longitude"] = lon

    new_results.append(result)

    print(
        f"Scenes found: "
        f"{result['sentinel_scene_count']}"
    )

    print(
        f"Cloud cover: "
        f"{result['cloud_cover']}"
    )

    print(
        f"Available: "
        f"{result['satellite_available']}"
    )

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    if len(new_results) >= BATCH_SIZE:

        batch_df = pd.DataFrame(
            new_results
        )

        if len(progress) > 0:

            progress = pd.concat(
                [
                    progress,
                    batch_df
                ],
                ignore_index=True
            )

        else:

            progress = batch_df

        progress.to_csv(
            TEMP_FILE,
            index=False
        )

        print()
        print(
            f"Checkpoint saved: "
            f"{len(progress)} rows"
        )

        new_results = []

    time.sleep(
        SLEEP_SECONDS
    )


# ============================================================
# SAVE FINAL REMAINING RESULTS
# ============================================================

if len(new_results) > 0:

    batch_df = pd.DataFrame(
        new_results
    )

    if len(progress) > 0:

        progress = pd.concat(
            [
                progress,
                batch_df
            ],
            ignore_index=True
        )

    else:

        progress = batch_df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

progress = progress.drop_duplicates(
    subset=["event_id"],
    keep="last"
)


# ============================================================
# SORT
# ============================================================

progress = progress.sort_values(
    "event_id"
).reset_index(drop=True)


# ============================================================
# SAVE FINAL FILE
# ============================================================

progress.to_csv(
    OUTPUT_FILE,
    index=False
)


# Save checkpoint too
progress.to_csv(
    TEMP_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

elapsed = (
    time.time() - start_time
) / 60

print()
print("=" * 70)
print("SATELLITE FEATURE COLLECTION COMPLETE")
print("=" * 70)

print(
    f"Total hotspots      : {len(df)}"
)

print(
    f"Satellite records   : {len(progress)}"
)

print(
    f"Satellite available : "
    f"{progress['satellite_available'].sum()}"
)

print(
    f"Unavailable          : "
    f"{(
        progress['satellite_available'] == 0
    ).sum()}"
)

print(
    f"Elapsed time         : "
    f"{elapsed:.1f} minutes"
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print(
    progress.head(10).to_string(
        index=False
    )
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)