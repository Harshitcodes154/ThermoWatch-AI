import os
import time
import requests
import pandas as pd
import numpy as np

print("=" * 60)
print("THERMO WATCH - SATELLITE FEATURE TEST")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "satellite_features_test.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total hotspots:", len(df))

# Test only first 10
test_df = df.head(10).copy()

print("Test hotspots:", len(test_df))

# ============================================================
# SENTINEL-2 FEATURE SOURCE
# ============================================================
#
# We use Microsoft Planetary Computer STAC API.
# This searches Sentinel-2 L2A scenes around each hotspot.
#
# No large image download yet.
# We only test whether suitable satellite scenes exist.
# ============================================================

STAC_URL = (
    "https://planetarycomputer.microsoft.com/"
    "api/stac/v1/search"
)

# ============================================================
# FUNCTION: FIND SENTINEL-2 SCENES
# ============================================================

def find_sentinel_scenes(lat, lon):

    # Approx. 5 km search box
    delta = 0.05

    bbox = [
        lon - delta,
        lat - delta,
        lon + delta,
        lat + delta
    ]

    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": "2025-08-01T00:00:00Z/"
                   "2026-05-31T23:59:59Z",
        "limit": 5
    }

    try:

        response = requests.post(
            STAC_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        features = data.get(
            "features",
            []
        )

        return features

    except Exception as e:

        print(
            "Satellite search error:",
            str(e)
        )

        return []


# ============================================================
# PROCESS HOTSPOTS
# ============================================================

results = []

for index, row in test_df.iterrows():

    lat = row["latitude"]
    lon = row["longitude"]

    print("\n" + "-" * 60)

    print(
        f"[{len(results)+1}/{len(test_df)}] "
        f"Searching: {lat:.6f}, {lon:.6f}"
    )

    scenes = find_sentinel_scenes(
        lat,
        lon
    )

    # --------------------------------------------------------
    # Basic satellite information
    # --------------------------------------------------------

    if scenes:

        scene = scenes[0]

        properties = scene.get(
            "properties",
            {}
        )

        cloud_cover = properties.get(
            "eo:cloud_cover",
            np.nan
        )

        scene_date = properties.get(
            "datetime",
            None
        )

        satellite_available = 1

        scene_count = len(scenes)

        print(
            "Satellite scenes found:",
            scene_count
        )

        print(
            "Cloud cover:",
            cloud_cover
        )

        print(
            "Latest/test scene:",
            scene_date
        )

    else:

        satellite_available = 0
        scene_count = 0
        cloud_cover = np.nan
        scene_date = None

        print(
            "No Sentinel-2 scene found."
        )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "event_id":
            row["event_id"],

        "latitude":
            lat,

        "longitude":
            lon,

        "ai_label":
            row.get(
                "ai_label",
                None
            ),

        "risk_score":
            row.get(
                "risk_score",
                None
            ),

        "source_type":
            row.get(
                "source_type",
                None
            ),

        "persistence_days":
            row.get(
                "persistence_days",
                None
            ),

        "mean_frp":
            row.get(
                "mean_frp",
                None
            ),

        "max_frp":
            row.get(
                "max_frp",
                None
            ),

        "distance_to_facility_km":
            row.get(
                "distance_to_facility_km",
                None
            ),

        "satellite_available":
            satellite_available,

        "sentinel_scene_count":
            scene_count,

        "cloud_cover":
            cloud_cover,

        "satellite_scene_date":
            scene_date
    })

    time.sleep(1)


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 60)
print("SATELLITE TEST COMPLETE")
print("=" * 60)

print(
    "Test hotspots:",
    len(result_df)
)

print(
    "Satellite available:",
    result_df[
        "satellite_available"
    ].sum()
)

print(
    "Satellite unavailable:",
    (
        result_df[
            "satellite_available"
        ] == 0
    ).sum()
)

print("\nRESULT:")
print(
    result_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 60)
print("DONE")
print("=" * 60)