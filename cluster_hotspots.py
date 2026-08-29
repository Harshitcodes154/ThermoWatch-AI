import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

INPUT = r"processed\firms_india.csv"
OUTPUT = r"processed\thermal_events.csv"

print("Loading FIRMS India data...")

df = pd.read_csv(INPUT)

print("Rows loaded:", len(df))

# ---------------------------------------
# 1. Prepare coordinates
# ---------------------------------------

coords = df[["latitude", "longitude"]].to_numpy()

# ---------------------------------------
# 2. DBSCAN clustering
# ---------------------------------------
#
# eps = ~1 km approximately
# At India's latitude, 0.01 degree ≈ 1 km
#
# min_samples = 2 means at least
# 2 detections can form a cluster
#

print("Clustering thermal detections...")

clusterer = DBSCAN(
    eps=0.01,
    min_samples=2,
    metric="euclidean",
    n_jobs=-1
)

df["cluster_id"] = clusterer.fit_predict(coords)

print("Clustering complete.")

# ---------------------------------------
# 3. Remove noise
# ---------------------------------------

clusters = df[df["cluster_id"] != -1].copy()

print("Clustered detections:", len(clusters))

# ---------------------------------------
# 4. Create event-level dataset
# ---------------------------------------

print("Creating thermal events...")

clusters["acq_date"] = pd.to_datetime(clusters["acq_date"])

events = (
    clusters
    .groupby("cluster_id")
    .agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),

        first_seen=("acq_date", "min"),
        last_seen=("acq_date", "max"),

        observation_count=("cluster_id", "size"),

        mean_frp=("frp", "mean"),
        max_frp=("frp", "max"),

        mean_brightness=("brightness", "mean"),
        max_brightness=("brightness", "max"),

        mean_confidence=("confidence", lambda x:
                         (x == "h").mean()),

        day_observations=("daynight", lambda x:
                          (x == "D").sum()),

        night_observations=("daynight", lambda x:
                            (x == "N").sum()),

        satellite_count=("satellite", "nunique")
    )
    .reset_index()
)

# ---------------------------------------
# 5. Persistence
# ---------------------------------------

events["persistence_days"] = (
    events["last_seen"] - events["first_seen"]
).dt.days + 1

# ---------------------------------------
# 6. Sort strongest/persistent events
# ---------------------------------------

events = events.sort_values(
    ["persistence_days", "max_frp"],
    ascending=False
)

# ---------------------------------------
# 7. Create readable Event ID
# ---------------------------------------

events.insert(
    0,
    "event_id",
    ["THERMAL_" + str(i).zfill(6)
     for i in range(1, len(events) + 1)]
)

# ---------------------------------------
# 8. Save
# ---------------------------------------

events.to_csv(OUTPUT, index=False)

print()
print("===================================")
print("THERMAL EVENT DATASET CREATED")
print("===================================")

print("Total FIRMS detections :", len(df))
print("Clustered detections   :", len(clusters))
print("Thermal events         :", len(events))

print()
print("Saved to:", OUTPUT)

print()
print("TOP 10 EVENTS:")
print(
    events[
        [
            "event_id",
            "latitude",
            "longitude",
            "persistence_days",
            "observation_count",
            "mean_frp",
            "max_frp"
        ]
    ].head(10).to_string(index=False)
)