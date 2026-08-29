import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

INPUT = r"processed\firms_india.csv"
OUTPUT = r"processed\thermal_events_v2.csv"

print("Loading FIRMS India data...")
df = pd.read_csv(INPUT)

print("Rows:", len(df))

# --------------------------------------------------
# DATE
# --------------------------------------------------

df["acq_date"] = pd.to_datetime(df["acq_date"])

# --------------------------------------------------
# IMPORTANT:
# Create spatial grid
# ~1 km grid
# --------------------------------------------------

GRID_SIZE = 0.01

df["lat_grid"] = np.floor(df["latitude"] / GRID_SIZE)
df["lon_grid"] = np.floor(df["longitude"] / GRID_SIZE)

# --------------------------------------------------
# Sort by location and time
# --------------------------------------------------

df = df.sort_values(
    ["lat_grid", "lon_grid", "acq_date"]
).reset_index(drop=True)

# --------------------------------------------------
# Create temporal event groups
#
# Same spatial grid + gap <= 7 days
# = same thermal episode
# --------------------------------------------------

print("Creating temporal events...")

df["date_gap"] = (
    df.groupby(["lat_grid", "lon_grid"])["acq_date"]
      .diff()
      .dt.days
)

df["new_event"] = (
    df["date_gap"].isna() |
    (df["date_gap"] > 7)
)

df["event_number"] = (
    df.groupby(["lat_grid", "lon_grid"])["new_event"]
      .cumsum()
)

# --------------------------------------------------
# Create unique event ID
# --------------------------------------------------

df["event_id"] = (
    df["lat_grid"].astype(str)
    + "_"
    + df["lon_grid"].astype(str)
    + "_"
    + df["event_number"].astype(str)
)

# --------------------------------------------------
# Aggregate events
# --------------------------------------------------

print("Aggregating events...")

events = (
    df.groupby("event_id")
      .agg(
          latitude=("latitude", "mean"),
          longitude=("longitude", "mean"),

          first_seen=("acq_date", "min"),
          last_seen=("acq_date", "max"),

          observation_count=("event_id", "size"),

          mean_frp=("frp", "mean"),
          max_frp=("frp", "max"),

          mean_brightness=("brightness", "mean"),
          max_brightness=("brightness", "max"),

          high_confidence_count=(
              "confidence",
              lambda x: (x == "h").sum()
          ),

          medium_confidence_count=(
              "confidence",
              lambda x: (x == "n").sum()
          ),

          low_confidence_count=(
              "confidence",
              lambda x: (x == "l").sum()
          ),

          day_observations=(
              "daynight",
              lambda x: (x == "D").sum()
          ),

          night_observations=(
              "daynight",
              lambda x: (x == "N").sum()
          ),

          satellite_count=("satellite", "nunique")
      )
      .reset_index()
)

# --------------------------------------------------
# Persistence
# --------------------------------------------------

events["persistence_days"] = (
    events["last_seen"] -
    events["first_seen"]
).dt.days + 1

# --------------------------------------------------
# Observation frequency
# --------------------------------------------------

events["observations_per_day"] = (
    events["observation_count"] /
    events["persistence_days"]
)

# --------------------------------------------------
# Persistent source flag
# --------------------------------------------------

events["persistent_source"] = (
    (events["persistence_days"] >= 14) &
    (events["observation_count"] >= 3)
)

# --------------------------------------------------
# Sort
# --------------------------------------------------

events = events.sort_values(
    ["persistence_days", "observation_count"],
    ascending=False
)

# --------------------------------------------------
# Save
# --------------------------------------------------

events.to_csv(OUTPUT, index=False)

print()
print("======================================")
print("TEMPORAL THERMAL EVENTS CREATED")
print("======================================")

print("FIRMS detections :", len(df))
print("Thermal events   :", len(events))

print(
    "Persistent sources:",
    events["persistent_source"].sum()
)

print()
print("Top 20 events:")
print(
    events[
        [
            "event_id",
            "latitude",
            "longitude",
            "first_seen",
            "last_seen",
            "persistence_days",
            "observation_count",
            "mean_frp",
            "max_frp",
            "persistent_source"
        ]
    ].head(20).to_string(index=False)
)

print()
print("Saved to:", OUTPUT)