import pandas as pd

INPUT = r"processed\thermal_events_v2.csv"
OUTPUT = r"processed\persistent_sources.csv"

print("Loading thermal events...")

df = pd.read_csv(INPUT)

print("Total events:", len(df))

# Persistent thermal sources only
persistent = df[df["persistent_source"] == True].copy()

print("Persistent sources:", len(persistent))

# Keep important columns
columns = [
    "event_id",
    "latitude",
    "longitude",
    "first_seen",
    "last_seen",
    "observation_count",
    "mean_frp",
    "max_frp",
    "mean_brightness",
    "max_brightness",
    "persistence_days",
    "observations_per_day",
    "satellite_count"
]

persistent = persistent[columns]

persistent.to_csv(OUTPUT, index=False)

print("\n================================")
print("PERSISTENT SOURCES CREATED")
print("================================")
print("Rows:", len(persistent))
print("Saved:", OUTPUT)

print("\nTOP 10:")
print(
    persistent
    .sort_values("persistence_days", ascending=False)
    .head(10)
    .to_string(index=False)
)