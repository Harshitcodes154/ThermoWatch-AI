import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("THERMO WATCH - RISK ANALYSIS")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "processed" / "risk_features.csv"
OUTPUT_DIR = BASE_DIR / "processed"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

print("\nLoading risk features...")

df = pd.read_csv(INPUT_FILE)

print(f"Total hotspots: {len(df)}")

# ---------------------------------------------------------
# BASIC ANALYSIS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RISK DISTRIBUTION")
print("=" * 60)

print(
    df["risk_category"]
    .value_counts()
)

print("\nPercentage:")

print(
    (df["risk_category"].value_counts(normalize=True) * 100)
    .round(2)
)

# ---------------------------------------------------------
# SOURCE TYPE ANALYSIS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SOURCE TYPE DISTRIBUTION")
print("=" * 60)

print(
    df["source_type"]
    .value_counts()
)

# ---------------------------------------------------------
# HIGH RISK
# ---------------------------------------------------------

high_risk = df[
    df["risk_category"].isin(["HIGH", "CRITICAL"])
].copy()

print("\n" + "=" * 60)
print("HIGH / CRITICAL HOTSPOTS")
print("=" * 60)

print(f"Count: {len(high_risk)}")

# ---------------------------------------------------------
# TOP 50
# ---------------------------------------------------------

top50 = (
    df.sort_values(
        "risk_score",
        ascending=False
    )
    .head(50)
)

top50_file = OUTPUT_DIR / "top_50_risk_hotspots.csv"

top50.to_csv(
    top50_file,
    index=False
)

print(f"\nSaved TOP 50:")
print(top50_file)

# ---------------------------------------------------------
# SOURCE RISK ANALYSIS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("AVERAGE RISK BY SOURCE TYPE")
print("=" * 60)

source_risk = (
    df.groupby("source_type")
    .agg(
        hotspot_count=("event_id", "count"),
        average_risk=("risk_score", "mean"),
        maximum_risk=("risk_score", "max"),
        average_frp=("mean_frp", "mean"),
        average_persistence=("persistence_days", "mean"),
    )
    .sort_values(
        "average_risk",
        ascending=False
    )
)

print(source_risk.round(2).to_string())

source_file = OUTPUT_DIR / "risk_by_source_type.csv"

source_risk.to_csv(source_file)

print("\nSaved:")
print(source_file)

# ---------------------------------------------------------
# PERSISTENT SOURCES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MOST PERSISTENT HOTSPOTS")
print("=" * 60)

persistent = (
    df.sort_values(
        "persistence_days",
        ascending=False
    )
    .head(20)
)

print(
    persistent[
        [
            "event_id",
            "latitude",
            "longitude",
            "persistence_days",
            "observation_count",
            "mean_frp",
            "risk_score",
            "risk_category",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# HIGHEST FRP
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HIGHEST FRP HOTSPOTS")
print("=" * 60)

highest_frp = (
    df.sort_values(
        "max_frp",
        ascending=False
    )
    .head(20)
)

print(
    highest_frp[
        [
            "event_id",
            "latitude",
            "longitude",
            "mean_frp",
            "max_frp",
            "persistence_days",
            "source_type",
            "risk_score",
            "risk_category",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# VERY CLOSE FACILITIES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HOTSPOTS VERY CLOSE TO FACILITIES")
print("=" * 60)

very_close = df[
    df["distance_to_facility_km"] <= 1
].copy()

print(
    f"Within 1 km: {len(very_close)}"
)

# ---------------------------------------------------------
# CRITICAL CANDIDATES
# ---------------------------------------------------------

critical_candidates = df[
    (
        df["risk_score"] >= 60
    )
    |
    (
        (df["max_frp"] >= 30)
        &
        (df["persistence_days"] >= 30)
    )
].copy()

critical_candidates = critical_candidates.sort_values(
    "risk_score",
    ascending=False
)

critical_file = (
    OUTPUT_DIR /
    "critical_hotspot_candidates.csv"
)

critical_candidates.to_csv(
    critical_file,
    index=False
)

print(
    f"\nCritical candidates: "
    f"{len(critical_candidates)}"
)

print("Saved:")
print(critical_file)

# ---------------------------------------------------------
# DASHBOARD DATA
# ---------------------------------------------------------

dashboard_columns = [
    "event_id",
    "latitude",
    "longitude",
    "first_seen",
    "last_seen",
    "mean_frp",
    "max_frp",
    "persistence_days",
    "observations_per_day",
    "distance_to_facility_km",
    "source_type",
    "source_confidence",
    "source_confidence_score",
    "risk_score",
    "risk_category",
]

dashboard_columns = [
    col
    for col in dashboard_columns
    if col in df.columns
]

dashboard = df[dashboard_columns].copy()

dashboard_file = (
    OUTPUT_DIR /
    "dashboard_hotspots.csv"
)

dashboard.to_csv(
    dashboard_file,
    index=False
)

print("\nDashboard dataset saved:")
print(dashboard_file)

# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("THERMO WATCH RISK ANALYSIS COMPLETE")
print("=" * 60)

print(f"""
Total hotspots       : {len(df)}
HIGH                 : {(df["risk_category"] == "HIGH").sum()}
CRITICAL             : {(df["risk_category"] == "CRITICAL").sum()}
MEDIUM               : {(df["risk_category"] == "MEDIUM").sum()}
LOW                  : {(df["risk_category"] == "LOW").sum()}

Within 1 km facility : {len(very_close)}

Highest risk score   : {df["risk_score"].max():.2f}
Average risk score   : {df["risk_score"].mean():.2f}
""")

print("Output files:")
print("1. top_50_risk_hotspots.csv")
print("2. risk_by_source_type.csv")
print("3. critical_hotspot_candidates.csv")
print("4. dashboard_hotspots.csv")

print("\n" + "=" * 60)