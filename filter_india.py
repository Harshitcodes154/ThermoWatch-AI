import pandas as pd
import geopandas as gpd

# -----------------------------------------
# PATHS
# -----------------------------------------

FIRMS_FILE = r"processed\viirs_unified.csv"
BOUNDARY_FILE = r"india_boundary\ne_10m_admin_0_countries.shp"
OUTPUT_FILE = r"processed\firms_india.csv"

# -----------------------------------------
# 1. LOAD FIRMS DATA
# -----------------------------------------

print("Loading FIRMS data...")

df = pd.read_csv(FIRMS_FILE)

print("Total FIRMS rows:", len(df))

# -----------------------------------------
# 2. LOAD COUNTRY BOUNDARIES
# -----------------------------------------

print("Loading country boundaries...")

countries = gpd.read_file(BOUNDARY_FILE)

print("Boundary CRS:", countries.crs)

# -----------------------------------------
# 3. SELECT INDIA
# -----------------------------------------

india = countries[countries["NAME"] == "India"].copy()

if india.empty:
    print("ERROR: India boundary not found!")
    exit()

print("India boundary loaded.")

# -----------------------------------------
# 4. CONVERT FIRMS POINTS TO GEODATA
# -----------------------------------------

print("Creating geographic points...")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"]
    ),
    crs="EPSG:4326"
)

# -----------------------------------------
# 5. FILTER POINTS INSIDE INDIA
# -----------------------------------------

print("Filtering FIRMS points inside India...")

india_points = gpd.sjoin(
    gdf,
    india[["geometry"]],
    predicate="within",
    how="inner"
)

# -----------------------------------------
# 6. REMOVE EXTRA GEOMETRY COLUMNS
# -----------------------------------------

india_points = india_points.drop(
    columns=["geometry", "index_right"],
    errors="ignore"
)

# -----------------------------------------
# 7. SAVE
# -----------------------------------------

india_points.to_csv(
    OUTPUT_FILE,
    index=False
)

# -----------------------------------------
# 8. SUMMARY
# -----------------------------------------

print()
print("===================================")
print("INDIA FIRMS DATASET CREATED")
print("===================================")

print("Original rows :", len(df))
print("India rows    :", len(india_points))

print(
    "Date range    :",
    india_points["acq_date"].min(),
    "to",
    india_points["acq_date"].max()
)

print(
    "Latitude range:",
    india_points["latitude"].min(),
    "to",
    india_points["latitude"].max()
)

print(
    "Longitude range:",
    india_points["longitude"].min(),
    "to",
    india_points["longitude"].max()
)

print()
print("Saved to:", OUTPUT_FILE)