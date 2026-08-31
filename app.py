import io
import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# PDF Report Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI - Satellite Risk Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION & REPOSITORIES
# ============================================================

HF_MODEL_URL = "https://huggingface.co/harshitcodes1544/sihharshit154/resolve/main/source_classifier_v5.joblib"
HF_OSM_URL = "https://huggingface.co/datasets/harshitcodes1544/thermowatch-osm-data/resolve/main/osm_facilities.csv"

FIRMS_API_KEY = st.secrets.get("FIRMS_API_KEY", os.environ.get("FIRMS_API_KEY", ""))
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
INDIA_BBOX = "68.0,6.0,97.5,37.5"
INDIA_CENTER = [22.9734, 78.6569]

# 100% Free Tile Layer (No API Key Required)
FREE_TILE_URL = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
FREE_TILE_ATTR = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>"

# ============================================================
# CACHED LOADERS (MODEL & OSM)
# ============================================================

@st.cache_resource
def load_model():
    response = requests.get(HF_MODEL_URL, timeout=120, allow_redirects=True)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

@st.cache_data(ttl=86400)
def load_osm_facilities():
    response = requests.get(HF_OSM_URL, timeout=180, allow_redirects=True)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))

# ============================================================
# NASA FIRMS INGESTION (SAFE, MULTI-SENSOR)
# ============================================================

@st.cache_data(ttl=1800)
def fetch_live_firms(day_range=1):
    if not FIRMS_API_KEY:
        return pd.DataFrame()

    sensors = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA21_NRT"]
    df = pd.DataFrame()

    for sensor in sensors:
        url = f"{FIRMS_URL}{FIRMS_API_KEY}/{sensor}/{INDIA_BBOX}/{day_range}"
        try:
            res = requests.get(url, timeout=120)
            if res.status_code == 200 and "latitude" in res.text:
                temp = pd.read_csv(io.StringIO(res.text))
                if not temp.empty:
                    df = temp
                    break
        except Exception:
            continue

    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "latitude": "latitude", "longitude": "longitude", "bright_ti4": "bright_ti4",
        "scan": "scan", "track": "track", "acq_date": "acq_date", "acq_time": "acq_time",
        "satellite": "satellite", "instrument": "instrument", "confidence": "confidence",
        "version": "version", "bright_ti5": "bright_ti5", "frp": "frp", "daynight": "daynight"
    }
    df = df.rename(columns=rename_map)

    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
        df["acquisition_datetime"] = pd.to_datetime(
            df["acq_date"].astype(str) + " " + df["acq_time"].str[:2] + ":" + df["acq_time"].str[2:],
            errors="coerce"
        )

    for col in ["latitude", "longitude", "bright_ti4", "bright_ti5", "frp", "scan", "track"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df["hotspot_id"] = [f"LIVE_{i:06d}" for i in range(1, len(df) + 1)]
    df["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    return df

# ============================================================
# CLUSTERING & FACILITY ATTRIBUTION
# ============================================================

def apply_spatial_clustering(df):
    if len(df) < 2:
        df["cluster_id"] = "CLUSTER_000"
        return df

    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    # 5 km spatial threshold
    db = DBSCAN(eps=5.0 / 6371.0, min_samples=2, metric='haversine')
    clusters = db.fit_predict(coords_rad)
    df["cluster_id"] = [f"CLUSTER_{c:03d}" if c != -1 else "ISOLATED" for c in clusters]
    return df

def attribute_facilities(live, facilities):
    if live.empty:
        return live

    live = live.copy()
    facilities = facilities.copy()

    for d in [live, facilities]:
        d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
        d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")

    live = live.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    facilities = facilities.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    if facilities.empty:
        live["distance_to_facility_km"] = 999.0
        live["proximity_category"] = "FAR"
        live["facility_match_quality"] = "VERY_LOW"
        return live

    facility_coords = np.radians(facilities[["latitude", "longitude"]].values)
    live_coords = np.radians(live[["latitude", "longitude"]].values)

    tree = BallTree(facility_coords, metric="haversine")
    distances, indices = tree.query(live_coords, k=1)

    distances_km = distances[:, 0] * 6371.0088
    nearest = facilities.iloc[indices[:, 0]].reset_index(drop=True)

    result = live.copy()
    fields = {
        "facility_osm_id": "osm_id", "facility_type": "feature_type", "facility_name": "name",
        "facility_power": "power", "facility_industrial": "industrial", "facility_landuse": "landuse",
        "facility_operator": "operator", "facility_plant_source": "plant_source", "facility_plant_method": "plant_method"
    }
    for out_col, in_col in fields.items():
        result[out_col] = nearest[in_col].values if in_col in nearest.columns else np.nan

    result["facility_latitude"] = nearest["latitude"].values
    result["facility_longitude"] = nearest["longitude"].values
    result["distance_to_facility_km"] = distances_km

    result["proximity_category"] = result["distance_to_facility_km"].apply(
        lambda d: "VERY_CLOSE" if d <= 1 else ("CLOSE" if d <= 2 else ("NEAR" if d <= 5 else ("DISTANT" if d <= 10 else "FAR")))
    )
    result["facility_match_quality"] = result["distance_to_facility_km"].apply(
        lambda d: "HIGH" if d <= 1 else ("MEDIUM" if d <= 5 else ("LOW" if d <= 10 else "VERY_LOW"))
    )
    return result

# ============================================================
# WEATHER & DISPERSION ENGINE (OPEN-METEO)
# ============================================================

@st.cache_data(ttl=1800)
def fetch_weather_vector(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m,wind_direction_10m,relative_humidity_2m"
        res = requests.get(url, timeout=5).json()
        current = res.get("current", {})
        speed = current.get("wind_speed_10m", 0)
        direction = current.get("wind_direction_10m", 0)
        humidity = current.get("relative_humidity_2m", 50)
        return speed, direction, humidity
    except Exception:
        return 0.0, 0, 50

def get_compass_bearing(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25) / 22.5) % 16
    return dirs[ix]

# ============================================================
# FEATURE ENGINEERING & MODEL PREDICTION
# ============================================================

FEATURES = [
    "mean_frp", "max_frp", "frp_ratio", "mean_brightness", "max_brightness", "brightness_range",
    "observation_count", "persistence_days", "observations_per_day", "activity_density", "satellite_count",
    "distance_to_facility_km", "facility_within_1km", "facility_within_5km",
    "persistent_long_term", "persistent_180_days", "facility_type", "facility_power", "facility_industrial", "facility_landuse",
    "frp_ratio_v5", "frp_excess_v5", "frp_log_v5", "max_frp_log_v5", "brightness_range_v5", "brightness_ratio_v5",
    "brightness_excess_v5", "persistence_log_v5", "persistence_months_v5", "persistent_30d_v5", "persistent_90d_v5",
    "persistent_180d_v5", "persistent_270d_v5", "observation_log_v5", "obs_per_persistence_v5", "activity_persistence_v5",
    "distance_log_v5", "very_close_v5", "within_2km_v5", "within_5km_v5", "within_10km_v5",
    "frp_persistence_v5", "frp_distance_signal_v5", "activity_distance_signal_v5"
]

def engineer_features(df):
    df = df.copy()
    df["mean_frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df["max_frp"] = df["mean_frp"]
    df["frp_ratio"] = df["max_frp"] / (df["mean_frp"] + 1e-6)
    df["mean_brightness"] = pd.to_numeric(df.get("bright_ti4", 0), errors="coerce").fillna(0)
    df["max_brightness"] = df["mean_brightness"]
    df["brightness_range"] = 0.0
    df["observation_count"] = 1
    df["persistence_days"] = 1
    df["observations_per_day"] = 1.0
    df["activity_density"] = 1.0
    df["satellite_count"] = 1
    df["distance_to_facility_km"] = pd.to_numeric(df.get("distance_to_facility_km", 999.0), errors="coerce").fillna(999.0)
    df["facility_within_1km"] = (df["distance_to_facility_km"] <= 1).astype(int)
    df["facility_within_5km"] = (df["distance_to_facility_km"] <= 5).astype(int)
    df["persistent_long_term"] = 0
    df["persistent_180_days"] = 0

    for col in ["facility_type", "facility_power", "facility_industrial", "facility_landuse"]:
        df[col] = df[col].fillna("unknown").astype(str) if col in df.columns else "unknown"

    df["frp_ratio_v5"] = df["max_frp"] / (df["mean_frp"] + 1e-6)
    df["frp_excess_v5"] = np.maximum(df["max_frp"] - 5.0, 0)
    df["frp_log_v5"] = np.log1p(df["mean_frp"])
    df["max_frp_log_v5"] = np.log1p(df["max_frp"])
    df["brightness_range_v5"] = 0.0
    df["brightness_ratio_v5"] = 1.0
    df["brightness_excess_v5"] = 0.0
    df["persistence_log_v5"] = np.log1p(df["persistence_days"])
    df["persistence_months_v5"] = df["persistence_days"] / 30.0
    df["persistent_30d_v5"] = (df["persistence_days"] >= 30).astype(int)
    df["persistent_90d_v5"] = (df["persistence_days"] >= 90).astype(int)
    df["persistent_180d_v5"] = (df["persistence_days"] >= 180).astype(int)
    df["persistent_270d_v5"] = (df["persistence_days"] >= 270).astype(int)
    df["observation_log_v5"] = np.log1p(df["observation_count"])
    df["obs_per_persistence_v5"] = df["observation_count"] / (df["persistence_days"] + 1e-6)
    df["activity_persistence_v5"] = df["activity_density"] * df["persistence_days"]
    df["distance_log_v5"] = np.log1p(df["distance_to_facility_km"])
    df["very_close_v5"] = (df["distance_to_facility_km"] <= 1).astype(int)
    df["within_2km_v5"] = (df["distance_to_facility_km"] <= 2).astype(int)
    df["within_5km_v5"] = (df["distance_to_facility_km"] <= 5).astype(int)
    df["within_10km_v5"] = (df["distance_to_facility_km"] <= 10).astype(int)
    df["frp_persistence_v5"] = df["mean_frp"] * df["persistence_days"]
    df["frp_distance_signal_v5"] = df["mean_frp"] / (df["distance_to_facility_km"] + 1)
    df["activity_distance_signal_v5"] = df["activity_density"] / (df["distance_to_facility_km"] + 1)
    return df

def run_v5_prediction(df, model):
    if df.empty:
        return df

    df = engineer_features(df)
    X = df[FEATURES].copy()

    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for cat in ["facility_type", "facility_power", "facility_industrial", "facility_landuse"]:
        X[cat] = X[cat].fillna("unknown").astype(str)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    conf = np.max(probs, axis=1) * 100

    df["predicted_source"] = preds
    df["confidence"] = conf.round(1)

    frp_c = np.clip(df["mean_frp"] * 3, 0, 35)
    mfrp_c = np.clip(df["max_frp"] * 0.5, 0, 20)
    conf_c = conf * 0.20
    prox_c = np.where(df["distance_to_facility_km"] <= 1.0, 25, np.where(df["distance_to_facility_km"] <= 5.0, 15, 0))

    risk = np.clip(frp_c + mfrp_c + conf_c + prox_c, 0, 100)
    df["live_risk_score"] = risk.round(2)
    df["live_risk_category"] = df["live_risk_score"].apply(
        lambda s: "CRITICAL" if s >= 75 else ("HIGH" if s >= 50 else ("MEDIUM" if s >= 25 else "LOW"))
    )
    return df

# ============================================================
# PDF REPORT BUILDER
# ============================================================

def generate_pdf_report(df, time_window_label):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#b91c1c"))
    story.append(Paragraph("ThermoWatch AI — Satellite Thermal Threat Audit", title_style))
    story.append(Paragraph(f"Window: {time_window_label} | Generated on: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles['Normal']))
    story.append(Spacer(1, 15))

    summary_text = (
        f"<b>Total Hotspots:</b> {len(df)} | "
        f"<b>Critical:</b> {(df['live_risk_category'] == 'CRITICAL').sum()} | "
        f"<b>High Risk:</b> {(df['live_risk_category'] == 'HIGH').sum()} | "
        f"<b>Average Risk Score:</b> {df['live_risk_score'].mean():.2f}/100"
    )
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 15))

    table_data = [["ID", "Lat/Lon", "Source", "Conf.", "Facility", "Dist(km)", "Risk"]]
    top_15 = df.sort_values("live_risk_score", ascending=False).head(15)

    for _, row in top_15.iterrows():
        table_data.append([
            str(row["hotspot_id"]),
            f"{row['latitude']:.2f}, {row['longitude']:.2f}",
            str(row["predicted_source"]),
            f"{row['confidence']}%",
            str(row.get("facility_name", "N/A"))[:15],
            f"{row['distance_to_facility_km']:.1f}",
            str(row["live_risk_category"])
        ])

    t = Table(table_data, colWidths=[65, 80, 80, 50, 125, 55, 65])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("🛡️ ThermoWatch Controls")

time_window_options = {
    "Last 24 Hours (1 Day)": 1,
    "Last 48 Hours (2 Days)": 2,
    "Last 72 Hours (3 Days)": 3,
    "Last 96 Hours (4 Days)": 4
}

selected_window_label = st.sidebar.selectbox(
    "Select Observation Time Window:",
    options=list(time_window_options.keys()),
    index=0
)
lookback_days = time_window_options[selected_window_label]

min_conf = st.sidebar.slider("Filter AI Confidence (%)", min_value=0, max_value=100, value=0)

if st.sidebar.button("🔄 Sync Satellite Feeds"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("Satellite: VIIRS NOAA-20 / SNPP / NOAA-21 (NASA FIRMS NRT Feed)")

# ============================================================
# PIPELINE EXECUTION
# ============================================================

st.title("🔥 ThermoWatch AI — Planetary Thermal Intelligence")
st.caption(f"Realtime Satellite Thermal Detection over India Region | Active Window: **{selected_window_label}**")

try:
    with st.spinner("Synchronizing AI weights & NASA FIRMS telemetry..."):
        model = load_model()
        facilities = load_osm_facilities()
        raw_firms = fetch_live_firms(day_range=lookback_days)

    if not raw_firms.empty:
        clustered = apply_spatial_clustering(raw_firms)
        attributed = attribute_facilities(clustered, facilities)
        predictions = run_v5_prediction(attributed, model)
        if min_conf > 0:
            predictions = predictions[predictions["confidence"] >= min_conf].reset_index(drop=True)
    else:
        predictions = pd.DataFrame()

except Exception as e:
    st.error("System encountered an unexpected exception while loading resources.")
    st.exception(e)
    st.stop()

# ============================================================
# GRACEFUL NO-DETECTION STATE (NO CRASH)
# ============================================================

if predictions.empty:
    st.info(
        f"ℹ️ **No Thermal Anomalies Found in {selected_window_label}**\n\n"
        f"NASA FIRMS reported **0 active fire/thermal observations** across the Indian airspace for this time window.\n\n"
        f"👉 **Tip:** To view previous activity or seasonal/stubble burns, select **48 Hours**, **72 Hours**, or **96 Hours** from the sidebar."
    )
    
    # 100% Free OpenStreetMap via Carto tiles
    m_empty = folium.Map(
        location=INDIA_CENTER, 
        zoom_start=5, 
        tiles=FREE_TILE_URL, 
        attr=FREE_TILE_ATTR
    )
    st_folium(m_empty, width="100%", height=450)
    st.stop()

# ============================================================
# EXECUTIVE METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hotspots Detected", len(predictions))
c2.metric("Spatial Clusters", predictions["cluster_id"].nunique())
c3.metric("Critical Hazards", int((predictions["live_risk_category"] == "CRITICAL").sum()))
c4.metric("High Risk Fires", int((predictions["live_risk_category"] == "HIGH").sum()))
c5.metric("Avg Fleet Risk", f"{predictions['live_risk_score'].mean():.1f} / 100")

# ============================================================
# INTERACTIVE INDIA HEATMAP & SPATIAL CLUSTERS
# ============================================================

st.subheader("🗺️ India Thermal Anomaly & HeatMap")

map_center = [predictions["latitude"].median(), predictions["longitude"].median()]

# Free Base Map Layer (No API Key Required)
m = folium.Map(
    location=map_center, 
    zoom_start=5, 
    tiles=FREE_TILE_URL, 
    attr=FREE_TILE_ATTR
)

# 1. Density HeatMap Layer
heat_data = [[row["latitude"], row["longitude"], float(row["frp"])] for _, row in predictions.iterrows()]
HeatMap(heat_data, radius=18, blur=20, min_opacity=0.4, max_zoom=10).add_to(m)

# 2. Risk Marker Points & Danger Buffers
risk_colors = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#eab308",
    "LOW": "#22c55e"
}

for _, row in predictions.head(200).iterrows():
    color = risk_colors.get(row["live_risk_category"], "#3b82f6")
    popup_html = f"""
    <div style='font-family:sans-serif; width:220px;'>
        <h4 style='margin:0; color:{color};'>{row['hotspot_id']} ({row['live_risk_category']})</h4>
        <b>Source:</b> {row['predicted_source']} ({row['confidence']}%)<br>
        <b>FRP:</b> {row['frp']} MW<br>
        <b>Facility:</b> {row.get('facility_name', 'Unknown')}<br>
        <b>Distance:</b> {row['distance_to_facility_km']:.2f} km<br>
        <b>Cluster:</b> {row['cluster_id']}
    </div>
    """
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5 + min((row["frp"] / 40.0), 12),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(m)

    # Highlight danger buffer around facility if <= 2km
    if row["distance_to_facility_km"] <= 2.0:
        folium.Circle(
            location=[row["facility_latitude"], row["facility_longitude"]],
            radius=1500,
            color="#dc2626",
            weight=1.5,
            fill=True,
            fill_opacity=0.15,
            tooltip=f"Industrial Zone: {row.get('facility_name', 'Facility')}"
        ).add_to(m)

st_folium(m, width="100%", height=530)

# ============================================================
# WEATHER & DISPERSION (TOP PRIORITY THREAT)
# ============================================================

top_threat = predictions.sort_values("live_risk_score", ascending=False).iloc[0]
w_speed, w_dir, w_humidity = fetch_weather_vector(top_threat["latitude"], top_threat["longitude"])
cardinal = get_compass_bearing(w_dir)

st.subheader("🌪️ Realtime Dispersion & Wind Vectors")
wc1, wc2, wc3, wc4 = st.columns(4)
wc1.info(f"📍 **Target Focus:** `{top_threat['hotspot_id']}` ({top_threat.get('facility_name', 'Open Area')})")
wc2.metric("Surface Wind Speed", f"{w_speed} km/h")
wc3.metric("Wind Direction", f"{w_dir}° ({cardinal})")
wc4.metric("Ambient Humidity", f"{w_humidity}%")
st.caption(f"⚠️ **Dispersion Trajectory:** Smoke & aerosol plumes are propagating towards the **{cardinal}** sector at **{w_speed} km/h**.")

# ============================================================
# CHARTS & ANALYTICS
# ============================================================

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("📊 AI Classification Breakdown")
    st.bar_chart(predictions["predicted_source"].value_counts())

with col_r:
    st.subheader("🚨 Risk Severity Levels")
    st.bar_chart(predictions["live_risk_category"].value_counts())

# ============================================================
# PRIORITY AUDIT QUEUE & EXPORT OPTIONS
# ============================================================

st.subheader("📋 Priority Hazard Audit Queue")
display_cols = [
    "hotspot_id", "cluster_id", "facility_name", "predicted_source",
    "confidence", "frp", "distance_to_facility_km", "live_risk_score", "live_risk_category"
]
st.dataframe(
    predictions.sort_values("live_risk_score", ascending=False)[display_cols].head(25),
    use_container_width=True,
    hide_index=True
)

st.subheader("📥 Export Intelligence Reports")
d1, d2 = st.columns(2)
with d1:
    csv_bytes = predictions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Telemetry CSV",
        data=csv_bytes,
        file_name=f"thermowatch_{lookback_days}d_telemetry.csv",
        mime="text/csv",
        use_container_width=True
    )
with d2:
    pdf_bytes = generate_pdf_report(predictions, selected_window_label)
    st.download_button(
        label="📑 Download Incident Audit PDF",
        data=pdf_bytes,
        file_name=f"thermowatch_{lookback_days}d_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.divider()
st.caption("ThermoWatch AI | Defense-Grade Environmental & Industrial Threat Intelligence")
