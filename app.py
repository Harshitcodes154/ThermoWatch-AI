
import streamlit as st
import pandas as pd
import numpy as np
import folium
from pathlib import Path
from datetime import datetime, timezone, timedelta
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

# ============================================================
# THERMOWATCH — PROFESSIONAL LIVE THERMAL INTELLIGENCE
# IMPORTANT: Dynamic UI uses Streamlit native components.
# This prevents HTML source from appearing on the page.
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

FINAL_FILE = PROCESSED_DIR / "live_final_predictions.csv"
V5_FILE = PROCESSED_DIR / "live_v5_predictions.csv"
FACILITY_FILE = PROCESSED_DIR / "live_facility_matches.csv"
FIRMS_FILE = PROCESSED_DIR / "live_firms_hotspots.csv"

# ============================================================
# CSS ONLY — NO DYNAMIC HTML IS USED BELOW
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 50% -20%, rgba(18,42,68,.32), transparent 42%),
      #05080d;
    color:#e7edf5;
}
header[data-testid="stHeader"] { background:#05080d; }
.block-container {
    padding-top:.7rem;
    padding-bottom:2rem;
    max-width:1600px;
}
[data-testid="stMetric"] {
    background:linear-gradient(180deg,#0c1520,#09111a);
    border:1px solid #1d3045;
    border-radius:9px;
    padding:10px 13px;
}
[data-testid="stMetricLabel"] {
    color:#75879b !important;
    font-size:9px !important;
    text-transform:uppercase;
    letter-spacing:1px;
}
[data-testid="stMetricValue"] {
    color:#f3f7fb !important;
    font-size:23px !important;
    font-weight:800 !important;
}
.tw-title {
    font-size:25px;
    font-weight:850;
    letter-spacing:.2px;
}
.tw-sub {
    color:#718096;
    font-size:10px;
    font-family:monospace;
    letter-spacing:.4px;
}
.tw-section {
    font-size:18px;
    font-weight:850;
    margin-top:4px;
}
.tw-caption {
    color:#718096;
    font-size:10px;
    font-family:monospace;
}
.alert-critical {
    border-left:3px solid #ff405d;
}
.alert-high {
    border-left:3px solid #ff9f32;
}
.alert-medium {
    border-left:3px solid #ffd23f;
}
.alert-low {
    border-left:3px solid #29a9ff;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:#1d3045 !important;
    background:rgba(9,17,26,.72);
}
.inspector-value {
    font-size:15px;
    font-weight:800;
}
.small-muted {
    color:#8fa1b5;
    font-size:11px;
}
[data-testid="stDataFrame"] {
    border:1px solid #1a2b3d;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def clean_text(value, fallback=""):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return fallback
    return text

def num(row, col, default=0.0):
    try:
        value = row.get(col, default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def confidence_level(value):
    v = num({"x": value}, "x")
    if v >= 80:
        return "HIGH"
    if v >= 60:
        return "MEDIUM"
    return "LOW"

def facility_label(row):
    name = clean_text(row.get("facility_name"), "")
    if name:
        return name

    ftype = clean_text(row.get("facility_type"), "").lower()
    power = clean_text(row.get("facility_power"), "").lower()
    industrial = clean_text(row.get("facility_industrial"), "").lower()

    if "substation" in ftype or "substation" in power:
        return "Unnamed Power Substation"
    if "generator" in ftype or "generator" in power or "plant" in power:
        return "Unnamed Power Facility"
    if industrial or "industrial" in ftype:
        return "Unnamed Industrial Facility"
    return "Unidentified Facility"

def risk_color(category):
    c = clean_text(category, "LOW").upper()
    return {
        "CRITICAL": "#ff405d",
        "HIGH": "#ff9f32",
        "MEDIUM": "#ffd23f",
        "LOW": "#29a9ff",
    }.get(c, "#29a9ff")

def risk_class(category):
    return clean_text(category, "LOW").lower()

def format_observed(value):
    if value is None or str(value).strip() == "":
        return "Unavailable"
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return str(value)
    return dt.strftime("%d %b %Y · %H:%M UTC")

def data_age(value):
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return "Unknown"
    now = pd.Timestamp.now(tz="UTC")
    hours = max(0, (now - dt).total_seconds() / 3600)
    if hours < 1:
        return f"{int(hours*60)} min ago"
    if hours < 24:
        return f"{hours:.1f} hours ago"
    return f"{hours/24:.1f} days ago"

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=60)
def load_data():
    candidates = [FINAL_FILE, V5_FILE, FACILITY_FILE]
    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        return pd.DataFrame(), None

    df = pd.read_csv(path)

    # Merge facility information when final file lacks it.
    if FACILITY_FILE.exists() and "hotspot_id" in df.columns:
        try:
            fac = pd.read_csv(FACILITY_FILE)
            if "hotspot_id" in fac.columns:
                useful = [
                    c for c in fac.columns
                    if c not in df.columns or c in [
                        "hotspot_id",
                        "facility_name",
                        "facility_type",
                        "facility_power",
                        "facility_industrial",
                        "distance_to_facility_km",
                        "facility_match_quality",
                    ]
                ]
                useful = list(dict.fromkeys(["hotspot_id"] + useful))
                df = df.merge(
                    fac[useful],
                    on="hotspot_id",
                    how="left",
                    suffixes=("", "_facility"),
                )
        except Exception:
            pass

    # Normalize required columns.
    defaults_numeric = [
        "latitude", "longitude", "frp", "confidence",
        "live_risk_score", "distance_to_facility_km"
    ]
    defaults_text = [
        "facility_name", "facility_type", "facility_power",
        "facility_industrial", "predicted_source",
        "confidence_level", "attribution_quality",
        "live_risk_category", "source_statement", "observed_at"
    ]

    for col in defaults_numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in defaults_text:
        if col not in df.columns:
            df[col] = ""

    if "hotspot_id" not in df.columns:
        df["hotspot_id"] = [
            f"LIVE_{i:06d}" for i in range(1, len(df) + 1)
        ]

    # Reconstruct observation timestamp from FIRMS if needed.
    if (
        FIRMS_FILE.exists()
        and df["observed_at"].astype(str).str.strip().eq("").all()
    ):
        try:
            firms = pd.read_csv(FIRMS_FILE)
            if "hotspot_id" in firms.columns:
                if "acquisition_datetime" in firms.columns:
                    obs = firms[["hotspot_id", "acquisition_datetime"]].copy()
                    obs.columns = ["hotspot_id", "observed_at"]
                    df = df.drop(columns=["observed_at"]).merge(
                        obs, on="hotspot_id", how="left"
                    )
                elif {"acq_date", "acq_time"}.issubset(firms.columns):
                    obs = firms[["hotspot_id", "acq_date", "acq_time"]].copy()
                    obs["observed_at"] = (
                        obs["acq_date"].astype(str)
                        + " "
                        + obs["acq_time"].astype(str).str.zfill(4)
                    )
                    df = df.drop(columns=["observed_at"]).merge(
                        obs[["hotspot_id", "observed_at"]],
                        on="hotspot_id",
                        how="left"
                    )
        except Exception:
            pass

    df["facility_display"] = df.apply(facility_label, axis=1)

    df["predicted_source"] = (
        df["predicted_source"].fillna("UNKNOWN").astype(str).str.upper()
    )
    df["live_risk_category"] = (
        df["live_risk_category"].fillna("LOW").astype(str).str.upper()
    )
    df["attribution_quality"] = (
        df["attribution_quality"].fillna("UNCONFIRMED").astype(str).str.upper()
    )

    df = df[
        df["latitude"].between(6, 38)
        & df["longitude"].between(68, 98)
    ].copy()

    df = df.sort_values(
        ["live_risk_score", "frp"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df, path

df, data_path = load_data()

if df.empty:
    st.error("No ThermoWatch prediction data found.")
    st.write("Run the live FIRMS → facility attribution → V5 inference pipeline first.")
    st.code(
        "processed/live_final_predictions.csv\n"
        "processed/live_v5_predictions.csv\n"
        "processed/live_facility_matches.csv"
    )
    st.stop()

# ============================================================
# LIVE CONTROLS
# ============================================================

ctl1, ctl2, ctl3 = st.columns([1.2, 1.2, 4.6])

with ctl1:
    refresh_now = st.button("🔄 Refresh data", use_container_width=True)

with ctl2:
    auto_refresh = st.toggle("Auto refresh", value=False)

if refresh_now:
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================

critical_count = int((df["live_risk_category"] == "CRITICAL").sum())
high_count = int((df["live_risk_category"] == "HIGH").sum())

h1, h2 = st.columns([2.2, 1.0])

with h1:
    st.markdown('<div class="tw-title">🔥 ThermoWatch <span style="color:#f5b42c">AI</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="tw-sub">SATELLITE THERMAL INTELLIGENCE · FACILITY ATTRIBUTION · V5 RISK ENGINE</div>',
        unsafe_allow_html=True
    )

with h2:
    st.success("● SATELLITE DATA LOADED", icon="🛰️")
    st.warning(
        f"ALERT LEVEL: {'CRITICAL' if critical_count else 'ELEVATED'} · {critical_count}",
        icon="🔥"
    )

st.divider()

# ============================================================
# FILTERS
# ============================================================

f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1.1, 1.15, 1.15, 1.8])

with f1:
    risk_filter = st.selectbox(
        "Risk",
        ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

with f2:
    conf_filter = st.selectbox(
        "Confidence",
        ["All", "≥80%", "≥60%", "≥40%"]
    )

with f3:
    source_options = ["All"] + sorted(df["predicted_source"].dropna().unique().tolist())
    source_filter = st.selectbox("AI Source", source_options)

with f4:
    quality_filter = st.selectbox(
        "Attribution",
        ["All", "STRONG", "MODERATE", "WEAK", "UNCONFIRMED"]
    )

with f5:
    time_filter = st.selectbox(
        "Observation Window",
        ["All", "Last 24h", "Last 48h", "Last 7d"]
    )

with f6:
    search = st.text_input(
        "Search",
        placeholder="Hotspot, facility, source, lat/lon..."
    )

filtered = df.copy()

if risk_filter != "All":
    filtered = filtered[filtered["live_risk_category"] == risk_filter]

if conf_filter != "All":
    threshold = int(conf_filter.replace("≥", "").replace("%", ""))
    filtered = filtered[filtered["confidence"] >= threshold]

if source_filter != "All":
    filtered = filtered[filtered["predicted_source"] == source_filter]

if quality_filter != "All":
    filtered = filtered[
        filtered["attribution_quality"] == quality_filter
    ]

if time_filter != "All":
    obs_dt = pd.to_datetime(filtered["observed_at"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    hours = {"Last 24h": 24, "Last 48h": 48, "Last 7d": 168}[time_filter]
    filtered = filtered[
        obs_dt.notna() & (obs_dt >= now_utc - pd.Timedelta(hours=hours))
    ]

if search.strip():
    q = search.strip().lower()
    searchable = (
        filtered["hotspot_id"].astype(str)
        + " " + filtered["facility_display"].astype(str)
        + " " + filtered["predicted_source"].astype(str)
        + " " + filtered["latitude"].astype(str)
        + " " + filtered["longitude"].astype(str)
    ).str.lower()
    filtered = filtered[searchable.str.contains(q, na=False)]

# ============================================================
# KPI
# ============================================================

total = len(df)
visible = len(filtered)
avg_risk = float(df["live_risk_score"].mean()) if total else 0
max_frp = float(df["frp"].max()) if total else 0

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("SATELLITE HOTSPOTS", f"{total:,}")
k2.metric("CRITICAL", f"{critical_count:,}")
k3.metric("HIGH RISK", f"{high_count:,}")
k4.metric("AVERAGE RISK", f"{avg_risk:.1f}")
k5.metric("MAXIMUM FRP", f"{max_frp:.2f} MW")

st.info(
    f"Showing {visible:,} of {total:,} satellite-detected hotspots. "
    "Observed timestamps are the latest satellite observations available in the dataset; "
    "they do not mean every event is occurring at this exact second.",
    icon="ℹ️"
)

# ============================================================
# MAIN THREE-COLUMN DASHBOARD
# ============================================================

left, center, right = st.columns([1.0, 2.15, 1.0], gap="small")

# ============================================================
# ALERT FEED
# ============================================================

with left:
    st.markdown("### ⚡ ALERTS FEED")
    st.caption("PRIORITY STREAM · RANKED BY RISK")

    alert_df = filtered.head(12)

    if alert_df.empty:
        st.info("No hotspots match the current filters.")
    else:
        for _, row in alert_df.iterrows():
            risk = clean_text(row["live_risk_category"], "LOW").upper()
            score = num(row, "live_risk_score")
            frp = num(row, "frp")
            facility = clean_text(row["facility_display"], "Unidentified Facility")
            source = clean_text(row["predicted_source"], "UNKNOWN")
            hotspot = clean_text(row["hotspot_id"], "UNKNOWN")

            with st.container(border=True):
                a1, a2 = st.columns([2.5, 1.0])
                with a1:
                    st.markdown(f"**{facility}**")
                with a2:
                    st.markdown(
                        f"<span style='color:{risk_color(risk)};font-family:monospace;font-weight:800'>{risk} · {score:.1f}</span>",
                        unsafe_allow_html=True
                    )
                st.caption(hotspot)
                st.caption(f"FRP {frp:.2f} MW · {source}")

# ============================================================
# INDIA THERMAL MAP
# ============================================================

with center:
    st.markdown("### 🇮🇳 INDIA THERMAL ACTIVITY MAP")
    st.caption(
        f"{len(filtered):,} SATELLITE-DETECTED HOTSPOTS · "
        "THERMAL INTENSITY + RISK CLUSTERS"
    )

    if filtered.empty:
        st.warning("No hotspots to display.")
        map_result = {}
    else:
        m = folium.Map(
            location=[22.5, 79.0],
            zoom_start=5,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        heat_data = [
            [
                float(r.latitude),
                float(r.longitude),
                max(float(r.live_risk_score) / 100.0, 0.05)
            ]
            for r in filtered.itertuples()
            if pd.notna(r.latitude) and pd.notna(r.longitude)
        ]

        if heat_data:
            HeatMap(
                heat_data,
                name="Thermal Risk Heatmap",
                radius=25,
                blur=20,
                min_opacity=0.28,
                max_zoom=8,
            ).add_to(m)

        cluster = MarkerCluster(
            name="Hotspots",
            options={
                "maxClusterRadius": 35,
                "disableClusteringAtZoom": 8,
            }
        ).add_to(m)

        for _, row in filtered.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]
            if pd.isna(lat) or pd.isna(lon):
                continue

            risk = clean_text(row["live_risk_category"], "LOW").upper()
            facility = clean_text(row["facility_display"], "Unidentified Facility")
            score = num(row, "live_risk_score")
            frp = num(row, "frp")
            distance = row["distance_to_facility_km"]
            distance_text = (
                f"{float(distance):.3f} km"
                if pd.notna(distance) else "Unavailable"
            )

            popup_html = f"""
            <div style="font-family:Arial;min-width:250px">
                <b>{facility}</b><br>
                Hotspot: {clean_text(row['hotspot_id'])}<br>
                Risk: {score:.1f}/100 · {risk}<br>
                FRP: {frp:.2f} MW<br>
                AI Source: {clean_text(row['predicted_source'],'UNKNOWN')}<br>
                Confidence: {num(row,'confidence'):.1f}%<br>
                Facility distance: {distance_text}<br>
                Observed: {format_observed(row['observed_at'])}
            </div>
            """

            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=max(4, min(11, 4 + score / 18)),
                color=risk_color(risk),
                weight=1.4,
                fill=True,
                fill_color=risk_color(risk),
                fill_opacity=.9,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{clean_text(row['hotspot_id'])} · {risk} · {score:.1f}",
            ).add_to(cluster)

        folium.LayerControl(collapsed=True).add_to(m)

        map_result = st_folium(
            m,
            width=None,
            height=590,
            returned_objects=["last_object_clicked"],
            key="thermowatch_india_map",
        )

    st.caption("🔴 CRITICAL   🟠 HIGH   🟡 MEDIUM   🔵 LOW · Heat intensity = relative thermal/risk concentration · Click a hotspot to inspect")

# ============================================================
# INSPECTOR
# ============================================================

with right:
    st.markdown("### 🔎 INSPECTOR")
    st.caption("SOURCE & RISK DIAGNOSTICS")

    selected = None

    clicked = map_result.get("last_object_clicked") if map_result else None

    if (
        clicked
        and clicked.get("lat") is not None
        and clicked.get("lng") is not None
        and not filtered.empty
    ):
        click_lat = float(clicked["lat"])
        click_lon = float(clicked["lng"])

        temp = filtered.copy()
        temp["_distance"] = (
            (temp["latitude"] - click_lat) ** 2
            + (temp["longitude"] - click_lon) ** 2
        )
        selected = temp.sort_values("_distance").iloc[0]

    if selected is None and not filtered.empty:
        selected = filtered.iloc[0]

    if selected is None:
        st.info("Select a hotspot to inspect.")
    else:
        facility = clean_text(selected["facility_display"], "Unidentified Facility")
        hotspot = clean_text(selected["hotspot_id"])
        source = clean_text(selected["predicted_source"], "UNKNOWN")
        confidence = num(selected, "confidence")
        confidence_lvl = clean_text(
            selected["confidence_level"],
            confidence_level(confidence)
        ).upper()
        risk = clean_text(selected["live_risk_category"], "LOW").upper()
        risk_score = num(selected, "live_risk_score")
        frp = num(selected, "frp")
        lat = num(selected, "latitude")
        lon = num(selected, "longitude")
        distance = selected["distance_to_facility_km"]
        distance_text = (
            f"{float(distance):.3f} km"
            if pd.notna(distance) else "Unavailable"
        )
        attribution = clean_text(
            selected["attribution_quality"], "UNCONFIRMED"
        ).upper()
        observed = format_observed(selected["observed_at"])
        age = data_age(selected["observed_at"])
        statement = clean_text(
            selected["source_statement"],
            "No source statement available."
        )

        with st.container(border=True):
            st.caption("LOCATION")
            st.markdown(f"**{facility}**")

            st.caption("HOTSPOT")
            st.code(hotspot, language=None)

            st.caption("LAT / LON")
            st.write(f"{lat:.5f}° N · {lon:.5f}° E")

            st.caption("OBSERVED AT")
            st.write(observed)
            st.caption(f"DATA AGE · {age}")

            st.caption("AI CLASSIFICATION")
            st.markdown(f"**{source}**")
            st.write(f"Confidence: **{confidence:.1f}% · {confidence_lvl}**")
            st.progress(min(max(confidence / 100, 0), 1))

            st.caption("THERMAL / SENSOR DATA")
            st.write(f"FRP: **{frp:.2f} MW**")
            st.write(f"Distance to mapped facility: **{distance_text}**")

            st.caption("ATTRIBUTION")
            st.markdown(
                f"**{attribution}**",
                help="OSM proximity/mapping quality for the matched facility."
            )

            st.caption("RISK DIAGNOSTICS")
            st.markdown(
                f"### {risk_score:.1f}/100 · {risk}"
            )

            st.caption("SOURCE STATEMENT")
            st.write(statement)

            st.caption(
                "Facility labels come from mapped OSM metadata. "
                "When OSM has no name, ThermoWatch uses an unnamed label "
                "instead of inventing a facility name."
            )

# ============================================================
# LIVE DATA TABLE
# ============================================================

st.markdown("### 📡 LIVE HOTSPOT DATA")
st.caption("AI-ATTRIBUTED SATELLITE OBSERVATIONS")

table_cols = [
    "hotspot_id",
    "facility_display",
    "latitude",
    "longitude",
    "observed_at",
    "frp",
    "predicted_source",
    "confidence",
    "attribution_quality",
    "live_risk_score",
    "live_risk_category",
]

table = filtered[table_cols].copy()

table = table.rename(columns={
    "hotspot_id": "Hotspot ID",
    "facility_display": "Facility",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "observed_at": "Observed At",
    "frp": "FRP (MW)",
    "predicted_source": "AI Source",
    "confidence": "Confidence %",
    "attribution_quality": "Attribution",
    "live_risk_score": "Risk Score",
    "live_risk_category": "Risk",
})

if not table.empty:
    table["Observed At"] = table["Observed At"].apply(format_observed)
    table["Latitude"] = table["Latitude"].round(5)
    table["Longitude"] = table["Longitude"].round(5)
    table["FRP (MW)"] = table["FRP (MW)"].round(2)
    table["Confidence %"] = table["Confidence %"].round(1)
    table["Risk Score"] = table["Risk Score"].round(1)

st.dataframe(
    table,
    use_container_width=True,
    height=390,
    hide_index=True,
)

# ============================================================
# EXPORT
# ============================================================

c1, c2 = st.columns([1, 5])
with c1:
    st.download_button(
        "⬇️ Export CSV",
        data=filtered.to_csv(index=False),
        file_name="thermowatch_filtered_hotspots.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# OPERATIONAL SUMMARY
# ============================================================

if not filtered.empty:
    op1, op2, op3, op4 = st.columns(4)

    with op1:
        st.metric("VISIBLE HOTSPOTS", f"{len(filtered):,}")

    with op2:
        st.metric(
            "CRITICAL VISIBLE",
            f"{int((filtered['live_risk_category'] == 'CRITICAL').sum()):,}"
        )

    with op3:
        strong = int((filtered["attribution_quality"] == "STRONG").sum())
        st.metric("STRONG ATTRIBUTIONS", f"{strong:,}")

    with op4:
        observed_dt = pd.to_datetime(
            filtered["observed_at"], errors="coerce", utc=True
        )
        if observed_dt.notna().any():
            newest = observed_dt.max()
            age_hours = max(
                0,
                (pd.Timestamp.now(tz="UTC") - newest).total_seconds() / 3600
            )
            st.metric("NEWEST OBSERVATION AGE", f"{age_hours:.1f} h")
        else:
            st.metric("NEWEST OBSERVATION AGE", "N/A")

# ============================================================
# DATA PROVENANCE
# ============================================================

latest = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
latest_text = (
    latest.max().strftime("%d %b %Y · %H:%M UTC")
    if latest.notna().any()
    else "Unavailable"
)

st.caption(
    f"THERMOWATCH · V5 AI SOURCE ATTRIBUTION · "
    f"LATEST OBSERVATION: {latest_text} · "
    f"DATA FILE: {data_path.name if data_path else 'Unavailable'} · "
    f"MAP: OPENSTREETMAP"
)
