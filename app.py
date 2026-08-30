import streamlit as st
import pandas as pd
import numpy as np
import folium
import html

from pathlib import Path
from datetime import datetime, timezone

from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster


# ============================================================
# THERMOWATCH - PROFESSIONAL LIVE THERMAL ACTIVITY DASHBOARD
# ============================================================

st.set_page_config(
    page_title="ThermoWatch AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

FINAL_FILE = PROCESSED_DIR / "live_final_predictions.csv"
V5_FILE = PROCESSED_DIR / "live_v5_predictions.csv"
FACILITY_FILE = PROCESSED_DIR / "live_facility_matches.csv"
FIRMS_FILE = PROCESSED_DIR / "live_firms_hotspots.csv"


# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 50% -20%, rgba(18, 42, 68, .35), transparent 42%),
            #05080d;
        color: #e7edf5;
    }

    header[data-testid="stHeader"] {
        background: #05080d;
    }

    .block-container {
        padding-top: 0.65rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    .tw-topbar {
        border: 1px solid #172333;
        background: rgba(8, 14, 23, .96);
        border-radius: 12px;
        padding: 13px 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .tw-brand {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: .2px;
    }

    .tw-ai {
        color: #f5b42c;
        border: 1px solid #735817;
        background: #241c09;
        border-radius: 6px;
        padding: 3px 7px;
        font-size: 11px;
        margin-left: 6px;
        vertical-align: middle;
    }

    .tw-live {
        color: #29e3a2;
        font-family: monospace;
        font-size: 12px;
        letter-spacing: 1px;
    }

    .tw-alert {
        color: #ff4d65;
        border: 1px solid #702536;
        background: #210c13;
        padding: 6px 10px;
        border-radius: 7px;
        font-family: monospace;
        font-size: 12px;
        letter-spacing: .5px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 800;
        margin: 8px 0 7px 0;
        letter-spacing: .2px;
    }

    .section-subtitle {
        color: #718096;
        font-size: 11px;
        margin-bottom: 9px;
        font-family: monospace;
    }

    .metric-card {
        background: linear-gradient(180deg, #0c1520, #09111a);
        border: 1px solid #1d3045;
        border-radius: 9px;
        padding: 11px 13px;
        min-height: 72px;
    }

    .metric-label {
        color: #75879b;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #f3f7fb;
        font-size: 23px;
        font-weight: 800;
        margin-top: 5px;
    }

    .metric-value.red { color: #ff405d; }
    .metric-value.orange { color: #ff9f32; }
    .metric-value.cyan { color: #28d7ff; }

    .alert-card {
        background: #0b131e;
        border: 1px solid #24364a;
        border-left: 3px solid #ff405d;
        border-radius: 8px;
        padding: 10px 11px;
        margin-bottom: 7px;
    }

    .alert-title {
        font-size: 12px;
        font-weight: 800;
    }

    .alert-id {
        color: #64788e;
        font-size: 9px;
        font-family: monospace;
        margin-top: 3px;
    }

    .alert-risk {
        float: right;
        color: #ff405d;
        font-family: monospace;
        font-size: 10px;
        font-weight: 800;
    }

    .alert-meta {
        color: #8da0b5;
        font-size: 9px;
        margin-top: 5px;
    }

    .inspector {
        background: linear-gradient(180deg, #0d1723, #09111a);
        border: 1px solid #1e3044;
        border-radius: 10px;
        padding: 15px;
    }

    .inspector-label {
        color: #71859a;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 9px;
    }

    .inspector-value {
        color: #f0f5fa;
        font-size: 15px;
        font-weight: 800;
        margin-top: 3px;
    }

    .inspector-small {
        color: #b5c2d0;
        font-size: 11px;
        margin-top: 3px;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 10px;
        font-weight: 800;
        margin-top: 5px;
    }

    .pill-critical {
        color: #ff4d65;
        background: #2a0e16;
        border: 1px solid #72263a;
    }

    .pill-high {
        color: #ffad3b;
        background: #291b08;
        border: 1px solid #73501c;
    }

    .pill-medium {
        color: #ffd34d;
        background: #29230a;
        border: 1px solid #73631b;
    }

    .pill-low {
        color: #42bfff;
        background: #091e2b;
        border: 1px solid #174e6a;
    }

    .notice {
        border: 1px solid #1c3043;
        background: #09121c;
        color: #8fa1b5;
        border-radius: 8px;
        padding: 9px 11px;
        font-size: 10px;
        margin-top: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1a2b3d;
        border-radius: 8px;
    }

    .stButton button {
        border-radius: 7px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def first_existing_file():
    for path in [FINAL_FILE, V5_FILE, FACILITY_FILE]:
        if path.exists():
            return path
    return None


def clean_text(value, fallback=""):
    if value is None:
        return fallback
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return fallback
    return text


def escape_html(text):
    """Safely escape user data for HTML rendering."""
    if text is None:
        return ""
    return html.escape(str(text))


def format_facility_name(row):
    """
    Professional facility naming.
    Never invents a real facility name when OSM has no name.
    """
    name = clean_text(row.get("facility_name"), "")

    if name:
        return name

    facility_type = clean_text(row.get("facility_type"), "").lower()
    power = clean_text(row.get("facility_power"), "").lower()
    industrial = clean_text(row.get("facility_industrial"), "").lower()

    if "substation" in power or "substation" in facility_type:
        return "Unnamed Power Substation"

    if "generator" in power or "plant" in power:
        return "Unnamed Power Facility"

    if industrial:
        return "Unnamed Industrial Facility"

    if "industrial" in facility_type:
        return "Unnamed Industrial Facility"

    return "Unnamed Facility"


def risk_class(category):
    value = clean_text(category, "LOW").upper()
    return value.lower()


def marker_color(category):
    value = clean_text(category, "LOW").upper()
    return {
        "CRITICAL": "#ff304f",
        "HIGH": "#ff8a24",
        "MEDIUM": "#ffd23f",
        "LOW": "#29a9ff",
    }.get(value, "#29a9ff")


def risk_icon(category):
    value = clean_text(category, "LOW").upper()
    return {
        "CRITICAL": "fire",
        "HIGH": "exclamation-triangle",
        "MEDIUM": "circle",
        "LOW": "circle",
    }.get(value, "circle")


def confidence_level_from_number(value):
    try:
        value = float(value)
    except Exception:
        return "LOW"

    if value >= 80:
        return "HIGH"
    if value >= 60:
        return "MEDIUM"
    return "LOW"


def observed_age_text(value):
    if pd.isna(value) or clean_text(value, "") == "":
        return "Observation time unavailable"

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "Observation time unavailable"

    now = pd.Timestamp.now(tz="UTC")
    delta_hours = max(0, (now - ts).total_seconds() / 3600)

    if delta_hours < 1:
        return f"{delta_hours * 60:.0f} min ago"
    if delta_hours < 24:
        return f"{delta_hours:.1f} hr ago"
    return f"{delta_hours / 24:.1f} days ago"


def format_observed_at(value):
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "Unavailable"
    return ts.strftime("%d %b %Y · %H:%M UTC")


def source_badge(source):
    source = clean_text(source, "UNKNOWN").upper()
    return source.replace("_", " ")


def make_popup(row):
    hotspot = escape_html(clean_text(row.get("hotspot_id"), "Unknown"))
    facility = escape_html(clean_text(row.get("display_facility_name"), "Unnamed Facility"))
    source = escape_html(source_badge(row.get("predicted_source")))
    confidence = float(row.get("confidence", 0) or 0)
    risk = float(row.get("live_risk_score", 0) or 0)
    category = escape_html(clean_text(row.get("live_risk_category"), "LOW").upper())
    frp = float(row.get("frp", 0) or 0)
    distance = row.get("distance_to_facility_km", np.nan)
    observed = escape_html(format_observed_at(row.get("observed_at")))

    distance_text = (
        f"{float(distance):.3f} km"
        if pd.notna(distance)
        else "Unavailable"
    )

    html_content = f"""
    <div style="font-family:Arial,sans-serif;min-width:240px;color:#111">
        <div style="font-size:16px;font-weight:800;margin-bottom:6px">
            {hotspot}
        </div>
        <div style="font-size:12px">
            <b>Facility:</b> {facility}<br>
            <b>AI Source:</b> {source}<br>
            <b>Confidence:</b> {confidence:.1f}%<br>
            <b>FRP:</b> {frp:.2f} MW<br>
            <b>Facility Distance:</b> {distance_text}<br>
            <b>Risk:</b> {risk:.1f}/100 · {category}<br>
            <b>Observed:</b> {observed}
        </div>
    </div>
    """
    return folium.Popup(html_content, max_width=330)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data(ttl=60)
def load_data():
    path = first_existing_file()

    if path is None:
        return pd.DataFrame(), None

    df = pd.read_csv(path)

    # If the final enriched file is unavailable, merge V5 + facility data.
    if path.name == V5_FILE.name and FACILITY_FILE.exists():
        fac = pd.read_csv(FACILITY_FILE)

        if "hotspot_id" in fac.columns:
            merge_cols = [
                c for c in fac.columns
                if c not in df.columns or c in {
                    "hotspot_id",
                    "facility_name",
                    "facility_type",
                    "facility_power",
                    "facility_industrial",
                    "distance_to_facility_km",
                    "facility_match_quality",
                }
            ]
            merge_cols = list(dict.fromkeys(["hotspot_id"] + merge_cols))
            df = df.merge(fac[merge_cols], on="hotspot_id", how="left")

    if "hotspot_id" not in df.columns:
        df["hotspot_id"] = [f"LIVE_{i:06d}" for i in range(1, len(df) + 1)]

    # Normalize expected fields.
    numeric_cols = [
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "live_risk_score",
        "distance_to_facility_km",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "facility_name",
        "facility_type",
        "facility_power",
        "facility_industrial",
        "predicted_source",
        "confidence_level",
        "attribution_quality",
        "live_risk_category",
        "source_statement",
        "observed_at",
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""

    df["display_facility_name"] = df.apply(format_facility_name, axis=1)

    # If observed_at is missing, attempt to reconstruct from FIRMS acquisition fields.
    if df["observed_at"].astype(str).str.strip().eq("").all() and FIRMS_FILE.exists():
        try:
            firms = pd.read_csv(FIRMS_FILE)
            if "hotspot_id" in firms.columns:
                candidate = None

                if "acquisition_datetime" in firms.columns:
                    candidate = "acquisition_datetime"
                elif {"acq_date", "acq_time"}.issubset(firms.columns):
                    firms["acquisition_datetime"] = (
                        firms["acq_date"].astype(str)
                        + " "
                        + firms["acq_time"].astype(str).str.zfill(4)
                    )
                    candidate = "acquisition_datetime"

                if candidate:
                    firms_small = firms[["hotspot_id", candidate]].copy()
                    firms_small = firms_small.rename(columns={candidate: "observed_at"})
                    df = df.drop(columns=["observed_at"]).merge(
                        firms_small,
                        on="hotspot_id",
                        how="left",
                    )
        except Exception:
            pass

    df["confidence"] = df["confidence"].fillna(0)
    df["live_risk_score"] = df["live_risk_score"].fillna(0)
    df["frp"] = df["frp"].fillna(0)

    df["confidence_level"] = np.where(
        df["confidence_level"].astype(str).str.strip().eq(""),
        df["confidence"].apply(confidence_level_from_number),
        df["confidence_level"].astype(str).str.upper(),
    )

    df["live_risk_category"] = (
        df["live_risk_category"]
        .fillna("LOW")
        .astype(str)
        .str.upper()
    )

    df["predicted_source"] = (
        df["predicted_source"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    # Valid coordinates only.
    df = df[
        df["latitude"].between(6, 38, inclusive="both")
        & df["longitude"].between(68, 98, inclusive="both")
    ].copy()

    # Stable order: highest risk first.
    df = df.sort_values(
        ["live_risk_score", "frp"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df, path


df, data_path = load_data()


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

if df.empty:
    st.error(
        "No live prediction data found. Run the FIRMS → facility attribution → "
        "V5 inference pipeline first."
    )
    st.code(
        "processed/live_final_predictions.csv\n"
        "processed/live_v5_predictions.csv\n"
        "processed/live_facility_matches.csv",
        language="text",
    )
    st.stop()

critical_count = int((df["live_risk_category"] == "CRITICAL").sum())
high_count = int((df["live_risk_category"] == "HIGH").sum())

st.markdown(
    f"""
    <div class="tw-topbar">
        <div>
            <span class="tw-brand">🔥 ThermoWatch</span>
            <span class="tw-ai">AI</span>
        </div>
        <div style="display:flex;gap:18px;align-items:center">
            <span class="tw-live">● DATA INGESTION ACTIVE</span>
            <span class="tw-alert">🔥 ALERT LEVEL: {"CRITICAL" if critical_count else "ELEVATED"} ({critical_count})</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# FILTER BAR
# ------------------------------------------------------------

f1, f2, f3, f4, f5 = st.columns([1.15, 1.1, 1.1, 1.1, 1.7])

with f1:
    risk_filter = st.selectbox(
        "Risk",
        ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        index=0,
    )

with f2:
    conf_filter = st.selectbox(
        "Confidence",
        ["All", "≥80%", "≥60%", "≥40%"],
        index=0,
    )

with f3:
    source_filter = st.selectbox(
        "AI Source",
        ["All"] + sorted(
            [x for x in df["predicted_source"].dropna().unique()]
        ),
    )

with f4:
    quality_filter = st.selectbox(
        "Attribution",
        ["All", "STRONG", "MODERATE", "WEAK", "UNCONFIRMED"],
        index=0,
    )

with f5:
    search = st.text_input(
        "Search",
        placeholder="Hotspot, facility, source, lat/lon...",
        label_visibility="visible",
    )


filtered = df.copy()

if risk_filter != "All":
    filtered = filtered[
        filtered["live_risk_category"] == risk_filter
    ]

if conf_filter != "All":
    threshold = int(conf_filter.replace("≥", "").replace("%", ""))
    filtered = filtered[filtered["confidence"] >= threshold]

if source_filter != "All":
    filtered = filtered[
        filtered["predicted_source"] == source_filter
    ]

if quality_filter != "All":
    filtered = filtered[
        filtered["attribution_quality"].astype(str).str.upper()
        == quality_filter
    ]

if search.strip():
    q = search.strip().lower()
    searchable = (
        filtered["hotspot_id"].astype(str)
        + " "
        + filtered["display_facility_name"].astype(str)
        + " "
        + filtered["predicted_source"].astype(str)
        + " "
        + filtered["latitude"].astype(str)
        + " "
        + filtered["longitude"].astype(str)
    ).str.lower()

    filtered = filtered[searchable.str.contains(q, na=False)]


# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

total = len(df)
visible = len(filtered)
critical = int((df["live_risk_category"] == "CRITICAL").sum())
high = int((df["live_risk_category"] == "HIGH").sum())
avg_risk = float(df["live_risk_score"].mean()) if total else 0
max_frp = float(df["frp"].max()) if total else 0

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Satellite Hotspots</div>'
        f'<div class="metric-value">{total:,}</div></div>',
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Critical</div>'
        f'<div class="metric-value red">{critical:,}</div></div>',
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">High Risk</div>'
        f'<div class="metric-value orange">{high:,}</div></div>',
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Average Risk</div>'
        f'<div class="metric-value cyan">{avg_risk:.1f}</div></div>',
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Maximum FRP</div>'
        f'<div class="metric-value">{max_frp:.2f} MW</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div class="notice">
        Showing <b>{visible:,}</b> of <b>{total:,}</b> satellite-detected hotspots.
        "Observed" timestamps represent the latest satellite observation available
        in the FIRMS feed; this dashboard does not claim that every event is occurring
        at this exact second.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------

left, center, right = st.columns([1.0, 2.05, 1.0], gap="small")


# ------------------------------------------------------------
# ALERT FEED
# ------------------------------------------------------------

with left:
    st.markdown(
        '<div class="section-title">⚡ ALERTS FEED</div>'
        '<div class="section-subtitle">REAL-TIME PRIORITY STREAM · RANKED BY RISK</div>',
        unsafe_allow_html=True,
    )

    alert_df = filtered.head(12)

    if alert_df.empty:
        st.info("No hotspots match the current filters.")
    else:
        for _, row in alert_df.iterrows():
            risk = clean_text(row["live_risk_category"], "LOW").upper()
            facility = escape_html(clean_text(
                row["display_facility_name"],
                "Unnamed Facility",
            ))
            source = escape_html(source_badge(row["predicted_source"]))
            frp = float(row["frp"])
            score = float(row["live_risk_score"])
            hotspot_id = escape_html(clean_text(row["hotspot_id"]))

            st.markdown(
                f"""
                <div class="alert-card">
                    <span class="alert-risk">{risk} · {score:.1f}</span>
                    <div class="alert-title">{facility}</div>
                    <div class="alert-id">{hotspot_id}</div>
                    <div class="alert-meta">
                        FRP {frp:.2f} MW · {source}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------
# INDIA THERMAL MAP
# ------------------------------------------------------------

with center:
    st.markdown(
        '<div class="section-title">🇮🇳 INDIA THERMAL ACTIVITY MAP</div>'
        f'<div class="section-subtitle">{total:,} SATELLITE-DETECTED HOTSPOTS · CLICK A HOTSPOT TO INSPECT</div>',
        unsafe_allow_html=True,
    )

    # API-key-free basemap: OpenStreetMap.
    m = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Dark visual layer using standard OSM plus a semi-transparent
    # thermal heat layer. No CARTO/Mapbox/API key required.
    heat_data = [
        [
            float(row.latitude),
            float(row.longitude),
            max(float(row.live_risk_score) / 100.0, 0.05),
        ]
        for row in filtered.itertuples()
        if pd.notna(row.latitude) and pd.notna(row.longitude)
    ]

    if heat_data:
        HeatMap(
            heat_data,
            name="Thermal Intensity",
            radius=22,
            blur=18,
            min_opacity=0.30,
            max_zoom=8,
        ).add_to(m)

    # Marker cluster keeps the India map readable.
    cluster = MarkerCluster(
        name="Hotspots",
        options={
            "maxClusterRadius": 35,
            "disableClusteringAtZoom": 8,
        },
    ).add_to(m)

    for _, row in filtered.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        if pd.isna(lat) or pd.isna(lon):
            continue

        risk = clean_text(row["live_risk_category"], "LOW").upper()
        color = marker_color(risk)

        radius = max(
            4,
            min(
                11,
                4 + float(row["live_risk_score"]) / 18,
            ),
        )

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=radius,
            color=color,
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.88,
            popup=make_popup(row),
            tooltip=(
                f'{escape_html(clean_text(row["hotspot_id"]))} · '
                f'{risk} · Risk {float(row["live_risk_score"]):.1f}'
            ),
        ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)

    map_result = st_folium(
        m,
        width=None,
        height=590,
        returned_objects=["last_object_clicked"],
        key="thermowatch_india_map",
    )


# ------------------------------------------------------------
# INSPECTOR
# ------------------------------------------------------------

with right:
    st.markdown(
        '<div class="section-title">🔎 INSPECTOR</div>'
        '<div class="section-subtitle">SOURCE & RISK DIAGNOSTICS</div>',
        unsafe_allow_html=True,
    )

    selected = None

    # Map click is used only to locate a nearby hotspot.
    clicked = map_result.get("last_object_clicked") if map_result else None

    if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
        click_lat = float(clicked["lat"])
        click_lon = float(clicked["lng"])

        temp = filtered.copy()
        temp["_map_distance"] = (
            (temp["latitude"] - click_lat) ** 2
            + (temp["longitude"] - click_lon) ** 2
        )

        if not temp.empty:
            selected = temp.sort_values("_map_distance").iloc[0]

    if selected is None and not filtered.empty:
        selected = filtered.iloc[0]

    if selected is None:
        st.info("Select a hotspot to inspect.")
    else:
        facility = escape_html(clean_text(
            selected["display_facility_name"],
            "Unnamed Facility",
        ))

        source = escape_html(source_badge(selected["predicted_source"]))
        confidence = float(selected["confidence"])
        risk_score = float(selected["live_risk_score"])
        risk_category = clean_text(
            selected["live_risk_category"],
            "LOW",
        ).upper()

        lat = float(selected["latitude"])
        lon = float(selected["longitude"])
        frp = float(selected["frp"])

        distance = selected["distance_to_facility_km"]
        if pd.notna(distance):
            distance_text = f"{float(distance):.3f} km"
        else:
            distance_text = "Unavailable"

        attribution = escape_html(clean_text(
            selected["attribution_quality"],
            "UNCONFIRMED",
        ).upper())

        confidence_level = clean_text(
            selected["confidence_level"],
            confidence_level_from_number(confidence),
        ).upper()

        statement = escape_html(clean_text(
            selected["source_statement"],
            "No source statement available.",
        ))

        observed = escape_html(format_observed_at(selected["observed_at"]))
        age = escape_html(observed_age_text(selected["observed_at"]))
        hotspot_id = escape_html(clean_text(selected["hotspot_id"]))

        st.markdown(
            f"""
            <div class="inspector">

                <div class="inspector-label">LOCATION</div>
                <div class="inspector-value">{facility}</div>

                <div class="inspector-label">HOTSPOT</div>
                <div class="inspector-small">
                    {hotspot_id}
                </div>

                <div class="inspector-label">LAT / LON</div>
                <div class="inspector-small">
                    {lat:.5f}° N · {lon:.5f}° E
                </div>

                <div class="inspector-label">OBSERVED AT</div>
                <div class="inspector-small">{observed}</div>
                <div class="inspector-small">DATA AGE · {age}</div>

                <div class="inspector-label">AI CLASSIFICATION</div>
                <div class="inspector-value">{source}</div>

                <div class="inspector-small">
                    Confidence · <b>{confidence:.1f}%</b>
                    · {confidence_level}
                </div>

                <div class="inspector-label">THERMAL / SENSOR DATA</div>
                <div class="inspector-small">
                    FRP · <b>{frp:.2f} MW</b>
                </div>

                <div class="inspector-small">
                    Distance to mapped facility · <b>{distance_text}</b>
                </div>

                <div class="inspector-label">ATTRIBUTION</div>
                <div class="status-pill pill-{risk_class(risk_category)}">
                    {attribution}
                </div>

                <div class="inspector-label">RISK DIAGNOSTICS</div>
                <div class="inspector-value">
                    {risk_score:.1f}/100 · {risk_category}
                </div>

                <div class="inspector-label">SOURCE STATEMENT</div>
                <div class="inspector-small">{statement}</div>

                <div class="notice">
                    Facility label is based on mapped OSM metadata.
                    If a name is unavailable, ThermoWatch uses an
                    <b>Unnamed Facility</b> label rather than inventing a name.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

st.markdown(
    """
    <div style="margin-top:7px;color:#8fa1b5;font-size:10px;font-family:monospace">
        <span style="color:#ff304f">● CRITICAL</span>&nbsp;&nbsp;
        <span style="color:#ff8a24">● HIGH</span>&nbsp;&nbsp;
        <span style="color:#ffd23f">● MEDIUM</span>&nbsp;&nbsp;
        <span style="color:#29a9ff">● LOW</span>
        &nbsp;&nbsp;·&nbsp;&nbsp; Heat intensity represents relative thermal/risk concentration.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LIVE DATA TABLE
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title" style="margin-top:18px">📡 LIVE HOTSPOT DATA</div>'
    '<div class="section-subtitle">AI-ATTRIBUTED SATELLITE OBSERVATIONS</div>',
    unsafe_allow_html=True,
)

table_cols = [
    "hotspot_id",
    "display_facility_name",
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

table = table.rename(
    columns={
        "hotspot_id": "Hotspot ID",
        "display_facility_name": "Facility",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "observed_at": "Observed At",
        "frp": "FRP (MW)",
        "predicted_source": "AI Source",
        "confidence": "Confidence %",
        "attribution_quality": "Attribution",
        "live_risk_score": "Risk Score",
        "live_risk_category": "Risk",
    }
)

if not table.empty:
    table["Observed At"] = table["Observed At"].apply(format_observed_at)
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


# ------------------------------------------------------------
# FOOTER / DATA PROVENANCE
# ------------------------------------------------------------

latest = None

if "observed_at" in df.columns:
    parsed = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
    if parsed.notna().any():
        latest = parsed.max()

latest_text = (
    latest.strftime("%d %b %Y · %H:%M UTC")
    if latest is not None
    else "Unavailable"
)

st.markdown(
    f"""
    <div style="
        margin-top:12px;
        padding:10px 13px;
        border-top:1px solid #182636;
        color:#617489;
        font-size:9px;
        font-family:monospace;
    ">
        THERMOWATCH · AI SOURCE ATTRIBUTION · V5 MODEL
        &nbsp; | &nbsp;
        LATEST OBSERVATION IN DATA: {latest_text}
        &nbsp; | &nbsp;
        SOURCE FILE: {data_path.name if data_path else "Unavailable"}
        &nbsp; | &nbsp;
        MAP: OPENSTREETMAP
    </div>
    """,
    unsafe_allow_html=True,
)
