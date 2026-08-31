import io
import math
import html
from pathlib import Path

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium

from sklearn.neighbors import BallTree
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

st.set_page_config(page_title="ThermoWatch AI", page_icon="🔥", layout="wide")

# =========================
# CONFIG
# =========================
MODEL_URL = "https://huggingface.co/harshitcodes1544/sihharshit/resolve/main/source_classifier_v5.joblib"
OSM_URL = "https://huggingface.co/datasets/harshitcodes1544/thermowatch-osm-data/resolve/main/osm_facilities.csv"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
INDIA_BBOX = "68.0,6.0,97.5,37.5"
EARTH_RADIUS_KM = 6371.0088

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 50% -20%,rgba(18,42,68,.4),transparent 45%),#05080d;color:#e7edf5}
.block-container{max-width:1600px;padding-top:1rem}
.tw-header,.no-fire,.notice,.online{border:1px solid #1b2a3a;background:rgba(8,14,23,.96);border-radius:14px;padding:20px;margin-bottom:15px}
.tw-title{font-size:30px;font-weight:900}
.tw-sub{color:#8fa2b8;font:13px monospace;margin-top:5px}
.live{display:inline-block;color:#29e3a2;border:1px solid #155e49;background:#071b16;border-radius:6px;padding:5px 10px;font:11px monospace;margin-top:10px}
.no-fire{text-align:center;border-color:#28445b;background:linear-gradient(135deg,#081927,#050c14)}
.no-fire-title{font-size:23px;font-weight:850}
.no-fire-text{color:#8295aa;font-size:13px;line-height:1.65;margin-top:9px}
.online{border-color:#155e49;background:#071b16;color:#29e3a2;font:12px monospace}
.notice{color:#9eb0c4;font-size:12px}
.section{font-size:18px;font-weight:850;margin:18px 0 7px}
.footer{color:#5f7083;font:10px monospace;border-top:1px solid #172536;padding-top:12px;margin-top:20px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tw-header">
<div class="tw-title">🔥 ThermoWatch <span style="color:#f5b42c">AI</span></div>
<div class="tw-sub">LIVE SATELLITE FIRE DETECTION & INDUSTRIAL RISK INTELLIGENCE</div>
<div class="tw-sub">NASA FIRMS → OSM FACILITIES → V5 AI → RISK ASSESSMENT</div>
<div class="live">● LIVE SATELLITE THERMAL MONITORING</div>
</div>
""", unsafe_allow_html=True)

def clean(v, default="UNKNOWN"):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    s = str(v).strip()
    return default if s.lower() in {"", "nan", "none", "null"} else s

def num(v, default=0.0):
    try:
        x=float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def fmt(v):
    d=pd.to_datetime(v,errors="coerce",utc=True)
    return "Unavailable" if pd.isna(d) else d.strftime("%d %b %Y · %H:%M UTC")

def facility_label(r):
    n=clean(r.get("facility_name"),"")
    return n if n else clean(r.get("facility_type"),"Unknown facility").replace("_"," ").title()

# =========================
# LOAD MODEL / OSM
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    r=requests.get(MODEL_URL,timeout=180,allow_redirects=True)
    if r.status_code==404:
        raise RuntimeError("V5 model not found. Check source_classifier_v5.joblib in the main branch of your Hugging Face model repository.")
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

@st.cache_data(ttl=86400,show_spinner=False)
def load_osm():
    r=requests.get(OSM_URL,timeout=240,allow_redirects=True)
    if r.status_code==404:
        raise RuntimeError("OSM dataset not found. Check osm_facilities.csv in the main branch of the thermowatch-osm-data Hugging Face dataset repository.")
    r.raise_for_status()
    d=pd.read_csv(io.BytesIO(r.content))
    for c in ["latitude","longitude"]:
        if c not in d.columns:
            raise RuntimeError(f"OSM dataset missing column: {c}")
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=["latitude","longitude"]).copy()

# =========================
# LIVE FIRMS
# =========================
@st.cache_data(ttl=900,show_spinner=False)
def fetch_firms(days):
    key=str(st.secrets.get("FIRMS_API_KEY","")).strip()
    if not key:
        raise RuntimeError("FIRMS_API_KEY is missing from Streamlit Secrets.")
    days=max(1,min(int(days),5))
    url=f"{FIRMS_URL}{key}/VIIRS_NOAA20_NRT/{INDIA_BBOX}/{days}"
    r=requests.get(url,timeout=240,allow_redirects=True)
    r.raise_for_status()
    if not r.text.strip():
        return pd.DataFrame()
    try:
        d=pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        raise RuntimeError("NASA FIRMS response could not be parsed as CSV.") from e
    if d.empty:
        return pd.DataFrame()

    for c in ["latitude","longitude","frp","bright_ti4","bright_ti5","acq_date","acq_time"]:
        if c not in d.columns:
            d[c]=np.nan

    for c in ["latitude","longitude","frp","bright_ti4","bright_ti5"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")

    t=pd.to_numeric(d["acq_time"],errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
    d["acquisition_datetime"]=pd.to_datetime(
        d["acq_date"].astype(str)+" "+t,
        format="%Y-%m-%d %H%M",errors="coerce",utc=True
    )
    d=d[d.latitude.between(6,37.5)&d.longitude.between(68,97.5)].copy()
    if d.empty:
        return pd.DataFrame()

    keys=[c for c in ["latitude","longitude","acq_date","acq_time","satellite","instrument"] if c in d.columns]
    if keys:
        d=d.drop_duplicates(keys)
    d=d.reset_index(drop=True)
    d["hotspot_id"]=[f"LIVE_{i:06d}" for i in range(1,len(d)+1)]
    return d

# =========================
# OSM ATTRIBUTION
# =========================
def attribute(live, facilities):
    out=live.reset_index(drop=True).copy()
    if out.empty:
        return out
    if facilities.empty:
        out["facility_type"]="unknown"; out["facility_name"]=np.nan
        out["facility_power"]="unknown"; out["facility_industrial"]="unknown"; out["facility_landuse"]="unknown"
        out["distance_to_facility_km"]=999.0
        out["proximity_category"]="FAR"; out["facility_match_quality"]="VERY_LOW"
        return out

    tree=BallTree(np.radians(facilities[["latitude","longitude"]].values),metric="haversine")
    dist,idx=tree.query(np.radians(out[["latitude","longitude"]].values),k=1)
    nearest=facilities.iloc[idx[:,0]].reset_index(drop=True)
    km=dist[:,0]*EARTH_RADIUS_KM

    def getcol(c,default=np.nan):
        return nearest[c].values if c in nearest.columns else np.full(len(nearest),default,dtype=object)

    out["facility_osm_id"]=getcol("osm_id")
    out["facility_type"]=getcol("feature_type","unknown")
    out["facility_name"]=getcol("name")
    out["facility_power"]=getcol("power","unknown")
    out["facility_industrial"]=getcol("industrial","unknown")
    out["facility_landuse"]=getcol("landuse","unknown")
    out["facility_operator"]=getcol("operator")
    out["facility_latitude"]=nearest["latitude"].values
    out["facility_longitude"]=nearest["longitude"].values
    out["distance_to_facility_km"]=km

    out["proximity_category"]=pd.cut(
        km,[-np.inf,1,2,5,10,np.inf],
        labels=["VERY_CLOSE","CLOSE","NEAR","DISTANT","FAR"],
        right=True
    ).astype(str)
    out["facility_match_quality"]=pd.cut(
        km,[-np.inf,1,5,10,np.inf],
        labels=["HIGH","MEDIUM","LOW","VERY_LOW"],
        right=True
    ).astype(str)
    return out

# =========================
# V5 FEATURES
# =========================
FEATURES=[
"mean_frp","max_frp","frp_ratio","mean_brightness","max_brightness","brightness_range",
"observation_count","persistence_days","observations_per_day","activity_density","satellite_count",
"distance_to_facility_km","facility_within_1km","facility_within_5km","persistent_long_term",
"persistent_180_days","facility_type","facility_power","facility_industrial","facility_landuse",
"frp_ratio_v5","frp_excess_v5","frp_log_v5","max_frp_log_v5","brightness_range_v5",
"brightness_ratio_v5","brightness_excess_v5","persistence_log_v5","persistence_months_v5",
"persistent_30d_v5","persistent_90d_v5","persistent_180d_v5","persistent_270d_v5",
"observation_log_v5","obs_per_persistence_v5","activity_persistence_v5","distance_log_v5",
"very_close_v5","within_2km_v5","within_5km_v5","within_10km_v5","frp_persistence_v5",
"frp_distance_signal_v5","activity_distance_signal_v5"
]

def predict(d,model):
    d=d.copy()
    d["frp"]=pd.to_numeric(d["frp"],errors="coerce").fillna(0)
    for c in ["bright_ti4","bright_ti5"]:
        d[c]=pd.to_numeric(d.get(c,0),errors="coerce").fillna(0)

    d["mean_frp"]=d["frp"]; d["max_frp"]=d["frp"]
    d["frp_ratio"]=d["max_frp"]/(d["mean_frp"]+1e-6)
    d["mean_brightness"]=d["bright_ti4"]; d["max_brightness"]=d["bright_ti4"]
    d["brightness_range"]=d["bright_ti4"]-d["bright_ti5"]
    d["observation_count"]=1; d["persistence_days"]=1; d["observations_per_day"]=1.0
    d["activity_density"]=1.0; d["satellite_count"]=1
    d["distance_to_facility_km"]=pd.to_numeric(d.get("distance_to_facility_km",999),errors="coerce").fillna(999)
    d["facility_within_1km"]=(d.distance_to_facility_km<=1).astype(int)
    d["facility_within_5km"]=(d.distance_to_facility_km<=5).astype(int)
    d["persistent_long_term"]=0; d["persistent_180_days"]=0

    for c in ["facility_type","facility_power","facility_industrial","facility_landuse"]:
        d[c]=d.get(c,"unknown")
        d[c]=d[c].fillna("unknown").astype(str)

    d["frp_ratio_v5"]=d.max_frp/(d.mean_frp+1e-6)
    d["frp_excess_v5"]=np.maximum(d.max_frp-5,0)
    d["frp_log_v5"]=np.log1p(d.mean_frp)
    d["max_frp_log_v5"]=np.log1p(d.max_frp)
    d["brightness_range_v5"]=d.brightness_range
    d["brightness_ratio_v5"]=(d.max_brightness+1e-6)/(d.mean_brightness+1e-6)
    d["brightness_excess_v5"]=np.maximum(d.max_brightness-300,0)
    d["persistence_log_v5"]=np.log1p(d.persistence_days)
    d["persistence_months_v5"]=d.persistence_days/30
    d["persistent_30d_v5"]=(d.persistence_days>=30).astype(int)
    d["persistent_90d_v5"]=(d.persistence_days>=90).astype(int)
    d["persistent_180d_v5"]=(d.persistence_days>=180).astype(int)
    d["persistent_270d_v5"]=(d.persistence_days>=270).astype(int)
    d["observation_log_v5"]=np.log1p(d.observation_count)
    d["obs_per_persistence_v5"]=d.observation_count/(d.persistence_days+1e-6)
    d["activity_persistence_v5"]=d.activity_density*d.persistence_days
    d["distance_log_v5"]=np.log1p(d.distance_to_facility_km)
    d["very_close_v5"]=(d.distance_to_facility_km<=1).astype(int)
    d["within_2km_v5"]=(d.distance_to_facility_km<=2).astype(int)
    d["within_5km_v5"]=(d.distance_to_facility_km<=5).astype(int)
    d["within_10km_v5"]=(d.distance_to_facility_km<=10).astype(int)
    d["frp_persistence_v5"]=d.mean_frp*d.persistence_days
    d["frp_distance_signal_v5"]=d.mean_frp/(d.distance_to_facility_km+1)
    d["activity_distance_signal_v5"]=d.activity_density/(d.distance_to_facility_km+1)

    X=d[FEATURES].copy()
    nums=X.select_dtypes(include=["number"]).columns
    X[nums]=X[nums].replace([np.inf,-np.inf],np.nan).fillna(0)
    for c in ["facility_type","facility_power","facility_industrial","facility_landuse"]:
        X[c]=X[c].fillna("unknown").astype(str)

    pred=model.predict(X)
    prob=model.predict_proba(X)
    conf=np.max(prob,axis=1)*100
    d["predicted_source"]=pred
    d["confidence"]=np.round(conf,1)

    risk=np.clip(d.mean_frp*3,0,35)+np.clip(d.max_frp*.5,0,20)+conf*.20
    d["live_risk_score"]=np.clip(risk,0,100).round(2)
    d["live_risk_category"]=pd.cut(
        d.live_risk_score,[-np.inf,25,50,75,np.inf],
        labels=["LOW","MEDIUM","HIGH","CRITICAL"]
    ).astype(str)
    d["display_facility_name"]=d.apply(facility_label,axis=1)
    return d

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## ⚙️ THERMOWATCH CONTROLS")
windows={"24 Hours":1,"48 Hours":2,"72 Hours":3,"96 Hours":4,"5 Days":5}
selected=st.sidebar.selectbox("Satellite observation window",list(windows))
days=windows[selected]

if st.sidebar.button("🔄 FETCH LATEST LIVE DATA",use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Maximum FIRMS area-query window: 5 days.")
st.sidebar.markdown(f"**CURRENT WINDOW**  
`{selected}`

**SOURCE**  
NASA FIRMS VIIRS NOAA-20 NRT

**AI**  
V5 ExtraTrees Classifier")

# =========================
# PIPELINE
# =========================
try:
    with st.spinner("Loading V5 AI model..."):
        model=load_model()
    with st.spinner("Loading OSM facilities..."):
        facilities=load_osm()
    with st.spinner(f"Fetching NASA FIRMS for {selected}..."):
        live=fetch_firms(days)
except Exception as e:
    st.error("ThermoWatch LIVE pipeline failed.")
    st.exception(e)
    st.stop()

# =========================
# ZERO-OBSERVATION STATE
# =========================
if live.empty:
    st.markdown(f"""
<div class="no-fire">
<div class="no-fire-title">🛰️ NO SATELLITE THERMAL OBSERVATIONS DETECTED</div>
<div class="no-fire-text">
NASA FIRMS returned no thermal observations inside the selected
<b>{html.escape(selected)}</b> window over the India monitoring region.
</div>
<div class="no-fire-text">
ThermoWatch is still <b>ONLINE</b>. This does not mean that no fire exists anywhere;
it means that no FIRMS observation was returned for the selected monitoring window.
</div>
</div>
""",unsafe_allow_html=True)

    a,b,c=st.columns(3)
    a.metric("Monitoring Window",selected)
    b.metric("Satellite Observations",0)
    c.metric("Pipeline Status","ONLINE")
    st.markdown(f"""
<div class="online"><b>STATUS:</b> ONLINE &nbsp;·&nbsp;
<b>WINDOW:</b> {html.escape(selected)} &nbsp;·&nbsp;
<b>SOURCE:</b> NASA FIRMS VIIRS NOAA-20 NRT &nbsp;·&nbsp;
<b>OBSERVATIONS:</b> 0</div>
""",unsafe_allow_html=True)
    st.info("No AI prediction or risk score is generated because there is no satellite observation to analyze. Select a wider window if needed.")
    st.stop()

# =========================
# ATTRIBUTION + AI
# =========================
with st.spinner("Matching hotspots with nearest OSM facilities..."):
    attributed=attribute(live,facilities)

with st.spinner("Running V5 AI inference and risk assessment..."):
    predictions=predict(attributed,model)

predictions["display_facility_name"]=predictions.apply(facility_label,axis=1)

latest=pd.to_datetime(predictions["acquisition_datetime"],errors="coerce",utc=True).max()
latest_text=fmt(latest)

st.markdown(f"""
<div class="online"><b>● LIVE PIPELINE COMPLETE</b>
&nbsp;·&nbsp; {len(predictions):,} observations analyzed
&nbsp;·&nbsp; Latest: {html.escape(latest_text)}</div>
""",unsafe_allow_html=True)

# =========================
# METRICS
# =========================
total=len(predictions)
critical=int((predictions.live_risk_category=="CRITICAL").sum())
high=int((predictions.live_risk_category=="HIGH").sum())
avg=float(predictions.live_risk_score.mean())
mx=float(predictions.live_risk_score.max())

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("LIVE HOTSPOTS",f"{total:,}")
m2.metric("CRITICAL",critical)
m3.metric("HIGH RISK",high)
m4.metric("AVG RISK",f"{avg:.1f}")
m5.metric("MAX RISK",f"{mx:.1f}")

# =========================
# FILTERS
# =========================
st.markdown('<div class="section">🔎 LIVE FILTERS</div>',unsafe_allow_html=True)
f1,f2,f3,f4=st.columns(4)

with f1:
    rf=st.selectbox("Risk",["All","CRITICAL","HIGH","MEDIUM","LOW"])
with f2:
    cf=st.selectbox("Confidence",["All","≥80%","≥60%","≥40%"])
with f3:
    sf=st.selectbox("AI Source",["All"]+sorted(predictions.predicted_source.astype(str).unique()))
with f4:
    pf=st.selectbox("Facility Proximity",["All"]+sorted(predictions.proximity_category.astype(str).unique()))

filtered=predictions.copy()
if rf!="All": filtered=filtered[filtered.live_risk_category==rf]
if cf!="All": filtered=filtered[filtered.confidence>=int(cf.replace("≥","").replace("%",""))]
if sf!="All": filtered=filtered[filtered.predicted_source.astype(str)==sf]
if pf!="All": filtered=filtered[filtered.proximity_category.astype(str)==pf]

# =========================
# MAP
# =========================
st.markdown('<div class="section">🗺️ INDIA LIVE THERMAL ZONES</div>',unsafe_allow_html=True)

mp=folium.Map(location=[22.5,79],zoom_start=5,tiles="CartoDB dark_matter")
heat=[]
cluster=MarkerCluster(name="LIVE Thermal Hotspots").add_to(mp)

for _,r in filtered.iterrows():
    lat=num(r.get("latitude"),np.nan); lon=num(r.get("longitude"),np.nan)
    if not(np.isfinite(lat) and np.isfinite(lon)): continue
    risk=num(r.get("live_risk_score"),0)
    heat.append([lat,lon,max(.1,risk/100)])
    cat=clean(r.get("live_risk_category"),"LOW")
    color={"CRITICAL":"red","HIGH":"orange","MEDIUM":"blue","LOW":"green"}.get(cat,"blue")
    popup=f"""
    <b>🔥 ThermoWatch Detection</b><br>
    Hotspot: {html.escape(clean(r.get("hotspot_id")))}<br>
    Facility: {html.escape(clean(r.get("display_facility_name")))}<br>
    AI Source: {html.escape(clean(r.get("predicted_source")))}<br>
    FRP: {num(r.get("frp")):.2f} MW<br>
    Confidence: {num(r.get("confidence")):.1f}%<br>
    Risk: {risk:.2f} ({html.escape(cat)})<br>
    Facility Distance: {num(r.get("distance_to_facility_km"),999):.3f} km<br>
    Observed: {html.escape(fmt(r.get("acquisition_datetime")))}
    """
    folium.Marker([lat,lon],tooltip=f"{cat} · Risk {risk:.1f}",
                  popup=folium.Popup(popup,max_width=330),
                  icon=folium.Icon(color=color,icon="fire",prefix="fa")).add_to(cluster)

if heat:
    HeatMap(heat,radius=18,blur=24,min_opacity=.35).add_to(mp)
folium.LayerControl().add_to(mp)
st_folium(mp,width=None,height=650,key="thermowatch_map")

# =========================
# CHARTS
# =========================
c1,c2=st.columns(2)
with c1:
    st.markdown('<div class="section">🔥 Risk Distribution</div>',unsafe_allow_html=True)
    st.bar_chart(filtered.live_risk_category.value_counts().reindex(["CRITICAL","HIGH","MEDIUM","LOW"],fill_value=0))
with c2:
    st.markdown('<div class="section">🧠 AI Source Classification</div>',unsafe_allow_html=True)
    st.bar_chart(filtered.predicted_source.value_counts())

# =========================
# TOP EVENTS
# =========================
st.markdown('<div class="section">🚨 HIGHEST RISK LIVE DETECTIONS</div>',unsafe_allow_html=True)
cols=["hotspot_id","display_facility_name","latitude","longitude","acquisition_datetime","frp",
      "predicted_source","confidence","distance_to_facility_km","proximity_category",
      "facility_match_quality","live_risk_score","live_risk_category"]
table=filtered[[c for c in cols if c in filtered.columns]].sort_values("live_risk_score",ascending=False).head(20).copy()
table=table.rename(columns={
"hotspot_id":"Hotspot ID","display_facility_name":"Facility","latitude":"Latitude","longitude":"Longitude",
"acquisition_datetime":"Observed At","frp":"FRP (MW)","predicted_source":"AI Source","confidence":"Confidence %",
"distance_to_facility_km":"Facility Distance (km)","proximity_category":"Proximity",
"facility_match_quality":"Attribution","live_risk_score":"Risk Score","live_risk_category":"Risk"})
if "Observed At" in table: table["Observed At"]=table["Observed At"].apply(fmt)
st.dataframe(table,use_container_width=True,height=430,hide_index=True)

# =========================
# DETAILS + EXPORT
# =========================
with st.expander("📍 View selected detection details"):
    if not filtered.empty:
        sid=st.selectbox("Detection",filtered.hotspot_id.astype(str).tolist())
        row=filtered[filtered.hotspot_id.astype(str)==sid].iloc[0]
        st.write({
            "Hotspot ID":clean(row.get("hotspot_id")),
            "Facility":clean(row.get("display_facility_name")),
            "AI Source":clean(row.get("predicted_source")),
            "Confidence %":num(row.get("confidence")),
            "FRP MW":num(row.get("frp")),
            "Risk Score":num(row.get("live_risk_score")),
            "Risk":clean(row.get("live_risk_category")),
            "Facility Distance km":num(row.get("distance_to_facility_km"),999),
            "Observed":fmt(row.get("acquisition_datetime"))
        })

st.markdown('<div class="section">⬇️ EXPORT LIVE RESULTS</div>',unsafe_allow_html=True)
st.download_button("⬇️ Download LIVE ThermoWatch CSV",
                   filtered.to_csv(index=False).encode("utf-8"),
                   file_name=f"thermowatch_live_{days}day.csv",
                   mime="text/csv",use_container_width=True)

st.markdown(f"""
<div class="footer">
THERMOWATCH AI · V5 MODEL · LIVE SATELLITE MONITORING |
WINDOW: {html.escape(selected)} |
OBSERVATIONS: {len(predictions):,} |
VISIBLE: {len(filtered):,} |
LATEST: {html.escape(latest_text)} |
SOURCE: NASA FIRMS VIIRS NOAA-20 NRT |
FACILITY CONTEXT: OPENSTREETMAP
</div>
""",unsafe_allow_html=True)
