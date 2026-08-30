# 🔥 ThermoWatch AI

### AI-Powered Satellite Thermal Hotspot Intelligence & Industrial Risk Detection

> **Detect → Attribute → Classify → Score → Investigate**

ThermoWatch AI is a geospatial intelligence platform that transforms near-real-time satellite thermal hotspot observations into actionable fire-risk intelligence.

It combines **NASA FIRMS satellite observations, OpenStreetMap facility data, spatial attribution, machine-learning source classification, and risk scoring** to identify potentially significant thermal events across India.

---

## 🚨 Why ThermoWatch?

Satellite fire datasets can tell us:

> "A thermal anomaly was detected here."

But that alone is not enough.

ThermoWatch attempts to answer the next questions:

- 🔥 Where is the thermal hotspot?
- 🕐 When was it actually observed?
- 🏭 Is there a nearby industrial or power facility?
- 📍 How close is the hotspot to that facility?
- 🤖 What type of source does the AI predict?
- 📊 How confident is the prediction?
- ⚠️ How severe is the calculated risk?
- 🚨 Which locations should be investigated first?

---

# 🧠 System Overview

```text
                 NASA FIRMS
                     │
                     ▼
          Satellite Hotspot Data
                     │
                     ▼
             India Filtering
                     │
                     ▼
           Temporal / Spatial
              Event Processing
                     │
                     ▼
            OpenStreetMap (OSM)
             Facility Database
                     │
                     ▼
          Nearest Facility Matching
                     │
                     ▼
            Facility Attribution
                     │
                     ▼
             V5 ML Classifier
                     │
                     ▼
          Source Classification
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   POWER_GENERATION          INDUSTRIAL
          │                     │
          └──────────┬──────────┘
                     ▼
                Risk Engine
                     │
                     ▼
           Risk Score 0–100
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
       LOW         MEDIUM       HIGH
                                  │
                                  ▼
                              CRITICAL
                     │
                     ▼
              ThermoWatch UI
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Thermal Map   Alerts Feed   Inspector
                             
                     │
                     ▼
              Incident Report
