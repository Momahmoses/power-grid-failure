# Electricity Grid Failure Prediction — Predictive Maintenance Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predictive maintenance platform for Nigerian Distribution Companies (DisCos) — forecasting transformer and feeder failures 90 days in advance to reduce unplanned outages and improve SAIDI/SAIFI reliability scores.

---

## Problem Statement

Nigeria generates only ~4,000 MW for 220 million people, and what is generated is further lost to aging grid infrastructure. Nigerian DisCos currently deploy maintenance reactively. This platform shifts operations to proactive, data-driven maintenance scheduling.

---

## Features

| Feature | Description |
|---------|-------------|
| 90-Day Failure Forecasting | Gradient Boosting on 13 asset condition features |
| 4-Tier Risk Classification | Low / Moderate / High / Critical labelling |
| 11 DisCo Coverage | Eko, Ikeja, Abuja, Enugu, Kano, Port Harcourt and more |
| Interactive Dashboard | Overview, risk map, single-asset predictor |
| Model Diagnostics | Confusion matrix and feature importance analysis |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | Gradient Boosting, scikit-learn |
| Geospatial | GeoPandas, Folium |
| Dashboard | Streamlit, Plotly |
| Data | pandas, NumPy |

---

## Project Structure

```
power-grid-failure/
├── src/
│   ├── data_loader.py     # Asset telemetry and weather data ingestion
│   ├── model.py           # Gradient Boosting training and prediction
│   └── visualize.py       # Risk maps and reliability charts
├── streamlit_app.py       # Dashboard entry point
├── .streamlit/config.toml
├── requirements.txt
└── runtime.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/power-grid-failure.git
cd power-grid-failure
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Data Sources

- NERC DisCo performance reports
- TCN transmission infrastructure register
- NIMET weather data (temperature, humidity, lightning strikes)
- DisCo asset condition registers (transformer age, load, fault history)

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
