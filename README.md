# Electricity Grid Failure & Outage Prediction

Predictive maintenance platform for Nigerian Distribution Companies (DisCos). Forecasts transformer and feeder failures 90 days in advance using asset condition telemetry, loading profiles, weather exposure, and fault history. Reduces unplanned outages and improves SAIDI/SAIFI scores.

---

## Features

- 4-tier risk classification: Low / Moderate / High / Critical
- 11 DisCo coverage including Eko, Ikeja, Abuja, Enugu, Kano
- Gradient Boosting model with 13 asset condition features
- Interactive Streamlit dashboard: overview, risk map, single-asset predictor
- Confusion matrix and feature importance analysis

---

## Setup & Run

```bash
git clone https://github.com/Momahmoses/power-grid-failure.git
cd power-grid-failure
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Tech Stack

`Python` · `scikit-learn` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`

---

## Author

**Momah Moses** — [github.com/Momahmoses](https://github.com/Momahmoses)
