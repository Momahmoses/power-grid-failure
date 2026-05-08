"""
Electricity Grid Failure & Outage Prediction
=============================================
Predictive maintenance platform for Nigerian DisCos. Forecasts transformer
and feeder failures 90 days in advance using asset condition, load, and weather data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_grid_dataset
from model import (
    FEATURE_COLS, METRICS_PATH, MODEL_PATH,
    load_model, predict_failure, save_model, train,
)

st.set_page_config(page_title="Grid Failure Prediction | Nigeria", page_icon="⚡", layout="wide")

TIER_COLORS = {"Critical": "#8E1B1B", "High": "#E74C3C", "Moderate": "#F39C12", "Low": "#2ECC71"}


@st.cache_resource(show_spinner="Training grid failure model…")
def get_model_and_data():
    df = generate_grid_dataset()
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        pipeline = load_model()
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        pipeline, metrics = train(df)
        save_model(pipeline, metrics)
    probs = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
    df["failure_probability"] = probs
    df["risk_tier"] = pd.cut(probs, bins=[0, 0.30, 0.50, 0.70, 1.0], labels=["Low", "Moderate", "High", "Critical"]).astype(str)
    return pipeline, metrics, df


pipeline, metrics, df = get_model_and_data()

with st.sidebar:
    st.title("⚡ Grid Failure Prediction")
    st.caption("Nigerian DisCo Predictive Maintenance")
    st.divider()
    page = st.radio("Navigation", ["Overview", "Asset Risk Map", "Predict an Asset", "Model Performance"], label_visibility="collapsed")
    st.divider()
    disco_filter = st.multiselect("Filter by DisCo", sorted(df["disco"].unique()), default=sorted(df["disco"].unique()))
    asset_filter = st.multiselect("Filter by Asset Type", sorted(df["asset_type"].unique()), default=sorted(df["asset_type"].unique()))

df_f = df[(df["disco"].isin(disco_filter)) & (df["asset_type"].isin(asset_filter))]

if page == "Overview":
    st.title("Electricity Grid Failure & Outage Prediction")
    st.markdown("Predictive maintenance platform that forecasts transformer and feeder failures 90 days ahead, enabling proactive intervention by Nigerian DisCos.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets Monitored", f"{len(df_f):,}")
    critical_count = (df_f["risk_tier"].isin(["Critical", "High"])).sum()
    c2.metric("Critical/High Risk Assets", f"{critical_count:,}", delta=f"{critical_count/len(df_f):.1%} of fleet", delta_color="inverse")
    c3.metric("Mean Loading (%)", f"{df_f['loading_pct'].mean():.1f}%")
    c4.metric("Model ROC-AUC", f"{metrics['roc_auc_test']:.4f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        disco_risk = df_f.groupby("disco")["failure_probability"].mean().reset_index().sort_values("failure_probability", ascending=False)
        fig = px.bar(disco_risk, x="disco", y="failure_probability", color="failure_probability",
                     color_continuous_scale="Reds", title="Mean Failure Probability by DisCo",
                     labels={"failure_probability": "P(Failure in 90d)", "disco": "Distribution Company"}, height=420)
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        tier_ct = df_f["risk_tier"].value_counts().reset_index()
        fig2 = px.pie(tier_ct, values="count", names="risk_tier",
                      color="risk_tier", color_discrete_map=TIER_COLORS,
                      title="Asset Fleet Risk Distribution", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df_f.sample(min(2000, len(df_f)), random_state=42),
                          x="loading_pct", y="age_years", color="risk_tier",
                          color_discrete_map=TIER_COLORS, opacity=0.55,
                          title="Loading % vs Asset Age by Risk Tier", height=380,
                          labels={"loading_pct": "Loading (%)", "age_years": "Age (years)"})
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fi = metrics["feature_importance"]
        fi_s = sorted(fi.items(), key=lambda x: x[1])
        fig4 = go.Figure(go.Bar(x=[v for _, v in fi_s], y=[k for k, _ in fi_s], orientation="h", marker_color="#E74C3C"))
        fig4.update_layout(title="Feature Importance", height=380, margin={"l": 220})
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Asset Risk Map":
    st.title("Grid Asset Risk Map")
    sample = df_f.sample(min(1500, len(df_f)), random_state=42)
    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color="risk_tier", color_discrete_map=TIER_COLORS,
        size="failure_probability", size_max=14,
        hover_name="disco",
        hover_data={"asset_type": True, "failure_probability": ":.2%", "risk_tier": True, "latitude": False, "longitude": False},
        mapbox_style="carto-positron", zoom=5,
        center={"lat": 7.5, "lon": 7.0},
        title="Power Grid Asset Failure Risk Map — Nigeria", height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Predict an Asset":
    st.title("Predict Failure Risk for a Grid Asset")
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.slider("Asset Age (years)", 0.5, 40.0, 8.0, 0.5)
        loading_pct = st.slider("Loading (%)", 5.0, 150.0, 80.0, 1.0)
        voltage_deviation_pct = st.slider("Voltage Deviation (%)", 0.0, 20.0, 4.0, 0.1)
        ambient_temp_c = st.slider("Ambient Temperature (°C)", 20.0, 45.0, 33.0, 0.5)
        humidity_pct = st.slider("Humidity (%)", 20.0, 98.0, 70.0, 1.0)
        rainfall_last30d_mm = st.slider("Rainfall Last 30 Days (mm)", 0.0, 400.0, 80.0, 5.0)
        months_since_last_maintenance = st.slider("Months Since Last Maintenance", 0, 48, 12, 1)
    with col2:
        n_faults_12months = st.number_input("Faults in Last 12 Months", 0, 20, 1)
        n_faults_3years = st.number_input("Faults in Last 3 Years", int(n_faults_12months), 40, max(5, int(n_faults_12months)))
        harmonic_distortion_pct = st.slider("Harmonic Distortion (%)", 0.0, 25.0, 3.5, 0.1)
        corrosion_index = st.slider("Corrosion Index (0=none, 1=severe)", 0.0, 1.0, 0.3, 0.01)
        overload_events_6months = st.number_input("Overload Events (6 months)", 0, 20, 2)
        insulation_resistance_mohm = st.slider("Insulation Resistance (MΩ)", 0.1, 1000.0, 120.0, 1.0)

    if st.button("Assess Failure Risk", type="primary"):
        result = predict_failure(pipeline, {
            "age_years": age_years, "loading_pct": loading_pct,
            "voltage_deviation_pct": voltage_deviation_pct, "ambient_temp_c": ambient_temp_c,
            "humidity_pct": humidity_pct, "rainfall_last30d_mm": rainfall_last30d_mm,
            "months_since_last_maintenance": months_since_last_maintenance,
            "n_faults_12months": n_faults_12months, "n_faults_3years": n_faults_3years,
            "harmonic_distortion_pct": harmonic_distortion_pct, "corrosion_index": corrosion_index,
            "overload_events_6months": overload_events_6months,
            "insulation_resistance_mohm": insulation_resistance_mohm,
        })
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Failure Probability (90d)", f"{result['failure_probability']:.2%}")
        r2.metric("Prediction", "Failure Likely" if result["failure_predicted"] else "Stable")
        r3.metric("Risk Tier", result["risk_tier"])
        if result["risk_tier"] in ("Critical", "High"):
            st.error(f"⚠️ {result['risk_tier'].upper()} RISK: Schedule immediate preventive maintenance or replacement.")
        elif result["risk_tier"] == "Moderate":
            st.warning("Moderate risk. Plan maintenance within the next 30 days.")
        else:
            st.success("Low risk. Continue routine monitoring schedule.")

elif page == "Model Performance":
    st.title("Model Performance — Gradient Boosting Classifier")
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{metrics['roc_auc_test']:.4f}")
    m2.metric("CV AUC", f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    m3.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m4.metric("Failure F1-Score", f"{metrics['f1_failure']:.4f}")

    cm = metrics["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=np.array(cm), x=["No Failure", "Failure"], y=["No Failure", "Failure"],
        colorscale="Reds", showscale=False,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont={"size": 18},
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full Metrics JSON"):
        st.json(metrics)
