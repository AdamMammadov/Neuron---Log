import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

from pathlib import Path

from src.visualization.map_builder import (
    build_route_map_data
)

from src.ai.recommendation_engine import (
    generate_ai_recommendations
)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Neuron Logistics AI",
    layout="wide"
)

st.title("🚚 NEURON LOGISTICS AI PLATFORM")

st.markdown(
    "AI-Powered Demand Forecasting & Shipment Optimization System"
)

# ==========================================
# LOAD FILES
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

forecast_path = (
    BASE_DIR /
    "outputs" /
    "Tahminlenen_Talep.xlsx"
)

plan_path = (
    BASE_DIR /
    "outputs" /
    "Arac_Planlama.xlsx"
)

forecast_df = pd.read_excel(forecast_path)

plan_df = pd.read_excel(plan_path)

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🎛️ AI Control Panel")

selected_risk = st.sidebar.multiselect(
    "Select Risk Level",
    options=plan_df["Risk Level"].unique(),
    default=plan_df["Risk Level"].unique()
)

selected_vehicle = st.sidebar.multiselect(
    "Select Vehicle Type",
    options=plan_df["Araç Türü"].unique(),
    default=plan_df["Araç Türü"].unique()
)

filtered_df = plan_df[
    (
        plan_df["Risk Level"].isin(
            selected_risk
        )
    )
    &
    (
        plan_df["Araç Türü"].isin(
            selected_vehicle
        )
    )
]

# ==========================================
# KPI SECTION
# ==========================================

total_forecast = round(
    forecast_df["Tahminlenen Desi"].sum(),
    2
)

total_cost = round(
    filtered_df["Toplam Maliyet"].sum(),
    2
)

avg_utilization = round(
    filtered_df["Doluluk Oranı"].mean(),
    2
)

shipment_count = len(filtered_df)

high_risk = len(
    filtered_df[
        filtered_df["Risk Level"] == "HIGH"
    ]
)

# KPI CARDS

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Forecasted Desi",
    f"{total_forecast:,.0f}"
)

col2.metric(
    "Logistics Cost",
    f"{total_cost:,.0f} TL"
)

col3.metric(
    "Vehicle Utilization",
    avg_utilization
)

col4.metric(
    "Shipment Count",
    shipment_count
)

col5.metric(
    "High Risk Shipments",
    high_risk
)

st.divider()

# ==========================================
# COST ANALYSIS
# ==========================================

st.subheader("📊 Cost Analysis")

cost_by_vehicle = (
    filtered_df
    .groupby("Araç Türü")["Toplam Maliyet"]
    .sum()
    .reset_index()
)

fig_cost = px.bar(
    cost_by_vehicle,
    x="Araç Türü",
    y="Toplam Maliyet",
    title="Total Cost by Vehicle Type"
)

st.plotly_chart(
    fig_cost,
    use_container_width=True
)

# ==========================================
# RISK DISTRIBUTION
# ==========================================

st.subheader("⚠️ Risk Distribution")

risk_chart = (
    filtered_df["Risk Level"]
    .value_counts()
    .reset_index()
)

risk_chart.columns = [
    "Risk Level",
    "Count"
]

fig_risk = px.pie(
    risk_chart,
    names="Risk Level",
    values="Count",
    title="Shipment Risk Levels"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# ==========================================
# UTILIZATION ANALYSIS
# ==========================================

st.subheader("🚛 Vehicle Utilization Analysis")

fig_util = px.histogram(
    filtered_df,
    x="Doluluk Oranı",
    nbins=20,
    title="Vehicle Utilization Distribution"
)

st.plotly_chart(
    fig_util,
    use_container_width=True
)

# ==========================================
# TOP EXPENSIVE ROUTES
# ==========================================

st.subheader("💰 Most Expensive Routes")

expensive_routes = (
    filtered_df
    .groupby(
        [
            "Çıkış Transfer Merkezi",
            "Varış Transfer Merkezi"
        ]
    )["Toplam Maliyet"]
    .sum()
    .reset_index()
)

expensive_routes = expensive_routes.sort_values(
    by="Toplam Maliyet",
    ascending=False
).head(10)

fig_routes = px.bar(
    expensive_routes,
    x="Toplam Maliyet",
    y="Varış Transfer Merkezi",
    orientation="h",
    title="Top Expensive Shipment Routes"
)

st.plotly_chart(
    fig_routes,
    use_container_width=True
)

# ==========================================
# AI ROUTE MAP
# ==========================================

st.subheader("🗺️ AI Logistics Route Map")

map_df = build_route_map_data(
    filtered_df
)

layer = pdk.Layer(
    "ArcLayer",
    data=map_df,

    get_source_position='[from_lon, from_lat]',
    get_target_position='[to_lon, to_lat]',

    get_source_color="color",
    get_target_color="color",

    auto_highlight=True,

    width_scale=0.0001,

    get_width="desi / 100",

    pickable=True
)

view_state = pdk.ViewState(
    latitude=39.0,
    longitude=35.0,
    zoom=5
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "text":
        "{origin} ➜ {destination}\n"
        "Risk: {risk}\n"
        "Desi: {desi}"
    }
)

st.pydeck_chart(deck)

# ==========================================
# AI INSIGHTS PANEL
# ==========================================

st.subheader("🧠 AI Insights")

if avg_utilization >= 0.85:
    st.success(
        "Fleet efficiency is excellent."
    )

elif avg_utilization >= 0.70:
    st.warning(
        "Fleet efficiency is acceptable."
    )

else:
    st.error(
        "Fleet efficiency needs improvement."
    )

if high_risk > 20:
    st.error(
        "High operational risk detected."
    )

else:
    st.success(
        "Operational risk is under control."
    )

if total_cost < 9000000:
    st.success(
        "Cost optimization successful."
    )

else:
    st.warning(
        "Transportation costs are high."
    )

# ==========================================
# AI RECOMMENDATION ENGINE
# ==========================================

st.subheader("🤖 AI Recommendations")

recommendations = generate_ai_recommendations(
    filtered_df
)

for rec in recommendations:

    st.info(rec)

# ==========================================
# RAW DATA VIEW
# ==========================================

st.subheader("📄 Shipment Plan Data")

st.dataframe(filtered_df)