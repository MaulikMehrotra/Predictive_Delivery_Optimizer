# app.py
"""
Interactive Streamlit dashboard for Predictive Delivery Optimizer
Requirements covered:
- Python + Streamlit
- Data analysis and visualization
- Interactivity (filters, user input, export)
- Code quality and comments
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

st.title("📦 Predictive Delivery Optimizer Dashboard")
st.markdown("""
Analyze delivery performance, visualize logistics insights, and predict potential delays.
---
""")

# --------------------------
# LOAD DATA AND MODEL
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("processed_data.csv")

@st.cache_resource
def load_model():
    model_artifacts = joblib.load("delivery_delay_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model_artifacts, encoders

try:
    df = load_data()
    model_artifacts, encoders = load_model()
    st.success("Data and model loaded successfully.")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

model = model_artifacts["model"]
scaler = model_artifacts["scaler"]
feature_columns = model_artifacts["feature_columns"]

# --------------------------
# SIDEBAR FILTERS
# --------------------------
st.sidebar.header("🔍 Filters")

# Safe categorical filters
def get_unique(col):
    return sorted(df[col].dropna().unique().tolist())

filter_priority = st.sidebar.multiselect("Priority", get_unique("Priority"))
filter_segment = st.sidebar.multiselect("Customer Segment", get_unique("Customer_Segment"))
filter_carrier = st.sidebar.multiselect("Carrier", get_unique("Carrier"))
filter_category = st.sidebar.multiselect("Product Category", get_unique("Product_Category"))

filtered_df = df.copy()

if filter_priority:
    filtered_df = filtered_df[filtered_df["Priority"].isin(filter_priority)]
if filter_segment:
    filtered_df = filtered_df[filtered_df["Customer_Segment"].isin(filter_segment)]
if filter_carrier:
    filtered_df = filtered_df[filtered_df["Carrier"].isin(filter_carrier)]
if filter_category:
    filtered_df = filtered_df[filtered_df["Product_Category"].isin(filter_category)]

st.write(f"### Showing {len(filtered_df)} records after filtering")

# --------------------------
# 1️⃣ DATA ANALYSIS INSIGHTS
# --------------------------
st.header("📊 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Customer Rating", round(filtered_df["Customer_Rating"].mean(), 2))
col2.metric("Avg Delay (days)", round(filtered_df["Delay_Days"].mean(), 2))
col3.metric("Delay %", f"{(filtered_df['Delayed'].mean() * 100):.2f}%")
col4.metric("Avg Delivery Cost (₹)", round(filtered_df["Delivery_Cost_INR"].mean(), 2))

# --------------------------
# 2️⃣ VISUALIZATIONS (>=4 types)
# --------------------------

st.header("📈 Visualizations")

tab1, tab2, tab3, tab4 = st.tabs([
    "Delivery Delay by Priority",
    "Distance vs Delay",
    "Satisfaction Distribution",
    "Feature Importance"
])

# Chart 1: Delivery Delay by Priority
with tab1:
    if "Priority" in filtered_df.columns:
        fig1 = px.box(filtered_df, x="Priority", y="Delay_Days", color="Priority",
                      title="Delivery Delay by Priority Level", points="all")
        st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Distance vs Delay
with tab2:
    if "Distance_KM" in filtered_df.columns:
        fig2 = px.scatter(filtered_df, x="Distance_KM", y="Delay_Days", color="Carrier",
                          size="Delivery_Cost_INR", title="Distance vs Delay Days")
        st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Customer satisfaction histogram
with tab3:
    if "Satisfaction_Index" in filtered_df.columns:
        fig3 = px.histogram(filtered_df, x="Satisfaction_Index", nbins=30, color="Delayed",
                            title="Customer Satisfaction Distribution", marginal="box")
        st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Feature importances (load from saved PNG if available)
with tab4:
    try:
        fig4 = plt.imread("feature_importances.png")
        st.image(fig4, caption="Model Feature Importances", use_container_width=True)
    except:
        st.info("Feature importance plot not found. Run main.py first.")

# --------------------------
# 3️⃣ INTERACTIVITY: USER PREDICTION
# --------------------------
st.header("🧠 Predict Delivery Delay")

st.markdown("Enter delivery details below to predict if it will be delayed:")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    distance = col1.number_input("Distance (KM)", min_value=0.0, step=0.1)
    traffic = col2.number_input("Traffic Delay (minutes)", min_value=0.0, step=1.0)
    order_value = col1.number_input("Order Value (₹)", min_value=0.0, step=10.0)
    delivery_cost = col2.number_input("Delivery Cost (₹)", min_value=0.0, step=5.0)

    priority = col1.selectbox("Priority", get_unique("Priority"))
    carrier = col2.selectbox("Carrier", get_unique("Carrier"))
    segment = col1.selectbox("Customer Segment", get_unique("Customer_Segment"))
    category = col2.selectbox("Product Category", get_unique("Product_Category"))
    weather_impact = st.selectbox(
    "Weather Impact",
    ["None", "Mild", "Severe"],
    index=0
)


    submitted = st.form_submit_button("Predict Delay")

if submitted:
    input_data = pd.DataFrame([{
        "Distance_KM": distance,
        "Traffic_Delay_Minutes": traffic,
        "Order_Value_INR": order_value,
        "Delivery_Cost_INR": delivery_cost,
        "Priority": priority,
        "Carrier": carrier,
        "Customer_Segment": segment,
        "Product_Category": category,
        "Fuel_Cost_per_KM": 0.0,
        "Delivery_Efficiency": 0.0,
        "Revenue_per_KM": 0.0,
        "Weather_Impact": [weather_impact]
    }])

    # Encode categorical inputs using saved encoders
    for col in input_data.select_dtypes(include="object").columns:
        if col in encoders:
            le = encoders[col]
            input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            input_data[col] = le.transform(input_data[col])
        else:
            input_data[col] = 0

    # Scale numerics
    num_cols = input_data.select_dtypes(include=np.number).columns.tolist()
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Predict
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if pred == 1:
        st.error(f"⚠️ Predicted: DELAYED ({prob:.1f}% probability)")
    else:
        st.success(f"✅ Predicted: ON TIME ({100 - prob:.1f}% probability)")

# --------------------------
# 4️⃣ EXPORT OPTION
# --------------------------
st.header("💾 Export Data")
csv_export = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered dataset as CSV",
    data=csv_export,
    file_name="filtered_deliveries.csv",
    mime="text/csv",
)
