import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import random

# --- Page Setup ---
st.set_page_config(page_title="Amazon Delivery Time Estimator", page_icon="📦", layout="wide")

# --- Load Model ---
try:
    model = joblib.load("delivery_time_model.pkl")
except:
    st.error("Model file not found. Please train the model first.")
    st.stop()

# --- Title ---
st.title("📦 Amazon Delivery Time Estimator")
st.caption("Estimate your expected delivery time based on real-world conditions.")

# --- Sidebar ---
st.sidebar.header("⚙️ Quick Tools")
if st.sidebar.button("✨ Auto-Fill Sample Data"):
    # Generate random but realistic values
    st.session_state.random_inputs = {
        "area": random.choice(["Urban", "Semi-Urban", "Rural"]),
        "category": random.choice(["Small", "Medium", "Large"]),
        "traffic": random.choice(["Low", "Medium", "High"]),
        "weather": random.choice(["Sunny", "Cloudy", "Rainy", "Stormy"]),
        "vehicle": random.choice(["Bike", "Car", "Van"]),
        "agent_age": random.randint(20, 45),
        "agent_rating": round(random.uniform(3.0, 5.0), 1),
        "distance": round(random.uniform(2, 25), 1)
    }
else:
    st.session_state.random_inputs = st.session_state.get("random_inputs", {})

r = st.session_state.random_inputs

# --- Inputs ---
st.header("📝 Enter Order Information")

col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("Delivery Area Type", ["Urban", "Semi-Urban", "Rural"],
                        index=["Urban", "Semi-Urban", "Rural"].index(r.get("area", "Urban")))
    category = st.selectbox("Order Size", ["Small", "Medium", "Large"],
                            index=["Small", "Medium", "Large"].index(r.get("category", "Small")))
    traffic = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"],
                           index=["Low", "Medium", "High"].index(r.get("traffic", "Low")))
    weather = st.selectbox("Weather Today", ["Sunny", "Cloudy", "Rainy", "Stormy"],
                           index=["Sunny", "Cloudy", "Rainy", "Stormy"].index(r.get("weather", "Sunny")))

with col2:
    vehicle = st.selectbox("Delivery Vehicle", ["Bike", "Car", "Van"],
                           index=["Bike", "Car", "Van"].index(r.get("vehicle", "Bike")))
    agent_age = st.slider("Delivery Partner Age", 18, 60, r.get("agent_age", 25))
    agent_rating = st.slider("Delivery Partner Rating", 1.0, 5.0, r.get("agent_rating", 4.5), step=0.1)
    distance = st.number_input("Approximate Delivery Distance (km)", 0.5, 50.0, r.get("distance", 5.0), step=0.5)

# --- Preparation Time ---
def estimate_prep_time(category, distance):
    if category == "Small": return 18 + distance * 0.2
    elif category == "Medium": return 25 + distance * 0.25
    else: return 35 + distance * 0.3

prep_time = estimate_prep_time(category, distance)
st.info(f"🕒 Estimated Order Preparation Time: **{prep_time:.1f} minutes**")

# --- Encoders ---
encode_maps = {
    "Weather": {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3},
    "Traffic": {"Low": 0, "Medium": 1, "High": 2},
    "Vehicle": {"Bike": 0, "Car": 1, "Van": 2},
    "Area": {"Urban": 0, "Semi-Urban": 1, "Rural": 2},
    "Category": {"Small": 0, "Medium": 1, "Large": 2}
}

# --- Prediction ---
if st.button("🚀 Estimate Delivery Time"):
    try:
        input_data = {
            "Agent_Age": agent_age, "Agent_Rating": agent_rating,
            "Store_Latitude": 12.9716, "Store_Longitude": 77.5946,
            "Drop_Latitude": 12.9352, "Drop_Longitude": 77.6245,
            "Weather": encode_maps["Weather"][weather],
            "Traffic": encode_maps["Traffic"][traffic],
            "Vehicle": encode_maps["Vehicle"][vehicle],
            "Area": encode_maps["Area"][area],
            "Category": encode_maps["Category"][category],
            "Prep_Time": prep_time, "Distance_km": distance
        }

        df = pd.DataFrame([input_data])
        pred_minutes = model.predict(df)[0]
        pred_hours = pred_minutes / 60

        st.success(f"✅ Estimated Delivery Time: **{pred_hours:.2f} hours**")

        # --- Visualization ---
        st.subheader("📊 Delivery Insights")
        col1, col2 = st.columns(2)

        # --- Bar Chart ---
        with col1:
            bar_data = pd.DataFrame({
                "Factors": ["Distance", "Traffic", "Preparation", "Rating"],
                "Impact": [distance, encode_maps["Traffic"][traffic]*10, prep_time, agent_rating*10]
            })
            fig1, ax1 = plt.subplots(figsize=(5,3))
            sns.barplot(x="Factors", y="Impact", data=bar_data, ax=ax1, palette="coolwarm")
            ax1.set_title("Key Factors Affecting Delivery")
            ax1.set_ylabel("Relative Impact")
            st.pyplot(fig1)

        # --- Dendrogram ---
        with col2:
            sample_data = np.random.rand(10, 4)
            linked = linkage(sample_data, 'ward')
            fig2, ax2 = plt.subplots(figsize=(5,3))
            dendrogram(linked, ax=ax2, color_threshold=1.5)
            ax2.set_title("Order Pattern Similarities")
            ax2.set_xlabel("Orders")
            ax2.set_ylabel("Distance Metric")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
