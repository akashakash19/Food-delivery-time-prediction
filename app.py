import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import folium
from streamlit_folium import st_folium

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model (2).pkl")
encoders = pickle.load(open("encoders.pkl", "rb"))

st.set_page_config(page_title="Delivery AI", layout="wide")

# -------------------------
# TITLE
# -------------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff7e00, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Smart Delivery Predictor</div>", unsafe_allow_html=True)

# -------------------------
# LOCATION INPUT
# -------------------------
st.subheader("📍 Location Input")

colA, colB = st.columns(2)

with colA:
    rest_lat = st.number_input("Restaurant Latitude", value=13.08)
    rest_lon = st.number_input("Restaurant Longitude", value=80.27)

with colB:
    user_lat = st.number_input("Delivery Latitude", value=12.97)
    user_lon = st.number_input("Delivery Longitude", value=77.59)

# -------------------------
# DISTANCE CALCULATION
# -------------------------
from geopy.distance import geodesic
distance = geodesic((rest_lat, rest_lon), (user_lat, user_lon)).km

# -------------------------
# MAP
# -------------------------
m = folium.Map(location=[rest_lat, rest_lon], zoom_start=10)

folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color="green")).add_to(m)
folium.Marker([user_lat, user_lon], tooltip="Customer", icon=folium.Icon(color="red")).add_to(m)

folium.PolyLine([[rest_lat, rest_lon], [user_lat, user_lon]], color="blue").add_to(m)

st_folium(m, width=800, height=350)

# -------------------------
# INPUT FEATURES
# -------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 25)
    rating = st.slider("Rating", 1.0, 5.0, 4.0)
    vehicle_condition = st.slider("Vehicle Condition", 1, 10, 5)
    deliveries = st.selectbox("Multiple Deliveries", [0,1,2,3])

with col2:
    prep_time = st.slider("Preparation Time", 5, 60, 20)

# TIME
col3, col4 = st.columns(2)

with col3:
    order_date = st.date_input("Order Date", datetime.today())
    order_time = st.time_input("Order Time")

with col4:
    picked_time = st.time_input("Picked Time")

# CATEGORIES
col5, col6, col7 = st.columns(3)

with col5:
    weather = st.selectbox("Weather", encoders["Weather_conditions"].classes_)

with col6:
    traffic = st.selectbox("Traffic", encoders["Road_traffic_density"].classes_)

with col7:
    vehicle = st.selectbox("Vehicle", encoders["Type_of_vehicle"].classes_)

col8, col9, col10 = st.columns(3)

with col8:
    order_type = st.selectbox("Order Type", encoders["Type_of_order"].classes_)

with col9:
    city = st.selectbox("City", encoders["City"].classes_)

with col10:
    festival = st.selectbox("Festival", encoders["Festival"].classes_)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def create_features():
    order_dt = datetime.combine(order_date, order_time)
    picked_dt = datetime.combine(order_date, picked_time)

    if picked_dt < order_dt:
        picked_dt += pd.Timedelta(days=1)

    order_hour = order_dt.hour
    prepare_time = (picked_dt - order_dt).total_seconds() / 60
    is_weekend = int(order_dt.weekday() >= 5)

    return order_hour, prepare_time, is_weekend

# -------------------------
# PREDICT
# -------------------------
if st.button("🚀 Predict Delivery Time"):

    with st.spinner("Calculating..."):
        order_hour, prepare_time, is_weekend = create_features()

        input_dict = {
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "order_hour": order_hour,
            "order_prepare_time": prepare_time,
            "Weather_conditions": encoders["Weather_conditions"].transform([weather])[0],
            "Road_traffic_density": encoders["Road_traffic_density"].transform([traffic])[0],
            "Vehicle_condition": vehicle_condition,
            "Type_of_order": encoders["Type_of_order"].transform([order_type])[0],
            "Type_of_vehicle": encoders["Type_of_vehicle"].transform([vehicle])[0],
            "multiple_deliveries": deliveries,
            "Festival": encoders["Festival"].transform([festival])[0],
            "City": encoders["City"].transform([city])[0],
            "distance": distance,
            "is_weekend": is_weekend
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

    # ETA RANGE
    lower = prediction - 5
    upper = prediction + 5

    st.success(f"⏳ Delivery Time: {lower:.0f} - {upper:.0f} minutes")

    # STATUS
    if prediction < 20:
        st.success("⚡ Fast Delivery")
    elif prediction < 40:
        st.warning("🚗 Moderate Delivery")
    else:
        st.error("🐢 Delay Expected")
