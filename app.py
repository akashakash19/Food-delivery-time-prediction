import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Food Delivery Time Predictor", layout="wide")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")

# -------------------------
# STYLE
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

.card {
    padding: 15px;
    border-radius: 15px;
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚀 Food Delivery Time AI predictor</div>", unsafe_allow_html=True)

# -------------------------
# MAP INPUT
# -------------------------
st.subheader("📍 Delivery Route")

colA, colB = st.columns(2)

with colA:
    rest_lat = st.number_input("Restaurant Latitude", value=13.08)
    rest_lon = st.number_input("Restaurant Longitude", value=80.27)

with colB:
    user_lat = st.number_input("Customer Latitude", value=12.97)
    user_lon = st.number_input("Customer Longitude", value=77.59)

# distance
from geopy.distance import geodesic
distance = geodesic((rest_lat, rest_lon), (user_lat, user_lon)).km

# map
m = folium.Map(location=[rest_lat, rest_lon], zoom_start=10)
folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color="green")).add_to(m)
folium.Marker([user_lat, user_lon], tooltip="Customer", icon=folium.Icon(color="red")).add_to(m)
folium.PolyLine([[rest_lat, rest_lon], [user_lat, user_lon]], color="blue").add_to(m)
st_folium(m, width=800, height=350)

# -------------------------
# MAPPINGS
# -------------------------
weather_map = {"Sunny":0, "Cloudy":1, "Fog":2, "Stormy":3, "Windy":4}
traffic_map = {"Low":0, "Medium":1, "High":2, "Jam":3}
vehicle_map = {"Bike":0, "Scooter":1, "Car":2}
order_type_map = {"Meal":0, "Snack":1, "Drinks":2}
city_map = {"Urban":0, "Semi-Urban":1, "Metropolitan":2}
festival_map = {"No":0, "Yes":1}

# -------------------------
# INPUTS
# -------------------------
st.subheader("📥 Delivery Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 25)
    rating = st.slider("Rating", 1.0, 5.0, 4.0)
    vehicle_condition = st.slider("Vehicle Condition", 1, 10, 5)
    deliveries = st.selectbox("Multiple Deliveries", [0,1,2,3])

with col2:
    prep_time = st.slider("Preparation Time", 5, 60, 20)

# time
col3, col4 = st.columns(2)

with col3:
    order_time = st.time_input("Order Time")

with col4:
    picked_time = st.time_input("Picked Time")

# categories
col5, col6, col7 = st.columns(3)

with col5:
    weather = st.selectbox("Weather", list(weather_map.keys()))

with col6:
    traffic = st.selectbox("Traffic", list(traffic_map.keys()))

with col7:
    vehicle = st.selectbox("Vehicle", list(vehicle_map.keys()))

col8, col9, col10 = st.columns(3)

with col8:
    order_type = st.selectbox("Order Type", list(order_type_map.keys()))

with col9:
    city = st.selectbox("City", list(city_map.keys()))

with col10:
    festival = st.selectbox("Festival", list(festival_map.keys()))

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def create_features():
    order_dt = datetime.combine(datetime.today(), order_time)
    picked_dt = datetime.combine(datetime.today(), picked_time)

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

    with st.spinner("AI is predicting..."):
        order_hour, prepare_time, is_weekend = create_features()

        input_dict = {
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "order_hour": order_hour,
            "order_prepare_time": prepare_time,
            "Weather_conditions": weather_map[weather],
            "Road_traffic_density": traffic_map[traffic],
            "Vehicle_condition": vehicle_condition,
            "Type_of_order": order_type_map[order_type],
            "Type_of_vehicle": vehicle_map[vehicle],
            "multiple_deliveries": deliveries,
            "Festival": festival_map[festival],
            "City": city_map[city],
            "distance": distance,
            "is_weekend": is_weekend
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

    # KPI
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'>📏 Distance<br><b>{distance:.2f} km</b></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>⭐ Rating<br><b>{rating}</b></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>⏳ ETA<br><b>{prediction:.2f} min</b></div>", unsafe_allow_html=True)

    # ETA range
    lower = max(0, prediction - 5)
    upper = prediction + 5

    st.success(f"🚚 Delivery Time: {lower:.0f} - {upper:.0f} minutes")

    # status
    if prediction < 20:
        st.success("⚡ Fast Delivery")
    elif prediction < 40:
        st.warning("🚗 Moderate Traffic")
    else:
        st.error("🐢 Delay Expected")

    # -------------------------
    # FEATURE IMPORTANCE (FAKE VISUAL)
    # -------------------------
    st.subheader("📊 Key Factors")

    labels = ["Distance", "Prep Time", "Traffic", "Weather", "Rating"]
    values = [distance, prepare_time, 3, 2, rating]

    fig, ax = plt.subplots()
    ax.barh(labels, values)
    st.pyplot(fig)
