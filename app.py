import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model Files
# ----------------------------
model = pickle.load(open("model_gbc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(page_title="AgriAI Crop Prediction", layout="wide")

# ----------------------------
# Background + Styling
# ----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(rgba(10,20,15,0.65), rgba(10,20,15,0.65)),
                url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

/* Clean Title */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #00ffa3;
    margin-top: 40px;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #dddddd;
    margin-bottom: 40px;
}

/* Main Glass Card */
.main-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 40px;
    border: 1px solid rgba(0,255,150,0.2);
    box-shadow: 0 0 30px rgba(0,255,150,0.08);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00ffa3, #00c97b);
    border-radius: 12px;
    border: none;
    color: black;
    font-weight: bold;
    padding: 12px 30px;
    font-size: 16px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Clean Header (No Box)
# ----------------------------
st.markdown("<div class='main-title'>🌾 AgriAI Crop Recommendation</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Smart Farming & Sustainable Agriculture Platform</div>", unsafe_allow_html=True)

# ----------------------------
# Prediction Card
# ----------------------------
#st.markdown("<div class='main-card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    n = st.number_input("Nitrogen (N)", 0.0, 150.0, 90.0)
    p = st.number_input("Phosphorus (P)", 0.0, 150.0, 40.0)
    k = st.number_input("Potassium (K)", 0.0, 150.0, 40.0)

with col2:
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    hum = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)

with col3:
    ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
    rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Prediction Logic
# ----------------------------
if st.button("🚀 Run AI Prediction"):

    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)

    crop_name = encoder.inverse_transform(prediction)[0]
    confidence = round(np.max(probabilities) * 100, 2)

    st.markdown(f"""
    <div style="text-align:center; margin-top:30px;">
        <h2 style="color:#00ffa3;">🌱 Recommended Crop: {crop_name}</h2>
        <p style="font-size:18px;">Suitability Score: {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)



