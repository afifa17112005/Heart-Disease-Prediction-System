import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Global styles and animated hearts background
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
:root{--accent:#ff6b6b;--accent-2:#ff9a9e;--muted:#062a37}
body, .stApp, .stApp * { font-family: 'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, Arial; }

/* Animated gradient */
.stApp { background: linear-gradient(270deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c1ee); background-size: 900% 900%; animation: gradientShift 20s ease infinite; }
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* Frosted container */
.block-container { padding: 2.2rem; border-radius: 14px; background: rgba(255,255,255,0.86); box-shadow: 0 18px 40px rgba(12,24,40,0.08); backdrop-filter: blur(8px); }

/* Hero typography */
.hero-title{ font-size:48px; margin:0; color:var(--muted); font-weight:800 }
.hero-sub{ font-size:18px; color:#2b2b2b; margin-top:8px }
.badge{ display:inline-block; margin-right:8px; background:#e6fff8; padding:8px 12px; border-radius:999px; border:1px solid rgba(0,0,0,0.06); font-weight:700; color:#007a5a }
.cta{ display:inline-block; background:#7fe3ff; padding:10px 16px; border-radius:12px; color:#062a37; font-weight:800; text-decoration:none }

/* Floating hearts */
.hearts { position: fixed; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden }
.hearts .heart { position:absolute; bottom:-80px; width:36px; height:36px; background-size:contain; background-repeat:no-repeat; opacity:0.95; animation: floatUp linear infinite }
@keyframes floatUp { 0% { transform: translateY(0) rotate(0deg) scale(0.9); opacity:0 } 10% { opacity:1 } 100% { transform: translateY(-120vh) rotate(360deg) scale(1.05); opacity:0 } }
.hearts .heart:nth-child(1){ left:5%; animation-duration:10s }
.hearts .heart:nth-child(2){ left:18%; animation-duration:14s; animation-delay:1s }
.hearts .heart:nth-child(3){ left:32%; animation-duration:12s; animation-delay:0.6s }
.hearts .heart:nth-child(4){ left:46%; animation-duration:16s; animation-delay:2s }
.hearts .heart:nth-child(5){ left:60%; animation-duration:11s; animation-delay:0.2s }
.hearts .heart:nth-child(6){ left:74%; animation-duration:13s; animation-delay:1.6s }
.hearts .heart:nth-child(7){ left:88%; animation-duration:15s; animation-delay:0.8s }

</style>

<div class="hearts">
  <div class="heart" style="background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 29"><path fill="%23ff6b6b" d="M23.6 0c-2.7 0-5 1.6-6.6 4-1.6-2.4-3.9-4-6.6-4C4.6 0 0 4.6 0 10.3 0 19 16 29 16 29s16-10 16-18.7C32 4.6 27.4 0 23.6 0z"/></svg>');"></div>
  <div class="heart" style="background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 29"><path fill="%23ff9a9e" d="M23.6 0c-2.7 0-5 1.6-6.6 4-1.6-2.4-3.9-4-6.6-4C4.6 0 0 4.6 0 10.3 0 19 16 29 16 29s16-10 16-18.7C32 4.6 27.4 0 23.6 0z"/></svg>');"></div>
  <div class="heart" style="background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 29"><path fill="%23ff3b3b" d="M23.6 0c-2.7 0-5 1.6-6.6 4-1.6-2.4-3.9-4-6.6-4C4.6 0 0 4.6 0 10.3 0 19 16 29 16 29s16-10 16-18.7C32 4.6 27.4 0 23.6 0z"/></svg>');"></div>
  <div class="heart" style="background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 29"><path fill="%23ffc1c1" d="M23.6 0c-2.7 0-5 1.6-6.6 4-1.6-2.4-3.9-4-6.6-4C4.6 0 0 4.6 0 10.3 0 19 16 29 16 29s16-10 16-18.7C32 4.6 27.4 0 23.6 0z"/></svg>');"></div>
  <div class="heart" style="background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 29"><path fill="%23ffd6e0" d="M23.6 0c-2.7 0-5 1.6-6.6 4-1.6-2.4-3.9-4-6.6-4C4.6 0 0 4.6 0 10.3 0 19 16 29 16 29s16-10 16-18.7C32 4.6 27.4 0 23.6 0z"/></svg>');"></div>
  <div class="heart" style="background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 29"><path fill="%23ff8aa1" d="M23.6 0c-2.7 0-5 1.6-6.6 4-1.6-2.4-3.9-4-6.6-4C4.6 0 0 4.6 0 10.3 0 19 16 29 16 29s16-10 16-18.7C32 4.6 27.4 0 23.6 0z"/></svg>');"></div>
</div>
''', unsafe_allow_html=True)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Hero (Streamlit-native)
with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.markdown("<h1 class='hero-title'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-sub'>Enter patient measurements below to get an immediate risk estimate using a trained XGBoost model.</p>", unsafe_allow_html=True)
        # badges removed per user request
        st.markdown("<a class='cta' href='#predict'>Predict Now</a>", unsafe_allow_html=True)
    with right:
        st.markdown("""
                <style>
                .heart-emoji{ font-size:120px; display:flex; align-items:center; justify-content:center; }
                @keyframes beat { 0%{transform:scale(1)} 25%{transform:scale(1.06)} 40%{transform:scale(0.98)} 60%{transform:scale(1.04)} 100%{transform:scale(1)} }
                .heart-emoji{ animation: beat 1.2s ease-in-out infinite; }
                </style>
                <div class='frame'>
                  <div class='heart-emoji'>🫀</div>
                </div>
                """, unsafe_allow_html=True)

# Prediction inputs will appear below the hero
st.markdown("<div class='prediction-area' id='predict'>", unsafe_allow_html=True)

age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.number_input("Rest ECG (0-2)", 0, 2)
thalach = st.number_input("Maximum Heart Rate")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope (0-2)", 0, 2)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4)
thal = st.number_input("Thal (0-3)", 0, 3)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

st.markdown("</div>", unsafe_allow_html=True)
