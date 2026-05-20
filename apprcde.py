import streamlit as st
import numpy as np
import os
import warnings

# ======================================================
# SUPPRESSION DES WARNINGS TENSORFLOW/KERAS
# ======================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from utils import preprocess_image

# ======================================================
# SESSION STATE INIT (DOIT ÊTRE TOUT EN HAUT)
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ======================================================
# UTILISATEURS AUTORISÉS (IMPORTANT: AVANT login_page)
# ======================================================
AUTHORIZED_USERS = {
    "admin.armel.sogo": {
        "password": "BreastAI@2026Secure",
        "role": "System Administrator",
        "full_name": "Armel Emmanuel SOGO",
        "department": "Health Data & AI Unit"
    },
    "dr.marie.kabore": {
        "password": "Radiology@BF2026",
        "role": "Senior Radiologist",
        "full_name": "Dr Marie Kaboré",
        "department": "Radiology Department"
    },
    "dr.issa.ouedraogo": {
        "password": "Oncology@Hospital2026",
        "role": "Clinical Oncologist",
        "full_name": "Dr Issa Ouédraogo",
        "department": "Oncology Department"
    }
}

# ======================================================
# CONFIGURATION GÉNÉRALE
# ======================================================
st.set_page_config(
    page_title="Farafin BreastCancer AI Clinical Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CSS PROFESSIONNEL
# ======================================================
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #0f172a;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(90deg, #ec4899, #f43f5e);
    color: white;
    border: none;
}

.login-box {
    background: white;
    padding: 30px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}

.result-box {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border-left: 5px solid #ec4899;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

.footer-box {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOGIN PAGE
# ======================================================
def login_page():

    st.markdown("""
    <div style='text-align:center; margin-bottom:10px;'>
        <img src='https://img.icons8.com/color/240/pink-ribbon.png' width='140'>
        <h1>Farafin BreastCancer AI</h1>
        <p style='color:#64748b;'>Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.subheader("Secure Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Access granted")
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# AUTH CHECK
# ======================================================
if not st.session_state.authenticated:
    login_page()
    st.stop()

# ======================================================
# USER CONNECTÉ
# ======================================================
current_user = AUTHORIZED_USERS[st.session_state.username]

# ======================================================
# MODEL
# ======================================================
MODEL_PATH = "model/best_mobilenet_model.h5"

@st.cache_resource
def load_my_model():
    tf.compat.v1.reset_default_graph()
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.title("User Panel")

    st.success(current_user["full_name"])
    st.write(current_user["role"])
    st.write(current_user["department"])

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

# ======================================================
# MAIN UI
# ======================================================
st.title("🩺 Breast Cancer AI Clinical Decision Support")
st.info("AI assists radiologists in mammography interpretation")

st.divider()

uploaded_file = st.file_uploader(
    "Upload mammography image",
    type=["jpg", "png", "jpeg", "IMG", "DICOM"]
)

if uploaded_file:

    col1, col2 = st.columns([2, 1])

    with col1:
        img, img_array = preprocess_image(uploaded_file)
        st.image(img, caption="Mammography image", use_container_width=True)

    with col2:
        st.info("Image loaded successfully")
        launch = st.button("Run AI Analysis")

    if launch:
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)[0][0]

        st.divider()

        if prediction > 0.5:
            st.error("Suspicious lesion detected")

            st.markdown("""
            <div class='result-box'>
            <h3>Clinical Interpretation</h3>
            <p>Model suggests possible malignant pattern.</p>

            <h4>Recommendations</h4>
            <ul>
                <li>Radiology review</li>
                <li>Oncology consultation</li>
                <li>Biopsy consideration</li>
                <li>Advanced imaging</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("No suspicious findings detected")

            st.markdown("""
            <div class='result-box'>
            <h3>Clinical Interpretation</h3>
            <p>No abnormal pattern detected.</p>

            <h4>Recommendations</h4>
            <ul>
                <li>Routine follow-up</li>
                <li>Clinical correlation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown("<p class='footer-box'>Farafin AI Clinical Platform – 2026</p>", unsafe_allow_html=True)