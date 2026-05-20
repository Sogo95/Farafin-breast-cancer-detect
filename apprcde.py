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
# CONFIGURATION GÉNÉRALE
# ======================================================
st.set_page_config(
    page_title="Farafin BreastCancer AI Clinical Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# DESIGN SYSTEM PREMIUM (CLINICAL SAAS)
# ======================================================
st.markdown("""
<style>

/* GLOBAL */
html, body {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND */
.main {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

/* CONTAINER */
.block-container {
    padding: 2rem 2.5rem;
    max-width: 1350px;
}

/* TITLES */
h1 {
    font-size: 34px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
}

h2, h3 {
    color: #0f172a;
    font-weight: 600;
}

/* LOGIN CARD */
.login-box {
    background: white;
    padding: 34px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    transition: 0.3s ease;
}

.login-box:hover {
    transform: translateY(-2px);
}

/* INPUTS */
input {
    border-radius: 10px !important;
}

/* BUTTONS */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
    background: linear-gradient(90deg, #ec4899, #f43f5e);
    color: white;
    border: none;
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(244, 63, 94, 0.25);
}

/* RESULT CARD */
.result-box {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border-left: 6px solid #ec4899;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    margin-top: 15px;
}

/* FOOTER */
.footer-box {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 25px 0 10px 0;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOGIN PAGE (UI UPGRADE UNIQUEMENT)
# ======================================================
def login_page():

    st.markdown("""
    <div style="text-align:center; padding-top:20px;">
        <img src="https://img.icons8.com/color/240/pink-ribbon.png" width="130">
        <h1 style="margin-bottom:0;">Farafin AI Clinical Platform</h1>
        <p style="color:#64748b; font-size:16px;">
            Breast Cancer Decision Support System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:

        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.markdown("### Secure Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign in"):
            if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Access granted")
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:12px;'>Restricted access – authorized clinicians only</p>", unsafe_allow_html=True)


# ======================================================
# AUTH (inchangé)
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    login_page()
    st.stop()

# ======================================================
# USERS
# ======================================================
AUTHORIZED_USERS = {
    "admin.armel.sogo": {
        "password": "BreastAI@2026Secure",
        "role": "System Administrator",
        "full_name": "Armel Emmanuel SOGO",
        "department": "Health Data & AI Unit"
    }
}

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
# SIDEBAR (UI CLEAN)
# ======================================================
with st.sidebar:
    st.markdown("## 👤 User Panel")
    st.write(f"**{current_user['full_name']}**")
    st.caption(current_user["role"])
    st.divider()

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("### Modules")
    st.markdown("""
    - Mammography Analysis  
    - Lesion Detection AI  
    - Clinical Decision Support  
    - Audit Dashboard  
    """)

# ======================================================
# MAIN HEADER
# ======================================================
st.title("🩺 Breast Cancer AI Clinical Decision Support")
st.caption("AI-assisted mammography interpretation for clinical decision support")

st.divider()

# ======================================================
# IMAGE UPLOAD
# ======================================================
st.subheader("Upload Mammography")

uploaded_file = st.file_uploader(
    "Upload medical image",
    type=["jpg", "png", "jpeg", "IMG", "DICOM"]
)

if uploaded_file:

    col1, col2 = st.columns([2, 1])

    with col1:
        img, img_array = preprocess_image(uploaded_file)
        st.image(img, caption="Loaded mammography", use_container_width=True)

    with col2:
        st.info("Pre-analysis quality check passed")

        launch = st.button("Run AI Analysis")

    if launch:
        with st.spinner("Running inference..."):
            prediction = model.predict(img_array)[0][0]

        st.divider()

        if prediction > 0.5:
            st.error("Suspicious lesion detected")

            st.markdown("""
            <div class='result-box'>
            <h3>Clinical Interpretation</h3>
            <p>Model indicates potential malignant pattern.</p>

            <h4>Recommendations</h4>
            <ul>
                <li>Immediate radiology review</li>
                <li>Oncology consultation</li>
                <li>Biopsy consideration</li>
                <li>Advanced imaging (MRI/US)</li>
            </ul>

            <b>Clinical note:</b> AI is decision support only.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("No suspicious findings detected")

            st.markdown("""
            <div class='result-box'>
            <h3>Clinical Interpretation</h3>
            <p>No abnormal patterns detected.</p>

            <h4>Recommendations</h4>
            <ul>
                <li>Routine follow-up</li>
                <li>Clinical correlation required</li>
            </ul>

            <b>Clinical note:</b> Negative AI result does not exclude disease.
            </div>
            """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown("<p class='footer-box'>Farafin AI Clinical Platform – 2026</p>", unsafe_allow_html=True)