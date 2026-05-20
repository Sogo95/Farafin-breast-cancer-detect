import streamlit as st
import numpy as np
import os
import warnings

# ======================================================
# ENV
# ======================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from utils import preprocess_image


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Farafin AI Clinical Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================
# DESIGN SYSTEM (CLINICAL GRADE)
# ======================================================
st.markdown("""
<style>

/* ========== GLOBAL ========== */
html, body, [class*="css"] {
    font-family: Inter, system-ui, sans-serif;
}

.stApp {
    background: #f4f6f9;
}

/* ========== TYPO ========== */
h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

h2, h3 {
    color: #111827 !important;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    height: 44px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background: #1d4ed8;
}

/* ========== INPUT FIX (IMPORTANT LOGIN VISIBILITY) ========== */
[data-testid="stTextInput"] input {
    background: white !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

[data-testid="stTextInput"] label {
    color: #374151 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: #0b1220;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ========== CARDS ========== */
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.2rem;
}

/* ========== IMAGE ========== */
img {
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

/* ========== DIVIDER ========== */
hr {
    border: none !important;
    height: 1px !important;
    background: #e5e7eb !important;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# USERS
# ======================================================
AUTHORIZED_USERS = {
    "admin.armel.sogo": {
        "password": "BreastAI@2026Secure",
        "role": "Admin",
        "full_name": "Armel Emmanuel SOGO"
    },
    "dr.marie.kabore": {
        "password": "Radiology@BF2026",
        "role": "Radiologist",
        "full_name": "Dr Marie Kaboré"
    }
}


# ======================================================
# SESSION STATE
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ======================================================
# LOGIN PAGE (PROFESSIONNEL + VISIBILITÉ OK)
# ======================================================
def login_page():

    st.markdown("""
    <style>
    .stApp {
        background: #0b1220;
    }

    .login-box {
        max-width: 420px;
        margin: 10vh auto;
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 2rem;
    }

    .title {
        text-align:center;
        color:white;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align:center;
        color:#9ca3af;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }

    /* FIX CRUCIAL VISIBILITÉ INPUT */
    input {
        background: #ffffff !important;
        color: #111827 !important;
        border-radius: 8px !important;
    }

    input::placeholder {
        color: #9ca3af !important;
    }

    label {
        color: #d1d5db !important;
        font-size: 0.8rem !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    st.markdown("<div class='title'>Farafin AI Clinical Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Radiology Decision Support System</div>", unsafe_allow_html=True)

    username = st.text_input("User ID", placeholder="Enter your ID")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Login"):
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# AUTH
# ======================================================
if not st.session_state.authenticated:
    login_page()
    st.stop()


# ======================================================
# USER
# ======================================================
user = AUTHORIZED_USERS[st.session_state.username]


# ======================================================
# MODEL
# ======================================================
MODEL_PATH = "model/best_mobilenet_model.h5"

@st.cache_resource
def load_model_fn():
    tf.compat.v1.reset_default_graph()
    return load_model(MODEL_PATH, compile=False)

model = load_model_fn()


# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.title("Clinician Panel")
    st.write(user["full_name"])
    st.write(user["role"])

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()


# ======================================================
# HEADER
# ======================================================
st.title("🩺 Mammography AI Decision Support")
st.info("Clinical support only — final validation by radiologist required.")


# ======================================================
# UPLOAD
# ======================================================
file = st.file_uploader(
    "Upload mammography image",
    type=["jpg", "png", "jpeg", "dcm"]
)

if file:

    img, img_array = preprocess_image(file)
    st.image(img, use_container_width=True)

    if st.button("Run AI Analysis"):

        with st.spinner("Analyzing..."):
            pred = model.predict(img_array)[0][0]

        st.divider()

        if pred > 0.5:
            st.error("⚠️ Suspicious lesion detected")
            st.markdown("""
            <div class="card">
            High probability of abnormal lesion.
            Recommend radiological confirmation.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("✅ No abnormality detected")
            st.markdown("""
            <div class="card">
            No suspicious findings detected by AI model.
            </div>
            """, unsafe_allow_html=True)


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Farafin AI for Health © 2026")