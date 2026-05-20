import streamlit as st
import numpy as np
import os
import warnings

# ======================================================
# ENV & WARNINGS
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
    page_title="Farafin BreastCancer AI Clinical Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================
# GLOBAL CSS (CLINICAL GRADE UI)
# ======================================================
st.markdown("""
<style>

/* =======================
   BASE
======================= */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f9fc;
}

/* =======================
   TITRES
======================= */
h1 { color: #111827 !important; font-weight: 700 !important; }
h2 { color: #1f2937 !important; }
h3 { color: #374151 !important; }

/* =======================
   BUTTONS
======================= */
.stButton > button {
    background: #ec4899;
    color: white;
    height: 48px;
    border-radius: 10px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background: #db2777;
}

/* =======================
   INPUTS GLOBAL (LIGHT MODE)
======================= */
[data-testid="stTextInput"] input {
    background: white !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.95rem !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #ec4899 !important;
    box-shadow: 0 0 0 3px rgba(236,72,153,0.12) !important;
}

/* LABELS */
[data-testid="stTextInput"] label {
    color: #374151 !important;
    font-weight: 600 !important;
}

/* =======================
   SIDEBAR
======================= */
[data-testid="stSidebar"] {
    background: #0f172a;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* =======================
   ALERTS
======================= */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* =======================
   IMAGE
======================= */
img {
    border-radius: 12px;
}

/* =======================
   DIVIDER
======================= */
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
        "role": "System Admin",
        "full_name": "Armel Emmanuel SOGO",
        "department": "AI Health"
    },
    "dr.marie.kabore": {
        "password": "Radiology@BF2026",
        "role": "Radiologist",
        "full_name": "Dr Marie Kaboré",
        "department": "Radiology"
    }
}


# ======================================================
# SESSION
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ======================================================
# LOGIN PAGE (FIX VISIBILITY ISSUE)
# ======================================================
def login_page():

    st.markdown("""
    <style>
    /* LOGIN FULL DARK BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
    }

    /* LOGIN CARD */
    .login-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px;
        padding: 2rem;
    }

    /* FIX INPUT VISIBILITY (IMPORTANT) */
    input {
        background: rgba(255,255,255,0.95) !important;
        color: #0f172a !important;
    }

    input::placeholder {
        color: #64748b !important;
    }

    label {
        color: #cbd5e1 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:40px;">
        <h1 style="color:white;">Farafin AI Clinical Platform</h1>
        <p style="color:#94a3b8;">Breast Cancer Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your ID")
        password = st.text_input("Password", type="password", placeholder="Enter password")

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
# AUTH
# ======================================================
if not st.session_state.authenticated:
    login_page()
    st.stop()


# ======================================================
# USER
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
    st.write(current_user["full_name"])
    st.write(current_user["role"])

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()


# ======================================================
# MAIN UI
# ======================================================
st.title("🩺 Mammography AI Analysis")
st.info("Clinical decision support only — not a diagnostic tool.")

uploaded_file = st.file_uploader(
    "Upload mammography image",
    type=["jpg", "png", "jpeg", "dcm"]
)

if uploaded_file:

    img, img_array = preprocess_image(uploaded_file)
    st.image(img, use_container_width=True)

    if st.button("Run AI Analysis"):

        with st.spinner("Processing..."):
            prediction = model.predict(img_array)[0][0]

        st.divider()

        if prediction > 0.5:
            st.error("Suspicious lesion detected")
        else:
            st.success("No abnormality detected")


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Farafin AI for Health © 2026")