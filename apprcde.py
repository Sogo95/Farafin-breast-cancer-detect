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
# CSS PROFESSIONNEL + DESIGN PREMIUM AMÉLIORÉ
# ======================================================
st.markdown("""
<style>

.main {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* TITRES */
h1, h2, h3 {
    color: #0f172a;
    letter-spacing: -0.3px;
}

/* BOUTONS */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(135deg, #ec4899, #db2777);
    color: white;
    border: none;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(236, 72, 153, 0.25);
}

/* LOGIN BOX */
.login-box {
    background: white;
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

/* RESULT BOX */
.result-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #dbeafe;
    margin-top: 10px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.04);
}

/* FOOTER */
.footer-box {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    padding-top: 20px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* INPUTS */
input {
    border-radius: 10px !important;
}

/* CARDS HOVER EFFECT */
.result-box:hover {
    transform: translateY(-2px);
    transition: 0.2s ease-in-out;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# UTILISATEURS AUTORISÉS
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


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""


# ======================================================
# PAGE DE CONNEXION PREMIUM
# ======================================================
def login_page():

    st.markdown("""
    <div style='text-align:center; margin-bottom:10px;'>
        <img src='https://img.icons8.com/color/240/pink-ribbon.png' width='160'>
    </div>

    <h1 style='text-align:center; margin-bottom:5px;'>
        Farafin BreastCancer AI Clinical Platform
    </h1>

    <h3 style='text-align:center; color:#64748b; margin-top:0px;'>
        Outil d'aide à la décision pour la détection précoce des lésions mammaires
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;height:2px;background:linear-gradient(to right,transparent,#ec4899,transparent);'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.subheader("Connexion sécurisée")

        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")

        if st.button("Se connecter"):

            if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Access Granted")
                st.rerun()
            else:
                st.error("Unauthorized Access or Invalid Credentials")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; color:gray; font-size:13px; margin-top:20px;'>
        Accès réservé aux professionnels de santé autorisés.
    </p>
    """, unsafe_allow_html=True)


# ======================================================
# AUTH CONTROL
# ======================================================
if not st.session_state.authenticated:
    login_page()
    st.stop()


# ======================================================
# UTILISATEUR CONNECTÉ
# ======================================================
current_user = AUTHORIZED_USERS[st.session_state.username]


# ======================================================
# MODÈLE IA
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
    st.title("Espace utilisateur")

    st.success(f"Connecté : {current_user['full_name']}")
    st.write(f"**Rôle :** {current_user['role']}")
    st.write(f"**Département :** {current_user['department']}")

    if st.button("Déconnexion"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

    st.markdown("---")

    st.markdown("""
### Modules disponibles
- Analyse mammographique
- Détection assistée des lésions
- Support décisionnel clinique
- Audit & supervision clinique
""")


# ======================================================
# HEADER
# ======================================================
st.title("🩺 Farafin Breast Cancer Clinical Decision Support")

st.info("Plateforme d'aide à la décision radiologique. Validation clinique obligatoire.")

st.divider()


# ======================================================
# UPLOAD IMAGE
# ======================================================
st.subheader("📤 Importation de l'image")

uploaded_file = st.file_uploader(
    "Importer une image médicale",
    type=["jpg", "png", "jpeg", "IMG", "DICOM"]
)

if uploaded_file is not None:

    col1, col2 = st.columns([2, 1])

    with col1:
        img, img_array = preprocess_image(uploaded_file)
        st.image(img, caption="Image mammographique chargée", use_container_width=True)

    with col2:
        st.warning("""
✔ Qualité acceptable  
✔ Résolution suffisante  
✔ Cadrage correct  
✔ Exploitable
""")

        launch = st.button("🔍 Lancer l'analyse clinique")

    if launch:

        with st.spinner("Analyse IA en cours..."):
            prediction = model.predict(img_array)[0][0]

        st.divider()

        if prediction > 0.5:

            st.error("⚠️ Suspicion de lésion détectée")

            st.markdown("""
<div class='result-box'>
<h3>Interprétation clinique</h3>
<p>Suspicion de lésion potentiellement maligne.</p>

<h3>Recommandations</h3>
<ul>
<li>Corrélation radiologique</li>
<li>Avis oncologique</li>
<li>Biopsie si nécessaire</li>
<li>IRM / échographie</li>
</ul>

<p><b>Support décisionnel uniquement</b></p>
</div>
""", unsafe_allow_html=True)

        else:

            st.success("Aucune anomalie détectée")

            st.markdown("""
<div class='result-box'>
<h3>Interprétation clinique</h3>
<p>Absence de signe radiologique suspect.</p>

<h3>Recommandations</h3>
<ul>
<li>Suivi habituel</li>
<li>Surveillance clinique</li>
<li>Corrélation symptomatique</li>
</ul>

<p><b>Ne remplace pas l'expertise médicale</b></p>
</div>
""", unsafe_allow_html=True)


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")

st.markdown("""
<div class='footer-box'>
Farafin AI for Health – 2026
</div>
""", unsafe_allow_html=True)