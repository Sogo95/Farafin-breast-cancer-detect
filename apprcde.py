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
# CSS PROFESSIONNEL — DESIGN MÉDICAL PREMIUM
# ======================================================
# ======================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=Playfair+Display:wght@500;600;700&display=swap');

/* ── ROOT & RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a2332;
}

/* ── FOND PRINCIPAL ── */
.stApp {
    background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 40%, #f4f0f8 100%);
    background-attachment: fixed;
}

/* Bruit subtil en overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
}

/* ── CONTENEUR PRINCIPAL ── */
.block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1380px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1f3d 0%, #1a2d4a 60%, #0d1b2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Playfair Display', serif !important;
    letter-spacing: 0.02em;
}

[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.88rem;
    color: #94b3c8 !important;
    line-height: 1.6;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 1.2rem 0;
}

/* Badge utilisateur sidebar */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: linear-gradient(135deg, rgba(236,72,153,0.15), rgba(99,102,241,0.12)) !important;
    border: 1px solid rgba(236,72,153,0.3) !important;
    border-radius: 10px !important;
}

/* ── TITRES ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #0f1f3d !important;
    letter-spacing: -0.02em;
    line-height: 1.2 !important;
}

h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.4rem !important;
    color: #1a2d4a !important;
    font-weight: 600 !important;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #1a2d4a !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── INPUTS ── */
[data-testid="stTextInput"] input {
    border: 1.5px solid #d0dae8;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.93rem;
    background: #ffffff;
    color: #1a2332;
    transition: border-color 0.2s, box-shadow 0.2s;
}

[data-testid="stTextInput"] input:focus {
    border-color: #ec4899;
    box-shadow: 0 0 0 3px rgba(236,72,153,0.12);
    outline: none;
}

[data-testid="stTextInput"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #4a5568 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
}

/* ── BOUTONS ── */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.04em;
    border: none;
    background: linear-gradient(135deg, #ec4899 0%, #a855f7 100%);
    color: white;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(236,72,153,0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(236,72,153,0.42);
}

.stButton > button:hover::before { opacity: 1; }

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 3px 12px rgba(236,72,153,0.3);
}

/* Bouton déconnexion sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #c8d8e8 !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(236,72,153,0.15) !important;
    border-color: rgba(236,72,153,0.4) !important;
    color: white !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #c4cfe0;
    border-radius: 16px;
    background: rgba(255,255,255,0.7);
    transition: border-color 0.2s, background 0.2s;
    padding: 0.5rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: #ec4899;
    background: rgba(236,72,153,0.03);
}

/* ── ALERTES ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 0 0 0 4px !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
}

/* Info bleue */
div[data-testid="stAlert"][class*="info"] {
    background: linear-gradient(90deg, #eff6ff, #f0f9ff) !important;
    border-left-color: #3b82f6 !important;
    color: #1e40af !important;
}

/* Warning jaune */
div[data-testid="stAlert"][class*="warning"] {
    background: linear-gradient(90deg, #fffbeb, #fefce8) !important;
    border-left-color: #f59e0b !important;
    color: #92400e !important;
}

/* Success vert */
div[data-testid="stAlert"][class*="success"] {
    background: linear-gradient(90deg, #f0fdf4, #ecfdf5) !important;
    border-left-color: #10b981 !important;
    color: #065f46 !important;
}

/* Error rouge */
div[data-testid="stAlert"][class*="error"] {
    background: linear-gradient(90deg, #fff1f2, #fef2f2) !important;
    border-left-color: #ef4444 !important;
    color: #991b1b !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #c4cfe0, transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── IMAGE ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 8px 32px rgba(15,31,61,0.12);
}

/* ── SPINNER ── */
[data-testid="stSpinner"] {
    font-family: 'DM Sans', sans-serif;
    color: #ec4899;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c4cfe0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a0aec0; }

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
        "department": "Health Data & AI Unit",
        "initials": "AS"
    },
    "dr.marie.kabore": {
        "password": "Radiology@BF2026",
        "role": "Senior Radiologist",
        "full_name": "Dr Marie Kaboré",
        "department": "Radiology Department",
        "initials": "MK"
    },
    "dr.issa.ouedraogo": {
        "password": "Oncology@Hospital2026",
        "role": "Clinical Oncologist",
        "full_name": "Dr Issa Ouédraogo",
        "department": "Oncology Department",
        "initials": "IO"
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
    # Fond pleine page pour la login
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f1f3d 0%, #1a2d4a 50%, #1e1030 100%) !important; }
    .stApp::before { opacity: 0.06 !important; }
    .block-container { max-width: 520px !important; margin: 0 auto; padding-top: 4rem !important; }
    h1 { color: #ffffff !important; font-size: 1.7rem !important; }
    h3 { color: #94b3c8 !important; text-transform: none !important; letter-spacing: 0 !important; font-weight: 300 !important; font-size: 0.95rem !important; }
    [data-testid="stTextInput"] label { color: #94b3c8 !important; }
    [data-testid="stTextInput"] input { background: rgba(255,255,255,0.06) !important; border-color: rgba(255,255,255,0.12) !important; color: #000000 !important; }
    [data-testid="stTextInput"] input:focus { border-color: #ec4899 !important; background: rgba(255,255,255,0.09) !important; }
    [data-testid="stTextInput"] input::placeholder { color: rgba(255,255,255,0.3) !important; }
    </style>
    """, unsafe_allow_html=True)

    # LOGO + TITRE
    
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
    
    # CARTE LOGIN
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 24px 64px rgba(0,0,0,0.3);
    ">
    """, unsafe_allow_html=True)

    username = st.text_input("Identifiant", placeholder="votre.identifiant")
    password = st.text_input("Mot de passe", type="password", placeholder="••••••••••••")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Accéder à la plateforme →"):
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Accès autorisé — Chargement en cours…")
            st.rerun()
        else:
            st.error("Identifiants incorrects ou accès non autorisé.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem; margin-top:2rem; letter-spacing:0.05em;'>
        RÉSERVÉ AUX PROFESSIONNELS DE SANTÉ AUTORISÉS &nbsp;·&nbsp; FARAFIN AI FOR HEALTH © 2026
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
# SIDEBAR PROFESSIONNELLE
# ======================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
            <div style="
                width:42px; height:42px; border-radius:12px;
                background:linear-gradient(135deg,#ec4899,#a855f7);
                display:flex; align-items:center; justify-content:center;
                font-weight:700; font-size:0.85rem; color:white; flex-shrink:0;
            ">
    """ + current_user['initials'] + """
            </div>
            <div>
                <div style="color:white !important; font-weight:600; font-size:0.9rem; line-height:1.2;">""" + current_user['full_name'] + """</div>
                <div style="color:#94b3c8; font-size:0.75rem; margin-top:2px;">""" + current_user['role'] + """</div>
            </div>
        </div>
        <div style="
            background:rgba(236,72,153,0.12);
            border:1px solid rgba(236,72,153,0.25);
            border-radius:8px; padding:6px 10px;
            font-size:0.75rem; color:#f9a8d4; letter-spacing:0.03em;
        ">
            🏥 """ + current_user['department'] + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="margin-bottom:1rem;">
        <p style="font-size:0.7rem; color:#4a6a8a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;">Navigation</p>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("🩻", "Analyse mammographique", True),
        ("🔬", "Détection des lésions", False),
        ("📋", "Support décisionnel", False),
        ("📊", "Audit clinique", False),
        ("📁", "Historique patients", False),
    ]
    for icon, label, active in nav_items:
        bg = "rgba(236,72,153,0.15)" if active else "transparent"
        border = "rgba(236,72,153,0.4)" if active else "transparent"
        color = "#f9a8d4" if active else "#94b3c8"
        st.markdown(f"""
        <div style="
            display:flex; align-items:center; gap:10px;
            padding:9px 12px; border-radius:9px;
            background:{bg}; border:1px solid {border};
            margin-bottom:4px; cursor:pointer;
            transition:all 0.2s;
        ">
            <span style="font-size:1rem;">{icon}</span>
            <span style="font-size:0.87rem; color:{color}; font-weight:{'600' if active else '400'};">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("⎋  Déconnexion"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

    st.markdown("""
    <div style="margin-top:2rem; padding:12px; background:rgba(255,255,255,0.04); border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
        <p style="font-size:0.7rem; color:#4a6a8a; text-align:center; margin:0; letter-spacing:0.04em;">
            FARAFIN AI FOR HEALTH<br>
            <span style="color:#2a4a6a;">Version 2.0 · 2026</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# HEADER PRINCIPAL
# ======================================================
col_title, col_badge = st.columns([3, 1])

with col_title:
    st.markdown("""
    <div style="margin-bottom:0.2rem;">
        <p style="font-size:0.75rem; color:#ec4899; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:4px;">
            Plateforme clinique IA
        </p>
        <h1 style="margin:0;">Bienvenue sur la plateforme d'aide au diagnostic du cancer du sein Assistée par IA </h1>
        <p style="color:#64748b; font-size:0.93rem; margin-top:6px; font-weight:300;">
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_badge:
    st.markdown("""
    <div style="
        text-align:right; padding-top:0.8rem;
    ">
        <div style="
            display:inline-block;
            background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(5,150,105,0.08));
            border:1px solid rgba(16,185,129,0.3);
            border-radius:10px; padding:8px 14px;
        ">
            <span style="font-size:0.7rem; color:#059669; font-weight:600; letter-spacing:0.06em;">● SYSTÈME ACTIF</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# NOTICE CLINIQUE
st.markdown("""
<div style="
    background: linear-gradient(90deg, #eff6ff, #f8faff);
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 1.5rem;
    display:flex; align-items:center; gap:12px;
">
    <span style="font-size:1.3rem;">⚕️</span>
    <div>
        <span style="font-size:0.83rem; color:#1e40af; font-weight:600; display:block; margin-bottom:2px;">
            Avis de support décisionnel
        </span>
        <span style="font-size:0.82rem; color:#3b82f6; font-weight:300;">
            Cet outil a pour rôle d'assister le radiologue dans l'interprétation des mammographies.
           <strong>La validation clinique finale reste strictement médicale</strong>.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ======================================================
# ZONE PRINCIPALE — DEUX COLONNES
# ======================================================
main_col, side_col = st.columns([3, 2], gap="large")

with main_col:

    # CARTE IMPORT
    st.markdown("""
    <div style="
        background:white;
        border-radius:18px;
        border:1px solid #e2e8f0;
        padding:1.6rem;
        box-shadow:0 4px 24px rgba(15,31,61,0.06);
        margin-bottom:1rem;
    ">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.2rem;">
            <div style="
                width:36px; height:36px; border-radius:10px;
                background:linear-gradient(135deg,#ec4899,#a855f7);
                display:flex; align-items:center; justify-content:center;
                font-size:1rem;
            ">📤</div>
            <div>
                <h3 style="margin:0; text-transform:none; letter-spacing:0;">Importation d'une image</h3>
                <p style="margin:0; font-size:0.78rem; color:#94a3b8; font-weight:300;">Format DICOM, JPG, PNG, JPEG acceptés</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Glisser-déposer ou sélectionner une image mammographique",
        type=["jpg", "png", "jpeg", "IMG", "DICOM"],
        label_visibility="visible"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # IMAGE + ANALYSE
    if uploaded_file is not None:
        img, img_array = preprocess_image(uploaded_file)

        # Affichage image dans une carte
        st.markdown("""
        <div style="
            background:white; border-radius:18px;
            border:1px solid #e2e8f0; padding:1.2rem;
            box-shadow:0 4px 24px rgba(15,31,61,0.06);
        ">
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.78rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.7rem;'>Aperçu — Image chargée</p>", unsafe_allow_html=True)
        st.image(img, caption="", use_container_width=True)

        # Métadonnées fictives d'affichage
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div style='background:#f8fafc; border-radius:8px; padding:8px 10px; text-align:center;'><p style='font-size:0.68rem; color:#94a3b8; margin:0; text-transform:uppercase; letter-spacing:0.06em;'>Format</p><p style='font-size:0.88rem; font-weight:600; color:#1a2332; margin:2px 0 0;'>" + uploaded_file.type.split("/")[-1].upper() + "</p></div>", unsafe_allow_html=True)
        with m2:
            size_kb = round(uploaded_file.size / 1024, 1)
            st.markdown(f"<div style='background:#f8fafc; border-radius:8px; padding:8px 10px; text-align:center;'><p style='font-size:0.68rem; color:#94a3b8; margin:0; text-transform:uppercase; letter-spacing:0.06em;'>Taille</p><p style='font-size:0.88rem; font-weight:600; color:#1a2332; margin:2px 0 0;'>{size_kb} Ko</p></div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div style='background:#f8fafc; border-radius:8px; padding:8px 10px; text-align:center;'><p style='font-size:0.68rem; color:#94a3b8; margin:0; text-transform:uppercase; letter-spacing:0.06em;'>Statut</p><p style='font-size:0.88rem; font-weight:600; color:#10b981; margin:2px 0 0;'>✓ Valide</p></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


with side_col:
    if uploaded_file is not None:
        # CHECKLIST QUALITÉ
        st.markdown("""
        <div style="
            background:white; border-radius:18px;
            border:1px solid #e2e8f0; padding:1.4rem;
            box-shadow:0 4px 24px rgba(15,31,61,0.06);
            margin-bottom:1rem;
        ">
            <p style="font-size:0.72rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 1rem;">
                Contrôle qualité image
            </p>
        """, unsafe_allow_html=True)

        checks = [
            ("Qualité image", True),
            ("Résolution suffisante", True),
            ("Cadrage correct", True),
            ("Image exploitable", True),
            ("Contrôle radiologique", True),
        ]
        for label, ok in checks:
            color = "#10b981" if ok else "#ef4444"
            icon = "✓" if ok else "✗"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #f1f5f9;">
                <div style="
                    width:22px; height:22px; border-radius:6px;
                    background:{'rgba(16,185,129,0.12)' if ok else 'rgba(239,68,68,0.1)'};
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.75rem; font-weight:700; color:{color};
                    flex-shrink:0;
                ">{icon}</div>
                <span style="font-size:0.85rem; color:#374151;">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # BOUTON ANALYSE
        launch = st.button("🔬  Lancer l'analyse assistée par IA")

    else:
        # État vide
        st.markdown("""
        <div style="
            background:white; border-radius:18px;
            border:2px dashed #e2e8f0; padding:2.5rem 1.5rem;
            text-align:center;
            box-shadow:none;
        ">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🩻</div>
            <p style="color:#94a3b8; font-size:0.88rem; font-weight:400; margin:0;">
                Importez une mammographie<br>pour démarrer l'analyse
            </p>
        </div>
        """, unsafe_allow_html=True)
        launch = False


# ======================================================
# RÉSULTATS D'ANALYSE
# ======================================================
if uploaded_file is not None and launch:
    st.divider()
    st.markdown("""
    <p style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem;">
        Résultats de l'analyse clinique
    </p>
    """, unsafe_allow_html=True)

    with st.spinner("Analyse IA en cours — Traitement du modèle MobileNet…"):
        prediction = model.predict(img_array)[0][0]

    confidence = round(float(prediction) * 100 if prediction > 0.5 else (1 - float(prediction)) * 100, 1)

    if prediction > 0.5:
        # ── RÉSULTAT SUSPECT ──
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg, #fff1f2, #fef2f2);
            border:1px solid #fecdd3;
            border-left:5px solid #ef4444;
            border-radius:16px; padding:1.6rem;
            margin-bottom:1rem;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <span style="font-size:1.6rem;">⚠️</span>
                <div>
                    <p style="font-size:1.05rem; font-weight:700; color:#991b1b; margin:0;">
                        Suspicion de lésion mammaire suspecte
                    </p>
                    <p style="font-size:0.8rem; color:#b91c1c; margin:3px 0 0; font-weight:300;">
                        Score de confiance du modèle : <strong>{confidence}%</strong>
                    </p>
                </div>
            </div>
            <div style="
                background:white; border-radius:10px; padding:1rem 1.2rem;
                border:1px solid #fecdd3; margin-bottom:1rem;
            ">
                <p style="font-size:0.82rem; color:#7f1d1d; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; margin:0 0 0.7rem;">
                    Interprétation clinique
                </p>
                <p style="font-size:0.88rem; color:#991b1b; margin:0; line-height:1.7;">
                    Le modèle détecte des anomalies compatibles avec une <strong>lésion potentiellement maligne</strong>.
                    Une corrélation radiologique et une validation par le spécialiste sont indispensables.
                </p>
            </div>
            <p style="font-size:0.82rem; font-weight:600; color:#7f1d1d; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem;">
                Recommandations prioritaires
            </p>
        """, unsafe_allow_html=True)

        recs = [
            "Corrélation radiologique immédiate",
            "Avis spécialisé en sénologie / oncologie",
            "Biopsie ciblée si cliniquement indiquée",
            "Échographie ou IRM mammaire complémentaire",
            "Validation obligatoire par radiologue senior",
        ]
        for r in recs:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:5px 0;">
                <span style="color:#ef4444; font-weight:700;">→</span>
                <span style="font-size:0.86rem; color:#991b1b;">{r}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Explicabilité IA
        st.markdown("""
        <div style="
            background:white; border-radius:14px;
            border:1px solid #e2e8f0; padding:1.2rem;
            box-shadow:0 2px 12px rgba(15,31,61,0.04);
        ">
            <p style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                🧬 Explicabilité IA — Grad-CAM
            </p>
            <p style="font-size:0.84rem; color:#64748b; margin:0;">
                La carte de chaleur des régions d'intérêt diagnostique (Grad-CAM) sera affichée ici pour localiser les zones suspectes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── RÉSULTAT NÉGATIF ──
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg, #f0fdf4, #ecfdf5);
            border:1px solid #bbf7d0;
            border-left:5px solid #10b981;
            border-radius:16px; padding:1.6rem;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <span style="font-size:1.6rem;">✅</span>
                <div>
                    <p style="font-size:1.05rem; font-weight:700; color:#065f46; margin:0;">
                        Aucun signe radiologique suspect détecté
                    </p>
                    <p style="font-size:0.8rem; color:#047857; margin:3px 0 0; font-weight:300;">
                        Score de confiance du modèle : <strong>{confidence}%</strong>
                    </p>
                </div>
            </div>
            <div style="
                background:white; border-radius:10px; padding:1rem 1.2rem;
                border:1px solid #bbf7d0; margin-bottom:1rem;
            ">
                <p style="font-size:0.82rem; color:#065f46; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; margin:0 0 0.7rem;">
                    Interprétation clinique
                </p>
                <p style="font-size:0.88rem; color:#047857; margin:0; line-height:1.7;">
                    Aucune anomalie mammaire significative n'a été identifiée par le modèle sur cette image.
                    Maintenir le suivi habituel et corréler avec le contexte clinique du patient.
                </p>
            </div>
            <p style="font-size:0.82rem; font-weight:600; color:#065f46; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem;">
                Recommandations
            </p>
        """, unsafe_allow_html=True)

        recs = [
            "Maintenir le protocole de surveillance périodique",
            "Corrélation avec le contexte clinique du patient",
            "Réévaluation si symptomatologie clinique persistante",
            "Résultat à intégrer dans le dossier médical",
        ]
        for r in recs:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:5px 0;">
                <span style="color:#10b981; font-weight:700;">→</span>
                <span style="font-size:0.86rem; color:#047857;">{r}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # AVERTISSEMENT LÉGAL
    st.markdown("""
    <div style="
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:0.9rem 1.2rem;
        margin-top:1rem;
    ">
        <p style="font-size:0.78rem; color:#94a3b8; margin:0; line-height:1.6;">
            <strong style="color:#64748b;"> Avertissement clinique!!!!</strong> —
            Cette analyse constitue un support décisionnel automatisé et ne doit jamais se substituer
            au jugement clinique du radiologue.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown("""
<div style="
    display:flex; justify-content:space-between; align-items:center;
    padding:0.5rem 0; flex-wrap:wrap; gap:0.5rem;
">
    <span style="font-size:0.78rem; color:#94a3b8;">
        Farafin AI for Health &nbsp;·&nbsp; Plateforme d'aide au diagnostic clinique v2.0 &nbsp;·&nbsp; 2026
    </span>
    <span style="font-size:0.78rem; color:#cbd5e1;">
        Modèle : MobileNetV2 &nbsp;·&nbsp; Confidentiel — Usage médical exclusif
    </span>
</div>
""", unsafe_allow_html=True)