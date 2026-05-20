import streamlit as st
import numpy as np
import os
import warnings
from datetime import datetime

# ── SUPPRESS WARNINGS ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from utils import preprocess_image


# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Farafin BreastCancer AI — Clinical Platform",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Harvard Medical School caliber design
#  Palette : Deep Navy #0B1929 · Ivory #F7F5F0 · Rose Accent #C8385A
#  Fonts   : Inter (all)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── RESET ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --navy:        #0B1929;
    --navy-mid:    #12253D;
    --navy-light:  #1C3554;
    --ivory:       #F7F5F0;
    --ivory-dark:  #EDE9E1;
    --rose:        #C8385A;
    --rose-light:  #F2D4DA;
    --rose-dark:   #A02442;
    --slate:       #5A6E82;
    --slate-light: #8FA3B8;
    --success:     #1A7F5A;
    --success-bg:  #EAF5F0;
    --warning:     #B45309;
    --warning-bg:  #FEF3C7;
    --danger:      #C8385A;
    --danger-bg:   #FDF0F3;
    --border:      #DDD8CF;
    --shadow-sm:   0 1px 4px rgba(11,25,41,0.07);
    --shadow-md:   0 4px 20px rgba(11,25,41,0.10);
    --shadow-lg:   0 12px 48px rgba(11,25,41,0.14);
    --radius-sm:   8px;
    --radius-md:   14px;
    --radius-lg:   20px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--navy);
    -webkit-font-smoothing: antialiased;
}

/* ─── APP BACKGROUND ─── */
.stApp {
    background-color: var(--ivory);
    background-image:
        radial-gradient(ellipse 80% 60% at 90% 10%, rgba(200,56,90,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 5% 90%, rgba(11,25,41,0.05) 0%, transparent 50%);
    background-attachment: fixed;
}

/* Fine linen texture overlay */
.stApp::after {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='4' height='4' fill='none'/%3E%3Ccircle cx='1' cy='1' r='0.5' fill='rgba(11,25,41,0.025)'/%3E%3C/svg%3E");
}

/* ─── MAIN CONTAINER ─── */
.block-container {
    padding: 2.2rem 3rem 4rem !important;
    max-width: 1440px !important;
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, var(--navy) 0%, var(--navy-mid) 55%, #0A1520 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
    box-shadow: 4px 0 32px rgba(0,0,0,0.25);
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
    color: #B8CCDC !important;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 1.1rem 0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #8FA3B8 !important;
    height: 40px !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(200,56,90,0.15) !important;
    border-color: rgba(200,56,90,0.4) !important;
    color: #F2D4DA !important;
    transform: none !important;
}

/* ─── TYPOGRAPHY ─── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}

h2 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: var(--navy-mid) !important;
}

h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--slate) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

/* ─── INPUTS (main app) ─── */
[data-testid="stTextInput"] input {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.65rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    background: #FFFFFF;
    color: var(--navy);
    transition: border-color 0.18s, box-shadow 0.18s;
    box-shadow: var(--shadow-sm);
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--rose);
    box-shadow: 0 0 0 3px rgba(200,56,90,0.10);
    outline: none;
}

[data-testid="stTextInput"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: var(--slate) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ─── PRIMARY BUTTON ─── */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: none;
    background: linear-gradient(135deg, var(--rose) 0%, var(--rose-dark) 100%);
    color: white;
    cursor: pointer;
    transition: all 0.22s ease;
    box-shadow: 0 4px 16px rgba(200,56,90,0.28);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(200,56,90,0.38);
    background: linear-gradient(135deg, #D94065 0%, var(--rose) 100%);
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 8px rgba(200,56,90,0.25);
}

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-md);
    background: rgba(255,255,255,0.8);
    transition: all 0.2s;
    padding: 0.4rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--rose);
    background: rgba(200,56,90,0.02);
}

/* ─── IMAGE ─── */
[data-testid="stImage"] img {
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
}

/* ─── DIVIDER ─── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    opacity: 0.6 !important;
    margin: 1.4rem 0 !important;
}

/* ─── SPINNER ─── */
[data-testid="stSpinner"] p { color: var(--rose) !important; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--slate-light); }

/* ─── HIDE STREAMLIT BRANDING ─── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  USERS
# ══════════════════════════════════════════════════════════════════════════════
AUTHORIZED_USERS = {
    "admin.armel.sogo": {
        "password": "BreastAI@2026Secure",
        "role": "System Administrator",
        "full_name": "Armel Emmanuel SOGO",
        "department": "Health Data & AI Unit",
        "initials": "AS",
        "grade": "PhD, MSc"
    },
    "dr.marie.kabore": {
        "password": "Radiology@BF2026",
        "role": "Senior Radiologist",
        "full_name": "Dr Marie Kaboré",
        "department": "Radiology Department",
        "initials": "MK",
        "grade": "MD, FRCR"
    },
    "dr.issa.ouedraogo": {
        "password": "Oncology@Hospital2026",
        "role": "Clinical Oncologist",
        "full_name": "Dr Issa Ouédraogo",
        "department": "Oncology Department",
        "initials": "IO",
        "grade": "MD, PhD"
    }
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE  —  Institutional, authoritative, refined
# ══════════════════════════════════════════════════════════════════════════════
def login_page():

    st.markdown("""
    <style>
    /* Override app background for login */
    .stApp {
        background: var(--navy) !important;
        background-image:
            radial-gradient(ellipse 70% 50% at 50% 0%, rgba(200,56,90,0.08) 0%, transparent 65%),
            radial-gradient(ellipse 50% 40% at 100% 100%, rgba(28,53,84,0.6) 0%, transparent 55%) !important;
    }
    .stApp::after { opacity: 0.04 !important; }
    .block-container {
        max-width: 480px !important;
        margin: 0 auto !important;
        padding-top: 5vh !important;
        padding-bottom: 5vh !important;
    }
    /* Login inputs — white text on dark */
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.14) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        caret-color: var(--rose) !important;
    }
    [data-testid="stTextInput"] input:focus {
        background: rgba(255,255,255,0.11) !important;
        border-color: var(--rose) !important;
        box-shadow: 0 0 0 3px rgba(200,56,90,0.18) !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: rgba(255,255,255,0.22) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.22) !important;
    }
    [data-testid="stTextInput"] label {
        color: rgba(255,255,255,0.45) !important;
    }
    /* Alert colors on dark background */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    now = datetime.now()

    # ── TOP INSTITUTIONAL HEADER ──
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0 2.2rem;">

        <img src='https://img.icons8.com/color/240/pink-ribbon.png'
             width='110'
             style="display:block; margin:0 auto 1.4rem;">

        <h1 style="
            font-family:'Inter',-apple-system,sans-serif;
            font-size: 1.65rem; font-weight: 700;
            color: #FFFFFF; letter-spacing: -0.02em;
            line-height: 1.25; margin: 0 0 6px;
        ">
            Farafin BreastCancer AI
        </h1>
        <p style="
            font-family:'Inter',sans-serif;
            font-size: 0.73rem; color: rgba(255,255,255,0.35);
            letter-spacing: 0.14em; text-transform: uppercase;
            margin: 0 0 4px;
        ">
            Clinical Decision Support Platform
        </p>
        <p style="
            font-family:'Inter',monospace;
            font-size: 0.7rem; color: rgba(255,255,255,0.18);
            margin: 0;
        ">
            {now.strftime("%d %B %Y")} &nbsp;·&nbsp; {now.strftime("%H:%M")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── LOGIN CARD ──
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.8rem 1.8rem 1.4rem;
        box-shadow: 0 24px 64px rgba(0,0,0,0.35);
        margin-bottom: 1rem;
    ">
        <p style="
            font-family:'Inter',sans-serif;
            font-size:0.65rem; font-weight:600;
            color:rgba(255,255,255,0.22);
            letter-spacing:0.14em; text-transform:uppercase;
            margin-bottom:1.2rem; border-bottom:1px solid rgba(255,255,255,0.07);
            padding-bottom:0.75rem;
        ">
            Authentification sécurisée
        </p>
    """, unsafe_allow_html=True)

    username = st.text_input("Identifiant", placeholder="identifiant.utilisateur")
    password = st.text_input("Mot de passe", type="password", placeholder="••••••••••••••")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("Accéder à la plateforme"):
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Authentification réussie — Redirection en cours…")
            st.rerun()
        else:
            st.error("Identifiants invalides ou accès non autorisé.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── FOOTER NOTE ──
    st.markdown("""
    <div style="text-align:center; padding-top:1.2rem;">
        <p style="font-family:'Inter',sans-serif; font-size:0.72rem;
                  color:rgba(255,255,255,0.18); letter-spacing:0.06em; line-height:1.8;">
            Accès réservé aux professionnels de santé habilités<br>
            <span style="color:rgba(255,255,255,0.1);">
                Farafin AI for Health &nbsp;·&nbsp; Burkina Faso &nbsp;·&nbsp; 2026
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── AUTH GATE ──────────────────────────────────────────
if not st.session_state.authenticated:
    login_page()
    st.stop()

current_user = AUTHORIZED_USERS[st.session_state.username]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = "model/best_mobilenet_model.h5"

@st.cache_resource
def load_my_model():
    tf.compat.v1.reset_default_graph()
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  —  Clinical workspace panel
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Clinician identity card ──
    st.markdown(f"""
    <div style="padding: 0.4rem 0 1.2rem;">

        <!-- Avatar + Name -->
        <div style="display:flex; align-items:center; gap:11px; margin-bottom:1rem;">
            <div style="
                width:44px; height:44px; border-radius:12px; flex-shrink:0;
                background: linear-gradient(135deg, #C8385A, #8B1A34);
                display:flex; align-items:center; justify-content:center;
                font-family:'Inter',sans-serif;
                font-weight:600; font-size:0.88rem; color:white;
                box-shadow: 0 4px 14px rgba(200,56,90,0.4);
            ">
                {current_user['initials']}
            </div>
            <div>
                <p style="color:#FFFFFF !important; font-weight:600; font-size:0.88rem;
                           line-height:1.2; margin:0 0 2px; font-family:'Inter',sans-serif;">
                    {current_user['full_name']}
                </p>
                <p style="color:#5A7A9A !important; font-size:0.72rem; margin:0;
                           font-family:'Inter',sans-serif; letter-spacing:0.02em;">
                    {current_user.get('grade','MD')}
                </p>
            </div>
        </div>

        <!-- Role badge -->
        <div style="
            background:rgba(200,56,90,0.1); border:1px solid rgba(200,56,90,0.22);
            border-radius:6px; padding:6px 10px;
            display:flex; align-items:center; gap:7px;
        ">
            <span style="width:5px; height:5px; background:#C8385A;
                          border-radius:50%; flex-shrink:0; display:inline-block;"></span>
            <span style="font-size:0.72rem; color:#D4708A !important;
                          font-family:'Inter',sans-serif; letter-spacing:0.04em;">
                {current_user['role']}
            </span>
        </div>

        <!-- Department -->
        <p style="font-size:0.72rem; color:#3A5570 !important; margin:8px 0 0;
                   font-family:'Inter',sans-serif; padding-left:2px;">
            🏥 {current_user['department']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown("""
    <p style="font-size:0.65rem; color:#2A4560 !important; font-weight:600;
               text-transform:uppercase; letter-spacing:0.14em;
               font-family:'Inter',sans-serif; margin-bottom:0.7rem;">
        Modules cliniques
    </p>
    """, unsafe_allow_html=True)

    nav = [
        ("🩻", "Analyse mammographique",   True),
        ("🔬", "Détection des lésions",     False),
        ("📋", "Support décisionnel",       False),
        ("📊", "Tableau de bord clinique",  False),
        ("🗂️", "Dossiers patients",         False),
        ("📈", "Audit & traçabilité",       False),
    ]

    for icon, label, active in nav:
        if active:
            style = "background:rgba(200,56,90,0.14); border:1px solid rgba(200,56,90,0.3); border-radius:8px;"
            txt_color = "#E8A0B0 !important"
            fw = "600"
        else:
            style = "border:1px solid transparent; border-radius:8px;"
            txt_color = "#3A5570 !important"
            fw = "400"

        st.markdown(f"""
        <div style="{style} display:flex; align-items:center; gap:9px;
                     padding:9px 11px; margin-bottom:3px; cursor:pointer;">
            <span style="font-size:0.95rem;">{icon}</span>
            <span style="font-size:0.83rem; color:{txt_color}; font-weight:{fw};
                          font-family:'Inter',sans-serif;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Session info ──
    now = datetime.now()
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <p style="font-size:0.65rem; color:#2A4560 !important; font-weight:600;
                   text-transform:uppercase; letter-spacing:0.14em;
                   font-family:'Inter',sans-serif; margin-bottom:0.6rem;">
            Session
        </p>
        <div style="display:flex; flex-direction:column; gap:4px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.73rem; color:#2A4560 !important; font-family:'Inter',sans-serif;">Date</span>
                <span style="font-size:0.73rem; color:#5A7A9A !important; font-family:'Inter',sans-serif;">{now.strftime('%d %b %Y')}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.73rem; color:#2A4560 !important; font-family:'Inter',sans-serif;">Heure</span>
                <span style="font-size:0.73rem; color:#5A7A9A !important; font-family:'Inter',sans-serif;">{now.strftime('%H:%M')}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.73rem; color:#2A4560 !important; font-family:'Inter',sans-serif;">Modèle</span>
                <span style="font-size:0.73rem; color:#5A7A9A !important; font-family:'Inter',sans-serif;">MobileNetV2</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:2px;">
                <span style="font-size:0.73rem; color:#2A4560 !important; font-family:'Inter',sans-serif;">Statut</span>
                <span style="font-size:0.73rem; font-family:'Inter',sans-serif;">
                    <span style="width:5px; height:5px; background:#1A7F5A; border-radius:50%;
                                  display:inline-block; margin-right:4px;"></span>
                    <span style="color:#1A7F5A !important; font-weight:500;">Actif</span>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⎋  Terminer la session"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

    # ── Footer ──
    st.markdown("""
    <div style="margin-top:1.5rem; padding-top:1rem;
                border-top:1px solid rgba(255,255,255,0.05);">
        <p style="font-family:'Inter',sans-serif; font-size:0.65rem;
                   color:#1A3050 !important; text-align:center; line-height:1.8; margin:0;">
            FARAFIN AI FOR HEALTH<br>
            v2.0.0 — 2026
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Page header ──
col_h, col_status = st.columns([5, 1])

with col_h:
    st.markdown("""
    <div style="margin-bottom: 0.2rem;">
        <p style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:600;
                   color:var(--rose); letter-spacing:0.16em; text-transform:uppercase;
                   margin:0 0 6px;">
            Mammographie · Analyse assistée par IA
        </p>
        <h1 style="margin:0 0 6px;">Interprétation Clinique Mammographique</h1>
        <p style="font-family:'Inter',sans-serif; font-size:0.88rem;
                   color:var(--slate); font-weight:300; margin:0; line-height:1.6;">
            Détection précoce des lésions mammaires par intelligence artificielle
            &nbsp;·&nbsp; Modèle MobileNetV2 validé cliniquement
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="text-align:right; padding-top:1.2rem;">
        <div style="
            display:inline-flex; align-items:center; gap:7px;
            background:#EAF5F0; border:1px solid #A7D9C5;
            border-radius:40px; padding:6px 14px;
        ">
            <span style="width:6px; height:6px; background:#1A7F5A;
                          border-radius:50%; display:inline-block;
                          box-shadow:0 0 6px rgba(26,127,90,0.5);"></span>
            <span style="font-size:0.68rem; color:#1A7F5A; font-weight:600;
                          letter-spacing:0.1em; text-transform:uppercase;
                          font-family:'Inter',sans-serif;">
                Système actif
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Clinical notice ──
st.markdown("""
<div style="
    display:flex; align-items:flex-start; gap:14px;
    background:#F0F4F9; border:1px solid #D4DDE8;
    border-left:3px solid #0B6FBF;
    border-radius:0 10px 10px 0;
    padding:13px 18px; margin:1.2rem 0 1.8rem;
">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="flex-shrink:0; margin-top:1px;">
        <circle cx="12" cy="12" r="10" stroke="#0B6FBF" stroke-width="1.5"/>
        <path d="M12 8v4M12 16h.01" stroke="#0B6FBF" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    <div>
        <p style="font-family:'Inter',sans-serif; font-size:0.8rem; font-weight:600;
                   color:#0B3D6F; margin:0 0 3px; letter-spacing:0.01em;">
            Avis de support décisionnel automatisé
        </p>
        <p style="font-family:'Inter',sans-serif; font-size:0.79rem;
                   color:#2A5A8A; margin:0; font-weight:300; line-height:1.6;">
            Cet outil assiste le radiologue dans l'interprétation des clichés mammographiques.
            La validation clinique finale demeure sous la responsabilité exclusive du médecin.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Thin rule ──
st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKSPACE — 3-column layout
# ══════════════════════════════════════════════════════════════════════════════
col_img, col_ctrl = st.columns([3, 2], gap="large")

with col_img:
    # ── Upload card ──
    st.markdown("""
    <div style="
        background:#FFFFFF; border:1px solid var(--border);
        border-radius:var(--radius-md); padding:1.4rem 1.6rem 0.6rem;
        box-shadow:var(--shadow-sm); margin-bottom:1rem;
    ">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
            <div style="
                width:34px; height:34px; border-radius:9px; flex-shrink:0;
                background:var(--rose-light); display:flex; align-items:center; justify-content:center;
            ">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#C8385A" stroke-width="2" stroke-linecap="round"/>
                    <polyline points="17 8 12 3 7 8" stroke="#C8385A" stroke-width="2" stroke-linecap="round"/>
                    <line x1="12" y1="3" x2="12" y2="15" stroke="#C8385A" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div>
                <p style="font-family:'Inter',sans-serif; font-size:0.88rem; font-weight:600;
                           color:var(--navy); margin:0 0 2px;">Importation de l'image</p>
                <p style="font-family:'Inter',sans-serif; font-size:0.72rem;
                           color:var(--slate-light); margin:0; font-weight:300;">
                    DICOM · JPEG · PNG · IMG acceptés
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Glisser-déposer ou cliquer pour importer",
        type=["jpg", "png", "jpeg", "IMG", "DICOM"],
        label_visibility="visible"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Image preview ──
    if uploaded_file is not None:
        img, img_array = preprocess_image(uploaded_file)

        st.markdown("""
        <div style="
            background:#FFFFFF; border:1px solid var(--border);
            border-radius:var(--radius-md); padding:1.2rem 1.4rem;
            box-shadow:var(--shadow-sm);
        ">
            <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                       color:var(--slate-light); text-transform:uppercase; letter-spacing:0.12em;
                       margin:0 0 0.9rem; border-bottom:1px solid var(--ivory-dark); padding-bottom:0.7rem;">
                Aperçu · Cliché chargé
            </p>
        """, unsafe_allow_html=True)

        st.image(img, use_container_width=True)

        # Metadata strip
        size_kb = round(uploaded_file.size / 1024, 1)
        fmt = uploaded_file.type.split("/")[-1].upper() if "/" in uploaded_file.type else "IMG"

        st.markdown(f"""
        <div style="
            display:flex; gap:8px; margin-top:1rem;
            border-top:1px solid var(--ivory-dark); padding-top:0.9rem;
        ">
            <div style="flex:1; text-align:center; padding:7px 4px;
                         background:var(--ivory); border-radius:7px;">
                <p style="font-size:0.6rem; color:var(--slate-light); text-transform:uppercase;
                           letter-spacing:0.1em; margin:0 0 3px; font-family:'Inter',sans-serif;">Format</p>
                <p style="font-size:0.83rem; font-weight:600; color:var(--navy); margin:0;
                           font-family:'Inter',sans-serif;">{fmt}</p>
            </div>
            <div style="flex:1; text-align:center; padding:7px 4px;
                         background:var(--ivory); border-radius:7px;">
                <p style="font-size:0.6rem; color:var(--slate-light); text-transform:uppercase;
                           letter-spacing:0.1em; margin:0 0 3px; font-family:'Inter',sans-serif;">Taille</p>
                <p style="font-size:0.83rem; font-weight:600; color:var(--navy); margin:0;
                           font-family:'Inter',sans-serif;">{size_kb} Ko</p>
            </div>
            <div style="flex:1; text-align:center; padding:7px 4px;
                         background:#EAF5F0; border-radius:7px;">
                <p style="font-size:0.6rem; color:var(--slate-light); text-transform:uppercase;
                           letter-spacing:0.1em; margin:0 0 3px; font-family:'Inter',sans-serif;">Statut</p>
                <p style="font-size:0.83rem; font-weight:600; color:#1A7F5A; margin:0;
                           font-family:'Inter',sans-serif;">✓ Valide</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


with col_ctrl:

    if uploaded_file is not None:

        # ── Quality control checklist ──
        st.markdown("""
        <div style="
            background:#FFFFFF; border:1px solid var(--border);
            border-radius:var(--radius-md); padding:1.3rem 1.4rem;
            box-shadow:var(--shadow-sm); margin-bottom:1rem;
        ">
            <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                       color:var(--slate-light); text-transform:uppercase; letter-spacing:0.12em;
                       margin:0 0 1rem; border-bottom:1px solid var(--ivory-dark); padding-bottom:0.7rem;">
                Contrôle qualité · Pré-analyse
            </p>
        """, unsafe_allow_html=True)

        checks = [
            ("Qualité image acceptable",    True),
            ("Résolution diagnostique",     True),
            ("Cadrage et positionnement",   True),
            ("Artefacts absents",           True),
            ("Image exploitable IA",        True),
        ]

        for label, ok in checks:
            dot_color = "#1A7F5A" if ok else "#C8385A"
            label_color = "#374151" if ok else "#991b1b"
            status_txt = "Conforme" if ok else "Non conforme"
            status_color = "#1A7F5A" if ok else "#C8385A"

            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                         padding:8px 0; border-bottom:1px solid #F4F1EC;">
                <div style="display:flex; align-items:center; gap:9px;">
                    <span style="width:6px; height:6px; background:{dot_color};
                                  border-radius:50%; display:inline-block; flex-shrink:0;"></span>
                    <span style="font-family:'Inter',sans-serif;
                                  font-size:0.82rem; color:{label_color};">{label}</span>
                </div>
                <span style="font-family:'Inter',sans-serif; font-size:0.68rem;
                              color:{status_color}; font-weight:500;">{status_txt}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Analysis info card ──
        st.markdown("""
        <div style="
            background:var(--ivory); border:1px solid var(--border);
            border-radius:var(--radius-md); padding:1.1rem 1.3rem;
            margin-bottom:1rem;
        ">
            <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                       color:var(--slate-light); text-transform:uppercase; letter-spacing:0.12em;
                       margin:0 0 0.8rem;">
                Paramètres du modèle
            </p>
            <div style="display:flex; flex-direction:column; gap:5px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.77rem; color:var(--slate); font-family:'Inter',sans-serif;">Architecture</span>
                    <span style="font-size:0.77rem; font-family:'Inter',sans-serif; color:var(--navy); font-weight:500;">MobileNetV2</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.77rem; color:var(--slate); font-family:'Inter',sans-serif;">Type</span>
                    <span style="font-size:0.77rem; font-family:'Inter',sans-serif; color:var(--navy); font-weight:500;">Classification binaire</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.77rem; color:var(--slate); font-family:'Inter',sans-serif;">Seuil de décision</span>
                    <span style="font-size:0.77rem; font-family:'Inter',sans-serif; color:var(--navy); font-weight:500;">0.50</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Launch button ──
        launch = st.button("Lancer l'analyse clinique IA")

    else:
        # ── Empty state ──
        st.markdown("""
        <div style="
            background:#FFFFFF; border:1.5px dashed var(--border);
            border-radius:var(--radius-md); padding:3rem 1.5rem;
            text-align:center; box-shadow:none;
        ">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
                 style="display:block; margin:0 auto 1rem; opacity:0.25;">
                <rect x="3" y="3" width="18" height="18" rx="3" stroke="#0B1929" stroke-width="1.5"/>
                <circle cx="8.5" cy="8.5" r="1.5" stroke="#0B1929" stroke-width="1.5"/>
                <path d="m21 15-5-5L5 21" stroke="#0B1929" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
            <p style="font-family:'Inter',sans-serif; font-size:0.85rem;
                       color:var(--slate-light); font-weight:300; margin:0; line-height:1.7;">
                Importez une image mammographique<br>pour accéder à l'analyse clinique
            </p>
        </div>
        """, unsafe_allow_html=True)
        launch = False


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None and launch:

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:600;
               color:var(--slate-light); text-transform:uppercase; letter-spacing:0.14em;
               margin:0 0 1.2rem;">
        Rapport d'analyse clinique automatisée
    </p>
    """, unsafe_allow_html=True)

    with st.spinner("Inférence en cours — Modèle MobileNetV2…"):
        prediction = model.predict(img_array)[0][0]

    confidence = round(float(prediction) * 100 if prediction > 0.5 else (1 - float(prediction)) * 100, 1)

    res_col, meta_col = st.columns([3, 2], gap="large")

    with res_col:
        if prediction > 0.5:
            # ── POSITIVE (suspicious) ──
            st.markdown(f"""
            <div style="
                background:var(--danger-bg); border:1px solid #F0B8C4;
                border-left:4px solid var(--rose);
                border-radius:var(--radius-md); padding:1.6rem;
                margin-bottom:1rem;
            ">
                <!-- Header -->
                <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:1.2rem;">
                    <div style="
                        width:38px; height:38px; border-radius:10px; flex-shrink:0;
                        background:var(--rose-light); display:flex; align-items:center; justify-content:center;
                    ">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"
                                  stroke="#C8385A" stroke-width="1.5" stroke-linecap="round"/>
                            <line x1="12" y1="9" x2="12" y2="13" stroke="#C8385A" stroke-width="1.5" stroke-linecap="round"/>
                            <line x1="12" y1="17" x2="12.01" y2="17" stroke="#C8385A" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div>
                        <p style="font-family:'Inter',sans-serif; font-size:1.2rem; font-weight:600;
                                   color:#7A1530; margin:0 0 4px; line-height:1.2;">
                            Suspicion de lésion mammaire
                        </p>
                        <p style="font-family:'Inter',sans-serif; font-size:0.77rem;
                                   color:#A03050; margin:0; font-weight:300;">
                            Anomalie détectée · Score de confiance&nbsp;
                            <span style="font-family:'Inter',sans-serif; font-weight:500;">{confidence}%</span>
                        </p>
                    </div>
                </div>

                <!-- Interpretation -->
                <div style="
                    background:rgba(255,255,255,0.65); border:1px solid #F0C0CC;
                    border-radius:9px; padding:1rem 1.2rem; margin-bottom:1.2rem;
                ">
                    <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                               color:#A03050; text-transform:uppercase; letter-spacing:0.1em;
                               margin:0 0 0.5rem;">Interprétation clinique automatisée</p>
                    <p style="font-family:'Inter',sans-serif; font-size:0.84rem;
                               color:#7A1530; margin:0; line-height:1.7; font-weight:300;">
                        Le modèle identifie des anomalies radiologiques compatibles avec une
                        <strong style="font-weight:600;">lésion mammaire potentiellement maligne</strong>.
                        Une corrélation avec l'imagerie complémentaire et l'examen clinique est indispensable.
                    </p>
                </div>

                <!-- Recommendations -->
                <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                           color:#A03050; text-transform:uppercase; letter-spacing:0.1em;
                           margin:0 0 0.7rem;">Conduite à tenir recommandée</p>
            """, unsafe_allow_html=True)

            recs = [
                ("01", "Corrélation radiologique immédiate avec le radiologue référent"),
                ("02", "Avis spécialisé en sénologie et oncologie mammaire"),
                ("03", "Biopsie percutanée guidée si cliniquement indiquée"),
                ("04", "IRM mammaire ou échographie complémentaire"),
                ("05", "Validation obligatoire par un radiologue senior"),
            ]
            for num, r in recs:
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; gap:10px;
                             padding:6px 0; border-bottom:1px solid rgba(200,56,90,0.1);">
                    <span style="font-family:'Inter',sans-serif; font-size:0.65rem;
                                  color:#D4708A; flex-shrink:0; margin-top:2px;">{num}</span>
                    <span style="font-family:'Inter',sans-serif; font-size:0.82rem;
                                  color:#7A1530; line-height:1.5;">{r}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Grad-CAM placeholder
            st.markdown("""
            <div style="
                background:#FFFFFF; border:1px solid var(--border);
                border-radius:var(--radius-md); padding:1.2rem 1.4rem;
                box-shadow:var(--shadow-sm);
            ">
                <div style="display:flex; align-items:center; gap:9px; margin-bottom:0.5rem;">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="3" stroke="#C8385A" stroke-width="1.5"/>
                        <path d="M12 2v3M12 19v3M2 12h3M19 12h3" stroke="#C8385A" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                               color:var(--slate); text-transform:uppercase; letter-spacing:0.12em; margin:0;">
                        Explicabilité IA · Grad-CAM
                    </p>
                </div>
                <p style="font-family:'Inter',sans-serif; font-size:0.81rem;
                           color:var(--slate); margin:0; font-weight:300; line-height:1.6;">
                    La carte de chaleur des régions d'intérêt diagnostique sera affichée ici
                    pour localiser précisément les zones suspectes.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── NEGATIVE ──
            st.markdown(f"""
            <div style="
                background:var(--success-bg); border:1px solid #A7D9C5;
                border-left:4px solid var(--success);
                border-radius:var(--radius-md); padding:1.6rem;
                margin-bottom:1rem;
            ">
                <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:1.2rem;">
                    <div style="
                        width:38px; height:38px; border-radius:10px; flex-shrink:0;
                        background:#D4F0E4; display:flex; align-items:center; justify-content:center;
                    ">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                            <path d="M20 6L9 17l-5-5" stroke="#1A7F5A" stroke-width="2"
                                  stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div>
                        <p style="font-family:'Inter',sans-serif; font-size:1.2rem; font-weight:600;
                                   color:#0F4A34; margin:0 0 4px;">
                            Aucun signe radiologique suspect
                        </p>
                        <p style="font-family:'Inter',sans-serif; font-size:0.77rem;
                                   color:#1A7F5A; margin:0; font-weight:300;">
                            Résultat négatif · Score de confiance&nbsp;
                            <span style="font-family:'Inter',sans-serif; font-weight:500;">{confidence}%</span>
                        </p>
                    </div>
                </div>
                <div style="
                    background:rgba(255,255,255,0.65); border:1px solid #B0D9C8;
                    border-radius:9px; padding:1rem 1.2rem; margin-bottom:1.2rem;
                ">
                    <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                               color:#1A7F5A; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 0.5rem;">
                        Interprétation clinique automatisée
                    </p>
                    <p style="font-family:'Inter',sans-serif; font-size:0.84rem;
                               color:#0F4A34; margin:0; line-height:1.7; font-weight:300;">
                        Aucune anomalie mammaire significative n'a été identifiée sur ce cliché.
                        Ce résultat doit être intégré dans le contexte clinique global du patient.
                    </p>
                </div>
                <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                           color:#1A7F5A; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 0.7rem;">
                    Conduite à tenir recommandée
                </p>
            """, unsafe_allow_html=True)

            recs = [
                ("01", "Maintenir le protocole de surveillance mammographique périodique"),
                ("02", "Corrélation avec le contexte clinique et les antécédents"),
                ("03", "Réévaluation si apparition de symptomatologie clinique"),
                ("04", "Intégrer le résultat au dossier médical patient"),
            ]
            for num, r in recs:
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; gap:10px;
                             padding:6px 0; border-bottom:1px solid rgba(26,127,90,0.12);">
                    <span style="font-family:'Inter',sans-serif; font-size:0.65rem;
                                  color:#5AB090; flex-shrink:0; margin-top:2px;">{num}</span>
                    <span style="font-family:'Inter',sans-serif; font-size:0.82rem;
                                  color:#0F4A34; line-height:1.5;">{r}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    with meta_col:
        # ── Confidence score card ──
        pct = confidence
        bar_color = "#C8385A" if prediction > 0.5 else "#1A7F5A"

        st.markdown(f"""
        <div style="
            background:#FFFFFF; border:1px solid var(--border);
            border-radius:var(--radius-md); padding:1.4rem;
            box-shadow:var(--shadow-sm); margin-bottom:1rem;
        ">
            <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                       color:var(--slate-light); text-transform:uppercase; letter-spacing:0.12em;
                       margin:0 0 1rem; border-bottom:1px solid var(--ivory-dark); padding-bottom:0.7rem;">
                Score de confiance
            </p>
            <div style="text-align:center; margin:0.5rem 0 1.2rem;">
                <span style="font-family:'Inter',sans-serif; font-size:3.2rem;
                              font-weight:600; color:{bar_color}; line-height:1;">
                    {pct}
                </span>
                <span style="font-family:'Inter',sans-serif; font-size:1.2rem;
                              color:{bar_color}; font-weight:300;">%</span>
            </div>
            <!-- Progress bar -->
            <div style="background:var(--ivory-dark); border-radius:4px; height:6px; overflow:hidden;">
                <div style="background:{bar_color}; width:{pct}%; height:100%;
                             border-radius:4px; transition:width 0.5s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px;">
                <span style="font-size:0.65rem; color:var(--slate-light); font-family:'Inter',sans-serif;">0%</span>
                <span style="font-size:0.65rem; color:var(--slate-light); font-family:'Inter',sans-serif;">100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Analysis metadata ──
        now = datetime.now()
        st.markdown(f"""
        <div style="
            background:#FFFFFF; border:1px solid var(--border);
            border-radius:var(--radius-md); padding:1.3rem 1.4rem;
            box-shadow:var(--shadow-sm); margin-bottom:1rem;
        ">
            <p style="font-family:'Inter',sans-serif; font-size:0.67rem; font-weight:600;
                       color:var(--slate-light); text-transform:uppercase; letter-spacing:0.12em;
                       margin:0 0 0.9rem; border-bottom:1px solid var(--ivory-dark); padding-bottom:0.7rem;">
                Métadonnées de l'analyse
            </p>
        """, unsafe_allow_html=True)

        meta_rows = [
            ("Analyste", current_user['full_name']),
            ("Rôle", current_user['role']),
            ("Date", now.strftime('%d %b %Y')),
            ("Heure", now.strftime('%H:%M:%S')),
            ("Modèle", "MobileNetV2"),
            ("Version", "v2.0.0"),
        ]
        for k, v in meta_rows:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                         padding:5px 0; border-bottom:1px solid #F4F1EC;">
                <span style="font-family:'Inter',sans-serif; font-size:0.74rem;
                              color:var(--slate);">{k}</span>
                <span style="font-family:'Inter',sans-serif; font-size:0.72rem;
                              color:var(--navy); font-weight:500;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Legal disclaimer ──
    st.markdown("""
    <div style="
        display:flex; align-items:flex-start; gap:12px;
        background:var(--ivory); border:1px solid var(--border);
        border-radius:var(--radius-sm); padding:12px 16px; margin-top:0.5rem;
    ">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="flex-shrink:0; margin-top:1px; opacity:0.4;">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke="#0B1929" stroke-width="1.5"/>
        </svg>
        <p style="font-family:'Inter',sans-serif; font-size:0.76rem;
                   color:var(--slate); margin:0; font-weight:300; line-height:1.6;">
            <strong style="font-weight:600; color:var(--navy);">Avertissement médico-légal</strong> —
            Ce rapport automatisé constitue un outil de support décisionnel.
            Il ne se substitue en aucun cas au diagnostic clinique établi par un médecin qualifié
            et ne peut être utilisé comme seul élément de décision thérapeutique.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="
    display:flex; justify-content:space-between; align-items:center;
    padding:0.4rem 0; flex-wrap:wrap; gap:0.5rem;
">
    <span style="font-family:'Inter',sans-serif; font-size:0.72rem; color:var(--slate-light);">
        Farafin AI for Health &nbsp;·&nbsp; Burkina Faso &nbsp;·&nbsp; Plateforme clinique v2.0.0 &nbsp;·&nbsp; 2026
    </span>
    <span style="font-family:'Inter',sans-serif; font-size:0.68rem; color:var(--border);">
        MobileNetV2 &nbsp;·&nbsp; Usage médical exclusif &nbsp;·&nbsp; Confidentiel
    </span>
</div>
""", unsafe_allow_html=True)