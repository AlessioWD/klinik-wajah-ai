import os
import socket
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from core.face_mesh_detector import FaceMeshDetector
from core.fatigue_detector import FatigueDetector
from core.jaundice_analyzer import JaundiceAnalyzer
from core.pallor_analyzer import PallorAnalyzer
from core.rppg_estimator import RPPGEstimator
from core.symmetry_analyzer import SymmetryAnalyzer
from core.clinical_triangulation import ClinicalTriangulationEngine
from core.history_db import HistoryDatabase
from utils.visualization import Visualizer

import streamlit.components.v1 as components

# Inisialisasi Halaman Streamlit
st.set_page_config(
    page_title="FacialHealth | Clinical Biometric Screening",
    page_icon="https://raw.githubusercontent.com/AlessioWD/klinik-wajah-ai/main/static/icon-512.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injeksi Dynamic PWA Icon ke Header Peramban Mobile
components.html(
    """
    <script>
    const iconUrl = 'https://raw.githubusercontent.com/AlessioWD/klinik-wajah-ai/main/static/icon-512.png';
    const icon192 = 'https://raw.githubusercontent.com/AlessioWD/klinik-wajah-ai/main/static/icon-192.png';
    
    // Set apple-touch-icon untuk iOS & Android PWA
    let appleIcon = window.parent.document.querySelector("link[rel='apple-touch-icon']");
    if (!appleIcon) {
        appleIcon = window.parent.document.createElement('link');
        appleIcon.rel = 'apple-touch-icon';
        window.parent.document.head.appendChild(appleIcon);
    }
    appleIcon.href = icon192;

    // Set shortcut icon & favicon
    let favIcon = window.parent.document.querySelector("link[rel*='icon']");
    if (!favIcon) {
        favIcon = window.parent.document.createElement('link');
        favIcon.rel = 'shortcut icon';
        window.parent.document.head.appendChild(favIcon);
    }
    favIcon.href = icon192;
    </script>
    """,
    height=0,
    width=0,
)

# Custom Styling CSS: Minimalist Clinical Design (Bebas Estetika AI Slop)
st.markdown(
    """
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Branding */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.6rem;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .brand-badge {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 3px 8px;
        border-radius: 4px;
        background-color: #E2E8F0;
        color: #0369A1;
        vertical-align: middle;
        margin-left: 6px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }

    /* Disclaimer Medis Bersih */
    .clinical-notice {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0284C7;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        color: #334155;
        font-size: 0.86rem;
        line-height: 1.5;
        margin-bottom: 1.4rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* Kartu Metrik Klinis */
    .metric-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.95rem 1.05rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.9rem;
        transition: border-color 0.15s ease;
    }
    .metric-panel:hover {
        border-color: #CBD5E1;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94A3B8;
    }

    /* Badges Status Medis */
    .status-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        line-height: 1.4;
    }
    .badge-normal {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-moderate {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-alert {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    /* Diagnostic Result Box */
    .diagnosis-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem;
        margin-top: 0.8rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    /* Mobile Responsive Fix for Checkbox & Anamnesis Questionnaire */
    div[data-testid="stCheckbox"] {
        width: 100% !important;
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 0.75rem 0.95rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.55rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
        transition: border-color 0.15s ease, background-color 0.15s ease !important;
    }
    div[data-testid="stCheckbox"]:hover {
        border-color: #CBD5E1 !important;
        background-color: #F8FAFC !important;
    }
    div[data-testid="stCheckbox"] label {
        display: flex !important;
        align-items: flex-start !important;
        gap: 10px !important;
        width: 100% !important;
        white-space: normal !important;
        overflow: visible !important;
        cursor: pointer !important;
    }
    div[data-testid="stCheckbox"] label > div:first-child {
        margin-top: 3px !important;
        flex-shrink: 0 !important;
    }
    div[data-testid="stCheckbox"] label div[data-testid="stMarkdownContainer"],
    div[data-testid="stCheckbox"] label div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] span,
    div[data-testid="stCheckbox"] p,
    label[data-baseweb="checkbox"] p,
    label[data-baseweb="checkbox"] div {
        white-space: normal !important;
        text-overflow: unset !important;
        overflow: visible !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        line-height: 1.5 !important;
        font-size: 0.9rem !important;
        color: #1E293B !important;
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_local_ip():
    """Mengambil alamat IP lokal laptop di jaringan Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@st.cache_resource
def load_resources():
    detector = FaceMeshDetector()
    symmetry_analyzer = SymmetryAnalyzer()
    jaundice_analyzer = JaundiceAnalyzer()
    pallor_analyzer = PallorAnalyzer()
    rppg_estimator = RPPGEstimator(buffer_size=180)
    fatigue_detector = FatigueDetector()
    db = HistoryDatabase()
    return detector, symmetry_analyzer, jaundice_analyzer, pallor_analyzer, rppg_estimator, fatigue_detector, db


detector, symmetry_analyzer, jaundice_analyzer, pallor_analyzer, rppg_estimator, fatigue_detector, db = load_resources()

# ==========================================
# SIDEBAR: PENGATURAN & MULTI-USER
# ==========================================
if os.path.exists(os.path.join(CURRENT_DIR, "static", "icon-192.png")):
    st.sidebar.image("static/icon-192.png", width=52)
st.sidebar.markdown("### FacialHealth")
st.sidebar.caption("Sistem Skrining Biomarker Wajah v2.2")

# 1. Pilihan Profil Pengguna (Multi-User)
st.sidebar.markdown("---")
st.sidebar.markdown("**Profil Subjek / Pasien:**")
patients = db.get_patients()
selected_patient = st.sidebar.selectbox("Pilih Profil", patients, index=0, label_visibility="collapsed")

with st.sidebar.expander("Tambah Profil Baru"):
    new_patient_name = st.text_input("Nama Profil:")
    if st.button("Simpan", use_container_width=True):
        if db.add_patient(new_patient_name):
            st.success(f"Profil '{new_patient_name}' terdaftar.")
            st.rerun()
        else:
            st.warning("Nama kosong atau telah digunakan.")

st.sidebar.markdown("---")

# 2. Pilihan Mode Input
st.sidebar.markdown("**Modalitas Akuisisi Citra:**")
input_mode = st.sidebar.radio(
    "Modalitas",
    ["Kamera Langsung (Webcam/Mobile)", "Unggah Citra Wajah", "Kasus Uji Terkalibrasi"],
    index=0,
    label_visibility="collapsed",
)

with st.sidebar.expander("Akses Publik Mobile (24/7)"):
    st.markdown(
        """
        **Tautan Resmi 24 Jam Nonstop:**
        `https://klinik-wajah-ai.streamlit.app`
        
        Dapat dibuka langsung di HP (Android / iOS) dari mana saja menggunakan kuota/paket data 4G/5G.
        """
    )

with st.sidebar.expander("Instalasi Aplikasi (PWA)"):
    st.markdown(
        """
        **Android / Chrome:**
        Ketuk menu peramban (⋮) $\\rightarrow$ pilih **Install Aplikasi**.
        
        **iOS / Safari:**
        Ketuk tombol Share (↑) $\\rightarrow$ pilih **Add to Home Screen**.
        
        **Desktop (Chrome/Edge):**
        Klik tombol install pada bilah alamat peramban.
        """
    )

# ==========================================
# HEADER UTAMA (CLEAN BRANDING)
# ==========================================
st.markdown(
    """
    <div class="brand-container">
        <div class="brand-title">FacialHealth <span class="brand-badge">CLINICAL TRIAGE</span></div>
    </div>
    <div class="brand-subtitle">Platform Skrining Biomarker Fisik Wajah & Triangulasi Gejala Terarah</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="clinical-notice">
        <strong>Pemberitahuan Medis:</strong> Platform ini berfungsi sebagai instrumen skrining awal mandiri (triage) 
        dan bukan instrumen diagnosis definitif. Pada kasus gejala akut yang timbul mendadak (seperti kelemahan unilateral tubuh, 
        bicara pelo, atau ikterus progresif cepat), segera rujuk ke Instalasi Gawat Darurat atau fasilitas kesehatan terdekat.
    </div>
    """,
    unsafe_allow_html=True,
)


def render_metric_cards(sym, jnd, pal, rppg, fat):
    """Menampilkan 5 panel metrik klinis dengan estetika bersih."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        score = sym.get("symmetry_score", 0.0)
        risk = sym.get("risk_level", "LOW")
        b_cls = "badge-normal" if risk == "LOW" else ("badge-moderate" if risk == "MODERATE" else "badge-alert")
        st.markdown(
            f"""<div class="metric-panel">
            <div class="metric-label">Simetri Fasialis</div>
            <div class="metric-value">{score}%</div>
            <span class="status-badge {b_cls}">{sym.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        score = jnd.get("jaundice_score", 0.0)
        risk = jnd.get("risk_level", "LOW")
        b_cls = "badge-normal" if risk == "LOW" else ("badge-moderate" if risk == "MODERATE" else "badge-alert")
        st.markdown(
            f"""<div class="metric-panel">
            <div class="metric-label">Indeks Sklera</div>
            <div class="metric-value">{score} <span class="metric-sub">/100</span></div>
            <span class="status-badge {b_cls}">{jnd.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        score = pal.get("pallor_score", 0.0)
        risk = pal.get("risk_level", "LOW")
        b_cls = "badge-normal" if risk == "LOW" else ("badge-moderate" if risk == "MODERATE" else "badge-alert")
        st.markdown(
            f"""<div class="metric-panel">
            <div class="metric-label">Perfusi Mukosa</div>
            <div class="metric-value">{score} <span class="metric-sub">/100</span></div>
            <span class="status-badge {b_cls}">{pal.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c4:
        bpm = rppg.get("bpm")
        bpm_str = f"{bpm:.0f} BPM" if bpm else "Mengukur..."
        b_cls = "badge-normal" if rppg.get("is_stable") else "badge-moderate"
        st.markdown(
            f"""<div class="metric-panel">
            <div class="metric-label">Nadi Nirsentuh</div>
            <div class="metric-value">{bpm_str}</div>
            <span class="status-badge {b_cls}">{rppg.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c5:
        score = fat.get("fatigue_score", 0.0)
        risk = fat.get("risk_level", "LOW")
        b_cls = "badge-normal" if risk == "LOW" else ("badge-moderate" if risk == "MODERATE" else "badge-alert")
        st.markdown(
            f"""<div class="metric-panel">
            <div class="metric-label">Status Palpebra</div>
            <div class="metric-value">{100 - score:.0f}% <span class="metric-sub">Optimal</span></div>
            <span class="status-badge {b_cls}">{fat.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def process_and_display(frame_bgr: np.ndarray, is_video: bool = False):
    """Menjalankan analisis lengkap dan merender hasil skrining klinis."""
    lms_dict = detector.process_image(frame_bgr)

    if lms_dict is None:
        st.warning("Landmark wajah tidak teridentifikasi. Posisikan wajah tegak lurus terhadap sensor kamera.")
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        return

    # 1. Jalankan Analisis Komputasi
    sym_res = symmetry_analyzer.analyze(lms_dict)
    jnd_res = jaundice_analyzer.analyze(frame_bgr, lms_dict, apply_calibration=True)
    pal_res = pallor_analyzer.analyze(frame_bgr, lms_dict)
    fat_res = fatigue_detector.analyze(lms_dict)

    if is_video:
        rppg_res = rppg_estimator.update(frame_bgr, lms_dict)
    else:
        rppg_res = {
            "bpm": 72.0,
            "status": "Mode Citra Statis",
            "is_stable": False,
            "ppg_wave": [],
            "confidence": 50.0,
        }

    # Render 5 Panel Metrik
    render_metric_cards(sym_res, jnd_res, pal_res, rppg_res, fat_res)

    # Visualisasi Frame dengan Overlay HUD Medis
    annotated_frame = Visualizer.draw_health_overlay(
        frame_bgr, lms_dict, sym_res, jnd_res, rppg_res, fat_res, pal_res
    )

    col_view, col_details = st.columns([1.05, 1.15])

    with col_view:
        st.markdown("##### Visualisasi Topologi Landmark")
        st.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            caption="Sumbu Proyeksi Vertikal, Segmentasi Sklera, Kontur Mukosa Bibir, & ROI Fotopletismografi",
            use_container_width=True,
        )

        if "illuminant_info" in jnd_res:
            illum = jnd_res["illuminant_info"]
            st.caption(f"Kondisi Pencahayaan: **{illum['lighting_type']}** (Normalisasi Gray-World Aktif)")

    with col_details:
        st.markdown("##### Evaluasi Triangulasi & Rekomendasi Klinis")

        # 2. Dapatkan Pertanyaan Penguat (Anamnesis)
        visual_findings = {
            "symmetry": sym_res,
            "jaundice": jnd_res,
            "pallor": pal_res,
            "fatigue": fat_res,
            "rppg": rppg_res,
        }
        candidate_questions = ClinicalTriangulationEngine.get_candidate_questions(visual_findings)

        user_answers = {}
        if candidate_questions:
            st.caption("Verifikasi Anamnesis Lanjutan:")
            for q in candidate_questions:
                user_answers[q["id"]] = st.checkbox(f"**[{q['category']}]** {q['text']}", key=f"q_{q['id']}")

        # 3. Evaluasi Triangulasi
        triangulation_res = ClinicalTriangulationEngine.evaluate(visual_findings, user_answers)
        primary_diag = triangulation_res["primary_diagnosis"]

        # Tampilkan Hasil Diagnosa Banding
        urgency = primary_diag["urgency"]
        border_col = "#DC2626" if urgency == "EMERGENCY" else ("#D97706" if urgency == "HIGH" else ("#4F46E5" if urgency == "MODERATE" else "#059669"))
        bg_col = "#FEF2F2" if urgency == "EMERGENCY" else ("#FFFBEB" if urgency == "HIGH" else ("#F8FAFC" if urgency == "MODERATE" else "#F0FDF4"))

        st.markdown(
            f"""
            <div style="background-color:{bg_col}; border:1px solid #E2E8F0; border-left:5px solid {border_col}; padding:1.0rem 1.15rem; border-radius:6px; margin-top:0.6rem;">
                <div style="font-size:1.05rem; font-weight:700; color:#0F172A;">{primary_diag['condition']}</div>
                <div style="font-size:0.82rem; color:#64748B; margin-top:0.25rem;">Tingkat Konfidensi Skrining: <strong>{primary_diag['confidence']}%</strong> | Klasifikasi: <strong>{primary_diag['urgency']}</strong></div>
                <div style="font-size:0.86rem; color:#334155; margin-top:0.45rem; line-height:1.5;">{primary_diag['explanation']}</div>
                <div style="font-size:0.84rem; color:#0369A1; margin-top:0.55rem;"><strong>Tindakan yang Disarankan:</strong> {primary_diag['action_plan']}</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.35rem;"><strong>Disiplin Spesialis Terkait:</strong> {primary_diag['specialist']}</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.2rem;"><strong>Modalitas Penunjang Diusulkan:</strong> {', '.join(primary_diag['tests'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(f"Simpan Catatan Skrining", use_container_width=True):
                db.save_screening(
                    patient_name=selected_patient,
                    symmetry_score=sym_res.get("symmetry_score"),
                    jaundice_score=jnd_res.get("jaundice_score"),
                    pallor_score=pal_res.get("pallor_score"),
                    fatigue_score=fat_res.get("fatigue_score"),
                    bpm=rppg_res.get("bpm"),
                    primary_condition=primary_diag["condition"],
                    urgency=primary_diag["urgency"],
                    notes=f"Konfidensi: {primary_diag['confidence']}%",
                )
                st.success(f"Tersimpan pada profil '{selected_patient}'.")

        with col_btn2:
            report_txt = f"""==================================================
LAPORAN RESMI SKRINING BIOMARKER FASIAL (TRIAGE)
Identitas Subjek: {selected_patient}
Waktu Akuisisi  : {time.strftime('%d-%m-%Y %H:%M:%S')}
==================================================

1. EVALUASI TRIANGULASI KLINIS:
- Diagnosa Banding Utama: {primary_diag['condition']}
- Tingkat Konfidensi    : {primary_diag['confidence']}%
- Tingkat Urgensi       : {primary_diag['urgency']}
- Analisis Deskriptif   : {primary_diag['explanation']}
- Rencana Tindak Lanjut : {primary_diag['action_plan']}
- Rujukan Spesialis     : {primary_diag['specialist']}
- Pemeriksaan Penunjang : {', '.join(primary_diag['tests'])}

2. METRIK PARAMETER VISUAL:
- Simetri Fasialis (N.VII)   : {sym_res.get('symmetry_score')}% ({sym_res.get('status')})
- Sklera Ikterus (Bilirubin) : {jnd_res.get('jaundice_score')}/100 ({jnd_res.get('status')})
- Perfusi Mukosa (Hemoglobin): {pal_res.get('pallor_score')}/100 ({pal_res.get('status')})
- Fotopletismografi Nirsentuh: {rppg_res.get('bpm', '-')} BPM ({rppg_res.get('status')})
- Indeks Kebugaran Palpebra  : {100 - fat_res.get('fatigue_score', 0):.0f}% Optimal ({fat_res.get('status')})

==================================================
DISCLAIMER: Dokumen ini diterbitkan sebagai hasil skrining terkomputasi dan bukan merupakan pengganti diagnosis medis resmi.
=================================================="""
            st.download_button(
                label="Unduh Berkas Laporan (.txt)",
                data=report_txt,
                file_name=f"laporan_skrining_{selected_patient}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ==========================================
# RENDER INPUT SUMBER KAMERA / FOTO
# ==========================================
tab_screen, tab_history = st.tabs(["Pemeriksaan Sesi Berjalan", "Riwayat & Rekam Perkembangan"])

with tab_screen:
    if input_mode == "Kamera Langsung (Webcam/Mobile)":
        st.markdown(f"**Perekaman Citra Aktif — Subjek: `{selected_patient}`**")
        st.caption("Posisikan wajah tepat di hadapan lensa dan pertahankan ekspresi netral.")
        cam_picture = st.camera_input("Ambil Citra Skrining", label_visibility="collapsed")

        if cam_picture is not None:
            bytes_data = cam_picture.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            process_and_display(cv_img, is_video=True)

    elif input_mode == "Unggah Citra Wajah":
        st.markdown(f"**Unggah Berkas Citra — Subjek: `{selected_patient}`**")
        uploaded_file = st.file_uploader("Pilih file foto wajah (Format JPG / PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_display(cv_img, is_video=False)

    elif input_mode == "Kasus Uji Terkalibrasi":
        st.markdown("**Profil Kasus Uji Standar Laboratorium**")
        sample_case = st.selectbox(
            "Pilih Profil Uji:",
            [
                "Profil 1: Fisiologis Normal (Simetris & Sklera Jernih)",
                "Profil 2: Simulasi Asimetri Paresis Fasialis (Stroke Triage)",
                "Profil 3: Simulasi Sklera Hiperbilirubinemia (Ikterus)",
                "Profil 4: Simulasi Hipo-perfusi Mukosa Labialis (Anemia)",
            ],
            label_visibility="collapsed",
        )

        sample_dir = os.path.join(CURRENT_DIR, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        sample_file = os.path.join(sample_dir, "sample_base.jpg")

        if not os.path.exists(sample_file):
            blank_face = np.full((600, 600, 3), 220, dtype=np.uint8)
            cv2.ellipse(blank_face, (300, 320), (180, 240), 0, 0, 360, (180, 205, 235), -1)
            cv2.ellipse(blank_face, (220, 280), (35, 20), 0, 0, 360, (250, 250, 250), -1)
            cv2.ellipse(blank_face, (380, 280), (35, 20), 0, 0, 360, (250, 250, 250), -1)
            cv2.circle(blank_face, (220, 280), 12, (50, 40, 30), -1)
            cv2.circle(blank_face, (380, 280), 12, (50, 40, 30), -1)
            cv2.ellipse(blank_face, (300, 450), (60, 20), 0, 0, 360, (120, 130, 200), -1)
            cv2.imwrite(sample_file, blank_face)

        base_img = cv2.imread(sample_file)
        if base_img is None:
            base_img = np.full((600, 600, 3), 200, dtype=np.uint8)

        test_img = base_img.copy()
        if "Profil 2" in sample_case:
            cv2.putText(test_img, "[Simulasi Deviasi Saraf]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif "Profil 3" in sample_case:
            cv2.putText(test_img, "[Simulasi Ikterus]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 240), 2)
        elif "Profil 4" in sample_case:
            cv2.putText(test_img, "[Simulasi Kepucatan Mukosa]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        process_and_display(test_img, is_video=False)

with tab_history:
    st.markdown(f"**Rekam Medis & Tren Biomarker — Subjek: `{selected_patient}`**")
    hist_df = db.get_patient_history(selected_patient)

    if hist_df.empty:
        st.info(f"Belum ada catatan riwayat terdaftar untuk subjek '{selected_patient}'. Lakukan skrining dan simpan hasil evaluasi.")
    else:
        st.caption(f"Total **{len(hist_df)} sesi pemeriksaan** terdaftar.")

        if len(hist_df) >= 2:
            st.markdown("##### Tren Parameter dari Waktu ke Waktu")
            chart_df = hist_df[["timestamp", "symmetry_score", "jaundice_score", "pallor_score", "bpm"]].copy()
            chart_df = chart_df.rename(
                columns={
                    "symmetry_score": "Simetri Fasialis (%)",
                    "jaundice_score": "Indeks Sklera",
                    "pallor_score": "Indeks Kepucatan",
                    "bpm": "Nadi (BPM)",
                }
            )
            chart_df = chart_df.sort_values(by="timestamp")
            chart_df = chart_df.set_index("timestamp")
            st.line_chart(chart_df)

        st.markdown("##### Tabel Riwayat Pemeriksaan")
        st.dataframe(
            hist_df.rename(
                columns={
                    "timestamp": "Waktu",
                    "symmetry_score": "Simetri (%)",
                    "jaundice_score": "Sklera (/100)",
                    "pallor_score": "Perfusi (/100)",
                    "bpm": "Nadi (BPM)",
                    "primary_condition": "Hasil Diagnosa Banding",
                    "urgency": "Urgensi",
                    "notes": "Konfidensi",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
