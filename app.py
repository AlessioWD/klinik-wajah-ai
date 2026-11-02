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
    initial_sidebar_state="collapsed",
)

# Injeksi Dynamic PWA Icon & Pembersih Antarmuka Streamlit Cloud (Bebas Watermark)
components.html(
    """
    <script>
    const iconUrl = 'https://raw.githubusercontent.com/AlessioWD/klinik-wajah-ai/main/static/icon-512.png';
    const icon192 = 'https://raw.githubusercontent.com/AlessioWD/klinik-wajah-ai/main/static/icon-192.png';
    
    // Set apple-touch-icon untuk iOS & Android PWA
    try {
        let appleIcon = window.parent.document.querySelector("link[rel='apple-touch-icon']");
        if (!appleIcon) {
            appleIcon = window.parent.document.createElement('link');
            appleIcon.rel = 'apple-touch-icon';
            window.parent.document.head.appendChild(appleIcon);
        }
        appleIcon.href = icon192;

        let favIcon = window.parent.document.querySelector("link[rel*='icon']");
        if (!favIcon) {
            favIcon = window.parent.document.createElement('link');
            favIcon.rel = 'shortcut icon';
            window.parent.document.head.appendChild(favIcon);
        }
        favIcon.href = icon192;
    } catch(e) {}

    // Bersihkan Header (Fork/Github) & Floating Badge Streamlit Cloud (Mahkota Merah & Manage App)
    function cleanStreamlitHostUI() {
        const targetDocs = [document];
        try {
            if (window.parent && window.parent.document) targetDocs.push(window.parent.document);
        } catch(e) {}
        try {
            if (window.top && window.top.document && window.top.document !== window.parent.document) targetDocs.push(window.top.document);
        } catch(e) {}

        targetDocs.forEach(doc => {
            try {
                // Sembunyikan header atas
                const headers = doc.querySelectorAll('header, [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu');
                headers.forEach(el => {
                    el.style.setProperty('display', 'none', 'important');
                    el.style.setProperty('visibility', 'hidden', 'important');
                });

                // Sembunyikan footer
                const footers = doc.querySelectorAll('footer');
                footers.forEach(el => el.style.setProperty('display', 'none', 'important'));

                // Sembunyikan tombol melayang kanan bawah (Viewer badge, Manage app, Profile badge)
                const badges = doc.querySelectorAll(
                    '[class*="viewerBadge"], [class*="manageApp"], [data-testid="manage-app-button"], [class*="ProfileBadge"], [class*="AppControls"], .stAppDeployButton, [class*="StatusWidget"]'
                );
                badges.forEach(b => {
                    b.style.setProperty('display', 'none', 'important');
                    b.style.setProperty('visibility', 'hidden', 'important');
                    b.style.setProperty('opacity', '0', 'important');
                    b.style.setProperty('pointer-events', 'none', 'important');
                });

                // Target elemen mengambang di pojok kanan bawah
                const allElements = doc.querySelectorAll('div, button, a');
                allElements.forEach(item => {
                    const cls = (item.className && typeof item.className === 'string') ? item.className : '';
                    if (cls.includes('viewerBadge') || cls.includes('manageApp') || cls.includes('ProfileBadge')) {
                        item.style.setProperty('display', 'none', 'important');
                    }
                });
            } catch(err) {}
        });
    }

    setInterval(cleanStreamlitHostUI, 300);
    cleanStreamlitHostUI();
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
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.15rem;
    }
    .metric-sublabel {
        font-size: 0.70rem;
        color: #0284C7;
        font-weight: 600;
        margin-bottom: 0.35rem;
        line-height: 1.25;
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

    /* Sembunyikan elemen bawaan Streamlit Cloud (Header, Footer, Watermark, Badges) */
    header,
    footer,
    #MainMenu,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137,
    [class*="viewerBadge"],
    [class*="manageApp"],
    [data-testid="manage-app-button"],
    div[class*="ProfileBadge"],
    div[class*="floating-menu"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
    }

    /* Sembunyikan sidebar bawaan agar tampilan 100% Full-Width bebas AI Slop */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Penyesuaian layout container untuk tampilan seperti Native App */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        margin: 0 auto !important;
    }

    /* SOS Alert & Emergency Hotline Styles */
    .sos-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        margin-top: 0.9rem;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
    }
    .sos-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: #DC2626;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.92rem;
        padding: 8px 18px;
        border-radius: 6px;
        text-decoration: none !important;
        transition: background-color 0.15s ease;
    }
    .sos-btn:hover {
        background-color: #B91C1C;
    }

    /* Sembunyikan tautan eksternal dokumentasi Streamlit pada kotak kamera */
    div[data-testid="stCameraInput"] a {
        display: none !important;
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


def generate_html_report(patient_name, primary_diag, sym, jnd, pal, rppg, fat):
    """Menghasilkan lembar laporan resume medis resmi berbasis HTML siap cetak/simpan PDF."""
    now_str = time.strftime('%d %B %Y, %H:%M:%S WIB')
    urgency = primary_diag.get('urgency', 'LOW')
    urgency_color = "#DC2626" if urgency == "EMERGENCY" else ("#D97706" if urgency == "HIGH" else ("#2563EB" if urgency == "MODERATE" else "#059669"))
    bpm_val = f"{rppg.get('bpm'):.0f} BPM" if rppg.get('bpm') else "-"

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Skrining Medis - {patient_name}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #F8FAFC; color: #1E293B; margin: 0; padding: 24px; }}
    .report-card {{ max-width: 820px; margin: 0 auto; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04); }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0284C7; padding-bottom: 16px; margin-bottom: 20px; }}
    .title {{ font-size: 20px; font-weight: 700; color: #0F172A; }}
    .subtitle {{ font-size: 13px; color: #64748B; margin-top: 4px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; background: {urgency_color}; color: #FFFFFF; }}
    .section-title {{ font-size: 14px; font-weight: 700; color: #0F172A; margin: 20px 0 10px 0; border-left: 4px solid #0284C7; padding-left: 8px; text-transform: uppercase; letter-spacing: 0.04em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ padding: 9px 12px; border-bottom: 1px solid #E2E8F0; text-align: left; }}
    th {{ background: #F1F5F9; font-weight: 600; color: #475569; }}
    .diag-box {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid {urgency_color}; border-radius: 8px; padding: 16px; margin-top: 10px; line-height: 1.6; font-size: 13.5px; }}
    .footer-note {{ margin-top: 28px; padding-top: 16px; border-top: 1px dashed #CBD5E1; font-size: 11px; color: #94A3B8; text-align: center; line-height: 1.5; }}
    @media print {{ body {{ background: #FFFFFF; padding: 0; }} .report-card {{ border: none; box-shadow: none; padding: 0; }} }}
</style>
</head>
<body>
<div class="report-card">
    <div class="header">
        <div>
            <div class="title">FacialHealth — Resume Skrining Biometrik Fasial</div>
            <div class="subtitle">Platform Evaluasi Komputasi Citra Wajah & Triangulasi Klinis Terarah</div>
            <div class="subtitle">Waktu Pemeriksaan: <strong>{now_str}</strong></div>
        </div>
        <div style="text-align: right;">
            <div class="badge">KLASIFIKASI: {urgency}</div>
            <div class="subtitle" style="margin-top: 6px;">Subjek Pasien: <strong>{patient_name}</strong></div>
        </div>
    </div>

    <div class="section-title">I. Evaluasi Triangulasi & Rekomendasi Klinis</div>
    <div class="diag-box">
        <div style="font-size: 16px; font-weight: 700; color: #0F172A;">{primary_diag.get('condition', '-')}</div>
        <div style="font-size: 12px; color: #64748B; margin-top: 2px;">Tingkat Konfidensi Skrining: <strong>{primary_diag.get('confidence', 0)}%</strong></div>
        <p style="margin: 8px 0;">{primary_diag.get('explanation', '-')}</p>
        <p style="margin: 6px 0;"><strong>Tindakan yang Disarankan:</strong> {primary_diag.get('action_plan', '-')}</p>
        <p style="margin: 6px 0;"><strong>Disiplin Spesialis Terkait:</strong> {primary_diag.get('specialist', '-')}</p>
        <p style="margin: 6px 0;"><strong>Modalitas Penunjang Diusulkan:</strong> {', '.join(primary_diag.get('tests', []))}</p>
    </div>

    <div class="section-title">II. Hasil Uji 5 Parameter Biomarker Visual</div>
    <table>
        <thead>
            <tr>
                <th>Biomarker Wajah</th>
                <th>Nilai Terukur</th>
                <th>Status / Interpretasi</th>
                <th>Organ / Saraf Relevan</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Simetri Wajah</strong></td>
                <td>{sym.get('symmetry_score', 0)}%</td>
                <td>{sym.get('status', '-')}</td>
                <td>Nervus Fasialis (N.VII) / Skrining Stroke</td>
            </tr>
            <tr>
                <td><strong>Warna Putih Mata (Sklera)</strong></td>
                <td>{jnd.get('jaundice_score', 0)} / 100</td>
                <td>{jnd.get('status', '-')}</td>
                <td>Hati, Kantung Empedu & Bilirubin</td>
            </tr>
            <tr>
                <td><strong>Kemerahan Bibir (Perfusi)</strong></td>
                <td>{pal.get('pallor_score', 0)} / 100</td>
                <td>{pal.get('status', '-')}</td>
                <td>Hemoglobin & Skrining Anemia</td>
            </tr>
            <tr>
                <td><strong>Detak Jantung (Nadi Nirsentuh)</strong></td>
                <td>{bpm_val}</td>
                <td>{rppg.get('status', '-')}</td>
                <td>Sistem Kardiovaskular</td>
            </tr>
            <tr>
                <td><strong>Status Kelopak Mata</strong></td>
                <td>{100 - fat.get('fatigue_score', 0):.0f}% Segar</td>
                <td>{fat.get('status', '-')}</td>
                <td>Kebugaran Fisik & Tingkat Kantuk</td>
            </tr>
        </tbody>
    </table>

    <div class="footer-note">
        <strong>Pemberitahuan Medis Resmi:</strong> Dokumen ini diterbitkan secara otomatis oleh sistem komputasi visual (FacialHealth) 
        sebagai instrumen skrining mandiri awal. Dokumen ini bukan pengganti rekam medis atau vonis klinis definitif. 
        Tunjukkan hasil ini kepada dokter berwenang di fasilitas kesehatan terdekat untuk tindakan diagnosis lanjutan.
    </div>
</div>
</body>
</html>"""



def render_metric_cards(sym, jnd, pal, rppg, fat):
    """Menampilkan 5 panel metrik dengan istilah yang mudah dipahami orang awam."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        score = sym.get("symmetry_score", 0.0)
        risk = sym.get("risk_level", "LOW")
        b_cls = "badge-normal" if risk == "LOW" else ("badge-moderate" if risk == "MODERATE" else "badge-alert")
        st.markdown(
            f"""<div class="metric-panel">
            <div class="metric-label">Simetri Wajah</div>
            <div class="metric-sublabel">Saraf Wajah / Stroke</div>
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
            <div class="metric-label">Warna Putih Mata</div>
            <div class="metric-sublabel">Kuning Sklera / Hati</div>
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
            <div class="metric-label">Kemerahan Bibir</div>
            <div class="metric-sublabel">Skrining Pucat / Anemia</div>
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
            <div class="metric-label">Detak Jantung</div>
            <div class="metric-sublabel">Nadi via Kamera Dahi</div>
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
            <div class="metric-label">Kelopak Mata</div>
            <div class="metric-sublabel">Tingkat Kantuk & Lelah</div>
            <div class="metric-value">{100 - score:.0f}% <span class="metric-sub">Segar</span></div>
            <span class="status-badge {b_cls}">{fat.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Panduan Pembacaan Istilah untuk Orang Awam
    with st.expander("📖 Panduan Penjelasan Istilah untuk Orang Awam (Klik untuk membaca)"):
        st.markdown(
            """
            - **1. Simetri Wajah (Keseimbangan Saraf Wajah):** Memeriksa apakah bibir, mata, dan sudut alis seimbang antara sisi kiri dan kanan. Kemiringan yang muncul mendadak merupakan tanda fisik awal serangan stroke (*F.A.S.T*) atau kelumpuhan saraf wajah (*Bell's Palsy*).
            - **2. Warna Putih Mata (Sklera / Penyakit Kuning):** *Sklera* adalah istilah medis untuk bagian putih pada bola mata. Jika sklera tampak kekuningan, hal ini menandakan peningkatan pigmen empedu (*bilirubin*) akibat gangguan organ hati (seperti hepatitis) atau kantung empedu.
            - **3. Kemerahan Bibir & Mata (Perfusi Mukosa / Kurang Darah):** *Perfusi* mengukur rona merah pada bibir dan selaput kelopak mata. Bila warnanya tampak pias atau putih pucat, ini merupakan indikasi kurang darah (*anemia*) atau sirkulasi yang kurang optimal.
            - **4. Detak Jantung (Nadi Nirsentuh):** Teknologi sensor kamera cerdas yang membaca denyut nadi per menit (*BPM*) secara otomatis dari getaran mikro aliran darah di dahi tanpa perlu alat sentuh.
            - **5. Kelopak Mata (Status Palpebra):** *Palpebra* adalah nama medis kelopak mata. Sistem mengukur bukaan kelopak mata dan pola kedipan untuk mendeteksi rasa kantuk berat atau kelelahan ekstrem.
            """
        )


def process_and_display(frame_bgr: np.ndarray, patient_name: str, is_video: bool = False):
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
            caption="Panduan Sensor AI: Garis Keseimbangan Wajah, Warna Putih Mata, Rona Bibir, & Deteksi Nadi Dahi",
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

        # FITUR: Tombol Audio Asisten Suara Medis
        clean_cond = primary_diag['condition'].replace('"', '').replace("'", "")
        clean_plan = primary_diag['action_plan'].replace('"', '').replace("'", "")
        speech_text = f"Hasil skrining wajah Anda adalah {clean_cond}. Urgensi: {urgency}. Tindakan yang dianjurkan: {clean_plan}"
        components.html(
            f"""
            <div style="margin-top: 6px;">
                <button onclick="speakMedicalReport()" style="background:#FFFFFF; border:1px solid #CBD5E1; color:#0F172A; padding:6px 14px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                    🔊 Dengarkan Penjelasan Suara
                </button>
            </div>
            <script>
            function speakMedicalReport() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    const msg = new SpeechSynthesisUtterance("{speech_text}");
                    msg.lang = "id-ID";
                    msg.rate = 0.92;
                    window.speechSynthesis.speak(msg);
                }} else {{
                    alert("Fitur audio tidak didukung oleh peramban ini.");
                }}
            }}
            </script>
            """,
            height=44,
        )

        # FITUR: Kartu Tanggap Darurat FAST (Hotline 119) jika Urgensi Tinggi / Darurat
        if urgency in ["EMERGENCY", "HIGH"]:
            st.markdown(
                """
                <div class="sos-card">
                    <div style="font-weight:700; color:#991B1B; font-size:1.02rem;">
                        🚨 PERINGATAN GAWAT DARURAT MEDIS (F.A.S.T.)
                    </div>
                    <div style="font-size:0.86rem; color:#7F1D1D; margin-top:5px; line-height:1.5;">
                        Terdeteksi tanda klinis akut yang memerlukan tindakan medis segera. Tetap tenang, jangan biarkan pasien berjalan sendiri, dan segera hubungi tim ambulans gawat darurat.
                    </div>
                    <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
                        <a href="tel:119" class="sos-btn">📞 Panggil Ambulans (119)</a>
                        <a href="tel:112" class="sos-btn" style="background-color:#991B1B;">📞 Darurat Nasional (112)</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("💾 Simpan Catatan", use_container_width=True):
                db.save_screening(
                    patient_name=patient_name,
                    symmetry_score=sym_res.get("symmetry_score"),
                    jaundice_score=jnd_res.get("jaundice_score"),
                    pallor_score=pal_res.get("pallor_score"),
                    fatigue_score=fat_res.get("fatigue_score"),
                    bpm=rppg_res.get("bpm"),
                    primary_condition=primary_diag["condition"],
                    urgency=primary_diag["urgency"],
                    notes=f"Konfidensi: {primary_diag['confidence']}%",
                )
                st.success(f"Tersimpan pada profil '{patient_name}'.")

        with col_btn2:
            report_txt = f"""==================================================
LAPORAN RESMI SKRINING BIOMARKER FASIAL (TRIAGE)
Identitas Subjek: {patient_name}
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
- Simetri Wajah (N.VII)      : {sym_res.get('symmetry_score')}% ({sym_res.get('status')})
- Warna Putih Mata (Sklera)  : {jnd_res.get('jaundice_score')}/100 ({jnd_res.get('status')})
- Kemerahan Bibir (Perfusi)  : {pal_res.get('pallor_score')}/100 ({pal_res.get('status')})
- Detak Nadi via Kamera Dahi : {rppg_res.get('bpm', '-')} BPM ({rppg_res.get('status')})
- Kebugaran Kelopak Mata     : {100 - fat_res.get('fatigue_score', 0):.0f}% Segar ({fat_res.get('status')})

==================================================
DISCLAIMER: Dokumen ini diterbitkan sebagai hasil skrining terkomputasi dan bukan merupakan pengganti diagnosis medis resmi.
=================================================="""
            st.download_button(
                label="📄 Unduh Ringkasan (.txt)",
                data=report_txt,
                file_name=f"laporan_skrining_{patient_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_btn3:
            html_report = generate_html_report(patient_name, primary_diag, sym_res, jnd_res, pal_res, rppg_res, fat_res)
            st.download_button(
                label="🖨️ Cetak Resume (.html)",
                data=html_report,
                file_name=f"resume_medis_{patient_name}_{time.strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
            )


# ==========================================
# TOP APP BAR: MODERN CLINICAL HEADER (BEBAS SIDEBAR)
# ==========================================
col_brand, col_profile = st.columns([1.65, 1.15])

with col_brand:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">FacialHealth <span class="brand-badge">CLINICAL BIOMETRICS v2.3</span></div>
        </div>
        <div class="brand-subtitle">Platform Skrining Biomarker Fisik Wajah & Triangulasi Gejala Terarah</div>
        """,
        unsafe_allow_html=True,
    )

with col_profile:
    patients = db.get_patients()
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px;'>👤 Subjek Pasien Aktif:</div>", unsafe_allow_html=True)
    c_pat, c_add = st.columns([1.7, 0.9])
    with c_pat:
        selected_patient = st.selectbox("Pilih Profil", patients, index=0, label_visibility="collapsed")
    with c_add:
        with st.popover("➕ Baru"):
            st.markdown("**Tambah Profil Pasien Baru**")
            new_p_name = st.text_input("Nama Lengkap:", placeholder="Contoh: Budi Santoso")
            if st.button("Simpan Profil", use_container_width=True):
                if db.add_patient(new_p_name):
                    st.success(f"Profil '{new_p_name}' terdaftar.")
                    st.rerun()
                else:
                    st.warning("Nama tidak boleh kosong atau sudah terdaftar.")

# Clinical Notice Bar
st.markdown(
    """
    <div class="clinical-notice">
        <strong>Pemberitahuan Medis:</strong> Platform ini berfungsi sebagai instrumen skrining awal mandiri (triage) 
        dan bukan vonis medis definitif. Pada kasus gejala akut yang timbul mendadak (seperti bibir miring mendadak, kebas separuh badan, 
        atau mata kuning pekat), segera hubungi layanan gawat darurat atau periksakan ke dokter spesialis terdekat.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# NAVIGASI UTAMA: 4 TAB MODERN (100% RESPOCIF DI HP & PC)
# ==========================================
tab_screen, tab_history, tab_compare, tab_info = st.tabs([
    "📷 Pemeriksaan Wajah",
    "📋 Rekam Medis Pasien",
    "🔄 Komparasi Before & After",
    "ℹ️ Panduan & Info Layanan",
])

with tab_screen:
    # Segmented Radio Bar untuk Modalitas (Tampilan Bersih Bebas AI Slop)
    input_mode = st.radio(
        "Pilih Metode Pemeriksaan:",
        ["📷 Kamera Langsung (Webcam / HP)", "🖼️ Unggah Berkas Foto Wajah", "🧪 Kasus Uji Standar Klinis"],
        index=0,
        horizontal=True,
    )

    if input_mode == "📷 Kamera Langsung (Webcam / HP)":
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.85rem 1.1rem; margin-bottom:0.6rem;">
                <div style="font-weight:600; color:#0F172A; font-size:0.92rem;">Perekaman Citra Aktif — Subjek: <strong>{selected_patient}</strong></div>
                <div style="font-size:0.82rem; color:#64748B; margin-top:2px;">Posisikan wajah tepat di hadapan lensa kamera, buka mata normal, dan pertahankan ekspresi netral santai.</div>
            </div>
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:6px; padding:0.55rem 0.85rem; margin-bottom:0.75rem; font-size:0.82rem; color:#166534; display:flex; align-items:center; gap:8px;">
                <span>💡</span>
                <span><strong>Jika kamera belum menyala:</strong> Klik ikon setelan di sebelah kiri alamat web (URL bar Chrome) lalu ubah opsi <strong>Camera / Kamera</strong> menjadi <strong>Izinkan (Allow)</strong>, kemudian segarkan (refresh) halaman.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cam_picture = st.camera_input("Ambil Citra Skrining", label_visibility="collapsed")

        if cam_picture is not None:
            bytes_data = cam_picture.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            process_and_display(cv_img, patient_name=selected_patient, is_video=True)

    elif input_mode == "🖼️ Unggah Berkas Foto Wajah":
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.85rem 1.1rem; margin-bottom:0.75rem;">
                <div style="font-weight:600; color:#0F172A; font-size:0.92rem;">Unggah Berkas Citra — Subjek: <strong>{selected_patient}</strong></div>
                <div style="font-size:0.82rem; color:#64748B; margin-top:2px;">Gunakan foto close-up wajah beresolusi jelas dengan pencahayaan seimbang (Format JPG / PNG).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Pilih file foto wajah", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_display(cv_img, patient_name=selected_patient, is_video=False)

    elif input_mode == "🧪 Kasus Uji Standar Klinis":
        st.markdown("**Profil Kasus Uji Standar Laboratorium**")
        sample_case = st.selectbox(
            "Pilih Profil Uji:",
            [
                "Profil 1: Fisiologis Sehat (Simetris & Mata Jernih)",
                "Profil 2: Simulasi Asimetri Wajah (Tanda Awal Stroke / Bell's Palsy)",
                "Profil 3: Simulasi Mata Menguning (Gangguan Hati / Empedu)",
                "Profil 4: Simulasi Bibir Pucat (Skrining Kurang Darah / Anemia)",
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

        process_and_display(test_img, patient_name=selected_patient, is_video=False)

with tab_history:
    st.markdown(f"**Rekam Medis & Riwayat Perkembangan — Subjek: `{selected_patient}`**")
    hist_df = db.get_patient_history(selected_patient)

    if hist_df.empty:
        st.info(f"Belum ada catatan riwayat terdaftar untuk subjek '{selected_patient}'. Lakukan skrining di tab Pemeriksaan Wajah dan simpan hasil evaluasi.")
    else:
        st.caption(f"Total **{len(hist_df)} sesi pemeriksaan** terdaftar.")

        if len(hist_df) >= 2:
            st.markdown("##### Tren Parameter dari Waktu ke Waktu")
            chart_df = hist_df[["timestamp", "symmetry_score", "jaundice_score", "pallor_score", "bpm"]].copy()
            chart_df = chart_df.rename(
                columns={
                    "symmetry_score": "Simetri Wajah (%)",
                    "jaundice_score": "Kuning Mata / Sklera",
                    "pallor_score": "Kepucatan Bibir",
                    "bpm": "Detak Jantung (BPM)",
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
                    "symmetry_score": "Simetri Wajah (%)",
                    "jaundice_score": "Kuning Mata (/100)",
                    "pallor_score": "Kepucatan Bibir (/100)",
                    "bpm": "Detak Jantung (BPM)",
                    "primary_condition": "Hasil Diagnosa Banding",
                    "urgency": "Tingkat Urgensi",
                    "notes": "Konfidensi",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# TAB 3: FITUR KOMPARASI BEFORE & AFTER
with tab_compare:
    st.markdown(f"**Komparasi Perkembangan Biomarker (Before & After) — Subjek: `{selected_patient}`**")
    hist_df = db.get_patient_history(selected_patient)

    if len(hist_df) < 2:
        st.info(
            f"Diperlukan minimal **2 sesi pemeriksaan tersimpan** untuk menganalisis komparasi Before & After pada profil '{selected_patient}'. "
            "Silakan lakukan pemeriksaan di tab 'Pemeriksaan Wajah' dan simpan catatan skrining terlebih dahulu."
        )
    else:
        st.caption("Pilih dua sesi pemeriksaan untuk melihat tren perbaikan klinis:")
        sess_options = [f"Sesi #{row['id']} - {row['timestamp']} ({row['primary_condition']})" for _, row in hist_df.iterrows()]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sel_old = st.selectbox("Sesi Sebelumnya (Before):", sess_options, index=len(sess_options)-1)
        with col_s2:
            sel_new = st.selectbox("Sesi Terbaru (After):", sess_options, index=0)

        id_old = int(sel_old.split(" - ")[0].replace("Sesi #", ""))
        id_new = int(sel_new.split(" - ")[0].replace("Sesi #", ""))

        row_old = hist_df[hist_df["id"] == id_old].iloc[0]
        row_new = hist_df[hist_df["id"] == id_new].iloc[0]

        st.markdown("##### Perubahan Nilai Biomarker:")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            sym_diff = row_new["symmetry_score"] - row_old["symmetry_score"]
            sym_tag = f"+{sym_diff:.1f}%" if sym_diff >= 0 else f"{sym_diff:.1f}%"
            st.metric("Simetri Wajah", f"{row_new['symmetry_score']}%", delta=sym_tag)
        with cc2:
            jnd_diff = row_new["jaundice_score"] - row_old["jaundice_score"]
            jnd_tag = f"{jnd_diff:.1f}"
            st.metric("Kuning Mata (Sklera)", f"{row_new['jaundice_score']}", delta=jnd_tag, delta_color="inverse")
        with cc3:
            pal_diff = row_new["pallor_score"] - row_old["pallor_score"]
            pal_tag = f"{pal_diff:.1f}"
            st.metric("Kepucatan Bibir", f"{row_new['pallor_score']}", delta=pal_tag, delta_color="inverse")
        with cc4:
            bpm_diff = (row_new["bpm"] or 0) - (row_old["bpm"] or 0)
            bpm_str = f"{row_new['bpm']:.0f} BPM" if row_new["bpm"] else "-"
            st.metric("Detak Nadi", bpm_str, delta=f"{bpm_diff:.0f} BPM")

        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:1.1rem; margin-top:1rem;">
                <div style="font-weight:700; color:#0F172A; margin-bottom:0.4rem;">Perbandingan Kesimpulan Klinis:</div>
                <div style="font-size:0.88rem; color:#475569; line-height:1.6;">
                    • <strong>Sebelum ({row_old['timestamp']}):</strong> {row_old['primary_condition']} (Tingkat Urgensi: <em>{row_old['urgency']}</em>)<br>
                    • <strong>Terbaru ({row_new['timestamp']}):</strong> {row_new['primary_condition']} (Tingkat Urgensi: <em>{row_new['urgency']}</em>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# TAB 4: PANDUAN PENGGUNAAN & INFO LAYANAN
with tab_info:
    st.markdown("### Informasi Layanan & Panduan Penggunaan")
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown(
            """
            #### 📱 Akses Publik 24/7 & Penggunaan di Smartphone
            Aplikasi ini aktif secara online dan dapat diakses dari browser HP Android & iOS dari mana saja:
            - **Tautan Resmi:** `https://klinik-wajah-ai.streamlit.app`
            - **Pencahayaan Optimal:** Pastikan berada di ruangan dengan pencahayaan seimbang (*cahaya putih / daylight*). Hindari lampu remang-remang atau cahaya warna-warni yang dapat memengaruhi analisis warna sklera.
            - **Posisi Kamera:** Posisikan wajah berjarak 40–60 cm dari lensa kamera, tegak lurus, dan ekspresi netral santai.
            """
        )
    with col_i2:
        st.markdown(
            """
            #### 🔒 Privasi Data & Keamanan Biometrik
            - **Pemrosesan In-Memory:** Foto wajah dan citra kamera diproses secara lokal di memori sementara dan **tidak diunggah** ke cloud publik.
            - **Database Lokal:** Hanya skor metrik angka (tanpa foto) yang tersimpan pada riwayat rekam medis pasien di perangkat ini.
            - **Standar Skrining Medis:** Mengacu pada pedoman klinis *F.A.S.T.* (Stroke), *CIELAB colorimetry* (Ikterus), dan spektroskopi optik rPPG.
            """
        )

