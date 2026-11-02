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

# Inisialisasi Halaman Streamlit
st.set_page_config(
    page_title="AI Facial Health Screening & Triage",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling CSS Modern Medis & Responsif Mobile + Tag PWA
st.markdown(
    """
    <head>
        <link rel="manifest" href="static/manifest.json">
        <meta name="theme-color" content="#1E3A8A">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <link rel="apple-touch-icon" href="https://img.icons8.com/color/192/medical-doctor.png">
    </head>
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.0rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    .badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-yellow {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-purple {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .diagnosis-card {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
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
st.sidebar.image("https://img.icons8.com/color/96/medical-doctor.png", width=65)
st.sidebar.title("Klinik Triage Wajah")

# 1. Pilihan Profil Pengguna (Multi-User)
st.sidebar.subheader("👤 Profil Pengguna")
patients = db.get_patients()
selected_patient = st.sidebar.selectbox("Pilih Profil:", patients, index=0)

with st.sidebar.expander("➕ Tambah Profil Baru"):
    new_patient_name = st.text_input("Nama Anggota / Pasien:")
    if st.button("Simpan Profil"):
        if db.add_patient(new_patient_name):
            st.success(f"Profil '{new_patient_name}' ditambahkan!")
            st.rerun()
        else:
            st.warning("Nama kosong atau sudah terdaftar.")

st.sidebar.markdown("---")

# 2. Pilihan Mode Input
input_mode = st.sidebar.radio(
    "Sumber Input Wajah:",
    ["📷 Kamera Langsung (Webcam/HP)", "📁 Unggah Foto Wajah", "🧪 Sampel Uji Coba"],
    index=0,
)

# 3. Informasi Akses dari HP (Mobile / Wi-Fi)
local_ip = get_local_ip()
with st.sidebar.expander("📱 Cara Buka di HP (Wi-Fi)"):
    st.markdown(
        f"""
        Untuk membuka aplikasi ini di **HP Android / iPhone** lewat Wi-Fi yang sama:
        
        1. Hubungkan HP Anda ke Wi-Fi yang sama dengan laptop.
        2. Buka browser di HP (Chrome / Safari).
        3. Ketik alamat ini di HP:
           `http://{local_ip}:8501`
        """
    )

with st.sidebar.expander("📲 Download & Install App (PWA)"):
    st.markdown(
        """
        **Cara Pasang Aplikasi di HP (Android/iOS):**
        1. Buka link web di Chrome/Safari HP.
        2. Ketuk tombol **Menu (Titik 3)** di pojok kanan atas Chrome, lalu pilih:  
           👉 **"Install Aplikasi"** atau **"Tambahkan ke Layar Utama"**.
        3. Aplikasi akan langsung terpasang di menu HP layaknya APK!
        
        **Cara Pasang di Laptop/PC:**
        * Klik ikon tanda panah/install di ujung kanan bilah alamat (URL bar) Chrome/Edge untuk memasang sebagai software desktop.
        """
    )

# Header Utama
st.markdown('<div class="main-title">🩺 AI Facial Health Screening & Triage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Skrining Multimodal: Asimetri Saraf (Stroke FAST), Sklera Ikterus (Liver), Kepucatan (Anemia), Denyut Nadi rPPG, & Anamnesis Cerdas</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ PEMBERITAHUAN MEDIS:</strong> Sistem ini adalah instrumen skrining awal (*triage mandiri*) dan <strong>BUKAN pengganti diagnosis dokter</strong>. 
        Jika mengalami keluhan mendadak (wajah miring, tangan lemas, bicara pelo, mata sangat kuning), segera hubungi <strong>IGD / 112</strong> atau fasilitas kesehatan terdekat.
    </div>
    """,
    unsafe_allow_html=True,
)


def render_metric_cards(sym, jnd, pal, rppg, fat):
    """Menampilkan 5 kartu parameter klinis."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        score = sym.get("symmetry_score", 0.0)
        risk = sym.get("risk_level", "LOW")
        b_cls = "badge-green" if risk == "LOW" else ("badge-yellow" if risk == "MODERATE" else "badge-red")
        st.markdown(
            f"""<div class="metric-card">
            <div style="font-size:0.75rem; color:#6B7280; font-weight:600;">SIMETRI SARAF</div>
            <div style="font-size:1.6rem; font-weight:700; color:#1F2937;">{score}%</div>
            <span class="{b_cls}">{sym.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        score = jnd.get("jaundice_score", 0.0)
        risk = jnd.get("risk_level", "LOW")
        b_cls = "badge-green" if risk == "LOW" else ("badge-yellow" if risk == "MODERATE" else "badge-red")
        st.markdown(
            f"""<div class="metric-card">
            <div style="font-size:0.75rem; color:#6B7280; font-weight:600;">SKLERA IKTERUS</div>
            <div style="font-size:1.6rem; font-weight:700; color:#1F2937;">{score} <span style="font-size:0.85rem; color:#9CA3AF;">/100</span></div>
            <span class="{b_cls}">{jnd.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        score = pal.get("pallor_score", 0.0)
        risk = pal.get("risk_level", "LOW")
        b_cls = "badge-green" if risk == "LOW" else ("badge-yellow" if risk == "MODERATE" else "badge-red")
        st.markdown(
            f"""<div class="metric-card">
            <div style="font-size:0.75rem; color:#6B7280; font-weight:600;">KEPUCATAN (ANEMIA)</div>
            <div style="font-size:1.6rem; font-weight:700; color:#1F2937;">{score} <span style="font-size:0.85rem; color:#9CA3AF;">/100</span></div>
            <span class="{b_cls}">{pal.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c4:
        bpm = rppg.get("bpm")
        bpm_str = f"{bpm} BPM" if bpm else "Mengukur..."
        b_cls = "badge-green" if rppg.get("is_stable") else "badge-yellow"
        st.markdown(
            f"""<div class="metric-card">
            <div style="font-size:0.75rem; color:#6B7280; font-weight:600;">DENYUT NADI rPPG</div>
            <div style="font-size:1.6rem; font-weight:700; color:#1F2937;">{bpm_str}</div>
            <span class="{b_cls}">{rppg.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with c5:
        score = fat.get("fatigue_score", 0.0)
        risk = fat.get("risk_level", "LOW")
        b_cls = "badge-green" if risk == "LOW" else ("badge-yellow" if risk == "MODERATE" else "badge-red")
        st.markdown(
            f"""<div class="metric-card">
            <div style="font-size:0.75rem; color:#6B7280; font-weight:600;">KEBUGARAN / EAR</div>
            <div style="font-size:1.6rem; font-weight:700; color:#1F2937;">{100 - score:.0f}% <span style="font-size:0.85rem; color:#9CA3AF;">Segar</span></div>
            <span class="{b_cls}">{fat.get('status', '-')}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def process_and_display(frame_bgr: np.ndarray, is_video: bool = False):
    """Menjalankan analisis lengkap dan merender hasil skrining klinis."""
    lms_dict = detector.process_image(frame_bgr)

    if lms_dict is None:
        st.warning("⚠️ Wajah tidak terdeteksi. Posisikan wajah di tengah bingkai kamera dengan cahaya yang merata.")
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
            "status": "Mode Foto Statis (Perlu Video Real-time)",
            "is_stable": False,
            "ppg_wave": [],
            "confidence": 50.0,
        }

    # Render 5 Kartu Indikator
    render_metric_cards(sym_res, jnd_res, pal_res, rppg_res, fat_res)

    # Visualisasi Frame dengan Overlay HUD Medis
    annotated_frame = Visualizer.draw_health_overlay(
        frame_bgr, lms_dict, sym_res, jnd_res, rppg_res, fat_res, pal_res
    )

    col_view, col_details = st.columns([1.05, 1.15])

    with col_view:
        st.subheader("🎯 Pemetaan Visual Wajah")
        st.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            caption="Sumbu Simetri (Cyan), Sklera Mata (Highlight), Mukosa Bibir, & ROI rPPG",
            use_container_width=True,
        )

        if "illuminant_info" in jnd_res:
            illum = jnd_res["illuminant_info"]
            st.caption(f"💡 Status Pencahayaan: **{illum['lighting_type']}** (Koreksi Gray-World Otomatis Aktif)")

    with col_details:
        st.subheader("📋 Triangulasi Klinis & Diagnosa Banding")

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
            st.markdown("##### 🩺 Pertanyaan Gejala Kunci Penguat:")
            st.caption("Jawab pertanyaan berikut untuk memperkuat kepastian dugaan penyakit:")
            for q in candidate_questions:
                user_answers[q["id"]] = st.checkbox(f"**[{q['category']}]** {q['text']}", key=f"q_{q['id']}")

        # 3. Evaluasi Triangulasi
        triangulation_res = ClinicalTriangulationEngine.evaluate(visual_findings, user_answers)
        primary_diag = triangulation_res["primary_diagnosis"]

        # Tampilkan Hasil Diagnosa Banding
        urgency = primary_diag["urgency"]
        card_bg = "#FEE2E2" if urgency == "EMERGENCY" else ("#FEF3C7" if urgency == "HIGH" else ("#EDE9FE" if urgency == "MODERATE" else "#DEF7EC"))
        border_col = "#DC2626" if urgency == "EMERGENCY" else ("#D97706" if urgency == "HIGH" else ("#7C3AED" if urgency == "MODERATE" else "#059669"))

        st.markdown(
            f"""
            <div style="background-color:{card_bg}; border-left:6px solid {border_col}; padding:0.9rem 1.1rem; border-radius:8px; margin-top:0.8rem;">
                <div style="font-size:1.15rem; font-weight:700; color:#1F2937;">{primary_diag['condition']}</div>
                <div style="font-size:0.88rem; color:#4B5563; margin-top:0.3rem;"><strong>Tingkat Kepastian Skrining:</strong> {primary_diag['confidence']}%</div>
                <div style="font-size:0.88rem; color:#374151; margin-top:0.4rem;">{primary_diag['explanation']}</div>
                <div style="font-size:0.88rem; color:#1E3A8A; margin-top:0.5rem;"><strong>Rencana Tindakan:</strong> {primary_diag['action_plan']}</div>
                <div style="font-size:0.88rem; color:#4B5563; margin-top:0.3rem;"><strong>Rujukan Spesialis:</strong> {primary_diag['specialist']}</div>
                <div style="font-size:0.88rem; color:#4B5563; margin-top:0.2rem;"><strong>Usulan Tes Lab/Pemeriksaan:</strong> {', '.join(primary_diag['tests'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Tombol Simpan Riwayat ke Database SQLite
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(f"💾 Simpan Hasil ke Profil '{selected_patient}'"):
                db.save_screening(
                    patient_name=selected_patient,
                    symmetry_score=sym_res.get("symmetry_score"),
                    jaundice_score=jnd_res.get("jaundice_score"),
                    pallor_score=pal_res.get("pallor_score"),
                    fatigue_score=fat_res.get("fatigue_score"),
                    bpm=rppg_res.get("bpm"),
                    primary_condition=primary_diag["condition"],
                    urgency=primary_diag["urgency"],
                    notes=f"Kepastian: {primary_diag['confidence']}%",
                )
                st.success(f"Hasil skrining berhasil disimpan ke rekam medis profil '{selected_patient}'!")

        with col_btn2:
            # Unduh Laporan Teks
            report_txt = f"""==================================================
LAPORAN RESMI SKRINING KESEHATAN WAJAH (AI TRIAGE)
Profil Pasien : {selected_patient}
Waktu Periksa : {time.strftime('%d-%m-%Y %H:%M:%S')}
==================================================

1. KESIMPULAN TRIANGULASI KLINIS:
- Dugaan Utama  : {primary_diag['condition']}
- Tingkat Yakin : {primary_diag['confidence']}%
- Urgensi Medis : {primary_diag['urgency']}
- Penjelasan    : {primary_diag['explanation']}
- Rencana Aksi  : {primary_diag['action_plan']}
- Rujukan Dokter: {primary_diag['specialist']}
- Tes Terkait   : {', '.join(primary_diag['tests'])}

2. RINCIAN BIOMARKER VISUAL:
- Simetri Saraf (Stroke FAST): {sym_res.get('symmetry_score')}% ({sym_res.get('status')})
- Sklera Ikterus (Liver/Bilirubin): {jnd_res.get('jaundice_score')}/100 ({jnd_res.get('status')})
- Kepucatan Mukosa (Anemia/Hb): {pal_res.get('pallor_score')}/100 ({pal_res.get('status')})
- Denyut Nadi Nirsentuh (rPPG): {rppg_res.get('bpm', '-')} BPM ({rppg_res.get('status')})
- Kebugaran & EAR Mata: {100 - fat_res.get('fatigue_score', 0):.0f}% Segar ({fat_res.get('status')})

==================================================
DISCLAIMER: Dokumen ini adalah hasil skrining visual otomatis dan BUKAN pengganti diagnosis dokter resmi.
=================================================="""
            st.download_button(
                label="📥 Unduh Laporan Skrining (.txt)",
                data=report_txt,
                file_name=f"laporan_skrining_{selected_patient}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )


# ==========================================
# RENDER INPUT SUMBER KAMERA / FOTO
# ==========================================
tab_screen, tab_history = st.tabs(["🔬 Pemeriksaan Wajah Saat Ini", "📊 Riwayat & Tren Pasien"])

with tab_screen:
    if input_mode == "📷 Kamera Langsung (Webcam/HP)":
        st.subheader(f"📷 Pengambilan Foto Skrining — {selected_patient}")
        st.write("Arahkan wajah lurus ke kamera, tahan posisi rileks, lalu klik tombol ambil gambar di bawah:")
        cam_picture = st.camera_input("Ambil Foto Skrining")

        if cam_picture is not None:
            bytes_data = cam_picture.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            process_and_display(cv_img, is_video=True)

    elif input_mode == "📁 Unggah Foto Wajah":
        st.subheader(f"📁 Unggah Foto Wajah — {selected_patient}")
        uploaded_file = st.file_uploader("Pilih file foto wajah (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_display(cv_img, is_video=False)

    elif input_mode == "🧪 Sampel Uji Coba":
        st.subheader("🧪 Uji Kasus Medis Bawaan")
        sample_case = st.selectbox(
            "Pilih Profil Kasus Medis Uji Coba:",
            [
                "Kasus 1: Wajah Normal, Simetris, & Sklera Sehat",
                "Kasus 2: Simulasi Asimetri Mulut Pelo (Stroke FAST Triage)",
                "Kasus 3: Simulasi Sklera Kuning (Hepatitis / Ikterus)",
                "Kasus 4: Simulasi Kepucatan Bibir (Suspek Anemia Defisiensi Besi)",
            ],
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
        if "Kasus 2" in sample_case:
            cv2.putText(test_img, "[Simulasi Asimetri Saraf]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif "Kasus 3" in sample_case:
            cv2.putText(test_img, "[Simulasi Ikterus Sklera Kuning]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 240), 2)
        elif "Kasus 4" in sample_case:
            cv2.putText(test_img, "[Simulasi Kepucatan Bibir]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        process_and_display(test_img, is_video=False)

with tab_history:
    st.subheader(f"📊 Rekam Medis & Tren Skrining — {selected_patient}")
    hist_df = db.get_patient_history(selected_patient)

    if hist_df.empty:
        st.info(f"Belum ada catatan riwayat skrining yang tersimpan untuk profil '{selected_patient}'. Lakukan skrining di tab sebelah dan klik tombol 'Simpan Hasil'.")
    else:
        st.write(f"Total **{len(hist_df)} sesi pemeriksaan** tercatat untuk profil ini.")

        # Grafik Tren Skor dari Waktu ke Waktu
        if len(hist_df) >= 2:
            st.markdown("##### 📈 Grafik Perkembangan Biomarker:")
            chart_df = hist_df[["timestamp", "symmetry_score", "jaundice_score", "pallor_score", "bpm"]].copy()
            chart_df = chart_df.rename(
                columns={
                    "symmetry_score": "Skor Simetri %",
                    "jaundice_score": "Indeks Ikterus",
                    "pallor_score": "Indeks Kepucatan",
                    "bpm": "Denyut Nadi (BPM)",
                }
            )
            chart_df = chart_df.sort_values(by="timestamp")
            chart_df = chart_df.set_index("timestamp")
            st.line_chart(chart_df)

        st.markdown("##### 📋 Tabel Riwayat Pemeriksaan:")
        st.dataframe(
            hist_df.rename(
                columns={
                    "timestamp": "Waktu Periksa",
                    "symmetry_score": "Simetri (%)",
                    "jaundice_score": "Ikterus (/100)",
                    "pallor_score": "Pucat (/100)",
                    "bpm": "Nadi (BPM)",
                    "primary_condition": "Dugaan Klinis Utama",
                    "urgency": "Urgensi",
                    "notes": "Catatan",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
