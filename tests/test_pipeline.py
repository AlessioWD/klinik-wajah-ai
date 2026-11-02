import os
import sys
import numpy as np

# Tambahkan direktori root proyek ke sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.face_mesh_detector import FaceMeshDetector
from core.fatigue_detector import FatigueDetector
from core.jaundice_analyzer import JaundiceAnalyzer
from core.rppg_estimator import RPPGEstimator
from core.symmetry_analyzer import SymmetryAnalyzer
from utils.visualization import Visualizer


def test_detector_initialization():
    print("[1/5] Menguji inisialisasi FaceMeshDetector...")
    detector = FaceMeshDetector()
    assert detector is not None
    print("   -> FaceMeshDetector berhasil diinisialisasi!")


def test_symmetry_analyzer():
    print("[2/5] Menguji SymmetryAnalyzer...")
    analyzer = SymmetryAnalyzer()

    # Buat 478 landmark sintetis yang simetris sempurna
    n_landmarks = 478
    norm_lms = np.zeros((n_landmarks, 3), dtype=np.float32)

    # Titik tengah
    norm_lms[analyzer.MID_BRIDGE] = [0.5, 0.35, 0.0]
    norm_lms[analyzer.MID_CHIN] = [0.5, 0.85, 0.0]

    # Sudut bibir simetris
    norm_lms[analyzer.MOUTH_RIGHT] = [0.42, 0.65, 0.0]
    norm_lms[analyzer.MOUTH_LEFT] = [0.58, 0.65, 0.0]

    # Bukaan mata simetris
    norm_lms[analyzer.R_EYE_TOP] = [0.40, 0.38, 0.0]
    norm_lms[analyzer.R_EYE_BOTTOM] = [0.40, 0.42, 0.0]
    norm_lms[analyzer.L_EYE_TOP] = [0.60, 0.38, 0.0]
    norm_lms[analyzer.L_EYE_BOTTOM] = [0.60, 0.42, 0.0]

    # Alis & pipi
    norm_lms[analyzer.R_EYEBROW] = [0.38, 0.30, 0.0]
    norm_lms[analyzer.L_EYEBROW] = [0.62, 0.30, 0.0]
    norm_lms[analyzer.R_CHEEK] = [0.35, 0.50, 0.0]
    norm_lms[analyzer.L_CHEEK] = [0.65, 0.50, 0.0]

    pixel_lms = (norm_lms * np.array([640, 480, 1])).astype(np.int32)

    mock_dict = {
        "norm_landmarks": norm_lms,
        "pixel_landmarks": pixel_lms,
        "blendshapes": {"mouthSmileRight": 0.0, "mouthSmileLeft": 0.0},
    }

    res = analyzer.analyze(mock_dict)
    assert res["symmetry_score"] >= 95.0, f"Expected high score for symmetric face, got {res['symmetry_score']}"
    assert res["risk_level"] == "LOW"
    print(f"   -> Hasil Simetri Normal: Skor {res['symmetry_score']}%, Status: {res['status']}")

    # Uji asimetri (bibir kanan turun / drooping)
    norm_lms_droop = norm_lms.copy()
    norm_lms_droop[analyzer.MOUTH_RIGHT] = [0.42, 0.72, 0.0]  # Turun drastis
    mock_dict_droop = {
        "norm_landmarks": norm_lms_droop,
        "pixel_landmarks": (norm_lms_droop * np.array([640, 480, 1])).astype(np.int32),
        "blendshapes": {},
    }
    res_droop = analyzer.analyze(mock_dict_droop)
    assert res_droop["symmetry_score"] < 75.0
    print(f"   -> Hasil Asimetri Saraf: Skor {res_droop['symmetry_score']}%, Status: {res_droop['status']}")


def test_jaundice_analyzer():
    print("[3/5] Menguji JaundiceAnalyzer...")
    analyzer = JaundiceAnalyzer()

    # Buat citra buatan dengan sklera mata putih vs kuning
    img_normal = np.full((300, 300, 3), 200, dtype=np.uint8)
    # Area mata (putih bersih)
    img_normal[100:150, 80:220] = [240, 240, 240]

    # Mock landmark
    mock_pixel_lms = np.zeros((478, 3), dtype=np.int32)
    for idx in analyzer.R_EYE_CONTOUR:
        mock_pixel_lms[idx] = [100, 120, 0]
    for idx in analyzer.L_EYE_CONTOUR:
        mock_pixel_lms[idx] = [200, 120, 0]
    mock_pixel_lms[analyzer.R_IRIS_CENTER] = [100, 120, 0]
    mock_pixel_lms[analyzer.L_IRIS_CENTER] = [200, 120, 0]

    res_normal = analyzer.analyze(img_normal, {"pixel_landmarks": mock_pixel_lms})
    print(f"   -> Sklera Normal: Skor Ikterus {res_normal['jaundice_score']}/100 ({res_normal['status']})")


def test_fatigue_detector():
    print("[4/5] Menguji FatigueDetector (EAR)...")
    detector = FatigueDetector()

    norm_lms = np.zeros((478, 3), dtype=np.float32)
    # Mata terbuka normal: p1-p4 = 0.1, vertical = 0.03 -> EAR ~ 0.30
    norm_lms[detector.R_P1] = [0.35, 0.40, 0.0]
    norm_lms[detector.R_P4] = [0.45, 0.40, 0.0]
    norm_lms[detector.R_P2] = [0.38, 0.385, 0.0]
    norm_lms[detector.R_P6] = [0.38, 0.415, 0.0]
    norm_lms[detector.R_P3] = [0.42, 0.385, 0.0]
    norm_lms[detector.R_P5] = [0.42, 0.415, 0.0]

    norm_lms[detector.L_P1] = [0.55, 0.40, 0.0]
    norm_lms[detector.L_P4] = [0.65, 0.40, 0.0]
    norm_lms[detector.L_P2] = [0.58, 0.385, 0.0]
    norm_lms[detector.L_P6] = [0.58, 0.415, 0.0]
    norm_lms[detector.L_P3] = [0.62, 0.385, 0.0]
    norm_lms[detector.L_P5] = [0.62, 0.415, 0.0]

    res = detector.analyze({"norm_landmarks": norm_lms})
    assert res["ear_avg"] > 0.25
    print(f"   -> Kondisi Mata Terbuka: EAR {res['ear_avg']}, Status: {res['status']}")


def test_rppg_estimator():
    print("[5/5] Menguji RPPGEstimator dengan sinyal pulsasi sintetis...")
    rppg = RPPGEstimator(buffer_size=120)

    # Simulasikan 90 frame pada 30 fps (durasi 3 detik) dengan denyut 72 BPM (1.2 Hz)
    fps = 30.0
    freq_heart = 1.2  # 72 BPM
    mock_lms = np.zeros((478, 3), dtype=np.int32)
    for idx in rppg.FOREHEAD_CORE:
        mock_lms[idx] = [100, 100, 0]

    mock_dict = {"pixel_landmarks": mock_lms}

    import time
    base_time = time.time()
    res = None
    for i in range(120):
        # Frame hijau dengan modulasi pulsa sinus 72 BPM
        intensity = 150.0 + 5.0 * np.sin(2 * np.pi * freq_heart * (i / fps))
        frame = np.full((200, 200, 3), [100, int(intensity), 100], dtype=np.uint8)
        sim_time = base_time + (i / fps)

        res = rppg.update(frame, mock_dict, timestamp=sim_time)
    print(f"   -> Estimasi BPM rPPG: {res.get('bpm')} BPM (Target: 72 BPM), Status: {res.get('status')}")
    assert res.get("bpm") is not None
    assert abs(res["bpm"] - 72.0) <= 6.0, f"Expected ~72 BPM, got {res['bpm']}"
    print("   -> Algoritma rPPG berhasil mendeteksi frekuensi denyut dengan presisi tinggi!")


if __name__ == "__main__":
    print("=== MEMULAI PENGUJIAN PIPELINE SKRINING KESEHATAN WAJAH ===")
    test_detector_initialization()
    test_symmetry_analyzer()
    test_jaundice_analyzer()
    test_fatigue_detector()
    test_rppg_estimator()
    print("\n[OK] SEMUA PENGUJIAN PIPELINE BERHASIL 100%!")
