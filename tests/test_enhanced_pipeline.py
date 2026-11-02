import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.color_calibrator import ColorCalibrator
from core.pallor_analyzer import PallorAnalyzer
from core.clinical_triangulation import ClinicalTriangulationEngine
from core.history_db import HistoryDatabase


def test_color_calibrator():
    print("[1/4] Menguji ColorCalibrator (Gray-World)...")
    # Gambar dengan lampu kuning kuat (R tinggi, B rendah)
    warm_img = np.full((100, 100, 3), [80, 180, 240], dtype=np.uint8)
    info = ColorCalibrator.estimate_ambient_illuminant(warm_img)
    assert info["is_warm"] is True
    print(f"   -> Estimasi Pencahayaan: {info['lighting_type']} (R/B: {info['rb_ratio']})")

    balanced_img = ColorCalibrator.apply_gray_world(warm_img)
    info_balanced = ColorCalibrator.estimate_ambient_illuminant(balanced_img)
    print(f"   -> Setelah Gray-World: R/B ratio turun dari {info['rb_ratio']} menjadi {info_balanced['rb_ratio']}")
    assert info_balanced["rb_ratio"] < info["rb_ratio"]
    print("   -> ColorCalibrator sukses menetralkan lampu kuning!")


def test_pallor_analyzer():
    print("[2/4] Menguji PallorAnalyzer (Anemia Screening)...")
    analyzer = PallorAnalyzer()

    # Buat landmark tiruan dengan poligon bibir seluas 20x20 piksel
    mock_pixel_lms = np.zeros((478, 3), dtype=np.int32)
    poly_box = [(40, 40), (60, 40), (60, 60), (40, 60)]
    for i, idx in enumerate(analyzer.LIP_CONTOUR):
        mock_pixel_lms[idx] = [poly_box[i % 4][0], poly_box[i % 4][1], 0]

    # Gambar bibir merah segar (Normal)
    img_healthy = np.zeros((100, 100, 3), dtype=np.uint8)
    img_healthy[40:60, 40:60] = [80, 90, 210]  # Merah tinggi
    res_healthy = analyzer.analyze(img_healthy, {"pixel_landmarks": mock_pixel_lms})
    print(f"   -> Bibir Normal: Skor Kepucatan {res_healthy['pallor_score']}, Status: {res_healthy['status']}")
    assert res_healthy["risk_level"] == "LOW"

    # Gambar bibir pucat pasi (Anemia)
    img_pale = np.zeros((100, 100, 3), dtype=np.uint8)
    img_pale[40:60, 40:60] = [170, 180, 190]  # Abu-abu / pucat
    res_pale = analyzer.analyze(img_pale, {"pixel_landmarks": mock_pixel_lms})
    print(f"   -> Bibir Pucat: Skor Kepucatan {res_pale['pallor_score']}, Status: {res_pale['status']}")
    assert res_pale["risk_level"] in ["MODERATE", "HIGH"]
    print("   -> PallorAnalyzer sukses mendeteksi tingkat kepucatan!")


def test_clinical_triangulation():
    print("[3/4] Menguji ClinicalTriangulationEngine...")
    engine = ClinicalTriangulationEngine()

    # Skenario 1: Asimetri + Kebas Lengan (Kondisi Stroke)
    findings_stroke = {
        "symmetry": {"symmetry_score": 62.0, "risk_level": "HIGH"},
        "jaundice": {"jaundice_score": 10.0},
        "pallor": {"pallor_score": 15.0},
    }
    answers_stroke = {
        "asym_sudden": True,
        "asym_limb_weakness": True,
    }
    eval_stroke = engine.evaluate(findings_stroke, answers_stroke)
    prim_stroke = eval_stroke["primary_diagnosis"]
    print(f"   -> Diagnosa Banding 1: {prim_stroke['condition']} (Keyakinan: {prim_stroke['confidence']}%)")
    assert prim_stroke["urgency"] == "EMERGENCY"
    assert "Stroke" in prim_stroke["condition"]

    # Skenario 2: Sklera Kuning + Urine Gelap seperti Teh (Hepatitis / Hati)
    findings_jaundice = {
        "symmetry": {"symmetry_score": 96.0},
        "jaundice": {"jaundice_score": 75.0, "risk_level": "HIGH"},
        "pallor": {"pallor_score": 20.0},
    }
    answers_jaundice = {
        "jaundice_dark_urine": True,
        "jaundice_ruq_pain": True,
    }
    eval_jaundice = engine.evaluate(findings_jaundice, answers_jaundice)
    prim_jaundice = eval_jaundice["primary_diagnosis"]
    print(f"   -> Diagnosa Banding 2: {prim_jaundice['condition']} (Keyakinan: {prim_jaundice['confidence']}%)")
    assert "Hepato-Bilier" in prim_jaundice["condition"]
    assert "SGOT" in prim_jaundice["tests"][1]
    print("   -> ClinicalTriangulationEngine sukses mengkorelasikan gejala!")


def test_history_database():
    print("[4/4] Menguji HistoryDatabase (SQLite & Multi-User)...")
    test_db_path = os.path.join(PROJECT_ROOT, "data", "test_screening.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = HistoryDatabase(db_path=test_db_path)
    patients = db.get_patients()
    assert len(patients) >= 2
    print(f"   -> Profil Pengguna Bawaan: {patients}")

    # Tambah profil baru
    db.add_patient("Pasien Uji Coba")
    assert "Pasien Uji Coba" in db.get_patients()

    # Simpan catatan skrining
    db.save_screening(
        patient_name="Pasien Uji Coba",
        symmetry_score=92.5,
        jaundice_score=15.0,
        pallor_score=18.0,
        fatigue_score=12.0,
        bpm=74.0,
        primary_condition="Profil Sehat",
        urgency="LOW",
        notes="Skrining rutin pagi hari",
    )

    history = db.get_patient_history("Pasien Uji Coba")
    assert len(history) == 1
    assert history.iloc[0]["bpm"] == 74.0
    print(f"   -> Berhasil menyimpan dan membaca riwayat database SQLite ({len(history)} data)")

    # Bersihkan file uji
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("   -> HistoryDatabase sukses terverifikasi!")


if __name__ == "__main__":
    print("=== PENGUJIAN PENINGKATAN SISTEM (TRIANGULASI, KALIBRASI, ANEMIA, DATABASE) ===")
    test_color_calibrator()
    test_pallor_analyzer()
    test_clinical_triangulation()
    test_history_database()
    print("\n[OK] SEMUA MODUL PENINGKATAN LULUS 100%!")
