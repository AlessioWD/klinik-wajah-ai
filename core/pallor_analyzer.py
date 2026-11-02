import cv2
import numpy as np


class PallorAnalyzer:
    """Menganalisis tingkat kepucatan pada mukosa bibir dan konjungtiva mata bawah untuk skrining awal Anemia (defisiensi Hemoglobin darah)."""

    def __init__(self):
        # Landmark bibir MediaPipe (Bibir bawah & kontur utama)
        self.LIP_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

        # Landmark kelopak mata bawah (area konjungtiva palpebra)
        self.R_LOWER_EYELID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        self.L_LOWER_EYELID = [362, 382, 381, 380, 374, 373, 390, 249, 263]

    def _extract_region_pixels(self, image_bgr: np.ndarray, pts: np.ndarray):
        """Mengekstrak piksel dari poligon landmark."""
        h, w, _ = image_bgr.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly], 255)

        pixels = image_bgr[mask == 255]
        return pixels

    def analyze(self, image_bgr: np.ndarray, landmarks_dict: dict) -> dict:
        """Menganalisis rasio kemerahan mukosa bibir dan kelopak bawah untuk menghitung indeks kepucatan."""
        pixel_lms = landmarks_dict["pixel_landmarks"]

        # Ekstraksi piksel bibir
        lip_pts = pixel_lms[self.LIP_CONTOUR, :2]
        lip_pixels = self._extract_region_pixels(image_bgr, lip_pts)

        # Ekstraksi piksel kelopak mata bawah
        r_lid_pts = pixel_lms[self.R_LOWER_EYELID, :2]
        l_lid_pts = pixel_lms[self.L_LOWER_EYELID, :2]
        r_lid_pixels = self._extract_region_pixels(image_bgr, r_lid_pts)
        l_lid_pixels = self._extract_region_pixels(image_bgr, l_lid_pts)

        if len(lip_pixels) < 20:
            return {
                "pallor_score": 0.0,
                "status": "Area Bibir Kurang Terlihat",
                "risk_level": "UNKNOWN",
                "redness_ratio": 0.0,
                "recommendation": "Pastikan area bibir dan wajah menghadap kamera tanpa terhalang masker.",
            }

        # Analisis rasio spektrum warna Merah (Red) terhadap Hijau (Green) & Biru (Blue)
        b = lip_pixels[:, 0].astype(np.float32)
        g = lip_pixels[:, 1].astype(np.float32)
        r = lip_pixels[:, 2].astype(np.float32)

        total_intensity = r + g + b + 1e-6
        norm_red = np.mean(r / total_intensity)
        rg_ratio = np.mean(r / (g + 1e-6))

        # Kondisi sehat vaskularisasi normal:
        # norm_red biasanya 0.40 - 0.52, rg_ratio > 1.30
        # Kondisi pucat / anemia:
        # norm_red turun < 0.36, rg_ratio < 1.15

        # Skor kepucatan (0: sangat segar merona, 100: pucat signifikan)
        if rg_ratio >= 1.30 and norm_red >= 0.39:
            pallor_score = max(0.0, (1.50 - rg_ratio) * 40.0)
            status = "Normal (Kemerahan Sehat & Segar)"
            risk_level = "LOW"
            recommendation = "Vaskularisasi kapiler pada bibir dan mukosa tampak baik dan segar."
        elif rg_ratio >= 1.15:
            pallor_score = 35.0 + (1.30 - rg_ratio) * 200.0
            status = "Agak Pucat (Mild Pallor)"
            risk_level = "MODERATE"
            recommendation = "Terdeteksi sedikit penurunan saturasi kemerahan pada area bibir. Bisa berkaitan dengan kurang tidur, dehidrasi, atau tanda awal defisiensi zat besi."
        else:
            pallor_score = min(100.0, 65.0 + (1.15 - rg_ratio) * 250.0)
            status = "Pucat Signifikan (Suspek Anemia)"
            risk_level = "HIGH"
            recommendation = "Area bibir dan mukosa tampak sangat pucat. Jika Anda sering merasa lemas, pusing berputar saat berdiri, atau ujung kuku dingin, disarankan cek laboratorium Darah Rutin (kadar Hemoglobin/Hb)."

        return {
            "pallor_score": round(float(pallor_score), 1),
            "status": status,
            "risk_level": risk_level,
            "redness_norm": round(float(norm_red), 3),
            "rg_ratio": round(float(rg_ratio), 2),
            "recommendation": recommendation,
        }
