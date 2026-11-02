import numpy as np


class FatigueDetector:
    """Mendeteksi kelelahan visual, kantuk kronis, dan risiko penurunan fokus menggunakan Eye Aspect Ratio (EAR) dan Mouth Aspect Ratio (MAR)."""

    def __init__(self):
        # Indeks mata kanan
        self.R_P1 = 33    # Outer canthus
        self.R_P4 = 133   # Inner canthus
        self.R_P2 = 160   # Top 1
        self.R_P6 = 144   # Bottom 1
        self.R_P3 = 158   # Top 2
        self.R_P5 = 153   # Bottom 2

        # Indeks mata kiri
        self.L_P1 = 263   # Outer canthus
        self.L_P4 = 362   # Inner canthus
        self.L_P2 = 385   # Top 1
        self.L_P6 = 380   # Bottom 1
        self.L_P3 = 387   # Top 2
        self.L_P5 = 373   # Bottom 2

        # Indeks mulut untuk deteksi menguap (Yawn / MAR)
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_CORNER_R = 61
        self.MOUTH_CORNER_L = 291

    def _calc_ear(self, norm_lms: np.ndarray, p1: int, p2: int, p3: int, p4: int, p5: int, p6: int) -> float:
        """Menghitung Eye Aspect Ratio (EAR) Soukupová & Čech."""
        pt1 = norm_lms[p1, :2]
        pt2 = norm_lms[p2, :2]
        pt3 = norm_lms[p3, :2]
        pt4 = norm_lms[p4, :2]
        pt5 = norm_lms[p5, :2]
        pt6 = norm_lms[p6, :2]

        # Jarak vertikal
        v1 = np.linalg.norm(pt2 - pt6)
        v2 = np.linalg.norm(pt3 - pt5)
        # Jarak horizontal
        h = np.linalg.norm(pt1 - pt4)

        if h < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def _calc_mar(self, norm_lms: np.ndarray) -> float:
        """Menghitung Mouth Aspect Ratio (MAR) untuk indikasi menguap."""
        top = norm_lms[self.MOUTH_TOP, :2]
        bot = norm_lms[self.MOUTH_BOTTOM, :2]
        cr = norm_lms[self.MOUTH_CORNER_R, :2]
        cl = norm_lms[self.MOUTH_CORNER_L, :2]

        v = np.linalg.norm(top - bot)
        h = np.linalg.norm(cr - cl)

        if h < 1e-6:
            return 0.0
        return float(v / h)

    def analyze(self, landmarks_dict: dict) -> dict:
        """Menganalisis tingkat kelelahan dan keterbukaan mata."""
        norm_lms = landmarks_dict["norm_landmarks"]

        ear_r = self._calc_ear(norm_lms, self.R_P1, self.R_P2, self.R_P3, self.R_P4, self.R_P5, self.R_P6)
        ear_l = self._calc_ear(norm_lms, self.L_P1, self.L_P2, self.L_P3, self.L_P4, self.L_P5, self.L_P6)
        ear_avg = (ear_r + ear_l) / 2.0

        mar = self._calc_mar(norm_lms)
        is_yawning = mar > 0.65

        # Normalisasi skor kelelahan (0: sangat segar, 100: sangat lelah/mengantuk)
        # EAR normal: ~0.28 - 0.35
        # EAR sayu / lelah: ~0.20 - 0.24
        # EAR tertutup: < 0.18
        if ear_avg >= 0.28:
            fatigue_score = max(0.0, (0.35 - ear_avg) * 100.0)
            status = "Segar & Fokus (Mata Terbuka Optimal)"
            risk_level = "LOW"
            recommendation = "Kondisi mata tampak prima dan responsif."
        elif ear_avg >= 0.21:
            fatigue_score = 40.0 + (0.28 - ear_avg) * 350.0
            status = "Mata Sayu / Kelelahan Ringan"
            risk_level = "MODERATE"
            recommendation = "Terdeteksi indikasi kelopak mata agak sayu (eye strain). Istirahatkan mata selama 5-10 menit dengan aturan 20-20-20."
        else:
            fatigue_score = min(100.0, 75.0 + (0.21 - ear_avg) * 300.0)
            status = "Sangat Lelah / Mengantuk Berat"
            risk_level = "HIGH"
            recommendation = "Kelopak mata cenderung menutup atau sangat berat. Sangat disarankan untuk tidak mengemudi atau mengoperasikan mesin berbahaya."

        if is_yawning:
            fatigue_score = min(100.0, fatigue_score + 25.0)
            status += " + Terdeteksi Menguap"

        return {
            "fatigue_score": round(fatigue_score, 1),
            "ear_avg": round(ear_avg, 3),
            "mar": round(mar, 3),
            "is_yawning": is_yawning,
            "status": status,
            "risk_level": risk_level,
            "recommendation": recommendation,
        }
