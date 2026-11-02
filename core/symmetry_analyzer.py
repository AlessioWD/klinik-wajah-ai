import numpy as np


class SymmetryAnalyzer:
    """Menganalisis simetri wajah untuk skrining risiko Bell's Palsy dan Stroke (FAST criteria).

    Menggunakan normalisasi vektor sumbu tengah (midline projection) agar kebal terhadap kemiringan kepala (head-tilt invariant).
    """

    def __init__(self):
        # Indeks landmark MediaPipe
        self.MID_TOP = 10       # Dahi atas
        self.MID_BRIDGE = 168   # Pangkal hidung (glabella)
        self.MID_NOSE_TIP = 1   # Ujung hidung
        self.MID_CHIN = 152     # Ujung dagu bawah

        # Sudut bibir
        self.MOUTH_RIGHT = 61   # Sudut bibir kanan subjek
        self.MOUTH_LEFT = 291   # Sudut bibir kiri subjek

        # Mata (bukaan kelopak mata atas-bawah)
        self.R_EYE_TOP = 159
        self.R_EYE_BOTTOM = 145
        self.L_EYE_TOP = 386
        self.L_EYE_BOTTOM = 374

        # Sudut mata luar
        self.R_EYE_OUTER = 33
        self.L_EYE_OUTER = 263

        # Alis
        self.R_EYEBROW = 70
        self.L_EYEBROW = 300

        # Pipi / Lipatan Nasolabial
        self.R_CHEEK = 205
        self.L_CHEEK = 425

    def analyze(self, landmarks_dict: dict) -> dict:
        """Menerima hasil deteksi landmark dan mengembalikan metrik simetri komprehensif."""
        norm_lms = landmarks_dict["norm_landmarks"]
        pixel_lms = landmarks_dict["pixel_landmarks"]
        blendshapes = landmarks_dict.get("blendshapes", {})

        # 1. Tentukan sumbu vertikal wajah (Midline: dari bridge ke dagu)
        p_bridge = norm_lms[self.MID_BRIDGE, :2]
        p_chin = norm_lms[self.MID_CHIN, :2]

        midline_vec = p_chin - p_bridge
        midline_len = np.linalg.norm(midline_vec)
        if midline_len < 1e-6:
            midline_len = 1.0

        u_midline = midline_vec / midline_len  # Vektor satuan searah sumbu wajah
        n_midline = np.array([-u_midline[1], u_midline[0]])  # Vektor satuan tegak lurus (horizontal relatif wajah)

        # Skala referensi wajah (panjang wajah) untuk normalisasi jarak
        ref_scale = midline_len

        # Fungsi helper proyeksi koordinat relatif ke sumbu wajah
        def get_rel_pos(idx):
            pt = norm_lms[idx, :2] - p_bridge
            v_pos = np.dot(pt, u_midline) / ref_scale  # Posisi vertikal
            h_pos = np.dot(pt, n_midline) / ref_scale  # Posisi horizontal
            return v_pos, h_pos

        # 2. Analisis Sudut Bibir (Mouth Corners)
        v_mouth_r, h_mouth_r = get_rel_pos(self.MOUTH_RIGHT)
        v_mouth_l, h_mouth_l = get_rel_pos(self.MOUTH_LEFT)

        # Selisih vertikal sudut bibir (asimetri mulut jatuh / drooping)
        mouth_v_diff = abs(v_mouth_r - v_mouth_l)
        # Rasio horizontal jarak sudut bibir ke sumbu tengah
        mouth_h_sym = min(abs(h_mouth_r), abs(h_mouth_l)) / max(abs(h_mouth_r), abs(h_mouth_l) + 1e-6)

        # 3. Analisis Bukaan Kelopak Mata (Palpebral Fissure Asymmetry)
        r_eye_h = np.linalg.norm(norm_lms[self.R_EYE_TOP, :2] - norm_lms[self.R_EYE_BOTTOM, :2])
        l_eye_h = np.linalg.norm(norm_lms[self.L_EYE_TOP, :2] - norm_lms[self.L_EYE_BOTTOM, :2])
        eye_open_ratio = min(r_eye_h, l_eye_h) / max(r_eye_h, l_eye_h, 1e-6)

        # 4. Analisis Ketinggian Alis
        v_brow_r, _ = get_rel_pos(self.R_EYEBROW)
        v_brow_l, _ = get_rel_pos(self.L_EYEBROW)
        eyebrow_v_diff = abs(v_brow_r - v_brow_l)

        # 5. Analisis Pipi / Lipatan Pipi
        v_chk_r, h_chk_r = get_rel_pos(self.R_CHEEK)
        v_chk_l, h_chk_l = get_rel_pos(self.L_CHEEK)
        cheek_v_diff = abs(v_chk_r - v_chk_l)

        # 6. Analisis Blendshapes (Aktivitas Otot Wajah Kiri vs Kanan)
        smile_r = blendshapes.get("mouthSmileRight", 0.0)
        smile_l = blendshapes.get("mouthSmileLeft", 0.0)
        smile_diff = abs(smile_r - smile_l)

        # 7. Kalkulasi Skor Simetri Keseluruhan (0 - 100%)
        # Penalti dihitung dari deviasi relatif
        mouth_penalty = min(mouth_v_diff * 400.0, 40.0)
        eye_penalty = (1.0 - eye_open_ratio) * 35.0
        brow_penalty = min(eyebrow_v_diff * 200.0, 15.0)
        cheek_penalty = min(cheek_v_diff * 150.0, 10.0)

        total_penalty = mouth_penalty + eye_penalty + brow_penalty + cheek_penalty
        symmetry_score = max(0.0, min(100.0, 100.0 - total_penalty))

        # 8. Klasifikasi Status Klinis Triage
        if symmetry_score >= 85.0:
            status = "Normal (Simetris Sehat)"
            risk_level = "LOW"
            recommendation = "Simetri wajah dalam batas wajar. Tidak ditemukan tanda penurunan otot wajah bermakna."
        elif symmetry_score >= 70.0:
            status = "Asimetri Ringan"
            risk_level = "MODERATE"
            recommendation = "Terdapat sedikit asimetri ringan (bisa merupakan variasi anatomi alami, posisi tidur, atau kebiasaan mengunyah sebelah)."
        else:
            status = "Asimetri Signifikan (Perhatian Klinis)"
            risk_level = "HIGH"
            # Deteksi sisi mana yang mengalami penurunan
            droop_side = "Kanan" if v_mouth_r > v_mouth_l else "Kiri"
            recommendation = (
                f"Terdeteksi penurunan elevasi wajah pada sisi {droop_side}. "
                "Jika keluhan ini muncul tiba-tiba disertai kebas atau kesulitan berbicara, "
                "segera konsultasikan ke IGD/Dokter Saraf untuk skrining Stroke FAST atau Bell's Palsy."
            )

        return {
            "symmetry_score": round(symmetry_score, 1),
            "status": status,
            "risk_level": risk_level,
            "mouth_deviation_score": round(mouth_v_diff * 100, 2),
            "eye_opening_symmetry_ratio": round(eye_open_ratio * 100, 1),
            "eyebrow_deviation_score": round(eyebrow_v_diff * 100, 2),
            "recommendation": recommendation,
            "keypoints": {
                "midline_start": pixel_lms[self.MID_BRIDGE][:2],
                "midline_end": pixel_lms[self.MID_CHIN][:2],
                "mouth_r": pixel_lms[self.MOUTH_RIGHT][:2],
                "mouth_l": pixel_lms[self.MOUTH_LEFT][:2],
                "eye_top_r": pixel_lms[self.R_EYE_TOP][:2],
                "eye_bot_r": pixel_lms[self.R_EYE_BOTTOM][:2],
                "eye_top_l": pixel_lms[self.L_EYE_TOP][:2],
                "eye_bot_l": pixel_lms[self.L_EYE_BOTTOM][:2],
                "eyebrow_r": pixel_lms[self.R_EYEBROW][:2],
                "eyebrow_l": pixel_lms[self.L_EYEBROW][:2],
            },
        }
