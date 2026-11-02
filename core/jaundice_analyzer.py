import cv2
import numpy as np
from core.color_calibrator import ColorCalibrator


class JaundiceAnalyzer:
    """Menganalisis warna sklera mata (bagian putih mata) untuk skrining dini Ikterus (Jaundice).

    Dilengkapi dengan normalisasi pencahayaan otomatis (Gray-World) untuk menetralkan lampu kuning ruangan.
    """

    def __init__(self):
        # Landmark mata MediaPipe
        self.R_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.L_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Landmark Iris (jika tersedia dari Face Mesh dengan iris / titik 468+)
        self.R_IRIS_CENTER = 468
        self.L_IRIS_CENTER = 473

    def _extract_sclera_pixels(self, image_bgr: np.ndarray, contour_pts: np.ndarray, iris_center_pt: np.ndarray):
        """Membuat mask sklera dengan mengecualikan iris, pupil, dan bayangan bulu mata."""
        h, w, _ = image_bgr.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Gambar poligon kontur mata penuh
        pts = contour_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # 2. Hitung radius estimasi iris (sekitar 25% dari lebar kontur mata)
        eye_rect = cv2.boundingRect(pts)
        iris_radius = max(4, int(eye_rect[2] * 0.22))

        # Kecualikan iris/pupil dari mask sklera
        cx, cy = int(iris_center_pt[0]), int(iris_center_pt[1])
        cv2.circle(mask, (cx, cy), iris_radius, 0, -1)

        # 3. Filter piksel gelap (bulu mata, bayangan kelopak)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        brightness_thresh = max(60, int(np.mean(gray[mask == 255]) * 0.75)) if np.any(mask == 255) else 60
        valid_mask = (mask == 255) & (gray > brightness_thresh)

        if not np.any(valid_mask):
            return None, mask

        return image_bgr[valid_mask], mask

    def analyze(self, image_bgr: np.ndarray, landmarks_dict: dict, apply_calibration: bool = True) -> dict:
        """Menganalisis warna sklera kedua mata dan menghasilkan skor risiko Ikterus."""
        illuminant_info = ColorCalibrator.estimate_ambient_illuminant(image_bgr)
        working_image = ColorCalibrator.apply_gray_world(image_bgr) if apply_calibration else image_bgr

        pixel_lms = landmarks_dict["pixel_landmarks"]
        h, w, _ = image_bgr.shape

        # Ambil koordinat kontur dan iris
        r_pts = pixel_lms[self.R_EYE_CONTOUR, :2]
        l_pts = pixel_lms[self.L_EYE_CONTOUR, :2]

        r_iris = pixel_lms[self.R_IRIS_CENTER, :2] if len(pixel_lms) > self.R_IRIS_CENTER else np.mean(r_pts, axis=0)
        l_iris = pixel_lms[self.L_IRIS_CENTER, :2] if len(pixel_lms) > self.L_IRIS_CENTER else np.mean(l_pts, axis=0)

        # Ekstraksi piksel sklera mata kanan & kiri dari citra terkalibrasi
        r_pixels, r_mask = self._extract_sclera_pixels(working_image, r_pts, r_iris)
        l_pixels, l_mask = self._extract_sclera_pixels(working_image, l_pts, l_iris)

        all_pixels = []
        if r_pixels is not None and len(r_pixels) > 20:
            all_pixels.append(r_pixels)
        if l_pixels is not None and len(l_pixels) > 20:
            all_pixels.append(l_pixels)

        if not all_pixels:
            return {
                "jaundice_score": 0.0,
                "status": "Mata Kurang Terlihat Jelas",
                "risk_level": "UNKNOWN",
                "yellowness_index": 0.0,
                "recommendation": "Pastikan mata terbuka wajar dan menghadap kamera dengan pencahayaan memadai.",
                "r_mask": r_mask,
                "l_mask": l_mask,
            }

        sclera_bgr = np.vstack(all_pixels)
        sclera_bgr_img = sclera_bgr.reshape(-1, 1, 3).astype(np.uint8)

        # 1. Analisis Ruang Warna CIELAB
        # Dalam OpenCV: L (0-255), a (0-255, center=128), b (0-255, center=128)
        # b* > 128 mengindikasikan komponen warna KUNING (+b*)
        lab = cv2.cvtColor(sclera_bgr_img, cv2.COLOR_BGR2LAB)
        mean_b_channel = np.mean(lab[:, :, 2])
        delta_b = mean_b_channel - 128.0  # Kelebihan warna kuning terhadap neutral white

        # 2. Analisis Ruang Warna HSV
        # Hue kuning: ~20 - 40 di OpenCV (skala 0-179)
        hsv = cv2.cvtColor(sclera_bgr_img, cv2.COLOR_BGR2HSV)
        hues = hsv[:, :, 0].flatten()
        sats = hsv[:, :, 1].flatten() / 255.0

        # Proporsi piksel sklera yang berada di spektrum kuning bersaturasi
        yellow_condition = (hues >= 18) & (hues <= 42) & (sats >= 0.18)
        yellow_pixel_ratio = np.sum(yellow_condition) / len(hues)

        # 3. Perhitungan Indeks Ikterus Gabungan (Jaundice Score 0 - 100)
        # Normal healthy: delta_b <= 5, yellow_pixel_ratio <= 0.05
        # Mild jaundice: delta_b 8 - 15, yellow_pixel_ratio 0.10 - 0.25
        # High jaundice: delta_b > 18, yellow_pixel_ratio > 0.35
        b_score = max(0.0, (delta_b - 3.0) * 4.5)
        ratio_score = yellow_pixel_ratio * 120.0

        jaundice_score = max(0.0, min(100.0, (b_score * 0.6) + (ratio_score * 0.4)))

        # 4. Status Klinis
        if jaundice_score < 30.0:
            status = "Normal (Sklera Jernih / Putih)"
            risk_level = "LOW"
            recommendation = "Warna sklera mata berada dalam batas normal. Tidak tampak tanda peningkatan pigmen kuning."
        elif jaundice_score < 60.0:
            status = "Kekuningan Ringan (Borderline)"
            risk_level = "MODERATE"
            recommendation = (
                "Terdeteksi semburat kekuningan ringan pada sklera. Hal ini sering disebabkan oleh pencahayaan lampu "
                "ruangan berwarna hangat (warm light), kelelahan mata, atau dehidrasi ringan. Jika mata terasa gatal/panas "
                "atau kulit ikut tampak menguning, pantau perkembangan secara berkala."
            )
        else:
            status = "Indikasi Sklera Ikterik (Kuning Signifikan)"
            risk_level = "HIGH"
            recommendation = (
                "Terdeteksi indeks warna kuning yang tinggi pada sklera mata. Ini merupakan tanda fisik khas "
                "ikterus (jaundice). Disarankan untuk berkonsultasi dengan dokter umum atau spesialis penyakit dalam "
                "guna pemeriksaan laboratorium darah (tes fungsi hati: Total Bilirubin, SGOT, SGPT)."
            )

        return {
            "jaundice_score": round(jaundice_score, 1),
            "status": status,
            "risk_level": risk_level,
            "delta_b_lab": round(delta_b, 2),
            "yellow_pixel_percentage": round(yellow_pixel_ratio * 100, 1),
            "recommendation": recommendation,
            "illuminant_info": illuminant_info,
            "r_mask": r_mask,
            "l_mask": l_mask,
        }
