import cv2
import numpy as np


class ColorCalibrator:
    """Melakukan normalisasi warna dan kompensasi pencahayaan ruangan menggunakan Gray-World Assumption.

    Berguna untuk menetralkan efek lampu bohlam kuning (warm white) agar analisis warna sklera mata dan kepucatan bibir menjadi objektif.
    """

    @staticmethod
    def apply_gray_world(image_bgr: np.ndarray) -> np.ndarray:
        """Menyeimbangkan warna gambar dengan asumsi rata-rata warna dunia nyata adalah netral abu-abu."""
        b, g, r = cv2.split(image_bgr.astype(np.float32))

        mean_b = np.mean(b) + 1e-6
        mean_g = np.mean(g) + 1e-6
        mean_r = np.mean(r) + 1e-6

        # Rata-rata intensitas abu-abu keseluruhan
        mean_gray = (mean_b + mean_g + mean_r) / 3.0

        # Skala tiap channel warna
        scale_b = mean_gray / mean_b
        scale_g = mean_gray / mean_g
        scale_r = mean_gray / mean_r

        # Batasi penguatan (gain) agar gambar tidak over-saturated (maks 1.5x)
        scale_b = np.clip(scale_b, 0.7, 1.5)
        scale_g = np.clip(scale_g, 0.7, 1.5)
        scale_r = np.clip(scale_r, 0.7, 1.5)

        b_balanced = np.clip(b * scale_b, 0, 255).astype(np.uint8)
        g_balanced = np.clip(g * scale_g, 0, 255).astype(np.uint8)
        r_balanced = np.clip(r * scale_r, 0, 255).astype(np.uint8)

        return cv2.merge([b_balanced, g_balanced, r_balanced])

    @staticmethod
    def estimate_ambient_illuminant(image_bgr: np.ndarray) -> dict:
        """Mengestimasi suhu warna lampu dan tingkat kecerahan ruangan (luminansi)."""
        b, g, r = cv2.split(image_bgr)
        mean_b = float(np.mean(b))
        mean_r = float(np.mean(r))

        ratio_rb = mean_r / (mean_b + 1e-6)

        if ratio_rb > 1.25:
            lighting_type = "Warm Light (Lampu Kuning / Sore Hari)"
            is_warm = True
        elif ratio_rb < 0.85:
            lighting_type = "Cool Light (Lampu Putih Kebiruan)"
            is_warm = False
        else:
            lighting_type = "Neutral Daylight (Cahaya Putih Seimbang)"
            is_warm = False

        # Hitung tingkat kecerahan rata-rata (Luminansi Grayscale 0 - 255)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        is_too_dark = brightness < 60.0
        is_too_bright = brightness > 220.0

        return {
            "lighting_type": lighting_type,
            "is_warm": is_warm,
            "rb_ratio": round(ratio_rb, 2),
            "brightness": round(brightness, 1),
            "is_too_dark": is_too_dark,
            "is_too_bright": is_too_bright,
        }

