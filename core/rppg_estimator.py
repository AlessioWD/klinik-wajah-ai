import time
from collections import deque
import cv2
import numpy as np
from scipy.signal import butter, filtfilt


class RPPGEstimator:
    """Mengestimasi denyut nadi (BPM) secara nirsentuh menggunakan Remote Photoplethysmography (rPPG).

    Membaca fluktuasi mikroskopis penyerapan cahaya spektrum hijau kapiler kulit wajah (dahi & pipi).
    """

    def __init__(self, buffer_size: int = 180):
        self.buffer_size = buffer_size
        self.timestamps = deque(maxlen=buffer_size)
        self.raw_signals = deque(maxlen=buffer_size)
        self.last_bpm = 72.0  # Nilai acuan default
        self.is_stable = False

        # Indeks dahi & pipi MediaPipe
        self.FOREHEAD_LANDMARKS = [10, 67, 109, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
        # Titik dahi utama: 10 (atas), 108, 151, 337, 338
        self.FOREHEAD_CORE = [108, 151, 337, 9, 8]
        # Pipi kanan & kiri
        self.R_CHEEK_CORE = [117, 118, 101, 205, 50]
        self.L_CHEEK_CORE = [346, 347, 330, 425, 280]

    def reset(self):
        """Mereset buffer sinyal."""
        self.timestamps.clear()
        self.raw_signals.clear()
        self.is_stable = False

    def _extract_roi_mean(self, image_bgr: np.ndarray, pixel_lms: np.ndarray) -> float:
        """Mengekstrak rata-rata intensitas kanal hijau pada ROI dahi dan pipi."""
        h, w, _ = image_bgr.shape

        # Ambil koordinat dahi
        forehead_pts = pixel_lms[self.FOREHEAD_CORE, :2]
        cx, cy = int(np.mean(forehead_pts[:, 0])), int(np.mean(forehead_pts[:, 1]))

        # Ukuran box dahi (sekitar 30x20 piksel)
        box_w, box_h = max(10, int(w * 0.08)), max(8, int(h * 0.05))
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w, cx + box_w // 2)
        y2 = min(h, cy + box_h // 2)

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        # Rata-rata nilai Green channel (paling sensitif terhadap penyerapan hemoglobin)
        mean_green = np.mean(roi[:, :, 1])
        return float(mean_green)

    def update(self, image_bgr: np.ndarray, landmarks_dict: dict, timestamp: float = None) -> dict:
        """Memperbarui buffer sinyal dengan frame terbaru dan menghitung estimasi BPM."""
        pixel_lms = landmarks_dict["pixel_landmarks"]
        now = time.time() if timestamp is None else timestamp

        val = self._extract_roi_mean(image_bgr, pixel_lms)
        if val > 0:
            self.timestamps.append(now)
            self.raw_signals.append(val)

        n_samples = len(self.raw_signals)
        min_required = 60  # Minimal 2-3 detik rekaman

        if n_samples < min_required:
            progress_pct = int((n_samples / min_required) * 100)
            return {
                "bpm": None,
                "status": f"Mengumpulkan sinyal kapiler wajah ({progress_pct}%)...",
                "is_stable": False,
                "ppg_wave": [],
                "confidence": 0.0,
            }

        # Hitung sampling rate aktual (fps)
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return {"bpm": None, "status": "Menunggu sinyal...", "is_stable": False, "ppg_wave": [], "confidence": 0.0}

        fps = (n_samples - 1) / duration
        if fps < 10.0:
            return {"bpm": None, "status": "FPS kamera terlalu rendah (< 10 fps)", "is_stable": False, "ppg_wave": [], "confidence": 0.0}

        # Normalisasi sinyal: kurangi moving average (detrending)
        sig = np.array(self.raw_signals)
        sig_normalized = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

        # Bandpass filter Butterworth (0.75 Hz - 2.5 Hz = 45 - 150 BPM)
        nyq = 0.5 * fps
        low = 0.75 / nyq
        high = 2.5 / nyq

        # Pastikan cutoff valid
        if low >= 1.0 or high >= 1.0 or low <= 0 or high <= low:
            return {"bpm": None, "status": "Kalibrasi frekuensi...", "is_stable": False, "ppg_wave": [], "confidence": 0.0}

        try:
            b, a = butter(2, [low, high], btype="band")
            filtered = filtfilt(b, a, sig_normalized)
        except Exception:
            filtered = sig_normalized

        # Analisis Spektrum Daya via FFT
        fft_len = 512
        fft_vals = np.abs(np.fft.rfft(filtered, n=fft_len))
        freqs = np.fft.rfftfreq(fft_len, d=1.0 / fps)

        # Ambil indeks frekuensi di rentang jantung (0.75 Hz - 2.5 Hz)
        valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]

        if len(valid_idx) == 0:
            return {"bpm": None, "status": "Mencari puncak pulsa...", "is_stable": False, "ppg_wave": [], "confidence": 0.0}

        valid_fft = fft_vals[valid_idx]
        peak_idx = valid_idx[np.argmax(valid_fft)]
        peak_freq = freqs[peak_idx]

        bpm_candidate = peak_freq * 60.0

        # Ukur rasio puncak terhadap noise (Signal Quality Index)
        peak_power = np.max(valid_fft)
        mean_power = np.mean(valid_fft) + 1e-6
        snr = peak_power / mean_power

        # Smoothing BPM agar stabil (Exponential Moving Average)
        if 48.0 <= bpm_candidate <= 140.0 and snr > 1.8:
            alpha = 0.25
            self.last_bpm = (1.0 - alpha) * self.last_bpm + alpha * bpm_candidate
            self.is_stable = True
            confidence = min(100.0, float(snr * 25.0))
            status = "Stabil (Terdeteksi)"
        else:
            self.is_stable = False
            confidence = max(20.0, float(snr * 15.0))
            status = "Sinyal Lemah / Subjek Bergerak"

        # Kirim 60 sampel terakhir gelombang PPG untuk visualisasi
        wave_snippet = filtered[-60:].tolist() if len(filtered) >= 60 else filtered.tolist()

        return {
            "bpm": round(self.last_bpm, 1),
            "status": status,
            "is_stable": self.is_stable,
            "ppg_wave": wave_snippet,
            "confidence": round(confidence, 1),
        }
