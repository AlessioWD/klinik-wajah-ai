import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")


class FaceMeshDetector:
    """Wrapper untuk MediaPipe FaceLandmarker Tasks API (MediaPipe 1.2+)."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Mengunduh model Face Landmarker ke {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model berhasil diunduh!")

    def process_image(self, image_bgr: np.ndarray):
        """Memproses gambar BGR dan mengembalikan landmarks dalam koordinat piksel dan normalisasi."""
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        # Ambil wajah pertama
        raw_landmarks = result.face_landmarks[0]
        pixel_landmarks = []
        norm_landmarks = []

        for lm in raw_landmarks:
            norm_landmarks.append((lm.x, lm.y, lm.z))
            pixel_landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))

        # Ekstraksi blendshapes jika ada
        blendshapes = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score

        return {
            "norm_landmarks": np.array(norm_landmarks),
            "pixel_landmarks": np.array(pixel_landmarks),
            "blendshapes": blendshapes,
            "image_shape": (h, w),
            "raw_result": result,
        }
