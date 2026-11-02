import cv2
import numpy as np


class Visualizer:
    """Menggambar overlay indikator medis klinis pada frame gambar/video."""

    # Palet warna BGR klinis
    COLOR_CYAN = (255, 200, 0)
    COLOR_GREEN = (80, 200, 80)
    COLOR_YELLOW = (0, 215, 255)
    COLOR_RED = (60, 60, 235)
    COLOR_WHITE = (245, 245, 245)
    COLOR_DARK = (25, 25, 25)

    @classmethod
    def get_status_color(cls, risk_level: str):
        if risk_level == "LOW":
            return cls.COLOR_GREEN
        elif risk_level == "MODERATE":
            return cls.COLOR_YELLOW
        else:
            return cls.COLOR_RED

    @classmethod
    def draw_health_overlay(
        cls,
        image_bgr: np.ndarray,
        landmarks_dict: dict,
        symmetry_res: dict = None,
        jaundice_res: dict = None,
        rppg_res: dict = None,
        fatigue_res: dict = None,
        pallor_res: dict = None,
    ) -> np.ndarray:
        """Menggambar visualisasi landmark, sumbu simetri, dan bounding box sklera."""
        out = image_bgr.copy()
        pixel_lms = landmarks_dict["pixel_landmarks"]

        # 1. Gambar Sumbu Simetri Wajah (Midline)
        if symmetry_res and "keypoints" in symmetry_res:
            kp = symmetry_res["keypoints"]
            pt_bridge = tuple(map(int, kp["midline_start"]))
            pt_chin = tuple(map(int, kp["midline_end"]))

            # Garis sumbu tengah wajah
            cv2.line(out, pt_bridge, pt_chin, cls.COLOR_CYAN, 2, cv2.LINE_AA)
            cv2.circle(out, pt_bridge, 4, cls.COLOR_CYAN, -1)
            cv2.circle(out, pt_chin, 4, cls.COLOR_CYAN, -1)

            # Titik sudut mulut kiri & kanan
            pt_mr = tuple(map(int, kp["mouth_r"]))
            pt_ml = tuple(map(int, kp["mouth_l"]))

            mouth_color = cls.get_status_color(symmetry_res.get("risk_level", "LOW"))
            cv2.line(out, pt_mr, pt_ml, mouth_color, 2, cv2.LINE_AA)
            cv2.circle(out, pt_mr, 5, mouth_color, -1)
            cv2.circle(out, pt_ml, 5, mouth_color, -1)

            # Titik mata kelopak
            for pt in [kp["eye_top_r"], kp["eye_bot_r"], kp["eye_top_l"], kp["eye_bot_l"]]:
                cv2.circle(out, tuple(map(int, pt)), 3, cls.COLOR_CYAN, -1)

        # 2. Highlight Kontur Sklera Mata
        if jaundice_res and "r_mask" in jaundice_res and jaundice_res["r_mask"] is not None:
            r_mask = jaundice_res["r_mask"]
            l_mask = jaundice_res.get("l_mask")
            combined_mask = r_mask
            if l_mask is not None:
                combined_mask = cv2.bitwise_or(r_mask, l_mask)

            # Berikan tint lembut pada sklera
            sclera_color = cls.get_status_color(jaundice_res.get("risk_level", "LOW"))
            overlay = out.copy()
            overlay[combined_mask == 255] = sclera_color
            cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

        # 3. Kotak ROI Dahi untuk rPPG
        if rppg_res:
            forehead_idx = [108, 151, 337, 9, 8]
            pts = pixel_lms[forehead_idx, :2]
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            w_roi, h_roi = int(out.shape[1] * 0.08), int(out.shape[0] * 0.05)
            x1, y1 = max(0, cx - w_roi // 2), max(0, cy - h_roi // 2)
            x2, y2 = min(out.shape[1], cx + w_roi // 2), min(out.shape[0], cy + h_roi // 2)

            rppg_color = cls.COLOR_GREEN if rppg_res.get("is_stable") else cls.COLOR_YELLOW
            cv2.rectangle(out, (x1, y1), (x2, y2), rppg_color, 1, cv2.LINE_AA)
            cv2.putText(out, "ROI rPPG", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, rppg_color, 1, cv2.LINE_AA)

        # 4. Highlight Kontur Bibir (Mukosa Vaskularisasi)
        if pallor_res:
            lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
            lip_pts = pixel_lms[lip_idx, :2].astype(np.int32).reshape((-1, 1, 2))
            lip_color = cls.get_status_color(pallor_res.get("risk_level", "LOW"))
            cv2.polylines(out, [lip_pts], isClosed=True, color=lip_color, thickness=1, lineType=cv2.LINE_AA)

        # 5. Siluet Panduan Oval Wajah Klinis (Smart Oval Framing Guide)
        h, w = out.shape[:2]
        center = (int(w / 2), int(h * 0.50))
        axes = (int(w * 0.28), int(h * 0.38))
        guide_overlay = out.copy()
        cv2.ellipse(guide_overlay, center, axes, 0, 0, 360, (200, 240, 220), 1, cv2.LINE_AA)
        ax_w, ax_h = axes[0], axes[1]
        cv2.line(guide_overlay, (center[0] - 12, center[1] - ax_h), (center[0] + 12, center[1] - ax_h), (34, 197, 94), 2, cv2.LINE_AA)
        cv2.line(guide_overlay, (center[0] - 12, center[1] + ax_h), (center[0] + 12, center[1] + ax_h), (34, 197, 94), 2, cv2.LINE_AA)
        cv2.line(guide_overlay, (center[0] - ax_w, center[1] - 12), (center[0] - ax_w, center[1] + 12), (34, 197, 94), 2, cv2.LINE_AA)
        cv2.line(guide_overlay, (center[0] + ax_w, center[1] - 12), (center[0] + ax_w, center[1] + 12), (34, 197, 94), 2, cv2.LINE_AA)
        cv2.addWeighted(guide_overlay, 0.4, out, 0.6, 0, out)

        return out

