"""
renderer_3d.py: Lightweight 3D orthographic/projective viewer built with OpenCV.
Visualizes the SVD mirror plane, normal vector, and the 3D Digital Twin skeleton.
"""

from __future__ import annotations

import cv2
import numpy as np


class Aura3DViewer:
    """
    Renders an interactive 3D virtual inspector window using OpenCV.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

        self.yaw = 0.0
        self.pitch = 0.2

        self.scale = 250.0
        self.center_x = width // 2

        # Lower the drawing center so the feet (origin) rest near the bottom of the window
        self.center_y = int(height * 0.8)

    def _project_point(self, pt_3d: np.ndarray) -> tuple[int, int]:
        x, y, z = pt_3d[0], pt_3d[1], pt_3d[2]

        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        x_rot = x * cos_y - z * sin_y
        z_rot = x * sin_y + z * cos_y

        cos_p, sin_p = np.cos(self.pitch), np.sin(self.pitch)
        y_rot = y * cos_p - z_rot * sin_p

        px = int(self.center_x + x_rot * self.scale)
        py = int(self.center_y + y_rot * self.scale)

        return px, py

    def render_scene(
        self, landmarks_3d: np.ndarray | None, plane_normal: np.ndarray | None = None
    ) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        origin = np.array([0.0, 0.0, 0.0])
        ox, oy = self._project_point(origin)

        xx, xy = self._project_point(np.array([0.5, 0.0, 0.0]))
        yx, yy = self._project_point(np.array([0.0, 0.5, 0.0]))
        zx, zy = self._project_point(np.array([0.0, 0.0, 0.5]))

        cv2.line(canvas, (ox, oy), (xx, xy), (0, 0, 255), 2)
        cv2.line(canvas, (ox, oy), (yx, yy), (0, 255, 0), 2)
        cv2.line(canvas, (ox, oy), (zx, zy), (255, 0, 0), 2)

        if plane_normal is not None:
            nx, ny = self._project_point(plane_normal * 0.8)
            cv2.arrowedLine(canvas, (ox, oy), (nx, ny), (0, 255, 255), 3, tipLength=0.2)
            cv2.putText(
                canvas,
                "SVD Mirror Normal (n)",
                (nx + 10, ny),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # ---> NEW: Calculate and display the mirror angle <---
            plane_yaw = np.degrees(np.arctan2(plane_normal[0], -plane_normal[2]))
            cv2.putText(
                canvas,
                f"Mirror estimated at {plane_yaw:.1f} deg relative to camera",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if landmarks_3d is not None and len(landmarks_3d) > 28:
            connections = [
                (11, 12),
                (11, 13),
                (13, 15),
                (12, 14),
                (14, 16),
                (11, 23),
                (12, 24),
                (23, 24),
                (23, 25),
                (25, 27),
                (24, 26),
                (26, 28),
            ]

            # --- ORIGIN SHIFT TO FEET ---
            pivot_y = (landmarks_3d[27, 1] + landmarks_3d[28, 1]) / 2.0
            display_landmarks = landmarks_3d.copy()
            display_landmarks[:, 1] -= pivot_y

            projected_pts = [self._project_point(pt) for pt in display_landmarks]

            for start_idx, end_idx in connections:
                if start_idx < len(projected_pts) and end_idx < len(projected_pts):
                    pt_a = projected_pts[start_idx]
                    pt_b = projected_pts[end_idx]
                    cv2.line(canvas, pt_a, pt_b, (200, 200, 200), 2, cv2.LINE_AA)

            for pt in projected_pts:
                cv2.circle(canvas, pt, 4, (0, 165, 255), -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "AURA 3D Digital Twin Inspector",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Use A/D keys to rotate Yaw | W/S to rotate Pitch",
            (15, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def handle_keyboard(self, key: int) -> None:
        step = 0.05
        if key == ord("a") or key == ord("A"):
            self.yaw -= step
        elif key == ord("d") or key == ord("D"):
            self.yaw += step
        elif key == ord("w") or key == ord("W"):
            self.pitch += step
        elif key == ord("s") or key == ord("S"):
            self.pitch -= step
