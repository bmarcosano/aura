"""Lightweight 3D skeleton viewer (visualization only, not a measurement engine)."""

from __future__ import annotations

import typing

import cv2
import numpy as np


class Aura3DViewer:
    """Project metric/relative 3D landmarks onto an orbit-controlled 2D canvas."""

    CONNECTIONS: typing.ClassVar[list[tuple[int, int]]] = [
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
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
    ]
    COLOR_MAP: typing.ClassVar[dict[str, tuple[int, int, int]]] = {
        "default": (245, 245, 245),
        "good": (0, 220, 0),
        "warning": (0, 210, 255),
        "bad": (0, 0, 255),
    }

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width, self.height = int(width), int(height)
        self.azimuth, self.elevation, self.zoom = 45.0, 18.0, 220.0

    def handle_keyboard(self, key: int) -> None:
        """Change viewer orbit without using AURA's letter shortcuts."""
        if key in (2424832, 65361, ord("4")):
            self.azimuth -= 5.0
        elif key in (2555904, 65363, ord("6")):
            self.azimuth += 5.0
        elif key in (2490368, 65362, ord("8")):
            self.elevation = min(85.0, self.elevation + 5.0)
        elif key in (2621440, 65364, ord("2")):
            self.elevation = max(-85.0, self.elevation - 5.0)

    def _project_point(self, pt_3d: np.ndarray) -> tuple[int, int]:
        """Apply an orthographic display projection; do not use for reconstruction."""
        point = np.asarray(pt_3d, dtype=float).reshape(3)
        if not np.isfinite(point).all():
            return -10000, -10000
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        R_y = np.array(
            [[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]]
        )
        R_x = np.array(
            [[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]]
        )
        rotated = R_x @ R_y @ point
        return int(rotated[0] * self.zoom + self.width / 2), int(
            -rotated[1] * self.zoom + self.height - 100
        )

    def render_scene(
        self,
        landmarks_3d: np.ndarray | None,
        plane_normal: np.ndarray | None = None,
        joint_colors: dict[int, str] | None = None,
        plane_d: float | None = None,
        reconstruction_mode: str = "MONOCULAR_RELATIVE",
    ) -> np.ndarray:
        """Render a skeleton and optional mirror plane in the same input frame."""
        canvas = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        joint_colors = joint_colors or {}
        for i in range(-4, 5):
            step = 0.3
            cv2.line(
                canvas,
                self._project_point([-1.2, 0, i * step]),
                self._project_point([1.2, 0, i * step]),
                (45, 45, 45),
                1,
            )
            cv2.line(
                canvas,
                self._project_point([i * step, 0, -1.2]),
                self._project_point([i * step, 0, 1.2]),
                (45, 45, 45),
                1,
            )
        if plane_normal is not None and plane_d is not None:
            normal = np.asarray(plane_normal, dtype=float).reshape(3)
            norm = np.linalg.norm(normal)
            if np.isfinite(normal).all() and norm > 1e-8:
                normal /= norm
                point = -float(plane_d) * normal
                end = point + 0.6 * normal
                cv2.arrowedLine(
                    canvas,
                    self._project_point(point),
                    self._project_point(end),
                    (255, 150, 30),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.2,
                )
        points = None if landmarks_3d is None else np.asarray(landmarks_3d, dtype=float)
        if points is not None and points.shape == (33, 3):
            projected = [self._project_point(p) for p in points]
            valid = np.isfinite(points).all(axis=1)
            for a, b in self.CONNECTIONS:
                if (
                    valid[a]
                    and valid[b]
                    and projected[a][0] > -9000
                    and projected[b][0] > -9000
                ):
                    cv2.line(
                        canvas,
                        projected[a],
                        projected[b],
                        (170, 170, 170),
                        2,
                        cv2.LINE_AA,
                    )
            for index, point in enumerate(projected):
                if valid[index] and point[0] > -9000:
                    color = self.COLOR_MAP.get(
                        joint_colors.get(index, "default"), self.COLOR_MAP["default"]
                    )
                    cv2.circle(
                        canvas,
                        point,
                        5 if index in (0, 11, 12, 23, 24) else 3,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
        cv2.putText(
            canvas,
            f"MODE: {reconstruction_mode}",
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Orbit: arrows / numpad 4-6-8-2",
            (14, self.height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        return canvas
