"""
renderer_3d.py: Unified 3D Digital Twin & SVD Inspector rendering module for AURA.
Combines 3D skeletal rendering, SVD mirror plane normal visualization as a directed vector,
floor grid grounding, dynamic joint color-coding, and collision-free orbital controls.
"""

from __future__ import annotations

import cv2
import numpy as np


class Aura3DViewer:
    """
    Renders a unified 3D Digital Twin, floor grid, SVD mirror plane normal vector,
    and dynamic joint status colors using an orbital camera model.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height

        # Orbital camera state
        self.azimuth = 45.0  # Horizontal rotation (degrees)
        self.elevation = 20.0  # Vertical tilt (degrees)
        self.zoom = 220.0  # Pixels per meter scale

        # Base Color Map (BGR for OpenCV)
        self.COLOR_MAP = {
            "default": (255, 255, 255),  # White
            "good": (0, 255, 0),  # Green
            "warning": (0, 255, 255),  # Yellow
            "bad": (0, 0, 255),  # Red
        }

        # Anatomical Skeleton Connections (33 MediaPipe joints)
        self.CONNECTIONS = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Upper body
            (11, 23),
            (12, 24),
            (23, 24),  # Torso
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),  # Lower body
            (27, 29),
            (28, 30),
            (29, 31),
            (30, 32),  # Feet
        ]

    def handle_keyboard(self, key: int) -> None:
        """
        Handles keyboard orbital camera controls using strict Arrow Keys and Numpad Keys
        to completely avoid collision with system letter commands (W/S/A/D/L etc).
        Requires cv2.waitKeyEx() to capture arrow keys properly.
        """
        # LEFT ARROW (Win: 2424832, Linux: 65361) or Numpad 4
        if key in (2424832, 65361, ord("4")):
            self.azimuth -= 5.0
        # RIGHT ARROW (Win: 2555904, Linux: 65363) or Numpad 6
        elif key in (2555904, 65363, ord("6")):
            self.azimuth += 5.0
        # UP ARROW (Win: 2490368, Linux: 65362) or Numpad 8
        elif key in (2490368, 65362, ord("8")):
            self.elevation = min(85.0, self.elevation + 5.0)
        # DOWN ARROW (Win: 2621440, Linux: 65364) or Numpad 2
        elif key in (2621440, 65364, ord("2")):
            self.elevation = max(-85.0, self.elevation - 5.0)

    def _project_point(self, pt_3d: np.ndarray) -> tuple[int, int]:
        """
        Projects 3D world point (meters) to 2D canvas pixels with correct Y-axis orientation.
        In MediaPipe world coordinates, Y is negative upwards (towards head).
        Center origin (0,0,0) is grounded near the bottom-center of the viewport canvas.
        """
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)

        # Rotation matrices around Y (azimuth) and X (elevation)
        R_y = np.array(
            [
                [np.cos(az_rad), 0, np.sin(az_rad)],
                [0, 1, 0],
                [-np.sin(az_rad), 0, np.cos(az_rad)],
            ]
        )

        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(el_rad), -np.sin(el_rad)],
                [0, np.sin(el_rad), np.cos(el_rad)],
            ]
        )

        rotated = pt_3d @ R_y.T @ R_x.T

        # Screen coordinates
        u = int(rotated[0] * self.zoom + (self.width // 2))
        v = int(rotated[1] * self.zoom + (self.height - 140))

        return u, v

    def render_scene(
        self,
        landmarks_3d: np.ndarray | None,
        plane_normal: np.ndarray | None = None,
        joint_colors: dict[int, str] | None = None,
    ) -> np.ndarray:
        """Renders floor grid, 3D Digital Twin skeleton, and directed SVD plane normal vector."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas.fill(18)  # Sleek dark background

        if joint_colors is None:
            joint_colors = {}

        # 1. Render Floor Grid (XZ plane at Y=0)
        grid_step = 0.3  # 30 cm grid spacing
        for i in range(-4, 5):
            # Lines parallel to Z
            p1 = self._project_point(np.array([-1.2, 0.0, i * grid_step]))
            p2 = self._project_point(np.array([1.2, 0.0, i * grid_step]))
            cv2.line(canvas, p1, p2, (45, 45, 45), 1, cv2.LINE_AA)

            # Lines parallel to X
            p3 = self._project_point(np.array([i * grid_step, 0.0, -1.2]))
            p4 = self._project_point(np.array([i * grid_step, 0.0, 1.2]))
            cv2.line(canvas, p3, p4, (45, 45, 45), 1, cv2.LINE_AA)

        # Draw origin cross (center of floor grid)
        o_pt = self._project_point(np.array([0.0, 0.0, 0.0]))
        cv2.circle(canvas, o_pt, 4, (100, 100, 100), -1)

        # 2. Render Directed SVD Mirror Plane Normal Vector on Floor Grid (Y=0)
        if plane_normal is not None and len(plane_normal) == 3:
            n_start = np.array([0.0, 0.0, 0.0])  # Floor origin Y=0
            n_end = n_start + (plane_normal * 0.6)  # 60cm vector length
            p_start = self._project_point(n_start)
            p_end = self._project_point(n_end)

            # Draw directed arrow vector (BGR Cyan/Orange: 255, 128, 0)
            cv2.arrowedLine(
                canvas, p_start, p_end, (255, 128, 0), 2, cv2.LINE_AA, tipLength=0.2
            )
            cv2.circle(canvas, p_start, 4, (255, 128, 0), -1)

            # Text label for SVD vector
            label_pos = (p_end[0] + 8, p_end[1] - 8)
            cv2.putText(
                canvas,
                "SVD Estimated Reflection Normal",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 160, 50),
                1,
                cv2.LINE_AA,
            )

        # 3. Render 3D Digital Twin Skeleton
        if landmarks_3d is not None and len(landmarks_3d) == 33:
            pts_2d = [self._project_point(pt) for pt in landmarks_3d]

            # Draw Bones
            for idx_a, idx_b in self.CONNECTIONS:
                pt_a = pts_2d[idx_a]
                pt_b = pts_2d[idx_b]
                cv2.line(canvas, pt_a, pt_b, (180, 180, 180), 2, cv2.LINE_AA)

            # Draw Joints with Dynamic Biomechanical Colors
            for i, pt_2d in enumerate(pts_2d):
                status = joint_colors.get(i, "default")
                color = self.COLOR_MAP.get(status, self.COLOR_MAP["default"])
                radius = 6 if i in (0, 11, 12, 23, 24) else 4
                cv2.circle(canvas, pt_2d, radius, color, -1, cv2.LINE_AA)

        # Overlay Camera Instructions
        cv2.putText(
            canvas,
            f"Orbit Controls: [ARROWS or NUMPAD 8/2/4/6] Elev ({self.elevation:.0f}deg) Azim ({self.azimuth:.0f}deg)",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return canvas
