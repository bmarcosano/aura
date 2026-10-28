"""
Module for 3D pose reconstruction, landmark fusion, and multi-view triangulation.

Combines direct-view and rectified mirror-view 3D landmarks into a unified
Digital Twin representation, applying visibility-weighted confidence fusion to handle
occlusions and oblique camera setups. Incorporates temporal EMA smoothing and spatial anchoring.
"""

from __future__ import annotations

import numpy as np


class PoseReconstructor:
    """
    Fuses multi-view 3D skeletal landmarks into a robust, unified 3D pose estimate,
    anchors the skeleton to the floor, and applies EMA temporal smoothing.
    """

    def __init__(self, confidence_weight: float = 0.5, alpha: float = 0.4) -> None:
        """
        Initializes the Pose Reconstructor.

        Args:
            confidence_weight (float): Base weight assigned to mirror-view landmarks.
            alpha (float): Smoothing factor for EMA (0.0 = max smoothing, 1.0 = no smoothing).
        """
        self.confidence_weight = confidence_weight
        self.alpha = alpha
        self._previous_fused: np.ndarray | None = None

    def fuse_landmarks(
        self,
        front_landmarks: np.ndarray | None,
        rectified_mirror_landmarks: np.ndarray | None,
        front_visibility: np.ndarray | None = None,
        mirror_visibility: np.ndarray | None = None,
        is_anthropometrically_valid: bool = True,
    ) -> np.ndarray | None:
        """
        Fuses front-view and rectified mirror-view 3D landmarks, anchors the feet to the floor,
        and applies temporal smoothing to reduce jitter.
        """
        if front_landmarks is None:
            return None

        # 1. Spatial Fusion
        if rectified_mirror_landmarks is None or not is_anthropometrically_valid:
            fused_landmarks = front_landmarks.copy()
        else:
            n_points = len(front_landmarks)
            v_front = (
                front_visibility if front_visibility is not None else np.ones(n_points)
            )
            v_mirror = (
                mirror_visibility
                if mirror_visibility is not None
                else np.ones(n_points)
            )

            w_front = v_front * (1.0 - self.confidence_weight)
            w_mirror = v_mirror * self.confidence_weight

            weight_sum = w_front + w_mirror + 1e-6
            w_front_norm = (w_front / weight_sum).reshape(-1, 1)
            w_mirror_norm = (w_mirror / weight_sum).reshape(-1, 1)

            fused_landmarks = (w_front_norm * front_landmarks) + (
                w_mirror_norm * rectified_mirror_landmarks
            )

        # 2. Spatial Anchoring (Grounding)
        # MediaPipe world landmarks are relative to the hip center (0,0,0).
        # To stop feet from sliding, we anchor the origin to the floor/mid-ankles.
        # Indices: 27=left_ankle, 28=right_ankle, 29=left_heel, 30=right_heel, 31=left_toe, 32=right_toe

        # Floor Y: maximum Y value among all foot points (Y is down in MediaPipe)
        floor_y = np.max(fused_landmarks[27:33, 1])

        # Center X and Z around the midpoint of the ankles
        mid_ankle_x = (fused_landmarks[27, 0] + fused_landmarks[28, 0]) / 2.0
        mid_ankle_z = (fused_landmarks[27, 2] + fused_landmarks[28, 2]) / 2.0

        fused_landmarks[:, 0] -= mid_ankle_x
        fused_landmarks[:, 1] -= floor_y
        fused_landmarks[:, 2] -= mid_ankle_z

        # 3. Temporal EMA Smoothing
        if self._previous_fused is None:
            self._previous_fused = fused_landmarks
        else:
            fused_landmarks = (self.alpha * fused_landmarks) + (
                (1.0 - self.alpha) * self._previous_fused
            )
            self._previous_fused = fused_landmarks

        return fused_landmarks

    def reset_smoothing(self) -> None:
        """Clears the temporal smoothing buffer, useful when the video restarts."""
        self._previous_fused = None
