"""
Module for 3D pose reconstruction, landmark fusion, and multi-view triangulation.

Combines direct-view and rectified mirror-view 3D landmarks into a unified
Digital Twin representation, applying visibility-weighted confidence fusion to handle
occlusions and oblique camera setups.
"""

from __future__ import annotations

import numpy as np


class PoseReconstructor:
    """
    Fuses multi-view 3D skeletal landmarks into a robust, unified 3D pose estimate.
    """

    def __init__(self, confidence_weight: float = 0.5) -> None:
        """
        Initializes the Pose Reconstructor.

        Args:
            confidence_weight (float): Base weight assigned to mirror-view landmarks
                                       during fusion (0.0 = front only, 1.0 = mirror only).
        """
        self.confidence_weight = confidence_weight

    def fuse_landmarks(
        self,
        front_landmarks: np.ndarray | None,
        rectified_mirror_landmarks: np.ndarray | None,
        front_visibility: np.ndarray | None = None,
        mirror_visibility: np.ndarray | None = None,
        is_anthropometrically_valid: bool = True,
    ) -> np.ndarray | None:
        """
        Fuses front-view and rectified mirror-view 3D landmarks based on visibility scores.
        If a landmark is occluded in one view, it trusts the view where it is visible.
        """
        if front_landmarks is None:
            return None

        # If validation failed strictly, fall back to front view
        if rectified_mirror_landmarks is None or not is_anthropometrically_valid:
            return front_landmarks

        # Default visibilities to 1.0 if not provided
        n_points = len(front_landmarks)
        v_front = (
            front_visibility if front_visibility is not None else np.ones(n_points)
        )
        v_mirror = (
            mirror_visibility if mirror_visibility is not None else np.ones(n_points)
        )

        # Weight calculation based on visibility AND the base confidence weight
        w_front = v_front * (1.0 - self.confidence_weight)
        w_mirror = v_mirror * self.confidence_weight

        # Normalize weights to sum to 1.0 per landmark (add epsilon to avoid div by zero)
        weight_sum = w_front + w_mirror + 1e-6
        w_front_norm = (w_front / weight_sum).reshape(-1, 1)
        w_mirror_norm = (w_mirror / weight_sum).reshape(-1, 1)

        # Dynamic point-by-point fusion
        fused_landmarks = (w_front_norm * front_landmarks) + (
            w_mirror_norm * rectified_mirror_landmarks
        )
        return fused_landmarks
