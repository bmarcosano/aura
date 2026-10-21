"""
Module for anthropometric validation and spatial rectification of mirror landmarks.

Leverages the isometric property of specular reflection (preservation of Euclidean
distances) to validate the quality of 3D keypoint detection. It transforms mirror-view
landmarks into the primary camera reference frame using a Householder reflection matrix
and filters out hallucinated or heavily occluded keypoints based on bone-length discrepancies.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class AnthropometricValidator:
    """
    Validates 3D pose landmarks by comparing direct-view and rectified mirror-view bone lengths.
    """

    # Key anatomical bone segments using MediaPipe Pose landmark indices
    BONE_SEGMENTS: ClassVar[dict[str, tuple[int, int]]] = {
        "left_upper_arm": (11, 13),  # Left Shoulder to Left Elbow
        "left_forearm": (13, 15),  # Left Elbow to Left Wrist
        "right_upper_arm": (12, 14),  # Right Shoulder to Right Elbow
        "right_forearm": (14, 16),  # Right Elbow to Right Wrist
        "left_thigh": (23, 25),  # Left Hip to Left Knee
        "left_calf": (25, 27),  # Left Knee to Left Ankle
        "right_thigh": (24, 26),  # Right Hip to Right Knee
        "right_calf": (26, 28),  # Right Knee to Right Ankle
    }

    def __init__(self, error_threshold: float = 0.10) -> None:
        """
        Initializes the Anthropometric Validator.

        Args:
            error_threshold (float): Maximum allowed relative discrepancy between direct
                                     and mirror bone lengths (e.g., 0.10 means 10% error allowed).
        """
        self.error_threshold = error_threshold

    def apply_mirror_reflection(
        self, mirror_landmarks: np.ndarray, reflection_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Maps mirror-view keypoints into the primary camera coordinate system.

        Applies the Householder transformation: P'_m = P_m * R^T

        Args:
            mirror_landmarks (np.ndarray): Shape (N, 3) 3D landmarks from the mirror.
            reflection_matrix (np.ndarray): Shape (3, 3) orthogonal reflection matrix.

        Returns:
            np.ndarray: Rectified mirror landmarks in the camera's spatial frame.
        """
        if mirror_landmarks.ndim != 2 or mirror_landmarks.shape[1] != 3:
            raise ValueError("mirror_landmarks must have shape (N, 3).")

        if reflection_matrix.shape != (3, 3):
            raise ValueError("reflection_matrix must be a (3, 3) numpy array.")

        # Matrix multiplication: project mirror points using the reflection matrix
        rectified_landmarks = mirror_landmarks @ reflection_matrix.T
        return rectified_landmarks

    def compute_bone_lengths(self, landmarks: np.ndarray) -> dict[str, float]:
        """
        Calculates the Euclidean distance for predefined anatomical bone segments.

        Args:
            landmarks (np.ndarray): Shape (N, 3) array of 3D joint coordinates.

        Returns:
            dict[str, float]: Dictionary mapping segment names to their physical lengths.
        """
        lengths: dict[str, float] = {}
        for segment_name, (idx_a, idx_b) in self.BONE_SEGMENTS.items():
            point_a = landmarks[idx_a]
            point_b = landmarks[idx_b]
            distance = float(np.linalg.norm(point_a - point_b))
            lengths[segment_name] = distance
        return lengths

    def validate_isometry(
        self, front_landmarks: np.ndarray, rectified_mirror_landmarks: np.ndarray
    ) -> tuple[bool, dict[str, float]]:
        """
        Compares bone lengths between direct view and rectified mirror view.

        Metric: E_k = |L_{f,k} - L_{m,k}| / L_{f,k}

        Args:
            front_landmarks (np.ndarray): Shape (N, 3) direct view landmarks.
            rectified_mirror_landmarks (np.ndarray): Shape (N, 3) mirror landmarks projected into camera space.

        Returns:
            tuple[bool, dict[str, float]]:
                - is_valid (bool): True if ALL bone discrepancies are below the error threshold.
                - discrepancies (dict[str, float]): Dictionary containing the E_k error score for each segment.
        """
        if front_landmarks.shape != rectified_mirror_landmarks.shape:
            raise ValueError(
                "Shape mismatch between front and rectified mirror landmarks."
            )

        front_lengths = self.compute_bone_lengths(front_landmarks)
        mirror_lengths = self.compute_bone_lengths(rectified_mirror_landmarks)

        discrepancies: dict[str, float] = {}
        is_valid = True

        for segment_name in self.BONE_SEGMENTS:
            lf = front_lengths[segment_name]
            lm = mirror_lengths[segment_name]

            # Prevent division by zero if MediaPipe collapses a joint
            if lf < 1e-6:
                error = 1.0
            else:
                error = abs(lf - lm) / lf

            discrepancies[segment_name] = error

            if error > self.error_threshold:
                is_valid = False

        return is_valid, discrepancies
