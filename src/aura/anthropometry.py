"""
Module for validating anthropometric isometry between direct and mirror-reflected poses.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class AnthropometricValidator:
    """
    Validates skeletal bone-length isometry between direct and mirror views.
    """

    BONE_SEGMENTS: ClassVar[list[tuple[int, int]]] = [
        (11, 12),  # Shoulders
        (11, 23),  # Left Torso
        (12, 24),  # Right Torso
        (23, 24),  # Hips
        (11, 13),  # Left Upper Arm
        (13, 15),  # Left Forearm
        (12, 14),  # Right Upper Arm
        (14, 16),  # Right Forearm
        (23, 25),  # Left Thigh
        (25, 27),  # Left Shin
        (24, 26),  # Right Thigh
        (26, 28),  # Right Shin
    ]

    def __init__(self, error_threshold: float = 0.40) -> None:
        self.error_threshold = error_threshold

    def apply_mirror_reflection(
        self, mirror_landmarks: np.ndarray, reflection_matrix: np.ndarray
    ) -> np.ndarray:
        """Applies Householder reflection transformation matrix to mirror landmarks."""
        if mirror_landmarks.shape != (33, 3):
            raise ValueError(
                "Shape mismatch: mirror_landmarks must have shape (33, 3)."
            )
        if reflection_matrix.shape != (3, 3):
            raise ValueError(
                "Shape mismatch: reflection_matrix must have shape (3, 3)."
            )
        return (reflection_matrix @ mirror_landmarks.T).T

    def validate_isometry(
        self, front_landmarks: np.ndarray, rectified_mirror: np.ndarray
    ) -> tuple[bool, float, str, str]:
        """
        Validates bone length consistency between direct view and rectified mirror view.
        """
        max_error = 0.0
        checked_bones = 0
        if front_landmarks.shape != (33, 3) or rectified_mirror.shape != (33, 3):
            raise ValueError(
                "Shape mismatch: front_landmarks and mirror_landmarks must have shape (33, 3)."
            )
        for start_idx, end_idx in self.BONE_SEGMENTS:
            if start_idx >= len(front_landmarks) or end_idx >= len(front_landmarks):
                continue
            if start_idx >= len(rectified_mirror) or end_idx >= len(rectified_mirror):
                continue

            len_front = np.linalg.norm(
                front_landmarks[start_idx] - front_landmarks[end_idx]
            )
            len_mirror = np.linalg.norm(
                rectified_mirror[start_idx] - rectified_mirror[end_idx]
            )

            if len_front > 1e-5:
                error = abs(len_front - len_mirror) / len_front
                max_error = max(max_error, error)
                checked_bones += 1

        is_valid = max_error <= self.error_threshold

        user_msg = (
            f"Skeletal isometry check: max bone discrepancy is {max_error * 100:.1f}%."
        )
        expert_msg = f"Isometry validation: {checked_bones} bone segments checked. Max error={max_error:.4f} (Threshold={self.error_threshold}). Status: {'PASS' if is_valid else 'FAIL'}."

        return is_valid, max_error, user_msg, expert_msg
