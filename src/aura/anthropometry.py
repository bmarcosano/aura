"""Anthropometric plausibility and cross-view isometry checks."""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class AnthropometricValidator:
    """Validate skeletal geometry without pretending to certify biomechanics.

    ``units`` is retained for compatibility with the validator previously
    defined in ``multi_detector.py``. ``apply_mirror_reflection`` accepts both
    a legacy 3x3 linear matrix and a complete 4x4 affine reflection.
    """

    BONE_SEGMENTS: ClassVar[list[tuple[int, int]]] = [
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]

    def __init__(
        self,
        error_threshold: float = 0.40,
        units: str = "meters",
        min_valid_bones: int = 4,
    ) -> None:
        if error_threshold <= 0 or min_valid_bones < 1:
            raise ValueError("Invalid anthropometric validation settings")
        self.error_threshold = float(error_threshold)
        self.units = units
        self.min_valid_bones = int(min_valid_bones)
        self.last_checked_bones = 0

    def apply_mirror_reflection(
        self, mirror_landmarks: np.ndarray, reflection_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply a 3x3 or 4x4 reflection to points in the same declared frame."""
        points = np.asarray(mirror_landmarks, dtype=np.float64)
        matrix = np.asarray(reflection_matrix, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("mirror_landmarks must have shape (N,3)")
        if matrix.shape == (3, 3):
            return (matrix @ points.T).T
        if matrix.shape == (4, 4):
            homogeneous = np.c_[points, np.ones(len(points))]
            return (matrix @ homogeneous.T).T[:, :3]
        raise ValueError("reflection_matrix must have shape (3,3) or (4,4)")

    def validate_isometry(
        self, front_landmarks: np.ndarray, rectified_mirror: np.ndarray
    ) -> tuple[bool, float, str, str]:
        """Compare corresponding bone lengths; fail closed if too few are usable."""
        front = np.asarray(front_landmarks, dtype=np.float64)
        mirror = np.asarray(rectified_mirror, dtype=np.float64)
        if front.shape != (33, 3) or mirror.shape != (33, 3):
            raise ValueError("Both inputs must have shape (33,3)")
        errors: list[float] = []
        for start, end in self.BONE_SEGMENTS:
            a, b = front[start] - front[end], mirror[start] - mirror[end]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if la > 1e-6 and lb > 1e-6:
                errors.append(abs(la - lb) / la)
        self.last_checked_bones = len(errors)
        if len(errors) < self.min_valid_bones:
            return (
                False,
                float("inf"),
                "Insufficient valid body segments.",
                f"Only {len(errors)} segments available; minimum is {self.min_valid_bones}.",
            )
        maximum = max(errors)
        valid = maximum <= self.error_threshold
        return (
            valid,
            maximum,
            f"Skeletal isometry discrepancy: {maximum * 100:.1f}%.",
            f"Checked {len(errors)} segments; maximum relative error={maximum:.4f}; threshold={self.error_threshold:.4f}.",
        )

    def validate_skeleton_3d(self, landmarks_3d: np.ndarray) -> bool:
        """Check finite points and broad segment plausibility in the configured units."""
        points = np.asarray(landmarks_3d, dtype=np.float64)
        if points.shape != (33, 3) or not np.isfinite(points).all():
            return False
        lengths = [
            float(np.linalg.norm(points[a] - points[b])) for a, b in self.BONE_SEGMENTS
        ]
        lengths = [value for value in lengths if np.isfinite(value) and value > 1e-4]
        if len(lengths) < self.min_valid_bones:
            return False
        # Broad limits only catch unit/scale failures; they are not normative body-size ranges.
        limits = (0.015, 1.5) if self.units == "meters" else (1.0, 200.0)
        return all(limits[0] <= value <= limits[1] for value in lengths)
