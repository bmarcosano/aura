"""Mirror-plane estimation utilities.

The metric ``estimate_plane`` method requires corresponding 3D points already
expressed in one common camera coordinate system. MediaPipe's two independent
hip-centred ``pose_world_landmarks`` arrays do NOT satisfy this requirement.
"""

from __future__ import annotations

import numpy as np


class MirrorPlaneEstimator:
    """Estimate a plane from 3D point pairs and build its Householder reflection.

    The constructor order is preserved from the current AURA release:
    ``MirrorPlaneEstimator(ema_alpha=0.15, history_size=90)``.
    """

    def __init__(self, ema_alpha: float = 0.15, history_size: int = 90) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self.ema_alpha = float(ema_alpha)
        self.history_size = int(history_size)
        self._diff_history: list[np.ndarray] = []
        self.is_estimated = False
        self.normal_vector = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self._reflection_matrix: np.ndarray | None = None  # Legacy 3x3 getter.
        self._affine_reflection_matrix: np.ndarray | None = None
        self.ema_normal: np.ndarray | None = None
        self.ema_centroid: np.ndarray | None = None
        self.last_planarity_rms = float("inf")
        self.last_plane_valid = False

    def add_frame_data(
        self, front_landmarks: np.ndarray, mirror_landmarks: np.ndarray
    ) -> tuple[bool, str, str]:
        """Store legacy 3D displacement observations for diagnostic normal estimation.

        This legacy method cannot recover metric plane offset and must not be
        treated as a replacement for ArUco-based mirror calibration.
        """
        front = np.asarray(front_landmarks, dtype=np.float64)
        mirror = np.asarray(mirror_landmarks, dtype=np.float64)
        if front.shape != mirror.shape or front.ndim != 2 or front.shape[1] != 3:
            raise ValueError("Both landmark arrays must have identical shape (N,3)")
        if not np.isfinite(front).all() or not np.isfinite(mirror).all():
            return (
                False,
                "Invalid calibration observations.",
                "Non-finite landmark coordinates.",
            )
        self._diff_history.append(front - mirror)
        self._diff_history = self._diff_history[-self.history_size :]
        count = len(self._diff_history)
        return (
            count >= self.history_size,
            f"Accumulating legacy plane data: {count}/{self.history_size}.",
            f"Stored {count} displacement frames.",
        )

    def estimate(self) -> tuple[np.ndarray, float, float, str, str]:
        """Estimate a legacy displacement direction (orientation only, not metric plane)."""
        if not self._diff_history:
            raise RuntimeError("Cannot estimate plane: history buffer is empty")
        data = np.vstack(self._diff_history)
        data = data[np.isfinite(data).all(axis=1)]
        if len(data) < 3:
            raise RuntimeError("Not enough finite displacement vectors")
        centered = data - np.mean(data, axis=0)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        # The dominant displacement direction is used only as a legacy heuristic.
        normal = vt[0]
        normal /= np.linalg.norm(normal)
        if normal[2] > 0:
            normal = -normal
        self.normal_vector = normal
        nx, ny, nz = normal
        self.yaw_deg = float(np.degrees(np.arctan2(nx, -nz)))
        self.pitch_deg = float(np.degrees(np.arcsin(np.clip(ny, -1.0, 1.0))))
        self._reflection_matrix = np.eye(3) - 2.0 * np.outer(normal, normal)
        self.is_estimated = True
        note = "Legacy normal estimate only; metric plane offset is unknown."
        return (
            normal.copy(),
            self.yaw_deg,
            self.pitch_deg,
            note,
            f"SVD singular values: {singular_values.tolist()}",
        )

    def get_reflection_matrix(self) -> np.ndarray:
        """Return the legacy 3x3 Householder linear component."""
        if self._reflection_matrix is None:
            raise RuntimeError("Mirror plane has not been estimated")
        return self._reflection_matrix.copy()

    def get_affine_reflection_matrix(self) -> np.ndarray:
        """Return the complete 4x4 reflection, including plane offset."""
        if self._affine_reflection_matrix is None:
            raise RuntimeError("A metric mirror plane has not been estimated")
        return self._affine_reflection_matrix.copy()

    def is_calibrated_state(self) -> bool:
        """Return True only when a metric 4x4 plane transform has been estimated."""
        return bool(
            self.last_plane_valid and self._affine_reflection_matrix is not None
        )

    def estimate_plane(
        self, direct_lms_3d: np.ndarray, reflected_lms_3d: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Fit a metric plane from corresponding points in one common 3D frame.

        The midpoint of each direct/reflected pair lies on the mirror plane.
        A total-least-squares plane fit is then obtained with SVD. Inputs must
        already be camera-space points; hip-centred MediaPipe arrays are invalid.
        Returns ``(4x4 reflection matrix, unit normal)`` or ``(None, None)``.
        """
        direct = np.asarray(direct_lms_3d, dtype=np.float64)
        reflected = np.asarray(reflected_lms_3d, dtype=np.float64)
        self.last_plane_valid = False
        if direct.shape != reflected.shape or direct.ndim != 2 or direct.shape[1] != 3:
            return None, None
        finite = np.isfinite(direct).all(axis=1) & np.isfinite(reflected).all(axis=1)
        direct, reflected = direct[finite], reflected[finite]
        if len(direct) < 4:
            return None, None

        midpoints = 0.5 * (direct + reflected)
        centroid = np.mean(midpoints, axis=0)
        centered = midpoints - centroid
        if np.linalg.matrix_rank(centered, tol=1e-7) < 2:
            return None, None
        _, _singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            return None, None
        normal = normal / norm
        if (
            self.ema_normal is not None
            and np.dot(normal, self.ema_normal) < 0
            or self.ema_normal is None
            and normal[2] < 0
        ):
            normal = -normal

        # Reject points that do not support a sufficiently planar midpoint set.
        residuals = centered @ normal
        self.last_planarity_rms = float(np.sqrt(np.mean(residuals**2)))
        spatial_scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if spatial_scale < 1e-5 or self.last_planarity_rms > max(
            0.02, 0.05 * spatial_scale
        ):
            return None, None

        if self.ema_normal is None:
            self.ema_normal = normal.copy()
            self.ema_centroid = centroid.copy()
        else:
            blended = self.ema_alpha * normal + (1.0 - self.ema_alpha) * self.ema_normal
            blended_norm = np.linalg.norm(blended)
            if blended_norm < 1e-10:
                return None, None
            self.ema_normal = blended / blended_norm
            self.ema_centroid = (
                self.ema_alpha * centroid + (1.0 - self.ema_alpha) * self.ema_centroid
            )

        n = self.ema_normal.copy()
        d = -float(np.dot(n, self.ema_centroid))
        H = np.eye(3) - 2.0 * np.outer(n, n)
        t = -2.0 * d * n
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = H
        transform[:3, 3] = t
        self.normal_vector = n
        self._reflection_matrix = H.copy()
        self._affine_reflection_matrix = transform
        self.is_estimated = True
        self.last_plane_valid = True
        self.yaw_deg = float(np.degrees(np.arctan2(n[0], -n[2])))
        self.pitch_deg = float(np.degrees(np.arcsin(np.clip(n[1], -1.0, 1.0))))
        return transform.copy(), n.copy()
