"""
mirror_plane.py: Estimates the 3D mirror plane using Singular Value Decomposition (SVD).
"""

from __future__ import annotations

import numpy as np


class MirrorPlaneEstimator:
    """
    Estimates the spatial plane of a mirror using SVD on landmark displacement vectors.
    """

    def __init__(self, ema_alpha: float = 0.15, history_size: int = 90) -> None:
        """
        Initializes the MirrorPlaneEstimator.

        Args:
            ema_alpha: Smoothing factor for the Exponential Moving Average filter.
            history_size: Number of frames to keep in the history buffer.
        """
        self.ema_alpha: float = ema_alpha
        self.history_size: int = history_size
        self._diff_history: list[np.ndarray] = []

        self.is_estimated: bool = False
        self.normal_vector: np.ndarray = np.array([0.0, 0.0, -1.0])
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self._reflection_matrix: np.ndarray | None = None

        self.ema_normal = None
        self.ema_centroid = None

    def add_frame_data(
        self, front_landmarks: np.ndarray, mirror_landmarks: np.ndarray
    ) -> tuple[bool, str, str]:
        """
        Accumulates 3D landmark difference vectors and returns detailed logs.
        """
        if front_landmarks.shape != mirror_landmarks.shape:
            raise ValueError(
                f"Landmark shape mismatch: Direct view {front_landmarks.shape} "
                f"vs Mirror view {mirror_landmarks.shape}"
            )

        diffs = front_landmarks - mirror_landmarks
        self._diff_history.append(diffs)

        if len(self._diff_history) > self.history_size:
            self._diff_history.pop(0)

        current_count = len(self._diff_history)
        user_msg = f"Accumulating mirror calibration data: frame {current_count}/{self.history_size}."
        expert_msg = f"Buffer append: {current_count}/{self.history_size} frames stored. Displaced landmarks shape: {diffs.shape}."

        is_ready = current_count >= self.history_size
        return is_ready, user_msg, expert_msg

    def estimate(self) -> tuple[np.ndarray, float, float, str, str]:
        """
        Computes mirror plane normal and returns detailed SVD execution logs.
        """
        if len(self._diff_history) == 0:
            raise RuntimeError(
                "Cannot estimate mirror plane: landmark history buffer is empty."
            )

        all_diffs = np.vstack(self._diff_history)
        _, S, Vh = np.linalg.svd(all_diffs, full_matrices=False)

        normal = Vh[0]
        if normal[2] > 0:
            normal = -normal

        norm_length = np.linalg.norm(normal)
        if norm_length > 0:
            normal = normal / norm_length

        self.normal_vector = normal
        nx, ny, nz = normal

        self.yaw_deg = float(np.degrees(np.arctan2(nx, -nz)))
        self.pitch_deg = float(np.degrees(np.arcsin(np.clip(ny, -1.0, 1.0))))

        self.is_estimated = True

        n = self.normal_vector.reshape(3, 1)
        self._reflection_matrix = np.eye(3) - 2.0 * (n @ n.T)

        user_msg = f"Mirror plane successfully calculated! Yaw: {self.yaw_deg:.1f}°, Pitch: {self.pitch_deg:.1f}°."
        expert_msg = (
            f"SVD executed on matrix {all_diffs.shape}. Singular values S={S[:3]}. "
            f"Normal n=[{nx:.4f}, {ny:.4f}, {nz:.4f}], Yaw={self.yaw_deg:.2f}deg, Pitch={self.pitch_deg:.2f}deg. "
            f"Householder matrix R computed."
        )

        return self.normal_vector, self.yaw_deg, self.pitch_deg, user_msg, expert_msg

    def get_reflection_matrix(self) -> np.ndarray:
        if not self.is_estimated or self._reflection_matrix is None:
            raise RuntimeError("Mirror plane has not been estimated yet.")
        return self._reflection_matrix

    def is_calibrated_state(self) -> bool:
        return self.is_estimated

    def estimate_plane(
        self, direct_lms_3d: np.ndarray, reflected_lms_3d: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Calculates the 4x4 reflection matrix given two (N, 3) numpy arrays of landmarks.
        Uses an Exponential Moving Average (EMA) filter for stability.

        Args:
             direct_lms_3d: 3D coordinates from the direct view.
             reflected_lms_3d: 3D coordinates from the reflected view.

        Returns:
             A tuple containing the 4x4 reflection matrix and the normal vector.
             Returns (None, None) if not enough landmarks are provided.
        """
        if len(direct_lms_3d) < 3 or len(reflected_lms_3d) < 3:
            return None, None

        # 1. Midpoints: physically lie on the mirror plane
        midpoints = (direct_lms_3d + reflected_lms_3d) / 2.0

        # 2. Centroid of the midpoints
        centroid = np.mean(midpoints, axis=0)

        # 3. Center points for SVD
        centered_midpoints = midpoints - centroid

        # 4. SVD: the normal is the vector with the smallest variance (last row of Vt)
        _U, _S, Vt = np.linalg.svd(centered_midpoints)
        normal = Vt[-1, :]

        # Ensure normal points towards the camera (Z > 0)
        if normal[2] < 0:
            normal = -normal

        # 5. EMA filter to stabilize the plane over time
        if self.ema_normal is None:
            self.ema_normal = normal
            self.ema_centroid = centroid
        else:
            self.ema_normal = (
                self.ema_alpha * normal + (1 - self.ema_alpha) * self.ema_normal
            )
            self.ema_normal /= np.linalg.norm(self.ema_normal)  # Re-normalize
            self.ema_centroid = (
                self.ema_alpha * centroid + (1 - self.ema_alpha) * self.ema_centroid
            )

        # 6. Build the 4x4 Householder affine matrix
        n = self.ema_normal
        c = self.ema_centroid

        R = np.eye(3) - 2 * np.outer(n, n)
        d = -np.dot(n, c)  # Distance from origin
        t = -2 * d * n

        reflection_matrix = np.eye(4)
        reflection_matrix[:3, :3] = R
        reflection_matrix[:3, 3] = t

        # Keep legacy variables in sync if needed
        self.normal_vector = self.ema_normal
        self._reflection_matrix = reflection_matrix[:3, :3]
        self.is_estimated = True

        return reflection_matrix, n
