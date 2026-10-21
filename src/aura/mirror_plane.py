"""
mirror_plane.py: Estimates the 3D mirror plane using Singular Value Decomposition (SVD).
"""

from __future__ import annotations

import numpy as np


class MirrorPlaneEstimator:
    """
    Estimates the spatial plane of a mirror using SVD on landmark displacement vectors.
    """

    def __init__(self, history_size: int = 90) -> None:
        self.history_size: int = history_size
        self._diff_history: list[np.ndarray] = []

        self.is_estimated: bool = False
        self.normal_vector: np.ndarray = np.array([0.0, 0.0, -1.0])
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self._reflection_matrix: np.ndarray | None = None

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
