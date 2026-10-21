"""
Module for markerless 3D mirror plane estimation using Singular Value Decomposition (SVD).

Estimates the 3D mirror plane normal vector n = (nx, ny, nz) and calculates 3D orientation
angles (Yaw and Pitch) from paired direct and specular MediaPipe 3D World Landmarks.
This removes the need for physical calibration targets (e.g., ArUco markers, AprilTags)
or manual camera-mirror alignment.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np


class MirrorPlaneEstimator:
    """
    Estimates 3D mirror plane orientation using SVD on landmark displacement vectors.

    Given corresponding 3D anatomical joints detected in direct view (P_f) and mirror 
    reflection view (P_m), the displacement vector v_i = P_{f,i} - P_{m,i} is orthogonal 
    to the mirror plane. By stacking these vectors across multiple frames, Singular Value 
    Decomposition (SVD) isolates the primary eigenvector corresponding to the plane normal.
    """

    def __init__(self, history_size: int = 90) -> None:
        """
        Initializes the mirror plane estimator.

        Args:
            history_size (int): Number of consecutive frames to accumulate displacement 
                                vectors for temporal convergence (e.g., 90 frames = 3 sec at 30 FPS).
        """
        self.history_size: int = history_size
        self._diff_history: List[np.ndarray] = []

        # Internal estimation state
        self.is_estimated: bool = False
        self.normal_vector: np.ndarray = np.array([0.0, 0.0, -1.0])
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0

    def add_frame_data(self, front_landmarks: np.ndarray, mirror_landmarks: np.ndarray) -> bool:
        """
        Accumulates 3D landmark difference vectors for the current frame.

        Args:
            front_landmarks (np.ndarray): Array of shape (N, 3) representing 3D world 
                                          coordinates from direct view.
            mirror_landmarks (np.ndarray): Array of shape (N, 3) representing 3D world 
                                           coordinates from mirror view.

        Returns:
            bool: True if the internal history buffer has accumulated enough frames to run 
                  estimation, False otherwise.

        Raises:
            ValueError: If the shapes of front and mirror landmark arrays do not match.
        """
        if front_landmarks.shape != mirror_landmarks.shape:
            raise ValueError(
                f"Landmark shape mismatch: Direct view {front_landmarks.shape} "
                f"vs Mirror view {mirror_landmarks.shape}"
            )

        # Compute displacement vectors v_i = P_f - P_m for all joints
        diffs = front_landmarks - mirror_landmarks
        self._diff_history.append(diffs)

        # Maintain sliding temporal window
        if len(self._diff_history) > self.history_size:
            self._diff_history.pop(0)

        return len(self._diff_history) >= self.history_size

    def estimate(self) -> Tuple[np.ndarray, float, float]:
        """
        Computes the mirror plane unit normal vector and orientation angles using SVD.

        The difference vectors are aggregated into an (M, 3) matrix where M = history_size * N_landmarks.
        Applying SVD yields Vh, where the first right-singular vector (riga 0) aligns with 
        the dominant variance direction—the plane normal vector n.

        Returns:
            Tuple[np.ndarray, float, float]: 
                - normal_vector (np.ndarray): Unit normal vector n = (nx, ny, nz).
                - yaw_deg (float): Horizontal rotation angle in degrees.
                - pitch_deg (float): Vertical tilt angle in degrees.

        Raises:
            RuntimeError: If called before any landmark data has been accumulated.
        """
        if len(self._diff_history) == 0:
            raise RuntimeError("Cannot estimate mirror plane: landmark history buffer is empty.")

        # Stack frame history into a unified 2D matrix of shape (M, 3)
        all_diffs = np.vstack(self._diff_history)

        # Singular Value Decomposition: all_diffs = U * S * Vh
        # Vh contains the principal orthogonal directions
        _, _, Vh = np.linalg.svd(all_diffs, full_matrices=False)

        # Principal eigenvector corresponds to the mirror plane normal
        normal = Vh[0]

        # Enforce camera-facing convention:
        # In MediaPipe camera coordinates, Z extends away from the camera.
        # We ensure the normal vector points back towards the observer (nz < 0).
        if normal[2] > 0:
            normal = -normal

        # Normalize to unit length
        norm_length = np.linalg.norm(normal)
        if norm_length > 0:
            normal = normal / norm_length

        self.normal_vector = normal

        nx, ny, nz = normal

        # Yaw (horizontal tilt/rotation): theta_yaw = arctan(nx / -nz)
        self.yaw_deg = float(np.degrees(np.arctan2(nx, -nz)))

        # Pitch (vertical tilt): theta_pitch = arcsin(ny)
        self.pitch_deg = float(np.degrees(np.arcsin(np.clip(ny, -1.0, 1.0))))

        self.is_estimated = True
        return self.normal_vector, self.yaw_deg, self.pitch_deg

    def get_reflection_matrix(self) -> np.ndarray:
        """
        Calculates the Householder reflection transformation matrix R = I - 2 * n * n^T.

        This matrix allows transforming mirror-view coordinates back into the primary 
        camera reference space: P'_m = R * P_m.

        Returns:
            np.ndarray: (3, 3) orthogonal transformation reflection matrix.

        Raises:
            RuntimeError: If called before mirror plane estimation has been executed.
        """
        if not self.is_calibrated_state():
            raise RuntimeError(
                "Mirror plane has not been estimated yet. Call 'estimate()' prior to requesting matrix."
            )

        n = self.normal_vector.reshape(3, 1)  # Column vector (3, 1)
        identity = np.eye(3)

        # Householder transformation matrix
        reflection_matrix = identity - 2.0 * (n @ n.T)

        return reflection_matrix

    def is_calibrated_state(self) -> bool:
        """Returns True if the mirror plane normal and orientation have been successfully estimated."""
        return self.is_estimated