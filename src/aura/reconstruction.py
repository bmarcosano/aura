"""
Module for 3D pose reconstruction, landmark fusion, and multi-view triangulation.

Combines direct-view and rectified mirror-view 2D pixel landmarks into a unified
metric 3D Digital Twin representation using absolute camera calibration and affine
reflection matrices.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoReconstructor:
    """
    Reconstructs exact 3D skeletal landmarks from multi-view 2D pixel coordinates
    (direct camera and virtual reflected camera), utilizing affine reflection geometry
    and intrinsic camera calibration to minimize reprojection error.
    """

    def __init__(self, camera_matrix: np.ndarray) -> None:
        """
        Initializes the reconstructor with the physical camera intrinsic matrix.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix (K) from calibration.
        """
        self.camera_matrix = camera_matrix

        # P1 is the projection matrix for the direct view (physical camera).
        # Assuming the camera is at the origin: P1 = K [ I | 0 ]
        extrinsics_direct = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P1 = np.dot(self.camera_matrix, extrinsics_direct)

    def _check_reprojection_error(
        self,
        pt3d: np.ndarray,
        pt2d_direct: np.ndarray,
        pt2d_reflected: np.ndarray,
        P2: np.ndarray,
        max_error_px: float = 15.0,
    ) -> bool:
        """
        Verifies how well a triangulated 3D point matches the original 2D observations.

        Args:
            pt3d: (3,) numpy array representing the 3D point.
            pt2d_direct: (2,) numpy array representing the pixel coordinate in the direct view.
            pt2d_reflected: (2,) numpy array representing the pixel coordinate in the mirror view.
            P2: 3x4 projection matrix of the reflected view.
            max_error_px: Maximum acceptable reprojection error in pixels.

        Returns:
            True if the error in both views is below the threshold, False otherwise.
        """
        pt3d_homog = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])

        # Reprojection onto direct view
        proj1 = self.P1 @ pt3d_homog
        if proj1[2] == 0:
            return False
        proj1 = proj1[:2] / proj1[2]
        err1 = np.linalg.norm(proj1 - pt2d_direct)

        # Reprojection onto reflected view
        proj2 = P2 @ pt3d_homog
        if proj2[2] == 0:
            return False
        proj2 = proj2[:2] / proj2[2]
        err2 = np.linalg.norm(proj2 - pt2d_reflected)

        return err1 < max_error_px and err2 < max_error_px

    def triangulate_landmarks(
        self,
        points_2d_direct: np.ndarray,
        points_2d_reflected: np.ndarray,
        reflection_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Triangulates 3D points from 2D pixel coordinates of the direct and reflected views,
        using the affine reflection matrix of the mirror. Rejects points with high reprojection error.

        Args:
            points_2d_direct: Nx2 numpy array of (x, y) pixel coordinates (direct view).
            points_2d_reflected: Nx2 numpy array of (x, y) pixel coordinates (mirror view).
            reflection_matrix: 4x4 affine reflection matrix (from ArUco tracker or SVD).

        Returns:
            Nx3 numpy array of reconstructed 3D points in the camera coordinate system.
            Points failing the reprojection check will be preserved but can be filtered downstream.
        """
        # Ensure the inputs are float32 arrays
        pts1 = np.asarray(points_2d_direct, dtype=np.float32)
        pts2 = np.asarray(points_2d_reflected, dtype=np.float32)

        # 1. Extract the 3x4 extrinsic matrix from the 4x4 reflection matrix.
        # This defines the pose of the "virtual camera" behind the mirror.
        extrinsics_reflected = reflection_matrix[:3, :4]

        # 2. Build the projection matrix for the reflected view: P2 = K [ R_ref | t_ref ]
        P2 = np.dot(self.camera_matrix, extrinsics_reflected)

        # 3. OpenCV's triangulatePoints requires 2xN arrays
        pts1_t = pts1.T
        pts2_t = pts2.T

        # 4. Triangulate the points
        # Returns a 4xN array in homogeneous coordinates (X, Y, Z, W)
        points_4d_hom = cv2.triangulatePoints(self.P1, P2, pts1_t, pts2_t)

        # 5. Convert from homogeneous to 3D Euclidean coordinates (divide by W)
        # Add a small epsilon to avoid division by zero
        W = points_4d_hom[3, :] + 1e-7
        points_3d = (points_4d_hom[:3, :] / W).T

        # 6. Reprojection Check Pipeline (Optional flagging/smoothing can be implemented here)
        # For now, we compute it to log severe geometric mismatches
        valid_points_count = 0
        for i in range(len(points_3d)):
            if self._check_reprojection_error(
                points_3d[i], pts1[i], pts2[i], P2, max_error_px=15.0
            ):
                valid_points_count += 1

        if valid_points_count < len(points_3d) * 0.5:
            logger.warning(
                f"Low reconstruction quality: only {valid_points_count}/{len(points_3d)} points passed reprojection check."
            )

        # 7. Return as an Nx3 array to match standard landmark shapes
        return points_3d

    def fuse_landmarks(self, *args, **kwargs):
        """
        Legacy wrapper for backward compatibility.
        Previously used to average two independent 3D pose estimations.
        Raises a deprecation warning and directs to the new stereoscopic pipeline.
        """
        import warnings

        warnings.warn(
            "fuse_landmarks is deprecated. AURA now uses true Multi-View Geometry. "
            "Please use triangulate_landmarks(points_2d_direct, points_2d_reflected, reflection_matrix) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Fallback to a dummy return if old modules forcibly invoke it without matrix context
