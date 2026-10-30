"""
aruco_tracker.py: Multi-marker ArUco tracking for absolute spatial metrology.
Detects markers and estimates their 6D pose (rotation and translation) in real-time.
"""

from __future__ import annotations

import cv2
import numpy as np


class ArucoMultiTracker:
    """
    Detects ArUco markers to establish absolute 3D reference frames
    (e.g., origin on the floor, reflection plane on the mirror).
    """

    def __init__(
        self, dict_type: int = cv2.aruco.DICT_4X4_50, marker_size_m: float = 0.18
    ) -> None:
        """
        Args:
            dict_type: The ArUco dictionary to use (default: 4x4 squares, up to 50 IDs).
            marker_size_m: The physical size of the printed marker side in meters (default: 18 cm).
        """
        self.marker_size = marker_size_m

        # Initialize ArUco dictionary and parameters (compatible with OpenCV 4.7+)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Define the 3D coordinates of the marker's corners in its local coordinate system
        half_size = self.marker_size / 2.0
        self.obj_points = np.array(
            [
                [-half_size, half_size, 0],  # Top-left
                [half_size, half_size, 0],  # Top-right
                [half_size, -half_size, 0],  # Bottom-right
                [-half_size, -half_size, 0],  # Bottom-left
            ],
            dtype=np.float32,
        )

    def process_frame(
        self, frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
        """
        Detects markers in the frame and computes their rotation (rvec) and translation (tvec).

        Returns:
            - The annotated frame with drawn markers and coordinate axes.
            - A dictionary mapping marker IDs to their (rvec, tvec) poses.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        poses = {}

        if ids is not None:
            # Draw the square borders and IDs around detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                # Solve Perspective-n-Point for each marker to find its 3D pose
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    corners[i][0],
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if success:
                    poses[marker_id] = (rvec, tvec)
                    # Draw the 3D coordinate axes (X=Red, Y=Green, Z=Blue) on the marker
                    cv2.drawFrameAxes(
                        frame, camera_matrix, dist_coeffs, rvec, tvec, self.marker_size
                    )

                    # Display distance in text
                    distance_cm = np.linalg.norm(tvec) * 100
                    text_pos = (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10)
                    cv2.putText(
                        frame,
                        f"Dist: {distance_cm:.1f}cm",
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        return frame, poses

    def estimate_mirror_plane(
        self, rvec: np.ndarray, tvec: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """
        Calculates the exact mirror plane equation (n*P + d = 0) and the full 4x4
        affine reflection matrix using the ArUco marker pose.

        Args:
            rvec: Rotation vector from cv2.solvePnP (shape: 3x1 or 1x3).
            tvec: Translation vector from cv2.solvePnP (shape: 3x1 or 1x3).

        Returns:
            A dictionary containing the normal vector, point on plane, d, and reflection_matrix.
        """
        # 1. Convert the rotation vector to a 3x3 rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # 2. The normal vector of the mirror is the Z-axis of the ArUco marker
        # transformed into the camera coordinate system (3rd column of R).
        normal = R[:, 2]

        # Ensure the normal is a strictly normalized unit vector
        normal = normal / np.linalg.norm(normal)

        # 3. The translation vector (tvec) represents a physical 3D point on the mirror plane
        point_on_plane = tvec.reshape(3)

        # 4. Calculate the scalar distance 'd' for the plane equation: n_x*X + n_y*Y + n_z*Z + d = 0
        # Therefore, d = - dot(normal, point_on_plane)
        d = -np.dot(normal, point_on_plane)

        # 5. Construct the 4x4 Affine Reflection Matrix (Householder transformation)
        # M_ref = [ I - 2*n*n^T   |  -2*d*n ]
        #         [       0       |     1   ]
        M_ref = np.eye(4, dtype=np.float32)
        n = normal.reshape(3, 1)

        # Rotational part (3x3)
        M_ref[:3, :3] = np.eye(3) - 2 * (n @ n.T)

        # Translational part (3x1)
        M_ref[:3, 3] = -2 * d * normal

        return {
            "normal": normal,
            "point_on_plane": point_on_plane,
            "d": float(d),
            "reflection_matrix": M_ref,
        }
