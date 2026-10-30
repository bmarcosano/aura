"""ArUco detection, pose estimation, and mirror-plane reflection geometry."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ArucoMultiTracker:
    """Track square ArUco markers and estimate their poses relative to the camera.

    The legacy ``process_frame`` return type is preserved: ``(annotated_frame,
    {marker_id: (rvec, tvec)})``. Marker corners must lie on the mirror plane
    (or have a known, compensated offset) for metric mirror calibration.
    """

    def __init__(
        self,
        dictionary_id: int = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else 0,
        marker_size_m: float = 0.18,
    ) -> None:
        if marker_size_m <= 0:
            raise ValueError("marker_size_m must be positive")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV ArUco module is unavailable; install opencv-contrib-python"
            )
        self.marker_size = float(marker_size_m)
        self.dictionary_id = int(dictionary_id)
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:  # OpenCV 4.8 compatibility.
            self.parameters = cv2.aruco.DetectorParameters_create()
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.last_reprojection_errors: dict[int, float] = {}

    def _detect(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

    def _solve_square_pose(
        self, corners: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Select the lowest-error positive-depth IPPE solution for one marker."""
        half = self.marker_size / 2.0
        # Required point order for SOLVEPNP_IPPE_SQUARE.
        object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        candidates = []
        if hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            try:
                result = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if result and result[0]:
                    for rvec, tvec in zip(result[1], result[2]):
                        candidates.append((rvec, tvec))
            except cv2.error:
                candidates = []
        if not candidates:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=getattr(cv2, "SOLVEPNP_ITERATIVE", 0),
            )
            if ok:
                candidates.append((rvec, tvec))

        best = None
        for rvec, tvec in candidates:
            if float(tvec.reshape(3)[2]) <= 0:
                continue
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            error = float(
                np.sqrt(
                    np.mean(
                        np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1)
                    )
                )
            )
            if np.isfinite(error) and (best is None or error < best[2]):
                best = (
                    np.asarray(rvec, dtype=np.float64),
                    np.asarray(tvec, dtype=np.float64),
                    error,
                )
        return best

    def process_frame(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
        """Detect markers and return their camera-relative Rodrigues poses."""
        if frame is None or frame.size == 0:
            return frame, {}
        K = np.asarray(camera_matrix, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        D = (
            np.zeros((5, 1), dtype=np.float64)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64)
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.last_reprojection_errors = {}
        if ids is None:
            return frame, poses

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for corner_array, marker_id in zip(corners, ids.flatten()):
            solved = self._solve_square_pose(corner_array, K, D)
            if solved is None:
                continue
            rvec, tvec, error = solved
            marker_id = int(marker_id)
            poses[marker_id] = (rvec, tvec)
            self.last_reprojection_errors[marker_id] = error
            cv2.drawFrameAxes(frame, K, D, rvec, tvec, self.marker_size * 0.5)
            distance_cm = float(np.linalg.norm(tvec)) * 100.0
            anchor = tuple(np.asarray(corner_array).reshape(4, 2)[0].astype(int))
            cv2.putText(
                frame,
                f"ID {marker_id}: {distance_cm:.1f} cm / {error:.1f}px",
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return frame, poses

    def estimate_mirror_plane(
        self, rvec: np.ndarray, tvec: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Return mirror plane and 4x4 affine reflection from a coplanar marker pose."""
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        normal = rotation[:, 2].astype(np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            raise ValueError("Degenerate marker orientation")
        normal /= norm
        point = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            raise ValueError("Marker translation contains non-finite values")
        d = -float(np.dot(normal, point))
        H = np.eye(3) - 2.0 * np.outer(normal, normal)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = H
        transform[:3, 3] = -2.0 * d * normal
        return {
            "normal": normal,
            "point_on_plane": point,
            "d": d,
            "reflection_matrix": transform,
        }
