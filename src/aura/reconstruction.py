"""Mirror-assisted 3D triangulation and a compatibility-only landmark fusion path.

The reflected view is modelled as a virtual camera. For a mirror plane
n.T @ X + d = 0, the reflection is X_ref = H @ X + t, where
H = I - 2*n*n.T and t = -2*d*n. H is an orthogonal reflection (det=-1),
not a physical camera rotation; it is nevertheless a valid projective mapping.
"""

from __future__ import annotations

import logging
import warnings

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoReconstructor:
    """Triangulate landmarks from direct and mirror-reflected image observations.

    Input pixel coordinates must refer to the full camera image (not local ROI
    coordinates). Both views must use the same calibrated camera intrinsics and
    the same image-coordinate convention. If images were already undistorted,
    pass ``dist_coeffs=None``.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        K = np.asarray(camera_matrix, dtype=np.float64)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or abs(np.linalg.det(K)) < 1e-12
        ):
            raise ValueError("camera_matrix must be a finite, invertible 3x3 matrix")
        self.camera_matrix = K
        self.dist_coeffs = (
            None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
        )
        self.P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.last_valid_mask = np.zeros(0, dtype=bool)
        self.last_reprojection_errors = np.empty((0, 2), dtype=np.float64)
        self.last_diagnostics: dict[str, float | int | str] = {}

    def _prepare_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort pixel observations when raw (distorted) frames are used."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("2D landmarks must have shape (N, 2)")
        if self.dist_coeffs is not None and np.any(np.abs(self.dist_coeffs) > 1e-12):
            finite = np.isfinite(pts).all(axis=1)
            safe_points = pts.copy()
            # OpenCV should never receive NaN/Inf input, even for landmarks later rejected.
            safe_points[~finite] = 0.0
            corrected = cv2.undistortPoints(
                safe_points.astype(np.float64).reshape(-1, 1, 2),
                self.camera_matrix,
                self.dist_coeffs,
                P=self.camera_matrix,
            ).reshape(-1, 2)
            corrected[~finite] = np.nan
            pts = corrected
        return pts

    @staticmethod
    def _projection_matrix(
        reflection_matrix: np.ndarray, camera_matrix: np.ndarray
    ) -> np.ndarray:
        """Build P2 = K [H | t] from a 4x4 affine reflection or a 3x4 mapping."""
        transform = np.asarray(reflection_matrix, dtype=np.float64)
        if transform.shape == (4, 4):
            extrinsics = transform[:3, :4]
        elif transform.shape == (3, 4):
            extrinsics = transform
        else:
            raise ValueError("reflection_matrix must have shape (4,4) or (3,4)")
        if not np.isfinite(extrinsics).all():
            raise ValueError("reflection_matrix contains non-finite values")
        if transform.shape == (4, 4):
            if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
                raise ValueError(
                    "The 4x4 transform must be affine with last row [0,0,0,1]"
                )
            H = transform[:3, :3]
            # The linear component must be an orthogonal reflection (or a numerically close estimate).
            if not np.allclose(H.T @ H, np.eye(3), atol=2e-2) or not np.isclose(
                np.linalg.det(H), -1.0, atol=2e-2
            ):
                raise ValueError(
                    "The 3x3 linear component is not a valid Householder reflection"
                )
        return camera_matrix @ extrinsics

    def _check_reprojection_error(
        self,
        pt3d: np.ndarray,
        pt2d_direct: np.ndarray,
        pt2d_reflected: np.ndarray,
        P2: np.ndarray,
        max_error_px: float = 15.0,
    ) -> bool:
        """Return True only when both projections are finite, in front, and close."""
        point = np.r_[np.asarray(pt3d, dtype=np.float64).reshape(3), 1.0]
        q1, q2 = self.P1 @ point, P2 @ point
        if q1[2] <= 1e-8 or q2[2] <= 1e-8:
            return False
        p1, p2 = q1[:2] / q1[2], q2[:2] / q2[2]
        err1 = float(np.linalg.norm(p1 - pt2d_direct))
        err2 = float(np.linalg.norm(p2 - pt2d_reflected))
        return bool(
            np.isfinite(err1 + err2) and err1 <= max_error_px and err2 <= max_error_px
        )

    def triangulate_landmarks(
        self,
        points_2d_direct: np.ndarray,
        points_2d_reflected: np.ndarray,
        reflection_matrix: np.ndarray,
        max_error_px: float = 15.0,
        direct_confidence: np.ndarray | None = None,
        reflected_confidence: np.ndarray | None = None,
        min_confidence: float = 0.35,
    ) -> np.ndarray:
        """Triangulate corresponding full-frame pixel landmarks.

        The legacy first three arguments are preserved. Invalid points are NaN
        in the returned ``(N,3)`` array and marked False in ``last_valid_mask``.
        This avoids silently presenting rejected observations as valid 3D data.
        """
        raw1 = np.asarray(points_2d_direct, dtype=np.float64)
        raw2 = np.asarray(points_2d_reflected, dtype=np.float64)
        if raw1.ndim != 2 or raw1.shape[1] != 2 or raw1.shape != raw2.shape:
            raise ValueError("Both 2D landmark arrays must have identical shape (N,2)")
        if raw1.shape[0] == 0:
            self.last_valid_mask = np.zeros(0, dtype=bool)
            self.last_reprojection_errors = np.empty((0, 2), dtype=np.float64)
            return np.empty((0, 3), dtype=np.float64)

        pts1, pts2 = self._prepare_points(raw1), self._prepare_points(raw2)
        P2 = self._projection_matrix(reflection_matrix, self.camera_matrix)
        n = len(pts1)
        conf1 = (
            np.ones(n)
            if direct_confidence is None
            else np.asarray(direct_confidence, dtype=float).reshape(-1)
        )
        conf2 = (
            np.ones(n)
            if reflected_confidence is None
            else np.asarray(reflected_confidence, dtype=float).reshape(-1)
        )
        if conf1.size != n or conf2.size != n:
            raise ValueError("Confidence arrays must contain one value per landmark")

        output = np.full((n, 3), np.nan, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        errors = np.full((n, 2), np.inf, dtype=np.float64)
        finite_observations = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(
            axis=1
        )
        confidence_ok = (conf1 >= min_confidence) & (conf2 >= min_confidence)

        indices = np.flatnonzero(finite_observations & confidence_ok)
        if indices.size:
            homogeneous = cv2.triangulatePoints(
                self.P1.astype(np.float64),
                P2.astype(np.float64),
                pts1[indices].T.astype(np.float64),
                pts2[indices].T.astype(np.float64),
            )
            for column, index in enumerate(indices):
                w = float(homogeneous[3, column])
                if not np.isfinite(w) or abs(w) < 1e-10:
                    continue
                point = homogeneous[:3, column] / w
                if not np.isfinite(point).all():
                    continue
                point_h = np.r_[point, 1.0]
                q1, q2 = self.P1 @ point_h, P2 @ point_h
                if q1[2] <= 1e-8 or q2[2] <= 1e-8:
                    continue
                reproj1, reproj2 = q1[:2] / q1[2], q2[:2] / q2[2]
                errors[index] = [
                    np.linalg.norm(reproj1 - pts1[index]),
                    np.linalg.norm(reproj2 - pts2[index]),
                ]
                if max(errors[index]) <= max_error_px:
                    output[index] = point
                    valid[index] = True

        self.last_valid_mask = valid
        self.last_reprojection_errors = errors
        accepted = int(valid.sum())
        finite_err = errors[valid]
        mean_error = float(np.mean(finite_err)) if finite_err.size else float("inf")
        self.last_diagnostics = {
            "accepted_points": accepted,
            "total_points": n,
            "mean_reprojection_error_px": mean_error,
            "status": "VALID" if accepted else "INVALID",
        }
        if accepted < max(1, int(np.ceil(0.5 * n))):
            logger.warning(
                "Low triangulation yield: %d/%d landmarks accepted", accepted, n
            )
        return output

    def fuse_landmarks(self, *args, **kwargs) -> np.ndarray:
        """Legacy confidence-weighted fusion of two already aligned 3D arrays.

        This method is retained for API compatibility only. Its output is not a
        calibrated camera-space reconstruction. New mirror code must call
        ``triangulate_landmarks`` with 2D image observations.
        """
        warnings.warn(
            "fuse_landmarks() is a legacy model-space blend; use triangulate_landmarks() for geometry.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Keep the historical variadic API while accepting descriptive keyword names.
        front_landmarks = (
            args[0] if len(args) > 0 else kwargs.pop("front_landmarks", None)
        )
        rectified_mirror_landmarks = (
            args[1] if len(args) > 1 else kwargs.pop("rectified_mirror_landmarks", None)
        )
        front_visibility = (
            args[2] if len(args) > 2 else kwargs.pop("front_visibility", None)
        )
        mirror_visibility = (
            args[3] if len(args) > 3 else kwargs.pop("mirror_visibility", None)
        )
        is_anthropometrically_valid = (
            args[4]
            if len(args) > 4
            else kwargs.pop("is_anthropometrically_valid", True)
        )
        if front_landmarks is None:
            raise TypeError("front_landmarks is required")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        front = np.asarray(front_landmarks, dtype=np.float64)
        if front.ndim != 2 or front.shape[1] != 3:
            raise ValueError("front_landmarks must have shape (N,3)")
        if rectified_mirror_landmarks is None or not is_anthropometrically_valid:
            return front.copy()
        mirror = np.asarray(rectified_mirror_landmarks, dtype=np.float64)
        if mirror.shape != front.shape:
            raise ValueError("Both legacy 3D arrays must have identical shape")
        wf = (
            np.ones(len(front))
            if front_visibility is None
            else np.asarray(front_visibility, dtype=float).reshape(-1)
        )
        wm = (
            np.ones(len(front))
            if mirror_visibility is None
            else np.asarray(mirror_visibility, dtype=float).reshape(-1)
        )
        if wf.size != len(front) or wm.size != len(front):
            raise ValueError("Visibility arrays must contain one value per landmark")
        wf = np.clip(np.nan_to_num(wf), 0.0, 1.0)
        wm = np.clip(np.nan_to_num(wm), 0.0, 1.0)
        denominator = wf + wm
        result = front.copy()
        good = (
            np.isfinite(front).all(axis=1)
            & np.isfinite(mirror).all(axis=1)
            & (denominator > 1e-8)
        )
        result[good] = (
            front[good] * wf[good, None] + mirror[good] * wm[good, None]
        ) / denominator[good, None]
        result[~np.isfinite(front).all(axis=1)] = np.nan
        return result
