"""Two-ROI pose inference for the direct and mirror-reflected image regions."""

from __future__ import annotations

import logging

import numpy as np

from aura.anthropometry import AnthropometricValidator  # Legacy import path.
from aura.detector import PoseDetector

logger = logging.getLogger(__name__)


class MultiSkeletonDetector:
    """Run two independent PoseDetectors and map both outputs to full-frame pixels.

    The left region is treated as the direct view and the right region as the
    mirror view. The mirror ROI is horizontally flipped only for pose inference;
    its landmarks are then unflipped and translated back to the original frame.
    The split ratio must be supplied by a reliable/assisted splitter.
    """

    def __init__(self, split_ratio: float = 0.5) -> None:
        if not 0.05 <= float(split_ratio) <= 0.95:
            raise ValueError("split_ratio must be between 0.05 and 0.95")
        self.split_ratio = float(split_ratio)
        self.front_detector = PoseDetector()
        self.mirror_detector = PoseDetector()
        self.validator = AnthropometricValidator(units="meters")
        self.last_frame_size: tuple[int, int] | None = None

    def _crop_rois(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a BGR frame into non-empty left/direct and right/mirror ROIs."""
        if frame is None or frame.ndim != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")
        _height, width = frame.shape[:2]
        split_x = round(width * self.split_ratio)
        split_x = min(max(split_x, 1), width - 1)
        return frame[:, :split_x].copy(), frame[:, split_x:].copy()

    @staticmethod
    def map_mirror_pixels_to_full_frame(
        flipped_roi_points: np.ndarray, roi_width: int, split_x: int
    ) -> np.ndarray:
        """Undo ROI flip and restore the crop offset to full-frame pixel coordinates."""
        points = np.asarray(flipped_roi_points, dtype=np.float64).copy()
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("flipped_roi_points must have shape (N,2)")
        if roi_width < 1 or split_x < 0:
            raise ValueError("Invalid ROI geometry")
        points[:, 0] = (roi_width - 1) - points[:, 0] + split_x
        return points

    def _get_2d_pixels(self, detector: PoseDetector) -> np.ndarray | None:
        """Return the detector's most recent image-pixel landmark array."""
        if detector._last_image_size is None:
            return None
        width, height = detector._last_image_size
        return detector.get_landmark_array(width, height)

    def process_frame(self, frame: np.ndarray) -> dict:
        """Infer both ROIs and return legacy keys plus explicit geometry metadata."""
        front_roi, mirror_roi = self._crop_rois(frame)
        height, full_width = frame.shape[:2]
        split_x = front_roi.shape[1]
        mirror_inference_roi = np.ascontiguousarray(mirror_roi[:, ::-1])

        front_raw = self.front_detector.detect(front_roi, draw=True)
        mirror_raw = self.mirror_detector.detect(mirror_inference_roi, draw=True)
        front_results = self.front_detector.results
        mirror_results = self.mirror_detector.results

        direct_2d = None
        reflected_2d = None
        direct_visibility = None
        reflected_visibility = None
        if front_raw is not None:
            direct_2d = front_raw[:, :2].astype(np.float64)
            direct_visibility = front_raw[:, 3].astype(np.float32)
            # The direct ROI starts at x=0, so local and full-frame X match.
        if mirror_raw is not None:
            # Convert from flipped ROI pixels to original full-frame pixels.
            reflected_2d = self.map_mirror_pixels_to_full_frame(
                mirror_raw[:, :2], mirror_roi.shape[1], split_x
            )
            reflected_visibility = mirror_raw[:, 3].astype(np.float32)

        # A valid pair means only that usable observations exist; it does not
        # assert that the mirror plane or 3D reconstruction has been validated.
        is_valid = bool(
            direct_2d is not None
            and reflected_2d is not None
            and direct_2d.shape == reflected_2d.shape == (33, 2)
            and np.isfinite(direct_2d).all()
            and np.isfinite(reflected_2d).all()
            and int(np.sum(direct_visibility >= 0.35)) >= 12
            and int(np.sum(reflected_visibility >= 0.35)) >= 12
        )

        logs: list[tuple[str, str]] = []
        if not is_valid:
            logs.append(
                (
                    "Both views are not currently reliable.",
                    "ROI observations missing or too many low-visibility landmarks.",
                )
            )
        return {
            "front_roi": front_roi,
            "mirror_roi": mirror_roi,
            "front_results": front_results,
            "mirror_results": mirror_results,
            "direct_pose_2d": direct_2d,
            "reflected_pose_2d": reflected_2d,
            "direct_visibility": direct_visibility,
            "reflected_visibility": reflected_visibility,
            "split_x": split_x,
            "frame_size": (full_width, height),
            "is_valid": is_valid,
            "logs": logs,
        }

    def close(self) -> None:
        """Release both MediaPipe graphs."""
        self.front_detector.close()
        self.mirror_detector.close()


__all__ = ["AnthropometricValidator", "MultiSkeletonDetector"]
