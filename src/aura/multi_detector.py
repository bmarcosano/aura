"""
multi_detector.py: Manages dual-view frame splitting, parallel pose detection, and validation.
"""

from __future__ import annotations

import cv2
import numpy as np

from aura.anthropometry import AnthropometricValidator
from aura.detector import PoseDetector
from aura.mirror_plane import MirrorPlaneEstimator


class MultiSkeletonDetector:
    """
    Splits a dual-view frame, runs MediaPipe detection, estimates the mirror plane
    via SVD, and validates the mirror skeleton's anthropometric integrity.
    """

    def __init__(self, split_ratio: float = 0.5, error_threshold: float = 0.10):
        """
        Initializes the MultiSkeletonDetector with geometric validation components.

        Args:
            split_ratio: Horizontal ratio to split the frame (default 0.5).
            error_threshold: Max allowed bone-length discrepancy (default 10%).
        """
        self.split_ratio = split_ratio

        self.front_detector = PoseDetector()
        self.mirror_detector = PoseDetector()

        self.plane_estimator = MirrorPlaneEstimator(history_size=90)
        self.validator = AnthropometricValidator(error_threshold=error_threshold)

    def _crop_rois(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crops the frame into front-view ROI and mirror-view ROI based on split ratio."""
        _, w, _ = frame.shape
        split_x = int(w * self.split_ratio)
        return frame[:, :split_x], frame[:, split_x:]

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Processes the dual view, extracts landmarks, runs SVD calibration,
        and applies anthropometric filtering if the mirror is calibrated.
        """
        front_roi, mirror_roi = self._crop_rois(frame)
        flipped_mirror_roi = cv2.flip(mirror_roi, 1)

        # 1. Run MediaPipe Detection
        self.front_detector.detect(front_roi, draw=True)
        self.mirror_detector.detect(flipped_mirror_roi, draw=True)

        front_results = getattr(self.front_detector, "results", None)
        mirror_results = getattr(self.mirror_detector, "results", None)

        is_valid = True

        # 2. Extract Real-World 3D Landmarks for Geometry Engine
        if front_results and mirror_results:
            front_world = self.front_detector.get_world_landmark_array()
            mirror_world = self.mirror_detector.get_world_landmark_array()

            if front_world is not None and mirror_world is not None:
                # 3. Feed SVD Calibration Engine
                buffer_ready = self.plane_estimator.add_frame_data(
                    front_world, mirror_world
                )

                if buffer_ready and not self.plane_estimator.is_calibrated_state():
                    self.plane_estimator.estimate()
                    print("✅ Mirror Plane Calibrated via SVD!")

                # 4. Anthropometric Validation (Only if calibrated)
                if self.plane_estimator.is_calibrated_state():
                    try:
                        reflection_matrix = self.plane_estimator.get_reflection_matrix()
                        rectified_mirror = self.validator.apply_mirror_reflection(
                            mirror_world, reflection_matrix
                        )

                        is_valid, _ = self.validator.validate_isometry(
                            front_world, rectified_mirror
                        )

                        if not is_valid:
                            # 🛑 FALLBACK: Drop hallucinated mirror data for this frame
                            print("⚠️ Anthropometric mismatch! Dropping mirror data.")
                            mirror_results = None

                    except (ValueError, RuntimeError) as e:
                        print(f"⚠️ Validation Error: {e}")
                        mirror_results = None
                        is_valid = False

        return {
            "front_roi": front_roi,
            "mirror_roi": mirror_roi,
            "front_results": front_results,
            "mirror_results": mirror_results,
            "is_valid": is_valid,
        }
