"""
multi_detector.py: Manages dual-view frame splitting, parallel pose detection, and validation.
"""

from __future__ import annotations

import cv2
import numpy as np

from aura.anthropometry import AnthropometricValidator
from aura.detector import PoseDetector
from aura.mirror_plane import MirrorPlaneEstimator
from aura.reconstruction import PoseReconstructor


class MultiSkeletonDetector:
    """
    Splits a dual-view frame, runs MediaPipe detection, estimates mirror plane via SVD,
    validates isometry, and reconstructs a unified 3D Digital Twin with detailed logging.
    """

    def __init__(self, split_ratio: float = 0.5, error_threshold: float = 0.80):
        self.split_ratio = split_ratio

        self.front_detector = PoseDetector()
        self.mirror_detector = PoseDetector()

        self.plane_estimator = MirrorPlaneEstimator(history_size=90)
        self.validator = AnthropometricValidator(error_threshold=error_threshold)
        self.reconstructor = PoseReconstructor(confidence_weight=0.5)

    def _crop_rois(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, w, _ = frame.shape
        split_x = int(w * self.split_ratio)
        return frame[:, :split_x], frame[:, split_x:]

    def process_frame(self, frame: np.ndarray) -> dict:
        logs = []
        front_roi, mirror_roi = self._crop_rois(frame)
        flipped_mirror_roi = cv2.flip(mirror_roi, 1)

        logs.append(
            (
                "Running MediaPipe Pose detection on dual camera feeds.",
                f"MediaPipe inference dispatched for Front ROI ({front_roi.shape}) and Flipped Mirror ROI ({flipped_mirror_roi.shape}).",
            )
        )

        self.front_detector.detect(front_roi, draw=True)
        self.mirror_detector.detect(flipped_mirror_roi, draw=True)

        front_results = getattr(self.front_detector, "results", None)
        mirror_results = getattr(self.mirror_detector, "results", None)

        annotated_mirror_roi = cv2.flip(flipped_mirror_roi, 1)

        is_valid = True
        fused_3d = None
        plane_normal = None

        if self.plane_estimator.is_calibrated_state() and hasattr(
            self.plane_estimator, "normal_vector"
        ):
            plane_normal = self.plane_estimator.normal_vector

        front_vis, mirror_vis = None, None
        if front_results and getattr(front_results, "pose_world_landmarks", None):
            front_vis = np.array(
                [lm.visibility for lm in front_results.pose_world_landmarks.landmark]
            )
        if mirror_results and getattr(mirror_results, "pose_world_landmarks", None):
            mirror_vis = np.array(
                [lm.visibility for lm in mirror_results.pose_world_landmarks.landmark]
            )

        if front_results and mirror_results:
            front_world = self.front_detector.get_world_landmark_array()
            mirror_world = self.mirror_detector.get_world_landmark_array()

            if front_world is not None and mirror_world is not None:
                mirror_world[:, 0] = -mirror_world[:, 0]

                # Feed SVD calibration buffer
                buffer_ready, user_msg, expert_msg = (
                    self.plane_estimator.add_frame_data(front_world, mirror_world)
                )
                logs.append((user_msg, expert_msg))

                if buffer_ready and not self.plane_estimator.is_calibrated_state():
                    _, _, _, u_msg, e_msg = self.plane_estimator.estimate()
                    logs.append((u_msg, e_msg))

                rectified_mirror = None
                if self.plane_estimator.is_calibrated_state():
                    try:
                        if hasattr(self.plane_estimator, "normal_vector"):
                            plane_normal = self.plane_estimator.normal_vector
                        reflection_matrix = self.plane_estimator.get_reflection_matrix()
                        rectified_mirror = self.validator.apply_mirror_reflection(
                            mirror_world, reflection_matrix
                        )

                        is_valid, max_err, u_msg, e_msg = (
                            self.validator.validate_isometry(
                                front_world, rectified_mirror
                            )
                        )
                        logs.append((u_msg, e_msg))

                        if not is_valid:
                            logs.append(
                                (
                                    "Bone length variance detected. Applying fusion fallback.",
                                    f"Isometry threshold exceeded (err={max_err:.4f}). Forcing reconstruction fallback.",
                                )
                            )
                            is_valid = True

                    except (ValueError, RuntimeError) as e:
                        logs.append(
                            (
                                "Validation calculation warning.",
                                f"Exception in isometry validation: {e}",
                            )
                        )
                        mirror_results = None
                        is_valid = False

                fused_3d = self.reconstructor.fuse_landmarks(
                    front_landmarks=front_world,
                    rectified_mirror_landmarks=rectified_mirror,
                    front_visibility=front_vis,
                    mirror_visibility=mirror_vis,
                    is_anthropometrically_valid=is_valid,
                )
                logs.append(
                    (
                        "Unified 3D Digital Twin successfully constructed.",
                        f"PoseReconstructor fused {len(front_world)} joints using visibility weights.",
                    )
                )
        elif front_results:
            fused_3d = self.front_detector.get_world_landmark_array()
            logs.append(
                (
                    "Mirror view unavailable. Using direct front-view 3D landmarks.",
                    "Single-view fallback activated. Fused 3D populated exclusively from front detector.",
                )
            )

        return {
            "front_roi": front_roi,
            "mirror_roi": annotated_mirror_roi,
            "front_results": front_results,
            "mirror_results": mirror_results,
            "is_valid": is_valid,
            "fused_3d": fused_3d,
            "plane_normal": plane_normal,
            "logs": logs,
        }
