"""Dual-view frame splitting and synchronized MediaPipe pose observations."""

from __future__ import annotations

import time

import cv2
import numpy as np

from aura.detector import PoseDetector
from aura.pose_types import PoseObservation


class AnthropometricValidator:
    """Legacy metric validator retained for the calibrated reconstruction path."""

    def __init__(self, units: str = "meters") -> None:
        self.scale = 1.0 if units == "meters" else 1000.0
        self.thresholds = {
            "shoulder_width": (0.25, 0.60),
            "femur": (0.35, 0.65),
            "tibia": (0.30, 0.55),
            "torso": (0.40, 0.80),
        }
        self.thresholds = {
            key: (limits[0] * self.scale, limits[1] * self.scale)
            for key, limits in self.thresholds.items()
        }

    def validate_skeleton_3d(self, landmarks_3d: np.ndarray) -> bool:
        try:
            points = np.asarray(landmarks_3d, dtype=np.float32)
            if points.shape != (33, 3) or not np.isfinite(points).all():
                return False
            shoulder_w = np.linalg.norm(points[11] - points[12])
            left_femur = np.linalg.norm(points[23] - points[25])
            left_tibia = np.linalg.norm(points[25] - points[27])
            torso = np.linalg.norm(
                (points[11] + points[12]) * 0.5 - (points[23] + points[24]) * 0.5
            )
            return bool(
                self.thresholds["shoulder_width"][0]
                <= shoulder_w
                <= self.thresholds["shoulder_width"][1]
                and self.thresholds["femur"][0]
                <= left_femur
                <= self.thresholds["femur"][1]
                and self.thresholds["tibia"][0]
                <= left_tibia
                <= self.thresholds["tibia"][1]
                and self.thresholds["torso"][0] <= torso <= self.thresholds["torso"][1]
            )
        except (IndexError, TypeError, ValueError):
            return False


class MultiSkeletonDetector:
    """Run two MediaPipe detectors on synchronized ROIs from one source frame.

    The mirror detector receives a horizontally flipped ROI so MediaPipe sees
    the reflected subject in the same apparent anatomical orientation as the
    front detector. Its 2D pixels are mapped back to the original frame and its
    world X coordinate is unflipped for the estimated alignment pipeline.
    """

    def __init__(self, split_ratio: float = 0.5) -> None:
        if not 0.0 < float(split_ratio) < 1.0:
            raise ValueError("split_ratio must be strictly between 0 and 1")
        self.split_ratio = float(split_ratio)
        self.front_detector = PoseDetector()
        self.mirror_detector = PoseDetector()
        self._frame_counter = 0

    def _crop_rois(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame is None or frame.ndim != 3 or frame.shape[1] < 2:
            raise ValueError("frame must be a non-empty BGR image")
        _, width, _ = frame.shape
        split_x = int(width * self.split_ratio)
        split_x = min(max(split_x, 1), width - 1)
        return frame[:, :split_x], frame[:, split_x:]

    def _get_2d_pixels(self, detector: PoseDetector) -> np.ndarray | None:
        """Extract normalized image coordinates (x, y), preserving the old API."""
        results = getattr(detector, "results", None)
        if not results or not getattr(results, "pose_landmarks", None):
            return None
        return np.asarray(
            [[lm.x, lm.y] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )

    @staticmethod
    def _confidence(
        detector: PoseDetector,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        visibility = None
        presence = None
        get_visibility = getattr(detector, "get_visibility_array", None)
        get_presence = getattr(detector, "get_presence_array", None)
        if callable(get_visibility):
            visibility = get_visibility()
        if callable(get_presence):
            presence = get_presence()
        return visibility, presence

    def _make_observation(
        self,
        detector: PoseDetector,
        view: str,
        frame_id: int,
        timestamp: float,
        landmarks_2d: np.ndarray | None,
        landmarks_3d: np.ndarray | None,
        raw_landmarks_3d: np.ndarray | None,
        image_size: tuple[int, int],
        roi_offset: tuple[int, int],
        input_flipped: bool,
    ) -> PoseObservation:
        visibility, presence = self._confidence(detector)
        results = getattr(detector, "results", None)
        has_2d_pose = bool(results and getattr(results, "pose_landmarks", None))
        has_world_pose = landmarks_3d is not None
        detection_valid = has_2d_pose and has_world_pose and landmarks_2d is not None

        world = (
            None
            if landmarks_3d is None
            else np.asarray(landmarks_3d, dtype=np.float32).copy()
        )
        world_x_unflipped = False
        if view == "mirror" and world is not None and input_flipped:
            # The mirror ROI was flipped before MediaPipe inference. Undo the
            # corresponding sign change in its local horizontal world axis.
            world[:, 0] *= -1.0
            world_x_unflipped = True

        if visibility is None and detection_valid:
            visibility = np.ones(33, dtype=np.float32)
        if presence is None and visibility is not None:
            presence = visibility.copy()

        return PoseObservation(
            view=view,  # type: ignore[arg-type]
            frame_id=frame_id,
            timestamp=timestamp,
            landmarks_2d=landmarks_2d,
            landmarks_3d=world,
            raw_landmarks_3d=raw_landmarks_3d,
            visibility=visibility,
            presence=presence,
            image_size=image_size,
            roi_offset=roi_offset,
            input_flipped=input_flipped,
            world_x_unflipped=world_x_unflipped,
            detection_valid=detection_valid,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> dict:
        """Process both views and return legacy fields plus typed observations.

        ``frame_id`` and ``timestamp`` are optional to preserve existing callers.
        Both observations always receive the same values because they originate
        from the same captured source frame.
        """
        if frame is None or frame.ndim != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")

        self._frame_counter += 1
        current_frame_id = self._frame_counter if frame_id is None else int(frame_id)
        current_timestamp = (
            time.perf_counter() if timestamp is None else float(timestamp)
        )
        height, _width = frame.shape[:2]
        front_roi, mirror_roi = self._crop_rois(frame)
        split_x = front_roi.shape[1]
        flipped_mirror_roi = cv2.flip(mirror_roi, 1)

        logs = [
            (
                "Running MediaPipe Pose detection on dual views.",
                f"Frame {current_frame_id}: Front ROI {front_roi.shape}; Flipped Mirror ROI {flipped_mirror_roi.shape}.",
            )
        ]

        self.front_detector.detect(front_roi, draw=True)
        self.mirror_detector.detect(flipped_mirror_roi, draw=True)
        front_results = getattr(self.front_detector, "results", None)
        mirror_results = getattr(self.mirror_detector, "results", None)
        annotated_mirror_roi = cv2.flip(flipped_mirror_roi, 1)

        front_2d = self._get_2d_pixels(self.front_detector)
        mirror_2d = self._get_2d_pixels(self.mirror_detector)

        if front_2d is not None:
            front_2d[:, 0] *= front_roi.shape[1]
            front_2d[:, 1] *= height

        if mirror_2d is not None:
            roi_width = flipped_mirror_roi.shape[1]
            mirror_2d[:, 0] *= roi_width
            mirror_2d[:, 1] *= height
            # Undo input flip, then translate from mirror ROI to full-frame pixels.
            mirror_2d[:, 0] = (roi_width - 1) - mirror_2d[:, 0]
            mirror_2d[:, 0] += split_x

        front_world = self.front_detector.get_world_landmark_array()
        mirror_world = self.mirror_detector.get_world_landmark_array()

        front_observation = self._make_observation(
            detector=self.front_detector,
            view="front",
            frame_id=current_frame_id,
            timestamp=current_timestamp,
            landmarks_2d=front_2d,
            landmarks_3d=front_world,
            raw_landmarks_3d=front_world,
            image_size=(front_roi.shape[1], front_roi.shape[0]),
            roi_offset=(0, 0),
            input_flipped=False,
        )
        mirror_observation = self._make_observation(
            detector=self.mirror_detector,
            view="mirror",
            frame_id=current_frame_id,
            timestamp=current_timestamp,
            landmarks_2d=mirror_2d,
            landmarks_3d=mirror_world,
            raw_landmarks_3d=mirror_world,
            image_size=(flipped_mirror_roi.shape[1], flipped_mirror_roi.shape[0]),
            roi_offset=(split_x, 0),
            input_flipped=True,
        )

        # Keep all historical keys/signatures. New fusion code consumes the
        # typed observations; existing diagnostic scripts can keep using arrays.
        return {
            "front_roi": front_roi,
            "mirror_roi": annotated_mirror_roi,
            "front_results": front_results,
            "mirror_results": mirror_results,
            "direct_pose_2d": front_2d,
            "reflected_pose_2d": mirror_2d,
            "front_world_landmarks": front_world,
            "mirror_world_landmarks": mirror_world,
            "front_observation": front_observation,
            "mirror_observation": mirror_observation,
            "frame_id": current_frame_id,
            "timestamp": current_timestamp,
            "is_valid": True,
            "logs": logs,
        }
