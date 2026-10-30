"""MediaPipe Pose adapter with explicit pixel and hip-centred coordinate APIs."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PoseDetector:
    """Wrap the legacy MediaPipe ``solutions.pose`` API.

    ``get_landmark_array`` returns image pixels in columns X/Y, model-relative
    Z, and visibility. ``get_world_landmark_array`` returns MediaPipe's
    hip-centred model coordinates in metres; those are not world/camera
    coordinates and must not be used as absolute calibration observations.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except (
            ImportError
        ) as exc:  # Keep non-vision modules importable without MediaPipe.
            raise RuntimeError(
                "MediaPipe is required to create PoseDetector. Install the project dependencies."
            ) from exc

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This AURA version requires MediaPipe's solutions.pose API."
            )
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.results: Any | None = None
        self._last_image_size: tuple[int, int] | None = None
        logger.info("PoseDetector initialized")

    def detect(self, frame: np.ndarray, draw: bool = True) -> np.ndarray | None:
        """Run pose inference and return ``(33, 4)`` image-pixel landmarks."""
        if frame is None or frame.ndim != 3 or frame.size == 0:
            self.results = None
            return None
        height, width = frame.shape[:2]
        self._last_image_size = (width, height)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            self.results = self.pose.process(rgb)
        finally:
            rgb.flags.writeable = True

        if draw and self.results and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                self.mp_draw.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
            )
        return self.get_landmark_array(width, height)

    def find_pose(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """Legacy helper that returns an annotated copy of the input frame."""
        image = frame.copy()
        self.detect(image, draw=draw)
        return image

    def get_landmark_array(
        self, frame_width: int, frame_height: int
    ) -> np.ndarray | None:
        """Return ``(33, 4)``: X/Y pixels, relative Z, and visibility."""
        if not self.results or not self.results.pose_landmarks:
            return None
        rows = [
            [lm.x * frame_width, lm.y * frame_height, lm.z, lm.visibility]
            for lm in self.results.pose_landmarks.landmark
        ]
        return np.asarray(rows, dtype=np.float32)

    def get_world_landmark_array(self) -> np.ndarray | None:
        """Return MediaPipe's hip-centred 3D pose estimate in metres (not camera XYZ)."""
        if not self.results or not self.results.pose_world_landmarks:
            return None
        rows = [[lm.x, lm.y, lm.z] for lm in self.results.pose_world_landmarks.landmark]
        return np.asarray(rows, dtype=np.float32)

    def get_visibility_array(self) -> np.ndarray | None:
        """Return one visibility value per landmark, or None when no pose is detected."""
        if not self.results or not self.results.pose_landmarks:
            return None
        return np.asarray(
            [lm.visibility for lm in self.results.pose_landmarks.landmark],
            dtype=np.float32,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if getattr(self, "pose", None) is not None:
            self.pose.close()
            self.pose = None
