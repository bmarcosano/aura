"""
detector.py: MediaPipe Pose estimation and webcam stream management.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """Handles video capture and real-time 3D pose landmark extraction using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Direct initialization using modern MediaPipe solutions structure
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.results = None
        print("✅ PoseDetector initialized successfully.")

    def detect(self, frame: np.ndarray, draw: bool = True) -> np.ndarray | None:
        """
        Processes via MediaPipe, draws the skeleton overlay,
        and returns the pixel landmark array.
        """
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        self.results = self.pose.process(image_rgb)

        image_rgb.flags.writeable = True

        # Draw skeleton lines on the frame if requested and landmarks are found
        if self.results.pose_landmarks and draw:
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

        return self.get_landmark_array(w, h)

    def find_pose(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """Legacy helper: processes frame and returns the annotated image copy."""
        image = frame.copy()
        self.detect(image, draw=draw)
        return image

    def get_landmark_array(
        self, frame_width: int, frame_height: int
    ) -> np.ndarray | None:
        """Extracts landmark coordinates (x, y in pixels, z, visibility) as a NumPy array."""
        if not self.results or not self.results.pose_landmarks:
            return None

        landmarks = []
        for lm in self.results.pose_landmarks.landmark:
            # Map normalized coordinates back to pixel space for x and y
            landmarks.append(
                [lm.x * frame_width, lm.y * frame_height, lm.z, lm.visibility]
            )

        return np.array(landmarks, dtype=np.float32)

    def get_world_landmark_array(self) -> np.ndarray | None:
        """Return MediaPipe's per-view 3D pose estimate, hip-centred and in metres.

        These are model-estimated body-relative coordinates, not absolute camera
        coordinates and not a calibrated stereo reconstruction.
        """
        if not self.results or not self.results.pose_world_landmarks:
            return None

        landmarks = []
        for lm in self.results.pose_world_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32)

    def get_visibility_array(self) -> np.ndarray | None:
        """Return MediaPipe visibility for all 33 landmarks (legacy-compatible)."""
        if not self.results or not self.results.pose_landmarks:
            return None
        values = [
            float(getattr(lm, "visibility", 0.0))
            for lm in self.results.pose_landmarks.landmark
        ]
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)

    def get_presence_array(self) -> np.ndarray | None:
        """Return MediaPipe presence where available, otherwise visibility.

        MediaPipe versions differ in whether ``presence`` is exposed by the
        normalized pose landmark object. Falling back to visibility keeps the
        confidence contract stable across supported versions.
        """
        if not self.results or not self.results.pose_landmarks:
            return None
        landmarks = self.results.pose_landmarks.landmark
        values = []
        for lm in landmarks:
            visibility = float(getattr(lm, "visibility", 0.0))
            presence = None
            has_field = getattr(lm, "HasField", None)
            if callable(has_field):
                try:
                    if has_field("presence"):
                        presence = float(lm.presence)
                except (ValueError, TypeError):
                    presence = None
            else:
                raw_presence = getattr(lm, "presence", None)
                if raw_presence is not None:
                    presence = float(raw_presence)
            # Proto3 scalar fields can read as 0 when unset. If that happens
            # while visibility is positive, use visibility as the fallback.
            if presence is None or (presence <= 0.0 < visibility):
                presence = visibility
            values.append(presence)
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
