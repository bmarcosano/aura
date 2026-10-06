from __future__ import annotations

"""detector.py: MediaPipe Pose estimation and webcam stream management."""

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
    ):
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

    def detect(self, frame: np.ndarray, draw: bool = True) -> np.ndarray | None:
        """Processes an OpenCV frame, draws the skeleton overlay, and returns the landmark array."""
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
        """Extracts landmark coordinates (x, y, z, visibility) as a structured NumPy array."""
        if not self.results or not self.results.pose_landmarks:
            return None

        landmarks = []
        for lm in self.results.pose_landmarks.landmark:
            # Map normalized coordinates back to pixel space for x and y; keep z and visibility
            landmarks.append(
                [lm.x * frame_width, lm.y * frame_height, lm.z, lm.visibility]
            )

        return np.array(landmarks, dtype=np.float32)
