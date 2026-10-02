from __future__ import annotations

import cv2
import numpy as np

from aura.detector import PoseDetector


class MultiSkeletonDetector:
    def __init__(self, split_ratio: float = 0.5):
        """
        Initializes the MultiSkeletonDetector.
        :param split_ratio: Horizontal ratio to split the frame (default 0.5 -> left/right halves)
        """
        self.split_ratio = split_ratio
        self.front_detector = PoseDetector()
        self.mirror_detector = PoseDetector()

    def _crop_rois(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crops the frame into front-view ROI and mirror-view ROI based on split ratio."""
        _, w, _ = frame.shape
        split_x = int(w * self.split_ratio)

        # Left side: Front view
        front_roi = frame[:, :split_x]

        # Right side: Mirror view
        mirror_roi = frame[:, split_x:]

        return front_roi, mirror_roi

    def detect_dual_view(self, frame: np.ndarray) -> dict:
        """
        Processes a single frame, extracts ROIs, applies mirror flip,
        and runs MediaPipe detection on both views.
        """
        front_roi, mirror_roi = self._crop_rois(frame)

        # Apply horizontal flip to the mirror ROI so MediaPipe analyzes it as facing forward
        flipped_mirror_roi = cv2.flip(mirror_roi, 1)

        # Run detection (assuming PoseDetector returns processed frame and landmarks)
        # Note: We will integrate actual PoseDetector calls here
        front_landmarks = self.front_detector.get_landmark_array(
            front_roi
        )  # or equivalent method
        mirror_landmarks = self.mirror_detector.get_landmark_array(flipped_mirror_roi)

        return {"front": front_landmarks, "mirror": mirror_landmarks}
