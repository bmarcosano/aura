"""multi_detector.py: Manages dual-view frame splitting and parallel pose detection."""

from __future__ import annotations

import cv2
import numpy as np

from aura.detector import PoseDetector


class MultiSkeletonDetector:
    """Splits a dual-view frame into ROIs and processes them via independent detectors."""

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

        # Apply horizontal flip to the mirror ROI so MediaPipe analyzes it properly
        flipped_mirror_roi = cv2.flip(mirror_roi, 1)

        # Process both ROIs using PoseDetector (which handles detection and landmark extraction)
        front_landmarks = self.front_detector.detect(front_roi, draw=True)
        mirror_landmarks = self.mirror_detector.detect(flipped_mirror_roi, draw=True)

        return {"front": front_landmarks, "mirror": mirror_landmarks}
