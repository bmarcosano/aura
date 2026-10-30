"""Conservative motion-based proposal for a vertical direct/mirror ROI boundary."""

from __future__ import annotations

import cv2
import numpy as np


class AutoROISplitter:
    """Estimate a split from two large moving blobs; never invent a timeout split.

    Motion centroids are only a proposal. They do not prove that the second blob
    is a mirror reflection; the caller must still validate the two pose tracks
    and mirror geometry before enabling metric reconstruction.
    """

    def __init__(self, required_frames: int = 5, timeout_frames: int = 60) -> None:
        if required_frames < 3 or timeout_frames < required_frames:
            raise ValueError("Invalid splitter frame settings")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.split_ratios: list[float] = []
        self.required_frames = int(required_frames)
        self.timeout_frames = int(timeout_frames)
        self.frame_count = 0
        self.timed_out = False
        self.last_mad = float("inf")

    def update(self, frame: np.ndarray) -> float | None:
        """Return a stable split ratio or None while estimating/when timed out."""
        if frame is None or frame.ndim != 3 or frame.size == 0:
            return None
        if self.is_split_ready():
            return float(np.median(self.split_ratios))
        if self.timed_out:
            return None
        self.frame_count += 1
        if self.frame_count > self.timeout_frames:
            self.timed_out = True
            return None

        height, width = frame.shape[:2]
        mask = self.bg_subtractor.apply(frame, learningRate=-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = width * height * 0.008
        candidates = [c for c in contours if cv2.contourArea(c) >= minimum_area]
        candidates.sort(key=cv2.contourArea, reverse=True)
        if len(candidates) < 2:
            return None

        centers = []
        for contour in candidates[:2]:
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                return None
            centers.append(moments["m10"] / moments["m00"])
        left, right = sorted(centers)
        separation = right - left
        # Avoid accepting nearly coincident blobs or a split at the image edge.
        if separation < 0.12 * width:
            return None
        ratio = (left + right) / (2.0 * width)
        if not 0.15 <= ratio <= 0.85:
            return None
        self.split_ratios.append(float(ratio))
        self.split_ratios = self.split_ratios[-max(self.required_frames * 3, 15) :]
        if len(self.split_ratios) >= self.required_frames:
            values = np.asarray(self.split_ratios[-self.required_frames :])
            median = float(np.median(values))
            self.last_mad = float(np.median(np.abs(values - median)))
            if self.last_mad <= 0.015:
                cv2.line(
                    frame,
                    (int(median * width), 0),
                    (int(median * width), height),
                    (0, 255, 0),
                    2,
                )
        return (
            float(np.median(self.split_ratios[-self.required_frames :]))
            if self.is_split_ready()
            else None
        )

    def is_split_ready(self) -> bool:
        """Require enough readings and low robust dispersion."""
        if len(self.split_ratios) < self.required_frames:
            return False
        values = np.asarray(self.split_ratios[-self.required_frames :])
        median = np.median(values)
        self.last_mad = float(np.median(np.abs(values - median)))
        return self.last_mad <= 0.015
