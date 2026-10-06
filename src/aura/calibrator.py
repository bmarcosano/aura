"""Module for automatic spatial calibration using motion detection."""

from __future__ import annotations

import cv2
import numpy as np


class AutoSplitCalibrator:
    """
    Calibrates the split ratio by analyzing synchronized movements.
    Includes a debug view to visualize the motion mask.
    """

    def __init__(self, required_frames: int = 5, timeout_frames: int = 60):
        # learningRate=-1 lets the algorithm automatically decide the background update speed
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.split_ratios: list[float] = []
        self.required_frames = required_frames
        self.timeout_frames = timeout_frames
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> float | None:
        """
        Analyzes the frame for motion. Returns the averaged split ratio
        once enough valid frames are collected or if a timeout occurs.
        """
        if self.is_calibrated():
            # Ensure debug window is closed when done
            if cv2.getWindowProperty("AURA - Debug Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("AURA - Debug Mask")
            return float(np.mean(self.split_ratios))

        self.frame_count += 1

        if self.frame_count > self.timeout_frames:
            print("⚠️ Calibration timeout reached. Forcing split ratio.")
            if cv2.getWindowProperty("AURA - Debug Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("AURA - Debug Mask")
            return float(np.mean(self.split_ratios)) if self.split_ratios else 0.5

        h, w = frame.shape[:2]

        # Apply MOG2 to get the motion mask
        mask = self.bg_subtractor.apply(frame, learningRate=-1)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)

        # 🛑 DEBUG: Show what the algorithm is actually seeing
        # cv2.imshow("AURA - Debug Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Drastically lowered area threshold (0.5% of the screen)
        min_area = (w * h) * 0.005
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(large_contours) >= 2:
            large_contours.sort(key=cv2.contourArea, reverse=True)
            top_2_contours = large_contours[:2]

            centers_x = []
            for c in top_2_contours:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    centers_x.append(int(M["m10"] / M["m00"]))

            if len(centers_x) == 2:
                midpoint_x = sum(centers_x) / 2.0
                ratio = midpoint_x / w
                self.split_ratios.append(ratio)

                cv2.line(
                    frame, (int(midpoint_x), 0), (int(midpoint_x), h), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Calibrating: {len(self.split_ratios)}/{self.required_frames}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        return None

    def is_calibrated(self) -> bool:
        return len(self.split_ratios) >= self.required_frames
