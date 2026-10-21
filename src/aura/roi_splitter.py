"""
Module for automatic 2D Region of Interest (ROI) split ratio estimation using motion detection.

Once the kinematics layer confirms a mirror setup, this module estimates the exact vertical
boundary (split ratio) separating the direct view of the user from the specular reflection.
It utilizes a lightweight Mixture of Gaussians (MOG2) background subtractor to track the
centroids of the two primary moving bodies and computes their spatial midpoint.
"""

from __future__ import annotations

import cv2
import numpy as np


class AutoROISplitter:
    """
    Estimates the vertical ROI split ratio by analyzing motion vectors in the frame.

    Instead of running heavy deep-learning inference, it employs pixel-level motion analysis
    to detect the two largest moving blobs (user and mirror reflection) and determines the
    optical midpoint.
    """

    def __init__(self, required_frames: int = 5, timeout_frames: int = 60) -> None:
        """
        Initializes the ROI splitter.

        Args:
            required_frames (int): Number of consistent frame calculations required to finalize split.
            timeout_frames (int): Fallback limit to force split estimation if motion stalls.
        """
        # MOG2 isolates moving foreground objects from the static background.
        # learningRate=-1 allows automatic adaptation speed.
        # detectShadows=False optimizes throughput and prevents shadow artifacts from skewing centroids.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        # Stores valid split ratio readings (0.0 to 1.0) to average out noise and outliers.
        self.split_ratios: list[float] = []

        # Target number of valid frame readings before finalizing the split ratio
        self.required_frames = required_frames

        # Fallback limit: forces a split ratio if motion stalls or lighting is inadequate.
        self.timeout_frames = timeout_frames
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> float | None:
        """
        Processes an incoming video frame to detect motion and compute the ROI boundary.

        Args:
            frame (np.ndarray): Input BGR image from the camera feed.

        Returns:
            float | None: Averaged normalized split ratio (0.0 to 1.0) when complete,
                          or None while accumulating data.
        """
        # --- 1. COMPLETION CHECK ---
        if self.is_split_ready():
            if cv2.getWindowProperty("AURA - Debug Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("AURA - Debug Mask")
            return float(np.mean(self.split_ratios))

        self.frame_count += 1

        # --- 2. TIMEOUT FALLBACK ---
        if self.frame_count > self.timeout_frames:
            print("⚠️ ROI split timeout reached. Forcing split ratio.")
            if cv2.getWindowProperty("AURA - Debug Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("AURA - Debug Mask")
            return float(np.mean(self.split_ratios)) if self.split_ratios else 0.5

        h, w = frame.shape[:2]

        # --- 3. MOTION MASK GENERATION ---
        # Generate binary mask: moving foreground = 255 (white), static background = 0 (black).
        mask = self.bg_subtractor.apply(frame, learningRate=-1)

        # --- 4. NOISE REDUCTION & MORPHOLOGY ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # MORPH_OPEN removes high-frequency noise and pixel jitter.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # MORPH_DILATE merges fragmented body contours into unified blobs.
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)

        # --- 5. CONTOUR EXTRACTION & FILTERING ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Keep only contours exceeding 0.5% of total frame area
        min_area = (w * h) * 0.005
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # --- 6. GEOMETRIC MIDPOINT CALCULATION ---
        if len(large_contours) >= 2:
            # Select top 2 largest moving bodies (User + Reflection)
            large_contours.sort(key=cv2.contourArea, reverse=True)
            top_2_contours = large_contours[:2]

            centers_x = []
            for c in top_2_contours:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    # Centroid X-coordinate: M10 / M00
                    centers_x.append(int(M["m10"] / M["m00"]))

            if len(centers_x) == 2:
                # Calculate optical boundary midpoint between user and reflection
                midpoint_x = sum(centers_x) / 2.0
                ratio = midpoint_x / w
                self.split_ratios.append(ratio)

                # --- 7. VISUAL FEEDBACK ---
                cv2.line(
                    frame, (int(midpoint_x), 0), (int(midpoint_x), h), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Estimating Split: {len(self.split_ratios)}/{self.required_frames}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        return None

    def is_split_ready(self) -> bool:
        """Returns True if the target number of stable split readings has been reached."""
        return len(self.split_ratios) >= self.required_frames
