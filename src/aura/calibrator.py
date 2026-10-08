"""
Module for automatic spatial calibration using motion detection (The Calibrator).

Once the Sync Analyzer (in main.py) confirms the presence of a mirror via kinematics,
this module takes over to pinpoint the exact geometric boundary (split_ratio) between 
the real user and the reflection. It uses a lightweight background subtractor (MOG2) 
to track the center of mass of the two moving bodies and calculates their exact midpoint.
"""

from __future__ import annotations

import cv2
import numpy as np


class AutoSplitCalibrator:
    """
    Calibrates the split ratio by analyzing synchronized movements in the frame.
    Instead of heavy neural networks, it uses pixel-level motion detection to find 
    the two largest moving objects (user and mirror reflection) and splits the difference.
    """

    def __init__(self, required_frames: int = 5, timeout_frames: int = 60):
        # MOG2 (Mixture of Gaussians) isolates moving objects from the static background.
        # learningRate=-1 lets the algorithm automatically decide the background update speed.
        # detectShadows=False improves performance and prevents shadows from distorting the center of mass.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        # Stores valid split ratios calculated across multiple frames to average out anomalies.
        self.split_ratios: list[float] = []
        
        # How many successful readings we need to consider the calibration complete
        self.required_frames = required_frames
        
        # Fallback mechanism: if the user stops moving or lighting is bad, force a 
        # split after this many frames to prevent the system from hanging indefinitely.
        self.timeout_frames = timeout_frames
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> float | None:
        """
        Analyzes the frame for motion. Returns the averaged split ratio (float 0.0 - 1.0)
        once enough valid frames are collected, or if a timeout occurs. Returns None while working.
        """
        # --- 1. COMPLETION CHECK ---
        if self.is_calibrated():
            # Ensure debug window is closed when calibration finishes
            if cv2.getWindowProperty("AURA - Debug Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("AURA - Debug Mask")
            return float(np.mean(self.split_ratios))

        self.frame_count += 1

        # --- 2. TIMEOUT FALLBACK ---
        if self.frame_count > self.timeout_frames:
            print("⚠️ Calibration timeout reached. Forcing split ratio.")
            if cv2.getWindowProperty("AURA - Debug Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("AURA - Debug Mask")
            # If we collected some data before timeout, average it. Otherwise, default to 50% (0.5)
            return float(np.mean(self.split_ratios)) if self.split_ratios else 0.5

        h, w = frame.shape[:2]

        # --- 3. MOTION MASK GENERATION ---
        # Apply MOG2. Moving pixels become white (255), static pixels become black (0).
        mask = self.bg_subtractor.apply(frame, learningRate=-1)

        # --- 4. NOISE REDUCTION (MORPHOLOGICAL OPERATIONS) ---
        # Define a 3x3 circular kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # MORPH_OPEN: Erases tiny white dots (salt-and-pepper noise/camera artifacts)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # MORPH_DILATE: Expands the remaining white areas to merge fragmented body parts 
        # into a single solid blob for each person/reflection.
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)

        # 🛑 DEBUG: Uncomment to see the black/white motion mask in a separate window
        # cv2.imshow("AURA - Debug Mask", mask)

        # --- 5. CONTOUR EXTRACTION ---
        # Find the boundaries of the white blobs in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small irrelevant movements (e.g., a fan moving in the background)
        # We only care about blobs that take up at least 0.5% of the total screen area
        min_area = (w * h) * 0.005
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # --- 6. GEOMETRIC SPLIT CALCULATION ---
        # If we see at least two distinct large moving bodies (User + Mirror)
        if len(large_contours) >= 2:
            # Sort blobs by size (largest first) and grab the top 2
            large_contours.sort(key=cv2.contourArea, reverse=True)
            top_2_contours = large_contours[:2]

            centers_x = []
            for c in top_2_contours:
                # Calculate Image Moments (spatial characteristics of the shape)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    # Center of Mass (Centroid) X-coordinate formula: M10 / M00
                    centers_x.append(int(M["m10"] / M["m00"]))

            # If we successfully found the center of both bodies
            if len(centers_x) == 2:
                # The optical split line is exactly halfway between the real user and the reflection
                midpoint_x = sum(centers_x) / 2.0
                
                # Convert the absolute pixel X-coordinate into a normalized ratio (0.0 to 1.0)
                ratio = midpoint_x / w
                self.split_ratios.append(ratio)

                # --- 7. VISUAL FEEDBACK ---
                # Draw a green vertical line showing where the algorithm thinks the mirror edge is
                cv2.line(
                    frame, (int(midpoint_x), 0), (int(midpoint_x), h), (0, 255, 0), 2
                )
                # Show collection progress (e.g., 1/5, 2/5)
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
        """Returns True if the required number of stable split calculations has been reached."""
        return len(self.split_ratios) >= self.required_frames