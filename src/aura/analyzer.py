"""analyzer.py: Posture analysis engine computing geometric metrics from pose landmarks."""

import numpy as np


class PostureAnalyzer:
    """Analyzes posture metrics (shoulder tilt, hip alignment, head posture) from MediaPipe landmarks."""

    def __init__(self, shoulder_threshold_warning: float = 2.0, shoulder_threshold_poor: float = 5.0):
        self.shoulder_threshold_warning = shoulder_threshold_warning
        self.shoulder_threshold_poor = shoulder_threshold_poor

    def analyze_shoulders(self, landmarks: np.ndarray) -> dict:
        """Calculates shoulder tilt angle relative to the horizontal axis in degrees and returns status."""
        if landmarks is None or len(landmarks) <= 12:
            return {"tilt_angle_degrees": 0.0, "status": "Unknown"}

        # MediaPipe landmark indices: Left Shoulder = 11, Right Shoulder = 12
        left_shoulder = landmarks[11][:2]   # [x, y]
        right_shoulder = landmarks[12][:2]  # [x, y]

        # Vector components between left and right shoulders
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]

        # Calculate angle relative to horizontal axis
        angle_rad = np.arctan2(dy, dx)
        tilt_angle_degrees = abs(np.degrees(angle_rad))

        # Normalize deviation from horizontal (0° or 180°)
        if tilt_angle_degrees > 90.0:
            tilt_angle_degrees = 180.0 - tilt_angle_degrees

        # Evaluate posture status based on configured thresholds
        status = "Optimal"
        if tilt_angle_degrees >= self.shoulder_threshold_poor:
            status = "Poor"
        elif tilt_angle_degrees >= self.shoulder_threshold_warning:
            status = "Warning"

        return {
            "tilt_angle_degrees": float(tilt_angle_degrees),
            "status": status
        }