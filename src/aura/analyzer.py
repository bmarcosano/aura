"""analyzer.py: Posture analysis engine computing geometric metrics from pose landmarks."""

import numpy as np


class PostureAnalyzer:
    """Analyzes posture metrics (shoulder tilt, hip alignment, head forward posture) from MediaPipe landmarks."""

    def __init__(
        self,
        shoulder_threshold_warning: float = 2.0,
        shoulder_threshold_poor: float = 5.0,
        hip_threshold_warning: float = 2.0,
        hip_threshold_poor: float = 5.0,
        head_threshold_warning: float = 15.0,
        head_threshold_poor: float = 30.0,
    ):
        self.shoulder_threshold_warning = shoulder_threshold_warning
        self.shoulder_threshold_poor = shoulder_threshold_poor
        self.hip_threshold_warning = hip_threshold_warning
        self.hip_threshold_poor = hip_threshold_poor
        self.head_threshold_warning = head_threshold_warning
        self.head_threshold_poor = head_threshold_poor

    def analyze_shoulders(self, landmarks: np.ndarray) -> dict:
        """Calculates shoulder tilt angle relative to the horizontal axis in degrees."""
        if landmarks is None or len(landmarks) <= 12:
            return {"tilt_angle_degrees": 0.0, "status": "Unknown"}

        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]

        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]

        angle_rad = np.arctan2(dy, dx)
        tilt_angle_degrees = abs(np.degrees(angle_rad))

        if tilt_angle_degrees > 90.0:
            tilt_angle_degrees = 180.0 - tilt_angle_degrees

        status = "Optimal"
        if tilt_angle_degrees >= self.shoulder_threshold_poor:
            status = "Poor"
        elif tilt_angle_degrees >= self.shoulder_threshold_warning:
            status = "Warning"

        return {"tilt_angle_degrees": float(tilt_angle_degrees), "status": status}

    def analyze_hips(self, landmarks: np.ndarray) -> dict:
        """Calculates hip alignment tilt angle relative to the horizontal axis in degrees."""
        if landmarks is None or len(landmarks) <= 24:
            return {"tilt_angle_degrees": 0.0, "status": "Unknown"}

        # MediaPipe landmark indices: Left Hip = 23, Right Hip = 24
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]

        dx = right_hip[0] - left_hip[0]
        dy = right_hip[1] - left_hip[1]

        angle_rad = np.arctan2(dy, dx)
        tilt_angle_degrees = abs(np.degrees(angle_rad))

        if tilt_angle_degrees > 90.0:
            tilt_angle_degrees = 180.0 - tilt_angle_degrees

        status = "Optimal"
        if tilt_angle_degrees >= self.hip_threshold_poor:
            status = "Poor"
        elif tilt_angle_degrees >= self.hip_threshold_warning:
            status = "Warning"

        return {"tilt_angle_degrees": float(tilt_angle_degrees), "status": status}

    def analyze_head_posture(self, landmarks: np.ndarray) -> dict:
        """Calculates forward head posture based on horizontal offset between head and shoulder midpoint."""
        if landmarks is None or len(landmarks) <= 12:
            return {"forward_offset_pixels": 0.0, "status": "Unknown"}

        # Shoulders midpoint (Left = 11, Right = 12)
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

        # Head reference (Nose = 0 or midpoint of ears 7 & 8)
        # Using Nose (0) as primary head landmark point
        head_x = landmarks[0][0]
        head_y = landmarks[0][1]

        # Forward offset in pixels (horizontal deviation from shoulder midpoint)
        # Positive forward/backward offset magnitude
        forward_offset = (
            head_x - shoulder_mid_x
        )  # In 2D profile view or calibrated setup

        # Alternatively, absolute horizontal distance
        abs_offset = abs(forward_offset)

        status = "Optimal"
        if abs_offset >= self.head_threshold_poor:
            status = "Poor"
        elif abs_offset >= self.head_threshold_warning:
            status = "Warning"

        return {"forward_offset_pixels": float(forward_offset), "status": status}
