"""analyzer.py: Posture analysis engine computing geometric metrics from pose landmarks."""

import numpy as np


class PostureAnalyzer:
    """Analyzes posture metrics independent of camera distance using angles and anatomical normalization."""

    def __init__(
        self,
        shoulder_threshold_warning: float = 3.0,  # Degrees
        shoulder_threshold_poor: float = 7.0,
        hip_threshold_warning: float = 3.0,
        hip_threshold_poor: float = 7.0,
        head_tilt_warning: float = 5.0,  # Frontal lateral tilt (Degrees)
        head_fwd_warning: float = 15.0,  # Sagittal forward angle (Degrees)
    ):
        self.shoulder_threshold_warning = shoulder_threshold_warning
        self.shoulder_threshold_poor = shoulder_threshold_poor
        self.hip_threshold_warning = hip_threshold_warning
        self.hip_threshold_poor = hip_threshold_poor
        self.head_tilt_warning = head_tilt_warning
        self.head_fwd_warning = head_fwd_warning

    def _calculate_angle_horizontal(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Helper to calculate the absolute angle relative to the horizontal axis."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = abs(np.degrees(angle_rad))
        return float(angle_deg if angle_deg <= 90.0 else 180.0 - angle_deg)

    def analyze_shoulders(self, landmarks: np.ndarray) -> dict:
        """Calculates frontal shoulder tilt angle."""
        if landmarks is None or len(landmarks) <= 12:
            return {"tilt_angle_degrees": 0.0, "status": "Unknown"}

        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]

        tilt_angle = self._calculate_angle_horizontal(left_shoulder, right_shoulder)

        status = "Optimal"
        if tilt_angle >= self.shoulder_threshold_poor:
            status = "Poor"
        elif tilt_angle >= self.shoulder_threshold_warning:
            status = "Warning"

        return {"tilt_angle_degrees": tilt_angle, "status": status}

    def analyze_hips(self, landmarks: np.ndarray) -> dict:
        """Calculates frontal hip alignment tilt angle."""
        if landmarks is None or len(landmarks) <= 24:
            return {"tilt_angle_degrees": 0.0, "status": "Unknown"}

        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]

        tilt_angle = self._calculate_angle_horizontal(left_hip, right_hip)

        status = "Optimal"
        if tilt_angle >= self.hip_threshold_poor:
            status = "Poor"
        elif tilt_angle >= self.hip_threshold_warning:
            status = "Warning"

        return {"tilt_angle_degrees": tilt_angle, "status": status}

    def analyze_head_posture_frontal(self, landmarks: np.ndarray) -> dict:
        """
        Calculates LATERAL head deviation (frontal view) using anatomical normalization.
        Replaces pixel-based forward offset to ensure scale invariance.
        """
        if landmarks is None or len(landmarks) <= 12:
            return {"lateral_shift_ratio": 0.0, "status": "Unknown"}

        nose = landmarks[0][:2]
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]

        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0

        # Calculate shoulder width for scale normalization (Euclidean distance)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        # Prevent division by zero if tracking fails
        if shoulder_width < 1e-6:
            return {"lateral_shift_ratio": 0.0, "status": "Unknown"}

        # Ratio: deviation distance / shoulder width. Scale invariant.
        raw_offset_x = abs(nose[0] - shoulder_mid_x)
        shift_ratio = raw_offset_x / shoulder_width

        status = "Optimal"
        # Example thresholds: > 10% of shoulder width is a warning
        if shift_ratio >= 0.15:
            status = "Poor"
        elif shift_ratio >= 0.08:
            status = "Warning"

        return {"lateral_shift_ratio": float(shift_ratio), "status": status}

    def analyze_head_posture_sagittal(
        self, landmarks: np.ndarray, is_facing_right: bool = True
    ) -> dict:
        """
        Calculates TRUE Forward Head Posture from a profile (mirror) view.
        Measures the angle between the vertical axis and the ear-to-shoulder line.
        """
        if landmarks is None or len(landmarks) <= 12:
            return {"forward_angle_degrees": 0.0, "status": "Unknown"}

        # Select landmarks based on which side is visible in the mirror
        ear_idx = 8 if is_facing_right else 7
        shoulder_idx = 12 if is_facing_right else 11

        ear = landmarks[ear_idx][:2]
        shoulder = landmarks[shoulder_idx][:2]

        # Calculate angle with respect to the vertical axis (Y-axis)
        dx = ear[0] - shoulder[0]
        dy = shoulder[1] - ear[1]  # Flipped Y because image coordinates go top-down

        # Angle from vertical
        angle_rad = np.arctan2(dx, dy)
        forward_angle = np.degrees(angle_rad)

        status = "Optimal"
        if forward_angle >= self.head_fwd_warning * 2:  # Example poor threshold
            status = "Poor"
        elif forward_angle >= self.head_fwd_warning:
            status = "Warning"

        return {"forward_angle_degrees": float(forward_angle), "status": status}
