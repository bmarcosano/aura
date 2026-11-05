"""analyzer.py: Posture analysis engine computing geometric metrics from pose landmarks using geometry.py."""

from __future__ import annotations

import numpy as np

from aura.geometry import calculate_angle


class PostureAnalyzer:
    """Analyzes posture metrics independent of camera distance using angles and anatomical normalization."""

    def __init__(
        self,
        shoulder_threshold_warning: float = 3.0,
        shoulder_threshold_poor: float = 7.0,
        hip_threshold_warning: float = 3.0,
        hip_threshold_poor: float = 7.0,
        head_tilt_warning: float = 5.0,
        head_fwd_warning: float = 15.0,
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
        """Calculates LATERAL head deviation (frontal view) using anatomical normalization."""
        if landmarks is None or len(landmarks) <= 12:
            return {"lateral_shift_ratio": 0.0, "status": "Unknown"}

        nose = landmarks[0][:2]
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]

        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_width < 1e-6:
            return {"lateral_shift_ratio": 0.0, "status": "Unknown"}

        raw_offset_x = abs(nose[0] - shoulder_mid_x)
        shift_ratio = raw_offset_x / shoulder_width

        status = "Optimal"
        if shift_ratio >= 0.15:
            status = "Poor"
        elif shift_ratio >= 0.08:
            status = "Warning"

        return {"lateral_shift_ratio": float(shift_ratio), "status": status}

    def analyze_head_posture_sagittal(
        self, landmarks: np.ndarray, is_facing_right: bool = True
    ) -> dict:
        """Calculates TRUE Forward Head Posture from a profile (mirror) view."""
        if landmarks is None or len(landmarks) <= 12:
            return {"forward_angle_degrees": 0.0, "status": "Unknown"}

        ear_idx = 8 if is_facing_right else 7
        shoulder_idx = 12 if is_facing_right else 11

        ear = landmarks[ear_idx][:2]
        shoulder = landmarks[shoulder_idx][:2]

        dx = ear[0] - shoulder[0]
        dy = shoulder[1] - ear[1]

        angle_rad = np.arctan2(dx, dy)
        forward_angle = np.degrees(angle_rad)

        status = "Optimal"
        if forward_angle >= self.head_fwd_warning * 2:
            status = "Poor"
        elif forward_angle >= self.head_fwd_warning:
            status = "Warning"

        return {"forward_angle_degrees": float(forward_angle), "status": status}

    def analyze_squat(
        self, landmarks: np.ndarray, optimal_angles: dict[str, float] | None = None
    ) -> dict:
        """
        Analyzes squat posture, evaluating knee angles against optimal golden targets using geometry.calculate_angle.
        """
        if landmarks is None or len(landmarks) <= 28:
            return {
                "left_knee_angle": 0.0,
                "right_knee_angle": 0.0,
                "status": "Unknown",
            }

        left_hip = landmarks[23][:2]
        left_knee = landmarks[25][:2]
        left_ankle = landmarks[27][:2]

        right_hip = landmarks[24][:2]
        right_knee = landmarks[26][:2]
        right_ankle = landmarks[28][:2]

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        target_left = (
            optimal_angles.get("left_knee", 170.0) if optimal_angles else 170.0
        )
        target_right = (
            optimal_angles.get("right_knee", 170.0) if optimal_angles else 170.0
        )

        diff_left = abs(left_knee_angle - target_left)
        diff_right = abs(right_knee_angle - target_right)
        max_diff = max(diff_left, diff_right)

        status = "Optimal"
        if max_diff > 20.0:
            status = "Poor"
        elif max_diff > 8.0:
            status = "Warning"

        return {
            "left_knee_angle": float(left_knee_angle),
            "right_knee_angle": float(right_knee_angle),
            "status": status,
            "max_deviation": float(max_diff),
        }
