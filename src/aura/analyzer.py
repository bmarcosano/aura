"""Interpretable posture metrics computed from MediaPipe or reconstructed landmarks."""

from __future__ import annotations

import numpy as np

from aura.geometry import calculate_angle


class PostureAnalyzer:
    """Compute descriptive alignment metrics; thresholds are coaching heuristics."""

    def __init__(
        self,
        shoulder_threshold_warning: float = 3.0,
        shoulder_threshold_poor: float = 7.0,
        hip_threshold_warning: float = 3.0,
        hip_threshold_poor: float = 7.0,
        head_tilt_warning: float = 5.0,
        head_fwd_warning: float = 15.0,
    ) -> None:
        self.shoulder_threshold_warning = shoulder_threshold_warning
        self.shoulder_threshold_poor = shoulder_threshold_poor
        self.hip_threshold_warning = hip_threshold_warning
        self.hip_threshold_poor = hip_threshold_poor
        self.head_tilt_warning = head_tilt_warning
        self.head_fwd_warning = head_fwd_warning

    @staticmethod
    def _valid(landmarks: np.ndarray | None, highest_index: int) -> bool:
        if landmarks is None:
            return False
        values = np.asarray(landmarks, dtype=np.float64)
        return (
            values.ndim == 2
            and len(values) > highest_index
            and values.shape[1] >= 2
            and np.isfinite(values[:, :2]).all()
        )

    def _calculate_angle_horizontal(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Return the acute absolute angle of a line relative to horizontal."""
        delta = np.asarray(p2, dtype=float)[:2] - np.asarray(p1, dtype=float)[:2]
        if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-8:
            return float("nan")
        angle = abs(float(np.degrees(np.arctan2(delta[1], delta[0]))))
        return angle if angle <= 90.0 else 180.0 - angle

    @staticmethod
    def _status(value: float, warning: float, poor: float) -> str:
        if not np.isfinite(value):
            return "Unknown"
        if value >= poor:
            return "Poor"
        if value >= warning:
            return "Warning"
        return "Optimal"

    def analyze_shoulders(self, landmarks: np.ndarray) -> dict:
        """Estimate frontal shoulder tilt in image coordinates."""
        if not self._valid(landmarks, 12):
            return {"tilt_angle_degrees": float("nan"), "status": "Unknown"}
        angle = self._calculate_angle_horizontal(landmarks[11][:2], landmarks[12][:2])
        return {
            "tilt_angle_degrees": angle,
            "status": self._status(
                angle, self.shoulder_threshold_warning, self.shoulder_threshold_poor
            ),
        }

    def analyze_hips(self, landmarks: np.ndarray) -> dict:
        """Estimate frontal pelvis tilt in image coordinates."""
        if not self._valid(landmarks, 24):
            return {"tilt_angle_degrees": float("nan"), "status": "Unknown"}
        angle = self._calculate_angle_horizontal(landmarks[23][:2], landmarks[24][:2])
        return {
            "tilt_angle_degrees": angle,
            "status": self._status(
                angle, self.hip_threshold_warning, self.hip_threshold_poor
            ),
        }

    def analyze_head_posture_frontal(self, landmarks: np.ndarray) -> dict:
        """Estimate lateral nose offset normalized by shoulder width."""
        if not self._valid(landmarks, 12):
            return {"lateral_shift_ratio": float("nan"), "status": "Unknown"}
        nose = np.asarray(landmarks[0][:2], dtype=float)
        left, right = np.asarray(landmarks[11][:2]), np.asarray(landmarks[12][:2])
        width = float(np.linalg.norm(left - right))
        if width < 1e-8:
            return {"lateral_shift_ratio": float("nan"), "status": "Unknown"}
        ratio = float(abs(nose[0] - (left[0] + right[0]) / 2.0) / width)
        status = "Poor" if ratio >= 0.15 else "Warning" if ratio >= 0.08 else "Optimal"
        return {"lateral_shift_ratio": ratio, "status": status}

    def analyze_head_posture_sagittal(
        self, landmarks: np.ndarray, is_facing_right: bool = True
    ) -> dict:
        """Estimate profile head/shoulder offset; this is a 2D proxy, not diagnosis."""
        if not self._valid(landmarks, 12):
            return {"forward_angle_degrees": float("nan"), "status": "Unknown"}
        ear_idx, shoulder_idx = (8, 12) if is_facing_right else (7, 11)
        ear, shoulder = (
            np.asarray(landmarks[ear_idx][:2]),
            np.asarray(landmarks[shoulder_idx][:2]),
        )
        delta = ear - shoulder
        angle = float(np.degrees(np.arctan2(delta[0], -delta[1])))
        magnitude = abs(angle)
        status = (
            "Poor"
            if magnitude >= self.head_fwd_warning * 2
            else "Warning"
            if magnitude >= self.head_fwd_warning
            else "Optimal"
        )
        return {"forward_angle_degrees": angle, "status": status}

    def analyze_squat(
        self, landmarks: np.ndarray, optimal_angles: dict[str, float] | None = None
    ) -> dict:
        """Return instantaneous knee angles and optional deviations from trainer targets.

        Default 170-degree targets are retained for backwards compatibility, but
        an instantaneous angle must be interpreted in the exercise phase context.
        """
        values = np.asarray(landmarks, dtype=float) if landmarks is not None else None
        if (
            values is None
            or values.ndim != 2
            or len(values) <= 28
            or values.shape[1] < 3
        ):
            return {
                "left_knee_angle": float("nan"),
                "right_knee_angle": float("nan"),
                "status": "Unknown",
            }
        left = calculate_angle(values[23, :3], values[25, :3], values[27, :3])
        right = calculate_angle(values[24, :3], values[26, :3], values[28, :3])
        if not np.isfinite([left, right]).all():
            return {
                "left_knee_angle": left,
                "right_knee_angle": right,
                "status": "Unknown",
            }
        target_l = optimal_angles.get("left_knee", 170.0) if optimal_angles else 170.0
        target_r = optimal_angles.get("right_knee", 170.0) if optimal_angles else 170.0
        deviation = max(abs(left - target_l), abs(right - target_r))
        status = "Poor" if deviation > 20 else "Warning" if deviation > 8 else "Optimal"
        return {
            "left_knee_angle": left,
            "right_knee_angle": right,
            "status": status,
            "max_deviation": float(deviation),
        }
