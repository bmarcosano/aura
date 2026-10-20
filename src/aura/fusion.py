"""
AURA - Multi-View Sensor Fusion Engine (v2.2.0 - Advanced Coaching Update)
Combines front-view symmetry and knee alignment analysis with
mirror-view profile biomechanics into a unified safety and form score.
"""

from __future__ import annotations


class AuraFusionEngine:
    """Combines multi-view pose evaluations into a unified form and safety status."""

    def __init__(self):
        # Weighting factors for global decision making
        self.front_weight = 0.40  # Symmetry & Balance
        self.mirror_weight = 0.60  # Spine & Torso safety (profile view)

    def evaluate_posture(self, front_results, mirror_results):
        """
        Takes MediaPipe results from both views and returns
        a unified posture evaluation with status and specific messages.
        """
        front_status = self._analyze_front(front_results)
        mirror_status = self._analyze_mirror(mirror_results)

        # Global override logic: Safety/Spine (Mirror) takes precedence if critical
        if mirror_status["score"] == 0 or front_status["score"] == 0:
            global_state = "POOR"
            global_color = (0, 0, 255)  # Red
        elif mirror_status["score"] == 1 or front_status["score"] == 1:
            global_state = "WARNING"
            global_color = (0, 140, 255)  # Orange
        else:
            global_state = "OPTIMAL"
            global_color = (0, 200, 0)  # Green

        return {
            "state": global_state,
            "color": global_color,
            "front_msg": front_status["msg"],
            "mirror_msg": mirror_status["msg"],
        }

    def _analyze_front(self, results):
        """Analyzes front-view symmetry, shoulder/hip tilt, and knee alignment."""
        if (
            not results
            or not hasattr(results, "pose_landmarks")
            or not results.pose_landmarks
        ):
            return {"score": 2, "msg": "Waiting for front view..."}

        try:
            lm = results.pose_landmarks.landmark
            shoulder_tilt = abs(lm[11].y - lm[12].y)
            hip_tilt = abs(lm[23].y - lm[24].y)

            # Advanced kinematic check: Knee lateral deviation (Valgus / Tracking check)
            l_knee_drift = abs(lm[25].x - lm[23].x)
            r_knee_drift = abs(lm[26].x - lm[24].x)

            if shoulder_tilt > 0.07 or hip_tilt > 0.07:
                return {"score": 0, "msg": "Front: Severe Asymmetry"}
            elif l_knee_drift > 0.12 or r_knee_drift > 0.12:
                return {"score": 1, "msg": "Front: Check Knee Alignment"}
            elif shoulder_tilt > 0.03 or hip_tilt > 0.03:
                return {"score": 1, "msg": "Front: Mild Tilt"}

            return {"score": 2, "msg": "Front: Balanced"}
        except (AttributeError, IndexError):
            return {"score": 2, "msg": "Front: Tracking..."}

    def _analyze_mirror(self, results):
        """Analyzes mirror-view profile/biomechanics (Spine and Torso lean)."""
        if (
            not results
            or not hasattr(results, "pose_landmarks")
            or not results.pose_landmarks
        ):
            return {"score": 2, "msg": "Waiting for mirror/profile..."}

        try:
            lm = results.pose_landmarks.landmark
            torso_lean = abs(lm[11].x - lm[23].x)

            if torso_lean > 0.15:
                return {"score": 0, "msg": "Mirror: Torso Lean Risk!"}
            elif torso_lean > 0.08:
                return {"score": 1, "msg": "Mirror: Check Spine Form"}
            return {"score": 2, "msg": "Mirror: Profile Optimal"}
        except (AttributeError, IndexError):
            return {"score": 2, "msg": "Mirror: Tracking..."}
