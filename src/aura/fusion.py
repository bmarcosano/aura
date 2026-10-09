"""
AURA - Multi-View Sensor Fusion Engine (v2.2.0 - Advanced Coaching Update)
Combines front-view symmetry and knee alignment analysis with
mirror-view profile biomechanics into a unified safety and form score.
"""


class AuraFusionEngine:
    def __init__(self):
        # Weighting factors for global decision making[cite: 2]
        self.front_weight = 0.40  # Symmetry & Balance[cite: 2]
        self.mirror_weight = 0.60  # Spine & Torso safety (profile view)[cite: 2]

    def evaluate_posture(self, front_results, mirror_results):
        """
        Takes MediaPipe results from both views and returns
        a unified posture evaluation with status and specific messages.[cite: 2]
        """
        front_status = self._analyze_front(front_results)
        mirror_status = self._analyze_mirror(mirror_results)

        # Global override logic: Safety/Spine (Mirror) takes precedence if critical[cite: 2]
        if mirror_status["score"] == 0 or front_status["score"] == 0:
            global_state = "POOR"
            global_color = (0, 0, 255)  # Red[cite: 2]
        elif mirror_status["score"] == 1 or front_status["score"] == 1:
            global_state = "WARNING"
            global_color = (0, 140, 255)  # Orange[cite: 2]
        else:
            global_state = "OPTIMAL"
            global_color = (0, 200, 0)  # Green[cite: 2]

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
            return {"score": 2, "msg": "Waiting for front view..."}  # [cite: 2]

        try:
            lm = results.pose_landmarks.landmark
            shoulder_tilt = abs(lm[11].y - lm[12].y)  # [cite: 2]
            hip_tilt = abs(lm[23].y - lm[24].y)  # [cite: 2]

            # Advanced kinematic check: Knee lateral deviation (Valgus / Tracking check)
            # Evaluates the horizontal spread of knees (landmarks 25 and 26) relative to hips
            l_knee_drift = abs(lm[25].x - lm[23].x)
            r_knee_drift = abs(lm[26].x - lm[24].x)

            if shoulder_tilt > 0.07 or hip_tilt > 0.07:
                return {"score": 0, "msg": "Front: Severe Asymmetry"}  # [cite: 2]
            elif l_knee_drift > 0.12 or r_knee_drift > 0.12:
                return {"score": 1, "msg": "Front: Check Knee Alignment"}
            elif shoulder_tilt > 0.03 or hip_tilt > 0.03:
                return {"score": 1, "msg": "Front: Mild Tilt"}  # [cite: 2]

            return {"score": 2, "msg": "Front: Balanced"}  # [cite: 2]
        except (AttributeError, IndexError):
            return {"score": 2, "msg": "Front: Tracking..."}  # [cite: 2]

    def _analyze_mirror(self, results):
        """Analyzes mirror-view profile/biomechanics (Spine and Torso lean)."""
        if (
            not results
            or not hasattr(results, "pose_landmarks")
            or not results.pose_landmarks
        ):
            return {"score": 2, "msg": "Waiting for mirror/profile..."}  # [cite: 2]

        try:
            lm = results.pose_landmarks.landmark
            # Profile analysis approximation using shoulder and hip horizontal offset[cite: 2]
            torso_lean = abs(lm[11].x - lm[23].x)  # [cite: 2]

            if torso_lean > 0.15:
                return {"score": 0, "msg": "Mirror: Torso Lean Risk!"}  # [cite: 2]
            elif torso_lean > 0.08:
                return {"score": 1, "msg": "Mirror: Check Spine Form"}  # [cite: 2]
            return {"score": 2, "msg": "Mirror: Profile Optimal"}  # [cite: 2]
        except (AttributeError, IndexError):
            return {"score": 2, "msg": "Mirror: Tracking..."}  # [cite: 2]
