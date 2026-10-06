"""
AURA - Multi-View Sensor Fusion Engine
Combines front-view symmetry analysis and mirror-view profile biomechanics
into a unified safety and form score.
"""

class AuraFusionEngine:
    def __init__(self):
        # Weighting factors for global decision making
        self.front_weight = 0.40  # Symmetry & Balance
        self.mirror_weight = 0.60 # Spine & Torso safety (profile view)

    def evaluate_posture(self, front_results, mirror_results):
        """
        Takes MediaPipe (or detector) results from both views and returns 
        a unified posture evaluation with status and specific messages.
        """
        front_status = self._analyze_front(front_results)
        mirror_status = self._analyze_mirror(mirror_results)

        # Global override logic: Safety/Spine (Mirror) takes precedence if critical
        if mirror_status["score"] == 0 or front_status["score"] == 0:
            global_state = "POOR"
            global_color = (0, 0, 255) # Red
        elif mirror_status["score"] == 1 or front_status["score"] == 1:
            global_state = "WARNING"
            global_color = (0, 140, 255) # Orange
        else:
            global_state = "OPTIMAL"
            global_color = (0, 200, 0) # Green

        return {
            "state": global_state,
            "color": global_color,
            "front_msg": front_status["msg"],
            "mirror_msg": mirror_status["msg"]
        }

    def _analyze_front(self, results):
        """Analyzes front-view symmetry (Shoulders and Hips tilt)."""
        if not results or not hasattr(results, 'pose_landmarks') or not results.pose_landmarks:
            return {"score": 2, "msg": "Waiting for front view..."}
        
        try:
            lm = results.pose_landmarks.landmark
            shoulder_tilt = abs(lm[11].y - lm[12].y)
            hip_tilt = abs(lm[23].y - lm[24].y)

            if shoulder_tilt > 0.07 or hip_tilt > 0.07:
                return {"score": 0, "msg": "Front: Severe Asymmetry"}
            elif shoulder_tilt > 0.03 or hip_tilt > 0.03:
                return {"score": 1, "msg": "Front: Mild Tilt"}
            return {"score": 2, "msg": "Front: Balanced"}
        except (AttributeError, IndexError):
            return {"score": 2, "msg": "Front: Tracking..."}

    def _analyze_mirror(self, results):
        """Analyzes mirror-view profile/biomechanics (Spine and Torso lean)."""
        if not results or not hasattr(results, 'pose_landmarks') or not results.pose_landmarks:
            return {"score": 2, "msg": "Waiting for mirror/profile..."}
        
        try:
            lm = results.pose_landmarks.landmark
            # Profile analysis approximation using shoulder, hip and knee alignment
            # (Assuming mirror captures side profile or angled reflection)
            torso_lean = abs(lm[11].x - lm[23].x) # Horizontal offset between shoulder and hip

            if torso_lean > 0.15:
                return {"score": 0, "msg": "Mirror: Torso Lean Risk!"}
            elif torso_lean > 0.08:
                return {"score": 1, "msg": "Mirror: Check Spine Form"}
            return {"score": 2, "msg": "Mirror: Profile Optimal"}
        except (AttributeError, IndexError):
            return {"score": 2, "msg": "Mirror: Tracking..."}