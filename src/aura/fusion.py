"""Fuse front/profile coaching observations without treating missing data as success."""

from __future__ import annotations

import numpy as np


class AuraFusionEngine:
    """Combine independent view-level heuristic assessments into one status."""

    def __init__(self):
        self.front_weight = 0.40
        self.mirror_weight = 0.60

    @staticmethod
    def _extract(results):
        if results is None or not getattr(results, "pose_landmarks", None):
            return None
        return results.pose_landmarks.landmark

    def evaluate_posture(self, front_results, mirror_results):
        front = self._analyze_front(front_results)
        mirror = self._analyze_mirror(mirror_results)
        weighted_items = []
        if front["score"] >= 0:
            weighted_items.append((front["score"], self.front_weight))
        if mirror["score"] >= 0:
            weighted_items.append((mirror["score"], self.mirror_weight))
        if not weighted_items:
            weighted_score = float("nan")
            state, color = "INSUFFICIENT_DATA", (160, 160, 160)
        else:
            total_weight = sum(weight for _, weight in weighted_items)
            weighted_score = (
                sum(score * weight for score, weight in weighted_items) / total_weight
            )
            # A critical view-level failure takes precedence over the weighted mean.
            if any(score == 0 for score, _ in weighted_items):
                state, color = "POOR", (0, 0, 255)
            elif len(weighted_items) < 2:
                state, color = "PARTIAL", (0, 200, 255)
            elif weighted_score < 1.5:
                state, color = "WARNING", (0, 140, 255)
            else:
                state, color = "OPTIMAL", (0, 200, 0)
        return {
            "state": state,
            "color": color,
            "front_msg": front["msg"],
            "mirror_msg": mirror["msg"],
            "weighted_score": float(weighted_score),
        }

    def _analyze_front(self, results):
        landmarks = self._extract(results)
        if landmarks is None:
            return {"score": -1, "msg": "Front: insufficient data"}
        try:
            required = [11, 12, 23, 24, 25, 26]
            if any(landmarks[i].visibility < 0.35 for i in required):
                return {"score": -1, "msg": "Front: low landmark visibility"}
            shoulder_tilt = abs(landmarks[11].y - landmarks[12].y)
            hip_tilt = abs(landmarks[23].y - landmarks[24].y)
            left_drift = abs(landmarks[25].x - landmarks[23].x)
            right_drift = abs(landmarks[26].x - landmarks[24].x)
            if shoulder_tilt > 0.07 or hip_tilt > 0.07:
                return {"score": 0, "msg": "Front: marked asymmetry"}
            if left_drift > 0.12 or right_drift > 0.12:
                return {"score": 1, "msg": "Front: review knee tracking"}
            if shoulder_tilt > 0.03 or hip_tilt > 0.03:
                return {"score": 1, "msg": "Front: mild alignment deviation"}
            return {"score": 2, "msg": "Front: balanced"}
        except (AttributeError, IndexError, TypeError):
            return {"score": -1, "msg": "Front: tracking unavailable"}

    def _analyze_mirror(self, results):
        landmarks = self._extract(results)
        if landmarks is None:
            return {"score": -1, "msg": "Profile: insufficient data"}
        try:
            if landmarks[11].visibility < 0.35 or landmarks[23].visibility < 0.35:
                return {"score": -1, "msg": "Profile: low landmark visibility"}
            dx = landmarks[11].x - landmarks[23].x
            dy = landmarks[11].y - landmarks[23].y
            lean = abs(float(np.degrees(np.arctan2(dx, max(abs(dy), 1e-8)))))
            if lean > 25:
                return {"score": 0, "msg": "Profile: pronounced torso inclination"}
            if lean > 15:
                return {"score": 1, "msg": "Profile: review torso inclination"}
            return {"score": 2, "msg": "Profile: no large inclination detected"}
        except (AttributeError, IndexError, TypeError):
            return {"score": -1, "msg": "Profile: tracking unavailable"}
