"""2D OpenCV overlays for controls, recording state, and descriptive posture metrics."""

from __future__ import annotations

import cv2
import numpy as np

from aura.analyzer import PostureAnalyzer


class AuraRenderer:
    """Draw compact status panels without changing pose or measurement data."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_control_panel(
        self,
        frame,
        recording_state,
        dtw_result,
        frame_count,
        has_golden,
        live_pacing_msg,
    ) -> None:
        """Render command hints and the most recent DTW result."""
        if frame is None or frame.size == 0:
            return
        _, width = frame.shape[:2]
        panel_w, panel_h = min(300, max(220, width // 3)), 330
        x1, y1 = max(0, width - panel_w - 12), 12
        x2, y2 = width - 8, min(frame.shape[0] - 8, y1 + panel_h)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.putText(
            frame,
            "AURA CONTROLS",
            (x1 + 12, y1 + 22),
            self.font,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        commands = [
            "G: record trainer reference",
            "U: record user",
            "S: stop/evaluate",
            "W: save reference",
            "L: load reference",
            "D: delete reference",
            "X: toggle mirror mode",
            "M: toggle log detail",
            "Q: quit",
        ]
        y = y1 + 48
        for text in commands:
            cv2.putText(
                frame,
                text,
                (x1 + 12, y),
                self.font,
                0.36,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
            y += 18
        y += 4
        cv2.putText(
            frame,
            f"REC: {recording_state}",
            (x1 + 12, y),
            self.font,
            0.42,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Golden: READY" if has_golden else "Golden: EMPTY",
            (x1 + 130, y),
            self.font,
            0.38,
            (0, 220, 0) if has_golden else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        y += 25
        if live_pacing_msg:
            cv2.putText(
                frame,
                str(live_pacing_msg)[:38],
                (x1 + 12, y),
                self.font,
                0.36,
                (0, 190, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22
        if dtw_result:
            score = float(dtw_result.get("score", 0.0))
            feedback = str(dtw_result.get("feedback", dtw_result.get("error", "")))
            color = (
                (0, 220, 0)
                if score >= 85
                else (0, 165, 255)
                if score >= 60
                else (0, 0, 255)
            )
            cv2.putText(
                frame,
                f"LAST SCORE: {score:.1f}/100",
                (x1 + 12, min(y + 4, y2 - 30)),
                self.font,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                feedback[:35],
                (x1 + 12, min(y + 27, y2 - 10)),
                self.font,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_postural_hud(
        self,
        frame,
        pose_results,
        analyzer: PostureAnalyzer,
        is_mirror_view: bool = False,
    ) -> None:
        """Draw frontal/profile alignment estimates; these remain 2D coaching proxies."""
        statuses = {"Shoulders": "Unknown", "Pelvis": "Unknown", "Head": "Unknown"}
        if pose_results and getattr(pose_results, "pose_landmarks", None):
            landmarks = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            statuses["Shoulders"] = analyzer.analyze_shoulders(landmarks)["status"]
            statuses["Pelvis"] = analyzer.analyze_hips(landmarks)["status"]
            head = (
                analyzer.analyze_head_posture_sagittal(landmarks)
                if is_mirror_view
                else analyzer.analyze_head_posture_frontal(landmarks)
            )
            statuses["Head"] = head["status"]
        color_map = {
            "Optimal": (0, 200, 0),
            "Warning": (0, 140, 255),
            "Poor": (0, 0, 255),
            "Unknown": (170, 170, 170),
        }
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (255, 132), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(
            frame,
            "POSTURE (2D ESTIMATES)",
            (22, 35),
            self.font,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        for index, (label, status) in enumerate(statuses.items()):
            cv2.putText(
                frame,
                f"{label}: {status}",
                (22, 65 + index * 22),
                self.font,
                0.42,
                color_map[status],
                1,
                cv2.LINE_AA,
            )
