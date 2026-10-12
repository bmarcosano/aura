"""
renderer.py: Handles all OpenCV UI drawing, HUDs, and visual overlays.
"""

import cv2
import numpy as np

from aura.analyzer import PostureAnalyzer


class AuraRenderer:
    """Handles the rendering of all graphical overlays on OpenCV frames."""

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
        """Renders the sidebar menu with recording status and DTW evaluation."""
        _, w, _ = frame.shape
        panel_w, panel_h = 280, 340
        x1, y1 = w - panel_w - 15, 15
        x2, y2 = w - 15, y1 + panel_h

        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 1. TITLE & MENU CONTROLS
        cv2.putText(
            frame,
            "--- AURA CONTROLS ---",
            (x1 + 15, y1 + 22),
            self.font,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        commands = [
            ("[G] Rec Golden Rep (Single)", (220, 220, 220)),
            ("[U] Rec User Execution", (220, 220, 220)),
            ("[S] Stop & Evaluate", (220, 220, 220)),
            ("[W] Save Golden (JSON)", (220, 220, 220)),
            ("[L] Load Golden (JSON)", (220, 220, 220)),
            ("[V] View Info & Status", (220, 220, 220)),
            ("[D] Delete JSON File", (220, 220, 220)),
            ("[Q] Quit AURA", (150, 150, 150)),
        ]

        for i, (text, color) in enumerate(commands):
            cv2.putText(
                frame,
                text,
                (x1 + 15, y1 + 45 + (i * 18)),
                self.font,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.line(frame, (x1 + 15, y1 + 195), (x2 - 15, y1 + 195), (100, 100, 100), 1)

        # 2. RECORDING STATUS
        cv2.putText(
            frame,
            "STATUS:",
            (x1 + 15, y1 + 215),
            self.font,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        if recording_state == "IDLE":
            cv2.putText(
                frame,
                "WAITING / IDLE",
                (x1 + 75, y1 + 215),
                self.font,
                0.42,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )
        else:
            if frame_count % 15 < 10:
                cv2.circle(frame, (x1 + 80, y1 + 211), 5, (0, 0, 255), -1)

            status_text = "GOLDEN" if recording_state == "RECORD_GOLDEN" else "USER"
            cv2.putText(
                frame,
                f"REC {status_text}",
                (x1 + 92, y1 + 215),
                self.font,
                0.42,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        mem_color = (0, 200, 0) if has_golden else (0, 0, 255)
        mem_text = "Mem: READY" if has_golden else "Mem: EMPTY"
        cv2.putText(
            frame,
            mem_text,
            (x2 - 95, y1 + 215),
            self.font,
            0.38,
            mem_color,
            1,
            cv2.LINE_AA,
        )

        # 3. LIVE PACING FEEDBACK
        if recording_state == "RECORD_USER" and live_pacing_msg:
            cv2.putText(
                frame,
                f"{live_pacing_msg}",
                (x1 + 15, y1 + 242),
                self.font,
                0.38,
                (0, 140, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. DTW EVALUATION RESULT
        if dtw_result:
            score = dtw_result.get("score", 0)
            feedback = dtw_result.get("feedback", "")
            color = (
                (0, 255, 0)
                if score >= 85
                else ((0, 165, 255) if score >= 60 else (0, 0, 255))
            )

            cv2.putText(
                frame,
                "LAST SCORE:",
                (x1 + 15, y1 + 270),
                self.font,
                0.43,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{score}/100",
                (x1 + 15, y1 + 300),
                self.font,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                feedback,
                (x1 + 95, y1 + 298),
                self.font,
                0.38,
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
        """Draws the postural alignment metrics HUD."""
        sh_status, sh_color = "WAITING...", (180, 180, 180)
        hip_status, hip_color = "WAITING...", (180, 180, 180)
        head_status, head_color = "WAITING...", (180, 180, 180)

        if (
            pose_results
            and hasattr(pose_results, "pose_landmarks")
            and pose_results.pose_landmarks
        ):
            landmarks_np = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
            )

            sh_data = analyzer.analyze_shoulders(landmarks_np)
            hip_data = analyzer.analyze_hips(landmarks_np)
            head_data = (
                analyzer.analyze_head_posture_sagittal(landmarks_np)
                if is_mirror_view
                else analyzer.analyze_head_posture_frontal(landmarks_np)
            )

            def map_color(status: str):
                if status == "Optimal":
                    return (0, 200, 0)
                elif status == "Warning":
                    return (0, 140, 255)
                elif status == "Poor":
                    return (0, 0, 255)
                return (180, 180, 180)

            sh_status, sh_color = (
                sh_data["status"].upper(),
                map_color(sh_data["status"]),
            )
            hip_status, hip_color = (
                hip_data["status"].upper(),
                map_color(hip_data["status"]),
            )
            head_status, head_color = (
                head_data["status"].upper(),
                map_color(head_data["status"]),
            )

        overlay = frame.copy()
        x1, y1, x2, y2 = 15, 15, 275, 145
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        cv2.putText(
            frame,
            "--- POSTURE HUD ---",
            (x1 + 10, y1 + 22),
            self.font,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Shoulders: {sh_status}",
            (x1 + 10, y1 + 55),
            self.font,
            0.52,
            sh_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Pelvis/Hips: {hip_status}",
            (x1 + 10, y1 + 88),
            self.font,
            0.52,
            hip_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Head Balance: {head_status}",
            (x1 + 10, y1 + 121),
            self.font,
            0.52,
            head_color,
            2,
            cv2.LINE_AA,
        )
