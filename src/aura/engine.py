"""
engine.py: Core application loop and state management for AURA.
"""

import math
import os
import sys

import cv2
import mediapipe as mp

from aura.analyzer import PostureAnalyzer
from aura.detector import PoseDetector
from aura.fusion import AuraFusionEngine
from aura.multi_detector import MultiSkeletonDetector

# Newly created UI and Input modules
from aura.renderer import AuraRenderer
from aura.roi_splitter import AutoROISplitter
from aura.temporal import DTWTemporalAnalyzer
from aura.utils.input_handler import KeyboardController

# State Constants
STATE_SINGLE = "SINGLE_MODE"
STATE_MIRROR = "MIRROR_MODE"
REC_IDLE = "IDLE"
REC_GOLDEN = "RECORD_GOLDEN"
REC_USER = "RECORD_USER"


class AuraEngine:
    """Orchestrates the camera feed, biomechanical models, and UI rendering."""

    def __init__(self, source="0", exercise="squats"):
        self.source = int(source) if str(source).isdigit() else source
        self.exercise = exercise
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"❌ Error: Could not open video source '{self.source}'")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / fps) if fps > 0 else 33

        # --- COMPONENT INITIALIZATION ---
        self.single_detector = PoseDetector()
        self.posture_analyzer = PostureAnalyzer()

        self.splitter = AutoROISplitter()
        self.multi_detector = None
        self.fusion_engine = AuraFusionEngine()
        self.dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)

        # New modular components
        self.renderer = AuraRenderer()

        # Silent Split setup
        mp_pose = mp.solutions.pose
        self.silent_checker_left = mp_pose.Pose(min_detection_confidence=0.5)
        self.silent_checker_right = mp_pose.Pose(min_detection_confidence=0.5)
        self.pearson_buffer_left = []
        self.pearson_buffer_right = []

        # System state
        self.current_state = STATE_SINGLE
        self.recording_state = REC_IDLE
        self.dtw_last_result = None
        self.frame_count = 0

    def run(self):
        """Starts the main video loop."""
        print(f"🚀 AURA Engine Started in SINGLE_MODE (Source: {self.source})")

        while self.cap.isOpened():
            success, frame = self.cap.read()

            # Handle end of video (if using an .mp4 file for testing)
            if not success or frame is None or frame.size == 0:
                if self.recording_state != REC_IDLE:
                    print(
                        "\n⚠️ Video ended during recording. Auto-stopping to evaluate..."
                    )
                    if self.recording_state == REC_USER:
                        self.dtw_last_result = self.dtw_analyzer.evaluate_technique()
                    self.recording_state = REC_IDLE

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
                if not success:
                    print("❌ Failed to re-read the video.")
                    break

            self.frame_count += 1
            _, w = frame.shape[:2]
            active_frame = None
            active_results = None
            front_roi = None
            mirror_roi = None

            # ==========================================
            # PHASE 1: SINGLE MODE
            # ==========================================
            if self.current_state == STATE_SINGLE:
                display_frame = frame.copy()
                active_frame = display_frame

                self.single_detector.detect(display_frame, draw=True)
                active_results = getattr(self.single_detector, "results", None)

                # Render HUD using the new UI class
                self.renderer.draw_postural_hud(
                    display_frame,
                    active_results,
                    self.posture_analyzer,
                    is_mirror_view=False,
                )

                # Background check for mirror presence
                self._check_silent_split(frame, w)

            # ==========================================
            # PHASE 2: MIRROR MODE
            # ==========================================
            elif self.current_state == STATE_MIRROR:
                if self.multi_detector is None:
                    split_ratio = self.splitter.update(frame)
                    if split_ratio is not None:
                        self.multi_detector = MultiSkeletonDetector(
                            split_ratio=split_ratio
                        )
                        cv2.destroyWindow("AURA - Calibration")
                    else:
                        cv2.imshow("AURA - Calibration", frame)
                else:
                    # Clean delegation to MultiSkeletonDetector
                    pipeline_data = self.multi_detector.process_frame(frame)

                    front_roi = pipeline_data["front_roi"]
                    mirror_roi = pipeline_data["mirror_roi"]
                    active_results = pipeline_data["front_results"]
                    mirror_results = pipeline_data["mirror_results"]

                    active_frame = front_roi

                    # Fusion evaluation (will handle None mirror_results gracefully if validated failed)
                    fusion_result = self.fusion_engine.evaluate_posture(
                        active_results, mirror_results
                    )

                    # Render the two split HUDs
                    if self._is_valid_roi(front_roi):
                        self.renderer.draw_postural_hud(
                            front_roi,
                            active_results,
                            self.posture_analyzer,
                            is_mirror_view=False,
                        )
                        cv2.putText(
                            front_roi,
                            f"FUSION: {fusion_result['state']}",
                            (15, 175),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            fusion_result["color"],
                            2,
                            cv2.LINE_AA,
                        )

                    if self._is_valid_roi(mirror_roi) and mirror_results:
                        self.renderer.draw_postural_hud(
                            mirror_roi,
                            mirror_results,
                            self.posture_analyzer,
                            is_mirror_view=True,
                        )

            # ==========================================
            # PHASE 3: DTW EXTRACTION & PACING
            # ==========================================
            if (
                active_results
                and hasattr(active_results, "pose_landmarks")
                and active_results.pose_landmarks
            ):
                nose_y = active_results.pose_landmarks.landmark[0].y
                if self.recording_state == REC_GOLDEN:
                    self.dtw_analyzer.golden_reference.append(nose_y)
                elif self.recording_state == REC_USER:
                    self.dtw_analyzer.add_user_frame(nose_y)

            live_pacing_msg = ""
            if self.recording_state == REC_USER:
                live_pacing_msg = self.dtw_analyzer.check_live_pacing()

            # ==========================================
            # PHASE 4: CONTROL PANEL RENDERING
            # ==========================================
            if active_frame is not None:
                has_golden = len(self.dtw_analyzer.golden_reference) > 0
                self.renderer.draw_control_panel(
                    active_frame,
                    self.recording_state,
                    self.dtw_last_result,
                    self.frame_count,
                    has_golden,
                    live_pacing_msg,
                )

            # ==========================================
            # PHASE 5: OPENCV WINDOW OUTPUT
            # ==========================================
            if self.current_state == STATE_SINGLE and active_frame is not None:
                cv2.imshow("AURA - Active Tracking", active_frame)
            elif self.current_state == STATE_MIRROR and self.multi_detector is not None:
                if self._is_valid_roi(front_roi):
                    cv2.imshow("AURA - Front View", front_roi)
                if self._is_valid_roi(mirror_roi):
                    cv2.imshow("AURA - Mirror View", mirror_roi)

            # ==========================================
            # PHASE 6: KEYBOARD INPUT HANDLING
            # ==========================================
            cmd = KeyboardController.get_command(self.delay)
            if not self._handle_commands(cmd):
                break

        self.shutdown()

    # --- PRIVATE UTILITY METHODS ---

    def _is_valid_roi(self, roi) -> bool:
        """Checks that ROI dimensions are safe for OpenCV rendering."""
        return (
            roi is not None and roi.ndim == 3 and roi.shape[0] > 0 and roi.shape[1] > 0
        )

    def _check_silent_split(self, frame, w) -> None:
        """Asynchronously checks for mirror presence (Pearson correlation)."""
        if self.frame_count % 10 != 0:
            return

        mid = w // 2
        left_half = cv2.cvtColor(frame[:, :mid], cv2.COLOR_BGR2RGB)
        right_half = cv2.cvtColor(frame[:, mid:], cv2.COLOR_BGR2RGB)

        res_left = self.silent_checker_left.process(left_half)
        res_right = self.silent_checker_right.process(right_half)

        if not (res_left.pose_landmarks and res_right.pose_landmarks):
            self.pearson_buffer_left.clear()
            self.pearson_buffer_right.clear()
            return

        left_x = [
            lm.x * mid for lm in res_left.pose_landmarks.landmark if lm.visibility > 0.5
        ]
        right_x = [
            mid + (lm.x * (w - mid))
            for lm in res_right.pose_landmarks.landmark
            if lm.visibility > 0.5
        ]

        if not (left_x and right_x):
            self.pearson_buffer_left.clear()
            self.pearson_buffer_right.clear()
            return

        body_gap = min(right_x) - max(left_x)
        if body_gap <= w * 0.04:
            self.pearson_buffer_left.clear()
            self.pearson_buffer_right.clear()
            return

        self.pearson_buffer_left.append(res_left.pose_landmarks.landmark[0].y)
        self.pearson_buffer_right.append(res_right.pose_landmarks.landmark[0].y)

        if len(self.pearson_buffer_left) > 15:
            self.pearson_buffer_left.pop(0)
            self.pearson_buffer_right.pop(0)

            n = len(self.pearson_buffer_left)
            mean_l = sum(self.pearson_buffer_left) / n
            mean_r = sum(self.pearson_buffer_right) / n

            num = sum(
                (self.pearson_buffer_left[i] - mean_l)
                * (self.pearson_buffer_right[i] - mean_r)
                for i in range(n)
            )
            den_l = math.sqrt(
                sum((self.pearson_buffer_left[i] - mean_l) ** 2 for i in range(n))
            )
            den_r = math.sqrt(
                sum((self.pearson_buffer_right[i] - mean_r) ** 2 for i in range(n))
            )

            pearson_r = (num / (den_l * den_r)) if (den_l > 0 and den_r > 0) else 0.0

            if pearson_r > 0.95:
                print(f"👀 Mirror detected (r={pearson_r:.2f}). Transitioning...")
                self.current_state = STATE_MIRROR
                cv2.destroyWindow("AURA - Active Tracking")

    def _handle_commands(self, cmd: str) -> bool:
        """Executes business logic based on the input command."""
        if cmd == "QUIT":
            print("❌ Terminating...")
            return False

        elif cmd == "REC_GOLDEN":
            print("🔴 REC: Golden Standard Rep (Trainer)")
            self.recording_state = REC_GOLDEN
            self.dtw_analyzer.golden_reference.clear()
            self.dtw_last_result = None

        elif cmd == "REC_USER":
            if not self.dtw_analyzer.golden_reference:
                print("⚠️ WARNING: Record or Load a Golden Standard first!")
            else:
                print("🔴 REC: User Execution")
                self.recording_state = REC_USER
                self.dtw_analyzer.reset_user_buffer()
                self.dtw_last_result = None

        elif cmd == "STOP_REC":
            if self.recording_state != REC_IDLE:
                print("⏹️ STOP Recording")
                if self.recording_state == REC_USER:
                    self.dtw_last_result = self.dtw_analyzer.evaluate_technique()
                self.recording_state = REC_IDLE

        elif cmd == "SAVE_GOLDEN":
            self.dtw_analyzer.save_golden_standard(exercise=self.exercise)

        elif cmd == "LOAD_GOLDEN":
            self.dtw_analyzer.load_golden_standard_from_file(exercise=self.exercise)

        elif cmd == "VIEW_STATUS":
            print("\n📊 --- AURA STATUS REPORT ---")
            print(f"Target Exercise: {self.exercise}")
            print(
                f"Loaded Golden Frames in Memory: {len(self.dtw_analyzer.golden_reference)}"
            )
            target_path = os.path.join(
                "golden_executions", self.exercise, "golden.json"
            )
            file_exists = os.path.exists(target_path)
            print(
                f"JSON File on Disk ('{target_path}'): {'EXISTS' if file_exists else 'NOT FOUND'}"
            )
            print("-----------------------------\n")

        elif cmd == "DELETE_GOLDEN":
            self.dtw_analyzer.delete_golden_standard_file(exercise=self.exercise)

        return True  # Return True to continue the loop

    def shutdown(self):
        """Releases hardware resources."""
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AURA Engine")
    parser.add_argument(
        "--source", default="0", help="Video source (0 for webcam, or path to video)"
    )
    parser.add_argument("--exercise", default="squats", help="Exercise to analyze")
    args = parser.parse_args()

    try:
        app = AuraEngine(source=args.source, exercise=args.exercise)
        app.run()
    except Exception as e:  # noqa: BLE001
        print(f"❌ Critical System Failure: {e}")
        sys.exit(1)
    finally:
        print("AURA shutdown complete.")


if __name__ == "__main__":
    main()
