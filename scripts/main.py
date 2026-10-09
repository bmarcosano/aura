"""
AURA - Main Entry Point (v2.6.0 - Full File Management & UI Update)
Dynamic biomechanical analysis system transitioning autonomously from Single Mode
to Mirror Mode using a lightweight "Silent Split" correlation algorithm.
Features real-time Postural HUD, Multi-View Sensor Fusion, DTW Temporal Analysis,
JSON Data Persistence with in-app management, and live pacing feedback.
"""

import argparse
import math
import os
import sys
import warnings

import cv2
import mediapipe as mp

# --- HARD SUPPRESSION OF C++ / TENSORFLOW / MEDIAPIPE LOGS ---
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)

from aura.calibrator import AutoSplitCalibrator
from aura.detector import PoseDetector
from aura.fusion import AuraFusionEngine
from aura.multi_detector import MultiSkeletonDetector
from aura.temporal import DTWTemporalAnalyzer

# Define core system states
STATE_SINGLE = "SINGLE_MODE"
STATE_MIRROR = "MIRROR_MODE"

# DTW Recording States
REC_IDLE = "IDLE"
REC_GOLDEN = "RECORD_GOLDEN"
REC_USER = "RECORD_USER"


def draw_control_panel(
    frame, recording_state, dtw_result, frame_count, has_golden, live_pacing_msg
) -> None:
    """
    Renders an on-screen sidebar menu providing visual instructions,
    current recording status, loaded assets, live pacing feedback,
    and the latest DTW evaluation results.
    """
    _, w, _ = frame.shape

    panel_w = 280
    panel_h = 340
    x1, y1 = w - panel_w - 15, 15
    x2, y2 = w - 15, y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- 1. TITLE & MENU CONTROLS ---
    cv2.putText(
        frame,
        "--- AURA CONTROLS ---",
        (x1 + 15, y1 + 22),
        font,
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
            font,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.line(frame, (x1 + 15, y1 + 195), (x2 - 15, y1 + 195), (100, 100, 100), 1)

    # --- 2. RECORDING STATUS ---
    cv2.putText(
        frame,
        "STATUS:",
        (x1 + 15, y1 + 215),
        font,
        0.43,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if recording_state == REC_IDLE:
        cv2.putText(
            frame,
            "WAITING / IDLE",
            (x1 + 75, y1 + 215),
            font,
            0.42,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    else:
        if frame_count % 15 < 10:
            cv2.circle(frame, (x1 + 80, y1 + 211), 5, (0, 0, 255), -1)

        status_text = "GOLDEN" if recording_state == REC_GOLDEN else "USER"
        cv2.putText(
            frame,
            f"REC {status_text}",
            (x1 + 92, y1 + 215),
            font,
            0.42,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    mem_color = (0, 200, 0) if has_golden else (0, 0, 255)
    mem_text = "Mem: READY" if has_golden else "Mem: EMPTY"
    cv2.putText(
        frame, mem_text, (x2 - 95, y1 + 215), font, 0.38, mem_color, 1, cv2.LINE_AA
    )

    # --- 3. LIVE PACING FEEDBACK ---
    if recording_state == REC_USER and live_pacing_msg:
        cv2.putText(
            frame,
            f"{live_pacing_msg}",
            (x1 + 15, y1 + 242),
            font,
            0.38,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

    # --- 4. DTW EVALUATION RESULT ---
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
            font,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{score}/100",
            (x1 + 15, y1 + 300),
            font,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, feedback, (x1 + 95, y1 + 298), font, 0.38, color, 1, cv2.LINE_AA
        )


def draw_postural_hud(frame, pose_results) -> None:
    """Draws the Postural alignment metrics HUD in the top-left corner."""
    _h, _w, _ = frame.shape
    sh_status, sh_color = "WAITING...", (180, 180, 180)
    hip_status, hip_color = "WAITING...", (180, 180, 180)
    head_status, head_color = "WAITING...", (180, 180, 180)

    if (
        pose_results
        and hasattr(pose_results, "pose_landmarks")
        and pose_results.pose_landmarks
    ):
        try:
            lm_list = pose_results.pose_landmarks.landmark
            l_shoulder, r_shoulder = lm_list[11], lm_list[12]
            l_hip, r_hip = lm_list[23], lm_list[24]
            nose = lm_list[0]

            shoulder_tilt = abs(l_shoulder.y - r_shoulder.y)
            hip_tilt = abs(l_hip.y - r_hip.y)
            head_offset = abs(nose.x - ((l_shoulder.x + r_shoulder.x) / 2.0))

            def get_status(val, warn_thresh, poor_thresh):
                if val < warn_thresh:
                    return "OPTIMAL", (0, 200, 0)
                elif val < poor_thresh:
                    return "WARNING", (0, 140, 255)
                else:
                    return "POOR", (0, 0, 255)

            sh_status, sh_color = get_status(shoulder_tilt, 0.03, 0.07)
            hip_status, hip_color = get_status(hip_tilt, 0.03, 0.07)
            head_status, head_color = get_status(head_offset, 0.04, 0.08)
        except (AttributeError, IndexError):
            pass

    overlay = frame.copy()
    x1, y1, x2, y2 = 15, 15, 275, 145
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        "--- POSTURE HUD ---",
        (x1 + 10, y1 + 22),
        font,
        0.43,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Shoulders: {sh_status}",
        (x1 + 10, y1 + 55),
        font,
        0.52,
        sh_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Pelvis/Hips: {hip_status}",
        (x1 + 10, y1 + 88),
        font,
        0.52,
        hip_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Head Balance: {head_status}",
        (x1 + 10, y1 + 121),
        font,
        0.52,
        head_color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA - Autonomous Pose Tracking")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source path or 0 for local webcam",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="squats",
        help="Target exercise subfolder name (e.g., squats, push_up)",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    # --- COMPONENT INITIALIZATION ---
    single_detector = PoseDetector()
    front_pose_detector = PoseDetector()
    mirror_pose_detector = PoseDetector()

    calibrator = AutoSplitCalibrator()
    multi_detector = None
    fusion_engine = AuraFusionEngine()

    dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)
    recording_state = REC_IDLE
    dtw_last_result = None

    mp_pose = mp.solutions.pose
    silent_checker_left = mp_pose.Pose(min_detection_confidence=0.5)
    silent_checker_right = mp_pose.Pose(min_detection_confidence=0.5)

    current_state = STATE_SINGLE
    frame_count = 0

    print(f"🚀 AURA Started in SINGLE_MODE (Source: {source})")

    while cap.isOpened():
        success, frame = cap.read()

        if not success or frame is None or frame.size == 0:
            if recording_state != REC_IDLE:
                print("\n⚠️ Video ended during recording. Auto-stopping to evaluate...")
                if recording_state == REC_USER:
                    dtw_last_result = dtw_analyzer.evaluate_technique()
                recording_state = REC_IDLE

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
            if not success:
                break

        frame_count += 1
        h, w = frame.shape[:2]
        active_frame = None
        active_results = None
        front_roi = None
        mirror_roi = None

        # ==========================================
        # STATE 1: SINGLE MODE
        # ==========================================
        if current_state == STATE_SINGLE:
            display_frame = frame.copy()
            active_frame = display_frame

            single_detector.detect(display_frame, draw=True)
            active_results = getattr(single_detector, "results", None)
            draw_postural_hud(display_frame, active_results)

            if "pearson_buffer_left" not in locals():
                pearson_buffer_left = []
                pearson_buffer_right = []

            # ---------------------------------------------------------
            # SILENT SPLIT LOGIC
            # ---------------------------------------------------------
            if frame_count % 10 == 0:
                mid = w // 2
                left_half = cv2.cvtColor(frame[:, :mid], cv2.COLOR_BGR2RGB)
                right_half = cv2.cvtColor(frame[:, mid:], cv2.COLOR_BGR2RGB)

                res_left = silent_checker_left.process(left_half)
                res_right = silent_checker_right.process(right_half)

                if res_left.pose_landmarks and res_right.pose_landmarks:
                    left_x_coords = [
                        lm.x * mid
                        for lm in res_left.pose_landmarks.landmark
                        if lm.visibility > 0.5
                    ]
                    right_x_coords = [
                        mid + (lm.x * (w - mid))
                        for lm in res_right.pose_landmarks.landmark
                        if lm.visibility > 0.5
                    ]

                    if left_x_coords and right_x_coords:
                        max_x_left = max(left_x_coords)
                        min_x_right = min(right_x_coords)
                        body_gap = min_x_right - max_x_left
                        min_required_gap = w * 0.04

                        if body_gap > min_required_gap:
                            y_left = res_left.pose_landmarks.landmark[0].y
                            y_right = res_right.pose_landmarks.landmark[0].y

                            pearson_buffer_left.append(y_left)
                            pearson_buffer_right.append(y_right)

                            if len(pearson_buffer_left) > 15:
                                pearson_buffer_left.pop(0)
                                pearson_buffer_right.pop(0)

                                n = len(pearson_buffer_left)
                                mean_l = sum(pearson_buffer_left) / n
                                mean_r = sum(pearson_buffer_right) / n

                                num = sum(
                                    (pearson_buffer_left[i] - mean_l)
                                    * (pearson_buffer_right[i] - mean_r)
                                    for i in range(n)
                                )
                                den_l = math.sqrt(
                                    sum(
                                        (pearson_buffer_left[i] - mean_l) ** 2
                                        for i in range(n)
                                    )
                                )
                                den_r = math.sqrt(
                                    sum(
                                        (pearson_buffer_right[i] - mean_r) ** 2
                                        for i in range(n)
                                    )
                                )

                                pearson_r = (
                                    (num / (den_l * den_r))
                                    if (den_l > 0 and den_r > 0)
                                    else 0.0
                                )

                                if pearson_r > 0.95:
                                    print(
                                        f"👀 Mirror detected (r={pearson_r:.2f}). Transitioning..."
                                    )
                                    current_state = STATE_MIRROR
                                    cv2.destroyWindow("AURA - Active Tracking")
                        else:
                            pearson_buffer_left.clear()
                            pearson_buffer_right.clear()
                    else:
                        pearson_buffer_left.clear()
                        pearson_buffer_right.clear()
                else:
                    pearson_buffer_left.clear()
                    pearson_buffer_right.clear()

        # ==========================================
        # STATE 2: MIRROR MODE
        # ==========================================
        elif current_state == STATE_MIRROR:
            if multi_detector is None:
                split_ratio = calibrator.update(frame)
                if split_ratio is not None:
                    multi_detector = MultiSkeletonDetector(split_ratio=split_ratio)
                    cv2.destroyWindow("AURA - Calibration")
                else:
                    cv2.imshow("AURA - Calibration", frame)
            else:
                front_roi, mirror_roi = multi_detector._crop_rois(frame)

                if (
                    front_roi is not None
                    and front_roi.size > 0
                    and mirror_roi is not None
                    and mirror_roi.size > 0
                ):
                    front_pose_detector.detect(front_roi, draw=True)
                    mirror_pose_detector.detect(mirror_roi, draw=True)

                    active_frame = front_roi
                    active_results = getattr(front_pose_detector, "results", None)

                    fusion_result = fusion_engine.evaluate_posture(
                        active_results,
                        getattr(mirror_pose_detector, "results", None),
                    )

                    draw_postural_hud(front_roi, active_results)
                    draw_postural_hud(
                        mirror_roi, getattr(mirror_pose_detector, "results", None)
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

        # ==========================================
        # DTW DATA EXTRACTION & LIVE PACING EVALUATION
        # ==========================================
        if (
            active_results
            and hasattr(active_results, "pose_landmarks")
            and active_results.pose_landmarks
        ):
            nose_y = active_results.pose_landmarks.landmark[0].y

            if recording_state == REC_GOLDEN:
                dtw_analyzer.golden_reference.append(nose_y)
            elif recording_state == REC_USER:
                dtw_analyzer.add_user_frame(nose_y)

        # Evaluate live pacing feedback in real-time during user execution
        live_pacing_msg = ""
        if recording_state == REC_USER:
            live_pacing_msg = dtw_analyzer.check_live_pacing()

        # Overlay the Control Panel on the active tracking view
        if active_frame is not None:
            has_golden = len(dtw_analyzer.golden_reference) > 0
            draw_control_panel(
                active_frame,
                recording_state,
                dtw_last_result,
                frame_count,
                has_golden,
                live_pacing_msg,
            )

        # Render Windows with safe dimension checking
        if current_state == STATE_SINGLE:
            if display_frame is not None and display_frame.size > 0:
                cv2.imshow("AURA - Active Tracking", display_frame)
        elif current_state == STATE_MIRROR and multi_detector is not None:
            if (
                front_roi is not None
                and front_roi.ndim == 3
                and front_roi.shape[0] > 0
                and front_roi.shape[1] > 0
            ):
                cv2.imshow("AURA - Front View", front_roi)
            if (
                mirror_roi is not None
                and mirror_roi.ndim == 3
                and mirror_roi.shape[0] > 0
                and mirror_roi.shape[1] > 0
            ):
                cv2.imshow("AURA - Mirror View", mirror_roi)

        # ==========================================
        # KEYBOARD CONTROLS (Including File Management)
        # ==========================================
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            print("❌ Terminating...")
            break
        elif key == ord("g"):
            print("🔴 REC: Golden Standard Rep (Trainer)")
            recording_state = REC_GOLDEN
            dtw_analyzer.golden_reference.clear()
            dtw_last_result = None
        elif key == ord("u"):
            if not dtw_analyzer.golden_reference:
                print("⚠️ WARNING: Record or Load a Golden Standard first!")
            else:
                print("🔴 REC: User Execution")
                recording_state = REC_USER
                dtw_analyzer.reset_user_buffer()
                dtw_last_result = None
        elif key == ord("s"):
            if recording_state != REC_IDLE:
                print("⏹️ STOP Recording")
                if recording_state == REC_USER:
                    dtw_last_result = dtw_analyzer.evaluate_technique()
                recording_state = REC_IDLE
        elif key == ord("w"):
            dtw_analyzer.save_golden_standard(exercise=args.exercise)
        elif key == ord("l"):
            dtw_analyzer.load_golden_standard_from_file(exercise=args.exercise)
        elif key == ord("v"):
            print("\n📊 --- AURA STATUS REPORT ---")
            print(f"Target Exercise: {args.exercise}")
            print(
                f"Loaded Golden Frames in Memory: {len(dtw_analyzer.golden_reference)}"
            )
            target_path = os.path.join(
                "golden_executions", args.exercise, "golden.json"
            )
            file_exists = os.path.exists(target_path)
            print(
                f"JSON File on Disk ('{target_path}'): {'EXISTS' if file_exists else 'NOT FOUND'}"
            )
            print("-----------------------------\n")
        elif key == ord("d"):
            dtw_analyzer.delete_golden_standard_file(exercise=args.exercise)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
