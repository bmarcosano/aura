"""
AURA - Main Entry Point (v2.1.0)
Dynamic system transitioning autonomously from Single Mode to Mirror Mode 
using the "Silent Split" kinematic correlation algorithm.
Features real-time Postural HUD feedback and Multi-View Sensor Fusion.
"""

import argparse
import math
import os
import sys
import warnings

import cv2
import mediapipe as mp

# --- HARD SUPPRESSION OF C++ / TENSORFLOW / MEDIAPIPE LOGS ---
# Suppresses annoying backend warnings to keep the terminal clean for AURA logs.
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)

from aura.calibrator import AutoSplitCalibrator
from aura.detector import PoseDetector
from aura.fusion import AuraFusionEngine
from aura.multi_detector import MultiSkeletonDetector

# Define core system execution states
STATE_SINGLE = "SINGLE_MODE"
STATE_MIRROR = "MIRROR_MODE"


def draw_postural_hud(frame, pose_results) -> None:
    """
    Draws a semi-transparent HUD in the top-left corner displaying
    real-time postural alignment metrics. It calculates absolute geometric 
    deviations directly from the MediaPipe raw landmarks.
    """
    _h, _w, _ = frame.shape

    # Default statuses when no subject is detected
    sh_status, sh_color = "WAITING...", (180, 180, 180)
    hip_status, hip_color = "WAITING...", (180, 180, 180)
    head_status, head_color = "WAITING...", (180, 180, 180)

    # Extract landmarks safely from PoseDetector results
    if (
        pose_results
        and hasattr(pose_results, "pose_landmarks")
        and pose_results.pose_landmarks
    ):
        try:
            lm_list = pose_results.pose_landmarks.landmark
            l_shoulder = lm_list[11]
            r_shoulder = lm_list[12]
            l_hip = lm_list[23]
            r_hip = lm_list[24]
            nose = lm_list[0]

            # Calculate vertical deviation between left and right joints
            shoulder_tilt = abs(l_shoulder.y - r_shoulder.y)
            hip_tilt = abs(l_hip.y - r_hip.y)

            # Calculate how far the head deviates from the torso's vertical center
            torso_center_x = (l_shoulder.x + r_shoulder.x) / 2.0
            head_offset = abs(nose.x - torso_center_x)

            # Define visual feedback colors
            COLOR_GREEN = (0, 200, 0)
            COLOR_ORANGE = (0, 140, 255)
            COLOR_RED = (0, 0, 255)

            def get_status(val, warn_thresh, poor_thresh):
                """Evaluates deviation value against specific anatomical thresholds."""
                if val < warn_thresh:
                    return "OPTIMAL", COLOR_GREEN
                elif val < poor_thresh:
                    return "WARNING", COLOR_ORANGE
                else:
                    return "POOR", COLOR_RED

            sh_status, sh_color = get_status(shoulder_tilt, 0.03, 0.07)
            hip_status, hip_color = get_status(hip_tilt, 0.03, 0.07)
            head_status, head_color = get_status(head_offset, 0.04, 0.08)
        except (AttributeError, IndexError):
            pass

    # Draw semi-transparent background box (Top-Left) for readability
    overlay = frame.copy()
    x1, y1, x2, y2 = 15, 15, 275, 145
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (210, 210, 210), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

    # Render HUD Text Lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2

    cv2.putText(
        frame,
        "--- AURA POSTURE HUD ---",
        (x1 + 10, y1 + 22),
        font,
        0.43,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Shoulders: {sh_status}",
        (x1 + 10, y1 + 55),
        font,
        font_scale,
        sh_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Pelvis/Hips: {hip_status}",
        (x1 + 10, y1 + 88),
        font,
        font_scale,
        hip_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Head Balance: {head_status}",
        (x1 + 10, y1 + 121),
        font,
        font_scale,
        head_color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    # --- CLI ARGUMENT SETUP ---
    parser = argparse.ArgumentParser(description="AURA - Autonomous Pose Tracking")
    parser.add_argument(
        "--source", type=str, default="0", help="Video source (0 for webcam)"
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    # --- CORE COMPONENT INITIALIZATION ---
    # Standard detector for Single Mode
    single_detector = PoseDetector()
    
    # Dual-detectors dedicated for the Mirror Mode ROIs
    front_pose_detector = PoseDetector()
    mirror_pose_detector = PoseDetector()

    calibrator = AutoSplitCalibrator()
    multi_detector = None
    fusion_engine = AuraFusionEngine()

    # --- SILENT CHECKER (Motion Sentinel) ---
    # Ultra-lightweight native MediaPipe instances running in the background.
    # Used exclusively to detect mirror reflections without triggering full processing.
    mp_pose = mp.solutions.pose
    silent_checker_left = mp_pose.Pose(min_detection_confidence=0.5)
    silent_checker_right = mp_pose.Pose(min_detection_confidence=0.5)

    current_state = STATE_SINGLE
    frame_count = 0

    print("🚀 AURA Started in SINGLE_MODE.")
    print("Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()

        # Automatic loop if video file ends (useful for testing)
        if not success or frame is None or frame.size == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
            if not success:
                break

        frame_count += 1
        h, w = frame.shape[:2]

        # ==========================================
        # STATE 1: SINGLE MODE
        # Default full-screen execution state.
        # ==========================================
        if current_state == STATE_SINGLE:
            display_frame = frame.copy()

            # Run standard detection and draw skeleton
            single_detector.detect(display_frame, draw=True)

            # Pass detector results directly to the HUD
            draw_postural_hud(display_frame, getattr(single_detector, "results", None))

            cv2.imshow("AURA - Active Tracking", display_frame)

            # Initialize historical tracking buffers for Pearson correlation
            if "pearson_buffer_left" not in locals():
                pearson_buffer_left = []
                pearson_buffer_right = []

            # --- 2. SILENT SPLIT & SYNC ANALYZER ---
            # Every 10 frames (~3 times per second), check for mirror setup.
            # This logic saves immense CPU power by not running every frame.
            if frame_count % 10 == 0:
                mid = w // 2

                # Split frame strictly in half to search for dual subjects
                left_half = cv2.cvtColor(frame[:, :mid], cv2.COLOR_BGR2RGB)
                right_half = cv2.cvtColor(frame[:, mid:], cv2.COLOR_BGR2RGB)

                res_left = silent_checker_left.process(left_half)
                res_right = silent_checker_right.process(right_half)

                if res_left.pose_landmarks and res_right.pose_landmarks:
                    # --- GAP TEST: Verify spatial separation ---
                    # Prevents a single person standing in the middle from being registered as two.
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
                        
                        # Demand a physical gap of at least 4% of screen width
                        min_required_gap = w * 0.04

                        if body_gap > min_required_gap:
                            # --- KINEMATIC GATEKEEPER (PEARSON CORRELATION) ---
                            # Extract the Y coordinate (height) of the nose from both halves
                            y_left = res_left.pose_landmarks.landmark[0].y
                            y_right = res_right.pose_landmarks.landmark[0].y

                            pearson_buffer_left.append(y_left)
                            pearson_buffer_right.append(y_right)

                            # Maintain a rolling window of the last 15 samples (~1.5 seconds)
                            if len(pearson_buffer_left) > 15:
                                pearson_buffer_left.pop(0)
                                pearson_buffer_right.pop(0)

                                # Calculate Pearson Correlation Coefficient (r)
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

                                # Threshold Check: Human imitation is flawed. Only a true physical 
                                # mirror reflection can maintain > 95% perfect statistical synchronization.
                                if pearson_r > 0.95:
                                    print(
                                        f"👀 Synchronized movement confirmed! Mirror detected (r={pearson_r:.2f}). Transitioning..."
                                    )
                                    current_state = STATE_MIRROR
                                    cv2.destroyWindow("AURA - Active Tracking")
                        else:
                            # Flush buffers if the gap closes
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
        # Split-screen execution analyzing sagittal and frontal planes.
        # ==========================================
        elif current_state == STATE_MIRROR:
            # Phase 1: Calibration - Pinpoint the exact optical split line
            if multi_detector is None:
                split_ratio = calibrator.update(frame)

                if split_ratio is not None:
                    print(f"✅ Calibration complete! Split ratio: {split_ratio:.3f}")
                    # Initialize the MultiSkeleton module to slice the dual regions
                    multi_detector = MultiSkeletonDetector(split_ratio=split_ratio)
                    cv2.destroyWindow("AURA - Calibration")
                else:
                    cv2.imshow("AURA - Calibration", frame)
            
            # Phase 2: Continuous Dual-Tracking
            else:
                # Crop the physical frame into two separate matrices
                front_roi, mirror_roi = multi_detector._crop_rois(frame)

                if front_roi.size > 0 and mirror_roi.size > 0:
                    # Run independent MediaPipe inference simultaneously on both perspectives
                    front_pose_detector.detect(front_roi, draw=True)
                    mirror_pose_detector.detect(mirror_roi, draw=True)

                    # --- SENSOR FUSION STEP ---
                    # Combine frontal symmetry parameters (40% weight) with 
                    # mirror profile safety parameters (60% weight, with safety overrides)
                    fusion_result = fusion_engine.evaluate_posture(
                        getattr(front_pose_detector, "results", None),
                        getattr(mirror_pose_detector, "results", None),
                    )

                    # Print Fused Intelligence to console
                    print(
                        f"🔥 [FUSION] Global: {fusion_result['state']} | {fusion_result['front_msg']} || {fusion_result['mirror_msg']}"
                    )
                    
                    # Render independent metric HUDs on both window perspectives
                    draw_postural_hud(
                        front_roi, getattr(front_pose_detector, "results", None)
                    )
                    draw_postural_hud(
                        mirror_roi, getattr(mirror_pose_detector, "results", None)
                    )

                    # Superimpose the unified global fusion verdict on the primary view
                    status_text = f"FUSION: {fusion_result['state']}"
                    cv2.putText(
                        front_roi,
                        status_text,
                        (15, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        fusion_result["color"],
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("AURA - Front View", front_roi)
                    cv2.imshow("AURA - Mirror View", mirror_roi)

        # --- KEYBOARD LISTENER ---
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("❌ Exit command received. Terminating...")
            break

    # Graceful teardown
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()