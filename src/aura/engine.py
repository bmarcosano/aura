"""
engine.py: Core application loop and state management for AURA.
Includes multi-marker ArUco tracking for absolute spatial anchoring,
Markerless SVD Fallback, dynamic floor grounding, Auto Splitter UI window,
and unified 3D Digital Twin rendering.
"""

from __future__ import annotations

import math
import os

import cv2
import mediapipe as mp
import numpy as np

from aura.analyzer import PostureAnalyzer
from aura.detector import PoseDetector
from aura.logger import AuraLogger
from aura.mirror_plane import MirrorPlaneEstimator

# New Refactored Modules
from aura.multi_detector import AnthropometricValidator, MultiSkeletonDetector
from aura.reconstruction import StereoReconstructor

# UI and Input modules
from aura.renderer import AuraRenderer
from aura.renderer_3d import Aura3DViewer
from aura.roi_splitter import AutoROISplitter
from aura.temporal import DTWTemporalAnalyzer

# Aruco Tracking for absolute spatial metrology
from aura.utils.aruco_tracker import ArucoMultiTracker
from aura.utils.input_handler import KeyboardController

# State Constants
STATE_SINGLE = "SINGLE_MODE"
STATE_MIRROR = "MIRROR_MODE"
REC_IDLE = "IDLE"
REC_GOLDEN = "RECORD_GOLDEN"
REC_USER = "RECORD_USER"


class AuraEngine:
    """Orchestrates the camera feed, biomechanical models, ArUco tracking, and UI rendering."""

    def __init__(self, source: str | int = "0", exercise: str = "squats") -> None:
        self.source = int(source) if str(source).isdigit() else source
        self.exercise = exercise
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"❌ Error: Could not open video source '{self.source}'")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / fps) if fps > 0 else 33

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- CAMERA CALIBRATION & ARUCO ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.apply_calibration = False

        loaded_camera_matrix = None
        loaded_dist_coeffs = None
        calib_path = "camera_calibration.npz"

        if os.path.exists(calib_path):
            with np.load(calib_path) as data:
                loaded_camera_matrix = data["camera_matrix"]
                loaded_dist_coeffs = data["dist_coeffs"]

        if isinstance(self.source, int):
            # Physical webcam: default to applying calibration if available
            if loaded_camera_matrix is not None:
                self.apply_calibration = True
                self.camera_matrix = loaded_camera_matrix
                self.dist_coeffs = loaded_dist_coeffs
        else:
            # Video file: prompt the user, as the physical lens might not match the video
            print(f"\n[INFO] Video source detected: {self.source}")
            ans = (
                input("Apply optical calibration (undistort)? [y/N]: ").strip().lower()
            )
            if ans == "y" and loaded_camera_matrix is not None:
                self.apply_calibration = True
                self.camera_matrix = loaded_camera_matrix
                self.dist_coeffs = loaded_dist_coeffs
            else:
                self.apply_calibration = False

        # Fallback: Generate a generic pinhole camera matrix for synthetic or uncalibrated video
        # This prevents solvePnP and triangulation from failing due to mismatched focal lengths.
        if self.camera_matrix is None:
            focal_length = float(frame_width)
            self.camera_matrix = np.array(
                [
                    [focal_length, 0, frame_width / 2.0],
                    [0, focal_length, frame_height / 2.0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            print(
                f"[INFO] Using generic pinhole camera matrix (Focal length: {focal_length}).\n"
            )

        # Initialize ArUco tracker (marker size accurately set to 18cm = 0.18m)
        self.aruco_tracker = ArucoMultiTracker(marker_size_m=0.18)

        # --- COMPONENT INITIALIZATION ---
        self.single_detector = PoseDetector()
        self.posture_analyzer = PostureAnalyzer()

        self.splitter = AutoROISplitter()
        self.multi_detector: MultiSkeletonDetector | None = None

        # Structural Subsystems
        self.reconstructor: StereoReconstructor | None = None
        if self.camera_matrix is not None:
            self.reconstructor = StereoReconstructor(self.camera_matrix)

        self.svd_estimator = MirrorPlaneEstimator(ema_alpha=0.15)
        self.validator = AnthropometricValidator(units="meters")
        self.dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)

        # UI & Rendering
        self.renderer = AuraRenderer()
        self.viewer_3d = Aura3DViewer(width=800, height=600)
        self.sys_logger = AuraLogger()

        # Silent Split setup (Temporal coherence validation)
        mp_pose = mp.solutions.pose
        self.silent_checker_left = mp_pose.Pose(min_detection_confidence=0.5)
        self.silent_checker_right = mp_pose.Pose(min_detection_confidence=0.5)
        self.pearson_buffer_left: list[float] = []
        self.pearson_buffer_right: list[float] = []

        # System state
        self.current_state = STATE_SINGLE
        self.recording_state = REC_IDLE
        self.dtw_last_result = None
        self.frame_count = 0
        self._window_placed = False

    def _ground_landmarks(self, landmarks_3d: np.ndarray | None) -> np.ndarray | None:
        """
        Dynamically grounds raw 3D world landmarks onto the floor grid (Y=0) every frame.
        Finds the absolute lowest point of the feet and forces it to Y=0, absorbing
        the squat dynamically without vertical floating.
        """
        if landmarks_3d is None or len(landmarks_3d) != 33:
            return landmarks_3d

        grounded = landmarks_3d.copy()
        feet_indices = [27, 28, 29, 30, 31, 32]  # Ankles, heels, toes

        # Use np.max for Y to guarantee the lowest physical point touches the floor
        target_y = float(np.max(grounded[feet_indices, 1]))
        target_x = float((grounded[27, 0] + grounded[28, 0]) / 2.0)
        target_z = float((grounded[27, 2] + grounded[28, 2]) / 2.0)

        # Instant subtraction locks the feet centroid at (X=0, Z=0) and height Y=0
        grounded[:, 0] -= target_x
        grounded[:, 1] -= target_y
        grounded[:, 2] -= target_z

        return grounded

    def _evaluate_joint_colors(
        self, active_results, is_fused_valid: bool = True
    ) -> dict[int, str]:
        """
        Evaluates 3D joint posture against biomechanical thresholds and temporal DTW,
        returning a color mapping for the 3D viewport joints.
        """
        joint_colors: dict[int, str] = {}

        if (
            not active_results
            or not hasattr(active_results, "pose_landmarks")
            or not active_results.pose_landmarks
        ):
            return joint_colors

        landmarks = active_results.pose_landmarks.landmark

        left_hip = np.array([landmarks[23].x, landmarks[23].y])
        left_knee = np.array([landmarks[25].x, landmarks[25].y])
        left_ankle = np.array([landmarks[27].x, landmarks[27].y])

        right_hip = np.array([landmarks[24].x, landmarks[24].y])
        right_knee = np.array([landmarks[26].x, landmarks[26].y])
        right_ankle = np.array([landmarks[28].x, landmarks[28].y])

        def compute_angle(a, b, c):
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (
                np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
            )
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

        left_knee_angle = compute_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = compute_angle(right_hip, right_knee, right_ankle)

        for knee_idx, knee_angle in [(25, left_knee_angle), (26, right_knee_angle)]:
            if knee_angle < 70.0:
                joint_colors[knee_idx] = "warning"
            elif knee_angle > 175.0:
                joint_colors[knee_idx] = "good"
            else:
                joint_colors[knee_idx] = "good"

        if not is_fused_valid:
            joint_colors[23] = "bad"
            joint_colors[24] = "bad"
            joint_colors[11] = "bad"
            joint_colors[12] = "bad"

        if self.recording_state == REC_USER and self.dtw_last_result:
            distance = 0.0
            if isinstance(self.dtw_last_result, dict):
                distance = self.dtw_last_result.get("deviation_per_frame", 0)
            elif (
                isinstance(self.dtw_last_result, tuple)
                and len(self.dtw_last_result) > 0
            ):
                distance = self.dtw_last_result[0]

            if distance > 1.5:
                joint_colors[27] = "warning"
                joint_colors[28] = "warning"

        return joint_colors

    def run(self) -> None:
        """Starts the main video loop."""

        init_log = "AURA System Initialized. Looking for user..."
        if self.camera_matrix is not None:
            status = "applied" if self.apply_calibration else "loaded (not applied)"
            init_log = (
                f"AURA Initialized. Calibration file {status}. Looking for user..."
            )

        self.sys_logger.log(
            init_log,
            f"AuraEngine booted. Source: {self.source}, Target: {self.exercise}. FSM State: {STATE_SINGLE}.",
        )

        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success or frame is None or frame.size == 0:
                if self.recording_state != REC_IDLE:
                    self.sys_logger.log(
                        "Video ended. Auto-stopping evaluation.",
                        "Stream EOF detected. Forcing FSM REC_IDLE fallback.",
                    )
                    if self.recording_state == REC_USER:
                        self.dtw_last_result = self.dtw_analyzer.evaluate_technique()
                    self.recording_state = REC_IDLE

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
                if not success:
                    break

            self.frame_count += 1

            # --- PHASE 1: OPTICAL CORRECTION AND ARUCO TRACKING ---
            aruco_poses = {}
            active_dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            if self.camera_matrix is not None and self.dist_coeffs is not None:
                if self.apply_calibration:
                    # Physically undistort the frame using calibration data
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                else:
                    # Frame remains distorted, pass native coefficients for tracking math
                    active_dist_coeffs = self.dist_coeffs

                frame, aruco_poses = self.aruco_tracker.process_frame(
                    frame, self.camera_matrix, active_dist_coeffs
                )

            # Extract ArUco tracking data for Mirror Plane
            aruco_rvec = None
            aruco_tvec = None

            if 0 in aruco_poses:
                _rvec_floor, tvec_floor = aruco_poses[0]
                floor_dist = np.linalg.norm(tvec_floor)
                self.sys_logger.log_metric("Floor Dist", f"{floor_dist:.2f} m")

            if 1 in aruco_poses:
                aruco_rvec, aruco_tvec = aruco_poses[1]
                mirror_dist = np.linalg.norm(aruco_tvec)
                self.sys_logger.log_metric("Mirror Dist", f"{mirror_dist:.2f} m")

            _, w = frame.shape[:2]
            active_frame = None
            active_results = None
            front_roi = None
            mirror_roi = None
            fused_3d = None
            plane_normal = None

            is_stereo_validated = False
            hud_status = "3D FUSION: SINGLE"

            # ==========================================
            # PHASE 2: SINGLE MODE
            # ==========================================
            if self.current_state == STATE_SINGLE:
                display_frame = frame.copy()
                active_frame = display_frame

                self.single_detector.detect(display_frame, draw=True)
                active_results = getattr(self.single_detector, "results", None)

                raw_landmarks = self.single_detector.get_world_landmark_array()
                fused_3d = self._ground_landmarks(raw_landmarks)

                self.renderer.draw_postural_hud(
                    display_frame,
                    active_results,
                    self.posture_analyzer,
                    is_mirror_view=False,
                )
                self._check_silent_split(frame, w)

            # ==========================================
            # PHASE 3: MIRROR MODE (THE NEW PIPELINE)
            # ==========================================
            elif self.current_state == STATE_MIRROR:
                if self.multi_detector is None:
                    split_ratio = self.splitter.update(frame)
                    window_title = "Auto Splitter: Detecting..."
                    if split_ratio is not None:
                        self.sys_logger.log(
                            "Auto Splitter calibration successful.",
                            "AutoROISplitter finalized ratio. Instantiating MultiSkeletonDetector.",
                        )
                        self.multi_detector = MultiSkeletonDetector(
                            split_ratio=split_ratio
                        )
                        cv2.destroyWindow(window_title)
                    else:
                        cv2.imshow(window_title, frame)
                        self.single_detector.detect(frame, draw=False)
                        raw_landmarks = self.single_detector.get_world_landmark_array()
                        fused_3d = self._ground_landmarks(raw_landmarks)
                        active_frame = frame
                else:
                    # 1. Extract 2D Pixel Landmarks
                    pipeline_data = self.multi_detector.process_frame(frame)

                    for user_msg, expert_msg in pipeline_data.get("logs", []):
                        self.sys_logger.log(user_msg, expert_msg)

                    front_roi = pipeline_data["front_roi"]
                    mirror_roi = pipeline_data["mirror_roi"]
                    active_results = pipeline_data["front_results"]
                    mirror_results = pipeline_data["mirror_results"]

                    direct_pose_2d = pipeline_data.get("direct_pose_2d")
                    reflected_pose_2d = pipeline_data.get("reflected_pose_2d")

                    active_frame = front_roi
                    reflection_matrix = None

                    # 2. Attempt Full Stereoscopic Pipeline (Priority Route)
                    if (
                        direct_pose_2d is not None
                        and reflected_pose_2d is not None
                        and self.reconstructor is not None
                    ):
                        direct_3d_local = self.multi_detector.front_detector.get_world_landmark_array()
                        reflected_3d_local = self.multi_detector.mirror_detector.get_world_landmark_array()

                        # --- PLANE ESTIMATION ---
                        if aruco_rvec is not None and aruco_tvec is not None:
                            # Mode 1: Exact ArUco Tracker Metric Plane (Physical Setup)
                            mirror_data = self.aruco_tracker.estimate_mirror_plane(
                                aruco_rvec, aruco_tvec
                            )
                            reflection_matrix = mirror_data["reflection_matrix"]
                            plane_normal = mirror_data["normal"]
                            hud_status = "3D FUSION: STEREO (ARUCO)"
                        else:
                            # Mode 2: SVD Markerless Fallback (Synthetic/Uncalibrated Video)
                            if (
                                direct_3d_local is not None
                                and reflected_3d_local is not None
                            ):
                                # In synthetic video, MediaPipe's world landmarks are usually upright.
                                # We revert the mirror's horizontal flip to restore spatial symmetry
                                reflected_3d_unflipped = reflected_3d_local.copy()
                                reflected_3d_unflipped[:, 0] *= -1

                                # Calculate the mirror plane directly from the relative 3D vectors
                                # without attempting to project them into camera space via solvePnP
                                reflection_matrix, plane_normal = (
                                    self.svd_estimator.estimate_plane(
                                        direct_3d_local, reflected_3d_unflipped
                                    )
                                )

                                # In purely uncalibrated scenarios, the SVD plane often rests at Z=0.
                                # We add a small artificial depth (e.g., 2 meters) to the translation
                                # vector so the triangulated points don't collapse into the camera lens.
                                if reflection_matrix is not None:
                                    reflection_matrix[2, 3] = 2.0
                                    hud_status = "3D FUSION: STEREO (SVD)"

                        # --- TRIANGULATION & CORRECTION ---
                        if reflection_matrix is not None:
                            raw_landmarks_3d = self.reconstructor.triangulate_landmarks(
                                direct_pose_2d, reflected_pose_2d, reflection_matrix
                            )

                            # Validate metric integrity against anatomical thresholds
                            if self.validator.validate_skeleton_3d(raw_landmarks_3d):
                                fused_3d = self._ground_landmarks(raw_landmarks_3d)
                                is_stereo_validated = True
                            else:
                                self.sys_logger.log(
                                    "Biometric invalidation.",
                                    "AnthropometricValidator rejected 3D lengths. Falling back to single view.",
                                )

                    # 3. Fallback Route (Base Frontal Pose)
                    if not is_stereo_validated and active_results:
                        front_world_lm = self.multi_detector.front_detector.get_world_landmark_array()
                        fused_3d = self._ground_landmarks(front_world_lm)
                        hud_status = "3D FUSION: SINGLE"

                    # HUD Drawing
                    if self._is_valid_roi(front_roi):
                        self.renderer.draw_postural_hud(
                            front_roi,
                            active_results,
                            self.posture_analyzer,
                            is_mirror_view=False,
                        )
                        cv2.putText(
                            front_roi,
                            hud_status,
                            (15, 175),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0) if is_stereo_validated else (0, 0, 255),
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
            # PHASE 4: KINEMATIC EXTRACTION & PACING
            # ==========================================
            # Protect Golden Standard recordings from single-view relative estimations
            if fused_3d is not None:
                if is_stereo_validated or self.current_state == STATE_SINGLE:
                    kinematic_features = self.dtw_analyzer.extract_kinematic_features(
                        fused_3d
                    )

                    if self.recording_state == REC_GOLDEN:
                        self.dtw_analyzer.golden_reference.append(
                            kinematic_features.tolist()
                        )
                    elif self.recording_state == REC_USER:
                        self.dtw_analyzer.add_user_frame(kinematic_features)
                else:
                    if self.recording_state in [REC_GOLDEN, REC_USER]:
                        self.sys_logger.log(
                            "Warning: Frame dropped from recording.",
                            "Stereo triangulation inactive. Cannot record monocular fallback into Golden Standard.",
                        )

            live_pacing_msg = ""
            if self.recording_state == REC_USER:
                live_pacing_msg = self.dtw_analyzer.check_live_pacing()

            # ==========================================
            # PHASE 5: UI & UNIFIED 3D DIGITAL TWIN RENDER
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

            if self.current_state == STATE_SINGLE and active_frame is not None:
                cv2.imshow("AURA - Active Tracking", active_frame)
            elif self.current_state == STATE_MIRROR and self.multi_detector is not None:
                if self._is_valid_roi(front_roi):
                    cv2.imshow("AURA - Front View", front_roi)
                if self._is_valid_roi(mirror_roi):
                    cv2.imshow("AURA - Mirror View", mirror_roi)

            joint_colors = self._evaluate_joint_colors(
                active_results, is_stereo_validated
            )

            cv2.imshow(
                "AURA - 3D Digital Twin & SVD Inspector",
                self.viewer_3d.render_scene(fused_3d, plane_normal, joint_colors),
            )
            cv2.imshow("AURA - System Logs", self.sys_logger.render_events())
            cv2.imshow("AURA - Live Telemetry", self.sys_logger.render_metrics())

            if not self._window_placed:
                cv2.moveWindow("AURA - System Logs", 0, 0)
                self._window_placed = True

            # ==========================================
            # PHASE 6: KEYBOARD INPUT HANDLING
            # ==========================================
            cmd, raw_key = KeyboardController.get_command(self.delay)

            if raw_key != -1:
                self.viewer_3d.handle_keyboard(raw_key)

            if cmd != "NONE" and not self._handle_commands(cmd):
                break

        self.shutdown()

    def _is_valid_roi(self, roi: np.ndarray | None) -> bool:
        return (
            roi is not None and roi.ndim == 3 and roi.shape[0] > 0 and roi.shape[1] > 0
        )

    def _check_silent_split(self, frame: np.ndarray, w: int) -> None:
        """Evaluates Pearson correlation between the two halves to trigger the MIRROR_MODE."""
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
                self.sys_logger.log(
                    "Mirror detected! Processing dual views...",
                    f"Temporal correlation positive (Pearson r={pearson_r:.2f} > 0.95). FSM State -> MIRROR_MODE.",
                )
                self.current_state = STATE_MIRROR
                cv2.destroyWindow("AURA - Active Tracking")

    def _handle_commands(self, cmd: str) -> bool:
        if cmd == "QUIT":
            self.sys_logger.log(
                "Shutting down...", "SIGINT received. Terminating AURA gracefully."
            )
            return False
        elif cmd == "TOGGLE_LOG_MODE":
            self.sys_logger.toggle_mode()
        elif cmd == "REC_GOLDEN":
            self.sys_logger.log(
                "Starting Trainer Recording.",
                "Initiating REC_GOLDEN state. Buffer cleared.",
            )
            self.recording_state = REC_GOLDEN
            self.dtw_analyzer.golden_reference.clear()
            self.dtw_last_result = None
        elif cmd == "REC_USER":
            if not self.dtw_analyzer.golden_reference:
                self.sys_logger.log(
                    "Error: Need a Trainer recording first!",
                    "REC_USER aborted. DTW Golden reference array is empty.",
                )
            else:
                self.sys_logger.log(
                    "Starting User Recording.",
                    "Initiating REC_USER state. Live DTW tracing started.",
                )
                self.recording_state = REC_USER
                self.dtw_analyzer.reset_user_buffer()
                self.dtw_last_result = None
        elif cmd == "STOP_REC":
            if self.recording_state != REC_IDLE:
                self.sys_logger.log(
                    "Recording stopped. Analyzing technique...",
                    f"FSM Transition {self.recording_state} -> REC_IDLE.",
                )
                if self.recording_state == REC_USER:
                    self.dtw_last_result = self.dtw_analyzer.evaluate_technique()

                    raw_dist = 0.0
                    if isinstance(self.dtw_last_result, dict):
                        raw_dist = self.dtw_last_result.get("raw_distance", 0)
                    elif (
                        isinstance(self.dtw_last_result, tuple)
                        and len(self.dtw_last_result) > 1
                    ):
                        raw_dist = self.dtw_last_result[1]

                    self.sys_logger.log(
                        "Analysis complete.", f"DTW Path Cost Distance: {raw_dist:.2f}"
                    )
                self.recording_state = REC_IDLE
        elif cmd == "SAVE_GOLDEN":
            self.sys_logger.log(
                "Saving Trainer reference...", "Serializing golden reference to JSON."
            )
            self.dtw_analyzer.save_golden_standard(exercise=self.exercise)
        elif cmd == "LOAD_GOLDEN":
            self.sys_logger.log(
                "Loading Trainer reference...",
                f"Deserializing golden reference from disk (exercise={self.exercise}).",
            )
            self.dtw_analyzer.load_golden_standard_from_file(exercise=self.exercise)
        elif cmd == "VIEW_STATUS":
            self.sys_logger.log(
                "Printing system status to console.",
                "VIEW_STATUS requested. Dumping metrics to stdout.",
            )
            print("\n📊 --- AURA STATUS REPORT ---")
            print(f"Target Exercise: {self.exercise}")
        elif cmd == "DELETE_GOLDEN":
            self.sys_logger.log(
                "Deleted Trainer reference.",
                "Unlinking golden reference JSON from disk.",
            )
            self.dtw_analyzer.delete_golden_standard_file(exercise=self.exercise)

        return True

    def shutdown(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
