"""
engine.py: Core application loop and state management for AURA.
Includes automatic dual-view estimated fusion, optional calibrated ArUco
metric reconstruction, automatic ROI splitting, and one unified 3D Digital Twin.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from aura.analyzer import PostureAnalyzer
from aura.detector import PoseDetector
from aura.logger import AuraLogger
from aura.mirror_plane import MirrorPlaneEstimator  # Imported the new math estimator

# New Refactored Modules
from aura.multi_detector import AnthropometricValidator, MultiSkeletonDetector
from aura.pose_fusion import DualViewPoseFuser
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
        # A generic pinhole matrix may be useful for image/UI experiments, but
        # it must never enable metric stereo reconstruction.
        self.camera_matrix = None
        self.dist_coeffs = None
        self.apply_calibration = False
        self.camera_is_calibrated = False
        self.calibration_file_available = False
        self.calibration_rms: float | None = None
        self.calibration_view_count: int | None = None

        calib_path = "camera_calibration.npz"
        loaded_camera_matrix = None
        loaded_dist_coeffs = None
        if os.path.exists(calib_path):
            try:
                with np.load(calib_path) as data:
                    loaded_camera_matrix = np.asarray(
                        data["camera_matrix"], dtype=np.float32
                    )
                    loaded_dist_coeffs = np.asarray(
                        data["dist_coeffs"], dtype=np.float32
                    )
                    stored_width = (
                        int(data["image_width"]) if "image_width" in data else None
                    )
                    stored_height = (
                        int(data["image_height"]) if "image_height" in data else None
                    )
                    self.calibration_rms = float(data["rms"]) if "rms" in data else None
                    self.calibration_view_count = (
                        int(data["num_views"]) if "num_views" in data else None
                    )
                valid_intrinsics = (
                    loaded_camera_matrix.shape == (3, 3)
                    and loaded_dist_coeffs.size >= 4
                    and np.isfinite(loaded_camera_matrix).all()
                    and np.isfinite(loaded_dist_coeffs).all()
                )
                resolution_matches = (
                    stored_width is None
                    or stored_height is None
                    or frame_width <= 0
                    or frame_height <= 0
                    or (stored_width == frame_width and stored_height == frame_height)
                )
                self.calibration_file_available = bool(
                    valid_intrinsics and resolution_matches
                )
                if not self.calibration_file_available:
                    print(
                        "[WARNING] Calibration file is invalid or its stored resolution "
                        f"does not match the capture ({frame_width}x{frame_height}). "
                        "Estimated dual-view mode will be used."
                    )
                else:
                    print(f"\n[INFO] Calibration file found: {calib_path}")
                    if stored_width is None or stored_height is None:
                        print(
                            "[WARNING] This calibration file has no stored resolution metadata. "
                            "Confirm that it was generated at the current capture resolution."
                        )
                    if self.calibration_rms is None:
                        print(
                            "[WARNING] Calibration RMS is not stored in this legacy file; "
                            "optical calibration quality is unknown."
                        )
                    else:
                        print(
                            f"[INFO] Calibration RMS: {self.calibration_rms:.3f} px "
                            f"({self.calibration_view_count or 'unknown'} views)."
                        )
                        if self.calibration_rms > 1.5:
                            print(
                                "[WARNING] Calibration RMS is relatively high. "
                                "Consider recalibrating with more diverse checkerboard poses."
                            )
                    answer = (
                        input("Use this camera calibration for metric stereo? [y/N]: ")
                        .strip()
                        .lower()
                    )
                    if answer == "y":
                        self.camera_matrix = loaded_camera_matrix
                        self.dist_coeffs = loaded_dist_coeffs.reshape(-1, 1)
                        self.apply_calibration = True
                        self.camera_is_calibrated = True
                    else:
                        print(
                            "[INFO] Calibration not activated. Using estimated dual-view fusion."
                        )
            except (OSError, KeyError, ValueError) as exc:
                print(f"[WARNING] Cannot load camera calibration: {exc}")

        if self.camera_matrix is None:
            focal_length = float(max(frame_width, 1))
            self.camera_matrix = np.array(
                [
                    [focal_length, 0.0, frame_width / 2.0],
                    [0.0, focal_length, frame_height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            self.apply_calibration = False
            self.camera_is_calibrated = False
            print(
                "[INFO] Metric calibration inactive. Dual View Estimated Fusion "
                "will be used when the mirror split is ready.\n"
            )

        # Initialize ArUco tracker (marker size accurately set to 18cm = 0.18m)
        self.aruco_tracker = ArucoMultiTracker(marker_size_m=0.18)

        # --- COMPONENT INITIALIZATION ---
        self.single_detector = PoseDetector()
        self.posture_analyzer = PostureAnalyzer()

        self.splitter = AutoROISplitter()
        self.multi_detector: MultiSkeletonDetector | None = None

        # New Structural Subsystems
        self.reconstructor: StereoReconstructor | None = None
        if self.camera_is_calibrated:
            self.reconstructor = StereoReconstructor(self.camera_matrix)

        # Estimated fusion is the default after automatic splitting. It is
        # independent from the calibrated metric reconstruction branch.
        self.pose_fuser = DualViewPoseFuser()
        self.last_fusion_result = None  # Diagnostic snapshots; not rendered as PiP.

        # Buffer for EMA filter on render offset and floor stabilization
        self._smoothed_render_offset: np.ndarray | None = None
        self._smoothed_floor_y: float | None = None
        # Smoothing factor for vertical jitter reduction (higher = more responsive, lower = smoother)
        self._floor_ema_alpha: float = 0.35
        # Smoothing factor for horizontal jitter reduction
        self._ema_alpha: float = 0.15

        # Estimated root motion is tracked from the Front View image-space pelvis.
        # It moves only the rendered skeleton and never changes the fused pose.
        self._estimated_root_reference_2d: np.ndarray | None = None
        self._estimated_root_filtered_2d: np.ndarray | None = None
        self._estimated_root_base_translation = np.zeros(3, dtype=np.float32)
        self._estimated_root_translation = np.zeros(3, dtype=np.float32)
        self._estimated_root_scale_samples: deque[float] = deque(maxlen=15)
        self._estimated_root_scale: float | None = None
        self._estimated_root_scale_frozen = False
        self._estimated_root_axis_signs: np.ndarray | None = None
        self._estimated_root_missing_frames = 0
        self._estimated_root_needs_rebase = False
        # Floor marker ID 0 defines a stable camera-to-floor frame for metric
        # visualization. It is captured once, not re-estimated every frame.
        self._floor_frame_basis: np.ndarray | None = None
        self._floor_frame_origin: np.ndarray | None = None
        self._floor_frame_candidates: list[tuple[np.ndarray, np.ndarray]] = []

        # Retained for compatibility/diagnostic access; no longer used as an
        # uncalibrated stereo fallback.
        self.svd_estimator = MirrorPlaneEstimator(ema_alpha=self._ema_alpha)
        self.validator = AnthropometricValidator(units="meters")
        self.dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)

        # UI & Rendering
        self.renderer = AuraRenderer()
        self.viewer_3d = Aura3DViewer(width=800, height=600)
        self.sys_logger = AuraLogger()

        # Silent split detection uses two short-lived MediaPipe graphs. They are
        # released as soon as the mirror has been confirmed; the primary detector
        # is reused later as the Dual View front detector.
        mp_pose = mp.solutions.pose
        self.silent_checker_left = mp_pose.Pose(min_detection_confidence=0.5)
        self.silent_checker_right = mp_pose.Pose(min_detection_confidence=0.5)
        self._silent_checkers_closed = False
        self._shutdown = False
        self.pearson_buffer_left: list[float] = []
        self.pearson_buffer_right: list[float] = []

        # System state
        self.current_state = STATE_SINGLE
        self.recording_state = REC_IDLE
        # True only when the current frame has passed calibrated metric-stereo validation.
        # Keyboard recording commands use this flag to prevent estimated/metric mixing.
        self._metric_stereo_validated_current_frame = False
        self.dtw_last_result = None
        self.frame_count = 0
        self._window_placed = False

    def _ground_landmarks(self, landmarks_3d: np.ndarray | None) -> np.ndarray | None:
        """
        Dynamically grounds raw MediaPipe 3D world landmarks onto the floor grid (Y=0) EVERY FRAME.
        MediaPipe world_landmarks are inherently hip-centered. By calculating the feet centroid
        dynamically and applying it instantly (without EMA lag), the feet stay perfectly glued
        to the floor, absorbing the squat naturally without vertical floating or lateral skating.
        """
        if landmarks_3d is None or len(landmarks_3d) != 33:
            return landmarks_3d

        grounded = landmarks_3d.copy()
        feet_indices = [27, 28, 29, 30, 31, 32]  # Ankles, heels, toes

        # Use np.max for Y to guarantee the absolute lowest physical point touches the floor
        target_y = float(np.max(grounded[feet_indices, 1]))
        target_x = float((grounded[27, 0] + grounded[28, 0]) / 2.0)
        target_z = float((grounded[27, 2] + grounded[28, 2]) / 2.0)

        # Instant subtraction: forces the feet to be locked at X=0, Z=0 and lowest point at Y=0
        grounded[:, 0] -= target_x
        grounded[:, 1] -= target_y
        grounded[:, 2] -= target_z

        return grounded

    def _prepare_estimated_render_landmarks(
        self,
        landmarks_3d: np.ndarray | None,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Rigid Floor Baseline Anchor.

        Locks the vertical floor level (Y) to the initial standing baseline
        captured at startup. This prevents the feet from sinking or floating
        when the user drops into a deep squat.
        """
        if landmarks_3d is None or np.asarray(landmarks_3d).shape != (33, 3):
            return landmarks_3d

        # Create a copy to avoid polluting the original biometric data
        points = np.asarray(landmarks_3d, dtype=np.float32).copy()
        if not np.isfinite(points).all():
            return points

        # Use the ankles (27, 28) as the central horizontal pivot.
        left_ankle = points[27]
        right_ankle = points[28]

        center_x = float((left_ankle[0] + right_ankle[0]) / 2.0)
        center_z = float((left_ankle[2] + right_ankle[2]) / 2.0)

        # Calculate instantaneous lowest point of the feet
        feet_indices = [27, 28, 29, 30, 31, 32]
        instant_target_y = float(np.max(points[feet_indices, 1]))

        # Lock the floor baseline permanently using the first valid standing pose.
        # This stops the floor from dropping down with you during a squat.
        if self._smoothed_floor_y is None:
            self._smoothed_floor_y = instant_target_y

        # Shift entire skeleton using the frozen standing floor level
        points[:, 0] -= center_x
        points[:, 1] -= self._smoothed_floor_y
        points[:, 2] -= center_z

        return points

    def _reset_estimated_root_motion(self) -> None:
        """Reset the image-space root-motion estimator for a new tracking session."""
        self._estimated_root_reference_2d = None
        self._estimated_root_filtered_2d = None
        self._estimated_root_base_translation = np.zeros(3, dtype=np.float32)
        self._estimated_root_translation = np.zeros(3, dtype=np.float32)
        self._estimated_root_scale_samples.clear()
        self._estimated_root_scale = None
        self._estimated_root_scale_frozen = False
        self._estimated_root_axis_signs = None
        self._estimated_root_missing_frames = 0
        self._estimated_root_needs_rebase = False
        self._smoothed_floor_y = None  # Reset floor stabilization buffer

    @staticmethod
    def _root_motion_measurement(observation):
        """Return a reliable pelvis center and relative pixel-to-world scale estimate."""
        if observation is None or not getattr(observation, "detection_valid", False):
            return None

        points_2d = getattr(observation, "landmarks_2d", None)
        points_3d = getattr(observation, "landmarks_3d", None)
        confidence = getattr(observation, "joint_confidence", None)
        if points_2d is None or points_3d is None or confidence is None:
            return None

        points_2d = np.asarray(points_2d, dtype=np.float32)
        points_3d = np.asarray(points_3d, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if (
            points_2d.shape != (33, 2)
            or points_3d.shape != (33, 3)
            or confidence.shape != (33,)
        ):
            return None

        anchor_indices = np.asarray([11, 12, 23, 24], dtype=int)
        if not (
            np.isfinite(points_2d[anchor_indices]).all()
            and np.isfinite(points_3d[anchor_indices]).all()
            and np.isfinite(confidence[anchor_indices]).all()
        ):
            return None
        if np.min(confidence[anchor_indices]) < 0.25:
            return None

        hip_weights = np.maximum(confidence[[23, 24]], 0.05)
        pelvis_center = np.average(points_2d[[23, 24]], axis=0, weights=hip_weights)

        shoulder_width_pixels = float(np.linalg.norm(points_2d[11] - points_2d[12]))
        shoulder_width_world = float(np.linalg.norm(points_3d[11] - points_3d[12]))
        if shoulder_width_pixels < 15.0 or not 0.15 <= shoulder_width_world <= 0.80:
            return None

        # MediaPipe world coordinates are relative to the hips. The ratio below
        # provides an approximate, session-relative conversion for image motion.
        scale = shoulder_width_world / shoulder_width_pixels
        if not np.isfinite(scale) or not 0.0002 <= scale <= 0.03:
            return None

        # Infer axis signs from the observed shoulder and torso directions instead
        # of assuming that image and MediaPipe coordinate conventions match.
        image_shoulder_dx = float(points_2d[12, 0] - points_2d[11, 0])
        world_shoulder_dx = float(points_3d[12, 0] - points_3d[11, 0])
        image_torso_dy = float(
            np.mean(points_2d[[11, 12], 1]) - np.mean(points_2d[[23, 24], 1])
        )
        world_torso_dy = float(
            np.mean(points_3d[[11, 12], 1]) - np.mean(points_3d[[23, 24], 1])
        )
        if abs(image_shoulder_dx) < 5.0 or abs(world_shoulder_dx) < 0.03:
            return None
        if abs(image_torso_dy) < 10.0 or abs(world_torso_dy) < 0.08:
            return None

        axis_signs = np.asarray(
            [
                1.0 if image_shoulder_dx * world_shoulder_dx > 0.0 else -1.0,
                1.0 if image_torso_dy * world_torso_dy > 0.0 else -1.0,
            ],
            dtype=np.float32,
        )
        return pelvis_center.astype(np.float32), float(scale), axis_signs

    def _update_estimated_root_motion(self, front_observation) -> np.ndarray:
        """Estimate global X/Y movement from the Front View for rendering only."""
        measurement = self._root_motion_measurement(front_observation)
        if measurement is None:
            self._estimated_root_missing_frames += 1
            if self._estimated_root_missing_frames >= 12:
                # Rebase after a prolonged loss to avoid a large reacquisition jump.
                self._estimated_root_needs_rebase = True
            return self._estimated_root_translation.copy()

        pelvis_center, scale_sample, axis_signs = measurement
        self._estimated_root_missing_frames = 0

        if self._estimated_root_axis_signs is None:
            self._estimated_root_axis_signs = axis_signs

        if not self._estimated_root_scale_frozen:
            self._estimated_root_scale_samples.append(scale_sample)
            # Use the first valid scale immediately, then make it robust with a
            # growing median window while the session is still initializing.
            scale_values = np.asarray(
                self._estimated_root_scale_samples, dtype=np.float32
            )
            self._estimated_root_scale = float(np.median(scale_values))
            if len(self._estimated_root_scale_samples) >= 15:
                self._estimated_root_scale_frozen = True

        if (
            self._estimated_root_reference_2d is None
            or self._estimated_root_needs_rebase
        ):
            self._estimated_root_reference_2d = pelvis_center.copy()
            self._estimated_root_filtered_2d = pelvis_center.copy()
            if self._estimated_root_needs_rebase:
                self._estimated_root_base_translation = (
                    self._estimated_root_translation.copy()
                )
            else:
                self._estimated_root_base_translation = np.zeros(3, dtype=np.float32)
            self._estimated_root_needs_rebase = False
            return self._estimated_root_translation.copy()

        if self._estimated_root_filtered_2d is None:
            self._estimated_root_filtered_2d = pelvis_center.copy()

        # A light EMA suppresses landmark jitter without adding a heavy temporal model.
        delta_pixels = pelvis_center - self._estimated_root_filtered_2d
        max_step = 0.06 * max(
            float(getattr(front_observation, "image_size", (640, 480))[0]), 1.0
        )
        delta_length = float(np.linalg.norm(delta_pixels))
        if delta_length > max_step > 0.0:
            delta_pixels *= max_step / delta_length
        self._estimated_root_filtered_2d += 0.45 * delta_pixels

        if (
            self._estimated_root_scale is None
            or self._estimated_root_axis_signs is None
        ):
            return self._estimated_root_translation.copy()

        displacement_pixels = (
            self._estimated_root_filtered_2d - self._estimated_root_reference_2d
        )
        translation = np.zeros(3, dtype=np.float32)
        translation[0] = (
            self._estimated_root_axis_signs[0]
            * displacement_pixels[0]
            * self._estimated_root_scale
        )
        translation[1] = (
            self._estimated_root_axis_signs[1]
            * displacement_pixels[1]
            * self._estimated_root_scale
        )
        translation[2] = 0.0  # Depth is intentionally not inferred in estimated mode.

        self._estimated_root_translation = (
            self._estimated_root_base_translation + translation
        ).astype(np.float32)
        return self._estimated_root_translation.copy()

    @staticmethod
    def _observation_quality(observation) -> float:
        if observation is None or not observation.detection_valid:
            return 0.0
        confidence = observation.joint_confidence
        valid = observation.valid_joint_mask
        return float(np.mean(confidence[valid])) if valid.any() else 0.0

    def _floor_coordinates(self, points_camera: np.ndarray) -> np.ndarray:
        """Convert metric camera-frame points into the frozen floor-marker frame."""
        if self._floor_frame_basis is None or self._floor_frame_origin is None:
            return np.asarray(points_camera, dtype=np.float32).copy()
        points = np.asarray(points_camera, dtype=np.float32)
        return ((points - self._floor_frame_origin) @ self._floor_frame_basis).astype(
            np.float32
        )

    def _evaluate_joint_colors(
        self, active_results, is_fused_valid: bool = True
    ) -> dict[int, str]:
        """
        Evaluates 3D joint posture against biomechanical golden standards and returns a
        color mapping for rendering joints in the 3D viewport.
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
            # Handle both Tuple and Dict returns gracefully for temporal.py compatibility
            distance = 0.0
            if isinstance(self.dtw_last_result, dict):
                distance = self.dtw_last_result.get("deviation_per_frame", 0)
            elif (
                isinstance(self.dtw_last_result, tuple)
                and len(self.dtw_last_result) > 0
            ):
                distance = self.dtw_last_result[0]

            if distance > 1.5:  # Adjusted for multidimensional feature vectors
                joint_colors[27] = "warning"
                joint_colors[28] = "warning"

        return joint_colors

    def run(self) -> None:
        """Starts the main video loop."""

        init_log = "AURA System Initialized. Looking for user..."
        if self.camera_is_calibrated:
            init_log = "AURA Initialized. Metric camera calibration active. Looking for user..."
        else:
            init_log = "AURA Initialized. Dual View Estimated Fusion is the default mode. Looking for user..."

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

                if not isinstance(self.source, int):
                    self._reset_estimated_root_motion()
                    self._smoothed_render_offset = None
                    self._smoothed_floor_y = None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
                if not success:
                    break

            self.frame_count += 1

            # --- PHASE 1: OPTICAL CORRECTION AND ARUCO TRACKING ---
            aruco_poses = {}
            if (
                self.camera_is_calibrated
                and self.camera_matrix is not None
                and self.dist_coeffs is not None
            ):
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                # solvePnP must not apply distortion a second time to an already
                # undistorted frame.
                active_dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                frame, aruco_poses = self.aruco_tracker.process_frame(
                    frame, self.camera_matrix, active_dist_coeffs
                )

            # Extract ArUco tracking data for Mirror Plane
            aruco_rvec = None
            aruco_tvec = None

            if 0 in aruco_poses:
                rvec_floor, tvec_floor = aruco_poses[0]
                floor_dist = np.linalg.norm(tvec_floor)
                self.sys_logger.log_metric("Floor Dist", f"{floor_dist:.2f} m")
                floor_error = self.aruco_tracker.last_pose_errors.get(0)
                if floor_error is not None:
                    self.sys_logger.log_metric(
                        "Floor ArUco Reproj", f"{floor_error:.2f} px"
                    )
                # Capture the floor reference once, from a plausible marker pose.
                # This aligns the metric Digital Twin with gravity/floor instead
                # of assuming that the webcam optical axes are already level.
                if (
                    self._floor_frame_basis is None
                    and floor_error is not None
                    and floor_error <= 3.0
                ):
                    try:
                        floor_frame = self.aruco_tracker.estimate_floor_frame(
                            rvec_floor, tvec_floor
                        )
                        self._floor_frame_candidates.append(
                            (
                                floor_frame["basis_camera"].astype(np.float64),
                                floor_frame["origin_camera"].astype(np.float64),
                            )
                        )
                        if len(self._floor_frame_candidates) >= 8:
                            bases = [
                                item[0] for item in self._floor_frame_candidates[-8:]
                            ]
                            origins = np.asarray(
                                [item[1] for item in self._floor_frame_candidates[-8:]],
                                dtype=np.float64,
                            )
                            mean_basis = np.mean(np.stack(bases), axis=0)
                            u, _singular, vt = np.linalg.svd(mean_basis)
                            correction = np.eye(3)
                            if np.linalg.det(u @ vt) < 0.0:
                                correction[-1, -1] = -1.0
                            self._floor_frame_basis = (u @ correction @ vt).astype(
                                np.float32
                            )
                            self._floor_frame_origin = np.median(
                                origins, axis=0
                            ).astype(np.float32)
                            self.sys_logger.log(
                                "Floor reference initialized.",
                                "Camera-to-floor frame robustly averaged over 8 ArUco observations (ID 0).",
                            )
                    except (ValueError, np.linalg.LinAlgError):
                        self._floor_frame_candidates.clear()
                        self.sys_logger.log(
                            "Floor reference unavailable.",
                            "Could not derive a stable camera-to-floor coordinate frame from marker ID 0.",
                        )

            if 1 in aruco_poses:
                aruco_rvec, aruco_tvec = aruco_poses[1]
                mirror_dist = np.linalg.norm(aruco_tvec)
                self.sys_logger.log_metric("Mirror Dist", f"{mirror_dist:.2f} m")
                mirror_error = self.aruco_tracker.last_pose_errors.get(1)
                if mirror_error is not None:
                    self.sys_logger.log_metric(
                        "Mirror ArUco Reproj", f"{mirror_error:.2f} px"
                    )

            if self.camera_is_calibrated and self.calibration_rms is not None:
                self.sys_logger.log_metric(
                    "Calibration RMS", f"{self.calibration_rms:.2f} px"
                )

            _, w = frame.shape[:2]
            active_frame = None
            active_results = None
            front_roi = None
            mirror_roi = None
            fused_3d = None
            plane_normal = None
            fused_pose_result = None

            # State flags for the current frame
            is_stereo_validated = False
            is_estimated_fusion = False
            hud_status = "3D FUSION: SINGLE"

            # ==========================================
            # PHASE 2: SINGLE MODE
            # ==========================================
            if self.current_state == STATE_SINGLE:
                display_frame = frame.copy()
                active_frame = display_frame

                self.single_detector.detect(display_frame, draw=True)
                active_results = getattr(self.single_detector, "results", None)

                # Ground raw single-view 3D landmarks onto floor Y=0 dynamically
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
                        # Stop using the temporary split-checker graphs before
                        # allocating the mirror graph. Reuse the primary detector
                        # for the front ROI instead of creating a duplicate graph.
                        self._release_silent_checkers()
                        self.single_detector.reset()
                        self.multi_detector = MultiSkeletonDetector(
                            split_ratio=split_ratio,
                            front_detector=self.single_detector,
                        )
                        self.pose_fuser.reset()
                        self.last_fusion_result = None
                        self._smoothed_render_offset = None  # Reset EMA buffer when initializing MultiSkeletonDetector
                        self._smoothed_floor_y = None
                        self._reset_estimated_root_motion()
                        cv2.destroyWindow(window_title)
                    else:
                        cv2.imshow(window_title, frame)
                        self.single_detector.detect(frame, draw=False)
                        raw_landmarks = self.single_detector.get_world_landmark_array()
                        fused_3d = self._ground_landmarks(raw_landmarks)
                        active_frame = frame
                else:
                    # 1. Get 2D Pixel Landmarks
                    pipeline_data = self.multi_detector.process_frame(
                        frame, frame_id=self.frame_count, timestamp=time.perf_counter()
                    )

                    for user_msg, expert_msg in pipeline_data.get("logs", []):
                        self.sys_logger.log(user_msg, expert_msg)

                    front_roi = pipeline_data["front_roi"]
                    mirror_roi = pipeline_data["mirror_roi"]
                    active_results = pipeline_data["front_results"]
                    mirror_results = pipeline_data["mirror_results"]

                    direct_pose_2d = pipeline_data.get("direct_pose_2d")
                    reflected_pose_2d = pipeline_data.get("reflected_pose_2d")

                    active_frame = front_roi
                    # 2. Metric reconstruction is enabled only after the user
                    # explicitly opted into a valid camera calibration and the
                    # mirror ArUco marker is currently detected.
                    metric_inputs_ready = (
                        self.camera_is_calibrated
                        and self.reconstructor is not None
                        and aruco_rvec is not None
                        and aruco_tvec is not None
                        and self._floor_frame_basis is not None
                        and self.aruco_tracker.last_pose_errors.get(1, float("inf"))
                        <= 5.0
                        and direct_pose_2d is not None
                        and reflected_pose_2d is not None
                    )

                    if metric_inputs_ready:
                        try:
                            mirror_data = self.aruco_tracker.estimate_mirror_plane(
                                aruco_rvec, aruco_tvec
                            )
                            reflection_matrix = mirror_data["reflection_matrix"]
                            metric_points = self.reconstructor.triangulate_landmarks(
                                direct_pose_2d, reflected_pose_2d, reflection_matrix
                            )
                            if self.validator.validate_skeleton_3d(metric_points):
                                fused_3d = self._floor_coordinates(metric_points)
                                if self._floor_frame_basis is not None:
                                    plane_normal = (
                                        np.asarray(mirror_data["normal"])
                                        @ self._floor_frame_basis
                                    )
                                    hud_status = "METRIC STEREO | FLOOR ALIGNED"
                                else:
                                    plane_normal = mirror_data["normal"]
                                    hud_status = "METRIC STEREO | CAMERA FRAME"
                                is_stereo_validated = True
                                self.last_fusion_result = None
                            else:
                                self.sys_logger.log(
                                    "Metric reconstruction rejected.",
                                    "Anthropometric validation failed; switching to Dual View Estimated Fusion for this frame.",
                                )
                        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                            self.sys_logger.log(
                                "Metric reconstruction unavailable.",
                                f"ArUco stereo reconstruction failed: {exc}. Using estimated fusion.",
                            )

                    # Default route: always retain a useful Front-referenced
                    # skeleton and integrate Mirror information as soon as the
                    # empirical alignment has accumulated enough observations.
                    if not is_stereo_validated:
                        front_observation = pipeline_data.get("front_observation")
                        mirror_observation = pipeline_data.get("mirror_observation")
                        fused_pose_result = self.pose_fuser.update(
                            front_observation, mirror_observation
                        )
                        self.last_fusion_result = fused_pose_result
                        if fused_pose_result is not None:
                            fused_3d = self._prepare_estimated_render_landmarks(
                                fused_pose_result.landmarks_3d,
                                fused_pose_result.joint_valid,
                            )
                            # Add Front-derived global movement to the render copy only.
                            # The anatomical fused pose remains pelvis-relative and untouched.
                            root_translation = self._update_estimated_root_motion(
                                front_observation
                            )
                            if fused_3d is not None:
                                fused_3d = np.asarray(
                                    fused_3d, dtype=np.float32
                                ) + root_translation.reshape(1, 3)
                            is_estimated_fusion = True
                            front_quality = self._observation_quality(front_observation)
                            mirror_quality = self._observation_quality(
                                mirror_observation
                            )
                            source_counts = {
                                source: fused_pose_result.joint_sources.count(source)
                                for source in set(fused_pose_result.joint_sources)
                            }
                            mirror_dominant_sources = {
                                "MIRROR_DOMINANT",
                                "MIRROR_ONLY",
                                "MIRROR_CONFLICT_DOMINANT",
                            }
                            mirror_dominant = sum(
                                count
                                for source, count in source_counts.items()
                                if source in mirror_dominant_sources
                            )
                            valid_count = int(
                                np.count_nonzero(fused_pose_result.joint_valid)
                            )
                            self.sys_logger.log_metric(
                                "Front Detect Q", f"{front_quality:.2f}"
                            )
                            self.sys_logger.log_metric(
                                "Mirror Detect Q", f"{mirror_quality:.2f}"
                            )
                            self.sys_logger.log_metric(
                                "Alignment Q",
                                f"{fused_pose_result.alignment_quality:.2f}",
                            )
                            self.sys_logger.log_metric(
                                "Fusion Q", f"{fused_pose_result.fusion_quality:.2f}"
                            )
                            self.sys_logger.log_metric(
                                "Mirror Dominant", f"{mirror_dominant}/33"
                            )
                            self.sys_logger.log_metric(
                                "Valid Joints", f"{valid_count}/33"
                            )
                            if fused_pose_result.alignment_ready:
                                hud_status = (
                                    "DUAL VIEW ESTIMATED | "
                                    f"FQ:{fused_pose_result.fusion_quality:.2f} "
                                    f"AQ:{fused_pose_result.alignment_quality:.2f}"
                                )
                            else:
                                hud_status = (
                                    "DUAL VIEW ESTIMATED | ALIGNING "
                                    f"({self.pose_fuser.aligner.result.sample_count}/"
                                    f"{self.pose_fuser.aligner.min_samples})"
                                )
                        else:
                            hud_status = "DUAL VIEW ESTIMATED | FRONT LOST"

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
                            0.55,
                            (0, 255, 0)
                            if is_stereo_validated
                            else (
                                (0, 190, 255) if is_estimated_fusion else (0, 0, 255)
                            ),
                            2,
                            cv2.LINE_AA,
                        )
                        if is_estimated_fusion and fused_pose_result is not None:
                            detail = (
                                f"DET F:{front_quality:.2f} M:{mirror_quality:.2f}  "
                                f"MIRROR DOM:{mirror_dominant}/33  "
                                f"VALID:{valid_count}/33"
                            )
                            cv2.putText(
                                front_roi,
                                detail,
                                (15, 199),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.43,
                                (180, 220, 255),
                                1,
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
            # Expose the effective mode of this exact frame to keyboard handlers.
            self._metric_stereo_validated_current_frame = bool(is_stereo_validated)

            # A metric recording must remain metric for its entire trajectory.
            # Abort rather than silently skipping estimated frames and later mixing modes.
            if (
                self.recording_state in (REC_GOLDEN, REC_USER)
                and not self._metric_stereo_validated_current_frame
            ):
                interrupted_state = self.recording_state
                if interrupted_state == REC_GOLDEN:
                    self.dtw_analyzer.golden_reference.clear()
                else:
                    self.dtw_analyzer.reset_user_buffer()
                self.recording_state = REC_IDLE
                self.dtw_last_result = None
                self.sys_logger.log(
                    "Metric recording aborted: calibrated stereo is no longer valid.",
                    (
                        f"{interrupted_state} aborted. The current frame is not a validated "
                        "Metric Stereo reconstruction; incomplete data was discarded to "
                        "prevent mixing metric and estimated poses."
                    ),
                )

            if fused_3d is not None and is_stereo_validated:
                kinematic_features = self.dtw_analyzer.extract_kinematic_features(
                    fused_3d
                )

                if self.recording_state == REC_GOLDEN:
                    self.dtw_analyzer.golden_reference.append(
                        kinematic_features.tolist()
                    )
                elif self.recording_state == REC_USER:
                    self.dtw_analyzer.add_user_frame(kinematic_features)

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

            # Evaluate biomechanical joint colors against golden standard
            joint_colors = self._evaluate_joint_colors(
                active_results, (is_stereo_validated or is_estimated_fusion)
            )

            # Render exactly one skeleton. Invalid estimated joints are masked
            # instead of drawing zero-filled placeholders as real anatomy.
            render_joint_valid = (
                fused_pose_result.joint_valid
                if fused_pose_result is not None and is_estimated_fusion
                else None
            )
            cv2.imshow(
                "AURA - 3D Digital Twin",
                self.viewer_3d.render_scene(
                    fused_3d,
                    plane_normal,
                    joint_colors,
                    joint_valid=render_joint_valid,
                ),
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

            # Pass key to 3D Viewer orbit controls (handles Arrow keys & Numpad keys)
            if raw_key != -1:
                self.viewer_3d.handle_keyboard(raw_key)

            if cmd != "NONE" and not self._handle_commands(cmd):
                break

        self.shutdown()

    def _is_valid_roi(self, roi: np.ndarray | None) -> bool:
        return (
            roi is not None and roi.ndim == 3 and roi.shape[0] > 0 and roi.shape[1] > 0
        )

    def _release_silent_checkers(self) -> None:
        """Close temporary split-detection graphs once automatic splitting is confirmed."""
        if getattr(self, "_silent_checkers_closed", False):
            return
        for attribute_name in ("silent_checker_left", "silent_checker_right"):
            checker = getattr(self, attribute_name, None)
            if checker is not None:
                close_method = getattr(checker, "close", None)
                if callable(close_method):
                    try:
                        close_method()
                    except Exception as exc:  # noqa: BLE001 - Continue releasing the remaining MediaPipe graphs.
                        print(f"[WARNING] Could not close {attribute_name}: {exc}")
                setattr(self, attribute_name, None)
        self._silent_checkers_closed = True

    def _check_silent_split(self, frame: np.ndarray, w: int) -> None:
        if self.frame_count % 10 != 0:
            return
        if self.silent_checker_left is None or self.silent_checker_right is None:
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
                # These graphs are no longer needed once the mirror is detected.
                self._release_silent_checkers()
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
            if not getattr(self, "_metric_stereo_validated_current_frame", False):
                self.sys_logger.log(
                    "Golden Standard unavailable in Estimated mode.",
                    (
                        "G command rejected: Dual View Estimated Fusion provides a relative "
                        "pose estimate, not a validated metric reconstruction. Record a "
                        "trainer Golden Standard only while calibrated Metric Stereo is active."
                    ),
                )
            else:
                self.sys_logger.log(
                    "Starting Metric Golden Standard recording.",
                    "Initiating REC_GOLDEN state with validated calibrated stereo. Buffer cleared.",
                )
                self.recording_state = REC_GOLDEN
                self.dtw_analyzer.golden_reference.clear()
                self.dtw_last_result = None
        elif cmd == "REC_USER":
            if not getattr(self, "_metric_stereo_validated_current_frame", False):
                self.sys_logger.log(
                    "Metric comparison unavailable in Estimated mode.",
                    (
                        "U command rejected: the active pose is estimated and cannot be "
                        "compared with a metric Golden Standard. Switch to validated "
                        "Calibrated Metric Stereo to record a comparable execution."
                    ),
                )
            elif not self.dtw_analyzer.golden_reference:
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

                    # Handle Tuple or Dict return safely
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
        """Release capture, MediaPipe graphs and UI resources exactly once."""
        if getattr(self, "_shutdown", False):
            return
        self._shutdown = True

        try:
            if getattr(self, "cap", None) is not None:
                self.cap.release()
        finally:
            # MultiSkeletonDetector closes only the mirror graph it owns.
            # Its front graph is injected and remains owned by AuraEngine.
            multi_detector = getattr(self, "multi_detector", None)
            if multi_detector is not None:
                try:
                    multi_detector.close()
                except Exception as exc:  # noqa: BLE001 - Shutdown must continue after cleanup failures.
                    print(f"[WARNING] Could not close dual-view detectors: {exc}")

            single_detector = getattr(self, "single_detector", None)
            if single_detector is not None:
                try:
                    single_detector.close()
                except Exception as exc:  # noqa: BLE001 - Shutdown must continue after cleanup failures.
                    print(f"[WARNING] Could not close primary detector: {exc}")

            self._release_silent_checkers()
            cv2.destroyAllWindows()
