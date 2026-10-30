"""Main AURA application loop and finite-state recording controller.

Metric mirror reconstruction is enabled only when camera intrinsics have been
validated for the current frame size and a calibrated ArUco marker is detected
coplanar with the mirror plane. Markerless MediaPipe world
landmarks are never silently promoted to camera-space metric coordinates.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from aura.analyzer import PostureAnalyzer
from aura.anthropometry import AnthropometricValidator
from aura.detector import PoseDetector
from aura.logger import AuraLogger
from aura.mirror_plane import MirrorPlaneEstimator
from aura.multi_detector import MultiSkeletonDetector
from aura.reconstruction import StereoReconstructor
from aura.renderer import AuraRenderer
from aura.renderer_3d import Aura3DViewer
from aura.roi_splitter import AutoROISplitter
from aura.temporal import DTWTemporalAnalyzer
from aura.utils.aruco_tracker import ArucoMultiTracker
from aura.utils.input_handler import KeyboardController

logger = logging.getLogger(__name__)

STATE_SINGLE = "SINGLE_MODE"
STATE_MIRROR = "MIRROR_MODE"
REC_IDLE = "IDLE"
REC_GOLDEN = "RECORD_GOLDEN"
REC_USER = "RECORD_USER"


class AuraEngine:
    """Orchestrate video, pose estimation, optional mirror reconstruction, and UI.

    The public constructor and ``run``/``shutdown`` methods retain the current
    AURA API. The ``x`` key explicitly enters/leaves mirror mode; automatic
    reflection recognition is intentionally not used as a calibration signal.
    """

    def __init__(self, source: str | int = "0", exercise: str = "squats") -> None:
        self.source = int(source) if str(source).isdigit() else source
        self.exercise = exercise
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source '{self.source}'")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = max(1, int(1000 / fps)) if fps and fps > 0 else 33
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.apply_calibration = False
        self.camera_is_calibrated = False
        self.calibration_rms = float("inf")
        self.calibration_path = os.environ.get(
            "AURA_CALIBRATION_FILE", "camera_calibration.npz"
        )
        self._load_camera_calibration(self.calibration_path)

        self.reconstructor: StereoReconstructor | None = None
        if self.camera_is_calibrated:
            # Frames are undistorted before inference, therefore the reconstructor
            # receives pixel coordinates in the calibrated undistorted image.
            self.reconstructor = StereoReconstructor(
                self.camera_matrix, dist_coeffs=None
            )

        self.aruco_tracker: ArucoMultiTracker | None = None
        try:
            marker_size = float(os.environ.get("AURA_MARKER_SIZE_M", "0.18"))
            self.aruco_tracker = ArucoMultiTracker(marker_size_m=marker_size)
        except (RuntimeError, ValueError) as exc:
            logger.warning("ArUco metric mode unavailable: %s", exc)

        self.single_detector = PoseDetector()
        self.posture_analyzer = PostureAnalyzer()
        self.validator = AnthropometricValidator(
            units="meters"
        )  # Retained legacy engine attribute.
        self.svd_estimator = MirrorPlaneEstimator(
            ema_alpha=0.15
        )  # Diagnostic fallback only; not metric calibration.
        self.splitter = AutoROISplitter()
        self.multi_detector: MultiSkeletonDetector | None = None

        self.dtw_analyzer = DTWTemporalAnalyzer(sensitivity=5.0)
        self.renderer = AuraRenderer()
        self.viewer_3d = Aura3DViewer(width=800, height=600)
        self.sys_logger = AuraLogger()

        self.current_state = STATE_SINGLE
        self.recording_state = REC_IDLE
        self.dtw_last_result: dict | None = None
        self.frame_count = 0
        self._record_buffer: list[np.ndarray] = []
        self._recording_geometry_mode: str | None = None
        self._golden_geometry_mode: str | None = None
        self._last_render_landmarks: np.ndarray | None = None
        self._last_grounding_offset = np.zeros(3, dtype=np.float64)
        self._last_plane_normal: np.ndarray | None = None
        self._last_plane_d: float | None = None
        self._last_reconstruction_mode = "MONOCULAR_RELATIVE"
        self._window_placed = False

        if self.camera_is_calibrated:
            self.sys_logger.log_static(
                "Calibration",
                f"Valid K loaded ({self.frame_width}x{self.frame_height}, RMS {self.calibration_rms:.3f}px)",
            )
        else:
            self.sys_logger.log_static(
                "Calibration", "No valid camera calibration; metric stereo disabled"
            )

    def _load_camera_calibration(self, path: str) -> None:
        """Load only calibrations whose stored image size matches this camera."""
        file_path = Path(path)
        if not file_path.is_file():
            return
        try:
            with np.load(file_path, allow_pickle=False) as data:
                K = np.asarray(data["camera_matrix"], dtype=np.float64)
                D = np.asarray(data["dist_coeffs"], dtype=np.float64)
                stored_size = tuple(
                    int(v) for v in np.asarray(data["image_size"]).reshape(2)
                )
                rms = (
                    float(np.asarray(data["rms"]).reshape(()))
                    if "rms" in data
                    else float("inf")
                )
            if (
                K.shape != (3, 3)
                or not np.isfinite(K).all()
                or not np.isfinite(D).all()
            ):
                raise ValueError("Invalid calibration arrays")
            if stored_size != (self.frame_width, self.frame_height):
                raise ValueError(
                    f"Calibration size {stored_size} != camera size {(self.frame_width, self.frame_height)}"
                )
            # A legacy file without image_size is deliberately not accepted as metric.
            if not np.isfinite(rms) or rms > 2.0:
                raise ValueError(f"Calibration RMS is not acceptable: {rms}")
            self.camera_matrix, self.dist_coeffs = K, D
            self.camera_is_calibrated = True
            self.calibration_rms = rms
            self.apply_calibration = True
        except (OSError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Camera calibration ignored: %s", exc)

    def _ground_landmarks(self, landmarks_3d: np.ndarray | None) -> np.ndarray | None:
        """Return a display-only grounded copy; never mutate metric coordinates."""
        if landmarks_3d is None:
            return None
        points = np.asarray(landmarks_3d, dtype=np.float64)
        if points.shape != (33, 3):
            return None
        grounded = points.copy()
        feet = [27, 28, 29, 30, 31, 32]
        valid_rows = np.isfinite(grounded[feet]).all(axis=1)
        if not valid_rows.any():
            self._last_grounding_offset = np.zeros(3, dtype=np.float64)
            return None
        visible_feet = grounded[np.asarray(feet)[valid_rows]]
        # MediaPipe's model-space Y increases downward; this is only a display convention.
        target_y = float(np.max(visible_feet[:, 1]))
        target_x = float(np.median(visible_feet[:, 0]))
        target_z = float(np.median(visible_feet[:, 2]))
        self._last_grounding_offset = np.array(
            [target_x, target_y, target_z], dtype=np.float64
        )
        grounded[:, 0] -= target_x
        grounded[:, 1] -= target_y
        grounded[:, 2] -= target_z
        grounded[~np.isfinite(grounded)] = np.nan
        return grounded

    def _is_valid_roi(self, roi: np.ndarray | None) -> bool:
        """Compatibility helper: report whether an ROI is a usable BGR image."""
        return bool(
            roi is not None
            and isinstance(roi, np.ndarray)
            and roi.ndim == 3
            and roi.size > 0
            and roi.shape[0] > 1
            and roi.shape[1] > 1
        )

    def _check_silent_split(self, frame: np.ndarray, w: int) -> None:
        """Legacy hook retained for callers; automatic split guesses do not enable stereo mode.

        A reflection is a geometric hypothesis, not something that can be
        confirmed by simple nose-motion correlation. Press X to enter the
        mirror workflow or configure AURA_MIRROR_SPLIT_RATIO explicitly.
        """
        return

    def _evaluate_joint_colors(
        self, active_results, is_fused_valid: bool = True
    ) -> dict[int, str]:
        """Return simple display colors from image-space knee angles."""
        colors: dict[int, str] = {}
        if not active_results or not getattr(active_results, "pose_landmarks", None):
            return colors
        lm = active_results.pose_landmarks.landmark
        for knee_idx, hip_idx, ankle_idx in ((25, 23, 27), (26, 24, 28)):
            indices = (hip_idx, knee_idx, ankle_idx)
            if any(lm[i].visibility < 0.35 for i in indices):
                colors[knee_idx] = "default"
                continue
            a = np.array([lm[hip_idx].x, lm[hip_idx].y])
            b = np.array([lm[knee_idx].x, lm[knee_idx].y])
            c = np.array([lm[ankle_idx].x, lm[ankle_idx].y])
            from aura.geometry import calculate_angle

            angle = calculate_angle(a, b, c)
            colors[knee_idx] = (
                "warning" if np.isfinite(angle) and angle < 70 else "good"
            )
        if not is_fused_valid:
            colors[23] = colors[24] = colors[11] = colors[12] = "warning"
        return colors

    def _try_stereo_reconstruction(self, pipeline_data: dict, marker_poses: dict):
        """Return (points, normal, d, mode, valid) for the current mirror frame."""
        if (
            not self.camera_is_calibrated
            or self.reconstructor is None
            or self.aruco_tracker is None
        ):
            return None, None, None, "MONOCULAR_RELATIVE", False
        direct = pipeline_data.get("direct_pose_2d")
        reflected = pipeline_data.get("reflected_pose_2d")
        if (
            direct is None
            or reflected is None
            or not pipeline_data.get("is_valid", False)
        ):
            return None, None, None, "MONOCULAR_RELATIVE", False
        # Marker ID 1 is reserved for the marker attached to the mirror plane.
        if 1 not in marker_poses:
            return None, None, None, "MONOCULAR_RELATIVE", False
        rvec, tvec = marker_poses[1]
        diagnostics = self.aruco_tracker.last_reprojection_errors.get(1, float("inf"))
        if diagnostics > 2.0:
            return None, None, None, "INVALID", False
        plane = self.aruco_tracker.estimate_mirror_plane(rvec, tvec)
        points = self.reconstructor.triangulate_landmarks(
            direct,
            reflected,
            plane["reflection_matrix"],
            max_error_px=8.0,
            direct_confidence=pipeline_data.get("direct_visibility"),
            reflected_confidence=pipeline_data.get("reflected_visibility"),
            min_confidence=0.35,
        )
        mask = self.reconstructor.last_valid_mask
        required = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        enough = int(mask.sum()) >= 20 and all(mask[index] for index in required)
        if not enough:
            return None, plane["normal"], plane["d"], "INVALID", False
        return points, plane["normal"], float(plane["d"]), "STEREO_ARUCO", True

    def _record_features(
        self,
        feature_landmarks: np.ndarray | None,
        feature_mode: str = "MONOCULAR_RELATIVE",
    ) -> None:
        """Append features only when the source mode matches the active reference.

        Using one mode consistently prevents a trainer trajectory captured from a
        monocular pose model from being silently mixed with stereo camera-space data.
        """
        if feature_landmarks is None:
            return
        features = self.dtw_analyzer.extract_kinematic_features(feature_landmarks)
        if not np.isfinite(features).all():
            return
        if self.recording_state == REC_GOLDEN:
            if self._recording_geometry_mode is None:
                self._recording_geometry_mode = feature_mode
            if feature_mode == self._recording_geometry_mode:
                self._record_buffer.append(features.copy())
        elif self.recording_state == REC_USER:
            expected = self.dtw_analyzer.reference_metadata.get("reconstruction_mode")
            if expected and expected != feature_mode:
                return
            self.dtw_analyzer.add_user_frame(features)

    def run(self) -> None:
        """Run the camera loop until the user quits or the source ends."""
        self.sys_logger.log(
            "AURA initialized. Press X to enter mirror mode.",
            f"Source={self.source}; exercise={self.exercise}; metric_calibration={self.camera_is_calibrated}",
        )
        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok or frame is None or frame.size == 0:
                    if isinstance(self.source, str):
                        # Loop file sources; do not loop a disconnected camera.
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self.cap.read()
                    if not ok or frame is None:
                        break
                self.frame_count += 1

                if self.camera_is_calibrated and self.apply_calibration:
                    frame = cv2.undistort(
                        frame,
                        self.camera_matrix,
                        self.dist_coeffs,
                        None,
                        self.camera_matrix,
                    )
                    active_distortion = np.zeros((5, 1), dtype=np.float64)
                else:
                    active_distortion = self.dist_coeffs

                marker_poses: dict = {}
                if self.aruco_tracker is not None and self.camera_is_calibrated:
                    frame, marker_poses = self.aruco_tracker.process_frame(
                        frame, self.camera_matrix, active_distortion
                    )

                # The full-frame detector remains the consistent source for HUD
                # and Golden Standard feature extraction in every application mode.
                display_frame = frame.copy()
                self.single_detector.detect(display_frame, draw=True)
                active_results = self.single_detector.results
                mono_landmarks = self.single_detector.get_world_landmark_array()
                raw_landmarks = (
                    mono_landmarks.copy() if mono_landmarks is not None else None
                )
                plane_normal = plane_d = None
                reconstruction_mode = "MONOCULAR_RELATIVE"
                stereo_valid = False

                if self.current_state == STATE_MIRROR:
                    if self.multi_detector is None:
                        ratio = self.splitter.update(frame)
                        if ratio is not None:
                            try:
                                self.multi_detector = MultiSkeletonDetector(
                                    split_ratio=ratio
                                )
                                self.sys_logger.log(
                                    "Mirror ROIs initialized.",
                                    f"Split ratio={ratio:.4f}",
                                )
                            except RuntimeError as exc:
                                self.sys_logger.log(
                                    "Mirror pose detector unavailable.", str(exc)
                                )
                        if self.multi_detector is None:
                            cv2.putText(
                                display_frame,
                                "Mirror mode: estimating ROI split",
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 200, 255),
                                2,
                            )
                    else:
                        pipeline = self.multi_detector.process_frame(frame)
                        (
                            raw_landmarks,
                            plane_normal,
                            plane_d,
                            reconstruction_mode,
                            stereo_valid,
                        ) = self._try_stereo_reconstruction(pipeline, marker_poses)
                        # Display both ROI crops for transparent debugging.
                        if pipeline.get("front_roi") is not None:
                            cv2.imshow("AURA - Direct ROI", pipeline["front_roi"])
                        if pipeline.get("mirror_roi") is not None:
                            cv2.imshow("AURA - Mirror ROI", pipeline["mirror_roi"])
                        if not stereo_valid and mono_landmarks is not None:
                            raw_landmarks = mono_landmarks.copy()
                            reconstruction_mode = "MONOCULAR_RELATIVE"

                # The rendered copy is grounded; raw metric or model coordinates are retained.
                render_landmarks = self._ground_landmarks(raw_landmarks)
                self._last_render_landmarks = render_landmarks
                self._last_plane_normal = plane_normal
                # Translate the plane equation along with the display-only skeleton grounding.
                render_plane_d = (
                    None
                    if plane_d is None or plane_normal is None
                    else float(
                        plane_d + np.dot(plane_normal, self._last_grounding_offset)
                    )
                )
                self._last_plane_d = render_plane_d
                self._last_reconstruction_mode = reconstruction_mode

                self.renderer.draw_postural_hud(
                    display_frame, active_results, self.posture_analyzer
                )
                feature_landmarks = raw_landmarks if stereo_valid else mono_landmarks
                feature_mode = (
                    reconstruction_mode if stereo_valid else "MONOCULAR_RELATIVE"
                )
                self._record_features(feature_landmarks, feature_mode)
                live_pacing = (
                    self.dtw_analyzer.check_live_pacing()
                    if self.recording_state == REC_USER
                    else ""
                )
                self.renderer.draw_control_panel(
                    display_frame,
                    self.recording_state,
                    self.dtw_last_result,
                    self.frame_count,
                    bool(self.dtw_analyzer.golden_reference),
                    live_pacing,
                )
                cv2.putText(
                    display_frame,
                    f"3D: {reconstruction_mode}",
                    (15, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 220) if stereo_valid else (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("AURA - Magic Mirror", display_frame)

                colors = self._evaluate_joint_colors(
                    active_results,
                    is_fused_valid=stereo_valid or self.current_state == STATE_SINGLE,
                )
                scene = self.viewer_3d.render_scene(
                    render_landmarks,
                    plane_normal,
                    colors,
                    render_plane_d,
                    reconstruction_mode,
                )
                cv2.imshow("AURA - 3D Digital Twin", scene)

                command, raw_key = KeyboardController.get_command(self.delay)
                self.viewer_3d.handle_keyboard(raw_key)
                if not self._handle_commands(command):
                    break
        finally:
            self.shutdown()

    def _handle_commands(self, cmd: str) -> bool:
        """Handle legacy recording commands plus explicit mirror-mode toggle."""
        if cmd == "QUIT":
            return False
        if cmd == "TOGGLE_LOG_MODE":
            self.sys_logger.toggle_mode()
        elif cmd == "TOGGLE_MIRROR":
            self.current_state = (
                STATE_SINGLE if self.current_state == STATE_MIRROR else STATE_MIRROR
            )
            if self.current_state == STATE_SINGLE:
                if self.multi_detector is not None:
                    self.multi_detector.close()
                    self.multi_detector = None
            else:
                self.splitter = AutoROISplitter()
                manual_ratio = os.environ.get("AURA_MIRROR_SPLIT_RATIO")
                if manual_ratio is not None:
                    try:
                        self.multi_detector = MultiSkeletonDetector(
                            split_ratio=float(manual_ratio)
                        )
                        self.sys_logger.log(
                            "Mirror ROIs initialized from configured split ratio."
                        )
                    except (ValueError, RuntimeError) as exc:
                        self.multi_detector = None
                        self.sys_logger.log(
                            "Configured mirror split rejected.", str(exc)
                        )
            self.sys_logger.log(f"Mode changed: {self.current_state}")
        elif cmd == "REC_GOLDEN":
            self.recording_state = REC_GOLDEN
            self._record_buffer = []
            self._recording_geometry_mode = None
            self.dtw_last_result = None
            self.sys_logger.log(
                "Recording trainer reference.",
                "Recording four 3D joint-angle features from the full-frame MediaPipe pose.",
            )
        elif cmd == "REC_USER":
            if not self.dtw_analyzer.golden_reference:
                self.sys_logger.log("Load or record a trainer reference first.")
            else:
                self.recording_state = REC_USER
                self.dtw_analyzer.reset_user_buffer()
                self.dtw_last_result = None
                expected_mode = self.dtw_analyzer.reference_metadata.get(
                    "reconstruction_mode"
                )
                self.sys_logger.log(
                    "Recording user execution.",
                    f"Reference reconstruction mode={expected_mode or 'legacy/unspecified'}",
                )
        elif cmd == "STOP_REC":
            if self.recording_state == REC_GOLDEN:
                if len(self._record_buffer) >= 5:
                    self.dtw_analyzer.load_golden_standard(
                        np.vstack(self._record_buffer)
                    )
                    self._golden_geometry_mode = self._recording_geometry_mode
                    self.dtw_analyzer.reference_metadata = {
                        "reconstruction_mode": self._golden_geometry_mode,
                        "exercise": self.exercise,
                    }
                    self.sys_logger.log(
                        f"Trainer reference captured ({len(self._record_buffer)} frames, mode={self._golden_geometry_mode})."
                    )
                else:
                    self.sys_logger.log(
                        "Trainer reference rejected: not enough valid frames."
                    )
            elif self.recording_state == REC_USER:
                self.dtw_last_result = self.dtw_analyzer.evaluate_technique()
                self.sys_logger.log(
                    "User execution evaluated.", str(self.dtw_last_result)
                )
            self.recording_state = REC_IDLE
            self._recording_geometry_mode = None
        elif cmd == "SAVE_GOLDEN":
            success = self.dtw_analyzer.save_golden_standard(self.exercise)
            self.sys_logger.log(
                "Golden Standard saved." if success else "No Golden Standard to save."
            )
        elif cmd == "LOAD_GOLDEN":
            success = self.dtw_analyzer.load_golden_standard_from_file(self.exercise)
            self._golden_geometry_mode = self.dtw_analyzer.reference_metadata.get(
                "reconstruction_mode"
            )
            self.sys_logger.log(
                "Golden Standard loaded."
                if success
                else "Golden Standard not found or incompatible."
            )
        elif cmd == "DELETE_GOLDEN":
            success = self.dtw_analyzer.delete_golden_standard_file(self.exercise)
            self.sys_logger.log(
                "Golden Standard deleted."
                if success
                else "No Golden Standard file found."
            )
        elif cmd == "VIEW_STATUS":
            self.sys_logger.log(
                f"Mode={self.current_state}; calibration={self.camera_is_calibrated}; reconstruction={self._last_reconstruction_mode}; recording={self.recording_state}"
            )
        return True

    def shutdown(self) -> None:
        """Release camera, MediaPipe instances, and OpenCV windows safely."""
        if getattr(self, "multi_detector", None) is not None:
            self.multi_detector.close()
            self.multi_detector = None
        if getattr(self, "single_detector", None) is not None:
            self.single_detector.close()
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        cv2.destroyAllWindows()
