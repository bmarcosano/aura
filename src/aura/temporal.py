"""Multidimensional DTW comparison and versioned Golden Standard persistence."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import numpy as np

from aura.geometry import calculate_angle


class DTWTemporalAnalyzer:
    """Compare exercise feature trajectories with a trainer-recorded reference.

    The legacy method signatures and the ``golden_reference`` JSON key are
    preserved. Scalar legacy references can be loaded, but are never silently
    compared with a multidimensional feature sequence.
    """

    DEFAULT_FEATURES: typing.ClassVar[list[str]] = [
        "left_knee",
        "right_knee",
        "left_elbow",
        "right_elbow",
    ]

    def __init__(self, sensitivity: float = 5.0) -> None:
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        self.golden_reference: list = []
        self.user_buffer: list = []
        self.sensitivity = float(sensitivity)
        self.feature_names = list(self.DEFAULT_FEATURES)
        self.schema_version = 2
        # Optional provenance prevents mixing incompatible reference/user pipelines.
        self.reference_metadata: dict = {}

    def _compute_angle_3d(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute a 3D joint angle; NaN marks missing/degenerate observations."""
        return calculate_angle(a, b, c)

    def extract_kinematic_features(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """Extract left/right knee and elbow angles from 33 3D landmarks."""
        points = np.asarray(landmarks_3d, dtype=np.float64)
        if points.shape != (33, 3):
            return np.full(4, np.nan, dtype=np.float32)
        values = [
            self._compute_angle_3d(points[23], points[25], points[27]),
            self._compute_angle_3d(points[24], points[26], points[28]),
            self._compute_angle_3d(points[11], points[13], points[15]),
            self._compute_angle_3d(points[12], points[14], points[16]),
        ]
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _as_trajectory(trajectory: list | np.ndarray) -> np.ndarray:
        array = np.asarray(trajectory, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)  # Preserve one-feature legacy recordings.
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("Trajectory must be a non-empty 1D or 2D array")
        if not np.isfinite(array).all():
            raise ValueError("Trajectory contains non-finite feature values")
        return array

    def load_golden_standard(self, trajectory: list | np.ndarray) -> None:
        """Load a 1D legacy or 2D feature trajectory into memory."""
        array = self._as_trajectory(trajectory)
        self.golden_reference = array.tolist()
        self.reference_metadata = {}

    def save_golden_standard(
        self, exercise: str = "squats", filename: str = "golden.json"
    ) -> bool:
        """Save the Golden Standard in a versioned JSON envelope."""
        if not self.golden_reference:
            return False
        path = Path("golden_executions") / exercise / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": self.schema_version,
                "exercise": exercise,
                "feature_names": self.feature_names,
                "coordinate_units": "degrees",
                "metadata": self.reference_metadata,
                "golden_reference": self.golden_reference,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except (OSError, TypeError, ValueError):
            return False

    def load_golden_standard_from_file(
        self, exercise: str = "squats", filename: str = "golden.json"
    ) -> bool:
        """Load current or legacy Golden Standard JSON without dimension guessing."""
        path = Path("golden_executions") / exercise / filename
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            reference = self._as_trajectory(payload["golden_reference"])
            names = payload.get("feature_names")
            if names is not None and len(names) != reference.shape[1]:
                return False
            self.golden_reference = reference.tolist()
            if names:
                self.feature_names = list(names)
            elif reference.shape[1] == 1:
                self.feature_names = ["legacy_scalar"]
            self.reference_metadata = dict(payload.get("metadata", {}))
            return True
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def delete_golden_standard_file(
        self, exercise: str = "squats", filename: str = "golden.json"
    ) -> bool:
        """Delete a saved Golden Standard file."""
        path = Path("golden_executions") / exercise / filename
        try:
            if not path.is_file():
                return False
            path.unlink()
            return True
        except OSError:
            return False

    def add_user_frame(self, frame_features: np.ndarray | list[float] | float) -> None:
        """Append one finite feature vector; skip invalid frames instead of storing zeros."""
        values = np.asarray(frame_features, dtype=np.float32).reshape(-1)
        if values.size and np.isfinite(values).all():
            self.user_buffer.append(values.tolist())

    def reset_user_buffer(self) -> None:
        """Clear current user sequence."""
        self.user_buffer.clear()

    def check_live_pacing(self) -> str:
        """Return a coarse duration-based pacing hint, not a technique score."""
        if not self.golden_reference or not self.user_buffer:
            return "PACE: IDLE"
        ratio = len(self.user_buffer) / max(1, len(self.golden_reference))
        if ratio > 1.25:
            return "PACE: TOO SLOW"
        if ratio < 0.75 and len(self.user_buffer) > 10:
            return "PACE: TOO FAST"
        return "PACE: OPTIMAL"

    @staticmethod
    def _dtw_distance(
        sequence_a: np.ndarray, sequence_b: np.ndarray
    ) -> tuple[float, int]:
        """Compute exact DTW using Euclidean frame cost and return path length."""
        n, m = len(sequence_a), len(sequence_b)
        cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        length = np.zeros((n + 1, m + 1), dtype=np.int32)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                local = float(np.linalg.norm(sequence_a[i - 1] - sequence_b[j - 1]))
                options = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
                move = int(np.argmin(options))
                if move == 0:
                    pi, pj = i - 1, j - 1
                elif move == 1:
                    pi, pj = i - 1, j
                else:
                    pi, pj = i, j - 1
                cost[i, j] = local + options[move]
                length[i, j] = length[pi, pj] + 1
        return float(cost[n, m]), int(length[n, m])

    def evaluate_technique(self) -> dict:
        """Align feature sequences and return a transparent algorithmic score."""
        if not self.golden_reference:
            return {
                "error": "No Golden Standard loaded.",
                "score": 0.0,
                "raw_distance": 0.0,
                "status": "NO_REFERENCE",
            }
        if len(self.user_buffer) < 5:
            return {
                "error": "User movement too short for analysis.",
                "score": 0.0,
                "raw_distance": 0.0,
                "status": "TOO_SHORT",
            }
        try:
            user = self._as_trajectory(self.user_buffer)
            golden = self._as_trajectory(self.golden_reference)
        except ValueError as exc:
            return {
                "error": str(exc),
                "score": 0.0,
                "raw_distance": float("inf"),
                "status": "INVALID_DATA",
            }
        if user.shape[1] != golden.shape[1]:
            return {
                "error": f"Feature mismatch: user has {user.shape[1]}, reference has {golden.shape[1]}.",
                "score": 0.0,
                "raw_distance": float("inf"),
                "status": "INCOMPATIBLE_REFERENCE",
            }
        raw, path_length = self._dtw_distance(user, golden)
        average = raw / max(1, path_length)
        score = float(np.clip(100.0 - self.sensitivity * average, 0.0, 100.0))
        feedback = (
            "EXCELLENT"
            if score >= 90
            else "GOOD"
            if score >= 75
            else "FAIR - Review movement form"
            if score >= 50
            else "DEVIATION - Trainer review recommended"
        )
        return {
            "status": "SUCCESS",
            "score": round(score, 1),
            "raw_distance": raw,
            "deviation_per_frame": round(average, 3),
            "feedback": feedback,
            "frames_analyzed": len(user),
            "feature_count": user.shape[1],
        }
