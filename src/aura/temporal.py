"""
Module for Temporal Biomechanical Analysis using Multidimensional Dynamic Time Warping (DTW).

This module analyzes movement execution over time by extracting 3D joint angles
from reconstructed skeletal landmarks and comparing the live sequence against a pre-recorded
"Golden Standard" (e.g., a trainer). It supports JSON persistence for multi-feature trajectories.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.distance import cdist


class DTWTemporalAnalyzer:
    """
    Analyzes temporal execution of movement using multidimensional 3D kinematic feature vectors
    (e.g., knee and elbow angles) and Dynamic Time Warping (DTW).
    """

    def __init__(self, sensitivity: float = 5.0) -> None:
        """
        Initializes the Temporal Analyzer.

        Args:
            sensitivity: Scale factor for scoring penalty as DTW distance increases.
        """
        self.golden_reference: list = []  # Time x Features or Time x 1
        self.user_buffer: list = []  # Time x Features or Time x 1
        self.sensitivity = sensitivity

    def _compute_angle_3d(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Computes the 3D angle (in degrees) between three 3D points (a, b, c) with b as the vertex.
        """
        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cosine_angle)))

    def extract_kinematic_features(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Extracts key biomechanical joint angles (knee & elbow 3D angles) from a 3D skeleton.
        Returns a 4-dimensional feature vector.
        """
        try:
            left_knee = self._compute_angle_3d(
                landmarks_3d[23], landmarks_3d[25], landmarks_3d[27]
            )
            right_knee = self._compute_angle_3d(
                landmarks_3d[24], landmarks_3d[26], landmarks_3d[28]
            )
            left_elbow = self._compute_angle_3d(
                landmarks_3d[11], landmarks_3d[13], landmarks_3d[15]
            )
            right_elbow = self._compute_angle_3d(
                landmarks_3d[12], landmarks_3d[14], landmarks_3d[16]
            )

            return np.array(
                [left_knee, right_knee, left_elbow, right_elbow], dtype=np.float32
            )
        except (IndexError, TypeError):
            return np.zeros(4, dtype=np.float32)

    def load_golden_standard(self, trajectory: list | np.ndarray) -> None:
        """Loads the ideal movement execution feature matrix directly from memory."""
        if trajectory is None or len(trajectory) == 0:
            raise ValueError("Golden standard trajectory cannot be empty.")

        if isinstance(trajectory, np.ndarray):
            self.golden_reference = trajectory.tolist()
        else:
            self.golden_reference = list(trajectory)

        print(
            f"✅ Golden Standard loaded in memory ({len(self.golden_reference)} frames)."
        )

    # ==========================================
    # DATA PERSISTENCE & FILE MANAGEMENT
    # ==========================================

    def save_golden_standard(
        self, exercise: str = "squats", filename: str = "golden.json"
    ) -> bool:
        """
        Saves the currently recorded Golden Standard to a structured exercise directory.
        """
        if not self.golden_reference:
            print("⚠️ Error: No Golden Standard in memory to save.")
            return False

        dir_path = os.path.join("golden_executions", exercise)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, filename)

        try:
            with open(filepath, "w") as f:
                json.dump({"golden_reference": self.golden_reference}, f, indent=4)
            print(
                f"💾 Saved successfully: {filepath} ({len(self.golden_reference)} frames)"
            )
            return True
        except (OSError, TypeError) as e:
            print(f"❌ Error saving to file: {e}")
            return False

    def load_golden_standard_from_file(
        self, exercise: str = "squats", filename: str = "golden.json"
    ) -> bool:
        """
        Loads a previously saved Golden Standard from golden_executions/<exercise>/<filename>.
        Includes backward compatibility checks to prevent crashes with legacy 1D (scalar) recordings.
        """
        filepath = os.path.join("golden_executions", exercise, filename)

        if not os.path.exists(filepath):
            print(f"⚠️ Error: File '{filepath}' not found on disk.")
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if "golden_reference" in data:
                ref_data = np.array(data["golden_reference"])

                # CRITICAL FIX: Block legacy 1D scalar recordings from crashing scipy.cdist
                if ref_data.ndim == 1:
                    print(
                        "⚠️ WARNING: Found legacy Golden Standard (scalar). Incompatible with new 4-angle vectors."
                    )
                    self.golden_reference = []
                    return False

                self.golden_reference = ref_data.tolist()
                print(
                    f"📂 Loaded successfully: {filepath} ({len(self.golden_reference)} frames)."
                )
                return True
            else:
                print(f"⚠️ Error: '{filepath}' is missing 'golden_reference' data.")
                return False
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"❌ Error reading file: {e}")
            return False

    def delete_golden_standard_file(
        self, exercise: str = "squats", filename: str = "golden.json"
    ) -> bool:
        """Deletes the specific exercise Golden Standard file from disk."""
        filepath = os.path.join("golden_executions", exercise, filename)

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"🗑️ File deleted successfully: {filepath}")
                return True
            except OSError as e:
                print(f"❌ Error deleting file: {e}")
                return False
        else:
            print(f"⚠️ File '{filepath}' does not exist on disk.")
            return False

    # ==========================================
    # USER BUFFER & EVALUATION
    # ==========================================

    def add_user_frame(self, frame_features: np.ndarray | list[float] | float) -> None:
        """
        Appends a real-time biomechanical feature vector (or scalar) to the user buffer.
        """
        if isinstance(frame_features, np.ndarray):
            self.user_buffer.append(frame_features.tolist())
        elif isinstance(frame_features, (list, tuple)):
            self.user_buffer.append(list(frame_features))
        else:
            self.user_buffer.append(float(frame_features))

    def reset_user_buffer(self) -> None:
        """Clears the user buffer to prepare for the next repetition."""
        self.user_buffer.clear()

    def check_live_pacing(self) -> str:
        """
        Performs a lightweight real-time pace check by comparing
        the user's current progress against the loaded Golden Standard length.
        """
        if not self.golden_reference or not self.user_buffer:
            return "PACE: IDLE"

        ref_len = len(self.golden_reference)
        user_len = len(self.user_buffer)
        ratio = user_len / ref_len

        if ratio > 1.25:
            return "PACE: TOO SLOW (Speed up to match the trainer)"
        elif ratio < 0.75 and user_len > 10:
            return "PACE: TOO FAST (Slow down to match the trainer)"
        else:
            return "PACE: OPTIMAL"

    def evaluate_technique(self) -> dict:
        """
        Warp-aligns the user's feature buffer against the Golden Standard using
        multidimensional Dynamic Time Warping (DTW).

        Returns:
            Dictionary containing evaluation status, score, raw distance, and feedback.
        """
        if not self.golden_reference:
            return {
                "error": "No Golden Standard loaded.",
                "score": 0,
                "raw_distance": 0.0,
            }

        if len(self.user_buffer) < 5:
            return {
                "error": "User movement too short for analysis.",
                "score": 0,
                "raw_distance": 0.0,
            }

        seq1 = np.asarray(self.user_buffer, dtype=np.float32)
        seq2 = np.asarray(self.golden_reference, dtype=np.float32)

        if seq1.ndim == 1:
            seq1 = seq1.reshape(-1, 1)
        if seq2.ndim == 1:
            seq2 = seq2.reshape(-1, 1)

        dist_matrix = cdist(seq1, seq2, metric="euclidean")

        T, K = dist_matrix.shape
        cost_matrix = np.full((T + 1, K + 1), float("inf"))
        cost_matrix[0, 0] = 0.0

        for i in range(1, T + 1):
            for j in range(1, K + 1):
                cost = dist_matrix[i - 1, j - 1]
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i - 1, j],  # insertion
                    cost_matrix[i, j - 1],  # deletion
                    cost_matrix[i - 1, j - 1],  # match
                )

        raw_distance = cost_matrix[T, K]

        # Backtrack optimal warping path to find exact path length
        i, j = T, K
        path_len = 0
        while i > 0 and j > 0:
            path_len += 1
            c_match = cost_matrix[i - 1, j - 1]
            c_ins = cost_matrix[i - 1, j]
            c_del = cost_matrix[i, j - 1]
            m = min(c_match, c_ins, c_del)
            if m == c_match:
                i -= 1
                j -= 1
            elif m == c_ins:
                i -= 1
            else:
                j -= 1
        path_len += i + j

        avg_deviation = raw_distance / max(1, path_len)

        raw_score = 100.0 - (avg_deviation * self.sensitivity)
        final_score = max(0.0, min(100.0, raw_score))

        if final_score >= 90:
            feedback = "EXCELLENT"
        elif final_score >= 75:
            feedback = "GOOD"
        elif final_score >= 50:
            feedback = "FAIR - Form breaking down"
        else:
            feedback = "POOR - Biomechanical risk"

        # CRITICAL FIX: raw_distance added to return dict to prevent Engine errors
        return {
            "status": "SUCCESS",
            "score": round(float(final_score), 1),
            "raw_distance": float(raw_distance),
            "deviation_per_frame": round(float(avg_deviation), 3),
            "feedback": feedback,
            "frames_analyzed": len(self.user_buffer),
        }
