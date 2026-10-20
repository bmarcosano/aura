"""
Module for Temporal Biomechanical Analysis using Dynamic Time Warping (DTW).

This module analyzes the execution of a movement over time (a trajectory),
comparing a user's performance against a pre-recorded "Golden Standard" (e.g., a trainer).
It uses the fastdtw algorithm to align the sequences elastically.
It also supports saving, loading, and managing the Golden Standard to/from a JSON file.
"""

from __future__ import annotations

import json
import os

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class DTWTemporalAnalyzer:
    def __init__(self, sensitivity: float = 5.0):
        """
        Initializes the Temporal Analyzer.

        :param sensitivity: Determines how strictly the score drops as the DTW distance increases.
        """
        self.golden_reference: list[float] = []
        self.user_buffer: list[float] = []
        self.sensitivity = sensitivity

    def load_golden_standard(self, trajectory: list[float]) -> None:
        """Loads the ideal movement execution array directly from memory."""
        if not trajectory:
            raise ValueError("Golden standard trajectory cannot be empty.")
        self.golden_reference = trajectory
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
        Saves the currently recorded Golden Standard to a structured exercise directory
        under golden_executions/<exercise>/<filename>.
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
        """Loads a previously saved Golden Standard from golden_executions/<exercise>/<filename>."""
        filepath = os.path.join("golden_executions", exercise, filename)

        if not os.path.exists(filepath):
            print(f"⚠️ Error: File '{filepath}' not found on disk.")
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if "golden_reference" in data:
                self.golden_reference = data["golden_reference"]
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

    def add_user_frame(self, metric_value: float) -> None:
        """Appends a real-time biomechanical metric to the user buffer."""
        self.user_buffer.append(metric_value)

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
        """Warp-aligns the user's buffer against the Golden Standard using FastDTW."""
        if not self.golden_reference:
            return {"error": "No Golden Standard loaded.", "score": 0}

        if len(self.user_buffer) < 5:
            return {"error": "User movement too short for analysis.", "score": 0}

        ref_arr = np.array(self.golden_reference).reshape(-1, 1)
        user_arr = np.array(self.user_buffer).reshape(-1, 1)

        raw_distance, path = fastdtw(ref_arr, user_arr, dist=euclidean)

        avg_deviation = raw_distance / len(path)
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

        return {
            "status": "SUCCESS",
            "score": round(final_score, 1),
            "deviation_per_frame": round(avg_deviation, 3),
            "feedback": feedback,
            "frames_analyzed": len(self.user_buffer),
        }
