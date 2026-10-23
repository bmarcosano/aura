"""
logger.py: Dual-mode logging system for AURA.
Separates persistent system events from high-frequency live telemetry.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np


class AuraLogger:
    """
    Manages and renders real-time system logs in two distinct modes:
    - USER: Simple, intuitive explanations of system state.
    - EXPERT: Deep mathematical and implementation details (SVD, ML, thresholds).
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.static_events: dict[str, str] = {}
        self.metrics: dict[str, str] = {}
        self.max_events = 22
        self.expert_mode = True

    def toggle_mode(self) -> None:
        self.expert_mode = not self.expert_mode
        self.log_event(f"Log mode toggled. Expert view: {self.expert_mode}")

    def log(self, user_msg: str, expert_msg: str = "") -> None:
        """Backwards compatibility for existing engine.py calls."""
        msg = expert_msg if self.expert_mode and expert_msg else user_msg
        self.log_event(msg)

    def log_event(self, message: str) -> None:
        """Appends persistent events (initializations, state changes) to a scrolling list."""
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self.events.append(f"[{timestamp}] {message}")

        # Calculate how many slots are left for scrolling events
        available_slots = self.max_events - len(self.static_events)
        if len(self.events) > available_slots:
            self.events.pop(0)

    def log_static(self, key: str, message: str) -> None:
        """Updates a fixed-position log in the System Events window (no scrolling)."""
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self.static_events[key] = f"[{timestamp}] {message}"

    def log_metric(self, key: str, value: str) -> None:
        """Updates continuous live telemetry in the Live Telemetry window."""
        self.metrics[key] = value

    def render(self) -> np.ndarray:
        """Backwards compatibility: returns the event canvas by default."""
        return self.render_events()

    def render_events(self) -> np.ndarray:
        """Renders the scrolling list of system events and fixed static events."""
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "SYSTEM EVENTS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.line(canvas, (10, 40), (590, 40), (0, 255, 255), 1)

        y_offset = 65

        # 1. Render fixed/static events at the top (highlighted in soft orange)
        for key, msg in self.static_events.values():
            cv2.putText(
                canvas,
                msg,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 200, 255),
                1,
            )
            y_offset += 15

        # 2. Render scrolling events below them (standard grey)
        for ev in self.events:
            cv2.putText(
                canvas,
                ev,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            y_offset += 15

        return canvas

    def render_metrics(self) -> np.ndarray:
        """Renders the static dashboard for live telemetry metrics."""
        canvas = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "LIVE TELEMETRY",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.line(canvas, (10, 40), (390, 40), (0, 255, 0), 1)

        y_offset = 75
        for key, val in self.metrics.items():
            text = f"{key}: {val}"
            cv2.putText(
                canvas,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                1,
            )
            y_offset += 25

        return canvas
