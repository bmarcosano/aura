"""Compact OpenCV event and telemetry logger for the AURA interface."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class AuraLogger:
    """Store coach/expert messages and render bounded diagnostic panels."""

    def __init__(self) -> None:
        self.expert_mode = False
        self.events: deque[tuple[str, str]] = deque(maxlen=100)
        self.static_events: dict[str, str] = {}
        self.metrics: dict[str, str] = {}
        self.max_visible_events = 8

    def toggle_mode(self) -> None:
        self.expert_mode = not self.expert_mode

    def log(self, user_msg: str, expert_msg: str = "") -> None:
        self.events.append((str(user_msg), str(expert_msg or user_msg)))

    def log_event(self, message: str) -> None:
        self.log(message, message)

    def log_static(self, key: str, message: str) -> None:
        self.static_events[str(key)] = str(message)

    def log_metric(self, key: str, value: str) -> None:
        self.metrics[str(key)] = str(value)

    def render(self) -> np.ndarray:
        """Render the latest event panel and metrics panel into one image."""
        events = self.render_events()
        metrics = self.render_metrics()
        canvas = np.zeros(
            (
                max(events.shape[0], metrics.shape[0]),
                events.shape[1] + metrics.shape[1],
                3,
            ),
            dtype=np.uint8,
        )
        canvas[: events.shape[0], : events.shape[1]] = events
        canvas[: metrics.shape[0], events.shape[1] :] = metrics
        return canvas

    def render_events(self) -> np.ndarray:
        """Render static and recent events, safely bounded by the canvas height."""
        canvas = np.zeros((420, 560, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)
        cv2.putText(
            canvas,
            "AURA - SYSTEM EVENTS",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y = 52
        for key, message in list(self.static_events.items())[-4:]:
            if y > 390:
                break
            cv2.putText(
                canvas,
                f"[{key}] {message[:68]}",
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 210, 255),
                1,
                cv2.LINE_AA,
            )
            y += 19
        if self.static_events:
            y += 5
        for user_msg, expert_msg in list(self.events)[-self.max_visible_events :]:
            if y > 395:
                break
            text = expert_msg if self.expert_mode else user_msg
            cv2.putText(
                canvas,
                text[:72],
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 220, 180),
                1,
                cv2.LINE_AA,
            )
            y += 24
        return canvas

    def render_metrics(self) -> np.ndarray:
        """Render the most recent telemetry values."""
        canvas = np.zeros((420, 300, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        cv2.putText(
            canvas,
            "AURA - TELEMETRY",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y = 55
        for key, value in list(self.metrics.items())[-15:]:
            if y > 395:
                break
            cv2.putText(
                canvas,
                f"{key}: {value}"[:38],
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22
        return canvas
