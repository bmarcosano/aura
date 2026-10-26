"""
logger.py: Dual-mode system logger UI for real-time AI explainability (XAI).
Renders an OpenCV window with automatic text wrapping.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np


class AuraLogger:
    """
    Manages and renders real-time system logs in two distinct modes:
    - USER: Simple, intuitive explanations of system state.
    - EXPERT: Deep mathematical and implementation details (SVD, ML, thresholds).
    """

    def __init__(
        self, width: int = 450, height: int = 700, max_lines: int = 35
    ) -> None:
        self.width = width
        self.height = height
        self.max_lines = max_lines
        self.mode = "USER"  # Default mode
        self.logs = deque(maxlen=self.max_lines)

    def toggle_mode(self) -> None:
        """Switches verbosity mode between USER and EXPERT."""
        self.mode = "EXPERT" if self.mode == "USER" else "USER"
        self.log(
            user_msg=f"Switched to {self.mode} mode.",
            expert_msg=f"UI.Logger.Mode switched -> {self.mode}. Verbosity updated.",
        )

    def log(self, user_msg: str, expert_msg: str = "") -> None:
        """
        Pushes a new dual-mode message into the ring buffer.
        If expert_msg is not provided, user_msg is used for both.
        """
        time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        target_msg = expert_msg if (self.mode == "EXPERT" and expert_msg) else user_msg

        full_msg = f"[{time_str}] {target_msg}"

        # Simple text wrapping algorithm
        wrapped_lines = self._wrap_text(full_msg, max_chars=55)
        for line in wrapped_lines:
            self.logs.append(line)

    def _wrap_text(self, text: str, max_chars: int) -> list[str]:
        """Wraps text into multiple lines based on character limit."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        return lines

    def render(self) -> np.ndarray:
        """Draws the log history onto a dark canvas."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)  # Dark grey background

        # Header
        header_color = (0, 215, 255) if self.mode == "EXPERT" else (0, 255, 150)
        cv2.putText(
            canvas,
            f"AURA SYSTEM LOGS - {self.mode}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            header_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Press 'M' to toggle explanation mode",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        # Separator line
        cv2.line(canvas, (15, 75), (self.width - 15, 75), (100, 100, 100), 1)

        # Draw log lines from top to bottom
        y_offset = 105
        for line in self.logs:
            # Highlight time tag in a different color
            if line.startswith("["):
                time_part = line[:10]
                msg_part = line[10:]
                cv2.putText(
                    canvas,
                    time_part,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (100, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    msg_part,
                    (95, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
            else:
                # Wrapped lines without time tag
                cv2.putText(
                    canvas,
                    line,
                    (95, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            y_offset += 22

        return canvas
