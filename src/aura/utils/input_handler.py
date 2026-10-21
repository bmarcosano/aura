"""
input_handler.py: Maps keyboard inputs to core system commands and raw key events.
"""

from __future__ import annotations

import cv2


class KeyboardController:
    """Parses OpenCV waitKey events and translates them into system actions."""

    @staticmethod
    def get_command(delay: int) -> tuple[str, int]:
        """
        Waits for a key press for the given delay (ms).
        Returns a tuple of (string command, raw key code).
        """
        key = cv2.waitKey(delay) & 0xFF

        if key == 255:  # No key pressed
            return "NONE", -1

        cmd = "NONE"
        if key in (ord("q"), ord("Q"), 27):
            cmd = "QUIT"
        elif key in (ord("g"), ord("G")):
            cmd = "REC_GOLDEN"
        elif key in (ord("u"), ord("U")):
            cmd = "REC_USER"
        elif key in (ord("s"), ord("S")):
            cmd = "STOP_REC"
        elif key in (ord("l"), ord("L")):
            cmd = "LOAD_GOLDEN"
        elif key in (ord("v"), ord("V")):
            cmd = "VIEW_STATUS"
        elif key in (ord("x"), ord("X")):
            cmd = "DELETE_GOLDEN"
        elif key in (ord("m"), ord("M")):
            cmd = "TOGGLE_LOG_MODE"

        return cmd, key
