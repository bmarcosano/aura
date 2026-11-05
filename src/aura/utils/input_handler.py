"""
input_handler.py: Keypress controller and command mapper for AURA.
"""

from __future__ import annotations

from typing import ClassVar

import cv2


class KeyboardController:
    """Maps raw keyboard inputs to AURA FSM commands."""

    COMMAND_MAP: ClassVar[dict[int, str]] = {
        ord("g"): "REC_GOLDEN",
        ord("G"): "REC_GOLDEN",
        ord("u"): "REC_USER",
        ord("U"): "REC_USER",
        ord("s"): "STOP_REC",
        ord("S"): "STOP_REC",
        ord("w"): "SAVE_GOLDEN",
        ord("W"): "SAVE_GOLDEN",
        ord("l"): "LOAD_GOLDEN",
        ord("L"): "LOAD_GOLDEN",
        ord("v"): "VIEW_STATUS",
        ord("V"): "VIEW_STATUS",
        ord("d"): "DELETE_GOLDEN",
        ord("D"): "DELETE_GOLDEN",
        ord("q"): "QUIT",
        ord("Q"): "QUIT",
        ord("m"): "TOGGLE_LOG_MODE",
        ord("M"): "TOGGLE_LOG_MODE",
    }

    @classmethod
    def get_command(cls, delay: int = 33) -> tuple[str, int]:
        """
        Polls OpenCV keypress. Returns (command_name, raw_key_code).
        Uses waitKeyEx to properly capture extended keys (like Arrows).
        """
        raw_key = cv2.waitKeyEx(delay)
        if raw_key == -1:
            return "NONE", -1

        # Extract lower byte for standard ASCII matching if it's a standard key
        char_key = raw_key & 0xFF
        cmd = cls.COMMAND_MAP.get(char_key, "NONE")

        return cmd, raw_key
