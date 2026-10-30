"""Keyboard command mapping for the AURA OpenCV interface."""

from __future__ import annotations

from typing import ClassVar

import cv2


class KeyboardController:
    """Map keyboard events to stable command names."""

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
        ord("x"): "TOGGLE_MIRROR",
        ord("X"): "TOGGLE_MIRROR",
    }

    @classmethod
    def get_command(cls, delay: int = 33) -> tuple[str, int]:
        """Return (command, raw OpenCV key code); preserve the legacy signature."""
        raw_key = cv2.waitKeyEx(delay)
        if raw_key == -1:
            return "NONE", -1
        return cls.COMMAND_MAP.get(raw_key & 0xFF, "NONE"), raw_key
