"""
input_handler.py: Maps keyboard inputs to core system commands.
"""

import cv2


class KeyboardController:
    """Parses OpenCV waitKey events and translates them into system actions."""

    @staticmethod
    def get_command(delay: int) -> str:
        """
        Waits for a key press for the given delay (ms).
        Returns a string command mapping to the pressed key.
        """
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            return "QUIT"
        elif key == ord("g"):
            return "REC_GOLDEN"
        elif key == ord("u"):
            return "REC_USER"
        elif key == ord("s"):
            return "STOP_REC"
        elif key == ord("w"):
            return "SAVE_GOLDEN"
        elif key == ord("l"):
            return "LOAD_GOLDEN"
        elif key == ord("v"):
            return "VIEW_STATUS"
        elif key == ord("d"):
            return "DELETE_GOLDEN"

        return "NONE"
