"""Simple full-frame webcam test for MediaPipe and the posture analyzer."""

from __future__ import annotations

# Support direct execution from a source checkout before editable installation.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse

import cv2
import numpy as np

from aura.analyzer import PostureAnalyzer
from aura.detector import PoseDetector
from aura.renderer import AuraRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test AURA full-frame pose tracking")
    parser.add_argument("--source", default="0")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    detector = PoseDetector()
    analyzer = PostureAnalyzer()
    renderer = AuraRenderer()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            detector.detect(frame, draw=True)
            renderer.draw_postural_hud(frame, detector.results, analyzer)
            points = detector.get_world_landmark_array()
            if points is not None and np.isfinite(points).all():
                renderer.draw_control_panel(frame, "IDLE", None, 0, False, "")
            cv2.imshow("AURA - Webcam Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
