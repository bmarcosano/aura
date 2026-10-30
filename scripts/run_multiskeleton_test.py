"""Exercise the real MultiSkeletonDetector ROI-to-full-frame coordinate path."""

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

from aura.multi_detector import MultiSkeletonDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test AURA direct/mirror ROI pose detection"
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--split", type=float, default=0.5)
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open source: {source}")
    detector = MultiSkeletonDetector(split_ratio=args.split)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            data = detector.process_frame(frame)
            front = data["front_roi"].copy()
            mirror = data["mirror_roi"].copy()
            cv2.putText(
                front,
                f"Direct valid: {data['direct_pose_2d'] is not None}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                mirror,
                f"Mirror valid: {data['reflected_pose_2d'] is not None}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow("AURA - Direct ROI", front)
            cv2.imshow("AURA - Mirror ROI", mirror)
            # Debug output coordinates are expected in full-frame pixel coordinates.
            if (
                data["direct_pose_2d"] is not None
                and data["reflected_pose_2d"] is not None
            ):
                print(
                    "split_x=",
                    data["split_x"],
                    "direct[0]=",
                    data["direct_pose_2d"][0],
                    "mirror[0]=",
                    data["reflected_pose_2d"][0],
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
