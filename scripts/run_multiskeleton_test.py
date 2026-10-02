"""Script to test the MultiSkeletonDetector pipeline with video or webcam input including landmark rendering."""

from __future__ import annotations

import argparse
import sys
import cv2
from aura.detector import PoseDetector
from aura.multi_detector import MultiSkeletonDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test MultiSkeletonDetector and skeleton rendering on a video file or webcam."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: path to a video file or webcam index (default: '0' for webcam)",
    )
    args = parser.parse_args()

    # Parse source: if it's a digit, treat it as a webcam index, otherwise as a file path
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    # Initialize multi-detector for ROI slicing and separate pose detectors for each view
    multi_detector = MultiSkeletonDetector(split_ratio=0.6)
    front_pose_detector = PoseDetector()
    mirror_pose_detector = PoseDetector()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{source}'")
        sys.exit(1)

    # Retrieve video FPS to synchronize playback speed (fallback to ~30 FPS if unavailable/webcam)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    print(f"🚀 Starting test using source: {source} (FPS: {fps if fps > 0 else 'N/A'}). Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        
        # Safety check: ensure frame is successfully read and not empty
        if not success or frame is None or frame.size == 0:
            print("End of video stream or empty frame received.")
            break

        # 1. Crop the frame into Front View and Mirror View ROIs
        front_roi, mirror_roi = multi_detector._crop_rois(frame)

        # Safety check: ensure cropped ROIs have valid dimensions before processing
        if front_roi.size == 0 or mirror_roi.size == 0 or front_roi.shape[0] == 0 or mirror_roi.shape[0] == 0:
            print("⚠️ Warning: Invalid ROI dimensions, skipping frame.")
            continue

        # 2. Run MediaPipe pose detection (this modifies the ROIs in-place when draw=True)
        # We IGNORE the return value (which is just the numeric landmark array, not an image!)
        _ = front_pose_detector.detect(front_roi, draw=True)
        _ = mirror_pose_detector.detect(mirror_roi, draw=True)

        # 3. Display windows safely using the original ROIs, now containing the skeleton overlays
        cv2.imshow("AURA - Front View", front_roi)
        cv2.imshow("AURA - Mirror View (Flipped)", mirror_roi)

        # Wait according to calculated delay to maintain natural playback speed, exit on 'q'
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()