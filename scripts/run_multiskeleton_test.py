"""Script to test the MultiSkeletonDetector pipeline with Auto Calibration."""

from __future__ import annotations

import argparse
import sys
import cv2

from aura.detector import PoseDetector
from aura.multi_detector import MultiSkeletonDetector
from aura.calibrator import AutoSplitCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test MultiSkeletonDetector with Auto Calibration on a video or webcam."
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
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{source}'")
        sys.exit(1)

    # Retrieve video FPS to synchronize playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    # Initialize the auto-calibrator and pose detectors
    calibrator = AutoSplitCalibrator()
    front_pose_detector = PoseDetector()
    mirror_pose_detector = PoseDetector()
    
    # This will remain None until calibration is successfully completed
    multi_detector: MultiSkeletonDetector | None = None

    print(f"🚀 Starting test using source: {source} (FPS: {fps if fps > 0 else 'N/A'}).")
    print("⏳ CALIBRATION PHASE: Analysing motion to find the optimal mirror split line...")
    print("Press 'q' at any time to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        
        if not success or frame is None or frame.size == 0:
            print("End of video stream or empty frame received.")
            break

        # --- PHASE 1: AUTO-CALIBRATION ---
        if multi_detector is None:
            # Update calibrator with the current frame
            # The calibrator will draw feedback lines directly on this frame
            split_ratio = calibrator.update(frame)
            
            if split_ratio is not None:
                # Calibration complete! Initialize the MultiSkeletonDetector with the calculated ratio
                print(f"\n✅ Calibration complete! Optimal split ratio locked at: {split_ratio:.3f}")
                multi_detector = MultiSkeletonDetector(split_ratio=split_ratio)
                
                # Close the temporary calibration window
                cv2.destroyWindow("AURA - Calibration")
            else:
                # Still calibrating, show the full frame with green feedback lines
                cv2.imshow("AURA - Calibration", frame)
                
        # --- PHASE 2: DETECTION AND ROI CROPPING ---
        else:
            # 1. Crop the frame into Front View and Mirror View ROIs using the calibrated ratio
            front_roi, mirror_roi = multi_detector._crop_rois(frame)

            if front_roi.size == 0 or mirror_roi.size == 0 or front_roi.shape[0] == 0 or mirror_roi.shape[0] == 0:
                continue

            # 2. Run MediaPipe pose detection (modifies ROIs in-place)
            _ = front_pose_detector.detect(front_roi, draw=True)
            _ = mirror_pose_detector.detect(mirror_roi, draw=True)

            # 3. Display the separated and annotated views
            cv2.imshow("AURA - Front View", front_roi)
            cv2.imshow("AURA - Mirror View (Flipped)", mirror_roi)

        # Wait according to calculated delay, exit on 'q'
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()