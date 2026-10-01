"""run_webcam.py: Real-time or video file posture analysis script using OpenCV, PoseDetector, and PostureAnalyzer."""

import argparse
import cv2
import numpy as np

from aura.analyzer import PostureAnalyzer
from aura.detector import PoseDetector


def main():
    parser = argparse.ArgumentParser(description="AURA Posture Analysis Runner")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g., '0') or path to a local video file (e.g., 'assets/test_posture.mp4')",
    )
    args = parser.parse_args()

    # Determine if source is a webcam index or a video file path
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{args.source}'.")
        return

    detector = PoseDetector(model_complexity=1, min_detection_confidence=0.5)
    analyzer = PostureAnalyzer()

    print(f"🚀 AURA Posture Monitor started using source: {args.source}. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ End of video stream or failed to read frame.")
            break

        # Flip horizontally only if using a webcam (source 0), optional for pre-recorded videos
        if str(source) == "0":
            frame = cv2.flip(frame, 1)
        
        # 1. Extract landmarks using MediaPipe
        landmarks = detector.detect(frame)

        # 2. Analyze posture metrics
        shoulder_metrics = {"tilt_angle_degrees": 0.0, "status": "Unknown"}
        hip_metrics = {"tilt_angle_degrees": 0.0, "status": "Unknown"}
        head_metrics = {"forward_offset_pixels": 0.0, "status": "Unknown"}

        if landmarks is not None and len(landmarks) > 24:
            shoulder_metrics = analyzer.analyze_shoulders(landmarks)
            hip_metrics = analyzer.analyze_hips(landmarks)
            head_metrics = analyzer.analyze_head_posture(landmarks)

        # 3. Draw HUD overlay on screen
        cv2.putText(frame, "AURA Posture Monitoring", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Dynamic color based on shoulder status
        s_status = shoulder_metrics["status"]
        s_color = (0, 255, 0) if s_status == "Optimal" else (0, 165, 255) if s_status == "Warning" else (0, 0, 255)
        
        cv2.putText(frame, f"Shoulders Tilt: {shoulder_metrics['tilt_angle_degrees']:.1f} deg [{s_status}]", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Hips Tilt: {hip_metrics['tilt_angle_degrees']:.1f} deg [{hip_metrics['status']}]", 
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Head Offset: {head_metrics['forward_offset_pixels']:.1f} px [{head_metrics['status']}]", 
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        # Display the video frame
        cv2.imshow("AURA - Posture Analyzer", frame)

        # Control playback speed slightly for video files (e.g., 30ms delay per frame), or keep for webcam
        wait_time = 30 if not str(source).isdigit() else 1
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("✨ Session ended successfully.")


if __name__ == "__main__":
    main()