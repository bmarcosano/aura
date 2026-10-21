"""
calibrate.py: Offline camera calibration script using a checkerboard pattern.
Computes intrinsic camera matrix and distortion coefficients, saving them to disk.
"""

from __future__ import annotations

import cv2
import numpy as np


def calibrate_webcam(
    source: int | str = 0,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    output_filename: str = "camera_calibration.npz",
) -> None:
    """
    Captures frames from the video source, detects checkerboard corners,
    and performs camera calibration using OpenCV.

    Args:
        source: Webcam index (e.g., 0) or video file path.
        board_size: Inner corners of the chessboard (columns, rows), e.g., (9, 6).
        square_size: Physical size of a square side in meters (e.g., 0.025m = 2.5cm).
        output_filename: Path where calibration arrays will be saved.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"❌ Error: Could not open video source '{source}'")

    # Prepare 3D object points for the chessboard (e.g., (0,0,0), (1,0,0), (2,0,0) ... scaled by square_size)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size
    )

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    print("📸 --- WEBCAM CALIBRATION UTILITY ---")
    print(f"Target pattern: {board_size[0]}x{board_size[1]} inner corners.")
    print("Instructions:")
    print(" - Show the chessboard clearly to the camera.")
    print(" - Press 'SPACEBAR' to capture a frame (aim for 15-25 diverse frames).")
    print(" - Press 'ESC' or 'Q' when finished to compute calibration.\n")

    captured_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("❌ Failed to read frame from source.")
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            # Draw detected corners on live preview
            cv2.drawChessboardCorners(display_frame, board_size, corners, ret)
            cv2.putText(
                display_frame,
                f"Pattern Detected! Press SPACE to capture [{captured_count}]",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                display_frame,
                "Looking for chessboard pattern...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("AURA - Camera Calibration", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            print("⏹️ Calibration capture aborted by user.")
            break
        elif key == ord(" ") and ret:
            # Refine corner locations for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            objpoints.append(objp)
            imgpoints.append(refined_corners)
            captured_count += 1
            print(f"✅ Frame {captured_count} captured successfully.")

    cap.release()
    cv2.destroyAllWindows()

    if captured_count < 10:
        print(
            f"⚠️ Warning: Only {captured_count} frames captured. Recommended is at least 15-20 for accurate calibration."
        )

    if captured_count > 0:
        print(
            "\n⚙️ Computing camera intrinsic matrix and distortion coefficients via cv2.calibrateCamera..."
        )
        gray_shape = gray.shape[::-1]  # (width, height)

        ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray_shape, None, None
        )

        if ret:
            print("✅ Calibration successful!")
            print(f"Camera Matrix K:\n{camera_matrix}")
            print(f"Distortion Coefficients:\n{dist_coeffs.ravel()}")

            # Save calibration parameters to disk
            np.savez(
                output_filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
            )
            print(f"💾 Parameters successfully saved to '{output_filename}'.")
        else:
            print("❌ Camera calibration failed mathematically.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AURA Camera Calibration Utility")
    parser.add_argument("--source", default=0, help="Webcam device index or video path")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner columns")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner rows")
    parser.add_argument(
        "--size", type=float, default=0.025, help="Square size in meters"
    )
    parser.add_argument(
        "--output", default="camera_calibration.npz", help="Output .npz file path"
    )
    args = parser.parse_args()

    calibrate_webcam(
        source=int(args.source) if str(args.source).isdigit() else args.source,
        board_size=(args.cols, args.rows),
        square_size=args.size,
        output_filename=args.output,
    )
