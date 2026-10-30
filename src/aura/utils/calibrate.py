"""Interactive checkerboard camera calibration with quality metadata."""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def calibrate_webcam(
    source: int | str = 0,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    output_filename: str = "camera_calibration.npz",
) -> dict | None:
    """Capture diverse checkerboard views and save a metric intrinsic calibration.

    ``square_size`` is the measured side of one checkerboard square in metres.
    The saved ``image_size`` is (width, height); AURA checks it before using K.
    The function returns a metadata dictionary on success and None on failure.
    """
    if board_size[0] < 3 or board_size[1] < 3 or square_size <= 0:
        raise ValueError("Invalid checkerboard geometry")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    object_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_template[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    view_signatures: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

    print(
        "AURA camera calibration: move the checkerboard across the image and tilt it."
    )
    print(
        "SPACE captures a view; Q/ESC computes calibration; aim for 15-25 diverse views."
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(
                gray,
                board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, board_size, corners, found)
                cv2.putText(
                    preview,
                    f"Detected - SPACE to capture ({len(image_points)})",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    preview,
                    "Looking for checkerboard...",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("AURA - Camera Calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" ") and found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # A compact signature prevents collecting virtually identical poses.
                points = refined.reshape(-1, 2)
                signature = np.array(
                    [
                        points[:, 0].mean() / gray.shape[1],
                        points[:, 1].mean() / gray.shape[0],
                        np.ptp(points[:, 0]) / gray.shape[1],
                        np.ptp(points[:, 1]) / gray.shape[0],
                    ]
                )
                if (
                    view_signatures
                    and min(np.linalg.norm(signature - old) for old in view_signatures)
                    < 0.025
                ):
                    print(
                        "View too similar to a previous capture; move/tilt the board."
                    )
                    continue
                object_points.append(object_template.copy())
                image_points.append(refined)
                view_signatures.append(signature)
                print(f"Captured diverse view {len(image_points)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(image_points) < 10 or image_size is None:
        print(f"Calibration rejected: {len(image_points)} diverse views (minimum 10).")
        return None

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    if (
        not np.isfinite(rms)
        or not np.isfinite(camera_matrix).all()
        or not np.isfinite(dist_coeffs).all()
    ):
        print("Calibration rejected: non-finite solution.")
        return None

    # Calculate per-view reprojection RMS for diagnostics.
    per_view_errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        per_view_errors.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected.reshape(-1, 2) - observed.reshape(-1, 2)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
        )

    output_dir = os.path.dirname(os.path.abspath(output_filename))
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        output_filename,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.asarray(image_size, dtype=np.int32),
        rms=float(rms),
        per_view_errors=np.asarray(per_view_errors, dtype=np.float32),
        board_size=np.asarray(board_size, dtype=np.int32),
        square_size_m=float(square_size),
        calibration_version=2,
    )
    print(f"Calibration saved to {output_filename}; RMS={rms:.3f}px")
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "rms": float(rms),
        "per_view_errors": per_view_errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the AURA camera")
    parser.add_argument("--source", default="0")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument(
        "--size", type=float, default=0.025, help="Checker square side in metres"
    )
    parser.add_argument("--output", default="camera_calibration.npz")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    calibrate_webcam(source, (args.cols, args.rows), args.size, args.output)
