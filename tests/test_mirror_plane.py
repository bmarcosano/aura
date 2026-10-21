"""
Unit tests for MirrorPlaneEstimator SVD calculations and geometric transformations.
"""

import pytest
import numpy as np
from aura.mirror_plane import MirrorPlaneEstimator


def generate_synthetic_mirror_points(
    num_points: int, normal: np.ndarray, distance: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic paired 3D points reflected across a plane with given normal."""
    n = normal / np.linalg.norm(normal)
    R = np.eye(3) - 2.0 * np.outer(n, n)

    # Random direct points
    np.random.seed(42)
    front_pts = np.random.uniform(-1.0, 1.0, (num_points, 3))
    
    # Mirror points via Householder reflection (P_m = R * P_f + 2 * distance * n)
    # The displacement vector P_f - P_m will be parallel to n.
    mirror_pts = (R @ front_pts.T).T + 2.0 * distance * n
    return front_pts, mirror_pts


def test_estimator_initialization():
    estimator = MirrorPlaneEstimator(history_size=30)
    assert not estimator.is_estimated
    assert not estimator.is_calibrated_state()
    assert estimator.history_size == 30


def test_add_frame_data_shape_mismatch():
    estimator = MirrorPlaneEstimator()
    pts1 = np.zeros((10, 3))
    pts2 = np.zeros((8, 3))

    with pytest.raises(ValueError, match="Landmark shape mismatch"):
        estimator.add_frame_data(pts1, pts2)


def test_add_frame_data_accumulator():
    estimator = MirrorPlaneEstimator(history_size=5)
    pts_f = np.zeros((10, 3))
    pts_m = np.ones((10, 3))

    for i in range(4):
        ready = estimator.add_frame_data(pts_f, pts_m)
        assert not ready

    # 5th frame fills the history buffer
    ready = estimator.add_frame_data(pts_f, pts_m)
    assert ready


def test_estimation_empty_history_raises():
    estimator = MirrorPlaneEstimator()
    with pytest.raises(RuntimeError, match="history buffer is empty"):
        estimator.estimate()


def test_estimation_flat_mirror():
    estimator = MirrorPlaneEstimator(history_size=10)
    
    # Mirror perfectly parallel to camera (facing it)
    true_normal = np.array([0.0, 0.0, -1.0])
    pts_f, pts_m = generate_synthetic_mirror_points(num_points=33, normal=true_normal)

    for _ in range(10):
        estimator.add_frame_data(pts_f, pts_m)

    normal, yaw, pitch = estimator.estimate()

    assert estimator.is_estimated
    # Check normal vector points strictly along Z axis
    np.testing.assert_almost_equal(abs(normal[2]), 1.0, decimal=3)
    # Angles should be zero for a perfectly flat, non-tilted mirror
    assert abs(yaw) < 1.0
    assert abs(pitch) < 1.0


def test_estimation_yaw_tilted_mirror():
    estimator = MirrorPlaneEstimator(history_size=10)
    
    # Mirror rotated 15 degrees around Y axis (Yaw tilt)
    angle_rad = np.radians(15.0)
    # Sin(15) on X, Cos(15) on Z
    true_normal = np.array([np.sin(angle_rad), 0.0, -np.cos(angle_rad)])

    pts_f, pts_m = generate_synthetic_mirror_points(num_points=33, normal=true_normal)

    for _ in range(10):
        estimator.add_frame_data(pts_f, pts_m)

    _, yaw, pitch = estimator.estimate()

    # Yaw should be ~15 degrees, Pitch should be 0
    np.testing.assert_almost_equal(yaw, 15.0, decimal=1)
    np.testing.assert_almost_equal(pitch, 0.0, decimal=1)


def test_estimation_pitch_tilted_mirror():
    estimator = MirrorPlaneEstimator(history_size=10)
    
    # Mirror tilted backwards 10 degrees around X axis (Pitch tilt)
    angle_rad = np.radians(10.0)
    true_normal = np.array([0.0, np.sin(angle_rad), -np.cos(angle_rad)])

    pts_f, pts_m = generate_synthetic_mirror_points(num_points=33, normal=true_normal)

    for _ in range(10):
        estimator.add_frame_data(pts_f, pts_m)

    _, yaw, pitch = estimator.estimate()

    # Yaw should be 0, Pitch should be ~10 degrees
    np.testing.assert_almost_equal(yaw, 0.0, decimal=1)
    np.testing.assert_almost_equal(pitch, 10.0, decimal=1)


def test_reflection_matrix_uncalibrated_raises():
    estimator = MirrorPlaneEstimator()
    with pytest.raises(RuntimeError, match="Mirror plane has not been estimated yet"):
        estimator.get_reflection_matrix()


def test_reflection_matrix_properties():
    estimator = MirrorPlaneEstimator(history_size=5)
    
    # Arbitrary complex mirror orientation
    true_normal = np.array([0.2, 0.1, -0.97])
    true_normal /= np.linalg.norm(true_normal)

    pts_f, pts_m = generate_synthetic_mirror_points(num_points=20, normal=true_normal)

    for _ in range(5):
        estimator.add_frame_data(pts_f, pts_m)

    estimator.estimate()
    R = estimator.get_reflection_matrix()

    # A Householder reflection matrix must be orthogonal (R^T * R = I)
    np.testing.assert_almost_equal(R.T @ R, np.eye(3), decimal=5)
    
    # The determinant of a reflection matrix must be exactly -1
    np.testing.assert_almost_equal(np.linalg.det(R), -1.0, decimal=5)
