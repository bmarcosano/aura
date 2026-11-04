"""
Tests for the exact metric 3D reconstruction module (StereoReconstructor).
Replaces the old tests that checked the relative PoseReconstructor.
"""

import numpy as np
import pytest

from aura.reconstruction import StereoReconstructor


@pytest.fixture
def dummy_camera_matrix():
    """Mock intrinsic camera matrix for a 1920x1080 camera."""
    return np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


@pytest.fixture
def dummy_reflection_matrix():
    """Mock affine reflection matrix (mirror placed in front of the camera)."""
    M = np.eye(4, dtype=np.float32)
    M[2, 2] = -1.0  # Invert Z-axis
    M[2, 3] = 4.0  # Translation offset
    return M


def test_stereo_reconstructor_init(dummy_camera_matrix):
    """Test initialization and projection matrix P1 construction."""
    reconstructor = StereoReconstructor(dummy_camera_matrix)

    assert reconstructor.camera_matrix is not None
    assert reconstructor.P1.shape == (3, 4)
    np.testing.assert_array_equal(reconstructor.P1[:, :3], dummy_camera_matrix)


def test_triangulate_landmarks(dummy_camera_matrix, dummy_reflection_matrix):
    """Test the current Nx3 return contract of the metric triangulator."""
    reconstructor = StereoReconstructor(dummy_camera_matrix)

    num_landmarks = 33
    rng = np.random.default_rng(17)
    points_2d_direct = rng.random((num_landmarks, 2), dtype=np.float32) * 1000
    points_2d_reflected = rng.random((num_landmarks, 2), dtype=np.float32) * 1000

    # The current public API returns coordinates only; per-point reprojection
    # masks are a planned improvement to the calibrated reconstruction path.
    landmarks_3d = reconstructor.triangulate_landmarks(
        points_2d_direct, points_2d_reflected, dummy_reflection_matrix
    )

    assert landmarks_3d.shape == (num_landmarks, 3)
    assert np.isfinite(landmarks_3d).all(), "Triangulated points must be finite"
