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
    # Simple reflection across a plane parallel to the camera sensor
    M = np.eye(4, dtype=np.float32)
    M[2, 2] = -1.0  # Invert Z-axis
    M[2, 3] = 4.0  # Translation offset
    return M


def test_stereo_reconstructor_init(dummy_camera_matrix):
    """Test initialization and projection matrix P1 construction."""
    reconstructor = StereoReconstructor(dummy_camera_matrix)

    assert reconstructor.camera_matrix is not None
    assert reconstructor.P1.shape == (3, 4)
    # The left 3x3 block of P1 should be equal to the camera matrix
    np.testing.assert_array_equal(reconstructor.P1[:, :3], dummy_camera_matrix)


def test_triangulate_landmarks(dummy_camera_matrix, dummy_reflection_matrix):
    """Test the triangulation logic with mock 2D pixel data."""
    reconstructor = StereoReconstructor(dummy_camera_matrix)

    num_landmarks = 33

    # Generate mock 2D pixel coordinates for direct and reflected views
    points_2d_direct = np.random.rand(num_landmarks, 2).astype(np.float32) * 1000
    points_2d_reflected = np.random.rand(num_landmarks, 2).astype(np.float32) * 1000

    # Execute triangulation
    landmarks_3d = reconstructor.triangulate_landmarks(
        points_2d_direct, points_2d_reflected, dummy_reflection_matrix
    )

    # Verify the output shape and type
    assert landmarks_3d.shape == (num_landmarks, 3)
    assert not np.isnan(landmarks_3d).any(), "Triangulated points contain NaN values"
    assert not np.isinf(landmarks_3d).any(), (
        "Triangulated points contain Infinite values"
    )
