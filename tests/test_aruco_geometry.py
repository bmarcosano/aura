import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("opencv-contrib ArUco module is required", allow_module_level=True)

from aura.utils.aruco_tracker import ArucoMultiTracker


def test_marker_pose_creates_expected_affine_reflection():
    tracker = ArucoMultiTracker(marker_size_m=0.18)
    # Identity marker orientation: marker plane is z=2m in camera coordinates.
    plane = tracker.estimate_mirror_plane(
        np.zeros((3, 1)), np.array([[0.0], [0.0], [2.0]])
    )
    transform = plane["reflection_matrix"]
    assert transform.shape == (4, 4)
    assert np.allclose(transform[:3, :3], np.diag([1.0, 1.0, -1.0]))
    assert np.allclose(transform[:3, 3], [0.0, 0.0, 4.0])
    point = np.array([0.2, -0.3, 1.0, 1.0])
    reflected = transform @ point
    assert np.allclose(reflected[:3], [0.2, -0.3, 3.0])
