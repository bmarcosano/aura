import numpy as np

from aura.reconstruction import StereoReconstructor


def project(K, points):
    projected = (K @ points.T).T
    return projected[:, :2] / projected[:, 2:3]


def test_triangulation_recovers_synthetic_points():
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    reconstructor = StereoReconstructor(K)
    # Plane z=2m: reflected camera mapping is z' = 4-z.
    transform = np.eye(4)
    transform[2, 2] = -1.0
    transform[2, 3] = 4.0
    points = np.array(
        [
            [-0.35, -0.20, 0.80],
            [0.20, 0.10, 1.00],
            [0.45, -0.15, 1.25],
            [-0.10, 0.25, 1.45],
            [0.15, 0.32, 1.60],
        ]
    )
    direct = project(K, points)
    virtual_points = (transform @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    reflected = project(K, virtual_points)
    recovered = reconstructor.triangulate_landmarks(
        direct, reflected, transform, max_error_px=1.0
    )
    assert reconstructor.last_valid_mask.all()
    assert np.allclose(recovered, points, atol=1e-5)


def test_reprojection_outlier_is_rejected():
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    reconstructor = StereoReconstructor(K)
    transform = np.eye(4)
    transform[2, 2], transform[2, 3] = -1.0, 4.0
    points = np.array([[0.1, 0.0, 0.8], [-0.2, 0.1, 1.2]])
    direct = project(K, points)
    reflected_points = (transform @ np.c_[points, np.ones(2)].T).T[:, :3]
    reflected = project(K, reflected_points)
    reflected[1, 0] += 150.0
    recovered = reconstructor.triangulate_landmarks(
        direct, reflected, transform, max_error_px=3.0
    )
    assert reconstructor.last_valid_mask.tolist() == [True, False]
    assert np.isfinite(recovered[0]).all()
    assert np.isnan(recovered[1]).all()


def test_legacy_fusion_returns_array():
    reconstructor = StereoReconstructor(np.eye(3))
    front = np.ones((33, 3), dtype=float)
    result = reconstructor.fuse_landmarks(front)
    assert result.shape == (33, 3)
    assert np.allclose(result, front)


def test_invalid_nonreflection_matrix_is_rejected():
    reconstructor = StereoReconstructor(np.eye(3))
    with np.testing.assert_raises(ValueError):
        reconstructor.triangulate_landmarks(
            np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]]), np.eye(4)
        )
