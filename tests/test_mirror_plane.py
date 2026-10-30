import numpy as np

from aura.mirror_plane import MirrorPlaneEstimator


def reflect(points, normal, d):
    H = np.eye(3) - 2.0 * np.outer(normal, normal)
    t = -2.0 * d * normal
    return (H @ points.T).T + t


def test_svd_recovers_known_plane():
    rng = np.random.default_rng(7)
    direct = rng.uniform([-0.5, -0.5, 0.4], [0.5, 0.5, 1.6], size=(20, 3))
    normal = np.array([0.0, 0.0, 1.0])
    d = -2.0
    mirrored = reflect(direct, normal, d)
    estimator = MirrorPlaneEstimator(ema_alpha=1.0)
    transform, estimated_normal = estimator.estimate_plane(direct, mirrored)
    assert transform is not None
    assert abs(abs(estimated_normal[2]) - 1.0) < 1e-8
    assert np.allclose(transform[:3, :3], np.diag([1.0, 1.0, -1.0]), atol=1e-8)
    assert np.allclose(transform[:3, 3], [0.0, 0.0, 4.0], atol=1e-8)


def test_svd_rejects_collinear_midpoints():
    x = np.arange(5, dtype=float)
    direct = np.column_stack([x, np.zeros(5), np.ones(5)])
    reflected = direct.copy()
    estimator = MirrorPlaneEstimator()
    transform, normal = estimator.estimate_plane(direct, reflected)
    assert transform is None and normal is None


def test_metric_state_is_not_set_by_legacy_orientation_only_estimate():
    estimator = MirrorPlaneEstimator(history_size=1)
    estimator.add_frame_data(np.ones((4, 3)), np.zeros((4, 3)))
    estimator.estimate()
    assert estimator.is_calibrated_state() is False
