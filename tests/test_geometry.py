import numpy as np

from aura.geometry import calculate_angle, normalize_vector


def test_calculate_right_angle():
    assert calculate_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) == 90.0


def test_degenerate_angle_is_nan():
    assert np.isnan(calculate_angle([0, 0], [0, 0], [1, 1]))


def test_normalize_vector():
    value = normalize_vector(np.array([3.0, 4.0, 0.0]))
    assert np.allclose(value, [0.6, 0.8, 0.0])
    assert normalize_vector(np.zeros(3)) is None
