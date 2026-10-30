import numpy as np

from aura.anthropometry import AnthropometricValidator


def test_no_valid_bones_does_not_pass():
    points = np.full((33, 3), np.nan)
    valid, error, _, _ = AnthropometricValidator().validate_isometry(points, points)
    assert valid is False
    assert np.isinf(error)


def test_affine_reflection_applies_translation():
    points = np.array([[0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[2, 2] = -1.0
    transform[2, 3] = 4.0
    reflected = AnthropometricValidator().apply_mirror_reflection(points, transform)
    assert np.allclose(reflected, [[0.0, 0.0, 3.0]])
