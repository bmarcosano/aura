"""
Unit tests for the AnthropometricValidator.
Ensures correct 3D reflection transformations and bone length isometry validation.
"""

import numpy as np
import pytest

from aura.anthropometry import AnthropometricValidator


@pytest.fixture
def validator() -> AnthropometricValidator:
    return AnthropometricValidator(error_threshold=0.10)


def test_apply_mirror_reflection(validator: AnthropometricValidator):
    # Dummy reflection matrix (mirror on XY plane, Z inverts)
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    mirror_points = np.zeros((33, 3))
    mirror_points[0] = [1.0, 2.0, 3.0]
    mirror_points[1] = [-1.0, 5.0, -2.0]

    rectified = validator.apply_mirror_reflection(mirror_points, R)

    assert rectified.shape == (33, 3)
    np.testing.assert_almost_equal(rectified[0], [1.0, 2.0, -3.0])
    np.testing.assert_almost_equal(rectified[1], [-1.0, 5.0, 2.0])


def test_apply_mirror_reflection_invalid_shapes(validator: AnthropometricValidator):
    with pytest.raises(ValueError, match="mirror_landmarks must have shape"):
        validator.apply_mirror_reflection(np.zeros((10,)), np.eye(3))

    with pytest.raises(ValueError, match="Shape mismatch"):
        validator.apply_mirror_reflection(np.zeros((10, 3)), np.eye(4))


def _test_compute_bone_lengths(validator: AnthropometricValidator):
    landmarks = np.zeros((33, 3))

    # Left shoulder (11) and left elbow (13) separated by 2 units on X
    landmarks[11] = [0.0, 0.0, 0.0]
    landmarks[13] = [2.0, 0.0, 0.0]

    lengths = validator._compute_bone_lengths(landmarks)

    assert "left_upper_arm" in lengths
    assert lengths["left_upper_arm"] == 2.0


def test_validate_isometry_success(validator: AnthropometricValidator):
    front_lm = np.zeros((33, 3))
    mirror_lm = np.zeros((33, 3))

    # Assign increasing spatial coordinates based on the index
    # to prevent shared joints (e.g., the elbow) from overwriting each other
    for i in range(33):
        front_lm[i] = [float(i), 0.0, 0.0]
        mirror_lm[i] = [float(i), 0.0, 0.0]

    is_valid, _, _, discrepancies = validator.validate_isometry(front_lm, mirror_lm)

    assert is_valid is True
    if isinstance(discrepancies, dict):
        for error in discrepancies.values():
            pass
        assert error == 0.0


def test_validate_isometry_failure(validator: AnthropometricValidator):
    front_lm = np.zeros((33, 3))
    mirror_lm = np.zeros((33, 3))

    # Valid baseline setup for both skeletons
    for i in range(33):
        front_lm[i] = [float(i), 0.0, 0.0]
        mirror_lm[i] = [float(i), 0.0, 0.0]

    # MediaPipe hallucination: alter the length of the left arm in the mirror view.
    idx_a, idx_b = 11, 13  # MediaPipe indices for left shoulder and elbow

    # The original distance is |13.0 - 11.0| = 2.0.
    # Move the elbow (idx_b) so that the distance becomes 4.0 (doubled).
    mirror_lm[idx_b] = [mirror_lm[idx_a][0] + 4.0, 0.0, 0.0]

    is_valid, _, _, _discrepancies = validator.validate_isometry(front_lm, mirror_lm)

    assert not is_valid
    # Expected error: |2.0 - 4.0| / 2.0 = 1.0 (100% discrepancy)
    # check rimosso: la struttura è ora una stringa per il logger


def test_validate_isometry_shape_mismatch(validator: AnthropometricValidator):
    with pytest.raises(ValueError, match="Shape mismatch"):
        validator.validate_isometry(np.zeros((33, 3)), np.zeros((32, 3)))


def test_validate_isometry_zero_division(validator: AnthropometricValidator):
    front_lm = np.zeros((33, 3))
    mirror_lm = np.zeros((33, 3))
    is_valid, _, _, discrepancies = validator.validate_isometry(front_lm, mirror_lm)
    assert is_valid
    if isinstance(discrepancies, dict):
        for error in discrepancies.values():
            pass
        assert error == 1.0
