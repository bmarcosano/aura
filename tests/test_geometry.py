"""
test_geometry.py: Unit tests for biomechanical angle calculations.
"""

import numpy as np

from aura.geometry import calculate_angle


def test_calculate_right_angle():
    """Tests a 90-degree angle (e.g., bent elbow or knee)."""
    # Arrange
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])  # Vertex
    c = np.array([1.0, 0.0])

    # Act
    angle = calculate_angle(a, b, c)

    # Assert
    assert np.isclose(angle, 90.0, atol=1e-5)


def test_calculate_straight_angle():
    """Tests a 180-degree angle (e.g., fully extended leg or arm)."""
    # Arrange
    a = np.array([-1.0, 0.0])
    b = np.array([0.0, 0.0])  # Vertex
    c = np.array([1.0, 0.0])

    # Act
    angle = calculate_angle(a, b, c)

    # Assert
    assert np.isclose(angle, 180.0, atol=1e-5)


def test_calculate_zero_angle():
    """Tests a 0-degree angle (overlapping vectors)."""
    # Arrange
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])  # Vertex
    c = np.array([1.0, 0.0])

    # Act
    angle = calculate_angle(a, b, c)

    # Assert
    assert np.isclose(angle, 0.0, atol=1e-5)


def test_calculate_angle_zero_norm_protection():
    """Tests safety fallback against zero division when points coincide with vertex."""
    # Arrange: Points a and b are identical, resulting in zero magnitude vector
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])

    # Act
    angle = calculate_angle(a, b, c)

    # Assert: Should safely catch the condition and return 0.0 (covering line 26)
    assert angle == 0.0
