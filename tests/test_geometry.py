"""test_geometry.py: Unit tests for biomechanical angle calculations."""

import numpy as np
from aura.geometry import calculate_angle


def test_calculate_right_angle():
    """Tests a 90-degree angle (e.g., bent elbow or knee)."""
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])  # Vertex
    c = np.array([1.0, 0.0])

    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 90.0, atol=1e-5)


def test_calculate_straight_angle():
    """Tests a 180-degree angle (e.g., fully extended leg or arm)."""
    a = np.array([-1.0, 0.0])
    b = np.array([0.0, 0.0])  # Vertex
    c = np.array([1.0, 0.0])

    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 180.0, atol=1e-5)


def test_calculate_zero_angle():
    """Tests a 0-degree angle (overlapping vectors)."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])  # Vertex
    c = np.array([1.0, 0.0])

    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 0.0, atol=1e-5)
