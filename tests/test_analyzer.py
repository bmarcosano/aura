"""
test_analyzer.py: Unit tests for PostureAnalyzer module.
"""

import math

import numpy as np
import pytest

from aura.analyzer import PostureAnalyzer


def test_analyzer_initialization():
    """Tests default threshold values initialization."""
    # Arrange & Act: Initialize the analyzer with default parameters
    analyzer = PostureAnalyzer()

    # Assert: Verify that default thresholds match expected values
    assert analyzer.shoulder_threshold_warning == 3.0
    assert analyzer.head_fwd_warning == 15.0


@pytest.mark.parametrize(
    "landmarks,expected_status",
    [
        (None, "Unknown"),
        (np.zeros((5, 4), dtype=np.float32), "Unknown"),  # Too short landmark array
    ],
)
def test_analyze_shoulders_invalid_input(landmarks, expected_status):
    """Tests shoulder analysis with invalid or missing landmarks."""
    # Arrange
    analyzer = PostureAnalyzer()

    # Act
    result = analyzer.analyze_shoulders(landmarks)

    # Assert
    assert math.isnan(result["tilt_angle_degrees"])
    assert result["status"] == expected_status


def test_analyze_shoulders_optimal():
    """Tests optimal horizontal alignment for shoulders."""
    # Arrange
    analyzer = PostureAnalyzer()
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [100.0, 200.0, 0.0, 0.99]  # Left shoulder
    landmarks[12] = [200.0, 200.0, 0.0, 0.99]  # Right shoulder (same Y -> 0 deg tilt)

    # Act
    result = analyzer.analyze_shoulders(landmarks)

    # Assert
    assert result["status"] == "Optimal"
    assert result["tilt_angle_degrees"] == 0.0


def test_analyze_shoulders_warning():
    """Tests shoulder alignment warning status under slight tilt conditions."""
    # Arrange: Set a low warning threshold (2.0°) and high poor threshold (10.0°)
    analyzer = PostureAnalyzer(
        shoulder_threshold_warning=2.0, shoulder_threshold_poor=10.0
    )
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [100.0, 100.0, 0.0, 0.99]
    landmarks[12] = [200.0, 108.0, 0.0, 0.99]  # Slight tilt (approx 4.56°)

    # Act
    result = analyzer.analyze_shoulders(landmarks)

    # Assert
    assert result["status"] == "Warning"


def test_analyze_shoulders_poor():
    """Tests poor alignment (high tilt) for shoulders."""
    # Arrange: Set thresholds to isolate the Poor status branch
    analyzer = PostureAnalyzer(
        shoulder_threshold_warning=2.0, shoulder_threshold_poor=5.0
    )
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [100.0, 100.0, 0.0, 0.99]
    landmarks[12] = [200.0, 130.0, 0.0, 0.99]  # High vertical offset -> high tilt

    # Act
    result = analyzer.analyze_shoulders(landmarks)

    # Assert
    assert result["status"] == "Poor"
    assert result["tilt_angle_degrees"] >= 5.0


@pytest.mark.parametrize(
    "landmarks,expected_status",
    [
        (None, "Unknown"),
        (np.zeros((20, 4), dtype=np.float32), "Unknown"),  # Too short array for hips
    ],
)
def test_analyze_hips_invalid_input(landmarks, expected_status):
    """Tests hip analysis with invalid landmarks."""
    # Arrange
    analyzer = PostureAnalyzer()

    # Act
    result = analyzer.analyze_hips(landmarks)

    # Assert
    assert result["status"] == expected_status


def test_analyze_hips_warning():
    """Tests hip alignment warning status under slight tilt conditions."""
    # Arrange
    analyzer = PostureAnalyzer(hip_threshold_warning=2.0, hip_threshold_poor=10.0)
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[23] = [100.0, 100.0, 0.0, 0.99]  # Left hip
    landmarks[24] = [200.0, 108.0, 0.0, 0.99]  # Slight hip tilt

    # Act
    result = analyzer.analyze_hips(landmarks)

    # Assert
    assert result["status"] == "Warning"


def test_analyze_hips_poor():
    """Tests poor hip alignment status under high tilt conditions."""
    # Arrange
    analyzer = PostureAnalyzer(hip_threshold_warning=2.0, hip_threshold_poor=5.0)
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[23] = [100.0, 100.0, 0.0, 0.99]  # Left hip
    landmarks[24] = [200.0, 130.0, 0.0, 0.99]  # Right hip (high tilt)

    # Act
    result = analyzer.analyze_hips(landmarks)

    # Assert
    assert result["status"] == "Poor"


def test_analyze_head_frontal_invalid_input():
    """Tests frontal head posture with invalid input."""
    # Arrange
    analyzer = PostureAnalyzer()

    # Act & Assert
    result = analyzer.analyze_head_posture_frontal(None)
    assert result["status"] == "Unknown"
    assert math.isnan(result["lateral_shift_ratio"])

    result = analyzer.analyze_head_posture_frontal(np.zeros((5, 4)))
    assert result["status"] == "Unknown"
    assert math.isnan(result["lateral_shift_ratio"])


def test_analyze_head_frontal_zero_width():
    """Tests protection against zero division when shoulder width is negligible."""
    # Arrange
    analyzer = PostureAnalyzer()
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [200.0, 100.0, 0.0, 0.99]
    landmarks[12] = [200.0, 100.0, 0.0, 0.99]  # Same coordinates -> zero width

    # Act
    result = analyzer.analyze_head_posture_frontal(landmarks)

    # Assert
    assert result["status"] == "Unknown"
    assert math.isnan(result["lateral_shift_ratio"])


def test_analyze_head_frontal_optimal():
    """Tests perfectly centered head position."""
    # Arrange
    analyzer = PostureAnalyzer()
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [100.0, 200.0, 0.0, 0.99]
    landmarks[12] = [300.0, 200.0, 0.0, 0.99]  # Midpoint is 200
    landmarks[0] = [200.0, 100.0, 0.0, 0.99]  # Nose at 150 (zero offset)

    # Act
    result = analyzer.analyze_head_posture_frontal(landmarks)

    # Assert
    assert result["status"] == "Optimal"
    assert result["lateral_shift_ratio"] == 0.0


def test_analyze_head_frontal_warning():
    """Tests frontal head posture warning status under moderate lateral shift conditions."""
    # Arrange
    analyzer = PostureAnalyzer()
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [100.0, 200.0, 0.0, 0.99]
    landmarks[12] = [300.0, 200.0, 0.0, 0.99]  # Shoulder width = 200 (midpoint = 200)
    landmarks[0] = [
        220.0,
        150.0,
        0.0,
        0.99,
    ]  # Nose offset = 20 (ratio = 0.10 >= 0.08 warning threshold)

    # Act
    result = analyzer.analyze_head_posture_frontal(landmarks)

    # Assert
    assert result["status"] == "Warning"


def test_analyze_head_frontal_poor():
    """Tests frontal head posture poor status under severe lateral shift conditions."""
    # Arrange
    analyzer = PostureAnalyzer()
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[11] = [100.0, 100.0, 0.0, 0.99]
    landmarks[12] = [300.0, 100.0, 0.0, 0.99]  # Shoulder width = 200 (midpoint = 200)
    landmarks[0] = [
        240.0,
        50.0,
        0.0,
        0.99,
    ]  # Nose offset = 40 (ratio = 0.20 >= 0.15 poor threshold)

    # Act
    result = analyzer.analyze_head_posture_frontal(landmarks)

    # Assert
    assert result["status"] == "Poor"


def test_analyze_head_sagittal():
    """Tests sagittal (profile) head forward posture calculation with valid inputs."""
    # Arrange
    analyzer = PostureAnalyzer()
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[12] = [200.0, 100.0, 0.0, 0.99]  # Right shoulder
    landmarks[8] = [220.0, 80.0, 0.0, 0.99]  # Right ear forward

    # Act
    result_right = analyzer.analyze_head_posture_sagittal(
        landmarks, is_facing_right=True
    )
    result_left = analyzer.analyze_head_posture_sagittal(
        landmarks, is_facing_right=False
    )

    # Assert
    assert "forward_angle_degrees" in result_right
    assert "forward_angle_degrees" in result_left
    assert result_right["status"] in ["Optimal", "Warning", "Poor"]


def test_analyze_head_sagittal_invalid_input():
    """Tests sagittal head posture analysis with invalid or missing landmarks."""
    # Arrange
    analyzer = PostureAnalyzer()

    # Act & Assert: Verify handling of None and short arrays
    result = analyzer.analyze_head_posture_sagittal(None)
    assert result["status"] == "Unknown"
    assert math.isnan(result["forward_angle_degrees"])
    result = analyzer.analyze_head_posture_sagittal(np.zeros((5, 4)))
    assert result["status"] == "Unknown"
    assert math.isnan(result["forward_angle_degrees"])


def test_analyze_head_sagittal_warning_strict():
    """Tests exact Warning status trigger for sagittal head posture."""
    # Arrange: Set a moderate warning threshold so we can hit the Warning branch precisely
    analyzer = PostureAnalyzer(head_fwd_warning=25.0)

    # Configure coordinates to yield an angle between 25° and 50° (Warning range)
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[12] = [200.0, 150.0, 0.0, 0.99]  # Shoulder
    landmarks[8] = [
        220.0,
        120.0,
        0.0,
        0.99,
    ]  # Ear offset yielding a moderate forward angle

    # Act
    result = analyzer.analyze_head_posture_sagittal(landmarks, is_facing_right=True)

    # Assert
    assert result["status"] == "Warning"


def test_analyze_head_sagittal_warning_and_poor():
    """Tests sagittal head posture warning and poor status thresholds using trigonometric ratios."""
    # Arrange: Configure analyzer with a low warning threshold (e.g., 5.0°)
    analyzer = PostureAnalyzer(head_fwd_warning=5.0)

    # 1. Warning case: moderate horizontal offset relative to vertical distance
    landmarks_warn = np.zeros((33, 4), dtype=np.float32)
    landmarks_warn[12] = [200.0, 150.0, 0.0, 0.99]  # Shoulder
    landmarks_warn[8] = [
        220.0,
        120.0,
        0.0,
        0.99,
    ]  # Ear forward (triggers Warning range)

    # 2. Poor case: severe horizontal offset (>= head_fwd_warning * 2)
    landmarks_poor = np.zeros((33, 4), dtype=np.float32)
    landmarks_poor[12] = [200.0, 150.0, 0.0, 0.99]  # Shoulder
    landmarks_poor[8] = [
        260.0,
        110.0,
        0.0,
        0.99,
    ]  # Ear heavily forward (triggers Poor range)

    # Act
    res_warn = analyzer.analyze_head_posture_sagittal(
        landmarks_warn, is_facing_right=True
    )
    res_poor = analyzer.analyze_head_posture_sagittal(
        landmarks_poor, is_facing_right=True
    )

    # Assert
    assert res_warn["status"] in ["Warning", "Poor"]
    assert res_poor["status"] == "Poor"
