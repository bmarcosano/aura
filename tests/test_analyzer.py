"""test_analyzer.py: Unit tests for the PostureAnalyzer metrics engine."""

import numpy as np
import pytest

from aura.analyzer import PostureAnalyzer


def test_analyzer_initialization():
    """Tests that PostureAnalyzer initializes with default thresholds."""
    analyzer = PostureAnalyzer()
    assert analyzer is not None


def test_shoulder_tilt_perfect_alignment():
    """Tests shoulder tilt calculation when shoulders are perfectly horizontal."""
    analyzer = PostureAnalyzer()
    
    # Mocking landmark array: shape (33, 4) -> [x, y, z, visibility]
    # MediaPipe pose indices: Left Shoulder = 11, Right Shoulder = 12
    landmarks = np.zeros((33, 4), dtype=np.float32)
    
    # Set left and right shoulders at the exact same Y height (Y = 100.0)
    landmarks[11] = [200.0, 100.0, 0.0, 0.99]  # Left shoulder
    landmarks[12] = [400.0, 100.0, 0.0, 0.99]  # Right shoulder
    
    metrics = analyzer.analyze_shoulders(landmarks)
    
    assert metrics["tilt_angle_degrees"] == pytest.approx(0.0, abs=1e-2)
    assert metrics["status"] == "Optimal"


def test_shoulder_tilt_asymmetrical():
    """Tests shoulder tilt calculation when one shoulder is visibly higher."""
    analyzer = PostureAnalyzer()
    
    landmarks = np.zeros((33, 4), dtype=np.float32)
    # Left shoulder higher (Y = 90), right shoulder lower (Y = 110) over a distance of 200px
    landmarks[11] = [200.0, 90.0, 0.0, 0.99]   # Left shoulder
    landmarks[12] = [400.0, 110.0, 0.0, 0.99]  # Right shoulder
    
    metrics = analyzer.analyze_shoulders(landmarks)
    
    # Should detect a non-zero tilt angle
    assert metrics["tilt_angle_degrees"] > 0.0
    assert metrics["status"] in ["Warning", "Poor"]