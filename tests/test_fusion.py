"""
Unit Tests for AURA Multi-View Sensor Fusion Engine (AuraFusionEngine)
Tests posture evaluation, alignment checks, and safety override logic.
"""

import pytest

from aura.fusion import AuraFusionEngine


class MockLandmark:
    """Mock MediaPipe landmark with x and y coordinates."""

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class MockPoseResults:
    """Mock MediaPipe pose detection results container."""

    def __init__(self, landmarks):
        self.pose_landmarks = type("obj", (object,), {"landmark": landmarks})


@pytest.fixture
def fusion_engine():
    """Fixture providing an instance of AuraFusionEngine."""
    return AuraFusionEngine()


def test_fusion_optimal(fusion_engine):
    """Test that balanced front symmetry and optimal mirror profile return OPTIMAL state."""
    # Mock balanced front view landmarks (shoulders and hips level)
    front_landmarks = [MockLandmark() for _ in range(33)]
    front_landmarks[11] = MockLandmark(y=0.5)  # Left shoulder
    front_landmarks[12] = MockLandmark(y=0.501)  # Right shoulder (negligible tilt)
    front_landmarks[23] = MockLandmark(y=0.8)  # Left hip
    front_landmarks[24] = MockLandmark(y=0.801)  # Right hip (negligible tilt)

    # Mock optimal mirror/profile view landmarks (low horizontal torso lean)
    mirror_landmarks = [MockLandmark() for _ in range(33)]
    mirror_landmarks[11] = MockLandmark(x=0.5)  # Shoulder x
    mirror_landmarks[23] = MockLandmark(x=0.51)  # Hip x (small offset)

    front_res = MockPoseResults(front_landmarks)
    mirror_res = MockPoseResults(mirror_landmarks)

    result = fusion_engine.evaluate_posture(front_res, mirror_res)

    assert result["state"] == "OPTIMAL"
    assert result["color"] == (0, 200, 0)  # Green


def test_fusion_front_asymmetry(fusion_engine):
    """Test that severe shoulder tilt in the front view triggers a POOR state."""
    # Mock severe shoulder asymmetry (> 0.07 threshold)
    front_landmarks = [MockLandmark() for _ in range(33)]
    front_landmarks[11] = MockLandmark(y=0.4)
    front_landmarks[12] = MockLandmark(y=0.6)  # High vertical difference
    front_landmarks[23] = MockLandmark(y=0.8)
    front_landmarks[24] = MockLandmark(y=0.8)

    # Mock optimal mirror view
    mirror_landmarks = [MockLandmark() for _ in range(33)]
    mirror_landmarks[11] = MockLandmark(x=0.5)
    mirror_landmarks[23] = MockLandmark(x=0.5)

    front_res = MockPoseResults(front_landmarks)
    mirror_res = MockPoseResults(mirror_landmarks)

    result = fusion_engine.evaluate_posture(front_res, mirror_res)

    assert result["state"] == "POOR"
    assert "Asymmetry" in result["front_msg"]


def test_fusion_mirror_torso_lean(fusion_engine):
    """Test that excessive torso lean in the mirror profile view overrides and triggers POOR safety state."""
    # Mock optimal front view
    front_landmarks = [MockLandmark() for _ in range(33)]
    front_landmarks[11] = MockLandmark(y=0.5)
    front_landmarks[12] = MockLandmark(y=0.5)
    front_landmarks[23] = MockLandmark(y=0.8)
    front_landmarks[24] = MockLandmark(y=0.8)

    # Mock severe mirror torso lean (> 0.15 horizontal threshold)
    mirror_landmarks = [MockLandmark() for _ in range(33)]
    mirror_landmarks[11] = MockLandmark(x=0.7)  # Shoulder forward
    mirror_landmarks[23] = MockLandmark(x=0.5)  # Hip back (delta 0.2)

    front_res = MockPoseResults(front_landmarks)
    mirror_res = MockPoseResults(mirror_landmarks)

    result = fusion_engine.evaluate_posture(front_res, mirror_res)

    assert result["state"] == "POOR"
    assert "Torso Lean Risk" in result["mirror_msg"]


def test_fusion_none_inputs(fusion_engine):
    """Test safe handling when pose detection results are None (e.g., tracking lost)."""
    result = fusion_engine.evaluate_posture(None, None)

    assert "Waiting" in result["front_msg"]
    assert "Waiting" in result["mirror_msg"]
