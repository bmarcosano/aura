"""
test_fusion.py: Unit tests for AuraFusionEngine module.
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
    # Arrange
    front_landmarks = [MockLandmark() for _ in range(33)]
    front_landmarks[11] = MockLandmark(y=0.5)
    front_landmarks[12] = MockLandmark(y=0.501)
    front_landmarks[23] = MockLandmark(y=0.8)
    front_landmarks[24] = MockLandmark(y=0.801)

    mirror_landmarks = [MockLandmark() for _ in range(33)]
    mirror_landmarks[11] = MockLandmark(x=0.5)
    mirror_landmarks[23] = MockLandmark(x=0.51)

    front_res = MockPoseResults(front_landmarks)
    mirror_res = MockPoseResults(mirror_landmarks)

    # Act
    result = fusion_engine.evaluate_posture(front_res, mirror_res)

    # Assert
    assert result["state"] == "OPTIMAL"
    assert result["color"] == (0, 200, 0)


def test_fusion_front_asymmetry(fusion_engine):
    """Test that severe shoulder tilt triggers a POOR state."""
    # Arrange
    front_landmarks = [MockLandmark() for _ in range(33)]
    front_landmarks[11] = MockLandmark(y=0.4)
    front_landmarks[12] = MockLandmark(y=0.6)
    front_landmarks[23] = MockLandmark(y=0.8)
    front_landmarks[24] = MockLandmark(y=0.8)

    mirror_landmarks = [MockLandmark() for _ in range(33)]
    mirror_landmarks[11] = MockLandmark(x=0.5)
    mirror_landmarks[23] = MockLandmark(x=0.5)

    front_res = MockPoseResults(front_landmarks)
    mirror_res = MockPoseResults(mirror_landmarks)

    # Act
    result = fusion_engine.evaluate_posture(front_res, mirror_res)

    # Assert
    assert result["state"] == "POOR"
    assert "Asymmetry" in result["front_msg"]


def test_fusion_front_warning_states(fusion_engine):
    """Test front view Warning states (mild tilt and knee alignment)."""
    # Arrange 1: Mild tilt (> 0.03 and <= 0.07)
    front_landmarks_tilt = [MockLandmark(y=0.5) for _ in range(33)]
    front_landmarks_tilt[11] = MockLandmark(y=0.5)
    front_landmarks_tilt[12] = MockLandmark(y=0.54)
    front_landmarks_tilt[23] = MockLandmark(y=0.8)
    front_landmarks_tilt[24] = MockLandmark(y=0.8)

    # Arrange 2: Knee drift (> 0.12) with balanced shoulders and hips
    front_landmarks_knee = [MockLandmark(y=0.5) for _ in range(33)]
    front_landmarks_knee[11] = MockLandmark(y=0.5)
    front_landmarks_knee[12] = MockLandmark(y=0.5)
    front_landmarks_knee[23] = MockLandmark(x=0.5, y=0.8)
    front_landmarks_knee[24] = MockLandmark(x=0.7, y=0.8)
    front_landmarks_knee[25] = MockLandmark(x=0.65, y=0.9)

    mirror_landmarks = [MockLandmark(x=0.5, y=0.5) for _ in range(33)]
    mirror_landmarks[11] = MockLandmark(x=0.5)
    mirror_landmarks[23] = MockLandmark(x=0.5)
    mirror_res = MockPoseResults(mirror_landmarks)

    # Act & Assert (Mild tilt warning)
    res_tilt = fusion_engine.evaluate_posture(
        MockPoseResults(front_landmarks_tilt), mirror_res
    )
    assert res_tilt["state"] == "WARNING"
    assert "Mild Tilt" in res_tilt["front_msg"]

    # Act & Assert (Knee alignment warning)
    res_knee = fusion_engine.evaluate_posture(
        MockPoseResults(front_landmarks_knee), mirror_res
    )
    assert res_knee["state"] == "WARNING"
    assert "Knee Alignment" in res_knee["front_msg"]


def test_fusion_mirror_warning_and_poor(fusion_engine):
    """Test mirror profile warning and poor torso lean states."""
    # Arrange
    front_landmarks = [MockLandmark() for _ in range(33)]
    front_landmarks[11] = MockLandmark(y=0.5)
    front_landmarks[12] = MockLandmark(y=0.5)
    front_landmarks[23] = MockLandmark(y=0.8)
    front_landmarks[24] = MockLandmark(y=0.8)
    front_res = MockPoseResults(front_landmarks)

    # 1. Mirror Warning (torso lean > 0.08)
    mirror_warn = [MockLandmark() for _ in range(33)]
    mirror_warn[11] = MockLandmark(x=0.6)
    mirror_warn[23] = MockLandmark(x=0.5)

    # 2. Mirror Poor (torso lean > 0.15)
    mirror_poor = [MockLandmark() for _ in range(33)]
    mirror_poor[11] = MockLandmark(x=0.7)
    mirror_poor[23] = MockLandmark(x=0.5)

    # Act & Assert
    res_w = fusion_engine.evaluate_posture(front_res, MockPoseResults(mirror_warn))
    assert res_w["state"] == "WARNING"
    assert "Check Spine Form" in res_w["mirror_msg"]

    res_p = fusion_engine.evaluate_posture(front_res, MockPoseResults(mirror_poor))
    assert res_p["state"] == "POOR"
    assert "Torso Lean Risk" in res_p["mirror_msg"]


def test_fusion_exceptions_handling(fusion_engine):
    """Test robust handling of malformed landmark structures triggering exceptions."""
    # Arrange: Results object missing required landmark attributes
    malformed_res = type(
        "BadObj",
        (object,),
        {"pose_landmarks": type("BadLM", (object,), {"landmark": []})},
    )()

    # Act
    result = fusion_engine.evaluate_posture(malformed_res, malformed_res)

    # Assert
    assert result["front_msg"] == "Front: Tracking..."
    assert result["mirror_msg"] == "Mirror: Tracking..."


def test_fusion_none_inputs(fusion_engine):
    """Test safe handling when pose detection results are None."""
    result = fusion_engine.evaluate_posture(None, None)

    assert "Waiting" in result["front_msg"]
    assert "Waiting" in result["mirror_msg"]
