"""
test_multi_detector.py: Unit tests for MultiSkeletonDetector module.
"""

from unittest.mock import MagicMock

import numpy as np

from aura.multi_detector import MultiSkeletonDetector


def test_multi_skeleton_detector_initialization():
    """Test that MultiSkeletonDetector initializes correctly with default split ratios."""
    # Arrange & Act
    detector = MultiSkeletonDetector(split_ratio=0.5)

    # Assert
    assert detector is not None
    assert detector.split_ratio == 0.5
    assert detector.front_detector is not None
    assert detector.mirror_detector is not None


def test_roi_cropping_logic():
    """Test frame slicing into front and mirror ROIs."""
    # Arrange
    detector = MultiSkeletonDetector(split_ratio=0.5)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Act
    front_roi, mirror_roi = detector._crop_rois(dummy_frame)

    # Assert: Left half for front (width 320), right half for mirror (width 320)
    assert front_roi.shape == (480, 320, 3)
    assert mirror_roi.shape == (480, 320, 3)


def test_process_frame_basic(monkeypatch):
    """Test dual-view detection pipeline coordinates splitting, flipping, and detector calls."""
    # Arrange
    detector = MultiSkeletonDetector(split_ratio=0.5)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock detect methods to isolate multi-detector logic
    mock_detect_front = MagicMock()
    mock_detect_mirror = MagicMock()
    monkeypatch.setattr(detector.front_detector, "detect", mock_detect_front)
    monkeypatch.setattr(detector.mirror_detector, "detect", mock_detect_mirror)

    # Mock world landmarks to bypass SVD/validation logic for this basic routing test
    monkeypatch.setattr(
        detector.front_detector,
        "get_world_landmark_array",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        detector.mirror_detector,
        "get_world_landmark_array",
        MagicMock(return_value=None),
    )

    # Act
    result = detector.process_frame(dummy_frame)

    # Assert
    assert "front_roi" in result
    assert "mirror_roi" in result
    assert "front_results" in result
    assert "mirror_results" in result
    assert "direct_pose_2d" in result
    assert "reflected_pose_2d" in result
    mock_detect_front.assert_called_once()
    mock_detect_mirror.assert_called_once()


def test_process_frame_returns_synchronized_typed_observations(monkeypatch):
    """Both views must share frame timing and retain raw/normalized landmarks."""
    from types import SimpleNamespace

    detector = MultiSkeletonDetector(split_ratio=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = [
        SimpleNamespace(x=0.25, y=0.5, z=0.1, visibility=0.9, presence=0.8)
        for _ in range(33)
    ]
    results = SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmarks),
        pose_world_landmarks=SimpleNamespace(landmark=landmarks),
    )

    def set_results(target):
        def _detect(_frame, draw=True):
            target.results = results

        return _detect

    monkeypatch.setattr(
        detector.front_detector, "detect", set_results(detector.front_detector)
    )
    monkeypatch.setattr(
        detector.mirror_detector, "detect", set_results(detector.mirror_detector)
    )
    world = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (33, 1))
    monkeypatch.setattr(
        detector.front_detector, "get_world_landmark_array", lambda: world.copy()
    )
    monkeypatch.setattr(
        detector.mirror_detector, "get_world_landmark_array", lambda: world.copy()
    )

    output = detector.process_frame(frame, frame_id=19, timestamp=12.34)
    front = output["front_observation"]
    mirror = output["mirror_observation"]

    assert front.frame_id == mirror.frame_id == 19
    assert front.timestamp == mirror.timestamp == 12.34
    assert front.detection_valid and mirror.detection_valid
    assert mirror.input_flipped and mirror.world_x_unflipped
    assert np.allclose(mirror.raw_landmarks_3d[:, 0], 0.1)
    assert np.allclose(mirror.landmarks_3d[:, 0], -0.1)
    # x=.25 in the flipped 320px ROI -> 80px -> unflip to 239px + 320px offset.
    assert np.isclose(output["reflected_pose_2d"][0, 0], 559.0)
    assert np.allclose(front.visibility, 0.9)
    assert np.allclose(mirror.presence, 0.8)
