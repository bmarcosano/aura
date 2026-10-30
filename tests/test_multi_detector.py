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
    dummy_raw = np.zeros((33, 4), dtype=np.float32)
    mock_detect_front = MagicMock(return_value=dummy_raw)
    mock_detect_mirror = MagicMock(return_value=dummy_raw)
    monkeypatch.setattr(detector.front_detector, "detect", mock_detect_front)
    monkeypatch.setattr(detector.mirror_detector, "detect", mock_detect_mirror)
    dummy_2d = np.zeros((33, 2), dtype=np.float32)
    monkeypatch.setattr(
        detector.front_detector, "get_landmark_array", MagicMock(return_value=dummy_2d)
    )
    monkeypatch.setattr(
        detector.mirror_detector, "get_landmark_array", MagicMock(return_value=dummy_2d)
    )
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
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

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
