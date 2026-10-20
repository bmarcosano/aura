"""test_detector.py: Unit tests for PoseDetector class."""

from unittest.mock import MagicMock

import numpy as np

from aura.detector import PoseDetector


def test_detector_initialization():
    """Tests that PoseDetector initializes correctly with default parameters."""
    # Arrange & Act
    detector = PoseDetector()

    # Assert
    assert detector.pose is not None
    assert detector.results is None


def test_detector_with_dummy_frame():
    """Tests processing a synthetic blank frame (no human present)."""
    # Arrange
    detector = PoseDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Act: Process the frame
    processed_frame = detector.find_pose(dummy_frame, draw=True)

    # Assert: Verify that the output is still a valid OpenCV image of the same size
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == (480, 640, 3)

    # Since it's a blank black frame, MediaPipe should find no landmarks
    landmarks = detector.get_landmark_array(frame_width=640, frame_height=480)
    assert landmarks is None


def test_detector_detect_without_drawing():
    """Tests detection behavior when drawing is explicitly disabled (draw=False)."""
    # Arrange
    detector = PoseDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Act: Run detect with draw=False to cover the non-drawing branch
    landmarks = detector.detect(dummy_frame, draw=False)

    # Assert
    assert landmarks is None


def test_detector_detect_with_mocked_drawing(monkeypatch):
    """Tests detection rendering branch (line 47) by mocking MediaPipe process results."""
    # Arrange
    detector = PoseDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock a valid landmark structure so get_landmark_array doesn't raise an AttributeError
    mock_landmark = MagicMock()
    mock_landmark.x = 0.5
    mock_landmark.y = 0.5
    mock_landmark.z = 0.0
    mock_landmark.visibility = 0.99

    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [mock_landmark] * 33
    monkeypatch.setattr(detector.pose, "process", lambda img: mock_results)

    # Mock draw_landmarks to intercept the call and prevent internal errors
    mock_draw = MagicMock()
    monkeypatch.setattr(detector.mp_draw, "draw_landmarks", mock_draw)

    # Act: Run detect with draw=True
    landmarks = detector.detect(dummy_frame, draw=True)

    # Assert: Verify that draw_landmarks was successfully invoked
    mock_draw.assert_called_once()
    assert isinstance(landmarks, np.ndarray)


def test_find_pose_legacy_helper(monkeypatch):
    """Tests the legacy find_pose wrapper method (covers lines 74-81)."""
    # Arrange
    detector = PoseDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock process to return valid results for the helper method
    mock_results = MagicMock()
    mock_results.pose_landmarks = None
    monkeypatch.setattr(detector.pose, "process", lambda img: mock_results)

    # Act: Call legacy find_pose method
    annotated_frame = detector.find_pose(dummy_frame, draw=True)

    # Assert
    assert isinstance(annotated_frame, np.ndarray)
    assert annotated_frame.shape == dummy_frame.shape
