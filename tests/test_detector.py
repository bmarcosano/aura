"""test_detector.py: Unit tests for PoseDetector class."""

import numpy as np
import pytest
from aura.detector import PoseDetector


def test_detector_initialization():
    """Tests that PoseDetector initializes correctly with default parameters."""
    detector = PoseDetector()
    assert detector.pose is not None


def test_detector_with_dummy_frame():
    """Tests processing a synthetic blank frame (no human present)."""
    detector = PoseDetector()

    # Create a blank black frame (Height=480, Width=640, 3 channels)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process the frame
    processed_frame = detector.find_pose(dummy_frame, draw=True)

    # Verify that the output is still a valid OpenCV image of the same size
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == (480, 640, 3)

    # Since it's a blank black frame, MediaPipe should find no landmarks
    landmarks = detector.get_landmark_array(frame_width=640, frame_height=480)
    assert landmarks is None
