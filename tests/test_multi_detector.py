from __future__ import annotations

import numpy as np

from aura.multi_detector import MultiSkeletonDetector


def test_multi_skeleton_detector_initialization():
    """Test that MultiSkeletonDetector initializes correctly with default split ratios."""
    detector = MultiSkeletonDetector(split_ratio=0.5)
    assert detector is not None
    assert detector.split_ratio == 0.5


def test_roi_cropping_logic():
    """Test frame slicing into front and mirror ROIs."""
    detector = MultiSkeletonDetector(split_ratio=0.5)

    # Create a dummy synthetic frame (Height: 480, Width: 640, Channels: 3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    front_roi, mirror_roi = detector._crop_rois(dummy_frame)

    # Left half for front (width 320), right half for mirror (width 320)
    assert front_roi.shape == (480, 320, 3)
    assert mirror_roi.shape == (480, 320, 3)
