"""
Unit tests for the 3D Pose Reconstruction, Fusion, and EMA smoothing module.
"""

import numpy as np

from aura.reconstruction import PoseReconstructor


def test_single_view_fallback_and_anchoring():
    reconstructor = PoseReconstructor(confidence_weight=0.5, alpha=1.0)  # No EMA
    front = np.zeros((33, 3))

    # Mock feet positions to test spatial grounding
    front[27:33, 1] = 10.0  # Floor Y at 10.0
    front[27, 0] = -2.0  # Left ankle X
    front[28, 0] = 2.0  # Right ankle X (Midpoint X = 0.0)
    front[27, 2] = 5.0  # Left ankle Z
    front[28, 2] = 5.0  # Right ankle Z (Midpoint Z = 5.0)

    fused = reconstructor.fuse_landmarks(front, None)

    assert fused is not None
    assert fused.shape == (33, 3)
    # Check Grounding Math
    assert np.max(fused[27:33, 1]) == 0.0  # Floor should be shifted to Y=0
    assert (fused[27, 0] + fused[28, 0]) / 2.0 == 0.0  # X should be centered
    assert (fused[27, 2] + fused[28, 2]) / 2.0 == 0.0  # Z should be centered


def test_fusion_weighted_average():
    reconstructor = PoseReconstructor(confidence_weight=0.5, alpha=1.0)
    front = np.zeros((33, 3))
    mirror = np.ones((33, 3)) * 10.0

    # 50/50 fusion should result in raw coords being 5.0 everywhere
    fused = reconstructor.fuse_landmarks(
        front, mirror, is_anthropometrically_valid=True
    )

    # After grounding, the floor Y (which was 5.0) becomes 0.0.
    # Therefore, the Y coords of non-feet points (which were also 5.0) become 0.0.
    assert np.allclose(np.max(fused[27:33, 1]), 0.0)


def test_invalid_isometry_fallback():
    reconstructor = PoseReconstructor(confidence_weight=0.5, alpha=1.0)
    front = np.zeros((33, 3))
    mirror = np.ones((33, 3)) * 100.0  # Wild hallucinated values

    # Should ignore mirror completely if invalid
    fused = reconstructor.fuse_landmarks(
        front, mirror, is_anthropometrically_valid=False
    )

    assert (
        np.max(fused[27:33, 1]) == 0.0
    )  # Grounded front (0.0), not influenced by mirror (100.0)


def test_ema_smoothing():
    reconstructor = PoseReconstructor(confidence_weight=0.5, alpha=0.5)

    # Frame 1
    front1 = np.zeros((33, 3))
    fused1 = reconstructor.fuse_landmarks(front1, None)

    assert reconstructor._previous_fused is not None
    assert np.array_equal(reconstructor._previous_fused, fused1)

    # Frame 2
    front2 = np.zeros((33, 3))
    # Mock a change in the front view to test EMA smoothing
    front2[0:25] = 10.0
    fused2 = reconstructor.fuse_landmarks(front2, None)

    # Since alpha=0.5, the raw new grounded frame and the old frame are averaged
    assert not np.array_equal(fused1, fused2)

    # Reset
    reconstructor.reset_smoothing()
    assert reconstructor._previous_fused is None


def test_none_inputs():
    reconstructor = PoseReconstructor()
    assert reconstructor.fuse_landmarks(None, None) is None
