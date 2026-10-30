import numpy as np

from aura.multi_detector import MultiSkeletonDetector


def test_mirror_roi_mapping_undoes_flip_and_offset():
    width, split_x = 400, 640
    # A physical point at local original x=45 appears at 354 after flipping.
    flipped = np.array([[width - 1 - 45, 27], [0, 10], [399, 350]], dtype=float)
    mapped = MultiSkeletonDetector.map_mirror_pixels_to_full_frame(
        flipped, width, split_x
    )
    assert np.allclose(mapped, [[685, 27], [1039, 10], [640, 350]])
