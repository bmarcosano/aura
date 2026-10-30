import numpy as np

from aura.renderer_3d import Aura3DViewer


def test_renderer_skips_nonfinite_landmarks():
    points = np.full((33, 3), np.nan)
    points[11] = [0.0, 0.0, 0.0]
    points[12] = [0.2, 0.0, 0.0]
    image = Aura3DViewer().render_scene(points, reconstruction_mode="TEST")
    assert image.shape == (600, 800, 3)
