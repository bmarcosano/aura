"""
Unit tests for Aura3DViewer in renderer_3d.py, ensuring 3D projection,
orbital controls, upright skeletal orientation, and joint color-coding work properly.
"""

import numpy as np

from aura.renderer_3d import Aura3DViewer


def test_viewer_initialization():
    viewer = Aura3DViewer(width=800, height=600)
    assert viewer.width == 800
    assert viewer.height == 600
    assert viewer.azimuth == 45.0
    assert viewer.elevation == 20.0


def test_keyboard_orbit_controls():
    viewer = Aura3DViewer()

    # Test Azimuth via Numpad 6 / 4 (Right / Left)
    initial_az = viewer.azimuth
    viewer.handle_keyboard(ord("6"))
    assert viewer.azimuth == initial_az + 5.0
    viewer.handle_keyboard(ord("4"))
    assert viewer.azimuth == initial_az

    # Test Elevation via Numpad 8 / 2 (Up / Down)
    initial_el = viewer.elevation
    viewer.handle_keyboard(ord("8"))
    assert viewer.elevation == initial_el + 5.0
    viewer.handle_keyboard(ord("2"))
    assert viewer.elevation == initial_el


def test_projection_point_upright_orientation():
    viewer = Aura3DViewer(width=800, height=600)

    # Origin point (0,0,0) - feet floor center
    u_feet, v_feet = viewer._project_point(np.array([0.0, 0.0, 0.0]))
    assert u_feet == 400  # width // 2
    assert v_feet == 460  # height - 140

    # Head point (-1.7m in MediaPipe World Y)
    _u_head, v_head = viewer._project_point(np.array([0.0, -1.7, 0.0]))
    # Head must be HIGHER on screen than feet (smaller v)
    assert v_head < v_feet


def test_render_scene_with_colors_and_plane():
    viewer = Aura3DViewer(width=800, height=600)
    landmarks_3d = np.zeros((33, 3))
    plane_normal = np.array([0.0, 0.0, 1.0])
    colors = {25: "bad", 26: "warning"}

    canvas = viewer.render_scene(
        landmarks_3d, plane_normal=plane_normal, joint_colors=colors
    )

    assert canvas.shape == (600, 800, 3)
    assert np.any(canvas != 18)  # Background is not empty
