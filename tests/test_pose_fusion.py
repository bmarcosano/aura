import numpy as np

from aura.pose_alignment import MirrorToFrontAligner
from aura.pose_fusion import DualViewPoseFuser
from aura.pose_types import PoseObservation


def _base_pose():
    points = np.zeros((33, 3), dtype=np.float32)
    points[0] = [0.0, 0.55, 0.02]
    points[11] = [-0.20, 0.30, 0.0]
    points[12] = [0.20, 0.30, 0.0]
    points[13] = [-0.30, 0.15, 0.0]
    points[14] = [0.30, 0.15, 0.0]
    points[15] = [-0.35, 0.02, 0.0]
    points[16] = [0.35, 0.02, 0.0]
    points[23] = [-0.12, -0.12, 0.0]
    points[24] = [0.12, -0.12, 0.0]
    points[25] = [-0.13, -0.38, 0.0]
    points[26] = [0.13, -0.38, 0.0]
    points[27] = [-0.13, -0.68, 0.0]
    points[28] = [0.13, -0.68, 0.0]
    # Additional anchors for a non-degenerate torso fit.
    points[1] = [0.0, 0.50, 0.04]
    return points


def _obs(view, frame_id, points, visibility):
    return PoseObservation(
        view=view,
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        landmarks_2d=np.zeros((33, 2), dtype=np.float32),
        landmarks_3d=points,
        visibility=visibility,
        presence=np.full(33, 0.95, dtype=np.float32),
        detection_valid=True,
    )


def test_front_only_until_alignment_then_mirror_can_dominate_joint():
    points = _base_pose()
    aligner = MirrorToFrontAligner(min_samples=5)
    fuser = DualViewPoseFuser(aligner=aligner)
    front_visibility = np.full(33, 0.9, dtype=np.float32)
    mirror_visibility = np.full(33, 0.9, dtype=np.float32)

    # A second-view-only elbow position: it is deliberately different, while
    # torso anchors remain identical and establish the shared reference.
    mirror_points = points.copy()
    mirror_points[13] = [-0.45, 0.20, 0.16]
    mirror_points[15] = [-0.52, 0.08, 0.20]
    front_visibility[13] = 0.08
    front_visibility[15] = 0.08

    first_result = None
    for frame_id in range(5):
        first_result = fuser.update(
            _obs("front", frame_id, points, front_visibility),
            _obs("mirror", frame_id, mirror_points, mirror_visibility),
        )

    assert first_result is not None
    assert first_result.alignment_ready
    assert not first_result.is_metric
    assert first_result.reconstruction_mode == "estimated"
    assert first_result.joint_sources[13] in ("MIRROR_DOMINANT", "MIRROR_ONLY")
    assert np.linalg.norm(first_result.landmarks_3d[13] - points[13]) > 0.05
    # Pelvis remains Front-anchored even though both views are available.
    assert np.linalg.norm(first_result.landmarks_3d[23] - points[23]) < 0.01


def test_front_reference_is_not_replaced_when_front_temporarily_missing():
    points = _base_pose()
    visibility = np.full(33, 0.9, dtype=np.float32)
    fuser = DualViewPoseFuser(aligner=MirrorToFrontAligner(min_samples=3))

    result = None
    for frame_id in range(3):
        result = fuser.update(
            _obs("front", frame_id, points, visibility),
            _obs("mirror", frame_id, points, visibility),
        )
    assert result is not None

    # A mirror observation alone cannot become a new global reference.
    held = fuser.update(None, _obs("mirror", 3, points + 0.1, visibility))
    assert held is not None
    assert np.allclose(held.landmarks_3d, result.landmarks_3d)
    assert all(source == "HELD_FRONT" for source in held.joint_sources)


def test_fuser_handles_low_confidence_joints_without_nan_output():
    points = _base_pose()
    visibility = np.full(33, 0.9, dtype=np.float32)
    visibility[16] = 0.0
    points[16] = np.nan
    observation = _obs("front", 0, points, visibility)
    fuser = DualViewPoseFuser()
    result = fuser.update(observation, None)
    assert result is not None
    assert np.isfinite(result.landmarks_3d).all()
    assert not result.joint_valid[16]


def test_mismatched_frame_ids_never_fuse_stale_mirror_pose():
    points = _base_pose()
    visibility = np.full(33, 0.9, dtype=np.float32)
    fuser = DualViewPoseFuser(aligner=MirrorToFrontAligner(min_samples=3))

    result = None
    for frame_id in range(3):
        result = fuser.update(
            _obs("front", frame_id, points, visibility),
            _obs("mirror", frame_id, points, visibility),
        )
    assert result is not None and result.alignment_ready

    stale_mirror = _obs("mirror", 99, points + 0.2, visibility)
    current_front = _obs("front", 3, points, visibility)
    result = fuser.update(current_front, stale_mirror)
    assert result is not None
    assert all(source in ("FRONT", "LOW_CONFIDENCE") for source in result.joint_sources)
    assert np.allclose(result.landmarks_3d, points)
