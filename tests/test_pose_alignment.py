import numpy as np

from aura.pose_alignment import MirrorToFrontAligner
from aura.pose_types import PoseObservation


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _observation(view, frame_id, points, visibility=None):
    if visibility is None:
        visibility = np.full(33, 0.95, dtype=np.float32)
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


def test_alignment_recovers_fixed_similarity_transform():
    rng = np.random.default_rng(7)
    front = rng.normal(0.0, 0.25, size=(33, 3)).astype(np.float32)
    # Ensure the torso anchors are non-collinear.
    front[[0, 11, 12, 23, 24]] = np.array(
        [
            [0.00, 0.48, 0.02],
            [-0.19, 0.28, 0.00],
            [0.19, 0.28, 0.00],
            [-0.12, -0.12, 0.00],
            [0.12, -0.12, 0.00],
        ],
        dtype=np.float32,
    )
    rotation = _rotation_z(17.0)
    scale = 1.08
    translation = np.array([0.12, -0.04, 0.09], dtype=np.float32)
    mirror = ((front - translation) @ rotation.T) / scale

    aligner = MirrorToFrontAligner(min_samples=6)
    result = None
    for frame_id in range(6):
        # Small same-time movement across frames, applied to both views.
        movement = np.zeros((33, 3), dtype=np.float32)
        movement[11:17, 1] += frame_id * 0.002
        movement[23:33, 2] += frame_id * 0.001
        front_frame = front + movement
        mirror_frame = ((front_frame - translation) @ rotation.T) / scale
        result = aligner.update(
            _observation("front", frame_id, front_frame),
            _observation("mirror", frame_id, mirror_frame),
        )

    assert result is not None and result.ready
    assert abs(result.scale - scale) < 1e-3
    transformed = result.transform(mirror)
    assert transformed is not None
    assert np.mean(np.linalg.norm(transformed - front, axis=1)) < 2e-3
    assert result.quality > 0.95
    assert np.linalg.det(result.rotation) > 0.999


def test_alignment_requires_synchronized_confident_observations():
    points = np.zeros((33, 3), dtype=np.float32)
    points[0] = [0.0, 0.4, 0.0]
    points[11] = [-0.2, 0.2, 0.0]
    points[12] = [0.2, 0.2, 0.0]
    points[23] = [-0.1, -0.1, 0.0]
    points[24] = [0.1, -0.1, 0.0]
    low = np.full(33, 0.05, dtype=np.float32)
    aligner = MirrorToFrontAligner(min_samples=3)

    for frame_id in range(10):
        result = aligner.update(
            _observation("front", frame_id, points, low),
            _observation("mirror", frame_id + 1, points, low),
        )
    assert not result.ready
    assert result.sample_count == 0
