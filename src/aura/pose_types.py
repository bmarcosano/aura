"""Typed data structures shared by AURA's dual-view pose pipeline.

The estimated pipeline deliberately distinguishes MediaPipe's per-view pose
estimates from the fused result.  Coordinates in these classes are relative
MediaPipe world coordinates unless explicitly stated otherwise; they are not
camera-calibrated metric coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Array = np.ndarray


def _as_landmarks(
    value: Array | None, shape: tuple[int, ...], name: str
) -> Array | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.copy()


def _as_confidence(value: Array | None, fallback: float = 0.0) -> Array:
    if value is None:
        return np.full(33, fallback, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (33,):
        raise ValueError(f"Confidence arrays must have shape (33,), got {array.shape}")
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


@dataclass(slots=True)
class PoseObservation:
    """One synchronized detector observation for one side of the split frame."""

    view: Literal["front", "mirror"]
    frame_id: int
    timestamp: float
    landmarks_2d: Array | None
    landmarks_3d: Array | None
    raw_landmarks_3d: Array | None = None
    visibility: Array | None = None
    presence: Array | None = None
    image_size: tuple[int, int] = (0, 0)  # width, height of detector input
    roi_offset: tuple[int, int] = (0, 0)  # x, y offset in original full frame
    input_flipped: bool = False
    world_x_unflipped: bool = False
    detection_valid: bool = False

    def __post_init__(self) -> None:
        self.landmarks_2d = _as_landmarks(self.landmarks_2d, (33, 2), "landmarks_2d")
        self.landmarks_3d = _as_landmarks(self.landmarks_3d, (33, 3), "landmarks_3d")
        if self.raw_landmarks_3d is None:
            self.raw_landmarks_3d = (
                None if self.landmarks_3d is None else self.landmarks_3d.copy()
            )
        else:
            self.raw_landmarks_3d = _as_landmarks(
                self.raw_landmarks_3d, (33, 3), "raw_landmarks_3d"
            )
        self.visibility = _as_confidence(self.visibility)
        self.presence = _as_confidence(
            self.presence, fallback=1.0 if self.detection_valid else 0.0
        )
        if self.view not in ("front", "mirror"):
            raise ValueError("view must be 'front' or 'mirror'")
        finite_world_points = (
            int(np.isfinite(self.landmarks_3d).all(axis=1).sum())
            if self.landmarks_3d is not None
            else 0
        )
        self.detection_valid = bool(
            self.detection_valid
            and self.landmarks_2d is not None
            and self.landmarks_3d is not None
            and finite_world_points >= 8
        )

    @property
    def joint_confidence(self) -> Array:
        """Conservative per-joint confidence from visibility and presence."""
        return np.sqrt(self.visibility * self.presence).astype(np.float32)

    @property
    def valid_joint_mask(self) -> Array:
        if self.landmarks_3d is None:
            return np.zeros(33, dtype=bool)
        finite = np.isfinite(self.landmarks_3d).all(axis=1)
        return finite & (self.joint_confidence > 0.01)


@dataclass(slots=True)
class AlignmentResult:
    """Mirror-to-front similarity transform and its current quality estimate."""

    rotation: Array = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    translation: Array = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: float = 1.0
    ready: bool = False
    quality: float = 0.0
    residual_ratio: float = float("inf")
    sample_count: int = 0

    def transform(self, points: Array | None) -> Array | None:
        if points is None:
            return None
        points_array = np.asarray(points, dtype=np.float32)
        if points_array.shape != (33, 3):
            raise ValueError("points must have shape (33, 3)")
        return (self.scale * (points_array @ self.rotation) + self.translation).astype(
            np.float32
        )


@dataclass(slots=True)
class FusedPose:
    """Single Front-referenced estimated skeleton plus optional debug snapshots."""

    landmarks_3d: Array
    joint_confidence: Array
    joint_sources: tuple[str, ...]
    joint_valid: Array
    alignment_quality: float
    alignment_ready: bool
    fusion_quality: float
    reconstruction_mode: str = "estimated"
    coordinate_system: str = "front_view_relative"
    units: str = "relative"
    is_metric: bool = False
    # Kept in memory for diagnostics; the default viewer renders only landmarks_3d.
    front_landmarks_3d: Array | None = None
    mirror_landmarks_3d: Array | None = None
    aligned_mirror_landmarks_3d: Array | None = None
    frame_id: int = -1
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.landmarks_3d = _as_landmarks(self.landmarks_3d, (33, 3), "fused landmarks")
        self.joint_confidence = _as_confidence(self.joint_confidence)
        self.joint_valid = np.asarray(self.joint_valid, dtype=bool).reshape(33)
        if len(self.joint_sources) != 33:
            raise ValueError("joint_sources must contain 33 entries")
        self.front_landmarks_3d = _as_landmarks(
            self.front_landmarks_3d, (33, 3), "front_landmarks_3d"
        )
        self.mirror_landmarks_3d = _as_landmarks(
            self.mirror_landmarks_3d, (33, 3), "mirror_landmarks_3d"
        )
        self.aligned_mirror_landmarks_3d = _as_landmarks(
            self.aligned_mirror_landmarks_3d, (33, 3), "aligned_mirror_landmarks_3d"
        )
