"""Persistent alignment of MediaPipe mirror-world poses into the Front frame.

This is an empirical alignment for the UNCALIBRATED estimated mode.  It is not
camera calibration and must never be used to claim metric 3D reconstruction.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from aura.pose_types import AlignmentResult, PoseObservation


class MirrorToFrontAligner:
    """Estimate one stable similarity transform from mirror to front coordinates.

    A short synchronized observation buffer is used to estimate a single
    transform. Once accepted, that transform is frozen for the session so
    exercise motion cannot rotate/recenter the global skeleton on every frame.
    """

    # Torso anchors only: do not use elbows, wrists, knees or ankles to define
    # the global coordinate frame that the fuser is meant to supplement.
    ANCHOR_INDICES = (0, 11, 12, 23, 24)
    ANCHOR_BASE_WEIGHTS = np.asarray([0.35, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        min_samples: int = 6,
        max_samples: int = 24,
        min_anchor_confidence: float = 0.25,
        max_residual_ratio: float = 0.32,
        min_scale: float = 0.65,
        max_scale: float = 1.55,
    ) -> None:
        self.min_samples = max(3, int(min_samples))
        self.samples: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=max(self.min_samples, int(max_samples))
        )
        self.min_anchor_confidence = float(min_anchor_confidence)
        self.max_residual_ratio = float(max_residual_ratio)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.result = AlignmentResult()
        self._frozen = False

    def reset(self) -> None:
        self.samples.clear()
        self.result = AlignmentResult()
        self._frozen = False

    @staticmethod
    def _weighted_similarity(
        source: np.ndarray, target: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Fit row-vector transform: target ~= scale * source @ R + t."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        valid = (
            np.isfinite(source).all(axis=1)
            & np.isfinite(target).all(axis=1)
            & np.isfinite(weights)
            & (weights > 1e-6)
        )
        source, target, weights = source[valid], target[valid], weights[valid]
        if len(source) < 3:
            raise ValueError("At least three valid anchor pairs are required")

        weights = weights / weights.sum()
        source_mean = np.sum(source * weights[:, None], axis=0)
        target_mean = np.sum(target * weights[:, None], axis=0)
        source_centered = source - source_mean
        target_centered = target - target_mean

        covariance = (source_centered * weights[:, None]).T @ target_centered
        u, singular_values, vt = np.linalg.svd(covariance)
        correction = np.eye(3, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0:
            correction[-1, -1] = -1.0
        rotation = u @ correction @ vt

        source_variance = np.sum(weights * np.sum(source_centered**2, axis=1))
        if source_variance < 1e-10:
            raise ValueError("Degenerate source anchors")
        scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
        translation = target_mean - scale * (source_mean @ rotation)

        aligned = scale * (source @ rotation) + translation
        residual = np.linalg.norm(aligned - target, axis=1)
        rms = float(np.sqrt(np.sum(weights * residual**2)))
        target_spread = float(
            np.sqrt(np.sum(weights * np.sum(target_centered**2, axis=1)))
        )
        residual_ratio = rms / max(target_spread, 1e-6)
        return rotation, translation, scale, residual_ratio

    def _sample_from_pair(
        self, front: PoseObservation, mirror: PoseObservation
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if (
            front.landmarks_3d is None
            or mirror.landmarks_3d is None
            or not front.detection_valid
            or not mirror.detection_valid
            or front.frame_id != mirror.frame_id
        ):
            return None

        indices = np.asarray(self.ANCHOR_INDICES, dtype=int)
        front_conf = front.joint_confidence[indices]
        mirror_conf = mirror.joint_confidence[indices]
        weights = np.minimum(front_conf, mirror_conf) * self.ANCHOR_BASE_WEIGHTS
        valid = (
            (weights >= self.min_anchor_confidence)
            & np.isfinite(front.landmarks_3d[indices]).all(axis=1)
            & np.isfinite(mirror.landmarks_3d[indices]).all(axis=1)
        )
        if int(valid.sum()) < 3:
            return None

        source = mirror.landmarks_3d[indices[valid]]
        target = front.landmarks_3d[indices[valid]]
        sample_weights = weights[valid]
        # Reject nearly collinear / collapsed anchor configurations.
        if (
            np.linalg.matrix_rank(source - source.mean(axis=0), tol=1e-4) < 2
            or np.linalg.matrix_rank(target - target.mean(axis=0), tol=1e-4) < 2
        ):
            return None
        return source.copy(), target.copy(), sample_weights.copy()

    def update(
        self, front: PoseObservation | None, mirror: PoseObservation | None
    ) -> AlignmentResult:
        if self._frozen:
            self._update_validation_quality(front, mirror)
            return self.result

        if front is None or mirror is None:
            return self.result
        if front.frame_id != mirror.frame_id:
            return self.result

        sample = self._sample_from_pair(front, mirror)
        if sample is None:
            return self.result
        self.samples.append(sample)

        if len(self.samples) < self.min_samples:
            self.result.sample_count = len(self.samples)
            return self.result

        source = np.concatenate([sample[0] for sample in self.samples], axis=0)
        target = np.concatenate([sample[1] for sample in self.samples], axis=0)
        weights = np.concatenate([sample[2] for sample in self.samples], axis=0)

        try:
            rotation, translation, scale, residual_ratio = self._weighted_similarity(
                source, target, weights
            )
        except (ValueError, np.linalg.LinAlgError):
            self.result.sample_count = len(self.samples)
            return self.result

        if not (self.min_scale <= scale <= self.max_scale):
            self.result.sample_count = len(self.samples)
            return self.result

        quality = float(
            np.clip(1.0 - residual_ratio / self.max_residual_ratio, 0.0, 1.0)
        )
        if residual_ratio > self.max_residual_ratio:
            self.result = AlignmentResult(
                rotation=rotation.astype(np.float32),
                translation=translation.astype(np.float32),
                scale=float(scale),
                ready=False,
                quality=quality,
                residual_ratio=float(residual_ratio),
                sample_count=len(self.samples),
            )
            return self.result

        self.result = AlignmentResult(
            rotation=rotation.astype(np.float32),
            translation=translation.astype(np.float32),
            scale=float(scale),
            ready=True,
            quality=max(0.15, quality),
            residual_ratio=float(residual_ratio),
            sample_count=len(self.samples),
        )
        self._frozen = True
        return self.result

    def _update_validation_quality(
        self, front: PoseObservation | None, mirror: PoseObservation | None
    ) -> None:
        """Measure current anchor agreement without changing the frozen transform."""
        if (
            front is None
            or mirror is None
            or front.frame_id != mirror.frame_id
            or front.landmarks_3d is None
            or mirror.landmarks_3d is None
            or not front.detection_valid
            or not mirror.detection_valid
        ):
            self.result.quality *= 0.98
            return

        indices = np.asarray(self.ANCHOR_INDICES, dtype=int)
        weights = (
            np.minimum(
                front.joint_confidence[indices], mirror.joint_confidence[indices]
            )
            * self.ANCHOR_BASE_WEIGHTS
        )
        valid = weights >= self.min_anchor_confidence
        if int(valid.sum()) < 3:
            self.result.quality *= 0.98
            return

        mirror_aligned = self.result.transform(mirror.landmarks_3d)
        residuals = np.linalg.norm(
            mirror_aligned[indices[valid]] - front.landmarks_3d[indices[valid]], axis=1
        )
        front_scale = self._body_scale(front.landmarks_3d)
        residual_ratio = float(np.average(residuals, weights=weights[valid])) / max(
            front_scale, 1e-6
        )
        self.result.residual_ratio = residual_ratio
        instantaneous = float(np.clip(1.0 - residual_ratio / 0.45, 0.0, 1.0))
        self.result.quality = float(0.85 * self.result.quality + 0.15 * instantaneous)

    @staticmethod
    def _body_scale(points: np.ndarray) -> float:
        shoulder_center = (points[11] + points[12]) * 0.5
        hip_center = (points[23] + points[24]) * 0.5
        return float(max(np.linalg.norm(shoulder_center - hip_center), 0.15))
