"""Robust, persistent alignment of MediaPipe mirror poses into the Front frame.

The estimated path is empirical and relative.  It is deliberately separate
from calibrated camera/mirror geometry and must never be labelled metric.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from aura.pose_types import AlignmentResult, PoseObservation


class MirrorToFrontAligner:
    """Estimate a stable Mirror -> Front similarity transform.

    Each synchronized frame contributes one transform estimate.  Transform
    estimates are combined by robust consensus, rather than concatenating
    moving body landmarks across time (which confounds pose motion with the
    coordinate transform).
    """

    # Four torso anchors only.  The nose is intentionally excluded: head
    # motion and view-dependent face estimation must not rotate the whole body.
    ANCHOR_INDICES = (11, 12, 23, 24)
    ANCHOR_BASE_WEIGHTS = np.ones(4, dtype=np.float32)

    def __init__(
        self,
        min_samples: int = 6,
        max_samples: int = 24,
        min_anchor_confidence: float = 0.25,
        max_residual_ratio: float = 0.32,
        min_scale: float = 0.65,
        max_scale: float = 1.55,
        max_rotation_spread_deg: float = 22.0,
        max_translation_spread_ratio: float = 0.35,
        max_scale_spread_ratio: float = 0.25,
    ) -> None:
        self.min_samples = max(3, int(min_samples))
        self.samples: deque[
            tuple[np.ndarray, np.ndarray, float, float, float, float]
        ] = deque(maxlen=max(self.min_samples, int(max_samples)))
        self.min_anchor_confidence = float(min_anchor_confidence)
        self.max_residual_ratio = float(max_residual_ratio)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_rotation_spread = float(np.radians(max_rotation_spread_deg))
        self.max_translation_spread_ratio = float(max_translation_spread_ratio)
        self.max_scale_spread_ratio = float(max_scale_spread_ratio)
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

    @staticmethod
    def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
        relative = first.T @ second
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.arccos(cosine))

    @staticmethod
    def _average_rotations(
        rotations: list[np.ndarray], weights: np.ndarray
    ) -> np.ndarray:
        matrix = np.zeros((3, 3), dtype=np.float64)
        for rotation, weight in zip(rotations, weights):
            matrix += rotation * float(weight)
        u, _singular, vt = np.linalg.svd(matrix)
        correction = np.eye(3, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0:
            correction[-1, -1] = -1.0
        return u @ correction @ vt

    @staticmethod
    def _body_scale(points: np.ndarray) -> float:
        shoulder_center = (points[11] + points[12]) * 0.5
        hip_center = (points[23] + points[24]) * 0.5
        value = float(np.linalg.norm(shoulder_center - hip_center))
        return max(value, 0.15) if np.isfinite(value) else 0.15

    def _sample_from_pair(
        self, front: PoseObservation, mirror: PoseObservation
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float] | None:
        if (
            front.landmarks_3d is None
            or mirror.landmarks_3d is None
            or not front.detection_valid
            or not mirror.detection_valid
            or front.frame_id != mirror.frame_id
        ):
            return None

        # Timestamps are produced from the same source frame.  A mismatch is a
        # synchronization issue, not something the geometric fit should absorb.
        if abs(float(front.timestamp) - float(mirror.timestamp)) > 1e-3:
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
        if (
            np.linalg.matrix_rank(source - source.mean(axis=0), tol=1e-4) < 2
            or np.linalg.matrix_rank(target - target.mean(axis=0), tol=1e-4) < 2
        ):
            return None

        try:
            rotation, translation, scale, residual_ratio = self._weighted_similarity(
                source, target, sample_weights
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not (self.min_scale <= scale <= self.max_scale):
            return None
        if not np.isfinite(residual_ratio) or residual_ratio > self.max_residual_ratio:
            return None

        sample_weight = float(np.mean(sample_weights))
        body_scale = self._body_scale(front.landmarks_3d)
        return (
            rotation,
            translation,
            float(scale),
            float(residual_ratio),
            sample_weight,
            body_scale,
        )

    def _robust_consensus(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, int] | None:
        if len(self.samples) < self.min_samples:
            return None

        rotations = [sample[0] for sample in self.samples]
        translations = [sample[1] for sample in self.samples]
        scales = np.asarray([sample[2] for sample in self.samples], dtype=np.float64)
        residuals = np.asarray([sample[3] for sample in self.samples], dtype=np.float64)
        sample_weights = np.asarray([max(sample[4], 1e-4) for sample in self.samples])
        body_scales = np.asarray([max(sample[5], 0.15) for sample in self.samples])
        # Translation consistency is normalized by torso scale, not by the
        # absolute translation magnitude (which may be close to zero).
        translation_scale = max(float(np.median(body_scales)), 0.15)

        # Select the transform medoid: the estimate with the lowest aggregate
        # distance to the other per-frame transforms.
        pairwise = np.zeros((len(rotations), len(rotations)), dtype=np.float64)
        for i in range(len(rotations)):
            for j in range(i + 1, len(rotations)):
                angle = self._rotation_distance(rotations[i], rotations[j])
                shift = (
                    np.linalg.norm(translations[i] - translations[j])
                    / translation_scale
                )
                scale_delta = abs(np.log(max(scales[i], 1e-6) / max(scales[j], 1e-6)))
                distance = (
                    angle / max(self.max_rotation_spread, 1e-6)
                    + shift / max(self.max_translation_spread_ratio, 1e-6)
                    + scale_delta / max(self.max_scale_spread_ratio, 1e-6)
                )
                pairwise[i, j] = pairwise[j, i] = distance
        medoid_index = int(np.argmin(np.sum(pairwise, axis=1)))
        medoid_rotation = rotations[medoid_index]
        medoid_translation = translations[medoid_index]
        medoid_scale = scales[medoid_index]

        inlier_mask = np.asarray(
            [
                self._rotation_distance(medoid_rotation, rotation)
                <= self.max_rotation_spread
                and np.linalg.norm(translation - medoid_translation)
                <= self.max_translation_spread_ratio * translation_scale
                and abs(scale / medoid_scale - 1.0) <= self.max_scale_spread_ratio
                for rotation, translation, scale in zip(rotations, translations, scales)
            ],
            dtype=bool,
        )
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(self.samples)
        if inlier_count < self.min_samples or inlier_ratio < 0.65:
            return None

        inlier_rotations = [
            rotation for rotation, keep in zip(rotations, inlier_mask) if keep
        ]
        inlier_weights = sample_weights[inlier_mask]
        inlier_weights = inlier_weights / max(float(inlier_weights.sum()), 1e-9)
        rotation = self._average_rotations(inlier_rotations, inlier_weights)
        translation = np.median(
            np.asarray(translations, dtype=np.float64)[inlier_mask], axis=0
        )
        scale = float(np.median(scales[inlier_mask]))
        median_residual = float(np.median(residuals[inlier_mask]))

        angular_spread = float(
            np.median(
                [
                    self._rotation_distance(rotation, candidate)
                    for candidate in inlier_rotations
                ]
            )
        )
        translation_spread = float(
            np.median(
                np.linalg.norm(
                    np.asarray(translations, dtype=np.float64)[inlier_mask]
                    - translation,
                    axis=1,
                )
            )
        ) / max(translation_scale, 1e-6)
        scale_spread = float(
            np.median(np.abs(scales[inlier_mask] / max(scale, 1e-6) - 1.0))
        )
        stability = np.exp(
            -angular_spread / max(self.max_rotation_spread, 1e-6)
            - translation_spread / max(self.max_translation_spread_ratio, 1e-6)
            - scale_spread / max(self.max_scale_spread_ratio, 1e-6)
        )
        residual_quality = float(
            np.clip(
                1.0 - median_residual / max(self.max_residual_ratio, 1e-6), 0.0, 1.0
            )
        )
        quality = float(np.clip(inlier_ratio * stability * residual_quality, 0.0, 1.0))
        residual_summary = float(
            median_residual + angular_spread + translation_spread + scale_spread
        )
        return rotation, translation, scale, quality, residual_summary, inlier_count

    def update(
        self, front: PoseObservation | None, mirror: PoseObservation | None
    ) -> AlignmentResult:
        if self._frozen:
            self._update_validation_quality(front, mirror)
            return self.result

        if front is None or mirror is None or front.frame_id != mirror.frame_id:
            return self.result

        sample = self._sample_from_pair(front, mirror)
        if sample is None:
            return self.result
        self.samples.append(sample)
        self.result.sample_count = len(self.samples)

        consensus = self._robust_consensus()
        if consensus is None:
            # The count and current best single-frame residual remain visible
            # for diagnosis, but alignment is not yet trusted.
            self.result.quality = 0.0
            self.result.ready = False
            self.result.residual_ratio = float(np.median([s[3] for s in self.samples]))
            return self.result

        rotation, translation, scale, quality, residual_summary, inlier_count = (
            consensus
        )
        self.result = AlignmentResult(
            rotation=rotation.astype(np.float32),
            translation=translation.astype(np.float32),
            scale=float(scale),
            ready=quality >= 0.35,
            quality=max(0.0, quality),
            residual_ratio=float(residual_summary),
            sample_count=inlier_count,
        )
        if self.result.ready:
            self._frozen = True
        return self.result

    def _update_validation_quality(
        self, front: PoseObservation | None, mirror: PoseObservation | None
    ) -> None:
        """Measure current torso agreement without changing the frozen transform."""
        if (
            front is None
            or mirror is None
            or front.frame_id != mirror.frame_id
            or abs(float(front.timestamp) - float(mirror.timestamp)) > 1e-3
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
        valid = (
            (weights >= self.min_anchor_confidence)
            & np.isfinite(front.landmarks_3d[indices]).all(axis=1)
            & np.isfinite(mirror.landmarks_3d[indices]).all(axis=1)
        )
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
        self.result.quality = float(0.90 * self.result.quality + 0.10 * instantaneous)
