"""Front-referenced, confidence-weighted dual-view pose fusion."""

from __future__ import annotations

import numpy as np

from aura.pose_alignment import MirrorToFrontAligner
from aura.pose_types import FusedPose, PoseObservation


class DualViewPoseFuser:
    """Fuse two MediaPipe poses into one skeleton anchored to the Front View.

    This is an empirical, relative 3D estimator. It does not perform metric
    triangulation and must not be labelled as calibrated 3D.
    """

    def __init__(
        self,
        aligner: MirrorToFrontAligner | None = None,
        minimum_joint_confidence: float = 0.05,
        max_mirror_share: float = 0.90,
        conflict_ratio: float = 0.40,
        minimum_alignment_for_mirror_only: float = 0.35,
    ) -> None:
        self.aligner = aligner or MirrorToFrontAligner()
        self.minimum_joint_confidence = float(minimum_joint_confidence)
        self.max_mirror_share = float(max_mirror_share)
        self.conflict_ratio = float(conflict_ratio)
        self.minimum_alignment_for_mirror_only = float(
            minimum_alignment_for_mirror_only
        )
        self._last_front: np.ndarray | None = None
        self._last_result: FusedPose | None = None
        self._front_missing_frames = 0

        # The pelvis anchors the Front coordinate system. The mirror may refine
        # them slightly but cannot move the global root.
        self._mirror_share_caps = np.full(33, self.max_mirror_share, dtype=np.float32)
        # The pelvis is the permanent Front root. Keep a very small mirror
        # contribution only to absorb numerical noise, never to move the root.
        self._mirror_share_caps[[23, 24]] = 0.05
        # Head/face and shoulder anchors stay Front-led. Mirror may refine them
        # when useful, but must not drag the global frame around.
        self._mirror_share_caps[0:11] = 0.35
        self._mirror_share_caps[[11, 12]] = 0.35
        self._front_reference_indices = {11, 12, 23, 24}
        # Disagreement thresholds are joint-specific fractions of torso scale.
        self._joint_conflict_limits = np.full(33, self.conflict_ratio, dtype=np.float32)
        self._joint_conflict_limits[0:11] = min(self.conflict_ratio, 0.30)
        self._joint_conflict_limits[[11, 12, 23, 24]] = min(self.conflict_ratio, 0.25)

    def reset(self) -> None:
        self.aligner.reset()
        self._last_front = None
        self._last_result = None
        self._front_missing_frames = 0

    @staticmethod
    def _body_scale(points: np.ndarray) -> float:
        shoulder_center = (points[11] + points[12]) * 0.5
        hip_center = (points[23] + points[24]) * 0.5
        torso = np.linalg.norm(shoulder_center - hip_center)
        shoulder_width = np.linalg.norm(points[11] - points[12])
        values = [value for value in (torso, shoulder_width) if np.isfinite(value)]
        return float(max(values + [0.15]))

    def _front_only_result(
        self, front: PoseObservation, alignment_quality: float, alignment_ready: bool
    ) -> FusedPose:
        assert front.landmarks_3d is not None
        points = front.landmarks_3d.copy()
        confidence = front.joint_confidence.copy()
        finite = np.isfinite(points).all(axis=1)
        valid = finite & (confidence >= self.minimum_joint_confidence)
        confidence[~finite] = 0.0
        if self._last_front is not None:
            points[~finite] = self._last_front[~finite]
        else:
            points[~finite] = 0.0
        sources = tuple("FRONT" if ok else "LOW_CONFIDENCE" for ok in valid)
        result = FusedPose(
            landmarks_3d=points,
            joint_confidence=confidence,
            joint_sources=sources,
            joint_valid=valid,
            alignment_quality=alignment_quality,
            alignment_ready=alignment_ready,
            fusion_quality=float(np.mean(confidence[valid])) if valid.any() else 0.0,
            front_landmarks_3d=front.raw_landmarks_3d,
            frame_id=front.frame_id,
            timestamp=front.timestamp,
        )
        self._last_front = points.copy()
        self._last_result = result
        self._front_missing_frames = 0
        return result

    def _hold_front_reference(
        self, frame_id: int, timestamp: float
    ) -> FusedPose | None:
        """Hold the last Front-referenced pose briefly; never switch reference to mirror."""
        self._front_missing_frames += 1
        if self._last_result is None or self._front_missing_frames > 2:
            return None
        held = self._last_result.landmarks_3d.copy()
        confidence = self._last_result.joint_confidence * (
            0.65**self._front_missing_frames
        )
        valid = self._last_result.joint_valid.copy()
        valid &= confidence >= self.minimum_joint_confidence
        sources = tuple("HELD_FRONT" if ok else "LOW_CONFIDENCE" for ok in valid)
        result = FusedPose(
            landmarks_3d=held,
            joint_confidence=confidence,
            joint_sources=sources,
            joint_valid=valid,
            alignment_quality=self._last_result.alignment_quality,
            alignment_ready=self._last_result.alignment_ready,
            fusion_quality=float(np.mean(confidence[valid])) if valid.any() else 0.0,
            front_landmarks_3d=self._last_result.front_landmarks_3d,
            mirror_landmarks_3d=self._last_result.mirror_landmarks_3d,
            aligned_mirror_landmarks_3d=self._last_result.aligned_mirror_landmarks_3d,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        self._last_result = result
        return result

    def update(
        self,
        front: PoseObservation | None,
        mirror: PoseObservation | None,
    ) -> FusedPose | None:
        """Update the fused skeleton from synchronized observations."""
        frame_id = (
            front.frame_id if front is not None else (mirror.frame_id if mirror else -1)
        )
        timestamp = (
            front.timestamp
            if front is not None
            else (mirror.timestamp if mirror else 0.0)
        )

        if front is None or front.landmarks_3d is None or not front.detection_valid:
            return self._hold_front_reference(frame_id, timestamp)

        mirror_is_synchronized = (
            mirror is not None and mirror.frame_id == front.frame_id
        )
        alignment = self.aligner.update(
            front, mirror if mirror_is_synchronized else None
        )
        if (
            not mirror_is_synchronized
            or mirror is None
            or mirror.landmarks_3d is None
            or not mirror.detection_valid
            or not alignment.ready
        ):
            return self._front_only_result(front, alignment.quality, alignment.ready)

        aligned_mirror = alignment.transform(mirror.landmarks_3d)
        if aligned_mirror is None or not np.isfinite(aligned_mirror).all():
            return self._front_only_result(front, 0.0, False)

        front_points = front.landmarks_3d
        front_conf = front.joint_confidence
        mirror_conf = mirror.joint_confidence * float(
            np.clip(alignment.quality, 0.0, 1.0)
        )
        output = front_points.copy()
        confidence = np.zeros(33, dtype=np.float32)
        valid = np.zeros(33, dtype=bool)
        sources: list[str] = []
        body_scale = self._body_scale(front_points)

        for index in range(33):
            p_front = front_points[index]
            p_mirror = aligned_mirror[index]
            q_front = float(front_conf[index]) if np.isfinite(p_front).all() else 0.0
            q_mirror = float(mirror_conf[index]) if np.isfinite(p_mirror).all() else 0.0
            q_front = max(0.0, q_front)
            q_mirror = max(0.0, q_mirror)

            if (
                q_front < self.minimum_joint_confidence
                and q_mirror < self.minimum_joint_confidence
            ):
                if not np.isfinite(output[index]).all():
                    output[index] = (
                        self._last_front[index]
                        if self._last_front is not None
                        and np.isfinite(self._last_front[index]).all()
                        else np.zeros(3, dtype=np.float32)
                    )
                confidence[index] = 0.0
                valid[index] = False
                sources.append("LOW_CONFIDENCE")
                continue

            if q_front < self.minimum_joint_confidence:
                if index in self._front_reference_indices:
                    # Pelvis and shoulder anchors define the permanent Front
                    # reference. A mirror-only observation must not replace
                    # these global anchors when the Front detector is uncertain.
                    if np.isfinite(p_front).all():
                        output[index] = p_front
                    elif self._last_front is not None:
                        output[index] = self._last_front[index]
                    else:
                        output[index] = np.zeros(3, dtype=np.float32)
                    confidence[index] = q_front
                    valid[index] = False
                    sources.append("FRONT_REFERENCE_LOW_CONFIDENCE")
                elif alignment.quality >= self.minimum_alignment_for_mirror_only:
                    output[index] = p_mirror
                    confidence[index] = q_mirror * 0.75
                    valid[index] = True
                    sources.append("MIRROR_ONLY")
                else:
                    # The local mirror estimate is visible, but the coordinate
                    # alignment is not trustworthy enough to place it in Front space.
                    if not np.isfinite(p_front).all():
                        output[index] = (
                            self._last_front[index]
                            if self._last_front is not None
                            else np.zeros(3, dtype=np.float32)
                        )
                    else:
                        output[index] = p_front
                    confidence[index] = q_front
                    valid[index] = False
                    sources.append("MIRROR_ALIGNMENT_LOW")
                continue

            if q_mirror < self.minimum_joint_confidence:
                output[index] = p_front
                confidence[index] = q_front
                valid[index] = True
                sources.append("FRONT_ONLY")
                continue

            # A large disagreement is not averaged blindly. When one detector is
            # clearly more confident, trust it; otherwise retain Front and lower
            # the confidence to expose an unresolved conflict.
            disagreement = float(np.linalg.norm(p_front - p_mirror))
            normalized_disagreement = disagreement / max(body_scale, 1e-6)
            if normalized_disagreement > float(self._joint_conflict_limits[index]):
                mirror_cap = float(self._mirror_share_caps[index])
                if q_mirror > q_front * 1.5 and mirror_cap > 0.0:
                    requested_share = q_mirror / max(q_front + q_mirror, 1e-6)
                    mirror_share = min(requested_share, mirror_cap)
                    output[index] = (
                        1.0 - mirror_share
                    ) * p_front + mirror_share * p_mirror
                    confidence[index] = (
                        min(
                            (1.0 - mirror_share) * q_front + mirror_share * q_mirror,
                            0.55,
                        )
                        * 0.75
                    )
                    sources.append(
                        "MIRROR_DOMINANT"
                        if mirror_share >= 0.65
                        else "MIRROR_CONFLICT_DOMINANT"
                    )
                else:
                    output[index] = p_front
                    confidence[index] = min(q_front, 0.55) * 0.65
                    sources.append("FRONT_CONFLICT_DOMINANT")
                valid[index] = True
                continue

            mirror_share = q_mirror / max(q_front + q_mirror, 1e-6)
            mirror_share = min(mirror_share, float(self._mirror_share_caps[index]))
            front_share = 1.0 - mirror_share
            output[index] = front_share * p_front + mirror_share * p_mirror
            confidence[index] = float(
                np.clip(front_share * q_front + mirror_share * q_mirror, 0, 1)
            )
            valid[index] = True

            if mirror_share >= 0.65:
                sources.append("MIRROR_DOMINANT")
            elif mirror_share >= 0.25:
                sources.append("FUSED")
            else:
                sources.append("FRONT_DOMINANT")

        valid &= np.isfinite(output).all(axis=1)
        fusion_quality = float(np.mean(confidence[valid])) if valid.any() else 0.0
        result = FusedPose(
            landmarks_3d=output,
            joint_confidence=confidence,
            joint_sources=tuple(sources),
            joint_valid=valid,
            alignment_quality=float(alignment.quality),
            alignment_ready=True,
            fusion_quality=fusion_quality,
            front_landmarks_3d=front.raw_landmarks_3d,
            mirror_landmarks_3d=mirror.raw_landmarks_3d,
            aligned_mirror_landmarks_3d=aligned_mirror,
            frame_id=front.frame_id,
            timestamp=front.timestamp,
        )
        self._last_front = front_points.copy()
        self._last_result = result
        self._front_missing_frames = 0
        return result
