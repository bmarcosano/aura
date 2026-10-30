"""Small, well-tested geometric primitives used by AURA."""

from __future__ import annotations

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the unsigned angle ABC in degrees; return NaN for invalid geometry.

    The points may be 2D or 3D, but they must all belong to the same coordinate
    system. A missing/degenerate vector is not interpreted as a zero-degree angle.
    """
    pa = np.asarray(a, dtype=np.float64).reshape(-1)
    pb = np.asarray(b, dtype=np.float64).reshape(-1)
    pc = np.asarray(c, dtype=np.float64).reshape(-1)
    if pa.shape != pb.shape or pb.shape != pc.shape or pa.size not in (2, 3):
        return float("nan")
    if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
        return float("nan")

    ba, bc = pa - pb, pc - pb
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-10:
        return float("nan")
    cosine = float(np.dot(ba, bc) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def normalize_vector(vector: np.ndarray, eps: float = 1e-10) -> np.ndarray | None:
    """Return a finite unit vector, or None when the input is degenerate."""
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return value / norm


def reprojection_error(observed: np.ndarray, projected: np.ndarray) -> float:
    """Return Euclidean pixel reprojection error for two 2D points."""
    a = np.asarray(observed, dtype=np.float64).reshape(2)
    b = np.asarray(projected, dtype=np.float64).reshape(2)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.linalg.norm(a - b))
