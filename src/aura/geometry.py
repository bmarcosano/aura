"""geometry.py: Core biomechanical calculations using NumPy vectors."""

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates the angle (in degrees) between three 2D or 3D points.

    Point 'b' is the vertex (e.g., the knee, elbow, or shoulder).
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    # Vectors AB and BC
    ba = a - b
    bc = c - b

    # Product of vectors magnitudes
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)

    # Prevent division by zero safely without skewing normal values
    if norm_product < 1e-6:
        return 0.0

    # Dot product and norms to compute the angle's cosine
    cosine_angle = np.dot(ba, bc) / norm_product
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return float(np.degrees(angle))
