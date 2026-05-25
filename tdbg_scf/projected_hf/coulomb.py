"""Coulomb-kernel helpers for projected calculations."""
from __future__ import annotations

import numpy as np


def coulomb_2d_meV_A2(q_Ainv: np.ndarray | float, epsilon_r: float) -> np.ndarray:
    q = np.asarray(q_Ainv, dtype=float)
    if np.any(q <= 0.0):
        raise ValueError("q_Ainv must be positive")
    e2_meV_A = 14399.6454784255
    return 2.0 * np.pi * e2_meV_A / (float(epsilon_r) * q)
