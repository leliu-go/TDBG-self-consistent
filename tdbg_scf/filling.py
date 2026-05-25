"""Filling and density conversion helpers for moire unit cells."""
from __future__ import annotations

import numpy as np


def moire_cell_area_A2_from_weights(weights) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("sum(weights) must be positive")
    return 1.0 / total_weight


def filling_to_density_a2(nu: float, A_M_A2: float) -> float:
    return float(nu) / float(A_M_A2)


def filling_to_density_cm2(nu: float, A_M_A2: float) -> float:
    return filling_to_density_a2(nu, A_M_A2) * 1.0e16


def density_cm2_to_filling(n_cm2: float, A_M_A2: float) -> float:
    return float(n_cm2) * 1.0e-16 * float(A_M_A2)
