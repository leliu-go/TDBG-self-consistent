"""Energy diagnostics for projected flavor calculations."""
from __future__ import annotations

import numpy as np


def band_energy_meV_per_cell(evals: np.ndarray, occ: np.ndarray, weights: np.ndarray, A_M_A2: float) -> float:
    state = np.asarray(evals, dtype=float) * np.asarray(occ, dtype=float)
    weighted = state * np.asarray(weights, dtype=float)[None, :, None]
    return float(A_M_A2) * float(np.sum(weighted))
