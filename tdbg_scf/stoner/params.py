"""Parameters for the phenomenological four-flavor solver."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StonerParams:
    u0_meV_A2: float
    JH_meV_A2: float = 0.0
    temperature_K: float = 0.0
    max_iter: int = 200
    tol_energy_meV: float = 1e-8
    n_random_seeds: int = 20
    seed: int = 0
