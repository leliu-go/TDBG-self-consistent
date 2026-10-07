"""Parameters for the phenomenological four-flavor solver."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StonerParams:
    u0_meV_A2: float
    JH_meV_A2: float = 0.0
    temperature_K: float = 0.0
    max_iter: int = 200
    tol_energy_meV: float = 1e-8
    n_random_seeds: int = 20
    seed: int = 0
    flavor_nu_min: float = -2.0
    flavor_nu_max: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.u0_meV_A2, self.JH_meV_A2,
                            self.temperature_K, self.tol_energy_meV,
                            self.flavor_nu_min, self.flavor_nu_max]).all():
            raise ValueError("Stoner parameters must be finite")
        if self.flavor_nu_min >= self.flavor_nu_max:
            raise ValueError("flavor filling bounds require flavor_nu_min < flavor_nu_max")
        if self.temperature_K != 0:
            raise ValueError("Stoner currently supports temperature_K=0 only; finite-temperature free energy is not implemented")
        if self.tol_energy_meV <= 0:
            raise ValueError("tol_energy_meV must be positive")
        for name, value, lower in (("max_iter", self.max_iter, 1),
                                   ("n_random_seeds", self.n_random_seeds, 0),
                                   ("seed", self.seed, 0)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
