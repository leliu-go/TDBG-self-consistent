"""Initial-state helpers for projected flavor calculations."""
from __future__ import annotations

import numpy as np


def flavor_filling_seeds(nu_total: float, n_random: int = 0, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    seeds = [("balanced", np.full(4, float(nu_total) / 4.0))]
    for flavor in range(4):
        x = np.zeros(4)
        x[flavor] = float(nu_total)
        seeds.append((f"flavor_{flavor}", x))
    for index in range(int(n_random)):
        seeds.append((f"random_{index}", rng.dirichlet(np.ones(4)) * float(nu_total)))
    return seeds
