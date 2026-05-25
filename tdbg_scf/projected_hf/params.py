"""Parameters for flavor-resolved projected Hartree-Fock runs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProjectedHFParams:
    exchange_model: str = "coulomb_formfactor"
    finite_q_hartree: bool = False
    uniform_layer_hartree: bool = True
    epsilon_r: float = 10.0
    eps_perp: float = 4.0
    gate_distance_nm: float | None = None
    temperature_K: float = 0.0
    kBT_meV: float = 0.0
    max_iter: int = 100
    tol_density: float = 1e-7
    tol_U_meV: float = 1e-5
    tol_self_energy_meV: float = 1e-5
    mixing: float = 0.2
    n_random_seeds: int = 8
    seed: int = 0
    filling_reference: str = "neutrality"
    n_ref_per_flavor: float | None = None
    g_contact_meV: float = 0.0
    d_layer_nm: float = 0.335
    D_sign: float = -1.0
    seed_bias_meV: float = 10.0
