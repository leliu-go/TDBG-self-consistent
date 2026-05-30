from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SusceptibilityParams:
    """Numerical controls for finite-Q transverse susceptibility.

    The susceptibility is reported per moire unit cell per meV.
    """

    kBT_meV: float = 0.05
    denom_tol_meV: float = 1e-8
    energy_window_meV: float | None = 30.0
    max_bands_per_k: int | None = 24
    q_mode: str = "folded_grid"
    main_vertex_model: str = "su4_diag"
    also_run_hund_factor2: bool = True
    hund_transverse_factor: float = 2.0
    occupation_mode: str = "flavor_mu"
    spin_flip_mode: str = "both_pm"
    legacy_diagnostics: bool = True
    include_layer_matrix: bool = True
    hermitize_layer_matrix: bool = True
    layer_dipole_zeta: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5)
    finite_q_tol: float = 1e-3
