"""Physical constants and unit conversions used by the TDBG SCF code."""
from __future__ import annotations

E_CHARGE_C = 1.602176634e-19
EPS0_SI = 8.8541878128e-12
MEV_TO_J = 1.602176634e-22
ANGSTROM_TO_M = 1e-10
NM_TO_M = 1e-9


def cm2_to_a2(n_cm2: float) -> float:
    """Convert a density from cm^{-2} to Angstrom^{-2}."""
    return n_cm2 * 1.0e-16


def a2_to_cm2(n_a2: float) -> float:
    """Convert a density from Angstrom^{-2} to cm^{-2}."""
    return n_a2 / 1.0e-16


def cm2_to_nm2(n_cm2: float) -> float:
    return n_cm2 * 1.0e-14


def nm2_to_cm2(n_nm2: float) -> float:
    return n_nm2 / 1.0e-14


def D_Vnm_to_gate_density_a2(D_Vnm: float) -> float:
    """Return delta n = 2 eps0 D / e in Angstrom^{-2}."""
    D_Vm = D_Vnm * 1.0e9
    delta_m2 = 2.0 * EPS0_SI * D_Vm / E_CHARGE_C
    return delta_m2 / 1.0e20


def gate_densities_from_n_D_a2(n_total_a2: float, D_Vnm: float) -> tuple[float, float]:
    """
    Gate densities in the convention used for the Gauss-law update:
        n = sum_l n_l = -(n_t + n_b),
        D = e (n_b - n_t) / (2 eps0).
    Returns (n_t, n_b) in Angstrom^{-2}.
    """
    delta = D_Vnm_to_gate_density_a2(D_Vnm)
    n_b = (-n_total_a2 + delta) / 2.0
    n_t = (-n_total_a2 - delta) / 2.0
    return n_t, n_b


def gauss_coeff_meV_per_a2(d_layer_nm: float = 0.335, eps_perp: float = 4.0) -> float:
    """
    Coefficient C such that
        Delta U [meV] = - C * sheet_density [Angstrom^{-2}].
    """
    d_m = d_layer_nm * NM_TO_M
    coeff_J_per_a2 = E_CHARGE_C**2 * d_m * 1.0e20 / (eps_perp * EPS0_SI)
    return coeff_J_per_a2 / MEV_TO_J
