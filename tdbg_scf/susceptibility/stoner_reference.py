from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdbg_scf.stoner.full_band import stoner_mu_by_flavor
from tdbg_scf.stoner.params import StonerParams


FLAVOR_NAMES = ("K_up", "Kp_up", "K_down", "Kp_down")


@dataclass
class StonerReference:
    """Stoner-after reference data used by finite-Q susceptibility."""

    nu_f: np.ndarray
    mu_f_meV: np.ndarray
    sigma_f_meV: np.ndarray
    mu_common_meV: float
    mu_common_spread_meV: float
    u_cell_meV: float
    J_cell_meV: float
    flavor_names: tuple[str, str, str, str] = FLAVOR_NAMES
    sigma_full_f_meV: np.ndarray | None = None
    sigma_shifted_f_meV: np.ndarray | None = None
    mu_eff_spread_meV: float | None = None
    mu_eq_meV: float | None = None
    nu_f_reconstructed_common_mu: np.ndarray | None = None
    max_abs_nu_f_mismatch: float | None = None
    total_nu_mismatch: float | None = None
    reference_valid: bool = False
    reference_warning: str = ""
    soft_transverse_channel: str = "degenerate"

    def __post_init__(self) -> None:
        if self.sigma_shifted_f_meV is None:
            self.sigma_shifted_f_meV = np.asarray(self.sigma_f_meV, dtype=float)
        if self.sigma_full_f_meV is None:
            self.sigma_full_f_meV = np.asarray(self.sigma_shifted_f_meV, dtype=float)
        if self.mu_eff_spread_meV is None:
            self.mu_eff_spread_meV = float(self.mu_common_spread_meV)
        if self.mu_eq_meV is None:
            self.mu_eq_meV = float(self.mu_common_meV)
        if self.nu_f_reconstructed_common_mu is None:
            self.nu_f_reconstructed_common_mu = np.asarray(self.nu_f, dtype=float)
        if self.max_abs_nu_f_mismatch is None:
            self.max_abs_nu_f_mismatch = float(np.max(np.abs(np.asarray(self.nu_f_reconstructed_common_mu) - np.asarray(self.nu_f))))
        if self.total_nu_mismatch is None:
            self.total_nu_mismatch = float(np.sum(self.nu_f_reconstructed_common_mu) - np.sum(self.nu_f))

    def to_record(self) -> dict:
        out = {
            "mu_common_meV": float(self.mu_common_meV),
            "mu_eq_meV": float(self.mu_eq_meV),
            "mu_common_spread_meV": float(self.mu_common_spread_meV),
            "mu_eff_spread_meV": float(self.mu_eff_spread_meV),
            "max_abs_nu_f_mismatch": float(self.max_abs_nu_f_mismatch),
            "total_nu_mismatch": float(self.total_nu_mismatch),
            "reference_valid": bool(self.reference_valid),
            "reference_warning": str(self.reference_warning),
            "soft_transverse_channel": str(self.soft_transverse_channel),
            "u_cell_meV": float(self.u_cell_meV),
            "J_cell_meV": float(self.J_cell_meV),
        }
        for i, name in enumerate(self.flavor_names):
            out[f"nu_{name}"] = float(self.nu_f[i])
            out[f"nu_reconstructed_common_mu_{name}"] = float(self.nu_f_reconstructed_common_mu[i])
            out[f"mu_{name}_meV"] = float(self.mu_f_meV[i])
            out[f"sigma_{name}_meV"] = float(self.sigma_shifted_f_meV[i])
            out[f"sigma_shifted_{name}_meV"] = float(self.sigma_shifted_f_meV[i])
            out[f"sigma_full_{name}_meV"] = float(self.sigma_full_f_meV[i])
            out[f"mu_plus_sigma_{name}_meV"] = float(self.mu_f_meV[i] + self.sigma_shifted_f_meV[i])
            out[f"mu_eff_{name}_meV"] = float(self.mu_f_meV[i] + self.sigma_full_f_meV[i])
        return out


def stoner_self_energy_shifts_meV(
    nu_f: np.ndarray,
    params: StonerParams,
    A_M_A2: float,
    remove_common: bool = True,
) -> np.ndarray:
    """Return flavor-dependent Stoner shifts Sigma_f = dE_int/dnu_f."""

    nu = np.asarray(nu_f, dtype=float)
    if nu.shape != (4,):
        raise ValueError("nu_f must have shape (4,)")

    u_cell = float(params.u0_meV_A2) / float(A_M_A2)
    J_cell = float(params.JH_meV_A2) / float(A_M_A2)
    total = float(np.sum(nu))

    sigma = u_cell * (total - nu)
    m_K = float(nu[0] - nu[2])
    m_Kp = float(nu[1] - nu[3])
    sigma[0] += -J_cell * m_Kp
    sigma[2] += +J_cell * m_Kp
    sigma[1] += -J_cell * m_K
    sigma[3] += +J_cell * m_K

    if remove_common:
        sigma = sigma - float(np.mean(sigma))
    return np.asarray(sigma, dtype=float)


def soft_transverse_channel(nu_f: np.ndarray, spin_channel_tol: float = 1e-8) -> str:
    nu = np.asarray(nu_f, dtype=float)
    if nu.shape != (4,):
        raise ValueError("nu_f must have shape (4,)")
    m_spin = float((nu[0] + nu[1]) - (nu[2] + nu[3]))
    if m_spin > float(spin_channel_tol):
        return "minus"
    if m_spin < -float(spin_channel_tol):
        return "plus"
    return "degenerate"


def equilibrium_mu_for_shifted_self_energy_meV(ref: StonerReference) -> float:
    sigma_full = np.asarray(ref.sigma_full_f_meV, dtype=float)
    sigma_shifted = np.asarray(ref.sigma_shifted_f_meV, dtype=float)
    common_shift = float(np.mean(sigma_full - sigma_shifted))
    return float(ref.mu_eq_meV) - common_shift


def _nu_of_mu_zero_temperature(table, mu_meV: float) -> float:
    mu = float(mu_meV)
    lo_mu = table.mu_of_nu(float(table.nu_min))
    hi_mu = table.mu_of_nu(float(table.nu_max))
    if mu <= lo_mu:
        return float(table.nu_min)
    if mu >= hi_mu:
        return float(table.nu_max)
    return float(np.interp(mu, table.energy_grid_meV, table.nu_grid))


def _solve_common_mu_for_tables(tables, sigma_full: np.ndarray, nu_total: float) -> tuple[float, np.ndarray]:
    sigma = np.asarray(sigma_full, dtype=float)

    def fillings(mu_eq: float) -> np.ndarray:
        return np.asarray([_nu_of_mu_zero_temperature(table, float(mu_eq) - sigma[i]) for i, table in enumerate(tables)], dtype=float)

    lo = min(table.mu_of_nu(table.nu_min) + sigma[i] for i, table in enumerate(tables)) - 1.0
    hi = max(table.mu_of_nu(table.nu_max) + sigma[i] for i, table in enumerate(tables)) + 1.0
    target = float(nu_total)
    f_lo = float(np.sum(fillings(lo)) - target)
    f_hi = float(np.sum(fillings(hi)) - target)
    expand = 0
    while (f_lo > 0.0 or f_hi < 0.0) and expand < 64:
        width = hi - lo
        lo -= width
        hi += width
        f_lo = float(np.sum(fillings(lo)) - target)
        f_hi = float(np.sum(fillings(hi)) - target)
        expand += 1
    if f_lo > 0.0 or f_hi < 0.0:
        raise RuntimeError("common-mu reconstruction bracket failed")

    for _ in range(160):
        mid = 0.5 * (lo + hi)
        f_mid = float(np.sum(fillings(mid)) - target)
        if abs(f_mid) < 1e-12:
            lo = hi = mid
            break
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    mu_eq = 0.5 * (lo + hi)
    return float(mu_eq), fillings(mu_eq)


def make_stoner_reference(
    result,
    tables,
    params: StonerParams,
    A_M_A2: float,
    remove_common_shift: bool = True,
    mu_eff_spread_warn_meV: float = 0.02,
    mu_eff_spread_fail_meV: float = 0.10,
    filling_mismatch_warn: float = 2e-3,
    filling_mismatch_fail: float = 1e-2,
) -> StonerReference:
    """Convert an existing StonerResult into a reference state for chi(Q)."""

    nu_f = np.asarray(result.nu_f, dtype=float)
    mu_f = stoner_mu_by_flavor(result, tables)
    sigma_full = stoner_self_energy_shifts_meV(nu_f, params, A_M_A2, remove_common=False)
    sigma_shifted = sigma_full - float(np.mean(sigma_full)) if remove_common_shift else sigma_full.copy()
    mu_eff = mu_f + sigma_full
    mu_eq, nu_recon = _solve_common_mu_for_tables(tables, sigma_full, float(result.nu_total))
    mismatch = np.asarray(nu_recon, dtype=float) - nu_f
    max_abs_mismatch = float(np.max(np.abs(mismatch)))
    total_mismatch = float(np.sum(nu_recon) - float(result.nu_total))
    mu_eff_spread = float(np.max(mu_eff) - np.min(mu_eff))
    warnings = []
    if mu_eff_spread > float(mu_eff_spread_warn_meV):
        warnings.append(f"mu_eff_spread_meV={mu_eff_spread:.6g}")
    if max_abs_mismatch > float(filling_mismatch_warn):
        warnings.append(f"max_abs_nu_f_mismatch={max_abs_mismatch:.6g}")
    if abs(total_mismatch) > float(filling_mismatch_warn):
        warnings.append(f"total_nu_mismatch={total_mismatch:.6g}")
    reference_valid = bool(
        mu_eff_spread <= float(mu_eff_spread_fail_meV)
        and max_abs_mismatch <= float(filling_mismatch_fail)
        and abs(total_mismatch) <= float(filling_mismatch_fail)
    )

    return StonerReference(
        nu_f=nu_f,
        mu_f_meV=np.asarray(mu_f, dtype=float),
        sigma_f_meV=np.asarray(sigma_shifted, dtype=float),
        mu_common_meV=float(np.mean(mu_f + sigma_shifted)),
        mu_common_spread_meV=mu_eff_spread,
        u_cell_meV=float(params.u0_meV_A2) / float(A_M_A2),
        J_cell_meV=float(params.JH_meV_A2) / float(A_M_A2),
        sigma_full_f_meV=np.asarray(sigma_full, dtype=float),
        sigma_shifted_f_meV=np.asarray(sigma_shifted, dtype=float),
        mu_eff_spread_meV=mu_eff_spread,
        mu_eq_meV=mu_eq,
        nu_f_reconstructed_common_mu=nu_recon,
        max_abs_nu_f_mismatch=max_abs_mismatch,
        total_nu_mismatch=total_mismatch,
        reference_valid=reference_valid,
        reference_warning="; ".join(warnings),
        soft_transverse_channel=soft_transverse_channel(nu_f),
    )
