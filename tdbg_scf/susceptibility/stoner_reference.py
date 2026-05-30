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

    def __post_init__(self) -> None:
        if self.sigma_shifted_f_meV is None:
            self.sigma_shifted_f_meV = np.asarray(self.sigma_f_meV, dtype=float)
        if self.sigma_full_f_meV is None:
            self.sigma_full_f_meV = np.asarray(self.sigma_shifted_f_meV, dtype=float)
        if self.mu_eff_spread_meV is None:
            self.mu_eff_spread_meV = float(self.mu_common_spread_meV)

    def to_record(self) -> dict:
        out = {
            "mu_common_meV": float(self.mu_common_meV),
            "mu_common_spread_meV": float(self.mu_common_spread_meV),
            "mu_eff_spread_meV": float(self.mu_eff_spread_meV),
            "u_cell_meV": float(self.u_cell_meV),
            "J_cell_meV": float(self.J_cell_meV),
        }
        for i, name in enumerate(self.flavor_names):
            out[f"nu_{name}"] = float(self.nu_f[i])
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


def make_stoner_reference(
    result,
    tables,
    params: StonerParams,
    A_M_A2: float,
    remove_common_shift: bool = True,
) -> StonerReference:
    """Convert an existing StonerResult into a reference state for chi(Q)."""

    nu_f = np.asarray(result.nu_f, dtype=float)
    mu_f = stoner_mu_by_flavor(result, tables)
    sigma_full = stoner_self_energy_shifts_meV(nu_f, params, A_M_A2, remove_common=False)
    sigma_shifted = sigma_full - float(np.mean(sigma_full)) if remove_common_shift else sigma_full.copy()
    mu_eff = mu_f + sigma_full

    return StonerReference(
        nu_f=nu_f,
        mu_f_meV=np.asarray(mu_f, dtype=float),
        sigma_f_meV=np.asarray(sigma_shifted, dtype=float),
        mu_common_meV=float(np.mean(mu_f + sigma_shifted)),
        mu_common_spread_meV=float(np.max(mu_eff) - np.min(mu_eff)),
        u_cell_meV=float(params.u0_meV_A2) / float(A_M_A2),
        J_cell_meV=float(params.JH_meV_A2) / float(A_M_A2),
        sigma_full_f_meV=np.asarray(sigma_full, dtype=float),
        sigma_shifted_f_meV=np.asarray(sigma_shifted, dtype=float),
        mu_eff_spread_meV=float(np.max(mu_eff) - np.min(mu_eff)),
    )
