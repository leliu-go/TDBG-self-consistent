from __future__ import annotations

import numpy as np

from tdbg_scf.density import dos_at_mu_gaussian
from tdbg_scf.susceptibility.stoner_reference import equilibrium_mu_for_shifted_self_energy_meV


def robust_vhs_diagnostics(state, sigma_meV: float = 1.0) -> dict[str, float | str]:
    ref = state.stoner_reference
    sigma = np.asarray(ref.sigma_shifted_f_meV, dtype=float)
    mu_eq = equilibrium_mu_for_shifted_self_energy_meV(ref)
    evals_by_flavor = [
        np.asarray(state.evals_K_meV, dtype=float) + sigma[0],
        np.asarray(state.evals_Kp_meV, dtype=float) + sigma[1],
        np.asarray(state.evals_K_meV, dtype=float) + sigma[2],
        np.asarray(state.evals_Kp_meV, dtype=float) + sigma[3],
    ]
    dos_by_flavor = np.asarray(
        [dos_at_mu_gaussian(evals, state.weights, mu_eq, sigma_meV=float(sigma_meV), degeneracy=1) for evals in evals_by_flavor],
        dtype=float,
    )
    active = int(np.argmax(dos_by_flavor))
    return {
        "DOS_total_EF": float(np.sum(dos_by_flavor)),
        "DOS_active_flavor_EF": float(dos_by_flavor[active]),
        "DOS_active_flavor_index": active,
        "D_at_max_DOS_on_scan": np.nan,
        "vhs_status": "unresolved",
        "vhs_flavor": "",
        "vhs_band_index": -1,
        "vhs_kx_Ainv": np.nan,
        "vhs_ky_Ainv": np.nan,
        "E_vhs_meV": np.nan,
        "delta_E_vhs_meV": np.nan,
        "vhs_fit_residual_meV": np.nan,
    }
