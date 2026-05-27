"""Helpers for Stoner calculations using full continuum bands."""
from __future__ import annotations

import numpy as np

from ..density import dos_at_mu_gaussian, layer_weights_from_evecs
from ..filling import density_cm2_to_filling
from .dos import FlavorBandTable


def build_neutrality_referenced_band_table(
    energies_meV: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    energy_zero_meV: float = 0.0,
) -> FlavorBandTable:
    """Build a signed flavor table with ``nu=0`` at half filling.

    This matches the existing full-band density convention
    ``sum_kb weight_k * (occ_kb - 0.5)`` and avoids treating the filled
    sea as a huge positive Stoner density.
    """

    energies = np.asarray(energies_meV, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if energies.ndim != 2:
        raise ValueError("energies_meV must have shape (Nk, Nb)")
    if weights.shape != (energies.shape[0],):
        raise ValueError("weights must have shape (Nk,)")

    Nk, Nb = energies.shape
    state_e = energies.reshape(-1) - float(energy_zero_meV)
    state_w = np.repeat(weights, Nb) * float(A_M_A2)
    if np.any(state_w < 0.0):
        raise ValueError("weights must be non-negative")

    order = np.argsort(state_e, kind="mergesort")
    es = state_e[order]
    ws = state_w[order]
    nu_abs = np.concatenate([[0.0], np.cumsum(ws)])
    kinetic_abs = np.concatenate([[0.0], np.cumsum(ws * es)])
    energy_grid = np.concatenate([[es[0]], es])

    ref_abs = 0.5 * float(nu_abs[-1])
    kinetic_ref = float(np.interp(ref_abs, nu_abs, kinetic_abs))
    nu_grid = nu_abs - ref_abs
    kinetic_grid = kinetic_abs - kinetic_ref

    return FlavorBandTable(
        nu_grid=nu_grid,
        energy_grid_meV=energy_grid,
        kinetic_grid_meV=kinetic_grid,
        nu_min=float(nu_grid[0]),
        nu_max=float(nu_grid[-1]),
    )


def build_full_band_flavor_tables(
    evals_K_meV: np.ndarray,
    evals_Kp_meV: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
) -> list[FlavorBandTable]:
    """Return tables in flavor order K_up, Kp_up, K_down, Kp_down."""

    table_K = build_neutrality_referenced_band_table(evals_K_meV, weights, A_M_A2)
    table_Kp = build_neutrality_referenced_band_table(evals_Kp_meV, weights, A_M_A2)
    return [table_K, table_Kp, table_K, table_Kp]


def restrict_tables_to_carrier_sector(tables: list[FlavorBandTable], nu_total: float) -> list[FlavorBandTable]:
    """Restrict signed full-band tables to the carrier sector being doped.

    Positive total filling means added electrons only, so each flavor is
    constrained to ``[0, nu_max]``. Negative total filling means added holes
    only, so each flavor is constrained to ``[nu_min, 0]``. This prevents
    unphysical compensated electron-hole pair creation across flavors.
    """

    restricted = []
    for table in tables:
        if nu_total > 0.0:
            nu_min = max(0.0, float(table.nu_min))
            nu_max = float(table.nu_max)
        elif nu_total < 0.0:
            nu_min = float(table.nu_min)
            nu_max = min(0.0, float(table.nu_max))
        else:
            nu_min = 0.0
            nu_max = 0.0
        if nu_min > nu_max + 1e-12:
            raise ValueError("carrier-sector restriction leaves no available states")
        restricted.append(
            FlavorBandTable(
                nu_grid=table.nu_grid,
                energy_grid_meV=table.energy_grid_meV,
                kinetic_grid_meV=table.kinetic_grid_meV,
                nu_min=nu_min,
                nu_max=nu_max,
            )
        )
    return restricted


def density_cm2_to_nu_total(n_cm2: float, A_M_A2: float) -> float:
    return density_cm2_to_filling(float(n_cm2), float(A_M_A2))


def stoner_mu_by_flavor(result, tables: list[FlavorBandTable]) -> np.ndarray:
    return np.asarray([tables[f].mu_of_nu(float(result.nu_f[f])) for f in range(4)], dtype=float)


def explicit_flavor_dos_at_mu(
    evals_K_meV: np.ndarray,
    evals_Kp_meV: np.ndarray,
    weights: np.ndarray,
    mu_f_meV: np.ndarray,
    sigma_meV: float,
) -> float:
    return float(
        dos_at_mu_gaussian(evals_K_meV, weights, float(mu_f_meV[0]), sigma_meV=sigma_meV, degeneracy=1)
        + dos_at_mu_gaussian(evals_Kp_meV, weights, float(mu_f_meV[1]), sigma_meV=sigma_meV, degeneracy=1)
        + dos_at_mu_gaussian(evals_K_meV, weights, float(mu_f_meV[2]), sigma_meV=sigma_meV, degeneracy=1)
        + dos_at_mu_gaussian(evals_Kp_meV, weights, float(mu_f_meV[3]), sigma_meV=sigma_meV, degeneracy=1)
    )


def layer_filling_from_nu(
    energies_meV: np.ndarray,
    evecs: np.ndarray,
    weights: np.ndarray,
    layer_masks: np.ndarray,
    A_M_A2: float,
    nu: float,
) -> np.ndarray:
    """Layer-resolved filling for one flavor at signed filling ``nu``.

    The returned values are moire fillings per layer, using the same
    neutrality convention as the full-band SCF density: occupied states minus
    one half of every state.
    """

    energies = np.asarray(energies_meV, dtype=float)
    w = np.asarray(weights, dtype=float)
    if energies.ndim != 2:
        raise ValueError("energies_meV must have shape (Nk, Nb)")
    if w.shape != (energies.shape[0],):
        raise ValueError("weights must have shape (Nk,)")

    W = layer_weights_from_evecs(evecs, layer_masks)
    if W.shape[:2] != energies.shape:
        raise ValueError("evecs/layer_masks must produce layer weights matching energies")

    Nk, Nb = energies.shape
    state_e = energies.reshape(-1)
    state_dnu = np.repeat(w, Nb) * float(A_M_A2)
    state_layer_dnu = W.reshape(-1, W.shape[-1]) * state_dnu[:, None]
    total_capacity = float(np.sum(state_dnu))
    target_abs = 0.5 * total_capacity + float(nu)
    if target_abs < -1e-10 or target_abs > total_capacity + 1e-10:
        raise ValueError("nu outside available full-band capacity")
    target_abs = float(np.clip(target_abs, 0.0, total_capacity))

    order = np.argsort(state_e, kind="mergesort")
    dnu_sorted = state_dnu[order]
    layer_sorted = state_layer_dnu[order]
    cum_before = np.concatenate([[0.0], np.cumsum(dnu_sorted[:-1])])
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.clip((target_abs - cum_before) / dnu_sorted, 0.0, 1.0)
    fraction = np.where(dnu_sorted > 0.0, fraction, 0.0)

    occupied_layer = np.sum(layer_sorted * fraction[:, None], axis=0)
    neutral_reference = 0.5 * np.sum(state_layer_dnu, axis=0)
    return np.real(occupied_layer - neutral_reference)


def stoner_flavor_layer_fillings(
    evals_K_meV: np.ndarray,
    evecs_K: np.ndarray,
    evals_Kp_meV: np.ndarray,
    evecs_Kp: np.ndarray,
    weights: np.ndarray,
    layer_masks: np.ndarray,
    A_M_A2: float,
    nu_f: np.ndarray,
) -> np.ndarray:
    """Layer fillings for flavors ordered as K_up, Kp_up, K_down, Kp_down."""

    values = np.asarray(nu_f, dtype=float)
    if values.shape != (4,):
        raise ValueError("nu_f must have shape (4,)")
    return np.stack(
        [
            layer_filling_from_nu(evals_K_meV, evecs_K, weights, layer_masks, A_M_A2, values[0]),
            layer_filling_from_nu(evals_Kp_meV, evecs_Kp, weights, layer_masks, A_M_A2, values[1]),
            layer_filling_from_nu(evals_K_meV, evecs_K, weights, layer_masks, A_M_A2, values[2]),
            layer_filling_from_nu(evals_Kp_meV, evecs_Kp, weights, layer_masks, A_M_A2, values[3]),
        ],
        axis=0,
    )


def stoner_layer_fillings(
    evals_K_meV: np.ndarray,
    evecs_K: np.ndarray,
    evals_Kp_meV: np.ndarray,
    evecs_Kp: np.ndarray,
    weights: np.ndarray,
    layer_masks: np.ndarray,
    A_M_A2: float,
    nu_f: np.ndarray,
) -> np.ndarray:
    """Total Stoner layer filling summed over K/K' and spin flavors."""

    return np.sum(
        stoner_flavor_layer_fillings(
            evals_K_meV,
            evecs_K,
            evals_Kp_meV,
            evecs_Kp,
            weights,
            layer_masks,
            A_M_A2,
            nu_f,
        ),
        axis=0,
    )


def layer_dos_at_mu_gaussian(
    evals_meV: np.ndarray,
    evecs: np.ndarray,
    weights: np.ndarray,
    layer_masks: np.ndarray,
    mu_meV: float,
    sigma_meV: float,
) -> np.ndarray:
    """Single-flavor Gaussian DOS at ``mu`` projected onto layers."""

    evals = np.asarray(evals_meV, dtype=float)
    W = layer_weights_from_evecs(evecs, layer_masks)
    if W.shape[:2] != evals.shape:
        raise ValueError("evecs/layer_masks must produce layer weights matching evals")
    sigma = float(sigma_meV)
    x = (evals - float(mu_meV)) / sigma
    kernel = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sigma)
    return np.real(np.einsum("k,kb,kbl->l", weights, kernel, W, optimize=True))


def stoner_flavor_layer_dos_at_mu(
    evals_K_meV: np.ndarray,
    evecs_K: np.ndarray,
    evals_Kp_meV: np.ndarray,
    evecs_Kp: np.ndarray,
    weights: np.ndarray,
    layer_masks: np.ndarray,
    mu_f_meV: np.ndarray,
    sigma_meV: float,
) -> np.ndarray:
    """Layer DOS for flavors ordered as K_up, Kp_up, K_down, Kp_down."""

    mu = np.asarray(mu_f_meV, dtype=float)
    if mu.shape != (4,):
        raise ValueError("mu_f_meV must have shape (4,)")
    return np.stack(
        [
            layer_dos_at_mu_gaussian(evals_K_meV, evecs_K, weights, layer_masks, mu[0], sigma_meV),
            layer_dos_at_mu_gaussian(evals_Kp_meV, evecs_Kp, weights, layer_masks, mu[1], sigma_meV),
            layer_dos_at_mu_gaussian(evals_K_meV, evecs_K, weights, layer_masks, mu[2], sigma_meV),
            layer_dos_at_mu_gaussian(evals_Kp_meV, evecs_Kp, weights, layer_masks, mu[3], sigma_meV),
        ],
        axis=0,
    )


def stoner_layer_dos_at_mu(
    evals_K_meV: np.ndarray,
    evecs_K: np.ndarray,
    evals_Kp_meV: np.ndarray,
    evecs_Kp: np.ndarray,
    weights: np.ndarray,
    layer_masks: np.ndarray,
    mu_f_meV: np.ndarray,
    sigma_meV: float,
) -> np.ndarray:
    """Layer-resolved explicit-flavor DOS evaluated at Stoner flavor chemical potentials."""

    return np.sum(
        stoner_flavor_layer_dos_at_mu(
            evals_K_meV,
            evecs_K,
            evals_Kp_meV,
            evecs_Kp,
            weights,
            layer_masks,
            mu_f_meV,
            sigma_meV,
        ),
        axis=0,
    )


def layer_polarizations(n_layer_cm2: np.ndarray) -> dict[str, float]:
    n1, n2, n3, n4 = [float(x) for x in n_layer_cm2]
    return {
        "P_top_bottom_cm2": (n1 + n2) - (n3 + n4),
        "P_outer_inner_cm2": (n1 + n4) - (n2 + n3),
        "P_dipole_cm2": 1.5 * n1 + 0.5 * n2 - 0.5 * n3 - 1.5 * n4,
    }


def stoner_result_to_dict(result, tables: list[FlavorBandTable]) -> dict:
    mu_f = stoner_mu_by_flavor(result, tables)
    return {
        "nu_total": float(result.nu_total),
        "nu_f": result.nu_f.tolist(),
        "mu_f_meV": mu_f.tolist(),
        "energy_meV_per_cell": float(result.energy_meV_per_cell),
        "success": bool(result.success),
        "seed_name": str(result.seed_name),
        "spin_polarization": result.spin_polarization,
        "spin_polarization_norm": result.spin_polarization_norm,
        "valley_polarization": result.valley_polarization,
        "valley_polarization_norm": result.valley_polarization_norm,
        "spin_valley_polarization": result.spin_valley_polarization,
        "spin_valley_polarization_norm": result.spin_valley_polarization_norm,
        "flavor_polarization": result.flavor_polarization,
        "local_minima": [
            {
                "seed_name": item["seed_name"],
                "energy_meV_per_cell": float(item["energy_meV_per_cell"]),
                "success": bool(item["success"]),
                "nu_f": np.asarray(item["nu_f"], dtype=float).tolist(),
            }
            for item in result.local_minima
        ],
    }
