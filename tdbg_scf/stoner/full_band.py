"""Helpers for Stoner calculations using full continuum bands."""
from __future__ import annotations

import numpy as np
from .active_polarization import active_result_fields

from ..density import dos_at_mu_gaussian, layer_weights_from_evecs, validate_gaussian_sigma
from ..density import find_mu_for_density, full_layer_density
from ..filling import density_cm2_to_filling
from .dos import FlavorBandTable, _sorted_weighted_states, _cumulative_integrals, _stable_cumsum
from .energy import stoner_stationarity_intervals


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

    es, ws = _sorted_weighted_states(energies_meV, weights, A_M_A2, energy_zero_meV)
    nu_abs, _ = _cumulative_integrals(es, ws)
    energy_grid = np.concatenate([[es[0]], es])

    ref_abs = .5 * nu_abs[-1]
    ref_index = int(np.searchsorted(nu_abs, ref_abs, side="left"))
    fraction = (ref_abs-nu_abs[ref_index-1]) / (nu_abs[ref_index]-nu_abs[ref_index-1])
    nu_grid = np.asarray(nu_abs - ref_abs, dtype=float)
    # Integrate outwards from neutrality instead of subtracting two large
    # filled-sea energies. This preserves small slopes near the Fermi bands.
    increments = ws*es
    kinetic_grid = np.empty(len(increments)+1, dtype=float)
    kinetic_grid[:ref_index] = (-fraction*increments[ref_index-1]
                               -np.r_[0.,_stable_cumsum(increments[:ref_index-1][::-1])][::-1])
    kinetic_grid[ref_index:] = ((1-fraction)*increments[ref_index-1]
                               +np.r_[0.,_stable_cumsum(increments[ref_index:])])

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


def density_cm2_to_nu_total(n_cm2: float, A_M_A2: float) -> float:
    return density_cm2_to_filling(float(n_cm2), float(A_M_A2))


def stoner_mu_by_flavor(result, tables: list[FlavorBandTable]) -> np.ndarray:
    if getattr(result, "mu_f_meV", None) is not None:
        return np.asarray(result.mu_f_meV, dtype=float)
    return np.asarray([tables[f].mu_of_nu(float(result.nu_f[f])) for f in range(4)], dtype=float)



def stoner_mean_field_fermi_level(result, tables, params, A_M_A2,
                                 mu_tolerance_meV: float = 1e-4) -> dict:
    """Find one Fermi level for the existing constant-flavor Stoner shifts.

    Sigma_f is the unconstrained derivative of the interaction energy.
    Intersect K'_f(nu_f)+Sigma_f using the full physical spectral intervals,
    not the optimization-bound normal cones. No filling
    or energy is changed. An empty intersection cannot certify physical DOS.
    """
    check = stoner_stationarity_intervals(result.nu_f, tables, params, A_M_A2, mu_tolerance_meV)
    sigma = np.asarray(check["self_energy_f_meV"])
    lo,hi = check["mu_interval_lower_meV"],check["mu_interval_upper_meV"]
    valid,optimization_valid = check["physical_valid"],check["optimization_valid"]
    if valid:
        common = .5*(lo+hi) if np.isfinite([lo,hi]).all() else (hi if np.isfinite(hi) else lo)
        if not np.isfinite(common):
            common = 0.
        mu_f = common-sigma
    else:
        common, mu_f = float("nan"), np.full(4, np.nan)
    return {**check, "valid": valid, "mu_common_meV": float(common), "mu_f_meV": mu_f.tolist(),
            "reason": ("ok" if valid else "optimizer_kkt_mismatch" if not optimization_valid
                       else "filling_bound_limited" if check["filling_bound_limited"]
                       else "occupation_intervals_incompatible"),
            "convention": "common_mean_field_fermi_level_with_constant_flavor_shifts"}


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

    if not np.isfinite(A_M_A2) or A_M_A2 <= 0. or not np.isfinite(nu):
        raise ValueError("A_M_A2 must be finite and positive; nu must be finite")
    target = float(nu)/float(A_M_A2)
    mu = find_mu_for_density(energies, w, target, degeneracy=1, kBT_meV=0.)
    # Reuse the Hartree occupation rule: tied states receive an equal
    # fractional occupation, independent of the eigensolver's basis.
    return float(A_M_A2)*full_layer_density(
        mu, energies, evecs, w, layer_masks, degeneracy=1, kBT_meV=0.,
        target_density_a2=target)


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
    sigma = validate_gaussian_sigma(sigma_meV)
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
        **active_result_fields(result),
        "nu_total": float(result.nu_total),
        "nu_f": result.nu_f.tolist(),
        "flavor_bounds_nu": result.flavor_bounds_nu,
        "mu_f_meV": mu_f.tolist(),
        "mean_field_fermi_level": getattr(result, "mean_field_fermi_level", None),
        "mu_convention": ("common_mean_field_fermi_level_with_constant_flavor_shifts"
                          if getattr(result, "mu_f_meV", None) is not None else
                          "independent_flavor_filling_quantiles"),
        "energy_meV_per_cell": float(result.energy_meV_per_cell),
        "success": bool(result.success),
        "seed_name": str(result.seed_name),
        "minimum_validation": getattr(result, "minimum_validation", None),
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
                **{key: item[key] for key in (
                    "stationarity", "local_stability", "message",
                    "optimizer_iterations", "refinement_iterations",
                    "total_iterations", "energy_decrease_meV",
                    "de_seed", "de_converged", "de_function_evaluations",
                    "de_message", "stage", "refinement_budget_exhausted",
                ) if key in item},
            }
            for item in result.local_minima
        ],
    }
