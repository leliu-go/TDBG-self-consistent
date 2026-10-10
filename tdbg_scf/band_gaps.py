"""Indirect miniband gaps at the four-flavor filling boundaries.

Input spectra must contain every band of the continuum Hamiltonian, sorted
at each k, at the same layer potential U(n,D). These are Hartree band gaps
before Stoner flavor shifts, not charge-addition or post-Stoner gaps.
"""
from __future__ import annotations

import numpy as np


GAP_FILLINGS = {"nu0": 0, "nu_plus4": 4, "nu_minus4": -4}
GAP_SPECTRUM_SOURCE = "full_continuum_at_scf_U_before_stoner"
GAP_DEFINITION = "max(0, min_over_k_and_valleys(E_upper) - max_over_k_and_valleys(E_lower))"

STONER_GAP_FILLINGS = {
    ("nu0" if nu == 0 else f"nu_{'plus' if nu > 0 else 'minus'}{abs(nu)}"): nu
    for nu in range(-6, 7)
}
STONER_GAP_SPECTRUM_SOURCE = "full_continuum_at_scf_U_plus_stoner_self_energy"
STONER_GAP_DEFINITION = "max(0, min_over_k(E_sorted_4flavors[2*M+nu_ref]) - max_over_k(E_sorted_4flavors[2*M+nu_ref-1]))"
STONER_GAP_NUMERIC_KEYS = tuple(key for label in STONER_GAP_FILLINGS for key in (
    f"stoner_gap_{label}_meV", f"stoner_indirect_gap_{label}_meV",
    f"stoner_gap_{label}_lower_edge_meV", f"stoner_gap_{label}_upper_edge_meV",
))


def stoner_gap_analysis_metadata() -> dict:
    return {
        "stoner_gap_spectrum_source": STONER_GAP_SPECTRUM_SOURCE,
        "stoner_gap_definition": STONER_GAP_DEFINITION,
        "stoner_gap_flavor_count": 4,
        "stoner_gap_flavor_order": "K_up,Kp_up,K_down,Kp_down",
        "stoner_gap_sampling": "finite_2D_MBZ_grid",
        "stoner_gap_filling_labels": dict(STONER_GAP_FILLINGS),
        "stoner_gap_reference": "fixed_U_scf_and_self_energy_of_this_scan_point",
        "stoner_gap_equilibrium_interpretation": "use_scan_row_at_matching_integer_filling",
    }


def gap_analysis_metadata() -> dict:
    return {
        "gap_spectrum_source": GAP_SPECTRUM_SOURCE,
        "gap_definition": GAP_DEFINITION,
        "gap_spin_valley_degeneracy": 4,
        "gap_sampling": "finite_2D_MBZ_grid",
        "gap_filling_labels": dict(GAP_FILLINGS),
    }


def _validate_spectrum(evals: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(evals):
        raise ValueError(f"{name} must contain real eigenvalues")
    energy = np.asarray(evals, dtype=float)
    if energy.ndim != 2 or energy.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty 2-D (k, band) spectrum")
    if energy.shape[1] < 4 or energy.shape[1] % 2:
        raise ValueError(f"{name} needs an even complete band count >= 4")
    if not np.isfinite(energy).all():
        raise ValueError(f"{name} must contain finite eigenvalues")
    if np.any(np.diff(energy, axis=1) < 0):
        raise ValueError(f"{name} must be sorted by energy at every k")
    return energy


def four_flavor_band_gaps(evals_K: np.ndarray, evals_Kp: np.ndarray) -> dict[str, float]:
    """Return global indirect gaps at nu=0,+4,-4, with signed edge diagnostics.

    With M bands per valley, neutral filling occupies M/2 bands per flavor.
    Filling one additional band in each of four flavors gives nu=+4.
    Extrema include both valleys; a positive direct gap at each k alone is
    insufficient. No chemical potential, DOS width, or occupation threshold
    enters this definition. A gapless/touching/overlapping boundary returns 0.
    """
    K = _validate_spectrum(evals_K, "evals_K")
    Kp = _validate_spectrum(evals_Kp, "evals_Kp")
    if K.shape[1] != Kp.shape[1]:
        raise ValueError("The two valleys must have the same complete band count")
    center = K.shape[1] // 2
    result = {}
    for label, nu in GAP_FILLINGS.items():
        cut = center + nu // 4
        lower = float(max(K[:, cut - 1].max(), Kp[:, cut - 1].max()))
        upper = float(min(K[:, cut].min(), Kp[:, cut].min()))
        indirect = upper - lower
        result[f"gap_{label}_meV"] = max(0.0, indirect)
        result[f"indirect_gap_{label}_meV"] = indirect
        result[f"gap_{label}_lower_edge_meV"] = lower
        result[f"gap_{label}_upper_edge_meV"] = upper
    return result


def stoner_integer_band_gaps(evals_K: np.ndarray, evals_Kp: np.ndarray,
                             self_energy_f_meV: np.ndarray) -> dict[str, float]:
    """Indirect gaps of the frozen, flavor-split full mean-field spectrum.

    Each flavor has M full bands and M/2 occupied bands at neutral filling;
    an integer reference filling therefore cuts the merged spectrum after
    2*M + nu_ref bands. Flavor identities need not follow the same energy
    ordering at every k. This is a spectral gap at this point's fixed U and
    self-energies, not a charge-addition energy or a new solution at nu_ref.
    The caller must verify physical common-Fermi-level validity before using
    these diagnostics as equilibrium gaps.
    """
    K = _validate_spectrum(evals_K, "evals_K")
    Kp = _validate_spectrum(evals_Kp, "evals_Kp")
    if K.shape != Kp.shape:
        raise ValueError("Stoner flavor spectra require the same k mesh and complete band count")
    if np.iscomplexobj(self_energy_f_meV):
        raise ValueError("Stoner self-energies must be real")
    sigma = np.asarray(self_energy_f_meV, dtype=float)
    if sigma.shape != (4,) or not np.isfinite(sigma).all():
        raise ValueError("Stoner self-energies must be four finite numbers in flavor order")
    energies = np.concatenate([K + sigma[0], Kp + sigma[1], K + sigma[2], Kp + sigma[3]], axis=1)
    if not np.isfinite(energies).all():
        raise ValueError("Shifted Stoner flavor spectra must be finite")
    cuts = {label: 2 * K.shape[1] + nu for label, nu in STONER_GAP_FILLINGS.items()}
    if any(cut <= 0 or cut >= energies.shape[1] for cut in cuts.values()):
        raise ValueError("Complete spectrum does not enclose all requested integer gap boundaries")
    # Only these ranks are needed. In-place partition avoids a second full
    # copy and never sorts unrelated discrete k levels into a false DOS gap.
    energies.partition(sorted({index for cut in cuts.values() for index in (cut - 1, cut)}), axis=1)
    result = {}
    for label, cut in cuts.items():
        lower = float(energies[:, cut - 1].max())
        upper = float(energies[:, cut].min())
        indirect = upper - lower
        result.update({
            f"stoner_gap_{label}_meV": max(0., indirect),
            f"stoner_indirect_gap_{label}_meV": indirect,
            f"stoner_gap_{label}_lower_edge_meV": lower,
            f"stoner_gap_{label}_upper_edge_meV": upper,
        })
    return result
