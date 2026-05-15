"""Density, chemical-potential, and layer-weight routines."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def fermi(E_minus_mu: np.ndarray, kBT_meV: float) -> np.ndarray:
    x = np.asarray(E_minus_mu, dtype=float) / max(kBT_meV, 1e-12)
    x = np.clip(x, -100.0, 100.0)
    return 1.0 / (np.exp(x) + 1.0)


def total_density_from_evals(mu: float, evals: np.ndarray, weights: np.ndarray, degeneracy: int, kBT_meV: float) -> float:
    """Total density relative to charge neutrality, in A^{-2}."""
    occ = fermi(evals - mu, kBT_meV)
    dim = evals.shape[1]
    return degeneracy * np.sum(weights[:, None] * (occ - 0.5))


def find_mu_for_density(evals: np.ndarray, weights: np.ndarray, target_density_a2: float,
                        degeneracy: int = 4, kBT_meV: float = 0.1,
                        extra_window_meV: float = 500.0) -> float:
    lo = float(np.min(evals) - extra_window_meV)
    hi = float(np.max(evals) + extra_window_meV)

    def fn(mu: float) -> float:
        return total_density_from_evals(mu, evals, weights, degeneracy, kBT_meV) - target_density_a2

    flo = fn(lo)
    fhi = fn(hi)
    if flo > 0 or fhi < 0:
        raise RuntimeError(
            f"Chemical-potential bracket failed: f(lo)={flo:.3e}, f(hi)={fhi:.3e}, "
            f"target={target_density_a2:.3e}. Enlarge the energy window or active space."
        )
    return float(brentq(fn, lo, hi, maxiter=200, xtol=1e-10))


def layer_weights_from_evecs(evecs: np.ndarray, layer_masks: np.ndarray) -> np.ndarray:
    """
    Return W[ik, band, layer] = <u_{ik,band}|P_layer|u_{ik,band}>.

    evecs shape: (Nk, dim, dim), columns are eigenvectors.
    layer_masks shape: (n_layers, dim).
    """
    abs2 = np.abs(evecs) ** 2  # (Nk, orbital, band)
    W = np.einsum("koa,lo->kal", abs2, layer_masks, optimize=True)
    return np.real(W)


def full_layer_density(mu: float, evals: np.ndarray, evecs: np.ndarray, weights: np.ndarray,
                       layer_masks: np.ndarray, degeneracy: int = 4, kBT_meV: float = 0.1) -> np.ndarray:
    """Layer densities relative to charge neutrality, in A^{-2}."""
    occ = fermi(evals - mu, kBT_meV)  # (Nk, dim)
    W = layer_weights_from_evecs(evecs, layer_masks)  # (Nk, dim, L)
    # Half-filling subtraction: sum over bands of 1/2 * W = 1/2 * Tr P_l per k.
    n_layer = degeneracy * np.einsum("k,kb,kbl->l", weights, occ - 0.5, W, optimize=True)
    return np.real(n_layer)


def dos_at_mu_histogram(evals: np.ndarray, weights: np.ndarray, mu: float,
                        width_meV: float = 1.0, degeneracy: int = 4) -> float:
    mask = np.abs(evals - mu) <= 0.5 * width_meV
    return float(degeneracy * np.sum(weights[:, None] * mask) / width_meV)


def dos_at_mu_gaussian(evals: np.ndarray, weights: np.ndarray, mu: float,
                       sigma_meV: float = 1.0, degeneracy: int = 4) -> float:
    x = (evals - mu) / sigma_meV
    kernel = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sigma_meV)
    return float(degeneracy * np.sum(weights[:, None] * kernel))
