"""Density, chemical-potential, and layer-weight routines."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def validate_gaussian_sigma(sigma_meV: float) -> float:
    sigma = float(sigma_meV)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_meV must be finite and positive")
    return sigma


def _validate_spectrum(evals, weights, degeneracy, kBT_meV):
    energies = np.asarray(evals, dtype=float)
    measure = np.asarray(weights, dtype=float)
    if (energies.ndim != 2 or energies.size == 0 or not np.isfinite(energies).all()
            or measure.shape != (energies.shape[0],) or not np.isfinite(measure).all()
            or np.any(measure < 0) or not np.any(measure > 0)):
        raise ValueError("evals and weights need a finite (Nk, Nb) spectrum and a positive integration measure")
    if not np.isfinite(degeneracy) or degeneracy <= 0:
        raise ValueError("degeneracy must be finite and positive")
    if not np.isfinite(kBT_meV) or kBT_meV < 0:
        raise ValueError("kBT_meV must be finite and nonnegative")
    return energies, measure


def _zero_temperature_fill(evals, weights, target_density_a2, degeneracy):
    """Fill weighted levels exactly, sharing a partially filled degenerate level.

    A finite mesh generally needs a fractional occupation at the Fermi
    energy. A root of a step function cannot encode that fraction by mu alone.
    """
    state_e = evals.ravel()
    state_w = np.repeat(weights, evals.shape[1]) * degeneracy
    capacity = float(np.sum(state_w))
    target_abs = .5*capacity + float(target_density_a2)
    tolerance = 32*np.finfo(float).eps*capacity
    if not np.isfinite(target_abs) or target_abs < -tolerance or target_abs > capacity+tolerance:
        raise ValueError("target density outside the available full-band capacity")
    target_abs = float(np.clip(target_abs, 0., capacity))
    order = np.argsort(state_e, kind="mergesort")
    es, ws = state_e[order], state_w[order]
    positive = ws > 0
    es_positive, ws_positive = es[positive], ws[positive]
    energy_tolerance = max(1e-10, 16*np.finfo(float).eps*max(1., float(np.max(np.abs(es_positive)))))
    starts = np.r_[0, 1+np.flatnonzero(np.diff(es_positive) > energy_tolerance)]
    ends = np.r_[starts[1:], len(es_positive)]
    group_capacity = np.add.reduceat(ws_positive, starts)
    cumulative = np.r_[0., np.cumsum(group_capacity)]
    cumulative[-1] = capacity
    fraction = np.clip((target_abs-cumulative[:-1])/group_capacity, 0., 1.)
    occ_sorted = np.zeros_like(es)
    occ_sorted[positive] = np.repeat(fraction, ends-starts)
    occ = np.empty_like(state_e)
    occ[order] = occ_sorted

    if target_abs <= tolerance:
        mu = es_positive[0]-1.
    elif target_abs >= capacity-tolerance:
        mu = es_positive[-1]+1.
    else:
        node = int(np.searchsorted(cumulative, target_abs, side="left"))
        if abs(cumulative[node]-target_abs) <= tolerance:
            mu = .5*(es_positive[ends[node-1]-1]+es_positive[starts[node]])
        elif node > 0 and abs(cumulative[node-1]-target_abs) <= tolerance:
            mu = .5*(es_positive[ends[node-2]-1]+es_positive[starts[node-1]])
        else:
            mu = es_positive[starts[node-1]]
    return float(mu), occ.reshape(evals.shape)


def fermi(E_minus_mu: np.ndarray, kBT_meV: float) -> np.ndarray:
    if not np.isfinite(kBT_meV) or kBT_meV < 0:
        raise ValueError("kBT_meV must be finite and nonnegative")
    energy = np.asarray(E_minus_mu, dtype=float)
    if kBT_meV == 0:
        return np.where(energy < 0., 1., np.where(energy > 0., 0., .5))
    x = energy / kBT_meV
    x = np.clip(x, -100.0, 100.0)
    return 1.0 / (np.exp(x) + 1.0)


def total_density_from_evals(mu: float, evals: np.ndarray, weights: np.ndarray, degeneracy: int, kBT_meV: float,
                             target_density_a2: float | None = None) -> float:
    """Total density relative to charge neutrality, in A^{-2}."""
    evals, weights = _validate_spectrum(evals, weights, degeneracy, kBT_meV)
    occ = (_zero_temperature_fill(evals, weights, target_density_a2, degeneracy)[1]
           if kBT_meV == 0 and target_density_a2 is not None else fermi(evals - mu, kBT_meV))
    return degeneracy * np.sum(weights[:, None] * (occ - 0.5))


def find_mu_for_density(evals: np.ndarray, weights: np.ndarray, target_density_a2: float,
                        degeneracy: int = 4, kBT_meV: float = 0.1,
                        extra_window_meV: float = 500.0) -> float:
    evals, weights = _validate_spectrum(evals, weights, degeneracy, kBT_meV)
    if not np.isfinite(target_density_a2):
        raise ValueError("target_density_a2 must be finite")
    if kBT_meV == 0:
        return _zero_temperature_fill(evals, weights, target_density_a2, degeneracy)[0]
    lo = float(np.min(evals) - extra_window_meV)
    hi = float(np.max(evals) + extra_window_meV)

    def fn(mu: float) -> float:
        return total_density_from_evals(mu, evals, weights, degeneracy, kBT_meV) - target_density_a2

    flo = fn(lo)
    fhi = fn(hi)
    if flo > 0 or fhi < 0:
        raise RuntimeError(
            f"Chemical-potential bracket failed: f(lo)={flo:.3e}, f(hi)={fhi:.3e}, "
            f"target={target_density_a2:.3e}. Enlarge the energy window or continuum cutoff."
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
                       layer_masks: np.ndarray, degeneracy: int = 4, kBT_meV: float = 0.1,
                       target_density_a2: float | None = None) -> np.ndarray:
    """Layer densities relative to half filling, in A^{-2}.

    Pass the target at T=0 to fix a fractional Fermi-level occupation on a
    finite mesh. Equal-energy states receive equal occupation.
    """
    evals, weights = _validate_spectrum(evals, weights, degeneracy, kBT_meV)
    occ = (_zero_temperature_fill(evals, weights, target_density_a2, degeneracy)[1]
           if kBT_meV == 0 and target_density_a2 is not None else fermi(evals - mu, kBT_meV))
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
    sigma_meV = validate_gaussian_sigma(sigma_meV)
    x = (evals - mu) / sigma_meV
    kernel = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sigma_meV)
    return float(degeneracy * np.sum(weights[:, None] * kernel))
