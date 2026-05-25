"""Density matrices and fixed-filling chemical potential helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedDensity:
    P: np.ndarray
    nu_f: np.ndarray
    occ: np.ndarray
    trace_per_k: np.ndarray
    mu_meV: float


def _fermi(evals: np.ndarray, mu: float, kBT_meV: float) -> np.ndarray:
    x = (evals - float(mu)) / float(kBT_meV)
    x = np.clip(x, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def _zero_temperature_occupations(evals: np.ndarray, weights: np.ndarray, A_M_A2: float, target_abs: float):
    flat_e = evals.reshape(-1)
    n_flavor, n_k, n_band = evals.shape
    flat_w = np.tile(np.repeat(weights, n_band), n_flavor) * float(A_M_A2)
    order = np.argsort(flat_e, kind="mergesort")
    e_sorted = flat_e[order]
    w_sorted = flat_w[order]
    capacity = float(np.sum(w_sorted))
    if target_abs < -1e-10 or target_abs > capacity + 1e-10:
        raise ValueError("target filling is outside active-space capacity")

    occ_sorted = np.zeros_like(e_sorted, dtype=float)
    remaining = min(max(float(target_abs), 0.0), capacity)
    mu = float(e_sorted[0])
    start = 0
    while start < e_sorted.size:
        stop = start + 1
        while stop < e_sorted.size and abs(e_sorted[stop] - e_sorted[start]) <= 1e-12:
            stop += 1
        shell_weight = float(np.sum(w_sorted[start:stop]))
        if remaining > shell_weight + 1e-10:
            occ_sorted[start:stop] = 1.0
            remaining -= shell_weight
            start = stop
            continue
        if remaining <= 1e-10:
            below = e_sorted[start - 1] if start > 0 else e_sorted[start]
            above = e_sorted[start]
            mu = float(0.5 * (below + above))
            break
        fraction = remaining / shell_weight if shell_weight > 0.0 else 0.0
        occ_sorted[start:stop] = fraction
        if fraction >= 1.0 - 1e-10:
            below = e_sorted[stop - 1]
            above = e_sorted[stop] if stop < e_sorted.size else e_sorted[stop - 1]
            mu = float(0.5 * (below + above))
        else:
            mu = float(e_sorted[start])
        remaining = 0.0
        break
    else:
        mu = float(e_sorted[-1])

    occ_flat = np.zeros_like(occ_sorted)
    occ_flat[order] = occ_sorted
    return mu, occ_flat.reshape(n_flavor, n_k, n_band)


def _finite_temperature_occupations(
    evals: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    target_abs: float,
    kBT_meV: float,
):
    from scipy.optimize import brentq

    n_flavor, _, n_band = evals.shape
    state_weights = np.tile(np.repeat(weights, n_band), n_flavor) * float(A_M_A2)
    flat_e = evals.reshape(-1)

    def count(mu):
        return float(np.dot(state_weights, _fermi(flat_e, mu, kBT_meV)) - target_abs)

    lo = float(np.min(flat_e) - 80.0 * kBT_meV - 1.0)
    hi = float(np.max(flat_e) + 80.0 * kBT_meV + 1.0)
    mu = float(brentq(count, lo, hi))
    occ = _fermi(flat_e, mu, kBT_meV).reshape(n_flavor, -1, n_band)
    return mu, occ


def density_matrices_from_eigensystems(
    evals: np.ndarray,
    evecs: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    nu_target: float,
    n_ref_per_flavor: float,
    kBT_meV: float = 0.0,
) -> tuple[float, ProjectedDensity]:
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=np.complex128)
    weights = np.asarray(weights, dtype=float)
    if evals.ndim != 3:
        raise ValueError("evals must have shape (4, Nk, Nb)")
    if evecs.shape != evals.shape + (evals.shape[-1],):
        raise ValueError("evecs must have shape (4, Nk, Nb, Nb)")
    if weights.shape != (evals.shape[1],):
        raise ValueError("weights must have shape (Nk,)")

    n_flavor, n_k, n_band = evals.shape
    ref_abs = float(A_M_A2) * float(np.sum(weights)) * float(n_ref_per_flavor) * n_flavor
    target_abs = float(nu_target) + ref_abs

    if kBT_meV > 0.0:
        mu, occ = _finite_temperature_occupations(evals, weights, A_M_A2, target_abs, kBT_meV)
    else:
        mu, occ = _zero_temperature_occupations(evals, weights, A_M_A2, target_abs)

    P = np.zeros((n_flavor, n_k, n_band, n_band), dtype=np.complex128)
    traces = np.zeros((n_flavor, n_k), dtype=float)
    for f in range(n_flavor):
        for ik in range(n_k):
            vec = evecs[f, ik]
            P_fk = (vec * occ[f, ik][None, :]) @ vec.conj().T
            P[f, ik] = 0.5 * (P_fk + P_fk.conj().T)
            traces[f, ik] = float(np.real(np.trace(P[f, ik])))

    nu_f = float(A_M_A2) * np.sum(weights[None, :] * (traces - float(n_ref_per_flavor)), axis=1)
    density = ProjectedDensity(P=P, nu_f=nu_f, occ=occ, trace_per_k=traces, mu_meV=float(mu))
    return float(mu), density
