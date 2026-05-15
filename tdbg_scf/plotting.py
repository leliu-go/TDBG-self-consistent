"""Plotting helpers for TDBG SCF results."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .continuum import TDBGContinuumHamiltonian
from .density import dos_at_mu_histogram, dos_at_mu_gaussian


def compute_full_bands_along_path(ham: TDBGContinuumHamiltonian, U_meV: np.ndarray,
                                  points_per_segment: int = 80) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    kpts, dist, ticks, labels = ham.geom.high_symmetry_path(points_per_segment=points_per_segment)
    evals, _ = ham.diagonalize(kpts, U_meV)
    return dist, evals, ticks, labels


def compute_projected_bands_along_path(ham: TDBGContinuumHamiltonian,
                                       U_ref_meV: np.ndarray,
                                       U_meV: np.ndarray,
                                       mu_ref_meV: float,
                                       n_active: int = 8,
                                       selection: str = "closest",
                                       points_per_segment: int = 80) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    kpts, dist, ticks, labels = ham.geom.high_symmetry_path(points_per_segment=points_per_segment)
    from .solver_projected import build_projected_model_at_mu, projected_eigensystem

    model = build_projected_model_at_mu(
        ham,
        kpts,
        U_ref_meV=U_ref_meV,
        mu_ref_meV=mu_ref_meV,
        n_active=n_active,
        selection=selection,
    )
    evals, _ = projected_eigensystem(model, U_meV)
    return dist, evals, ticks, labels


def plot_bands(dist: np.ndarray, evals: np.ndarray, ticks: list[int], labels: list[str],
               mu_meV: float = 0.0, n_show: int = 12, title: str = "", out: str | None = None) -> None:
    n_bands = evals.shape[1]
    center = n_bands // 2
    lo = max(0, center - n_show // 2)
    hi = min(n_bands, lo + n_show)
    plt.figure(figsize=(4.0, 5.0))
    for j in range(lo, hi):
        plt.plot(dist, evals[:, j] - mu_meV, lw=1.0)
    for t in ticks:
        plt.axvline(dist[t], color="0.8", lw=0.8)
    plt.axhline(0.0, color="0.5", lw=0.8, ls="--")
    plt.xticks([dist[t] for t in ticks], labels)
    plt.ylabel(r"$E-\mu$ (meV)")
    plt.title(title)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=220)
    plt.close()


def plot_layer_profile(U_meV: np.ndarray, n_layer_cm2: np.ndarray, out: str | None = None) -> None:
    layers = np.arange(1, len(U_meV) + 1)
    fig, ax1 = plt.subplots(figsize=(4.2, 3.2))
    ax1.plot(layers, U_meV, marker="o")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(r"$U_l$ (meV)")
    ax2 = ax1.twinx()
    ax2.bar(layers, n_layer_cm2 / 1e12, alpha=0.25)
    ax2.set_ylabel(r"$n_l$ ($10^{12}$ cm$^{-2}$)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=220)
    plt.close(fig)


def save_dos_plot(evals: np.ndarray, weights: np.ndarray, mu_meV: float,
                  degeneracy: int = 4, out: str | None = None,
                  emin: float = -80.0, emax: float = 80.0, bins: int = 240) -> tuple[np.ndarray, np.ndarray]:
    energies = np.linspace(emin, emax, bins + 1)
    centers = 0.5 * (energies[:-1] + energies[1:])
    flat_E = (evals - mu_meV).ravel()
    # Repeat weights for each band.
    w = np.repeat(weights, evals.shape[1])
    counts, _ = np.histogram(flat_E, bins=energies, weights=w)
    dos = degeneracy * counts / np.diff(energies)
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(centers, dos)
    plt.xlabel(r"$E-\mu$ (meV)")
    plt.ylabel(r"DOS (A$^{-2}$ meV$^{-1}$)")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=220)
    plt.close()
    return centers, dos
