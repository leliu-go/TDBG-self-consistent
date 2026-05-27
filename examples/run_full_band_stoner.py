#!/usr/bin/env python3
"""Run projected/full SCF, then a four-flavor Stoner model on full bands."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdbg_scf import (
    FullSCFConfig,
    FullSCFSolver,
    ProjectedSCFConfig,
    ProjectedSCFSolver,
    TDBGContinuumHamiltonian,
    TDBGParameters,
    linear_potential_from_D,
    make_uniform_mbz_grid,
)
from tdbg_scf.constants import cm2_to_a2
from tdbg_scf.density import dos_at_mu_gaussian, find_mu_for_density
from tdbg_scf.filling import filling_to_density_cm2, moire_cell_area_A2_from_weights
from tdbg_scf.plotting import compute_full_bands_along_path, plot_bands, plot_layer_profile
from tdbg_scf.stoner.full_band import (
    build_full_band_flavor_tables,
    density_cm2_to_nu_total,
    explicit_flavor_dos_at_mu,
    layer_polarizations,
    restrict_tables_to_carrier_sector,
    stoner_flavor_layer_dos_at_mu,
    stoner_flavor_layer_fillings,
    stoner_mu_by_flavor,
    stoner_result_to_dict,
)
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu


FLAVOR_NAMES = ["K_up", "Kp_up", "K_down", "Kp_down"]
REFERENCE_STONER_U0_MEV_A2 = 7.9e4
REFERENCE_STONER_JH_MEV_A2 = 2.4e4


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_hamiltonian_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--n-cm2", type=float, default=2e12)
    ap.add_argument("--D-Vnm", type=float, default=0.4)
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1)
    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0)
    ap.add_argument("--a-cc-A", type=float, default=1.420)
    ap.add_argument("--gamma0-meV", type=float, default=2610.0)
    ap.add_argument("--gamma1-meV", type=float, default=361.0)
    ap.add_argument("--gamma3-meV", type=float, default=283.0)
    ap.add_argument("--gamma4-meV", type=float, default=138.0)


def add_full_scf_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--grid-n1", type=int, default=24)
    ap.add_argument("--grid-n2", type=int, default=24)
    ap.add_argument("--scf-source", choices=["projected_8band", "full"], default="projected_8band")
    ap.add_argument("--scf-valley", type=int, choices=[-1, 1], default=1)
    ap.add_argument("--full-scf-valley", type=int, choices=[-1, 1], default=None)
    ap.add_argument("--n-active", type=int, default=8)
    ap.add_argument("--selection", choices=["closest", "contiguous", "overlap"], default="overlap")
    ap.add_argument("--projector-refreshes", type=int, default=0)
    ap.add_argument("--projected-reference-mode", choices=["uploaded_D", "bare_D", "zero", "full_scf"], default="uploaded_D")
    ap.add_argument("--max-iter", "--projected-max-iter", dest="projected_max_iter", type=int, default=100)
    ap.add_argument("--tol-meV", "--projected-tol-meV", dest="projected_tol_meV", type=float, default=1e-4)
    ap.add_argument("--kBT-meV", type=float, default=0.2)
    ap.add_argument("--eps-perp", type=float, default=3.0)
    ap.add_argument("--d-layer-nm", type=float, default=0.335)
    ap.add_argument("--D-sign", type=float, default=-1.0)
    ap.add_argument("--full-max-iter", type=int, default=60)
    ap.add_argument("--full-tol-meV", type=float, default=1e-4)
    ap.add_argument("--mixer", choices=["linear", "anderson"], default="anderson")
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--anderson-beta", type=float, default=0.5)
    ap.add_argument("--anderson-memory", type=int, default=6)


def add_stoner_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--u0-meV-A2", type=float, default=REFERENCE_STONER_U0_MEV_A2)
    ap.add_argument("--JH-meV-A2", type=float, default=REFERENCE_STONER_JH_MEV_A2)
    ap.add_argument("--stoner-max-iter", type=int, default=200)
    ap.add_argument("--stoner-tol-energy-meV", type=float, default=1e-8)
    ap.add_argument("--stoner-random-seeds", type=int, default=20)
    ap.add_argument("--stoner-seed", type=int, default=0)


def add_analysis_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dos-sigma-meV", type=float, default=1.0)
    ap.add_argument("--dos-emin-meV", type=float, default=-80.0)
    ap.add_argument("--dos-emax-meV", type=float, default=80.0)
    ap.add_argument("--dos-bins", type=int, default=240)
    ap.add_argument("--points-per-segment", type=int, default=60)
    ap.add_argument("--n-show-full", type=int, default=12)
    ap.add_argument("--contour-n1", type=int, default=0, help="0 uses the SCF grid for full-band contour detail plots.")
    ap.add_argument("--contour-n2", type=int, default=0, help="0 uses the SCF grid for full-band contour detail plots.")
    ap.add_argument("--contour-band-count", type=int, default=6)


def params_from_args(args: argparse.Namespace, valley: int) -> TDBGParameters:
    return TDBGParameters(
        theta_deg=args.theta_deg,
        cutoff=args.cutoff,
        valley=valley,
        omega_meV=args.omega,
        wAA=args.wAA,
        wAB=args.wAB,
        a_cc_A=args.a_cc_A,
        gamma0_meV=args.gamma0_meV,
        gamma1_meV=args.gamma1_meV,
        gamma3_meV=args.gamma3_meV,
        gamma4_meV=args.gamma4_meV,
        sublattice_Z_meV=args.Z,
    )


def mixer_kwargs(args: argparse.Namespace) -> dict:
    if args.mixer == "linear":
        return {"alpha": args.alpha}
    return {"beta": args.anderson_beta, "memory": args.anderson_memory, "fallback_alpha": args.alpha}


def stoner_params_from_args(args: argparse.Namespace) -> StonerParams:
    return StonerParams(
        u0_meV_A2=args.u0_meV_A2,
        JH_meV_A2=args.JH_meV_A2,
        max_iter=args.stoner_max_iter,
        tol_energy_meV=args.stoner_tol_energy_meV,
        n_random_seeds=args.stoner_random_seeds,
        seed=args.stoner_seed,
    )


def scf_valley(args: argparse.Namespace) -> int:
    return int(args.full_scf_valley if args.full_scf_valley is not None else args.scf_valley)


def supplied_projector_U(args: argparse.Namespace, D_Vnm: float) -> np.ndarray | None:
    if args.projected_reference_mode == "full_scf":
        return None
    if args.projected_reference_mode == "zero":
        return np.zeros(4, dtype=float)
    strength = "bare" if args.projected_reference_mode == "bare_D" else "uploaded"
    U = linear_potential_from_D(float(D_Vnm), strength=strength)
    U -= np.mean(U)
    return U


def run_scf_reference(args: argparse.Namespace, ham: TDBGContinuumHamiltonian, kpts: np.ndarray, weights: np.ndarray, n_cm2: float, D_Vnm: float):
    if args.scf_source == "full":
        cfg = FullSCFConfig(
            target_density_cm2=n_cm2,
            D_Vnm=D_Vnm,
            degeneracy=4,
            kBT_meV=args.kBT_meV,
            max_iter=args.full_max_iter,
            tol_meV=args.full_tol_meV,
            eps_perp=args.eps_perp,
            d_layer_nm=args.d_layer_nm,
            D_sign=args.D_sign,
            mixer=args.mixer,
            mixer_kwargs=mixer_kwargs(args),
            keep_eigensystem=True,
        )
        result = FullSCFSolver(ham, kpts, weights).solve(cfg)
        return result, {"scf_source": "full", "reference_mode": "full"}

    supplied_U = supplied_projector_U(args, D_Vnm)
    full_cfg = None
    reference_mode = "supplied_U"
    if args.projected_reference_mode == "full_scf":
        reference_mode = "full_scf"
        full_cfg = FullSCFConfig(
            target_density_cm2=n_cm2,
            D_Vnm=D_Vnm,
            degeneracy=4,
            kBT_meV=args.kBT_meV,
            max_iter=args.full_max_iter,
            tol_meV=args.full_tol_meV,
            eps_perp=args.eps_perp,
            d_layer_nm=args.d_layer_nm,
            D_sign=args.D_sign,
            mixer=args.mixer,
            mixer_kwargs=mixer_kwargs(args),
            keep_eigensystem=False,
        )

    cfg = ProjectedSCFConfig(
        target_density_cm2=n_cm2,
        D_Vnm=D_Vnm,
        n_active=args.n_active,
        degeneracy=4,
        kBT_meV=args.kBT_meV,
        max_iter=args.projected_max_iter,
        tol_meV=args.projected_tol_meV,
        eps_perp=args.eps_perp,
        d_layer_nm=args.d_layer_nm,
        D_sign=args.D_sign,
        mixer=args.mixer,
        mixer_kwargs=mixer_kwargs(args),
        selection=args.selection,
        grid_shape=(args.grid_n1, args.grid_n2),
        projector_refreshes=args.projector_refreshes,
        reference_mode=reference_mode,
        supplied_U_ref_meV=supplied_U,
        full_scf_config=full_cfg,
    )
    result = ProjectedSCFSolver(ham, kpts, weights, grid_shape=(args.grid_n1, args.grid_n2)).solve(cfg)
    return result, {
        "scf_source": "projected_8band",
        "reference_mode": args.projected_reference_mode,
        "n_active": int(args.n_active),
        "selection": args.selection,
        "projector_refreshes": int(args.projector_refreshes),
    }


def gaussian_dos_curve(evals_by_flavor: list[np.ndarray], weights: np.ndarray, centers: np.ndarray, sigma_meV: float) -> np.ndarray:
    dos = np.zeros_like(centers, dtype=float)
    for evals in evals_by_flavor:
        flat_e = evals.reshape(-1)
        flat_w = np.repeat(weights, evals.shape[1])
        x = (flat_e[:, None] - centers[None, :]) / float(sigma_meV)
        kernel = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * float(sigma_meV))
        dos += np.sum(flat_w[:, None] * kernel, axis=0)
    return dos


def full_band_dos_summary(
    evals_K: np.ndarray,
    evals_Kp: np.ndarray,
    weights: np.ndarray,
    n_cm2: float,
    projected_mu_meV: float,
    sigma_meV: float,
    kBT_meV: float,
) -> dict[str, float]:
    evals_both = np.concatenate([evals_K, evals_Kp], axis=1)
    full_mu = find_mu_for_density(
        evals_both,
        weights,
        cm2_to_a2(float(n_cm2)),
        degeneracy=2,
        kBT_meV=float(kBT_meV),
    )
    dos_at_projected_mu = float(
        2.0 * dos_at_mu_gaussian(evals_K, weights, float(projected_mu_meV), sigma_meV=sigma_meV, degeneracy=1)
        + 2.0 * dos_at_mu_gaussian(evals_Kp, weights, float(projected_mu_meV), sigma_meV=sigma_meV, degeneracy=1)
    )
    dos_at_full_mu = float(
        2.0 * dos_at_mu_gaussian(evals_K, weights, full_mu, sigma_meV=sigma_meV, degeneracy=1)
        + 2.0 * dos_at_mu_gaussian(evals_Kp, weights, full_mu, sigma_meV=sigma_meV, degeneracy=1)
    )
    return {
        "full_mu_meV": float(full_mu),
        "dos_mu_full_at_projected_mu": dos_at_projected_mu,
        "dos_mu_full_at_full_mu": dos_at_full_mu,
    }


def save_dos_figure(out: Path, evals_K: np.ndarray, evals_Kp: np.ndarray, weights: np.ndarray,
                    reference_mu: float, mu_f: np.ndarray, args: argparse.Namespace) -> None:
    centers = np.linspace(args.dos_emin_meV, args.dos_emax_meV, args.dos_bins)
    evals_by_flavor = [evals_K - reference_mu, evals_Kp - reference_mu, evals_K - reference_mu, evals_Kp - reference_mu]
    dos = gaussian_dos_curve(evals_by_flavor, weights, centers, args.dos_sigma_meV)
    np.savez(out / "full_band_stoner_dos.npz", energy_meV=centers, dos=dos, full_mu_meV=reference_mu, mu_f_meV=mu_f)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(centers, dos, lw=1.2)
    ax.axvline(0.0, color="0.4", lw=0.8, ls="--", label="full-band mu")
    for name, mu in zip(FLAVOR_NAMES, mu_f):
        ax.axvline(float(mu - reference_mu), lw=0.8, alpha=0.65, label=name)
    ax.set_xlabel(r"$E-\mu_{\mathrm{full}}$ (meV)")
    ax.set_ylabel(r"DOS (A$^{-2}$ meV$^{-1}$)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "full_band_stoner_dos.png", dpi=220)
    plt.close(fig)


def save_flavor_occupations(out: Path, nu_f: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(FLAVOR_NAMES, nu_f)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_ylabel(r"$\nu_f$ relative to neutrality")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out / "flavor_occupations.png", dpi=220)
    plt.close(fig)


def save_layer_dos_bars(out: Path, layer_dos: np.ndarray) -> None:
    layers = np.arange(1, len(layer_dos) + 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(layers, layer_dos)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"DOS (A$^{-2}$ meV$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out / "stoner_layer_dos.png", dpi=220)
    plt.close(fig)


def save_flavor_layer_bars(out: Path, values: np.ndarray, ylabel: str, filename: str) -> None:
    layers = np.arange(1, values.shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True)
    for ax, flavor, layer_values in zip(axes.ravel(), FLAVOR_NAMES, values):
        ax.bar(layers, layer_values)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(flavor)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=220)
    plt.close(fig)


def layer_dos_polarizations(layer_dos: np.ndarray) -> dict[str, float]:
    d1, d2, d3, d4 = [float(x) for x in layer_dos]
    return {
        "layer_dos_top_bottom": (d1 + d2) - (d3 + d4),
        "layer_dos_outer_inner": (d1 + d4) - (d2 + d3),
        "layer_dos_dipole": 1.5 * d1 + 0.5 * d2 - 0.5 * d3 - 1.5 * d4,
    }


def save_stoner_flavor_bands(out: Path, dist: np.ndarray, bands_K: np.ndarray, bands_Kp: np.ndarray,
                             ticks: list[int], labels: list[str], mu_f: np.ndarray, n_show: int) -> None:
    bands_by_flavor = [bands_K, bands_Kp, bands_K, bands_Kp]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), sharex=True, sharey=True)
    for ax, name, bands, mu in zip(axes.ravel(), FLAVOR_NAMES, bands_by_flavor, mu_f):
        n_bands = bands.shape[1]
        center = n_bands // 2
        lo = max(0, center - n_show // 2)
        hi = min(n_bands, lo + n_show)
        for band in range(lo, hi):
            ax.plot(dist, bands[:, band] - float(mu), lw=0.8)
        for tick in ticks:
            ax.axvline(dist[tick], color="0.85", lw=0.6)
        ax.axhline(0.0, color="0.5", lw=0.7, ls="--")
        ax.set_title(name)
        ax.set_xticks([dist[t] for t in ticks], labels)
        ax.set_ylabel(r"$E-\mu_f$ (meV)")
    fig.tight_layout()
    fig.savefig(out / "stoner_flavor_bands.png", dpi=220)
    plt.close(fig)


def contour_band_indices(n_bands: int, count: int) -> list[int]:
    count = max(1, min(int(count), int(n_bands)))
    center = int(n_bands) // 2
    start = max(0, center - count // 2)
    end = min(int(n_bands), start + count)
    start = max(0, end - count)
    return list(range(start, end))


def diagonalize_evals_only(ham: TDBGContinuumHamiltonian, kpts: np.ndarray, U_layer_meV: np.ndarray) -> np.ndarray:
    evals = np.empty((len(kpts), ham.dim), dtype=float)
    for ik, (kx, ky) in enumerate(kpts):
        evals[ik] = np.linalg.eigvalsh(ham.hamiltonian(float(kx), float(ky), U_layer_meV))
    return evals


def plot_contour_panels(
    out: Path,
    filename: str,
    kpts: np.ndarray,
    panels: list[tuple[str, np.ndarray, float]],
    band_indices: list[int],
    title: str,
) -> None:
    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    x = kpts[:, 0]
    y = kpts[:, 1]
    for ax, (name, evals, mu) in zip(axes.ravel(), panels):
        shifted = evals[:, band_indices] - float(mu)
        nearest = np.min(np.abs(shifted), axis=1)
        if len(kpts) >= 4:
            levels = np.linspace(float(np.min(nearest)), float(np.max(nearest)), 41)
            if np.isclose(levels[0], levels[-1]):
                levels[-1] = levels[0] + 1e-12
            cf = ax.tricontourf(x, y, nearest, levels=levels, cmap="viridis_r")
            for band in band_indices:
                z = evals[:, band] - float(mu)
                if float(np.min(z)) <= 0.0 <= float(np.max(z)):
                    ax.tricontour(x, y, z, levels=[0.0], colors="k", linewidths=0.8)
            fig.colorbar(cf, ax=ax, shrink=0.82)
        else:
            sc = ax.scatter(x, y, c=nearest, cmap="viridis_r")
            fig.colorbar(sc, ax=ax, shrink=0.82)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=220)
    plt.close(fig)


def save_full_band_contours(
    out: Path,
    args: argparse.Namespace,
    ham_K: TDBGContinuumHamiltonian,
    ham_Kp: TDBGContinuumHamiltonian,
    U_meV: np.ndarray,
    scf_kpts: np.ndarray,
    evals_K: np.ndarray,
    evals_Kp: np.ndarray,
    full_mu: float,
    mu_f: np.ndarray,
) -> None:
    if args.contour_n1 > 0 and args.contour_n2 > 0:
        contour_kpts, _ = make_uniform_mbz_grid(ham_K.geom, args.contour_n1, args.contour_n2)
        contour_evals_K = diagonalize_evals_only(ham_K, contour_kpts, U_meV)
        contour_evals_Kp = diagonalize_evals_only(ham_Kp, contour_kpts, U_meV)
    else:
        contour_kpts = scf_kpts
        contour_evals_K = evals_K
        contour_evals_Kp = evals_Kp

    band_indices = contour_band_indices(contour_evals_K.shape[1], args.contour_band_count)
    np.savez(
        out / "full_band_contours_before_stoner.npz",
        kpts=contour_kpts,
        evals_K_meV=contour_evals_K,
        evals_Kp_meV=contour_evals_Kp,
        full_mu_meV=float(full_mu),
        band_indices=np.asarray(band_indices, dtype=int),
    )
    plot_contour_panels(
        out,
        "full_band_contours_before_stoner.png",
        contour_kpts,
        [("K before Stoner", contour_evals_K, float(full_mu)), ("Kp before Stoner", contour_evals_Kp, float(full_mu))],
        band_indices,
        "Full-band contours before Stoner",
    )

    np.savez(
        out / "full_band_contours_stoner_flavors.npz",
        kpts=contour_kpts,
        evals_K_meV=contour_evals_K,
        evals_Kp_meV=contour_evals_Kp,
        mu_f_meV=np.asarray(mu_f, dtype=float),
        band_indices=np.asarray(band_indices, dtype=int),
    )
    plot_contour_panels(
        out,
        "full_band_contours_stoner_flavors.png",
        contour_kpts,
        [
            ("K up after Stoner", contour_evals_K, float(mu_f[0])),
            ("Kp up after Stoner", contour_evals_Kp, float(mu_f[1])),
            ("K down after Stoner", contour_evals_K, float(mu_f[2])),
            ("Kp down after Stoner", contour_evals_Kp, float(mu_f[3])),
        ],
        band_indices,
        "Full-band contours after Stoner",
    )


def point_label(n_index: int, d_index: int, n_cm2: float, d_vnm: float) -> str:
    n_scaled = n_cm2 / 1e12
    return f"i{n_index:03d}_j{d_index:03d}_n{n_scaled:+.4f}e12_D{d_vnm:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")


def run_point(task: dict[str, Any], args: argparse.Namespace, detail_dir: Path | None = None) -> dict[str, Any]:
    started = time.time()
    n_cm2 = float(task["n_cm2"])
    D_Vnm = float(task["D_Vnm"])
    label = str(task.get("label", "point"))
    try:
        valley_scf = scf_valley(args)
        ham_scf = TDBGContinuumHamiltonian(params_from_args(args, valley_scf))
        ham_K = TDBGContinuumHamiltonian(params_from_args(args, +1))
        ham_Kp = TDBGContinuumHamiltonian(params_from_args(args, -1))
        kpts, weights = make_uniform_mbz_grid(ham_scf.geom, args.grid_n1, args.grid_n2)
        scf, scf_meta = run_scf_reference(args, ham_scf, kpts, weights, n_cm2, D_Vnm)
        if args.scf_source == "full" and valley_scf == 1 and scf.evals is not None and scf.evecs is not None:
            evals_K, evecs_K = scf.evals, scf.evecs
        else:
            evals_K, evecs_K = ham_K.diagonalize(kpts, scf.U_meV)
        if args.scf_source == "full" and valley_scf == -1 and scf.evals is not None and scf.evecs is not None:
            evals_Kp, evecs_Kp = scf.evals, scf.evecs
        else:
            evals_Kp, evecs_Kp = ham_Kp.diagonalize(kpts, scf.U_meV)

        A_M_A2 = moire_cell_area_A2_from_weights(weights)
        nu_total = density_cm2_to_nu_total(n_cm2, A_M_A2)
        tables = restrict_tables_to_carrier_sector(
            build_full_band_flavor_tables(evals_K, evals_Kp, weights, A_M_A2),
            nu_total,
        )
        stoner = solve_stoner_fixed_nu(nu_total, tables, stoner_params_from_args(args), A_M_A2)
        mu_f = stoner_mu_by_flavor(stoner, tables)
        layer_masks = ham_K.layer_projectors_diagonal()
        stoner_flavor_nu_layer = stoner_flavor_layer_fillings(
            evals_K,
            evecs_K,
            evals_Kp,
            evecs_Kp,
            weights,
            layer_masks,
            A_M_A2,
            stoner.nu_f,
        )
        stoner_nu_layer = np.sum(stoner_flavor_nu_layer, axis=0)
        stoner_flavor_n_layer_cm2 = np.asarray(
            [[filling_to_density_cm2(x, A_M_A2) for x in flavor_layer] for flavor_layer in stoner_flavor_nu_layer],
            dtype=float,
        )
        stoner_n_layer_cm2 = np.asarray([filling_to_density_cm2(x, A_M_A2) for x in stoner_nu_layer], dtype=float)
        stoner_flavor_layer_dos = stoner_flavor_layer_dos_at_mu(
            evals_K,
            evecs_K,
            evals_Kp,
            evecs_Kp,
            weights,
            layer_masks,
            mu_f,
            args.dos_sigma_meV,
        )
        stoner_layer_dos = np.sum(stoner_flavor_layer_dos, axis=0)

        n_layer_cm2 = np.asarray(scf.n_layer_cm2, dtype=float)
        pol = layer_polarizations(n_layer_cm2)
        stoner_pol = {f"stoner_{key}": value for key, value in layer_polarizations(stoner_n_layer_cm2).items()}
        stoner_layer_dos_pol = {
            "stoner_layer_dos_total": float(np.sum(stoner_layer_dos)),
            "stoner_layer_dos_top_bottom": float((stoner_layer_dos[0] + stoner_layer_dos[1]) - (stoner_layer_dos[2] + stoner_layer_dos[3])),
            "stoner_layer_dos_outer_inner": float((stoner_layer_dos[0] + stoner_layer_dos[3]) - (stoner_layer_dos[1] + stoner_layer_dos[2])),
            "stoner_layer_dos_dipole": float(1.5 * stoner_layer_dos[0] + 0.5 * stoner_layer_dos[1] - 0.5 * stoner_layer_dos[2] - 1.5 * stoner_layer_dos[3]),
        }
        full_dos = full_band_dos_summary(
            evals_K,
            evals_Kp,
            weights,
            n_cm2=n_cm2,
            projected_mu_meV=scf.mu_meV,
            sigma_meV=args.dos_sigma_meV,
            kBT_meV=args.kBT_meV,
        )
        row = {
            "status": "ok",
            "label": label,
            "scf_source": scf_meta["scf_source"],
            "scf_reference_mode": scf_meta["reference_mode"],
            "n_index": task.get("n_index", 0),
            "D_index": task.get("D_index", 0),
            "n_cm2": n_cm2,
            "nu_total": float(nu_total),
            "D_Vnm": D_Vnm,
            "A_M_A2": float(A_M_A2),
            "scf_mu_meV": float(scf.mu_meV),
            "scf_converged": bool(scf.converged),
            "scf_residual_meV": float(scf.residual_meV),
            "scf_iterations": int(scf.iterations),
            "full_mu_meV": full_dos["full_mu_meV"],
            "full_converged": bool(scf.converged),
            "full_residual_meV": float(scf.residual_meV),
            "full_iterations": int(scf.iterations),
            "stoner_success": bool(stoner.success),
            "stoner_seed": stoner.seed_name,
            "stoner_energy_meV_per_cell": float(stoner.energy_meV_per_cell),
            "nu_K_up": float(stoner.nu_f[0]),
            "nu_Kp_up": float(stoner.nu_f[1]),
            "nu_K_down": float(stoner.nu_f[2]),
            "nu_Kp_down": float(stoner.nu_f[3]),
            "mu_K_up_meV": float(mu_f[0]),
            "mu_Kp_up_meV": float(mu_f[1]),
            "mu_K_down_meV": float(mu_f[2]),
            "mu_Kp_down_meV": float(mu_f[3]),
            "spin_polarization": stoner.spin_polarization,
            "spin_polarization_norm": stoner.spin_polarization_norm,
            "valley_polarization": stoner.valley_polarization,
            "valley_polarization_norm": stoner.valley_polarization_norm,
            "spin_valley_polarization": stoner.spin_valley_polarization,
            "spin_valley_polarization_norm": stoner.spin_valley_polarization_norm,
            "flavor_polarization": stoner.flavor_polarization,
            "dos_mu_scf": full_dos["dos_mu_full_at_full_mu"],
            "dos_mu_full_scf": full_dos["dos_mu_full_at_full_mu"],
            "dos_mu_full_at_full_mu": full_dos["dos_mu_full_at_full_mu"],
            "dos_mu_full_at_projected_mu": full_dos["dos_mu_full_at_projected_mu"],
            "dos_mu_stoner": explicit_flavor_dos_at_mu(evals_K, evals_Kp, weights, mu_f, args.dos_sigma_meV),
            "U1_meV": float(scf.U_meV[0]),
            "U2_meV": float(scf.U_meV[1]),
            "U3_meV": float(scf.U_meV[2]),
            "U4_meV": float(scf.U_meV[3]),
            "n1_cm2": float(n_layer_cm2[0]),
            "n2_cm2": float(n_layer_cm2[1]),
            "n3_cm2": float(n_layer_cm2[2]),
            "n4_cm2": float(n_layer_cm2[3]),
            "stoner_n1_cm2": float(stoner_n_layer_cm2[0]),
            "stoner_n2_cm2": float(stoner_n_layer_cm2[1]),
            "stoner_n3_cm2": float(stoner_n_layer_cm2[2]),
            "stoner_n4_cm2": float(stoner_n_layer_cm2[3]),
            "stoner_n_total_layer_cm2": float(np.sum(stoner_n_layer_cm2)),
            "stoner_layer_dos_L1": float(stoner_layer_dos[0]),
            "stoner_layer_dos_L2": float(stoner_layer_dos[1]),
            "stoner_layer_dos_L3": float(stoner_layer_dos[2]),
            "stoner_layer_dos_L4": float(stoner_layer_dos[3]),
            "runtime_s": float(time.time() - started),
        }
        row.update(pol)
        row.update(stoner_pol)
        row.update(stoner_layer_dos_pol)
        for flavor_index, flavor_name in enumerate(FLAVOR_NAMES):
            flavor_n = stoner_flavor_n_layer_cm2[flavor_index]
            flavor_dos = stoner_flavor_layer_dos[flavor_index]
            flavor_prefix = f"stoner_{flavor_name}"
            for layer_index in range(4):
                row[f"{flavor_prefix}_n{layer_index + 1}_cm2"] = float(flavor_n[layer_index])
                row[f"{flavor_prefix}_layer_dos_L{layer_index + 1}"] = float(flavor_dos[layer_index])
            for key, value in layer_polarizations(flavor_n).items():
                row[f"{flavor_prefix}_{key}"] = value
            row[f"{flavor_prefix}_layer_dos_total"] = float(np.sum(flavor_dos))
            for key, value in layer_dos_polarizations(flavor_dos).items():
                row[f"{flavor_prefix}_{key}"] = value

        if detail_dir is not None:
            save_detail_outputs(
                detail_dir,
                row,
                args,
                ham_K,
                ham_Kp,
                kpts,
                weights,
                scf,
                evals_K,
                evals_Kp,
                evecs_K,
                evecs_Kp,
                stoner,
                tables,
            )
        return row
    except Exception as exc:
        error = {
            "status": "failed",
            "label": label,
            "n_index": task.get("n_index", 0),
            "D_index": task.get("D_index", 0),
            "n_cm2": n_cm2,
            "D_Vnm": D_Vnm,
            "runtime_s": float(time.time() - started),
            "error": repr(exc),
        }
        if detail_dir is not None:
            detail_dir.mkdir(parents=True, exist_ok=True)
            write_json(detail_dir / "error.json", error)
        return error


def save_detail_outputs(
    out: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    ham_K: TDBGContinuumHamiltonian,
    ham_Kp: TDBGContinuumHamiltonian,
    kpts: np.ndarray,
    weights: np.ndarray,
    scf,
    evals_K: np.ndarray,
    evals_Kp: np.ndarray,
    evecs_K: np.ndarray,
    evecs_Kp: np.ndarray,
    stoner,
    tables,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stoner_dict = stoner_result_to_dict(stoner, tables)
    write_json(out / "summary.json", {"point": row, "stoner": stoner_dict})
    write_json(out / "stoner_result.json", stoner_dict)
    pd.DataFrame(scf.history).to_csv(out / "scf_history.csv", index=False)
    pd.DataFrame({"layer": [1, 2, 3, 4], "U_meV": scf.U_meV, "n_cm2": scf.n_layer_cm2}).to_csv(out / "layers.csv", index=False)
    stoner_n_layer_cm2 = np.asarray([row[f"stoner_n{layer}_cm2"] for layer in range(1, 5)], dtype=float)
    stoner_layer_dos = np.asarray([row[f"stoner_layer_dos_L{layer}"] for layer in range(1, 5)], dtype=float)
    pd.DataFrame(
        {
            "layer": [1, 2, 3, 4],
            "U_meV": scf.U_meV,
            "scf_n_cm2": scf.n_layer_cm2,
            "stoner_n_cm2": stoner_n_layer_cm2,
            "stoner_layer_dos": stoner_layer_dos,
        }
    ).to_csv(out / "stoner_layers.csv", index=False)
    stoner_flavor_n_layer_cm2 = np.asarray(
        [[row[f"stoner_{flavor}_n{layer}_cm2"] for layer in range(1, 5)] for flavor in FLAVOR_NAMES],
        dtype=float,
    )
    stoner_flavor_layer_dos = np.asarray(
        [[row[f"stoner_{flavor}_layer_dos_L{layer}"] for layer in range(1, 5)] for flavor in FLAVOR_NAMES],
        dtype=float,
    )
    flavor_layer_rows = []
    for flavor_index, flavor in enumerate(FLAVOR_NAMES):
        for layer_index in range(4):
            flavor_layer_rows.append(
                {
                    "flavor": flavor,
                    "layer": layer_index + 1,
                    "U_meV": float(scf.U_meV[layer_index]),
                    "stoner_n_cm2": float(stoner_flavor_n_layer_cm2[flavor_index, layer_index]),
                    "stoner_layer_dos": float(stoner_flavor_layer_dos[flavor_index, layer_index]),
                }
            )
    pd.DataFrame(flavor_layer_rows).to_csv(out / "stoner_flavor_layers.csv", index=False)
    np.savez(
        out / "full_band_stoner_cache.npz",
        kpts=kpts,
        weights=weights,
        U_scf_meV=scf.U_meV,
        mu_scf_meV=scf.mu_meV,
        mu_full_meV=row["full_mu_meV"],
        evals_K_meV=evals_K,
        evals_Kp_meV=evals_Kp,
        evecs_K=evecs_K,
        evecs_Kp=evecs_Kp,
        n_layer_cm2=scf.n_layer_cm2,
        stoner_n_layer_cm2=stoner_n_layer_cm2,
        stoner_layer_dos=stoner_layer_dos,
        stoner_flavor_n_layer_cm2=stoner_flavor_n_layer_cm2,
        stoner_flavor_layer_dos=stoner_flavor_layer_dos,
        nu_f=stoner.nu_f,
        mu_f_meV=stoner_mu_by_flavor(stoner, tables),
    )

    plot_layer_profile(scf.U_meV, scf.n_layer_cm2, out=str(out / "layer_profile.png"))
    plot_layer_profile(scf.U_meV, stoner_n_layer_cm2, out=str(out / "stoner_layer_profile.png"))
    save_layer_dos_bars(out, stoner_layer_dos)
    save_flavor_layer_bars(out, stoner_flavor_n_layer_cm2, r"$n_l$ (cm$^{-2}$)", "stoner_flavor_layer_profile.png")
    save_flavor_layer_bars(out, stoner_flavor_layer_dos, r"DOS (A$^{-2}$ meV$^{-1}$)", "stoner_flavor_layer_dos.png")
    save_flavor_occupations(out, stoner.nu_f)
    mu_f = stoner_mu_by_flavor(stoner, tables)
    full_mu = float(row["full_mu_meV"])
    save_dos_figure(out, evals_K, evals_Kp, weights, full_mu, mu_f, args)

    dist_K, bands_K, ticks_K, labels_K = compute_full_bands_along_path(ham_K, scf.U_meV, points_per_segment=args.points_per_segment)
    dist_Kp, bands_Kp, ticks_Kp, labels_Kp = compute_full_bands_along_path(ham_Kp, scf.U_meV, points_per_segment=args.points_per_segment)
    np.savez(out / "full_bands_K.npz", dist=dist_K, bands=bands_K, ticks=ticks_K, labels=labels_K, mu_meV=full_mu)
    np.savez(out / "full_bands_Kp.npz", dist=dist_Kp, bands=bands_Kp, ticks=ticks_Kp, labels=labels_Kp, mu_meV=full_mu)
    plot_bands(dist_K, bands_K, ticks_K, labels_K, mu_meV=full_mu, n_show=args.n_show_full, title="Full bands K at SCF U", out=str(out / "full_bands_K.png"))
    plot_bands(dist_Kp, bands_Kp, ticks_Kp, labels_Kp, mu_meV=full_mu, n_show=args.n_show_full, title="Full bands K' at SCF U", out=str(out / "full_bands_Kp.png"))
    save_stoner_flavor_bands(out, dist_K, bands_K, bands_Kp, ticks_K, labels_K, mu_f, args.n_show_full)
    save_full_band_contours(
        out,
        args,
        ham_K,
        ham_Kp,
        scf.U_meV,
        kpts,
        evals_K,
        evals_Kp,
        full_mu,
        mu_f,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run projected/full SCF then full-band four-flavor Stoner at one n-D point.")
    add_hamiltonian_args(ap)
    add_full_scf_args(ap)
    add_stoner_args(ap)
    add_analysis_args(ap)
    ap.add_argument("--out", type=str, default="outputs/full_band_stoner")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    row = run_point({"n_cm2": args.n_cm2, "D_Vnm": args.D_Vnm, "label": "single_point"}, args, detail_dir=out)
    write_json(out / "run_summary.json", row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
