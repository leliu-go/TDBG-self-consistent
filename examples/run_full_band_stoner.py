#!/usr/bin/env python3
"""Run full self-consistent Hartree, then four-flavor full-band Stoner."""
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
    TDBGContinuumHamiltonian,
    TDBGParameters,
    make_uniform_mbz_grid,
)
from tdbg_scf.constants import cm2_to_a2
from tdbg_scf.density import dos_at_mu_gaussian, find_mu_for_density, full_layer_density, validate_gaussian_sigma
from tdbg_scf.electrostatics import LayerElectrostatics
from tdbg_scf.output_metadata import hamiltonian_metadata, hamiltonian_npz_metadata, displacement_metadata
from tdbg_scf.filling import filling_to_density_cm2, moire_cell_area_A2_from_weights
from tdbg_scf.plotting import compute_full_bands_along_path, plot_bands, plot_layer_profile
from tdbg_scf.stoner.full_band import (
    build_full_band_flavor_tables,
    density_cm2_to_nu_total,
    explicit_flavor_dos_at_mu,
    layer_polarizations,
    layer_dos_at_mu_gaussian,
    restrict_tables_to_carrier_sector,
    stoner_flavor_layer_dos_at_mu,
    stoner_flavor_layer_fillings,
    stoner_mu_by_flavor,
    stoner_mean_field_fermi_level,
    stoner_result_to_dict,
)
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu
from tdbg_scf.scf_outputs import scf_point_record
from tdbg_scf.scf_checks import scf_convergence_diagnostics, require_converged_scf


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
    ap.add_argument("--scf-valley", type=int, choices=[-1, 1], default=1)
    ap.add_argument("--kBT-meV", type=float, default=0.2)
    ap.add_argument("--eps-perp", type=float, default=3.0)
    ap.add_argument("--d-layer-nm", type=float, default=0.335)
    ap.add_argument("--D-sign", type=float, choices=[-1, 1], default=-1.0)
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
    return int(args.scf_valley)


def run_scf_reference(args: argparse.Namespace, ham: TDBGContinuumHamiltonian, kpts: np.ndarray, weights: np.ndarray, n_cm2: float, D_Vnm: float):
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


def gaussian_dos_curve(evals_by_flavor: list[np.ndarray], weights: np.ndarray, centers: np.ndarray, sigma_meV: float) -> np.ndarray:
    sigma_meV = validate_gaussian_sigma(sigma_meV)
    dos = np.zeros_like(centers, dtype=float)
    for evals in evals_by_flavor:
        flat_e = evals.reshape(-1)
        flat_w = np.repeat(weights, evals.shape[1])
        x = (flat_e[:, None] - centers[None, :]) / float(sigma_meV)
        kernel = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * float(sigma_meV))
        dos += np.sum(flat_w[:, None] * kernel, axis=0)
    return dos


def full_hartree_consistency(evals, evecs, weights, layer_masks, U_meV, n_cm2,
                             D_Vnm, kBT_meV, eps_perp, d_layer_nm, D_sign, tol_meV):
    """Evaluate full Hartree at the existing U without another diagonalization.

    Use the SCF valley and spin/valley degeneracy four, matching FullSCFSolver.
    This independently verifies the accepted solution using its final states.
    """
    mu = find_mu_for_density(evals, weights, cm2_to_a2(n_cm2), degeneracy=4, kBT_meV=kBT_meV)
    density = full_layer_density(mu, evals, evecs, weights, layer_masks,
                                 degeneracy=4, kBT_meV=kBT_meV,
                                 target_density_a2=cm2_to_a2(n_cm2))
    electro = LayerElectrostatics(len(U_meV), d_layer_nm=d_layer_nm, eps_perp=eps_perp, D_sign=D_sign)
    _, bottom = electro.gate_densities(cm2_to_a2(n_cm2), D_Vnm)
    residual = float(np.max(np.abs(electro.update_U_from_density(density, bottom)-U_meV)))
    charge_error = abs(float(np.sum(density)*1e16)-n_cm2)
    return {
        "scf_full_hartree_residual_meV": residual,
        "scf_full_hartree_consistent": bool(np.isfinite(residual) and residual < tol_meV
                                             and charge_error < max(1e4,abs(n_cm2)*1e-8)),
        "scf_full_hartree_mu_meV": float(mu),
        "scf_full_hartree_charge_error_cm2": charge_error,
        **{f"scf_full_hartree_n{layer+1}_cm2": float(value*1e16)
           for layer, value in enumerate(density)},
    }


def full_band_dos_summary(
    evals_K: np.ndarray,
    evals_Kp: np.ndarray,
    weights: np.ndarray,
    n_cm2: float,
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
    dos_at_full_mu = float(
        2.0 * dos_at_mu_gaussian(evals_K, weights, full_mu, sigma_meV=sigma_meV, degeneracy=1)
        + 2.0 * dos_at_mu_gaussian(evals_Kp, weights, full_mu, sigma_meV=sigma_meV, degeneracy=1)
    )
    return {
        "full_mu_meV": float(full_mu),
        "dos_mu_full_at_full_mu": dos_at_full_mu,
    }


def save_dos_figure(out: Path, evals_K: np.ndarray, evals_Kp: np.ndarray, weights: np.ndarray,
                    reference_mu: float, mu_f: np.ndarray, args: argparse.Namespace,
                    ham: TDBGContinuumHamiltonian, dos_data=None) -> None:
    if dos_data is None:
        centers = np.linspace(args.dos_emin_meV, args.dos_emax_meV, args.dos_bins)
        before = gaussian_dos_curve([evals_K-reference_mu, evals_Kp-reference_mu]*2,
                                     weights, centers, args.dos_sigma_meV)
    else:
        centers, before = dos_data
    # mu_f=mu_common-Sigma_f, so each curve has the same interacting Fermi zero.
    after = gaussian_dos_curve([E-mu for E,mu in zip([evals_K,evals_Kp]*2,mu_f)],
                               weights, centers, args.dos_sigma_meV)
    np.savez(out / "full_band_stoner_dos.npz", **hamiltonian_npz_metadata(ham),
             energy_meV=centers, dos=after, scf_dos=before, full_mu_meV=reference_mu,
             mu_f_meV=mu_f, energy_reference="E_plus_Sigma_f_minus_common_stoner_mu")

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(centers, before, lw=1., alpha=.6, label="SCF before Stoner")
    ax.plot(centers, after, lw=1.2, label="Stoner mean field")
    ax.axvline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"Energy relative to each stage's Fermi level (meV)")
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


def save_layer_dos_bars(out: Path, layer_dos: np.ndarray, filename="stoner_layer_dos.png") -> None:
    layers = np.arange(1, len(layer_dos) + 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(layers, layer_dos)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"DOS (A$^{-2}$ meV$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=220)
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
    """Apply the exact layer_polarizations coefficients to layer DOS."""
    pol = layer_polarizations(layer_dos)
    return {
        "layer_dos_top_bottom": pol["P_top_bottom_cm2"],
        "layer_dos_outer_inner": pol["P_outer_inner_cm2"],
        "layer_dos_dipole": pol["P_dipole_cm2"],
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
        ax.set_ylabel(r"$E+\Sigma_f-\mu$ (meV)")
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


def prepare_contour_data(args, ham_K, ham_Kp, U_meV, scf_kpts, evals_K, evals_Kp):
    if args.contour_n1 > 0 and args.contour_n2 > 0:
        kpts, _ = make_uniform_mbz_grid(ham_K.geom, args.contour_n1, args.contour_n2)
        evals_K = diagonalize_evals_only(ham_K, kpts, U_meV)
        evals_Kp = diagonalize_evals_only(ham_Kp, kpts, U_meV)
    else:
        kpts = scf_kpts
    return kpts, evals_K, evals_Kp, contour_band_indices(evals_K.shape[1], args.contour_band_count)


def write_scf_point_summary(out, row, ham):
    write_json(out / "summary.json", {"stage": "scf_before_stoner", "point": scf_point_record(row, ham),
                                     "hamiltonian": hamiltonian_metadata(ham)})


def save_scf_detail_outputs(out, row, args, ham_K, ham_Kp, kpts, weights, scf, evals_K, evals_Kp):
    """Save the unpolarized stage once, before Stoner; return data for its plots."""
    out.mkdir(parents=True, exist_ok=True)
    write_scf_point_summary(out, row, ham_K)
    pd.DataFrame(scf.history).to_csv(out / "scf_history.csv", index=False)
    layer_dos = np.array([row[f"scf_layer_dos_L{i}"] for i in range(1, 4 + 1)])
    pd.DataFrame({"layer": np.arange(1, 4 + 1), "U_meV": scf.U_meV,
                  "n_cm2": scf.n_layer_cm2, "layer_dos": layer_dos}).to_csv(out / "layers.csv", index=False)
    full_mu = float(row["full_mu_meV"])
    np.savez(out / "spectrum.npz", **hamiltonian_npz_metadata(ham_K),
             **displacement_metadata(args.D_sign), stage="scf_before_stoner", kpts=kpts, weights=weights,
             U_meV=scf.U_meV, scf_mu_meV=scf.mu_meV, full_mu_meV=full_mu,
             evals_K_meV=evals_K, evals_Kp_meV=evals_Kp, n_layer_cm2=scf.n_layer_cm2,
             layer_dos=layer_dos)
    centers = np.linspace(args.dos_emin_meV, args.dos_emax_meV, args.dos_bins)
    shifted = [evals_K - full_mu, evals_Kp - full_mu]
    dos = 2 * gaussian_dos_curve(shifted, weights, centers, args.dos_sigma_meV)
    np.savez(out / "dos.npz", **hamiltonian_npz_metadata(ham_K), stage="scf_before_stoner",
             energy_meV=centers, dos=dos, full_mu_meV=full_mu, scf_mu_meV=scf.mu_meV,
             dos_sigma_meV=args.dos_sigma_meV)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(centers, dos, lw=1.2)
    ax.axvline(0, color=".5", ls="--", lw=.8)
    ax.set(xlabel=r"$E-\mu_{\mathrm{full}}$ (meV)", ylabel=r"DOS (A$^{-2}$ meV$^{-1}$)",
           title=f"SCF before Stoner: ABBA")
    fig.tight_layout()
    fig.savefig(out / "dos.png", dpi=220)
    plt.close(fig)
    plot_layer_profile(scf.U_meV, scf.n_layer_cm2, out=str(out / "layer_profile.png"))
    save_layer_dos_bars(out, layer_dos, filename="layer_dos.png")
    bands_data = []
    for valley_name, ham in [("K", ham_K), ("Kp", ham_Kp)]:
        dist, bands, ticks, labels = compute_full_bands_along_path(ham, scf.U_meV, points_per_segment=args.points_per_segment)
        np.savez(out / f"full_bands_{valley_name}.npz", **hamiltonian_npz_metadata(ham),
                 dist=dist, bands=bands, ticks=ticks, labels=labels, mu_meV=full_mu)
        plot_bands(dist, bands, ticks, labels, mu_meV=full_mu, n_show=args.n_show_full,
                   title=f"SCF bands {valley_name}: ABBA", out=str(out / f"full_bands_{valley_name}.png"))
        bands_data.append((dist, bands, ticks, labels))
    contours = prepare_contour_data(args, ham_K, ham_Kp, scf.U_meV, kpts, evals_K, evals_Kp)
    contour_kpts, contour_K, contour_Kp, indices = contours
    np.savez(out / "full_band_contours_before_stoner.npz", **hamiltonian_npz_metadata(ham_K),
             kpts=contour_kpts, evals_K_meV=contour_K, evals_Kp_meV=contour_Kp,
             full_mu_meV=full_mu, band_indices=np.asarray(indices))
    plot_contour_panels(out, "full_band_contours_before_stoner.png", contour_kpts,
                        [("K before Stoner", contour_K, full_mu), ("Kp before Stoner", contour_Kp, full_mu)],
                        indices, "SCF contours before Stoner")
    return {"dos": (centers, dos), "bands": bands_data, "contours": contours}


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
    contour_data=None,
) -> None:
    if contour_data is None:
        contour_data = prepare_contour_data(args, ham_K, ham_Kp, U_meV, scf_kpts, evals_K, evals_Kp)
    contour_kpts, contour_evals_K, contour_evals_Kp, band_indices = contour_data

    np.savez(
        out / "full_band_contours_stoner_flavors.npz",
        **hamiltonian_npz_metadata(ham_K),
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


def clear_point_outputs(out: Path) -> None:
    """Remove only this workflow's named outputs before reusing a directory."""
    generated = (
        "summary.json", "run_summary.json", "error.json", "stoner_result.json",
        "stoner_diagnostics.json", "stoner_fermi_level.json", "stoner_layers.csv",
        "stoner_flavor_layers.csv", "full_band_stoner_cache.npz",
        "full_band_stoner_dos.npz", "full_band_stoner_dos.png",
        "stoner_layer_profile.png", "stoner_layer_dos.png",
        "stoner_flavor_layer_profile.png", "stoner_flavor_layer_dos.png",
        "flavor_occupations.png", "stoner_flavor_bands.png", "stoner_flavor_bands.npz",
        "full_band_contours_stoner_flavors.npz", "full_band_contours_stoner_flavors.png",
    )
    baseline = (
        "summary.json", "diagnostics.json", "scf_history.csv", "layers.csv", "spectrum.npz",
        "dos.npz", "dos.png", "layer_profile.png", "layer_dos.png",
        "full_bands_K.npz", "full_bands_Kp.npz", "full_bands_K.png", "full_bands_Kp.png",
        "full_band_contours_before_stoner.npz", "full_band_contours_before_stoner.png",
    )
    for directory, names in ((out, generated), (out / "scf", baseline)):
        for name in names:
            target = directory / name
            if target.is_file():
                target.unlink()


def run_point(task: dict[str, Any], args: argparse.Namespace, detail_dir: Path | None = None) -> dict[str, Any]:
    started = time.time()
    n_cm2 = float(task["n_cm2"])
    D_Vnm = float(task["D_Vnm"])
    label = str(task.get("label", "point"))
    row = {}
    ham_K = None
    try:
        if detail_dir is not None:
            clear_point_outputs(detail_dir)
        valley_scf = scf_valley(args)
        ham_scf = TDBGContinuumHamiltonian(params_from_args(args, valley_scf))
        ham_K = TDBGContinuumHamiltonian(params_from_args(args, +1))
        ham_Kp = TDBGContinuumHamiltonian(params_from_args(args, -1))
        kpts, weights = make_uniform_mbz_grid(ham_scf.geom, args.grid_n1, args.grid_n2)
        scf, scf_meta = run_scf_reference(args, ham_scf, kpts, weights, n_cm2, D_Vnm)
        scf_tolerance = args.full_tol_meV
        scf_checks = scf_convergence_diagnostics(scf, n_cm2, scf_tolerance)
        solver_metadata = {
            "scf_solver_method": args.mixer,
            **scf_checks,
        }
        row = {
            **hamiltonian_metadata(ham_scf), **displacement_metadata(args.D_sign),
            # Export readiness requires completed DOS and density calculations.
            # A converged solver alone does not make a complete map point.
            **solver_metadata, "scf_status": "failed",
            "scf_source": scf_meta["scf_source"], "scf_reference_mode": scf_meta["reference_mode"],
            "scf_residual_meV": float(scf.residual_meV),
            "scf_iterations": int(scf.iterations),
            "scf_mu_meV": float(scf.mu_meV),
            **{f"U{layer + 1}_meV": float(value) for layer, value in enumerate(scf.U_meV)},
            **{f"n{layer + 1}_cm2": float(value) for layer, value in enumerate(scf.n_layer_cm2)},
        }
        if detail_dir is not None:
            # Diagnostic states are retained even when physical analysis is
            # rejected. No DOS, gap, band figure or Stoner is computed here.
            write_json(detail_dir / "scf" / "diagnostics.json", {
                **row, "stage": "scf_diagnostics", "U_meV": np.asarray(scf.U_meV).tolist(),
                "n_layer_cm2": np.asarray(scf.n_layer_cm2).tolist(), "history": scf.history,
            })
        try:
            require_converged_scf(scf, n_cm2, scf_tolerance)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; DOS/bands/Stoner skipped") from exc
        if valley_scf == 1 and scf.evals is not None and scf.evecs is not None:
            evals_K, evecs_K = scf.evals, scf.evecs
        else:
            evals_K, evecs_K = ham_K.diagonalize(kpts, scf.U_meV)
        if valley_scf == -1 and scf.evals is not None and scf.evecs is not None:
            evals_Kp, evecs_Kp = scf.evals, scf.evecs
        else:
            evals_Kp, evecs_Kp = ham_Kp.diagonalize(kpts, scf.U_meV)

        A_M_A2 = moire_cell_area_A2_from_weights(weights)
        nu_total = density_cm2_to_nu_total(n_cm2, A_M_A2)
        layer_masks = ham_K.layer_projectors_diagonal()
        chosen_evals, chosen_evecs = (evals_K, evecs_K) if valley_scf == 1 else (evals_Kp, evecs_Kp)
        solver_metadata.update(full_hartree_consistency(
            chosen_evals, chosen_evecs, weights, layer_masks, scf.U_meV, n_cm2,
            D_Vnm, args.kBT_meV, args.eps_perp, args.d_layer_nm, args.D_sign, scf_tolerance))
        row.update(solver_metadata)
        if not solver_metadata["scf_full_hartree_consistent"]:
            row.update(scf_converged=False, scf_valid=False)
            raise RuntimeError("Independent full Hartree residual or charge check failed; DOS/bands/Stoner skipped")
        full_dos = full_band_dos_summary(
            evals_K, evals_Kp, weights, n_cm2=n_cm2,
            sigma_meV=args.dos_sigma_meV, kBT_meV=args.kBT_meV,
        )
        full_mu = full_dos["full_mu_meV"]
        # Two valleys with two spins each, at the density-matched full-band mu.
        scf_layer_dos = 2 * (
            layer_dos_at_mu_gaussian(evals_K, evecs_K, weights, layer_masks, full_mu, args.dos_sigma_meV)
            + layer_dos_at_mu_gaussian(evals_Kp, evecs_Kp, weights, layer_masks, full_mu, args.dos_sigma_meV)
        )
        n_layer_cm2 = np.asarray(scf.n_layer_cm2, dtype=float)
        row = {
            "status": "ok", "scf_status": "ok",
            **solver_metadata,
            **hamiltonian_metadata(ham_scf), **displacement_metadata(args.D_sign),
            "label": label, "n_index": task.get("n_index", 0), "D_index": task.get("D_index", 0),
            "n_cm2": n_cm2, "D_Vnm": D_Vnm, "nu_total": float(nu_total), "A_M_A2": float(A_M_A2),
            "scf_source": scf_meta["scf_source"], "scf_reference_mode": scf_meta["reference_mode"],
            "scf_valley": int(valley_scf), "eps_perp": float(args.eps_perp), "d_layer_nm": float(args.d_layer_nm),
            "scf_mu_meV": float(scf.mu_meV), "scf_converged": bool(scf.converged),
            "scf_residual_meV": float(scf.residual_meV), "scf_iterations": int(scf.iterations),
            "full_converged": bool(scf.converged), "full_residual_meV": float(scf.residual_meV),
            "full_iterations": int(scf.iterations), **full_dos,
            "dos_mu_scf": full_dos["dos_mu_full_at_full_mu"],
            "dos_mu_full_scf": full_dos["dos_mu_full_at_full_mu"],
            **{f"U{layer + 1}_meV": float(value) for layer, value in enumerate(scf.U_meV)},
            **{f"n{layer + 1}_cm2": float(value) for layer, value in enumerate(n_layer_cm2)},
            **layer_polarizations(n_layer_cm2),
            **{f"scf_layer_dos_L{layer + 1}": float(value) for layer, value in enumerate(scf_layer_dos)},
            "scf_layer_dos_total": float(np.sum(scf_layer_dos)),
            **{f"scf_{key}": value for key, value in layer_dos_polarizations(scf_layer_dos).items()},
        }
        detail_data = None
        if detail_dir is not None:
            detail_data = save_scf_detail_outputs(
                detail_dir / "scf", row, args, ham_K, ham_Kp, kpts, weights, scf, evals_K, evals_Kp,
            )
        tables = restrict_tables_to_carrier_sector(
            build_full_band_flavor_tables(evals_K, evals_Kp, weights, A_M_A2),
            nu_total,
        )
        stoner_params = stoner_params_from_args(args)
        stoner = solve_stoner_fixed_nu(nu_total, tables, stoner_params, A_M_A2)
        nu_f = np.asarray(stoner.nu_f, dtype=float)
        stoner_valid = bool(
            stoner.success and nu_f.shape == (4,) and np.isfinite(nu_f).all()
            and np.isfinite(stoner.energy_meV_per_cell)
            and abs(float(np.sum(nu_f)) - nu_total) < 1e-8
            and all(table.nu_min - 1e-8 <= value <= table.nu_max + 1e-8
                    for value, table in zip(nu_f, tables))
        )
        row["stoner_success"] = stoner_valid
        if detail_dir is not None:
            write_json(detail_dir / "stoner_diagnostics.json", {
                "accepted": stoner_valid, "optimizer_success": bool(stoner.success),
                "nu_total": nu_total, "nu_f": nu_f.tolist(),
                "energy_meV_per_cell": float(stoner.energy_meV_per_cell),
                "seed_name": stoner.seed_name,
                "local_minima": [
                    {**candidate, "nu_f": np.asarray(candidate["nu_f"]).tolist()}
                    for candidate in stoner.local_minima
                ],
            })
        if not stoner_valid:
            raise RuntimeError("Stoner did not converge to a finite, filling-conserving solution; "
                               "Stoner DOS/layer/band outputs skipped; SCF outputs retained")
        fermi_level = stoner_mean_field_fermi_level(stoner, tables, stoner_params, A_M_A2)
        stoner.mean_field_fermi_level = fermi_level
        row.update({
            "stoner_optimizer_success": True,
            "stoner_kkt_valid": fermi_level["optimization_valid"],
            "stoner_dos_valid": fermi_level["valid"],
            "stoner_mu_common_meV": fermi_level["mu_common_meV"],
            "stoner_mu_interval_mismatch_meV": fermi_level["interval_mismatch_meV"],
            "stoner_mu_constraint_mismatch_meV": fermi_level["constraint_mismatch_meV"],
            "stoner_mu_convention": fermi_level["convention"],
            "stoner_fermi_level_reason": fermi_level["reason"],
            **{f"stoner_self_energy_{flavor}_meV": float(value)
               for flavor, value in zip(FLAVOR_NAMES, fermi_level["self_energy_f_meV"])},
        })
        if detail_dir is not None:
            write_json(detail_dir / "stoner_fermi_level.json", fermi_level)
        if not fermi_level["optimization_valid"]:
            row["stoner_success"] = False
            raise RuntimeError("Stoner optimizer violates constrained stationarity; "
                               "Stoner outputs skipped; SCF outputs retained")
        # Retain carrier-sector occupations/energy if that restricted model
        # has no physical common Fermi level. DOS and spectral plots are NaN.
        stoner.mu_f_meV = np.asarray(fermi_level["mu_f_meV"], dtype=float)
        mu_f = stoner_mu_by_flavor(stoner, tables)
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
        ) if fermi_level["valid"] else np.full((4, 4), np.nan)
        stoner_layer_dos = np.sum(stoner_flavor_layer_dos, axis=0)

        stoner_pol = {f"stoner_{key}": value for key, value in layer_polarizations(stoner_n_layer_cm2).items()}
        stoner_layer_dos_pol = {
            "stoner_layer_dos_total": float(np.sum(stoner_layer_dos)),
            **{f"stoner_{key}": value for key, value in layer_dos_polarizations(stoner_layer_dos).items()},
        }
        row.update({
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
            "dos_mu_stoner": (explicit_flavor_dos_at_mu(evals_K, evals_Kp, weights, mu_f, args.dos_sigma_meV)
                              if fermi_level["valid"] else float("nan")),
            **{f"stoner_n{layer + 1}_cm2": float(value) for layer, value in enumerate(stoner_n_layer_cm2)},
            "stoner_n_total_layer_cm2": float(np.sum(stoner_n_layer_cm2)),
            **{f"stoner_layer_dos_L{layer + 1}": float(value) for layer, value in enumerate(stoner_layer_dos)},
            "runtime_s": float(time.time() - started),
        })
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
                detail_data,
            )
            write_scf_point_summary(detail_dir / "scf", row, ham_K)
        return row
    except Exception as exc:
        error = {
            **row,
            "status": "failed",
            "scf_status": row.get("scf_status", "failed"),
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
            if ham_K is not None and (detail_dir / "scf").is_dir():
                write_scf_point_summary(detail_dir / "scf", error, ham_K)
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
    detail_data,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stoner_dict = stoner_result_to_dict(stoner, tables)
    write_json(out / "summary.json", {
        "point": row,
        "stoner": stoner_dict,
        "hamiltonian": hamiltonian_metadata(ham_K),
        "displacement": displacement_metadata(args.D_sign),
    })
    write_json(out / "stoner_result.json", stoner_dict)
    stoner_n_layer_cm2 = np.asarray([row[f"stoner_n{layer}_cm2"] for layer in range(1, 4 + 1)], dtype=float)
    stoner_layer_dos = np.asarray([row[f"stoner_layer_dos_L{layer}"] for layer in range(1, 4 + 1)], dtype=float)
    pd.DataFrame(
        {
            "layer": np.arange(1, 4 + 1),
            "U_meV": scf.U_meV,
            "scf_n_cm2": scf.n_layer_cm2,
            "stoner_n_cm2": stoner_n_layer_cm2,
            "stoner_layer_dos": stoner_layer_dos,
        }
    ).to_csv(out / "stoner_layers.csv", index=False)
    stoner_flavor_n_layer_cm2 = np.asarray(
        [[row[f"stoner_{flavor}_n{layer}_cm2"] for layer in range(1, 4 + 1)] for flavor in FLAVOR_NAMES],
        dtype=float,
    )
    stoner_flavor_layer_dos = np.asarray(
        [[row[f"stoner_{flavor}_layer_dos_L{layer}"] for layer in range(1, 4 + 1)] for flavor in FLAVOR_NAMES],
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
        **hamiltonian_npz_metadata(ham_K),
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
        scf_layer_dos=np.array([row[f"scf_layer_dos_L{i}"] for i in range(1, 4 + 1)]),
        stoner_n_layer_cm2=stoner_n_layer_cm2,
        stoner_layer_dos=stoner_layer_dos,
        stoner_flavor_n_layer_cm2=stoner_flavor_n_layer_cm2,
        stoner_flavor_layer_dos=stoner_flavor_layer_dos,
        nu_f=stoner.nu_f,
        mu_f_meV=stoner_mu_by_flavor(stoner, tables),
        stoner_dos_valid=bool(row["stoner_dos_valid"]),
        stoner_mu_common_meV=row["stoner_mu_common_meV"],
        stoner_self_energy_f_meV=np.asarray(stoner.mean_field_fermi_level["self_energy_f_meV"]),
        stoner_mu_convention=row["stoner_mu_convention"],
    )

    plot_layer_profile(scf.U_meV, stoner_n_layer_cm2, out=str(out / "stoner_layer_profile.png"))
    if row["stoner_dos_valid"]:
        save_layer_dos_bars(out, stoner_layer_dos)
    save_flavor_layer_bars(out, stoner_flavor_n_layer_cm2, r"$n_l$ (cm$^{-2}$)", "stoner_flavor_layer_profile.png")
    if row["stoner_dos_valid"]:
        save_flavor_layer_bars(out, stoner_flavor_layer_dos, r"DOS (A$^{-2}$ meV$^{-1}$)", "stoner_flavor_layer_dos.png")
    save_flavor_occupations(out, stoner.nu_f)
    if not row["stoner_dos_valid"]:
        return
    mu_f = stoner_mu_by_flavor(stoner, tables)
    full_mu = float(row["full_mu_meV"])
    save_dos_figure(out, evals_K, evals_Kp, weights, full_mu, mu_f, args, ham_K, dos_data=detail_data["dos"])
    dist_K, bands_K, ticks_K, labels_K = detail_data["bands"][0]
    _, bands_Kp, _, _ = detail_data["bands"][1]
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
        contour_data=detail_data["contours"],
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run full self-consistent Hartree then full-band four-flavor Stoner at one n-D point.")
    add_hamiltonian_args(ap)
    add_full_scf_args(ap)
    add_stoner_args(ap)
    add_analysis_args(ap)
    ap.add_argument("--out", type=str, default="outputs/full_band_stoner")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    row = run_point({"n_cm2": args.n_cm2, "D_Vnm": args.D_Vnm, "label": "single_point"}, args, detail_dir=out)
    write_json(out / "run_summary.json", row)
    print(json.dumps(row, indent=2))
    return 1 if row.get("status") == "failed" or not row.get("stoner_dos_valid", False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
