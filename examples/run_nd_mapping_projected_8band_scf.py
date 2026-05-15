#!/usr/bin/env python3
"""Parallel n-D mapping with projected self-consistent TDBG calculations.

All n-D map scalar observables are computed from the final projected model.
Full Hamiltonian data are saved only for selected detail points as comparison
band structures.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

# Keep each worker single-threaded at the BLAS/OpenMP layer. Parallelism is
# handled by ProcessPoolExecutor over independent n-D points.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

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
    ProjectedSCFConfig,
    ProjectedSCFSolver,
    TDBGContinuumHamiltonian,
    TDBGParameters,
    make_uniform_mbz_grid,
)
from tdbg_scf.density import dos_at_mu_gaussian
from tdbg_scf.plotting import (
    compute_full_bands_along_path,
    compute_projected_bands_along_path,
    plot_bands,
    plot_layer_profile,
    save_dos_plot,
)
from tdbg_scf.solver_projected import build_projected_model_at_mu, projected_eigensystem


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def make_axis_values(explicit: str | None, lo: float, hi: float, count: int) -> list[float]:
    values = parse_float_list(explicit)
    if values is not None:
        return values
    if count < 1:
        raise ValueError("axis count must be >= 1")
    if count == 1:
        return [0.5 * (lo + hi)]
    return np.linspace(lo, hi, count).astype(float).tolist()


def evenly_spaced_indices(count: int, target_count: int) -> set[int]:
    if count <= 0 or target_count <= 0:
        return set()
    if target_count >= count:
        return set(range(count))
    return {int(round(x)) for x in np.linspace(0, count - 1, target_count)}


def point_label(n_index: int, d_index: int, n_cm2: float, d_vnm: float) -> str:
    n_scaled = n_cm2 / 1e12
    return f"i{n_index:03d}_j{d_index:03d}_n{n_scaled:+.4f}e12_D{d_vnm:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")


def params_from_config(config: dict[str, Any]) -> TDBGParameters:
    h = config["hamiltonian"]
    return TDBGParameters(
        theta_deg=h["theta_deg"],
        cutoff=h["cutoff"],
        valley=h["valley"],
        omega_meV=h["omega_meV"],
        wAA=h["wAA"],
        wAB=h["wAB"],
        a_cc_A=h["a_cc_A"],
        gamma0_meV=h["gamma0_meV"],
        gamma1_meV=h["gamma1_meV"],
        gamma3_meV=h["gamma3_meV"],
        gamma4_meV=h["gamma4_meV"],
        sublattice_Z_meV=h["sublattice_Z_meV"],
    )


def gap_around_mu(evals: np.ndarray, mu_mev: float) -> float:
    shifted = np.asarray(evals, dtype=float) - float(mu_mev)
    below = shifted[shifted <= 0.0]
    above = shifted[shifted >= 0.0]
    if below.size == 0 or above.size == 0:
        return float("nan")
    return float(np.min(above) - np.max(below))


def layer_polarizations(n_layer_cm2: np.ndarray) -> dict[str, float]:
    n1, n2, n3, n4 = [float(x) for x in n_layer_cm2]
    return {
        "P_top_bottom_cm2": (n1 + n2) - (n3 + n4),
        "P_outer_inner_cm2": (n1 + n4) - (n2 + n3),
        "P_dipole_cm2": 1.5 * n1 + 0.5 * n2 - 0.5 * n3 - 1.5 * n4,
    }


def scalar_result(task: dict[str, Any], config: dict[str, Any], result: Any, weights: np.ndarray) -> dict[str, Any]:
    n_layer = np.asarray(result.n_layer_cm2, dtype=float)
    pol = layer_polarizations(n_layer)
    diagnostics = result.model.selection_diagnostics or {}
    row: dict[str, Any] = {
        "status": "ok",
        "n_index": task["n_index"],
        "D_index": task["D_index"],
        "n_cm2": task["n_cm2"],
        "D_Vnm": task["D_Vnm"],
        "mu_meV": float(result.mu_meV),
        "gap_projected_meV": gap_around_mu(result.evals, result.mu_meV),
        "dos_mu_projected": dos_at_mu_gaussian(
            result.evals,
            weights,
            result.mu_meV,
            sigma_meV=config["analysis"]["dos_sigma_meV"],
            degeneracy=config["scf"]["degeneracy"],
        ),
        "U1_meV": float(result.U_meV[0]),
        "U2_meV": float(result.U_meV[1]),
        "U3_meV": float(result.U_meV[2]),
        "U4_meV": float(result.U_meV[3]),
        "n1_cm2": float(n_layer[0]),
        "n2_cm2": float(n_layer[1]),
        "n3_cm2": float(n_layer[2]),
        "n4_cm2": float(n_layer[3]),
        "n_total_layer_cm2": float(np.sum(n_layer)),
        "converged": bool(result.converged),
        "residual_meV": float(result.residual_meV),
        "iterations": int(result.iterations),
        "isolation_gap_meV": float(result.model.isolation_gap_meV),
        "max_abs_delta_U_meV": float(np.max(np.abs(result.U_meV - result.model.U_ref_meV))),
        "selection_unique_windows": diagnostics.get("unique_selected_windows"),
        "selection_min_overlap_score_gap": diagnostics.get("min_overlap_score_gap"),
        "selection_min_kept_overlap_score": diagnostics.get("min_kept_overlap_score"),
        "selection_ambiguous_overlap_points": diagnostics.get("ambiguous_overlap_points"),
    }
    row.update(pol)
    return row


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_history_and_layers(out: Path, result: Any) -> None:
    pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(
        {
            "layer": [1, 2, 3, 4],
            "U_meV": result.U_meV,
            "n_cm2": result.n_layer_cm2,
        }
    ).to_csv(out / "layers.csv", index=False)


def save_projected_model_npz(out: Path, result: Any, weights: np.ndarray, kpts: np.ndarray) -> None:
    np.savez(
        out / "projected_model.npz",
        U_ref_meV=result.model.U_ref_meV,
        U_scf_meV=result.U_meV,
        mu_ref_meV=result.model.mu_ref_meV,
        mu_scf_meV=result.mu_meV,
        miniband_energies0_meV=result.model.energies0_meV,
        final_evals_meV=result.evals,
        layer_mats=result.model.layer_mats,
        remote_density_per_k_layer=result.model.remote_density_per_k_layer,
        selected_indices=result.model.selected_indices,
        weights=weights,
        kpts=kpts,
    )


def save_band_data_and_figures(out: Path, ham: TDBGContinuumHamiltonian, result: Any, config: dict[str, Any]) -> None:
    analysis = config["analysis"]
    n_show_full = analysis["n_show_full"]
    n_active = config["scf"]["n_active"]
    points_per_segment = analysis["points_per_segment"]

    dist, ref_bands, ticks, labels = compute_full_bands_along_path(
        ham,
        result.model.U_ref_meV,
        points_per_segment=points_per_segment,
    )
    np.savez(out / "bands_reference_full.npz", dist=dist, bands=ref_bands, ticks=ticks, labels=labels, mu_meV=result.model.mu_ref_meV)
    plot_bands(
        dist,
        ref_bands,
        ticks,
        labels,
        mu_meV=result.model.mu_ref_meV,
        n_show=n_show_full,
        title="Full TDBG reference",
        out=str(out / "reference_full_bands.png"),
    )

    dist, final_full_bands, ticks, labels = compute_full_bands_along_path(
        ham,
        result.U_meV,
        points_per_segment=points_per_segment,
    )
    np.savez(out / "bands_final_full.npz", dist=dist, bands=final_full_bands, ticks=ticks, labels=labels, mu_meV=result.mu_meV)
    plot_bands(
        dist,
        final_full_bands,
        ticks,
        labels,
        mu_meV=result.mu_meV,
        n_show=n_show_full,
        title="Full Hamiltonian at projected-SCF U",
        out=str(out / "final_full_bands_at_projected_U.png"),
    )

    dist, projected_bands, ticks, labels = compute_projected_bands_along_path(
        ham,
        U_ref_meV=result.model.U_ref_meV,
        U_meV=result.U_meV,
        mu_ref_meV=result.model.mu_ref_meV,
        n_active=n_active,
        selection=config["scf"]["selection"],
        points_per_segment=points_per_segment,
    )
    np.savez(out / "bands_projected.npz", dist=dist, bands=projected_bands, ticks=ticks, labels=labels, mu_meV=result.mu_meV)
    plot_bands(
        dist,
        projected_bands,
        ticks,
        labels,
        mu_meV=result.mu_meV,
        n_show=n_active,
        title=f"Projected {n_active}-band at projected-SCF U",
        out=str(out / "projected_bands.png"),
    )


def contour_band_indices(n_bands: int, count: int) -> list[int]:
    count = max(1, min(count, n_bands))
    center = n_bands // 2
    start = max(0, center - count // 2)
    stop = min(n_bands, start + count)
    start = max(0, stop - count)
    return list(range(start, stop))


def save_projected_band_contours(
    out: Path,
    ham: TDBGContinuumHamiltonian,
    result: Any,
    scf_kpts: np.ndarray,
    config: dict[str, Any],
) -> None:
    analysis = config["analysis"]
    contour_n1 = analysis["contour_n1"]
    contour_n2 = analysis["contour_n2"]
    if contour_n1 > 0 and contour_n2 > 0:
        kpts, _ = make_uniform_mbz_grid(ham.geom, contour_n1, contour_n2)
        model = build_projected_model_at_mu(
            ham,
            kpts,
            U_ref_meV=result.model.U_ref_meV,
            mu_ref_meV=result.model.mu_ref_meV,
            n_active=config["scf"]["n_active"],
            selection=config["scf"]["selection"],
        )
        evals, _ = projected_eigensystem(model, result.U_meV)
    else:
        kpts = scf_kpts
        evals = result.evals

    band_indices = contour_band_indices(evals.shape[1], analysis["contour_band_count"])
    np.savez(
        out / "band_contours_projected.npz",
        kpts=kpts,
        evals=evals,
        mu_meV=result.mu_meV,
        band_indices=np.asarray(band_indices, dtype=int),
    )

    n_panels = len(band_indices)
    ncols = int(math.ceil(math.sqrt(n_panels)))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    x = kpts[:, 0]
    y = kpts[:, 1]
    for ax, band in zip(axes.ravel(), band_indices):
        z = evals[:, band] - result.mu_meV
        if len(kpts) >= 4:
            levels = np.linspace(float(np.min(z)), float(np.max(z)), 41)
            cf = ax.tricontourf(x, y, z, levels=levels, cmap="RdBu_r")
            ax.tricontour(x, y, z, levels=[0.0], colors="k", linewidths=0.8)
            fig.colorbar(cf, ax=ax, shrink=0.82)
        else:
            sc = ax.scatter(x, y, c=z, cmap="RdBu_r")
            fig.colorbar(sc, ax=ax, shrink=0.82)
        ax.set_title(f"band {band}: E-mu")
        ax.set_xlabel(r"$k_x$ ($A^{-1}$)")
        ax.set_ylabel(r"$k_y$ ($A^{-1}$)")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "band_contours_projected.png", dpi=220)
    plt.close(fig)


def save_detail_outputs(
    detail_dir: Path,
    task: dict[str, Any],
    config: dict[str, Any],
    ham: TDBGContinuumHamiltonian,
    result: Any,
    weights: np.ndarray,
    kpts: np.ndarray,
    scalar: dict[str, Any],
) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    point_summary = {
        "point": task,
        "scalar": scalar,
        "hamiltonian": config["hamiltonian"],
        "scf": config["scf"],
        "analysis": config["analysis"],
        "U_ref_meV": result.model.U_ref_meV.tolist(),
        "U_scf_meV": result.U_meV.tolist(),
        "mu_ref_meV": float(result.model.mu_ref_meV),
        "mu_scf_meV": float(result.mu_meV),
        "selection_diagnostics": result.model.selection_diagnostics or {},
    }
    write_json(detail_dir / "summary.json", point_summary)
    write_history_and_layers(detail_dir, result)
    save_projected_model_npz(detail_dir, result, weights, kpts)
    plot_layer_profile(result.U_meV, result.n_layer_cm2, out=str(detail_dir / "layer_profile.png"))
    save_dos_plot(
        result.evals,
        weights,
        result.mu_meV,
        config["scf"]["degeneracy"],
        out=str(detail_dir / "projected_dos.png"),
        emin=config["analysis"]["dos_emin_meV"],
        emax=config["analysis"]["dos_emax_meV"],
        bins=config["analysis"]["dos_bins"],
    )
    save_band_data_and_figures(detail_dir, ham, result, config)
    save_projected_band_contours(detail_dir, ham, result, kpts, config)


def run_one_point(task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    label = task["label"]
    try:
        params = params_from_config(config)
        ham = TDBGContinuumHamiltonian(params)
        grid_n1 = config["scf"]["grid_n1"]
        grid_n2 = config["scf"]["grid_n2"]
        kpts, weights = make_uniform_mbz_grid(ham.geom, grid_n1, grid_n2)
        mixer_kwargs = (
            {"alpha": config["scf"]["alpha"]}
            if config["scf"]["mixer"] == "linear"
            else {
                "beta": config["scf"]["anderson_beta"],
                "memory": config["scf"]["anderson_memory"],
                "fallback_alpha": config["scf"]["alpha"],
            }
        )
        full_cfg = FullSCFConfig(
            target_density_cm2=task["n_cm2"],
            D_Vnm=task["D_Vnm"],
            degeneracy=config["scf"]["degeneracy"],
            kBT_meV=config["scf"]["kBT_meV"],
            eps_perp=config["scf"]["eps_perp"],
            d_layer_nm=config["scf"]["d_layer_nm"],
            D_sign=config["scf"]["D_sign"],
            max_iter=config["scf"]["full_max_iter"],
            tol_meV=config["scf"]["full_tol_meV"],
            mixer=config["scf"]["mixer"],
            mixer_kwargs=mixer_kwargs,
            keep_eigensystem=False,
        )
        supplied_u_ref = config["scf"].get("supplied_U_ref_meV")
        cfg = ProjectedSCFConfig(
            target_density_cm2=task["n_cm2"],
            D_Vnm=task["D_Vnm"],
            n_active=config["scf"]["n_active"],
            degeneracy=config["scf"]["degeneracy"],
            kBT_meV=config["scf"]["kBT_meV"],
            max_iter=config["scf"]["max_iter"],
            tol_meV=config["scf"]["tol_meV"],
            eps_perp=config["scf"]["eps_perp"],
            d_layer_nm=config["scf"]["d_layer_nm"],
            D_sign=config["scf"]["D_sign"],
            mixer=config["scf"]["mixer"],
            mixer_kwargs=mixer_kwargs,
            selection=config["scf"]["selection"],
            grid_shape=(grid_n1, grid_n2),
            projector_refreshes=config["scf"]["projector_refreshes"],
            reference_mode=config["scf"]["reference_mode"],
            supplied_U_ref_meV=np.asarray(supplied_u_ref, dtype=float) if supplied_u_ref is not None else None,
            full_scf_config=full_cfg,
        )
        result = ProjectedSCFSolver(ham, kpts, weights, grid_shape=(grid_n1, grid_n2)).solve(cfg)
        row = scalar_result(task, config, result, weights)
        row["label"] = label
        row["runtime_s"] = float(time.time() - started)
        row["detail_saved"] = bool(task["detail"])
        if task["detail"]:
            detail_dir = Path(config["paths"]["details_dir"]) / label
            save_detail_outputs(detail_dir, task, config, ham, result, weights, kpts, row)
            row["detail_dir"] = str(detail_dir)
        return row
    except Exception as exc:
        error = {
            "status": "failed",
            "label": label,
            "n_index": task["n_index"],
            "D_index": task["D_index"],
            "n_cm2": task["n_cm2"],
            "D_Vnm": task["D_Vnm"],
            "detail_saved": bool(task["detail"]),
            "runtime_s": float(time.time() - started),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        if task["detail"]:
            detail_dir = Path(config["paths"]["details_dir"]) / label
            detail_dir.mkdir(parents=True, exist_ok=True)
            write_json(detail_dir / "error.json", error)
        return error


def plot_nd_map(df: pd.DataFrame, n_values: list[float], d_values: list[float], key: str, out: Path, title: str) -> None:
    if key not in df.columns:
        return
    arr = np.full((len(n_values), len(d_values)), np.nan, dtype=float)
    for _, row in df[df["status"] == "ok"].iterrows():
        arr[int(row["n_index"]), int(row["D_index"])] = float(row[key])
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    d_min, d_max = min(d_values), max(d_values)
    n_min, n_max = min(n_values) / 1e12, max(n_values) / 1e12
    if d_min == d_max:
        d_min -= 0.5
        d_max += 0.5
    if n_min == n_max:
        n_min -= 0.5
        n_max += 0.5
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[d_min, d_max, n_min, n_max],
        interpolation="nearest",
    )
    ax.set_xlabel("D (V/nm)")
    ax.set_ylabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_map_outputs(out: Path, rows: list[dict[str, Any]], n_values: list[float], d_values: list[float]) -> None:
    df = pd.DataFrame(rows)
    df = df.sort_values(["n_index", "D_index"], kind="stable")
    df.to_csv(out / "nd_map.csv", index=False)
    write_json(out / "nd_map.json", df.to_dict(orient="records"))
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        np.savez(
            out / "nd_map_scalars.npz",
            n_values_cm2=np.asarray(n_values, dtype=float),
            D_values_Vnm=np.asarray(d_values, dtype=float),
            rows=ok.to_records(index=False),
        )
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("gap_projected_meV", "Projected gap around mu (meV)"),
        ("dos_mu_projected", "Projected DOS at mu"),
        ("P_outer_inner_cm2", "Outer-inner layer polarization (cm^-2)"),
        ("P_top_bottom_cm2", "Top-bottom layer polarization (cm^-2)"),
        ("P_dipole_cm2", "Layer dipole scalar (cm^-2)"),
        ("residual_meV", "Projected SCF residual (meV)"),
        ("iterations", "Projected SCF iterations"),
    ]
    for key, title in plot_specs:
        plot_nd_map(df, n_values, d_values, key, fig_dir / f"nd_map_{key}.png", title)


def build_config(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    supplied = parse_float_list(args.U_ref)
    if supplied is not None and len(supplied) != 4:
        raise ValueError("--U-ref must contain exactly four comma-separated values")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "platform": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "paths": {
            "out": str(out),
            "details_dir": str(out / "details"),
        },
        "hamiltonian": {
            "theta_deg": args.theta_deg,
            "cutoff": args.cutoff,
            "valley": args.valley,
            "omega_meV": args.omega,
            "wAA": args.wAA,
            "wAB": args.wAB,
            "a_cc_A": args.a_cc_A,
            "gamma0_meV": args.gamma0_meV,
            "gamma1_meV": args.gamma1_meV,
            "gamma3_meV": args.gamma3_meV,
            "gamma4_meV": args.gamma4_meV,
            "sublattice_Z_meV": args.Z,
        },
        "scf": {
            "grid_n1": args.grid_n1,
            "grid_n2": args.grid_n2,
            "n_active": args.n_active,
            "selection": args.selection,
            "projector_refreshes": args.projector_refreshes,
            "degeneracy": args.degeneracy,
            "kBT_meV": args.kBT_meV,
            "eps_perp": args.eps_perp,
            "d_layer_nm": args.d_layer_nm,
            "D_sign": args.D_sign,
            "max_iter": args.max_iter,
            "tol_meV": args.tol_meV,
            "full_max_iter": args.full_max_iter,
            "full_tol_meV": args.full_tol_meV,
            "mixer": args.mixer,
            "alpha": args.alpha,
            "anderson_beta": args.anderson_beta,
            "anderson_memory": args.anderson_memory,
            "reference_mode": args.reference_mode,
            "supplied_U_ref_meV": supplied,
        },
        "analysis": {
            "dos_sigma_meV": args.dos_sigma_meV,
            "dos_emin_meV": args.dos_emin_meV,
            "dos_emax_meV": args.dos_emax_meV,
            "dos_bins": args.dos_bins,
            "points_per_segment": args.points_per_segment,
            "n_show_full": args.n_show_full,
            "contour_n1": args.contour_n1,
            "contour_n2": args.contour_n2,
            "contour_band_count": args.contour_band_count,
        },
        "parallel": {
            "max_workers": args.max_workers,
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
    }


def build_tasks(n_values: list[float], d_values: list[float], detail_n_count: int, detail_d_count: int) -> list[dict[str, Any]]:
    detail_i = evenly_spaced_indices(len(n_values), detail_n_count)
    detail_j = evenly_spaced_indices(len(d_values), detail_d_count)
    tasks = []
    for i, n_cm2 in enumerate(n_values):
        for j, d_vnm in enumerate(d_values):
            detail = i in detail_i and j in detail_j
            tasks.append(
                {
                    "n_index": i,
                    "D_index": j,
                    "n_cm2": float(n_cm2),
                    "D_Vnm": float(d_vnm),
                    "detail": detail,
                    "label": point_label(i, j, float(n_cm2), float(d_vnm)),
                }
            )
    return tasks


def resolve_workers(value: str) -> int:
    if value == "auto":
        return max(1, (os.cpu_count() or 2) - 2)
    workers = int(value)
    if workers < 1:
        raise ValueError("--max-workers must be >= 1 or 'auto'")
    return workers


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parallel projected-SCF n-D mapping for ABBA TDBG.")
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=2)
    ap.add_argument("--valley", type=int, default=1)
    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0)
    ap.add_argument("--a-cc-A", type=float, default=1.420)
    ap.add_argument("--gamma0-meV", type=float, default=2610.0)
    ap.add_argument("--gamma1-meV", type=float, default=361.0)
    ap.add_argument("--gamma3-meV", type=float, default=283.0)
    ap.add_argument("--gamma4-meV", type=float, default=138.0)

    ap.add_argument("--n-values-cm2", type=str, default=None, help="Comma-separated explicit n values in cm^-2.")
    ap.add_argument("--n-min-cm2", type=float, default=-4.0e12)
    ap.add_argument("--n-max-cm2", type=float, default=4.0e12)
    ap.add_argument("--n-count", type=int, default=21)
    ap.add_argument("--D-values-Vnm", type=str, default=None, help="Comma-separated explicit D values in V/nm.")
    ap.add_argument("--D-min-Vnm", type=float, default=-0.8)
    ap.add_argument("--D-max-Vnm", type=float, default=0.8)
    ap.add_argument("--D-count", type=int, default=17)
    ap.add_argument("--detail-n-count", type=int, default=10)
    ap.add_argument("--detail-D-count", type=int, default=10)

    ap.add_argument("--grid-n1", type=int, default=24)
    ap.add_argument("--grid-n2", type=int, default=24)
    ap.add_argument("--n-active", type=int, default=8)
    ap.add_argument("--selection", choices=["closest", "contiguous", "overlap"], default="overlap")
    ap.add_argument("--projector-refreshes", type=int, default=0)
    ap.add_argument("--degeneracy", type=int, default=4)
    ap.add_argument("--kBT-meV", type=float, default=0.2)
    ap.add_argument("--eps-perp", type=float, default=4.0)
    ap.add_argument("--d-layer-nm", type=float, default=0.335)
    ap.add_argument("--D-sign", type=float, default=-1.0)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol-meV", type=float, default=1e-4)
    ap.add_argument("--full-max-iter", type=int, default=60)
    ap.add_argument("--full-tol-meV", type=float, default=1e-4)
    ap.add_argument("--mixer", choices=["linear", "anderson"], default="anderson")
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--anderson-beta", type=float, default=0.5)
    ap.add_argument("--anderson-memory", type=int, default=6)
    ap.add_argument("--reference-mode", choices=["full_scf", "supplied_U"], default="full_scf")
    ap.add_argument("--U-ref", type=str, default=None, help="Comma-separated four layer potentials in meV if reference-mode=supplied_U.")

    ap.add_argument("--dos-sigma-meV", type=float, default=1.0)
    ap.add_argument("--dos-emin-meV", type=float, default=-80.0)
    ap.add_argument("--dos-emax-meV", type=float, default=80.0)
    ap.add_argument("--dos-bins", type=int, default=240)
    ap.add_argument("--points-per-segment", type=int, default=60)
    ap.add_argument("--n-show-full", type=int, default=12)
    ap.add_argument("--contour-n1", type=int, default=0, help="0 uses the SCF grid for projected band contours.")
    ap.add_argument("--contour-n2", type=int, default=0, help="0 uses the SCF grid for projected band contours.")
    ap.add_argument("--contour-band-count", type=int, default=6)

    ap.add_argument("--max-workers", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Write run_config.json and task manifest without running SCF.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    n_values = make_axis_values(args.n_values_cm2, args.n_min_cm2, args.n_max_cm2, args.n_count)
    d_values = make_axis_values(args.D_values_Vnm, args.D_min_Vnm, args.D_max_Vnm, args.D_count)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else Path("outputs") / f"nd_mapping_projected_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    config = build_config(args, out)
    config["n_values_cm2"] = n_values
    config["D_values_Vnm"] = d_values
    workers = resolve_workers(args.max_workers)
    config["parallel"]["resolved_max_workers"] = workers
    tasks = build_tasks(n_values, d_values, args.detail_n_count, args.detail_D_count)
    write_json(out / "run_config.json", config)
    write_json(out / "tasks.json", tasks)

    print(f"Output: {out}")
    print(f"n-D points: {len(tasks)} ({len(n_values)} x {len(d_values)}), detail points: {sum(t['detail'] for t in tasks)}")
    print(f"Workers: {workers}")
    if args.dry_run:
        print("Dry run complete.")
        return

    rows: list[dict[str, Any]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_label = {pool.submit(run_one_point, task, config): task["label"] for task in tasks}
        completed = 0
        for future in as_completed(future_to_label):
            row = future.result()
            rows.append(row)
            completed += 1
            status = row.get("status", "failed")
            label = row.get("label", future_to_label[future])
            print(f"[{completed:5d}/{len(tasks):5d}] {status:6s} {label} runtime={row.get('runtime_s', float('nan')):.1f}s", flush=True)
            if completed % 10 == 0 or completed == len(tasks):
                save_map_outputs(out, rows, n_values, d_values)

    save_map_outputs(out, rows, n_values, d_values)
    failed = [r for r in rows if r.get("status") != "ok"]
    if failed:
        with (out / "logs" / "failures.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in failed for k in row.keys()}))
            writer.writeheader()
            writer.writerows(failed)
    print(f"Finished in {(time.time() - started) / 60.0:.2f} min")
    print(f"Failed points: {len(failed)}")


if __name__ == "__main__":
    main()
