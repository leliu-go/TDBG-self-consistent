#!/usr/bin/env python3
"""Parallel n-D map: full self-consistent Hartree, then full-band Stoner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
for _path in (SCRIPT_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_full_band_stoner import (
    FLAVOR_NAMES,
    add_analysis_args,
    add_full_scf_args,
    add_hamiltonian_args,
    add_stoner_args,
    clear_point_outputs,
    hamiltonian_metadata,
    hamiltonian_npz_metadata,
    params_from_args,
    point_label,
    run_point,
    scf_valley,
    TDBGContinuumHamiltonian,
    write_json,
)
from tdbg_scf.output_metadata import displacement_metadata
from tdbg_scf.output_metadata import axis_edges as _axis_edges
from tdbg_scf.scf_outputs import scf_point_record, scf_scalar_grids


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


def build_tasks(n_values: list[float], d_values: list[float], detail_n_count: int, detail_d_count: int) -> list[dict[str, Any]]:
    detail_i = evenly_spaced_indices(len(n_values), detail_n_count)
    detail_j = evenly_spaced_indices(len(d_values), detail_d_count)
    tasks = []
    for i, n_cm2 in enumerate(n_values):
        for j, d_vnm in enumerate(d_values):
            tasks.append(
                {
                    "n_index": i,
                    "D_index": j,
                    "n_cm2": float(n_cm2),
                    "D_Vnm": float(d_vnm),
                    "detail": bool(i in detail_i and j in detail_j),
                    "label": point_label(i, j, float(n_cm2), float(d_vnm)),
                }
            )
    return tasks


def resolve_workers(value: str) -> int:
    if value == "auto":
        workers = max(1, (os.cpu_count() or 2) - 2)
        return min(workers, 61) if sys.platform == "win32" else workers
    workers = int(value)
    if workers < 1:
        raise ValueError("--max-workers must be >= 1 or auto")
    if sys.platform == "win32" and workers > 61:
        raise ValueError("Windows ProcessPoolExecutor supports at most 61 workers")
    return workers


def run_task(task: dict[str, Any], args: argparse.Namespace, details_dir: str) -> dict[str, Any]:
    detail_dir = Path(details_dir) / task["label"] if task.get("detail") else None
    row = run_point(task, args, detail_dir=detail_dir)
    row["detail_saved"] = bool(task.get("detail"))
    if task.get("detail") and row.get("status") == "ok":
        row["detail_dir"] = str(detail_dir)
    return row


def plot_nd_map(df: pd.DataFrame, n_values: list[float], d_values: list[float], key: str, out: Path, title: str) -> None:
    if key not in df.columns:
        out.unlink(missing_ok=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((len(n_values), len(d_values)), np.nan, dtype=float)
    for _, row in df[df["status"] == "ok"].iterrows():
        arr[int(row["n_index"]), int(row["D_index"])] = float(row[key])

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    n_axis, d_axis = np.asarray(n_values) / 1e12, np.asarray(d_values)
    i_order, j_order = np.argsort(n_axis), np.argsort(d_axis)
    im = ax.pcolormesh(_axis_edges(d_axis[j_order], .05), _axis_edges(n_axis[i_order], .1),
                       arr[np.ix_(i_order, j_order)], shading="flat")
    ax.set_xlabel("D (V/nm)")
    ax.set_ylabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_scf_map_outputs(out: Path, rows: list[dict[str, Any]], n_values: list[float], d_values: list[float],
                         ham: TDBGContinuumHamiltonian, D_sign: float) -> None:
    scf_dir = out / "scf"
    scf_dir.mkdir(parents=True, exist_ok=True)
    records = [scf_point_record({**displacement_metadata(D_sign), **row}, ham) for row in rows]
    frame = pd.DataFrame(records).sort_values(["n_index", "D_index"], kind="stable")
    frame.to_csv(scf_dir / "nd_map.csv", index=False)
    write_json(scf_dir / "nd_map.json", frame.to_dict(orient="records"))
    np.savez(scf_dir / "nd_map_scalars.npz", **hamiltonian_npz_metadata(ham),
             **displacement_metadata(D_sign), stage="scf_before_stoner",
             n_values_cm2=np.asarray(n_values, dtype=float), D_values_Vnm=np.asarray(d_values),
             **scf_scalar_grids(records, len(n_values), len(d_values)))
    specs = [
        ("dos_mu_scf", "SCF full-spectrum DOS at density-matched mu"),
        ("scf_mu_meV", "SCF chemical potential (meV)"),
        ("full_mu_meV", "SCF full-spectrum chemical potential (meV)"),
        ("P_top_bottom_cm2", "SCF top-bottom layer polarization (cm^-2)"),
        ("P_outer_inner_cm2", "SCF outer-inner layer polarization (cm^-2)"),
        ("P_dipole_cm2", "SCF layer dipole scalar (cm^-2)"),
        ("scf_layer_dos_top_bottom", "SCF top-bottom layer DOS"),
        ("scf_layer_dos_outer_inner", "SCF outer-inner layer DOS"),
        ("scf_layer_dos_dipole", "SCF layer DOS dipole scalar"),
        ("scf_residual_meV", "SCF residual (meV)"),
        ("scf_iterations", "SCF iterations"),
    ]
    for layer in range(1, 4 + 1):
        specs += [(f"U{layer}_meV", f"SCF layer {layer} electron potential (meV)"),
                  (f"n{layer}_cm2", f"SCF layer {layer} density (cm^-2)"),
                  (f"scf_layer_dos_L{layer}", f"SCF layer {layer} DOS")]
    for key, title in specs:
        plot_nd_map(frame, n_values, d_values, key, scf_dir / "figures" / f"nd_map_{key}.png", title)


def save_map_outputs(out: Path, rows: list[dict[str, Any]], n_values: list[float], d_values: list[float],
                     ham: TDBGContinuumHamiltonian, D_sign: float = -1.0) -> None:
    n_layers = max((int(row.get("n_layers", 4)) for row in rows), default=4)
    df = pd.DataFrame(rows).sort_values(["n_index", "D_index"], kind="stable")
    df.to_csv(out / "nd_map.csv", index=False)
    write_json(out / "nd_map.json", df.to_dict(orient="records"))
    save_scf_map_outputs(out, rows, n_values, d_values, ham, D_sign)
    ok = df[df["status"] == "ok"]
    # Replace the previous cache even when every point failed: empty rows then
    # explicitly describe this run instead of retaining an earlier valid map.
    np.savez(
        out / "nd_map_scalars.npz",
        **hamiltonian_npz_metadata(ham),
        **displacement_metadata(D_sign),
        n_values_cm2=np.asarray(n_values, dtype=float),
        D_values_Vnm=np.asarray(d_values, dtype=float),
        rows=ok.to_records(index=False),
    )

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("stoner_energy_meV_per_cell", "Full-band Stoner energy (meV/cell)"),
        ("flavor_polarization", "Flavor polarization"),
        ("spin_polarization", "Absolute spin polarization"),
        ("spin_polarization_norm", "Absolute normalized spin polarization"),
        ("valley_polarization", "Absolute valley polarization"),
        ("valley_polarization_norm", "Absolute normalized valley polarization"),
        ("spin_valley_polarization", "Absolute spin-valley polarization"),
        ("spin_valley_polarization_norm", "Absolute normalized spin-valley polarization"),
        ("dos_mu_scf", "Full-band DOS at full-band mu"),
        ("dos_mu_full_at_full_mu", "Full-band DOS at full-band mu"),
        ("dos_mu_stoner", "Explicit-flavor DOS at Stoner mu_f"),
        ("P_outer_inner_cm2", "Outer-inner layer polarization (cm^-2)"),
        ("P_top_bottom_cm2", "Top-bottom layer polarization (cm^-2)"),
        ("P_dipole_cm2", "Layer dipole scalar (cm^-2)"),
        ("stoner_P_outer_inner_cm2", "Stoner outer-inner layer polarization (cm^-2)"),
        ("stoner_P_top_bottom_cm2", "Stoner top-bottom layer polarization (cm^-2)"),
        ("stoner_P_dipole_cm2", "Stoner layer dipole scalar (cm^-2)"),
        *[(f"stoner_layer_dos_L{layer}", f"Stoner layer DOS L{layer}") for layer in range(1, n_layers + 1)],
        ("stoner_layer_dos_total", "Stoner total layer DOS"),
        ("stoner_layer_dos_outer_inner", "Stoner outer-inner layer DOS"),
        ("stoner_layer_dos_top_bottom", "Stoner top-bottom layer DOS"),
        ("stoner_layer_dos_dipole", "Stoner layer DOS dipole scalar"),
        ("scf_residual_meV", "SCF residual (meV)"),
        ("scf_iterations", "SCF iterations"),
    ]
    for flavor in FLAVOR_NAMES:
        prefix = f"stoner_{flavor}"
        plot_specs.extend(
            [
                (f"{prefix}_P_top_bottom_cm2", f"{flavor} Stoner top-bottom density polarization (cm^-2)"),
                (f"{prefix}_P_outer_inner_cm2", f"{flavor} Stoner outer-inner density polarization (cm^-2)"),
                (f"{prefix}_P_dipole_cm2", f"{flavor} Stoner density dipole scalar (cm^-2)"),
                *[(f"{prefix}_layer_dos_L{layer}", f"{flavor} Stoner layer DOS L{layer}") for layer in range(1, n_layers + 1)],
                (f"{prefix}_layer_dos_total", f"{flavor} Stoner total layer DOS"),
                (f"{prefix}_layer_dos_top_bottom", f"{flavor} Stoner top-bottom layer DOS"),
                (f"{prefix}_layer_dos_outer_inner", f"{flavor} Stoner outer-inner layer DOS"),
                (f"{prefix}_layer_dos_dipole", f"{flavor} Stoner layer DOS dipole scalar"),
            ]
        )
    for key, title in plot_specs:
        plot_nd_map(df, n_values, d_values, key, fig_dir / f"nd_map_{key}.png", title)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Server n-D mapping: full self-consistent Hartree per point, then full-band Stoner.")
    add_hamiltonian_args(ap)
    ap.add_argument("--n-values-cm2", type=str, default=None, help="Comma-separated explicit density values.")
    ap.add_argument("--n-min-cm2", type=float, default=-6e12)
    ap.add_argument("--n-max-cm2", type=float, default=6e12)
    ap.add_argument("--n-count", type=int, default=61)
    ap.add_argument("--D-values-Vnm", type=str, default=None, help="Comma-separated explicit D values.")
    ap.add_argument("--D-min-Vnm", type=float, default=-1.2)
    ap.add_argument("--D-max-Vnm", type=float, default=1.2)
    ap.add_argument("--D-count", type=int, default=121)
    ap.add_argument("--detail-n-count", type=int, default=11)
    ap.add_argument("--detail-D-count", type=int, default=21)
    add_full_scf_args(ap)
    add_stoner_args(ap)
    add_analysis_args(ap)
    ap.add_argument("--max-workers", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else Path("outputs") / f"nd_mapping_full_band_stoner_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    details_dir = out / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    ham = TDBGContinuumHamiltonian(params_from_args(args, scf_valley(args)))
    n_values = make_axis_values(args.n_values_cm2, args.n_min_cm2, args.n_max_cm2, args.n_count)
    d_values = make_axis_values(args.D_values_Vnm, args.D_min_Vnm, args.D_max_Vnm, args.D_count)
    tasks = build_tasks(n_values, d_values, args.detail_n_count, args.detail_D_count)
    workers = resolve_workers(args.max_workers)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "platform": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count()},
        "args": vars(args),
        "hamiltonian": hamiltonian_metadata(ham),
        "displacement": displacement_metadata(args.D_sign),
        "n_values_cm2": n_values,
        "D_values_Vnm": d_values,
        "max_workers_resolved": workers,
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    write_json(out / "config_used.json", config)

    rows: list[dict[str, Any]] = []
    started = time.time()
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            row = run_task(task, args, str(details_dir))
            rows.append(row)
            print(f"[{index}/{len(tasks)}] {task['label']} {row['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            def failed_worker(task, exc):
                detail_dir = details_dir / task["label"]
                clear_point_outputs(detail_dir)
                row = {
                    **hamiltonian_metadata(ham), **displacement_metadata(args.D_sign),
                    **task, "status": "failed", "scf_status": "failed", "scf_converged": False,
                    "stoner_success": False, "error_stage": "worker_transport",
                    "error": repr(exc), "detail_saved": False,
                }
                write_json(detail_dir / "error.json", row)
                return row

            futures = {}
            for task in tasks:
                try:
                    futures[pool.submit(run_task, task, args, str(details_dir))] = task
                except Exception as exc:
                    row = failed_worker(task, exc)
                    rows.append(row)
                    print(f"[{len(rows)}/{len(tasks)}] {task['label']} failed", flush=True)
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    row = failed_worker(futures[future], exc)
                rows.append(row)
                print(f"[{len(rows)}/{len(tasks)}] {row.get('label', '?')} {row['status']}", flush=True)

    save_map_outputs(out, rows, n_values, d_values, ham, D_sign=args.D_sign)
    summary = {
        "total_points": len(tasks),
        "ok_points": sum(1 for row in rows if row.get("status") == "ok"),
        "failed_points": sum(1 for row in rows if row.get("status") != "ok"),
        "invalid_stoner_dos_points": sum(1 for row in rows if row.get("status") == "ok"
                                          and not row.get("stoner_dos_valid", True)),
        "runtime_s": float(time.time() - started),
        "out": str(out),
    }
    write_json(out / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed_points"] or summary["invalid_stoner_dos_points"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
