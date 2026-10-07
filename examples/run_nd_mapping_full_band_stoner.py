#!/usr/bin/env python3
"""Parallel n-D map: full self-consistent Hartree, then full-band Stoner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
import traceback
import json
import os
from pathlib import Path
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
    hamiltonian_metadata,
    hamiltonian_npz_metadata,
    params_from_args,
    point_label,
    run_point,
    scf_valley,
    stoner_params_from_args,
    TDBGContinuumHamiltonian,
    write_json,
)
from tdbg_scf.output_metadata import displacement_metadata
from tdbg_scf.output_metadata import axis_edges as _axis_edges
from tdbg_scf.scf_outputs import scf_point_record, scf_scalar_grids


from tdbg_scf.scan_axes import add_density_axis_args, density_axis_metadata, resolve_density_axis
from tdbg_scf.run_storage import add_output_args, create_output_dir, detail_indices, estimate_storage, explicit_contour_kpoint_count, write_run_config
from tdbg_scf.run_progress import RunProgress, initial_runtime_estimate
from tdbg_scf.atomic_io import atomic_csv, atomic_json, atomic_npz, atomic_output
from tdbg_scf.live_map import CORE_FIELDS, SCF_CORE_FIELDS, LivePreview, complete_rows, scalar_grids
from tdbg_scf.scan_checkpoint import prepare_manifest, begin_attempt, load_points, save_point, restore_resume_args, scan_directory_lock
from tdbg_scf.worker_diagnostics import initialize_worker, close_worker_diagnostics, log_worker_event, memory_snapshot, runtime_diagnostics


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


def build_tasks(n_values, d_values, detail_count=10, detail_d_count=None):
    # Explicit legacy per-axis detail counts remain accepted; ordinary runs use
    # a bounded total count rather than their large Cartesian product.
    if detail_d_count is None:
        chosen = detail_indices(n_values, d_values, detail_count)
    else:
        chosen = {(i, j) for i in evenly_spaced_indices(len(n_values), detail_count)
                  for j in evenly_spaced_indices(len(d_values), detail_d_count)}
    return [{"n_index": i, "D_index": j, "n_cm2": float(n), "D_Vnm": float(d),
             "detail": (i, j) in chosen, "label": point_label(i, j, float(n), float(d))}
            for i, n in enumerate(n_values) for j, d in enumerate(d_values)]


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
    started = time.monotonic()
    out = Path(details_dir).parent
    attempt = getattr(args, "_checkpoint_attempt_id", None)
    detail_dir = (out / "checkpoints/work" / attempt / task["label"] if attempt else
                  Path(details_dir) / task["label"]) if task.get("detail") else None
    log_worker_event("task_start", **{key: value for key, value in task.items() if not key.startswith("_")},
                     memory=memory_snapshot())
    row = run_point(task, args, detail_dir=detail_dir)
    row["detail_saved"] = bool(task.get("detail")) and row.get("status") == "ok"
    row["runtime_s"] = time.monotonic() - started
    if task.get("detail") and row.get("status") == "ok":
        row["detail_dir"] = str(Path(details_dir) / task["label"])
    fingerprint = getattr(args, "_checkpoint_fingerprint", None)
    if fingerprint is not None:
        # Commit in the worker, even if the parent dies before receiving the future.
        # Stage details privately; only this attempt may publish them under the commit lock.
        save_point(out, task, row, fingerprint, attempt_id=attempt,
                   detail_source=detail_dir if attempt else None)
    elif row.get("status") != "ok":
        write_json(out / "errors" / f"{task['label']}.json", row)
    log_worker_event("task_returned", label=task["label"], status=row.get("status"),
                     runtime_s=row["runtime_s"], error=row.get("error"), memory=memory_snapshot())
    return row


def plot_nd_map(df: pd.DataFrame, n_values: list[float], d_values: list[float], key: str, out: Path, title: str) -> None:
    if key not in df.columns:
        return
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
    ax.set_facecolor("#eeeeee")
    with atomic_output(out) as temporary:
        fig.savefig(temporary, format="png", dpi=220)
    plt.close(fig)


def save_scf_map_outputs(out: Path, rows: list[dict[str, Any]], n_values: list[float], d_values: list[float],
                         ham: TDBGContinuumHamiltonian, D_sign: float, *, plot_figures=True) -> None:
    scf_dir = out / "scf"
    scf_dir.mkdir(parents=True, exist_ok=True)
    records = [scf_point_record({**displacement_metadata(D_sign), **row}, ham) for row in rows]
    frame = pd.DataFrame(records).sort_values(["n_index", "D_index"], kind="stable")
    expected_scf_fields = [*SCF_CORE_FIELDS, *[field for layer in range(1, 4 + 1)
                           for field in (f"U{layer}_meV", f"n{layer}_cm2", f"scf_layer_dos_L{layer}")]]
    for key in expected_scf_fields:
        if key not in frame:
            frame[key] = np.nan
    atomic_csv(scf_dir / "nd_map.csv", frame)
    atomic_json(scf_dir / "nd_map.json", frame.to_dict(orient="records"))
    grids = scf_scalar_grids(records, len(n_values), len(d_values))
    for key in frame:
        if key in expected_scf_fields and key not in grids:
            grids[key] = np.full((len(n_values), len(d_values)), np.nan)
    atomic_npz(scf_dir / "nd_map_scalars.npz", **hamiltonian_npz_metadata(ham),
             **displacement_metadata(D_sign), stage="scf_before_stoner",
             **density_axis_metadata(n_values, ham.geom), D_values_Vnm=np.asarray(d_values),
             **grids)
    if not plot_figures:
        return
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
                     ham: TDBGContinuumHamiltonian, D_sign: float = -1.0, *, plot_figures=True) -> None:
    rows = complete_rows(rows, n_values, d_values)
    n_layers = 4
    df = pd.DataFrame(rows).sort_values(["n_index", "D_index"], kind="stable")
    for key in CORE_FIELDS:
        if key not in df:
            df[key] = np.nan
    atomic_csv(out / "nd_map.csv", df)
    atomic_json(out / "nd_map.json", df.to_dict(orient="records"))
    save_scf_map_outputs(out, rows, n_values, d_values,
                         ham, D_sign, plot_figures=plot_figures)
    ok = df[df["status"] == "ok"]
    metadata = {**hamiltonian_npz_metadata(ham), **displacement_metadata(D_sign)}
    axis_metadata = {**density_axis_metadata(n_values, ham.geom), "D_values_Vnm": np.asarray(d_values, dtype=float)}
    atomic_npz(
            out / "nd_map_scalars.npz",
            **metadata,
            **axis_metadata,
            rows=ok.to_records(index=False),
            **scalar_grids(rows, (len(n_values), len(d_values)), 4, [*metadata, *axis_metadata]),
        )
    if not plot_figures:
        return

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


def parse_args():
    ap = argparse.ArgumentParser(description="Resumable n-D scan: full Hartree followed by bounded signed-flavor Stoner.")
    add_hamiltonian_args(ap)
    add_density_axis_args(ap)
    ap.add_argument("--D-values-Vnm", default=None)
    ap.add_argument("--D-min-Vnm", type=float, default=-1.2)
    ap.add_argument("--D-max-Vnm", type=float, default=1.2)
    ap.add_argument("--D-count", type=int, default=121)
    ap.add_argument("--detail-count", type=int, default=10)
    ap.add_argument("--detail-n-count", type=int, default=None)
    ap.add_argument("--detail-D-count", type=int, default=None)
    add_full_scf_args(ap)
    add_stoner_args(ap)
    add_analysis_args(ap)
    ap.add_argument("--max-workers", default="auto")
    ap.add_argument("--resume", action="store_true", help="Restore and continue the original checkpointed --out directory.")
    ap.add_argument("--live-plot-interval-s", type=float, default=30., help="Preview interval; 0 disables image refresh only.")
    add_output_args(ap)
    args = restore_resume_args(ap, ap.parse_args(), sys.argv[1:])
    stoner_params_from_args(args)
    if not np.isfinite(args.live_plot_interval_s) or args.live_plot_interval_s < 0:
        raise ValueError("live plot interval must be finite and nonnegative")
    if args.detail_count < 0 or any(value is not None and value < 0 for value in (args.detail_n_count, args.detail_D_count)):
        raise ValueError("detail counts must be nonnegative")
    return args


def main():
    args = parse_args()
    ham = TDBGContinuumHamiltonian(params_from_args(args, scf_valley(args)))
    n_values, scan_metadata = resolve_density_axis(args, ham.geom)
    if not np.isfinite([args.D_min_Vnm, args.D_max_Vnm]).all() or args.D_min_Vnm > args.D_max_Vnm:
        raise ValueError("D limits must be finite and ordered")
    d_values = make_axis_values(args.D_values_Vnm, args.D_min_Vnm, args.D_max_Vnm, args.D_count)
    if not d_values or not np.isfinite(d_values).all() or len(set(d_values)) != len(d_values):
        raise ValueError("D scan values must be nonempty, finite and unique")
    if args.detail_n_count is not None or args.detail_D_count is not None:
        tasks = build_tasks(n_values, d_values, args.detail_n_count or 0, args.detail_D_count or 0)
    else:
        tasks = build_tasks(n_values, d_values, args.detail_count)
    out = create_output_dir(args, "ndmap")
    with scan_directory_lock(out):
        return run_scan(args, ham, n_values, d_values, scan_metadata, tasks, out)


def run_scan(args, ham, n_values, d_values, scan_metadata, tasks, out):
    fingerprint = prepare_manifest(out, args, hamiltonian_metadata(ham), tasks)
    attempt = begin_attempt(out, args)
    rows_by_label, reused, rejected = load_points(out, tasks, fingerprint)
    pending = [task for task in tasks if task["label"] not in reused]
    workers = min(resolve_workers(args.max_workers), max(1, len(pending)))
    details_dir = out / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    storage = estimate_storage(ham.dim, 4, args.grid_n1 * args.grid_n2, len(tasks),
                               sum(bool(task["detail"]) for task in tasks), args.points_per_segment, args.dos_bins,
                               contour_nk=explicit_contour_kpoint_count(args))
    estimate = (initial_runtime_estimate(ham, args, len(pending), workers) if pending else
                {"low_s": 0., "high_s": 0., "active_workers": 0, "note": "All points restored; no SCF work remains."})
    progress = RunProgress(len(tasks), workers, estimate, resumed_completed=len(reused))
    print(f"Results: {out}", flush=True)
    print(f"Resume: {len(reused)} verified points reused; {len(pending)} points to compute; {len(rejected)} damaged/incompatible point records ignored.", flush=True)
    print(f"Detailed points: {storage['detail_points']}; saved results budget ~{storage['saved_results_estimate_GB']:.3f} GB; full eigenvectors are computed in RAM.", flush=True)
    print(progress.start_message(), flush=True)

    config = {
        **scan_metadata,
        "D_values_Vnm": d_values,
        "max_workers_resolved": workers,
        "detail_point_labels": [task["label"] for task in tasks if task["detail"]],
        "storage_estimate": storage,
        "initial_runtime_estimate": estimate,
        "scf_config_template": {"degeneracy": 4, "kBT_meV": args.kBT_meV,
            "max_iter": args.full_max_iter, "tol_meV": args.full_tol_meV,
            "eps_perp": args.eps_perp, "d_layer_nm": args.d_layer_nm, "D_sign": args.D_sign,
            "mixer": args.mixer, "anderson_beta": args.anderson_beta,
            "anderson_memory": args.anderson_memory, "linear_fallback_alpha": args.alpha},
        "scf_point_parameters": {"target_density_cm2": "n_values_cm2", "D_Vnm": "D_values_Vnm"},
        "stoner_config": asdict(stoner_params_from_args(args)),
    }
    if not args.resume or not (out / "config_used.json").exists():
        write_run_config(out, args, ham, "ndmap", {**config, "checkpoint_schema_version": 1,
                         "checkpoint_fingerprint": fingerprint, "live_preview": "nd_map_live.png"})
    diagnostics_dir = out / "diagnostics"
    atomic_json(diagnostics_dir / "runtime.json", runtime_diagnostics())
    atomic_json(diagnostics_dir / ("attempt_" + str(time.time_ns()) + ".json"), {
        "args": vars(args), "resumed_points": len(reused), "pending_points": len(pending),
        "rejected_checkpoints": rejected, "checkpoint_fingerprint": fingerprint,
    })
    print(f"Worker diagnostics: {diagnostics_dir}", flush=True)

    started = time.time()
    abort_reason = None
    returned = 0
    returned_labels = set()
    preview = LivePreview(out, n_values, d_values, args.live_plot_interval_s)

    def rows():
        return [rows_by_label[task["label"]] for task in tasks if task["label"] in rows_by_label]

    def summary(state):
        records = rows()
        value = {
            "scan_status": state, "total_points": len(tasks), "completed_points": len(records),
            "unprocessed_points": len(tasks) - len(records), "resumed_points": len(reused),
            "completed_this_run": returned, "pending_points": len(pending) - returned,
            "scan_aborted": abort_reason is not None, "abort_reason": abort_reason,
            "checkpoint_fingerprint": fingerprint, "diagnostics_directory": str(diagnostics_dir),
            "ok_points": sum(row.get("status") == "ok" for row in records),
            "failed_points": sum(row.get("status") != "ok" for row in records),
            "invalid_stoner_dos_points": sum(row.get("status") == "ok" and not row.get("stoner_dos_valid", True) for row in records),
            "stoner_bound_hit_points": sum(bool(row.get("stoner_filling_bound_hit", False)) for row in records),
            "stoner_bound_limited_points": sum(bool(row.get("stoner_filling_bound_limited", False)) for row in records),
            "hartree_iterations_performed": sum(int(row.get("scf_iterations_performed", row.get("scf_iterations", 0)))
                                                for row in records),
            "runtime_s": time.time() - started, "out": str(out), "max_workers_resolved": workers,
            "detail_points_requested": args.detail_count,
            "detail_points_selected": sum(bool(task["detail"]) for task in tasks),
            "detail_points_saved": sum(bool(row.get("detail_saved")) and row.get("status") == "ok" for row in records),
            "initial_runtime_estimate": estimate, "progress": progress.snapshot(),
        }
        if abort_reason is not None:
            value["progress"].update(eta_s=None, expected_completion_sgt=None, method="aborted")
        return value

    def save_live():
        save_map_outputs(out, rows(), n_values, d_values, ham, D_sign=args.D_sign,
                         plot_figures=False)
        preview.update(rows(), force=returned == 1)
        atomic_json(out / "run_summary.json", summary("running"))

    def append_row(row, task):
        nonlocal returned
        row = {**{key: value for key, value in task.items() if not key.startswith("_")}, **row}
        path = out / "checkpoints/points" / (task["label"] + ".json")
        if row.get("error_stage") != "worker_transport" and not path.exists():
            save_point(out, task, row, fingerprint, attempt_id=attempt)
        previous = rows_by_label.get(task["label"])
        rows_by_label[task["label"]] = row
        if task["label"] not in returned_labels:
            returned_labels.add(task["label"])
            returned += 1
        elif previous.get("status") != row.get("status"):
            if row.get("status") == "ok":
                progress.successful += 1
                progress.failed -= 1
            else:
                progress.successful -= 1
                progress.failed += 1
        print(progress.update(len(reused) + returned, task["label"], row["status"], row.get("runtime_s")), flush=True)
        save_live()

    def failed_worker(task, exc):
        committed, _, _ = load_points(out, [task], fingerprint, attempt_id=attempt)
        if task["label"] in committed:
            # A worker may die after committing, but before its future is delivered.
            # Keep valid Hartree results even when its later Stoner stage failed.
            return committed[task["label"]]
        # An earlier failed Stoner attempt may already contain a verified
        # Hartree stage. A process transport failure cannot invalidate it.
        previous, _, _ = load_points(out, [task], fingerprint)
        previous = previous.get(task["label"], {})
        preserve_scf = previous.get("scf_status") == "ok" and previous.get("scf_converged") is True
        scf_fields = scf_point_record(previous, ham) if preserve_scf else {}
        # No result was returned: physical convergence is unknown.
        row = {
            **scf_fields,
            **hamiltonian_metadata(ham), **displacement_metadata(args.D_sign),
            **task, "status": "failed", "scf_status": "ok" if preserve_scf else "unknown",
            "scf_converged": True if preserve_scf else None,
            "scf_preserved_from_checkpoint": preserve_scf,
            "stoner_success": None, "error_stage": "worker_transport",
            "error": repr(exc), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "detail_saved": False,
        }
        atomic_json(diagnostics_dir / ("transport_" + task["label"] + ".json"), row)
        # A transport error is not a completed calculation. Never overwrite a
        # checkpoint while a worker may concurrently commit its actual result.
        return row

    def abort_scan(task, exc):
        nonlocal abort_reason
        abort_reason = repr(exc)
        write_json(diagnostics_dir / "scan_abort.json", {
            "reason": abort_reason, "task": task, "memory": memory_snapshot(),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "note": "Scan interrupted. Unreturned points have no final result; this does not establish SCF nonconvergence.",
        })
        print(f"Scan interrupted. See {diagnostics_dir}. Unprocessed points will not be reported as physical failures.", flush=True)

    save_live()
    try:
        if pending and workers == 1:
            initialize_worker(str(diagnostics_dir))
            try:
                for task in pending:
                    append_row(run_task(task, args, str(details_dir)), task)
            finally:
                close_worker_diagnostics()
        elif pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                     initargs=(str(diagnostics_dir),)) as pool:
                futures = {}
                handled = set()
                try:
                    for task in pending:
                        try:
                            futures[pool.submit(run_task, task, args, str(details_dir))] = task
                        except BrokenProcessPool as exc:
                            append_row(failed_worker(task, exc), task)
                            abort_scan(task, exc)
                            break
                        except Exception as exc:
                            append_row(failed_worker(task, exc), task)
                    if abort_reason is None:
                        for future in as_completed(futures):
                            task = futures[future]
                            handled.add(future)
                            try:
                                row = future.result()
                            except BrokenProcessPool as exc:
                                append_row(failed_worker(task, exc), task)
                                abort_scan(task, exc)
                                break
                            except Exception as exc:
                                row = failed_worker(task, exc)
                            append_row(row, task)
                    if abort_reason is not None:
                        for future, task in futures.items():
                            future.cancel()
                            if future in handled or not future.done() or future.cancelled():
                                continue
                            try:
                                row = future.result()
                            except BrokenProcessPool:
                                continue
                            except Exception as exc:
                                row = failed_worker(task, exc)
                            append_row(row, task)
                except KeyboardInterrupt:
                    # Cancel queued tasks before the executor waits for active workers.
                    for future in futures:
                        future.cancel()
                    raise
    except (KeyboardInterrupt, BrokenProcessPool) as exc:
        abort_scan(None, exc)

    # Recover all worker-committed rows after both normal completion and interruption.
    # Scheduling eligibility is separate from whether a failed row contains valid SCF data.
    committed, _, _ = load_points(out, pending, fingerprint, attempt_id=attempt)
    for task in pending:
        if task["label"] in committed and (task["label"] not in returned_labels
                or rows_by_label[task["label"]].get("error_stage") == "worker_transport"):
            append_row(committed[task["label"]], task)

    print("Saving partial maps after process-pool failure..." if abort_reason else
          "All scan points finished; saving maps and figures...", flush=True)
    save_map_outputs(out, rows(), n_values, d_values, ham, D_sign=args.D_sign)
    preview.update(rows(), force=True)
    final = summary("aborted" if abort_reason else "completed")
    final["result_directory_size_bytes"] = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    atomic_json(out / "run_summary.json", final)
    print(json.dumps(final, indent=2))
    return 1 if abort_reason or final["failed_points"] or final["invalid_stoner_dos_points"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
