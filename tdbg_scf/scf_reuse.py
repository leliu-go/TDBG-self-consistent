"""Validate saved full-Hartree maps for Stoner replay without SCF iterations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .atomic_io import canonical_json
from .output_metadata import hamiltonian_metadata, hamiltonian_metadata_matches
from .scf_checks import require_converged_scf
from .solver_full import FullSCFResult
from .stoner_update import resolve_inplace_source


REPLAY_MUTABLE_ARGS = {
    "out", "data_root", "max_workers", "resume", "live_plot_interval_s", "detail_count",
    "u0_meV_A2", "JH_meV_A2", "dos_sigma_meV", "dos_emin_meV", "dos_emax_meV",
    "dos_bins", "points_per_segment", "n_show_full", "contour_n1", "contour_n2",
    "contour_band_count", "reuse_scf_run", "update_stoner_in_place", "restart_stoner_in_place",
}
PHYSICS_SOURCES = ("continuum.py", "density.py", "electrostatics.py")


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_config(source):
    path = Path(source) / "config_used.json"
    if not path.is_file():
        raise ValueError("Saved Hartree source lacks config_used.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("args"), dict):
        raise ValueError("Saved Hartree source has no valid argument record")
    if not (Path(source) / "nd_map.json").is_file():
        raise ValueError("Saved Hartree source lacks nd_map.json")
    recorded = config.get("source_sha256", {})
    for name in PHYSICS_SOURCES:
        data = (Path(__file__).parent / name).read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        hashes = {hashlib.sha256(payload).hexdigest() for payload in (data, lf, lf.replace(b"\n", b"\r\n"))}
        if recorded.get(name) not in hashes:
            raise ValueError("Saved Hartree source code mismatch: " + name)
    return config


def check_replay_output(source, out):
    """Reject writes into the old run before creating any output directories."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or source in out.parents:
        raise ValueError("Stoner replay output must be outside the original source run")


def restore_scf_reuse_args(parser, args, argv):
    source = resolve_inplace_source(args)
    if not source:
        return args
    source = Path(source).expanduser().resolve()
    config = _source_config(source)
    supplied = set()
    for token in argv:
        parsed = parser._parse_optional(token)
        if parsed is not None and parsed[0] is not None:
            supplied.add(parsed[0].dest)
    for name, value in config["args"].items():
        if name.startswith("_") or not hasattr(args, name):
            continue
        # Resume has already restored and checked every scientific parameter.
        mutable = name in REPLAY_MUTABLE_ARGS or name.startswith("stoner_")
        if getattr(args, "update_stoner_in_place", False) and name == "dos_sigma_meV":
            mutable = False
        if args.resume:
            if not mutable and canonical_json(getattr(args, name)) != canonical_json(value):
                raise ValueError("Saved Hartree source/resume inputs differ: " + name)
            continue
        if not mutable and name in supplied and canonical_json(getattr(args, name)) != canonical_json(value):
            raise ValueError("Incompatible saved Hartree input: " + name + " differs from the source scan")
        # Bounds describe the new Stoner calculation; all other omitted
        # analysis/Stoner options inherit the user's original values.
        if name not in supplied and name not in {
                "out", "data_root", "resume", "reuse_scf_run", "max_workers", "live_plot_interval_s",
                "update_stoner_in_place", "restart_stoner_in_place",
                "stoner_flavor_nu_min", "stoner_flavor_nu_max"}:
            setattr(args, name, value)
    args.reuse_scf_run = str(source)
    if args.out and not (getattr(args, "update_stoner_in_place", False)
                         and Path(args.out).resolve() == source):
        check_replay_output(source, args.out)
    elif not args.out:
        check_replay_output(source, args.data_root)
    record_reuse_source(args, source)
    return args


def record_reuse_source(args, source):
    source = Path(source).resolve()
    args._scf_reuse_metadata = {
        "source_directory": str(source),
        "source_config_sha256": _file_sha256(source / "config_used.json"),
        "source_nd_map_sha256": _file_sha256(source / "nd_map.json"),
        "physics_sources_verified": list(PHYSICS_SOURCES),
        "method": "two_full_valley_diagonalizations_at_saved_U_no_Hartree_iterations",
    }


def saved_scf_result(reference, n_cm2, D_Vnm, n_layers, tol_meV):
    """Reconstruct only a checked saved result, never a warm-start solver."""
    if reference["n_cm2"] != n_cm2 or reference["D_Vnm"] != D_Vnm:
        raise ValueError("Saved Hartree reference point identity mismatch")
    U = np.asarray(reference["U_meV"], dtype=float)
    density = np.asarray(reference["n_layer_cm2"], dtype=float)
    if U.shape != (n_layers,) or density.shape != (n_layers,):
        raise ValueError("Saved Hartree potential/density layer count mismatch")
    result = FullSCFResult(
        U_meV=U.copy(), mu_meV=float(reference["mu_meV"]), n_layer_a2=density * 1e-16,
        converged=reference["converged"] is True, residual_meV=float(reference["residual_meV"]),
        iterations=int(reference["iterations"]), history=[],
    )
    try:
        require_converged_scf(result, n_cm2, tol_meV)
    except RuntimeError as exc:
        raise ValueError("Saved Hartree point is not reusable: " + str(exc)) from exc
    return result


def attach_saved_scf_references(args, ham, tasks):
    """Validate all requested points before starting or writing a new scan."""
    source = getattr(args, "reuse_scf_run", None)
    if not source:
        return tasks
    config = _source_config(source)
    metadata = hamiltonian_metadata(ham)
    if not hamiltonian_metadata_matches(config.get("hamiltonian"), metadata):
        raise ValueError("Saved Hartree source Hamiltonian metadata mismatch")
    raw_rows = json.loads((Path(source) / "nd_map.json").read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("Saved Hartree source nd_map.json must be a list of points")
    rows = {}
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str):
            raise ValueError("Invalid saved Hartree point record")
        if row["label"] in rows:
            raise ValueError("Duplicate saved Hartree point label: " + row["label"])
        rows[row["label"]] = row
    enriched = []
    for task in tasks:
        row = rows.get(task["label"])
        if row is None:
            raise ValueError("Missing saved Hartree point: " + task["label"])
        if not hamiltonian_metadata_matches(row, metadata):
            raise ValueError("Saved Hartree point Hamiltonian mismatch: " + task["label"])
        for key in ("n_index", "D_index", "n_cm2", "D_Vnm"):
            expected = task.get(key, metadata.get(key))
            if canonical_json(row.get(key)) != canonical_json(expected):
                raise ValueError("Saved Hartree point mismatch: " + task["label"] + " / " + key)
        if row.get("scf_status") != "ok" or row.get("scf_converged") is not True:
            raise ValueError("Saved Hartree point did not converge: " + task["label"])
        try:
            reference = {
                "n_cm2": task["n_cm2"], "D_Vnm": task["D_Vnm"], "label": task["label"],
                "U_meV": [row[f"U{layer}_meV"] for layer in range(1, len(ham.layer_indices) + 1)],
                "n_layer_cm2": [row[f"n{layer}_cm2"] for layer in range(1, len(ham.layer_indices) + 1)],
                "mu_meV": row["scf_mu_meV"], "residual_meV": row["scf_residual_meV"],
                "iterations": row["scf_iterations"], "converged": row["scf_converged"],
                **args._scf_reuse_metadata,
            }
            saved_scf_result(reference, task["n_cm2"], task["D_Vnm"], len(ham.layer_indices), args.full_tol_meV)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid saved Hartree point: " + task["label"] + "; " + str(exc)) from exc
        enriched.append({**task, "_saved_scf": reference})
    args._scf_reuse_metadata["saved_points_selected"] = len(enriched)
    return enriched
