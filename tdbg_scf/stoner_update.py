"""Recoverable setup for Stoner replay in an existing result directory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

from .atomic_io import atomic_json, atomic_output
from .run_storage import now_sgt


STATE_NAME = "stoner_update_state.json"
SNAPSHOT_FILES = ("config_used.json", "nd_map.json", "nd_map.csv", "nd_map_scalars.npz",
                  "energy_gap_maps.npz", "stoner_energy_gap_maps.npz", "run_summary.json", "nd_map_live.png")
ARCHIVE_DIRECTORIES = ("details", "figures", "diagnostics", "errors", "checkpoints")


def _child(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError("Backup/archive path must be inside the original result directory")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked_snapshot(out, relative, checksum):
    backup = _child(out / "_stoner_backups", out / relative)
    manifest = json.loads((backup / "snapshot.json").read_text(encoding="utf-8"))
    if checksum != _sha256(backup / "snapshot.json"):
        raise ValueError("Stoner source backup manifest checksum mismatch")
    for name, info in manifest["files"].items():
        if name not in SNAPSHOT_FILES or _sha256(backup / name) != info["sha256"]:
            raise ValueError("Stoner source backup checksum mismatch: " + name)
    return backup


def _read_state(out):
    out = Path(out).resolve()
    path = out / STATE_NAME
    if not path.is_file():
        return None, None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1:
        raise ValueError("Incompatible Stoner update state version")
    backup = _checked_snapshot(out, state["backup_relative"], state.get("snapshot_sha256"))
    if "archive_backup_relative" in state:
        _checked_snapshot(out, state["archive_backup_relative"], state.get("archive_snapshot_sha256"))
    return state, backup


def resolve_inplace_source(args):
    """Resolve a prior setup/backup before the original map can be overwritten."""
    source = getattr(args, "reuse_scf_run", None)
    if getattr(args, "restart_stoner_in_place", False):
        if args.resume:
            raise ValueError("Stoner restart begins a new generation; use --resume without --restart-stoner-in-place to continue one")
        args.update_stoner_in_place = True
    if not getattr(args, "update_stoner_in_place", False):
        return source
    if not source:
        raise ValueError("--update-stoner-in-place requires --reuse-scf-run")
    source = Path(source).expanduser().resolve()
    if not args.out:
        args.out = str(source)
    out = Path(args.out).expanduser().resolve()
    state, backup = _read_state(out)
    if args.resume:
        if state is None or source != backup:
            raise ValueError("In-place Stoner resume source does not match the original backup")
        return str(backup)
    if source != out:
        raise ValueError("In-place Stoner output must be the same as the original source directory")
    if state is not None:
        manifest_path = out / "checkpoints/scan_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("saved_args", {}).get("update_stoner_in_place")
                    and state.get("setup_complete") and not getattr(args, "restart_stoner_in_place", False)):
                raise ValueError("Stoner update already initialized; use --resume --out to continue or --restart-stoner-in-place for a new calculation")
        return str(backup)
    return str(source)


def _snapshot_files(out):
    """Copy compact results before any old derived directories move."""
    out = Path(out).resolve()
    name = "before_" + now_sgt().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    backup = _child(out, out / "_stoner_backups" / name)
    backup.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in SNAPSHOT_FILES:
        source = _child(out, out / name)
        if not source.is_file():
            continue
        data = source.read_bytes()
        with atomic_output(backup / name) as temporary:
            temporary.write_bytes(data)
        files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if not {"config_used.json", "nd_map.json"}.issubset(files):
        raise ValueError("Original Stoner update source lacks its saved Hartree configuration/map")
    atomic_json(backup / "snapshot.json", {"files": files, "created_at": now_sgt().isoformat()})
    return backup


def _create_snapshot(out):
    out = Path(out).resolve()
    backup = _snapshot_files(out)
    state = {
        "schema_version": 1, "backup_relative": backup.relative_to(out).as_posix(),
        "snapshot_sha256": _sha256(backup / "snapshot.json"),
        "archive_completed": [], "setup_complete": False,
    }
    atomic_json(out / STATE_NAME, state)
    return state, backup


def _start_restart_archive(out, state):
    """Keep the original Hartree source; journal another Stoner generation."""
    backup = _snapshot_files(out)
    state.update({"archive_backup_relative": backup.relative_to(out).as_posix(),
                  "archive_snapshot_sha256": _sha256(backup / "snapshot.json"),
                  "archive_completed": [], "setup_complete": False,
                  "restart_generation": int(state.get("restart_generation", 0)) + 1})
    atomic_json(out / STATE_NAME, state)
    return state


def _archive_old_directories(out, state, backup):
    """Same-volume moves retain old large details without a second full copy."""
    for name in ARCHIVE_DIRECTORIES:
        if name in state["archive_completed"]:
            continue
        source, destination = _child(out, out / name), _child(out, backup / name)
        if source.exists():
            if destination.exists():
                raise ValueError("Conflicting original/archive directories: " + name)
            os.replace(str(source), str(destination))
        state["archive_completed"].append(name)
        atomic_json(out / STATE_NAME, state)
    state["setup_complete"] = True
    atomic_json(out / STATE_NAME, state)


def prepare_inplace_update(args, ham, tasks, out):
    """Called only after full point validation and while holding the scan lock."""
    if not getattr(args, "update_stoner_in_place", False):
        return tasks
    out = Path(out).resolve()
    state, backup = _read_state(out)
    if not args.resume and state is not None:
        manifest_path = out / "checkpoints/scan_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("saved_args", {}).get("update_stoner_in_place") and state.get("setup_complete"):
                if not getattr(args, "restart_stoner_in_place", False):
                    raise ValueError("Stoner update already initialized; use --resume --out to continue or --restart-stoner-in-place for a new calculation")
                state = _start_restart_archive(out, state)
    if state is None:
        if args.resume:
            raise ValueError("Missing Stoner update source backup")
        state, backup = _create_snapshot(out)
        expected = args._scf_reuse_metadata
        if (_sha256(backup / "config_used.json") != expected["source_config_sha256"]
                or _sha256(backup / "nd_map.json") != expected["source_nd_map_sha256"]):
            raise ValueError("Original Hartree source changed after preflight; no results were replaced")
    if not args.resume:
        # Fence any orphan worker from the previous generation before its
        # checkpoints move. A later begin_attempt creates the new worker token.
        from .scan_checkpoint import _commit_lock
        active = out / "checkpoints/active_attempt.json"
        if active.is_file():
            with _commit_lock(out):
                atomic_json(active, {"attempt_id": "archived_" + uuid.uuid4().hex,
                                     "fingerprint": "invalidated_before_stoner_replay"})
        archive = (out / state["archive_backup_relative"] if "archive_backup_relative" in state else backup)
        _archive_old_directories(out, state, archive)
    elif not state.get("setup_complete"):
        raise ValueError("Stoner update initialization is incomplete; repeat the original update command")

    # Delay these imports to keep source restoration independent of setup.
    from .scf_reuse import attach_saved_scf_references, record_reuse_source
    args.reuse_scf_run = str(backup)
    record_reuse_source(args, backup)
    args._scf_reuse_metadata.update({"update_stoner_in_place": True,
        "original_source_directory": str(out), "backup_directory": str(backup)})
    if "archive_backup_relative" in state:
        args._scf_reuse_metadata.update({"previous_stoner_archive_directory": str(out / state["archive_backup_relative"]),
                                        "restart_generation": int(state["restart_generation"])})
    args._inplace_update_prepared = True
    return attach_saved_scf_references(args, ham, tasks)
