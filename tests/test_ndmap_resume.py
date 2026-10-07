"""Durable completed points, missing-data maps and safe scan recovery."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import numpy as np
import pandas as pd
import pytest

from examples import run_nd_mapping_full_band_stoner as scan


def options(out, workers="1"):
    return ["--theta-deg", "1.35", "--cutoff", "0", "--Z", "0",
            "--grid-n1", "2", "--grid-n2", "2", "--kBT-meV", "2",
            "--full-max-iter", "100", "--stoner-random-seeds", "0",
            "--points-per-segment", "2", "--dos-bins", "12", "--detail-count", "0",
            "--n-values-cm2=1e11,2e11,3e11", "--D-values-Vnm=0.02",
            "--max-workers", workers, "--live-plot-interval-s", "0", "--out", str(out)]


def test_completed_point_is_durable_and_pending_cells_are_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["map", *options(tmp_path)])
    original = scan.run_task

    def interrupt(task, *args):
        if task["n_index"] == 1:
            raise SystemExit("simulated parent death")
        return original(task, *args)

    monkeypatch.setattr(scan, "run_task", interrupt)
    with pytest.raises(SystemExit, match="simulated parent death"):
        scan.main()
    assert len(list((tmp_path / "checkpoints/points").glob("*.json"))) == 1
    frame = pd.read_csv(tmp_path / "nd_map.csv")
    assert frame.status.tolist() == ["ok", "pending", "pending"]
    assert np.isfinite(frame.loc[0, "dos_mu_scf"])
    assert frame.loc[1:, "dos_mu_scf"].isna().all()
    with np.load(tmp_path / "nd_map_scalars.npz", allow_pickle=True) as maps:
        assert maps["point_ok"][:, 0].tolist() == [True, False, False]
        assert np.isnan(maps["dos_mu_stoner"][1:, 0]).all()
    with np.load(tmp_path / "scf/nd_map_scalars.npz") as maps:
        assert maps["scf_valid"][:, 0].tolist() == [True, False, False]


def test_resume_restores_physics_and_skips_only_completed_points(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "plot_nd_map", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["map", *options(tmp_path)])
    original, seen = scan.run_task, []

    def interrupt(task, *args):
        if task["n_index"] == 1:
            raise SystemExit("simulated parent death")
        return original(task, *args)

    monkeypatch.setattr(scan, "run_task", interrupt)
    with pytest.raises(SystemExit):
        scan.main()
    before = (tmp_path / "config_used.json").read_bytes()
    monkeypatch.setattr(scan, "run_task", lambda task, *a: seen.append(task["n_index"]) or original(task, *a))
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(tmp_path), "--max-workers", "1"])
    assert scan.main() == 0
    assert seen == [1, 2]
    assert (tmp_path / "config_used.json").read_bytes() == before
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["resumed_points"] == 1 and summary["completed_this_run"] == 2


def miniature(out, detail=False):
    from tdbg_scf.scan_checkpoint import prepare_manifest
    args = SimpleNamespace(out=str(out), resume=False, max_workers="1", live_plot_interval_s=0,
                           theta_deg=1.35, cutoff=0, stoner_flavor_nu_min=-2., stoner_flavor_nu_max=2.)
    tasks = scan.build_tasks([1e11, 2e11], [.02], 1 if detail else 0)
    fingerprint = prepare_manifest(out, args, {"model": "ABBA"}, tasks)
    return args, tasks, fingerprint


def row(task):
    return {**task, "status": "ok", "scf_status": "ok", "scf_converged": True,
            "stoner_success": True, "stoner_dos_valid": True, "detail_saved": False,
            "dos_mu_scf": .125, "dos_mu_stoner": .25}


def test_resume_rejects_changed_flavor_bounds_before_any_write(tmp_path, monkeypatch):
    miniature(tmp_path)
    path = tmp_path / "checkpoints/scan_manifest.json"
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(tmp_path), "--stoner-flavor-nu-max=4"])
    with pytest.raises(ValueError, match="Incompatible resume parameter"):
        scan.parse_args()
    assert path.read_bytes() == before


def test_resume_requires_an_existing_checkpointed_directory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["map", "--resume"])
    with pytest.raises(ValueError, match="--out"):
        scan.parse_args()


def test_damaged_record_and_failed_point_are_not_reused(tmp_path):
    from tdbg_scf.scan_checkpoint import save_point, load_points
    _, tasks, fingerprint = miniature(tmp_path)
    save_point(tmp_path, tasks[0], row(tasks[0]), fingerprint)
    path = tmp_path / "checkpoints/points" / (tasks[0]["label"] + ".json")
    data = json.loads(path.read_text())
    data["point"]["dos_mu_scf"] = 999.
    path.write_text(json.dumps(data))
    save_point(tmp_path, tasks[1], {**row(tasks[1]), "status": "failed"}, fingerprint)
    rows, reusable, rejected = load_points(tmp_path, tasks, fingerprint)
    assert not reusable and tasks[0]["label"] in rejected
    assert rows[tasks[1]["label"]]["status"] == "failed"


def test_missing_detailed_output_requires_recomputation(tmp_path):
    from tdbg_scf.scan_checkpoint import save_point, load_points
    _, tasks, fingerprint = miniature(tmp_path, detail=True)
    task = next(task for task in tasks if task["detail"])
    directory = tmp_path / "details" / task["label"]
    (directory / "scf").mkdir(parents=True)
    (directory / "summary.json").write_text("{}")
    (directory / "scf/summary.json").write_text("{}")
    save_point(tmp_path, task, {**row(task), "detail_saved": True}, fingerprint)
    assert task["label"] in load_points(tmp_path, tasks, fingerprint)[1]
    (directory / "scf/summary.json").unlink()
    assert task["label"] not in load_points(tmp_path, tasks, fingerprint)[1]


def test_directory_lock_rejects_two_writers_and_releases(tmp_path):
    from tdbg_scf.scan_checkpoint import scan_directory_lock
    with scan_directory_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already using"):
            with scan_directory_lock(tmp_path):
                pytest.fail("second writer acquired lock")
    with scan_directory_lock(tmp_path):
        pass


def test_atomic_output_keeps_previous_file_after_interruption(tmp_path):
    from tdbg_scf.atomic_io import atomic_output
    path = tmp_path / "data.json"
    path.write_text("previous complete result")
    with pytest.raises(RuntimeError, match="power loss"):
        with atomic_output(path) as temporary:
            temporary.write_text("partial")
            raise RuntimeError("power loss")
    assert path.read_text() == "previous complete result"


def test_orphan_worker_cannot_overwrite_a_new_attempt(tmp_path):
    from tdbg_scf.scan_checkpoint import begin_attempt, save_point, load_points
    args, tasks, fingerprint = miniature(tmp_path)
    old = begin_attempt(tmp_path, args)
    current = begin_attempt(tmp_path, args)
    assert not save_point(tmp_path, tasks[0], row(tasks[0]), fingerprint, attempt_id=old)
    assert save_point(tmp_path, tasks[0], row(tasks[0]), fingerprint, attempt_id=current)
    assert tasks[0]["label"] in load_points(tmp_path, tasks, fingerprint)[1]


def test_worker_commits_before_return_notification_failure(tmp_path, monkeypatch):
    from tdbg_scf.scan_checkpoint import load_points
    monkeypatch.setattr(sys, "argv", ["map", *options(tmp_path)])
    args = scan.parse_args()
    _, tasks, fingerprint = miniature(tmp_path)
    args._checkpoint_fingerprint = fingerprint

    def fail_notification(event, **kwargs):
        if event == "task_returned":
            raise RuntimeError("parent transport lost")

    monkeypatch.setattr(scan, "log_worker_event", fail_notification)
    with pytest.raises(RuntimeError, match="parent transport lost"):
        scan.run_task(tasks[0], args, str(tmp_path / "details"))
    assert tasks[0]["label"] in load_points(tmp_path, tasks, fingerprint)[1]


def test_default_filling_axis_includes_zero_and_plus_minus_four(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["map"])
    args = scan.parse_args()
    from tdbg_scf.scan_axes import resolve_density_axis
    from tdbg_scf import TDBGContinuumHamiltonian, TDBGParameters
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    densities, metadata = resolve_density_axis(args, ham.geom)
    assert len(densities) == 61
    assert all(value in metadata["nu_values"] for value in (-6., -4., 0., 4., 6.))
    assert densities[30] == 0.


def test_real_parent_hard_exit_then_parallel_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    argv = options(tmp_path)
    program = f'''import os, sys
from examples import run_nd_mapping_full_band_stoner as scan
sys.argv = ["map", *{argv!r}]
original = scan.run_task
def stop(task, *args):
    if task["n_index"] == 1:
        os._exit(77)
    return original(task, *args)
scan.run_task = stop
scan.main()
'''
    environment = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    killed = subprocess.run([sys.executable, "-c", program], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert killed.returncode == 77, killed.stdout + killed.stderr
    checkpoint = next((tmp_path / "checkpoints/points").glob("*.json"))
    contents, timestamp = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
    program = f'''import sys
from examples import run_nd_mapping_full_band_stoner as scan
sys.argv = ["map", "--resume", "--out", {str(tmp_path)!r}, "--max-workers", "2"]
scan.plot_nd_map = lambda *a, **kw: None
raise SystemExit(scan.main())
'''
    resumed = subprocess.run([sys.executable, "-c", program], cwd=root, env=environment,
                             capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert checkpoint.read_bytes() == contents and checkpoint.stat().st_mtime_ns == timestamp
    frame = pd.read_csv(tmp_path / "nd_map.csv")
    assert frame.status.tolist() == ["ok", "ok", "ok"]
    assert all(-2.-1e-8 <= value <= 2.+1e-8 for key in
               ("nu_K_up", "nu_Kp_up", "nu_K_down", "nu_Kp_down") for value in frame[key])
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["resumed_points"] == 1 and summary["completed_this_run"] == 2
    assert summary["pending_points"] == 0


def test_live_preview_is_updated_in_place_with_pending_data(tmp_path):
    from tdbg_scf.live_map import LivePreview
    preview = LivePreview(tmp_path, [0., 1e11], [-.02, .02], 30.)
    preview.update([], force=True)
    path = tmp_path / "nd_map_live.png"
    original = path.read_bytes()
    task = {"n_index": 0, "D_index": 0, "n_cm2": 0., "D_Vnm": -.02, "label": "point"}
    preview.update([{**row(task), "spin_polarization": 1.}], force=True)
    assert path.read_bytes().startswith(b"\x89PNG") and path.read_bytes() != original
    assert len(list(tmp_path.glob("nd_map_live*.png"))) == 1


def test_retry_transport_failure_keeps_previously_verified_hartree(tmp_path, monkeypatch):
    from tdbg_scf.scan_checkpoint import prepare_manifest, save_point
    monkeypatch.setattr(sys, "argv", ["map", *options(tmp_path, workers="2")])
    args = scan.parse_args()
    tasks = scan.build_tasks([1e11, 2e11, 3e11], [.02], 0)
    ham = scan.TDBGContinuumHamiltonian(scan.params_from_args(args, scan.scf_valley(args)))
    fingerprint = prepare_manifest(tmp_path, args, scan.hamiltonian_metadata(ham), tasks)
    previous = {**row(tasks[0]), "status": "failed", "stoner_success": False,
                "error": "previous Stoner failure", "error_stage": "point_workflow"}
    save_point(tmp_path, tasks[0], previous, fingerprint)
    path = tmp_path / "checkpoints/points" / (tasks[0]["label"] + ".json")
    before = path.read_bytes()

    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, *args):
            raise RuntimeError("retry transport lost before any commit")

    monkeypatch.setattr(scan, "ProcessPoolExecutor", Pool)
    original = scan.save_map_outputs
    monkeypatch.setattr(scan, "save_map_outputs", lambda *a, **kw:
                        original(*a, **{**kw, "plot_figures": False}))
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(tmp_path), "--max-workers", "2"])
    assert scan.main() == 1
    assert path.read_bytes() == before
    records = json.loads((tmp_path / "scf/nd_map.json").read_text())
    assert records[0]["status"] == "ok" and records[0]["dos_mu_scf"] == .125
    with np.load(tmp_path / "scf/nd_map_scalars.npz") as maps:
        assert maps["scf_valid"][:, 0].tolist() == [True, False, False]
