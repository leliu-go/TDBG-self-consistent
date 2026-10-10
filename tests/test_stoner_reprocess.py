"""Saved full-Hartree potentials suffice for complete Stoner-only map replay."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from examples import run_full_band_stoner as point
from examples import run_nd_mapping_full_band_stoner as scan
from tdbg_scf.solver_full import FullSCFSolver
from tdbg_scf.run_storage import write_run_config


@pytest.fixture
def source_run(tmp_path, monkeypatch, request):
    source = tmp_path / "source"
    source.mkdir()
    stacking, theta = getattr(request, "param", ("AB-ABC", 1.2))
    argv = ["map", "--theta-deg", str(theta), "--cutoff", "0",
            "--grid-n1", "2", "--grid-n2", "2", "--kBT-meV", "2", "--eps-perp", "4",
            "--full-max-iter", "100", "--detail-count", "0",
            "--n-values-cm2=1e11,2e11", "--D-values-Vnm=0.02", "--max-workers", "1",
            "--points-per-segment", "2", "--dos-bins", "12", "--live-plot-interval-s", "0",
            "--out", str(source)]
    monkeypatch.setattr(sys, "argv", argv)
    args = scan.parse_args()
    ham = point.TDBGContinuumHamiltonian(point.params_from_args(args, point.scf_valley(args)))
    tasks = scan.build_tasks([1e11, 2e11], [.02], 0)
    rows = [point.run_point(task, args) for task in tasks]
    assert all(row["status"] == "ok" for row in rows), rows
    config = write_run_config(source, args, ham, "ndmap", {
        "n_values_cm2": [1e11, 2e11], "D_values_Vnm": [.02],
    })
    # The user's completed run predates both the replay flag and signed bounds.
    for key in ("reuse_scf_run", "stoner_flavor_nu_min", "stoner_flavor_nu_max"):
        config["args"].pop(key, None)
    (source / "config_used.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "nd_map.json").write_text(json.dumps(rows), encoding="utf-8")
    pd.DataFrame(rows).to_csv(source / "nd_map.csv", index=False)
    original_save = scan.save_map_outputs
    monkeypatch.setattr(scan, "save_map_outputs", lambda *a, **kw:
                        original_save(*a, **{**kw, "plot_figures": False}))
    monkeypatch.setattr(scan.LivePreview, "update", lambda *a, **kw: None)
    return source, rows


def reuse_argv(source, out, *extra):
    return ["map", "--reuse-scf-run", str(source), "--out", str(out),
            "--max-workers", "1", "--detail-count", "0", "--live-plot-interval-s", "0", *extra]


def test_replay_restores_original_physics_and_new_signed_domain(source_run, tmp_path, monkeypatch):
    source, _ = source_run
    monkeypatch.setattr(sys, "argv", reuse_argv(source, tmp_path / "replayed"))
    args = scan.parse_args()
    assert args.theta_deg == 1.2 and args.cutoff == 0
    assert args.grid_n1 == args.grid_n2 == 2
    assert args.kBT_meV == 2 and args.eps_perp == 4
    assert args.n_values_cm2 == "1e11,2e11" and args.D_values_Vnm == "0.02"
    assert args.stoner_flavor_nu_min == -2 and args.stoner_flavor_nu_max == 2
    assert args.u0_meV_A2 == 79000 and args.JH_meV_A2 == 24000


def test_complete_replay_never_calls_hartree_solver(source_run, tmp_path, monkeypatch):
    source, original_rows = source_run
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    out = tmp_path / "replayed"
    monkeypatch.setattr(sys, "argv", reuse_argv(source, out))

    def forbidden(*a, **kw):
        pytest.fail("Stoner replay must not start a Hartree iteration")

    monkeypatch.setattr(FullSCFSolver, "solve", forbidden)
    original_diag = point.TDBGContinuumHamiltonian.diagonalize
    diagonalizations = []

    def record_diag(self, kpts, U):
        diagonalizations.append(np.asarray(U).copy())
        return original_diag(self, kpts, U)

    monkeypatch.setattr(point.TDBGContinuumHamiltonian, "diagonalize", record_diag)
    assert scan.main() == 0
    rows = json.loads((out / "nd_map.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 and len(diagonalizations) == 4
    for i, (old, new) in enumerate(zip(original_rows, rows)):
        assert new["scf_reused"] is True
        assert new["scf_iterations_performed"] == 0
        assert new["scf_full_hartree_consistent"] is True
        assert new["stoner_dos_valid"] is True
        U = [old[f"U{layer}_meV"] for layer in range(1, 5)]
        assert np.array_equal(diagonalizations[2*i], U)
        assert np.array_equal(diagonalizations[2*i+1], U)
        for key in ("dos_mu_scf", "dos_mu_stoner", "stoner_energy_meV_per_cell", "full_mu_meV",
                    "gap_nu0_meV", "gap_nu_plus4_meV", "gap_nu_minus4_meV",
                    "stoner_P_top_bottom_cm2", "stoner_layer_dos_total"):
            assert new[key] == pytest.approx(old[key], rel=1e-7, abs=1e-7), key
        for layer in range(1, 5):
            for template in ("U{}_meV", "n{}_cm2", "stoner_n{}_cm2", "stoner_layer_dos_L{}"):
                key = template.format(layer)
                assert new[key] == pytest.approx(old[key], rel=1e-7, abs=1e-7), key
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["saved_hartree_points_reused"] == 2
    assert summary["hartree_iterations_performed"] == 0
    assert summary["initial_runtime_estimate"]["hartree_iteration_range"] == [0, 0]
    assert summary["initial_runtime_estimate"]["full_valley_spectra_per_point"] == 2
    config = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
    assert config["scf_reuse"]["source_directory"] == str(source.resolve())
    assert config["scf_reuse"]["source_nd_map_sha256"]
    assert "_saved_scf" not in rows[0]


@pytest.mark.parametrize("option,value", [("--theta-deg", "1.3"), ("--grid-n1", "3"),
                                             ("--eps-perp", "5"), ("--D-values-Vnm", ".04"),
                                             ("--n-values-cm2", "3e11")])
def test_replay_rejects_changed_hartree_inputs(source_run, tmp_path, monkeypatch, option, value):
    source, _ = source_run
    out = tmp_path / "replayed"
    monkeypatch.setattr(sys, "argv", reuse_argv(source, out, option + "=" + value))
    with pytest.raises(ValueError, match="[Ii]ncompat|[Dd]iffer|[Mm]ismatch"):
        scan.main()
    assert not out.exists()


@pytest.mark.parametrize("damage", ["nonconverged", "missing_U", "nan_U", "missing_point", "wrong_density", "wrong_hamiltonian"])
def test_replay_rejects_bad_saved_points_without_hartree_fallback(source_run, tmp_path, monkeypatch, damage):
    source, rows = source_run
    if damage == "nonconverged":
        rows[0]["scf_converged"] = False
    elif damage == "missing_U":
        del rows[0]["U2_meV"]
    elif damage == "nan_U":
        rows[0]["U1_meV"] = float("nan")
    elif damage == "missing_point":
        rows.pop()
    elif damage == "wrong_density":
        rows[0]["n_cm2"] = 9e11
    else:
        rows[0]["stacking"] = "AB-ABA"
    (source / "nd_map.json").write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "replayed"
    monkeypatch.setattr(sys, "argv", reuse_argv(source, out))
    monkeypatch.setattr(FullSCFSolver, "solve", lambda *a, **kw: pytest.fail("No Hartree fallback"))
    with pytest.raises(ValueError, match="[Ss]aved|[Ss]ource|[Cc]onverg|[Mm]ismatch|[Mm]issing"):
        scan.main()
    assert not out.exists()


def test_replay_rejects_changed_hamiltonian_source_code(source_run, tmp_path, monkeypatch):
    source, _ = source_run
    path = source / "config_used.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["source_sha256"]["continuum.py"] = "0" * 64
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", reuse_argv(source, tmp_path / "replayed"))
    with pytest.raises(ValueError, match="[Ss]ource|[Cc]ode|[Mm]ismatch"):
        scan.main()


@pytest.mark.parametrize("inside", [False, True])
def test_replay_cannot_write_into_original_run(source_run, monkeypatch, inside):
    source, _ = source_run
    out = source / "new_stoner" if inside else source
    monkeypatch.setattr(sys, "argv", reuse_argv(source, out))
    with pytest.raises(ValueError, match="[Ss]ource|[Oo]riginal|[Oo]utput"):
        scan.main()
    assert not (source / "checkpoints").exists()


def test_replay_can_resume_completed_new_run(source_run, tmp_path, monkeypatch):
    source, _ = source_run
    out = tmp_path / "replayed"
    monkeypatch.setattr(sys, "argv", reuse_argv(source, out))
    monkeypatch.setattr(FullSCFSolver, "solve", lambda *a, **kw: pytest.fail("No Hartree during replay"))
    assert scan.main() == 0
    before = (out / "config_used.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(out), "--max-workers", "1"])
    monkeypatch.setattr(scan, "run_task", lambda *a, **kw: pytest.fail("Completed replay points must be reused"))
    assert scan.main() == 0
    assert (out / "config_used.json").read_bytes() == before
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["resumed_points"] == 2 and summary["completed_this_run"] == 0


PS = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")




@pytest.mark.skipif(not PS.exists(), reason="native PowerShell is required")
def test_powershell_inplace_runs_through_package_entry_point(source_run):
    source, originals = source_run
    before = (source / "nd_map.json").read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/run_ndmap_windows.ps1"
    result = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                             "-PythonBin", sys.executable, "-ReuseScfRun", str(source),
                             "-UpdateStonerInPlace", "-Workers", "20",
                             "-StonerFlavorNuMin", "-2", "-StonerFlavorNuMax", "2"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((source / "config_used.json").read_text(encoding="utf-8"))
    assert config["hamiltonian"]["stacking"] == "BA-AB"
    assert config["args"]["max_workers"] == "20"
    backup = Path(config["args"]["reuse_scf_run"])
    assert (backup / "nd_map.json").read_bytes() == before
    rows = json.loads((source / "nd_map.json").read_text(encoding="utf-8"))
    assert len(rows) == len(originals) == 2
    assert all(row["scf_reused"] and row["scf_iterations_performed"] == 0 for row in rows)
    assert all(row["stoner_dos_valid"] for row in rows)
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["failed_points"] == 0 and summary["hartree_iterations_performed"] == 0


@pytest.mark.skipif(not PS.exists(), reason="native PowerShell is required")
def test_powershell_replay_preview_restores_original_grid_and_forwards_only_overrides(source_run, tmp_path):
    source, _ = source_run
    script = Path(__file__).resolve().parents[1] / "scripts/run_ndmap_windows.ps1"
    result = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                             "-ReuseScfRun", str(source), "-Workers", "2", "-Out", str(tmp_path / "replayed"),
                             "-StonerFlavorNuMin", "-2", "-StonerFlavorNuMax", "2", "-Preview"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    preview = json.loads(result.stdout)
    argv = preview["python_arguments"]
    assert argv[argv.index("--reuse-scf-run") + 1] == str(source)
    assert "--theta-deg" not in argv and "--grid-n1" not in argv
    assert "--stoner-flavor-nu-min=-2" in argv and "--stoner-flavor-nu-max=2" in argv
    assert preview["resource_estimates"]["hamiltonian_dimension"] == 8


def inplace_argv(source, *extra):
    return ["map", "--reuse-scf-run", str(source), "--update-stoner-in-place",
            "--max-workers", "1", "--detail-count", "0", "--live-plot-interval-s", "0", *extra]


def test_inplace_updates_original_map_and_keeps_immutable_backup(source_run, monkeypatch):
    source, originals = source_run
    before = {name: (source / name).read_bytes() for name in ("config_used.json", "nd_map.json", "nd_map.csv")}
    for name in ("figures", "details", "diagnostics", "checkpoints"):
        (source / name).mkdir()
        (source / name / "old.txt").write_text("old artifact", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    monkeypatch.setattr(FullSCFSolver, "solve", lambda *a, **kw: pytest.fail("No Hartree during update"))
    assert scan.main() == 0
    config = json.loads((source / "config_used.json").read_text(encoding="utf-8"))
    backup = Path(config["args"]["reuse_scf_run"])
    assert backup.is_relative_to(source) if sys.version_info >= (3, 9) else source in backup.parents
    assert backup != source
    for name, content in before.items():
        assert (backup / name).read_bytes() == content
    for name in ("figures", "details", "diagnostics", "checkpoints"):
        assert (backup / name / "old.txt").read_text(encoding="utf-8") == "old artifact"
        assert not (source / name / "old.txt").exists()
    assert config["args"]["out"] == str(source)
    assert config["args"]["update_stoner_in_place"] is True
    rows = json.loads((source / "nd_map.json").read_text(encoding="utf-8"))
    assert all(row["scf_reused"] and row["scf_iterations_performed"] == 0 for row in rows)
    assert all(row["stoner_dos_valid"] for row in rows)
    with np.load(source / "scf/nd_map_scalars.npz") as maps:
        assert maps["scf_valid"].all()
        assert np.allclose(maps["dos_mu_scf"][:, 0], [r["dos_mu_scf"] for r in originals])
    with np.load(source / "energy_gap_maps.npz") as maps:
        assert maps["gap_valid"].all()


def test_inplace_partial_update_keeps_full_hartree_map_and_resumes_from_backup(source_run, monkeypatch):
    source, originals = source_run
    before_map = (source / "nd_map.json").read_bytes()
    original_task = scan.run_task

    def interrupt(task, *args):
        if task["n_index"] == 1:
            raise SystemExit("interrupted update")
        return original_task(task, *args)

    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    monkeypatch.setattr(scan, "run_task", interrupt)
    monkeypatch.setattr(FullSCFSolver, "solve", lambda *a, **kw: pytest.fail("No Hartree during update"))
    with pytest.raises(SystemExit, match="interrupted update"):
        scan.main()
    frame = pd.read_csv(source / "nd_map.csv")
    assert frame.status.tolist() == ["ok", "pending"]
    assert np.isfinite(frame.loc[0, "dos_mu_stoner"]) and np.isnan(frame.loc[1, "dos_mu_stoner"])
    with np.load(source / "scf/nd_map_scalars.npz") as maps:
        assert maps["scf_valid"].all()
    with np.load(source / "energy_gap_maps.npz") as maps:
        assert maps["gap_valid"].all()
    config = json.loads((source / "config_used.json").read_text(encoding="utf-8"))
    backup = Path(config["args"]["reuse_scf_run"])
    assert (backup / "nd_map.json").read_bytes() == before_map
    seen = []
    monkeypatch.setattr(scan, "run_task", lambda task, *a: seen.append(task["n_index"]) or original_task(task, *a))
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(source), "--max-workers", "1"])
    assert scan.main() == 0
    assert seen == [1]
    assert (backup / "nd_map.json").read_bytes() == before_map
    assert len(list((source / "_stoner_backups").iterdir())) == 1
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["resumed_points"] == 1 and summary["hartree_iterations_performed"] == 0


def test_inplace_setup_can_restart_after_backup_before_new_manifest(source_run, monkeypatch):
    source, _ = source_run
    before = (source / "nd_map.json").read_bytes()
    original_prepare = scan.prepare_manifest
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    monkeypatch.setattr(scan, "prepare_manifest", lambda *a, **kw: (_ for _ in ()).throw(SystemExit("setup interrupted")))
    with pytest.raises(SystemExit, match="setup interrupted"):
        scan.main()
    assert not (source / "checkpoints/scan_manifest.json").exists()
    assert (source / "nd_map.json").read_bytes() == before
    monkeypatch.setattr(scan, "prepare_manifest", original_prepare)
    assert scan.main() == 0
    assert len(list((source / "_stoner_backups").iterdir())) == 1


def test_inplace_repeat_requires_resume_and_keeps_completed_map(source_run, monkeypatch):
    source, _ = source_run
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    assert scan.main() == 0
    before = (source / "nd_map.json").read_bytes()
    with pytest.raises(ValueError, match="[Rr]esume|--resume"):
        scan.main()
    assert (source / "nd_map.json").read_bytes() == before
    assert len(list((source / "_stoner_backups").iterdir())) == 1


def test_explicit_inplace_restart_recomputes_all_stoner_points_and_preserves_hartree(source_run, monkeypatch):
    source, _ = source_run
    original_map = (source / "nd_map.json").read_bytes()
    monkeypatch.setattr(FullSCFSolver, "solve", lambda *a, **kw: pytest.fail("No Hartree during either replay"))
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    assert scan.main() == 0
    previous = {name: (source / name).read_bytes() for name in ("config_used.json", "nd_map.json", "stoner_energy_gap_maps.npz")
                if (source / name).exists()}
    original_backup = Path(json.loads(previous["config_used.json"])["args"]["reuse_scf_run"])
    (source / "details/previous_stoner.txt").write_text("previous Stoner detail", encoding="utf-8")
    original_task, seen = scan.run_task, []
    monkeypatch.setattr(scan, "run_task", lambda task, *a: seen.append(task["n_index"]) or original_task(task, *a))
    monkeypatch.setattr(sys, "argv", inplace_argv(source, "--restart-stoner-in-place", "--stoner-seed=11"))
    assert scan.main() == 0
    assert seen == [0, 1], "A new generation must not reuse previous Stoner checkpoints"
    config = json.loads((source / "config_used.json").read_text(encoding="utf-8"))
    assert Path(config["args"]["reuse_scf_run"]) == original_backup
    assert (original_backup / "nd_map.json").read_bytes() == original_map
    assert config["stoner_config"]["seed"] == 11
    archives = [p for p in (source / "_stoner_backups").iterdir() if p != original_backup]
    assert len(archives) == 1
    for name, content in previous.items():
        assert (archives[0] / name).read_bytes() == content
    assert (archives[0] / "details/previous_stoner.txt").read_text(encoding="utf-8") == "previous Stoner detail"
    rows = json.loads((source / "nd_map.json").read_text(encoding="utf-8"))
    assert all(row["scf_iterations_performed"] == 0 for row in rows)
    assert all(row["stoner_search_method"] == "differential_evolution_3d_two_seeds" for row in rows)
    assert all(row["stoner_gap_valid"] for row in rows)
    for row in rows:
        assert row["stoner_minimum_validation"]["de_seeds"] == [11, 12]


def test_restarted_inplace_generation_can_resume_without_repeating_hartree_or_completed_points(source_run, monkeypatch):
    source, _ = source_run
    monkeypatch.setattr(FullSCFSolver, "solve", lambda *a, **kw: pytest.fail("No Hartree in replay or resume"))
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    assert scan.main() == 0
    original_task = scan.run_task

    def interrupt(task, *a):
        if task["n_index"] == 1:
            raise SystemExit("new generation interrupted")
        return original_task(task, *a)

    monkeypatch.setattr(scan, "run_task", interrupt)
    monkeypatch.setattr(sys, "argv", inplace_argv(source, "--restart-stoner-in-place", "--stoner-seed=17"))
    with pytest.raises(SystemExit, match="new generation interrupted"):
        scan.main()
    with np.load(source / "stoner_energy_gap_maps.npz", allow_pickle=False) as maps:
        assert maps["stoner_gap_valid"][:, 0].tolist() == [True, False]
        assert np.isnan(maps["stoner_gap_nu_plus1_meV"][1, 0])
    seen = []
    monkeypatch.setattr(scan, "run_task", lambda task, *a: seen.append(task["n_index"]) or original_task(task, *a))
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(source), "--max-workers", "1"])
    assert scan.main() == 0
    assert seen == [1]
    assert len(list((source / "_stoner_backups").iterdir())) == 2
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["resumed_points"] == 1 and summary["hartree_iterations_performed"] == 0


def test_inplace_restart_recovers_interruption_during_archive_without_extra_generation(source_run, monkeypatch):
    from tdbg_scf import stoner_update

    source, _ = source_run
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    assert scan.main() == 0
    old_map = (source / "nd_map.json").read_bytes()
    (source / "details/previous_stoner.txt").write_text("previous detail", encoding="utf-8")
    original_replace = stoner_update.os.replace

    def stop_after_move(old, new):
        original_replace(old, new)
        if Path(old) == source / "details":
            raise SystemExit("restart archive interrupted")

    monkeypatch.setattr(stoner_update.os, "replace", stop_after_move)
    monkeypatch.setattr(sys, "argv", inplace_argv(source, "--restart-stoner-in-place"))
    with pytest.raises(SystemExit, match="restart archive interrupted"):
        scan.main()
    assert (source / "nd_map.json").read_bytes() == old_map
    assert len(list((source / "_stoner_backups").iterdir())) == 2
    monkeypatch.setattr(stoner_update.os, "replace", original_replace)
    assert scan.main() == 0
    assert len(list((source / "_stoner_backups").iterdir())) == 2
    rows = json.loads((source / "nd_map.json").read_text(encoding="utf-8"))
    assert all(row["scf_iterations_performed"] == 0 and row["stoner_gap_valid"] for row in rows)


@pytest.mark.skipif(not PS.exists(), reason="native PowerShell is required")
def test_powershell_restart_preview_forwards_explicit_new_generation_flag(source_run):
    source, _ = source_run
    script = Path(__file__).resolve().parents[1] / "scripts/run_ndmap_windows.ps1"
    result = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                             "-ReuseScfRun", str(source), "-RestartStonerInPlace", "-Workers", "20", "-Preview"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads(result.stdout)["python_arguments"]
    assert "--restart-stoner-in-place" in argv
    assert "--update-stoner-in-place" in argv
    assert argv[argv.index("--out") + 1] == str(source)
    assert "--resume" not in argv


def test_inplace_requires_a_source_and_rejects_different_output(source_run, tmp_path, monkeypatch):
    source, _ = source_run
    monkeypatch.setattr(sys, "argv", ["map", "--update-stoner-in-place"])
    with pytest.raises(ValueError, match="[Ss]ource|reuse-scf-run"):
        scan.parse_args()
    monkeypatch.setattr(sys, "argv", inplace_argv(source, "--out", str(tmp_path / "elsewhere")))
    with pytest.raises(ValueError, match="[Oo]utput|[Ss]ame|[Oo]riginal"):
        scan.main()
    assert not (source / "_stoner_backups").exists()


def test_inplace_rejects_physics_changes_before_backup(source_run, monkeypatch):
    source, _ = source_run
    monkeypatch.setattr(sys, "argv", inplace_argv(source, "--eps-perp=5"))
    with pytest.raises(ValueError, match="[Ii]ncompat|[Dd]iffer|[Mm]ismatch"):
        scan.main()
    assert not (source / "_stoner_backups").exists()


@pytest.mark.skipif(not PS.exists(), reason="native PowerShell is required")
def test_powershell_inplace_preview_uses_source_as_output(source_run):
    source, _ = source_run
    script = Path(__file__).resolve().parents[1] / "scripts/run_ndmap_windows.ps1"
    result = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                             "-ReuseScfRun", str(source), "-UpdateStonerInPlace", "-Workers", "20", "-Preview"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    preview = json.loads(result.stdout)
    argv = preview["python_arguments"]
    assert "--update-stoner-in-place" in argv
    assert argv[argv.index("--out") + 1] == str(source)
    assert argv[argv.index("--reuse-scf-run") + 1] == str(source)
    assert argv[argv.index("--max-workers") + 1] == "20"


def test_inplace_recovers_interruption_after_archiving_directory(source_run, monkeypatch):
    from tdbg_scf import stoner_update
    source, _ = source_run
    before = (source / "nd_map.json").read_bytes()
    (source / "details").mkdir()
    (source / "details/old.txt").write_text("original detail", encoding="utf-8")
    original_replace = stoner_update.os.replace

    def interrupt_after_move(old, new):
        original_replace(old, new)
        if Path(old) == source / "details":
            raise SystemExit("archive interrupted")

    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    monkeypatch.setattr(stoner_update.os, "replace", interrupt_after_move)
    with pytest.raises(SystemExit, match="archive interrupted"):
        scan.main()
    assert not (source / "details").exists()
    assert (source / "nd_map.json").read_bytes() == before
    monkeypatch.setattr(stoner_update.os, "replace", original_replace)
    assert scan.main() == 0
    backup = Path(json.loads((source / "config_used.json").read_text(encoding="utf-8"))["args"]["reuse_scf_run"])
    assert (backup / "details/old.txt").read_text(encoding="utf-8") == "original detail"


@pytest.mark.parametrize("physical_failure", [False, True])
def test_inplace_baseline_survives_transport_failure_but_rejects_physical_failure(source_run, monkeypatch, physical_failure):
    source, _ = source_run
    original = scan.run_task

    def failed(task, *args):
        if task["n_index"] == 0:
            return {"status": "failed", "scf_status": "failed" if physical_failure else "unknown",
                    "scf_converged": False if physical_failure else None, "stoner_success": False,
                    "error_stage": "point_workflow" if physical_failure else "worker_transport"}
        return original(task, *args)

    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    monkeypatch.setattr(scan, "run_task", failed)
    assert scan.main() == 1
    with np.load(source / "scf/nd_map_scalars.npz") as maps:
        assert maps["scf_valid"][:, 0].tolist() == [not physical_failure, True]
    with np.load(source / "energy_gap_maps.npz") as maps:
        assert maps["gap_valid"][:, 0].tolist() == [not physical_failure, True]


def test_inplace_resume_rejects_corrupted_original_backup(source_run, monkeypatch):
    source, _ = source_run
    monkeypatch.setattr(sys, "argv", inplace_argv(source))
    assert scan.main() == 0
    config = json.loads((source / "config_used.json").read_text(encoding="utf-8"))
    backup = Path(config["args"]["reuse_scf_run"])
    (backup / "nd_map.json").write_text("[]", encoding="utf-8")
    before = (source / "nd_map.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(source)])
    with pytest.raises(ValueError, match="[Cc]hecksum|[Bb]ackup|[Mm]ismatch"):
        scan.main()
    assert (source / "nd_map.json").read_bytes() == before


def test_replay_accepts_legacy_additive_metadata_without_repeating_hartree(source_run,tmp_path,monkeypatch):
    source,rows=source_run
    path=source/'config_used.json'
    cfg=json.loads(path.read_text(encoding='utf-8'))
    added={'stacking','stacking_top_to_bottom','bernal_AB_convention','upper_dimer','lower_dimer'}
    for key in added: cfg['hamiltonian'].pop(key,None)
    for row in rows:
        for key in added: row.pop(key,None)
        for key in list(row):
            if key.startswith(('gap_','indirect_gap_','stoner_gap_')): row.pop(key,None)
    path.write_text(json.dumps(cfg),encoding='utf-8')
    (source/'nd_map.json').write_text(json.dumps(rows),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',reuse_argv(source,tmp_path/'legacy_replay'))
    monkeypatch.setattr(FullSCFSolver,'solve',lambda *a,**kw:pytest.fail('Legacy replay repeated Hartree'))
    assert scan.main()==0
    new=json.loads((tmp_path/'legacy_replay'/'nd_map.json').read_text(encoding='utf-8'))
    assert all(row['scf_iterations_performed']==0 and row['stoner_dos_valid'] for row in new)
