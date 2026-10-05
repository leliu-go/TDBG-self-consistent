"""Correlated scans must obtain their bands from a converged full Hartree state."""
import json
from pathlib import Path

import pytest

from tdbg_scf.solver_full import FullSCFSolver
from tdbg_scf.stoner.scan import run_stoner_scan


def small_config(tmp_path):
    from tdbg_scf.correlated_config import load_correlated_config
    cfg = load_correlated_config(Path(__file__).resolve().parents[1]/"configs"/"correlated_example.json")
    cfg["shared"] = {"nu_grid": [-.5, .5], "D_Vnm_grid": [0., .02], "output_dir": str(tmp_path)}
    cfg["hamiltonian"]["cutoff"] = 1
    cfg["grid"] = {"n1": 2, "n2": 2}
    cfg["full_scf"].update(max_iter=80, tol_meV=1e-5, kBT_meV=2.)
    cfg["stoner"] = {"u0_meV_A2": 0., "n_random_seeds": 0}
    return cfg


def test_scan_uses_full_hartree_for_every_filling_and_displacement(tmp_path, monkeypatch):
    (tmp_path/"stoner").mkdir()
    (tmp_path/"stoner"/"error.json").write_text("old rejected scan")
    states = []
    original = FullSCFSolver.solve

    def record(self, cfg):
        result = original(self, cfg)
        states.append((cfg, result))
        return result

    monkeypatch.setattr(FullSCFSolver, "solve", record)
    results = run_stoner_scan(small_config(tmp_path))
    assert len(results) == len(states) == 4
    assert all(result.success for result in results)
    assert all(state.converged and state.evals.shape == (4, 72) for _, state in states)
    assert {cfg.D_Vnm for cfg, _ in states} == {0., .02}
    assert any(cfg.target_density_cm2 < 0 for cfg, _ in states)
    records = json.loads((tmp_path/"stoner"/"results.json").read_text())
    assert len(records) == 4
    assert all(record["scf"]["scf_valid"] for record in records)
    assert {record["D_Vnm"] for record in records} == {0., .02}
    assert not (tmp_path/"stoner"/"error.json").exists()


def test_scan_rejects_precomputed_tables_instead_of_bypassing_hartree(tmp_path):
    from tdbg_scf.stoner.dos import build_flavor_band_table
    cfg = small_config(tmp_path)
    table = build_flavor_band_table([[0., 1.]], [1.], 1.)
    cfg["shared"]["nu_grid"] = [.5]
    cfg["tables"] = [table]*4
    cfg["A_M_A2"] = 1.
    with pytest.raises(ValueError, match="full.*Hartree"):
        run_stoner_scan(cfg)


def test_scan_stops_before_stoner_when_hartree_is_unconverged(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)
    cfg["shared"]["nu_grid"], cfg["shared"]["D_Vnm_grid"] = [.5], [.02]
    cfg["full_scf"].update(max_iter=1, tol_meV=1e-12)
    (tmp_path/"stoner").mkdir()
    (tmp_path/"stoner"/"results.json").write_text('[{"nu_total": 0.25}]')
    (tmp_path/"stoner"/"user_notes.txt").write_text("preserve this")
    monkeypatch.setattr("tdbg_scf.stoner.scan.solve_stoner_fixed_nu", lambda *a, **kw:
        pytest.fail("Stoner must not run on an unconverged Hartree reference"))
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        run_stoner_scan(cfg)
    assert not (tmp_path/"stoner"/"results.json").exists()
    error = json.loads((tmp_path/"stoner"/"error.json").read_text())
    assert error["stage"] == "full_hartree"
    assert error["nu_total"] == .5
    assert error["scf"]["scf_valid"] is False
    assert (tmp_path/"stoner"/"user_notes.txt").read_text() == "preserve this"


def test_scan_rejects_an_unoptimized_stoner_seed(tmp_path):
    cfg = small_config(tmp_path)
    cfg["shared"]["nu_grid"], cfg["shared"]["D_Vnm_grid"] = [.5], [0.]
    cfg["stoner"]["max_iter"] = 1
    with pytest.raises(RuntimeError, match="Stoner.*converge"):
        run_stoner_scan(cfg)
    error = json.loads((tmp_path/"stoner"/"error.json").read_text())
    assert error["stage"] == "stoner"
    assert error["scf"]["scf_valid"]
    assert not (tmp_path/"stoner"/"results.json").exists()
