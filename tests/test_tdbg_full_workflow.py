"""End-to-end contracts for the original four-layer full Hartree workflow."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("script", [
    "run_full_band_stoner.py",
    "run_nd_mapping_full_band_stoner.py",
    "run_stoner_susceptibility_single.py",
    "run_nd_mapping_stoner_susceptibility.py",
    "run_fixed_nu_Dscan_stoner_susceptibility.py",
])
def test_all_entry_points_use_only_full_hartree(script):
    result = subprocess.run([sys.executable, f"examples/{script}", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--full-max-iter" in result.stdout
    assert "--full-tol-meV" in result.stdout
    for option in ("--n-active", "--scf-source", "--projector-refreshes", "--projected-reference-mode"):
        assert option not in result.stdout


def test_four_layer_model_and_cli_defaults_are_preserved():
    from examples.run_full_band_stoner import add_hamiltonian_args, params_from_args
    from tdbg_scf import TDBGContinuumHamiltonian
    parser = argparse.ArgumentParser()
    add_hamiltonian_args(parser)
    args = parser.parse_args([])
    p = params_from_args(args, +1)
    assert p.gamma1_meV == 361.0
    assert p.gamma4_meV == 138.0
    assert p.sublattice_Z_meV == 15.0
    ham = TDBGContinuumHamiltonian(p)
    assert ham.dim == 8 * (2*p.cutoff+1)**2
    assert ham.layer_projectors_diagonal().shape == (4, ham.dim)


def test_failed_hartree_skips_stoner_and_saves_diagnostics(monkeypatch, tmp_path):
    import examples.run_full_band_stoner as script
    from tdbg_scf.solver_full import FullSCFResult
    parser = argparse.ArgumentParser()
    script.add_hamiltonian_args(parser)
    script.add_full_scf_args(parser)
    script.add_stoner_args(parser)
    script.add_analysis_args(parser)
    args = parser.parse_args(["--cutoff", "0", "--grid-n1", "2", "--grid-n2", "2"])
    fake = FullSCFResult(U_meV=np.zeros(4), mu_meV=0., n_layer_a2=np.zeros(4),
                         converged=False, residual_meV=50., iterations=1, history=[])
    monkeypatch.setattr(script, "run_scf_reference", lambda *a: (fake, {"scf_source":"full", "reference_mode":"full"}))
    def forbidden(*a, **kw):
        pytest.fail("Stoner must never run for an invalid Hartree solution")
    monkeypatch.setattr(script, "solve_stoner_fixed_nu", forbidden)
    # A failed rerun must not leave a cache/figure from a previous accepted run.
    (tmp_path/"full_band_stoner_cache.npz").write_bytes(b"previous run")
    (tmp_path/"scf").mkdir()
    (tmp_path/"scf"/"dos.npz").write_bytes(b"previous run")
    (tmp_path/"user_notes.txt").write_text("preserve this", encoding="utf-8")
    import json
    for stage in ("before_stoner", "after_stoner"):
        folder = tmp_path / "origin_txt" / stage
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"format_version": 1, "status": "ok", "stage": stage}))
        (folder / "dos.txt").write_text("old accepted spectrum")
        (folder / "block_resolved_bands_and_contours.pdf").write_bytes(b"old accepted figure")
    row = script.run_point({"n_cm2":0., "D_Vnm":0., "label":"rejected"}, args, tmp_path)
    assert row["status"] == "failed"
    assert row["scf_converged"] is False
    assert row["scf_residual_meV"] == 50.
    for stage in ("before_stoner", "after_stoner"):
        folder = tmp_path / "origin_txt" / stage
        assert json.loads((folder / "manifest.json").read_text())["status"] != "ok"
        assert not (folder / "dos.txt").exists()
        assert not (folder / "block_resolved_bands_and_contours.pdf").exists()
    assert (tmp_path/"scf"/"diagnostics.json").is_file()
    assert not (tmp_path/"full_band_stoner_cache.npz").exists()
    assert not (tmp_path/"scf"/"dos.npz").exists()
    assert (tmp_path/"user_notes.txt").read_text(encoding="utf-8") == "preserve this"


def test_projection_modules_and_entry_points_are_deleted():
    root = Path(__file__).resolve().parents[1]
    for path in ("tdbg_scf/solver_projected.py", "tdbg_scf/projected_hf",
                 "examples/run_projected_8band_scf.py", "examples/run_nd_mapping_projected_8band_scf.py"):
        assert not (root/path).exists()


def test_abba_raw_moire_tunneling_has_hermitian_counterparts():
    from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian
    for valley in (-1, 1):
        ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=1, valley=valley))
        raw = np.zeros((ham.dim, ham.dim), dtype=complex)
        for g, (i, j) in enumerate(ham.lattice.G_indices):
            ham._add_tunnel_upper_to_lower(raw, g, i, j)
            ham._add_tunnel_lower_to_upper(raw, g, i, j)
        np.testing.assert_allclose(raw, raw.conj().T, atol=1e-13)
        # The same-G middle-layer block must retain the full omega coupling.
        rows = [ham._lower_index(4, i) for i in (0, 1)]
        cols = [ham._upper_index(4, i) for i in (2, 3)]
        np.testing.assert_allclose(raw[np.ix_(rows, cols)], ham.Tqb)


def test_abba_time_reversal_and_uniform_potential_gauge():
    from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian
    plus = TDBGContinuumHamiltonian(TDBGParameters(cutoff=1, valley=1))
    minus = TDBGContinuumHamiltonian(TDBGParameters(cutoff=1, valley=-1))
    reverse = np.empty(plus.dim, dtype=int)
    for g, (i, j) in enumerate(plus.lattice.G_indices):
        partner = plus.lattice.inv[(-i, -j)]
        for orbital in range(4):
            reverse[plus._upper_index(g, orbital)] = plus._upper_index(partner, orbital)
            reverse[plus._lower_index(g, orbital)] = plus._lower_index(partner, orbital)
    U = np.array([3., 1., -2., -2.])
    k = np.array([.002, -.007])
    H = plus.hamiltonian(*k, U)
    Htr = minus.hamiltonian(*(-k), U).conj()[np.ix_(reverse, reverse)]
    np.testing.assert_allclose(H, Htr, atol=1e-11)
    np.testing.assert_allclose(plus.hamiltonian(*k, U+7.5)-H,
                               7.5*np.eye(plus.dim), atol=1e-12)


def test_abba_zero_momentum_dimer_limit_and_capacitor_sign():
    from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian, LayerElectrostatics
    p = TDBGParameters(theta_deg=0., cutoff=0, omega_meV=0., sublattice_Z_meV=0.)
    H = TDBGContinuumHamiltonian(p).hamiltonian(0., 0., np.zeros(4))
    assert H[0, 3] == p.gamma1_meV  # A1 <-> B2
    assert H[5, 6] == p.gamma1_meV  # B3 <-> A4
    np.testing.assert_allclose(np.linalg.eigvalsh(H),
        [-p.gamma1_meV]*2 + [0.]*4 + [p.gamma1_meV]*2, atol=1e-12)
    electro = LayerElectrostatics(eps_perp=4., D_sign=-1.)
    _, bottom = electro.gate_densities(0., .4)
    U = electro.update_U_from_density(np.zeros(4), bottom)
    np.testing.assert_allclose(U[:-1]-U[1:], 1000*.335*.4/4., rtol=1e-12)
    assert np.mean(U) == pytest.approx(0., abs=1e-12)


def point_args(extra=()):
    import examples.run_full_band_stoner as script
    parser = argparse.ArgumentParser()
    for add in (script.add_hamiltonian_args, script.add_full_scf_args,
                script.add_stoner_args, script.add_analysis_args):
        add(parser)
    return parser.parse_args(["--cutoff", "1", "--grid-n1", "3", "--grid-n2", "3",
                             "--stoner-random-seeds", "0", "--points-per-segment", "3",
                             "--dos-bins", "41", *extra])


@pytest.mark.parametrize("valley", [-1, 1])
def test_real_full_point_conserves_charge_and_reuses_eigenstates(monkeypatch, valley):
    import examples.run_full_band_stoner as script
    calls = []
    original = script.TDBGContinuumHamiltonian.diagonalize
    def count(ham, *args, **kw):
        calls.append(ham.params.valley)
        return original(ham, *args, **kw)
    monkeypatch.setattr(script.TDBGContinuumHamiltonian, "diagonalize", count)
    row = script.run_point({"n_cm2":0., "D_Vnm":0.}, point_args(["--scf-valley", str(valley)]))
    assert row["status"] == "ok", row
    assert row["scf_full_hartree_consistent"]
    assert row["scf_full_hartree_residual_meV"] < 1e-4
    assert sum(row[f"n{i}_cm2"] for i in range(1, 5)) == pytest.approx(0., abs=1e4)
    assert row["scf_layer_dos_total"] == pytest.approx(row["dos_mu_scf"], rel=1e-10)
    assert calls.count(valley) == row["scf_iterations"]
    assert calls.count(-valley) == 1


def test_stoner_failure_retains_hartree_outputs(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import examples.run_full_band_stoner as script
    rejected = SimpleNamespace(success=False, nu_f=np.zeros(4), energy_meV_per_cell=0.,
                               seed_name="rejected", local_minima=[])
    monkeypatch.setattr(script, "solve_stoner_fixed_nu", lambda *a: rejected)
    row = script.run_point({"n_cm2":0., "D_Vnm":0.}, point_args(), tmp_path)
    assert row["status"] == "failed"
    assert row["scf_status"] == "ok"
    assert row["scf_converged"]
    assert (tmp_path/"scf"/"dos.npz").is_file()
    assert not (tmp_path/"full_band_stoner_cache.npz").exists()


def test_independent_residual_rejects_inaccurate_solver_flag(monkeypatch, tmp_path):
    import examples.run_full_band_stoner as script
    from tdbg_scf.density import full_layer_density, find_mu_for_density
    from tdbg_scf.solver_full import FullSCFResult
    args = point_args()
    ham = script.TDBGContinuumHamiltonian(script.params_from_args(args, 1))
    k, w = script.make_uniform_mbz_grid(ham.geom, args.grid_n1, args.grid_n2)
    U = np.array([100., -100., 50., -50.])
    E, V = ham.diagonalize(k, U)
    mu = find_mu_for_density(E, w, 0., kBT_meV=args.kBT_meV)
    n = full_layer_density(mu, E, V, w, ham.layer_projectors_diagonal(), kBT_meV=args.kBT_meV)
    fake = FullSCFResult(U_meV=U, mu_meV=mu, n_layer_a2=n, converged=True,
                         residual_meV=0., iterations=1, history=[], evals=E, evecs=V)
    monkeypatch.setattr(script, "run_scf_reference", lambda *a: (fake, {"scf_source":"full", "reference_mode":"full"}))
    def forbidden(*a, **kw):
        pytest.fail("Invalid full Hartree residual must stop Stoner")
    monkeypatch.setattr(script, "solve_stoner_fixed_nu", forbidden)
    row = script.run_point({"n_cm2":0., "D_Vnm":0.}, args, tmp_path)
    assert row["status"] == "failed"
    assert not row["scf_full_hartree_consistent"]
    assert row["scf_full_hartree_residual_meV"] > 1.
    assert not (tmp_path/"scf"/"dos.npz").exists()


def test_stoner_dos_figure_contains_mean_field_spectrum(tmp_path):
    from examples.run_full_band_stoner import save_dos_figure
    from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian
    args = point_args()
    E = np.array([[-2., 2.]])
    w = np.array([.25])
    mu_f = np.array([1., .5, -1., -.5])
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    save_dos_figure(tmp_path, E, E, w, 0., mu_f, args, ham)
    with np.load(tmp_path/"full_band_stoner_dos.npz") as data:
        centers = data["energy_meV"]
        expected = sum(.25*np.exp(-.5*((E.ravel()[:,None]-mu-centers)/args.dos_sigma_meV)**2).sum(axis=0)
                       /(np.sqrt(2*np.pi)*args.dos_sigma_meV) for mu in mu_f)
        np.testing.assert_allclose(data["dos"], expected, rtol=1e-13)
        assert not np.allclose(data["dos"], data["scf_dos"])


def test_scan_plot_nonuniform_cells_are_centered_on_input_points():
    from tdbg_scf.output_metadata import axis_edges
    np.testing.assert_allclose(axis_edges([0., 1., 3.]), [-.5, .5, 2., 4.])
    with pytest.raises(ValueError):
        axis_edges([1., 0.])


def test_hartree_only_rejected_rerun_removes_previous_physical_outputs(monkeypatch, tmp_path):
    import examples.run_full_scf as script
    from tdbg_scf.solver_full import FullSCFResult
    previous = ("full_scf_result.npz", "full_scf_bands.png", "dos.png", "layer_profile.png")
    for name in previous:
        (tmp_path/name).write_bytes(b"old accepted run")
    (tmp_path/"user_notes.txt").write_text("preserve this", encoding="utf-8")
    fake = FullSCFResult(U_meV=np.zeros(4), mu_meV=0., n_layer_a2=np.zeros(4),
                         converged=False, residual_meV=50., iterations=1, history=[])
    monkeypatch.setattr(script.FullSCFSolver, "solve", lambda *a: fake)
    monkeypatch.setattr(sys, "argv", ["run_full_scf.py", "--cutoff", "0", "--out", str(tmp_path)])
    assert script.main() == 1
    assert (tmp_path/"error.json").is_file()
    assert (tmp_path/"summary.json").is_file()
    assert all(not (tmp_path/name).exists() for name in previous)
    assert (tmp_path/"user_notes.txt").read_text(encoding="utf-8") == "preserve this"


def test_all_failed_map_replaces_old_cache_with_missing_data(monkeypatch, tmp_path):
    import examples.run_nd_mapping_full_band_stoner as script
    from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian
    (tmp_path/"nd_map_scalars.npz").write_bytes(b"old accepted map")
    figures = tmp_path/"figures"
    figures.mkdir()
    stale = figures/"nd_map_dos_mu_stoner.png"
    stale.write_bytes(b"old accepted map")
    unrelated = figures/"user_figure.png"
    unrelated.write_bytes(b"preserve this")
    monkeypatch.setattr(script, "save_scf_map_outputs", lambda *a, **kw: None)
    plot = script.plot_nd_map
    monkeypatch.setattr(script, "plot_nd_map", lambda frame, n, d, key, path, title:
                        plot(frame, n, d, key, path, title) if key == "dos_mu_stoner" else None)
    rows = [{"n_index":0, "D_index":0, "n_cm2":0., "D_Vnm":0.2, "status":"failed"}]
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    script.save_map_outputs(tmp_path, rows, [0.], [0.2], ham)
    with np.load(tmp_path/"nd_map_scalars.npz", allow_pickle=True) as data:
        assert len(data["rows"]) == 0
        np.testing.assert_array_equal(data["D_values_Vnm"], [0.2])
        assert np.isnan(data["dos_mu_stoner"]).all()
        assert not data["point_ok"].any()
    assert stale.read_bytes().startswith(b"\x89PNG"), "Old accepted figure must be replaced by the current blank map."
    assert unrelated.read_bytes() == b"preserve this"


def test_worker_transport_failure_preserves_worker_owned_detail_outputs(monkeypatch, tmp_path):
    import json
    import examples.run_nd_mapping_full_band_stoner as script
    class BrokenPool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, *args):
            raise RuntimeError("controlled worker submission failure")
    monkeypatch.setattr(sys, "argv", ["map", "--cutoff", "0", "--grid-n1", "2", "--grid-n2", "2",
        "--n-values-cm2=0,1e11", "--D-values-Vnm=0", "--max-workers", "2",
        "--live-plot-interval-s", "0", "--out", str(tmp_path)])
    args = script.parse_args()
    tasks = script.build_tasks([0., 1e11], [0.], args.detail_count)
    ham = script.TDBGContinuumHamiltonian(script.params_from_args(args, script.scf_valley(args)))
    script.prepare_manifest(tmp_path, args, script.hamiltonian_metadata(ham), tasks)
    task = tasks[0]
    detail = tmp_path/"details"/task["label"]
    (detail/"scf").mkdir(parents=True)
    (detail/"full_band_stoner_cache.npz").write_bytes(b"old accepted point")
    (detail/"scf"/"spectrum.npz").write_bytes(b"old accepted point")
    (detail/"user_notes.txt").write_text("preserve this")
    monkeypatch.setattr(script, "ProcessPoolExecutor", BrokenPool)
    monkeypatch.setattr(script, "save_map_outputs", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["map", "--resume", "--out", str(tmp_path), "--max-workers", "2"])
    assert script.main() == 1
    assert (detail/"full_band_stoner_cache.npz").read_bytes() == b"old accepted point"
    assert (detail/"scf"/"spectrum.npz").read_bytes() == b"old accepted point"
    diagnostics = json.loads((tmp_path/"diagnostics"/("transport_" + task["label"] + ".json")).read_text())
    assert diagnostics["error_stage"] == "worker_transport" and diagnostics["scf_converged"] is None
    assert not (detail/"error.json").exists()
    assert (detail/"user_notes.txt").read_text() == "preserve this"
