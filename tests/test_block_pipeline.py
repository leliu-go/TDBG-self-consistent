"""Detailed outputs retain compact full-basis weights from the solved states."""
import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from examples import run_full_band_stoner as workflow


def small_args():
    parser = argparse.ArgumentParser()
    workflow.add_hamiltonian_args(parser)
    workflow.add_full_scf_args(parser)
    workflow.add_stoner_args(parser)
    workflow.add_analysis_args(parser)
    return parser.parse_args([
        "--cutoff", "0", "--theta-deg", "1.2",
        "--grid-n1", "2", "--grid-n2", "2", "--points-per-segment", "2",
        "--dos-bins", "12", "--full-max-iter", "100", "--kBT-meV", "2",
    ])


def detail_fixture():
    args = small_args()
    hams = [workflow.TDBGContinuumHamiltonian(workflow.params_from_args(args, valley))
            for valley in (1, -1)]
    kpts, weights = workflow.make_uniform_mbz_grid(hams[0].geom, 2, 2)
    U = np.array([1.5, .5, -.5, -1.5])
    systems = [ham.diagonalize(kpts, U) for ham in hams]
    scf = SimpleNamespace(U_meV=U, mu_meV=.3, n_layer_cm2=np.zeros(4), history=[])
    row = {"status": "ok", "scf_status": "ok", "scf_converged": True,
           "full_mu_meV": .4, "gap_nu0_meV": 0., "gap_nu_plus4_meV": 0.,
           "gap_nu_minus4_meV": 0.,
           **{f"scf_layer_dos_L{layer}": 0. for layer in range(1, 5)}}
    return args, hams, kpts, weights, scf, row, systems


def independent_weights(ham, vectors):
    probability = np.abs(vectors) ** 2
    lower_mask = ham.layer_projectors_diagonal()[2:].sum(axis=0)
    return np.einsum("o,kob->kb", lower_mask, probability) / probability.sum(axis=1)


def isolate_detail_save(monkeypatch):
    # The exporter has its own integration tests; this test checks its input NPZs.
    monkeypatch.setattr(workflow, "export_scf_detail", lambda *a, **kw: None)
    monkeypatch.setattr(workflow, "plot_bands", lambda *a, **kw: None)
    monkeypatch.setattr(workflow, "plot_layer_profile", lambda *a, **kw: None)
    monkeypatch.setattr(workflow, "save_layer_dos_bars", lambda *a, **kw: None)
    monkeypatch.setattr(workflow, "plot_contour_panels", lambda *a, **kw: None)
    monkeypatch.setattr(workflow.plt.Figure, "savefig", lambda *a, **kw: None)
    monkeypatch.setattr(workflow.FullSCFSolver, "solve",
                        lambda *a, **kw: pytest.fail("detail export repeated SCF"))
    monkeypatch.setattr(workflow, "solve_stoner_fixed_nu",
                        lambda *a, **kw: pytest.fail("detail export repeated Stoner"))


def track_streaming_hamiltonians(monkeypatch, hams):
    calls = [[], []]
    for ham, valley_calls in zip(hams, calls):
        original = ham.hamiltonian

        def track(kx, ky, U, original=original, valley_calls=valley_calls):
            valley_calls.append([kx, ky])
            return original(kx, ky, U)

        monkeypatch.setattr(ham, "hamiltonian", track)
        monkeypatch.setattr(ham, "diagonalize",
                            lambda *a, **kw: pytest.fail("allocated a full eigenvector grid"))
    return calls


def save_detail(out, fixture, use_vectors=True, evals_K=None):
    args, hams, kpts, weights, scf, row, systems = fixture
    kwargs = {}
    if use_vectors:
        kwargs = dict(evecs_K=systems[0][1], evecs_Kp=systems[1][1])
    return workflow.save_scf_detail_outputs(
        out, row, args, hams[0], hams[1], kpts, weights, scf,
        systems[0][0] if evals_K is None else evals_K, systems[1][0], **kwargs)


def assert_compact(values, expected):
    assert values.shape == expected.shape
    assert values.dtype.kind == "f"
    assert np.isfinite(values).all()
    assert np.all((values >= 0.) & (values <= 1.))
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-14)


def test_detail_save_reuses_grid_vectors_and_preserves_path_contour_data(tmp_path, monkeypatch):
    fixture = detail_fixture()
    args, hams, kpts, _, scf, _, systems = fixture
    paths = [ham.geom.high_symmetry_path(args.points_per_segment) for ham in hams]
    path_systems = [ham.diagonalize(path[0], scf.U_meV) for ham, path in zip(hams, paths)]
    isolate_detail_save(monkeypatch)
    calls = track_streaming_hamiltonians(monkeypatch, hams)

    data = save_detail(tmp_path, fixture)

    with np.load(tmp_path / "spectrum.npz", allow_pickle=False) as saved:
        assert "evecs_K" not in saved.files and "evecs_Kp" not in saved.files
        for valley, ham, (energies, vectors) in zip(("K", "Kp"), hams, systems):
            np.testing.assert_array_equal(saved[f"evals_{valley}_meV"], energies)
            assert_compact(saved[f"lower_block_weight_{valley}"], independent_weights(ham, vectors))
    for valley, ham, path, (energies, vectors), returned, valley_calls in zip(
            ("K", "Kp"), hams, paths, path_systems, data["bands"], calls):
        with np.load(tmp_path / f"full_bands_{valley}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["kpts"], path[0])
            np.testing.assert_allclose(saved["bands"], energies, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(saved["dist"], path[1])
            np.testing.assert_array_equal(saved["ticks"], path[2])
            np.testing.assert_array_equal(saved["labels"], path[3])
            assert_compact(saved["lower_block_weight"], independent_weights(ham, vectors))
        assert len(returned) == 4
        np.testing.assert_array_equal(returned[0], path[1])
        np.testing.assert_allclose(returned[1], energies, atol=1e-10, rtol=0)
        assert returned[2] == path[2] and returned[3] == path[3]
        # Default contours and spectrum use the supplied grid vectors directly.
        np.testing.assert_array_equal(valley_calls, path[0])
    assert len(data["contours"]) == 4
    np.testing.assert_array_equal(data["contours"][0], kpts)
    with np.load(tmp_path / "full_band_contours_before_stoner.npz", allow_pickle=False) as saved:
        for valley, ham, (energies, vectors), returned in zip(
                ("K", "Kp"), hams, systems, data["contours"][1:3]):
            np.testing.assert_array_equal(saved[f"evals_{valley}_meV"], energies)
            np.testing.assert_array_equal(returned, energies)
            assert_compact(saved[f"lower_block_weight_{valley}"], independent_weights(ham, vectors))


def test_explicit_contour_grid_streams_one_full_diagonalization_per_valley(tmp_path, monkeypatch):
    fixture = detail_fixture()
    args, hams, _, _, scf, _, _ = fixture
    args.contour_n1, args.contour_n2 = 3, 2
    contour_kpts, _ = workflow.make_uniform_mbz_grid(hams[0].geom, 3, 2)
    expected = [ham.diagonalize(contour_kpts, scf.U_meV) for ham in hams]
    paths = [ham.geom.high_symmetry_path(args.points_per_segment)[0] for ham in hams]
    isolate_detail_save(monkeypatch)
    calls = track_streaming_hamiltonians(monkeypatch, hams)
    monkeypatch.setattr(workflow, "diagonalize_evals_only",
                        lambda *a, **kw: pytest.fail("separate contour eigvalsh pass"))

    data = save_detail(tmp_path, fixture)

    assert len(data["contours"]) == 4
    np.testing.assert_array_equal(data["contours"][0], contour_kpts)
    with np.load(tmp_path / "full_band_contours_before_stoner.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["kpts"], contour_kpts)
        for valley, ham, (energies, vectors), returned, path, valley_calls in zip(
                ("K", "Kp"), hams, expected, data["contours"][1:3], paths, calls):
            np.testing.assert_allclose(saved[f"evals_{valley}_meV"], energies, atol=1e-10, rtol=0)
            np.testing.assert_allclose(returned, energies, atol=1e-10, rtol=0)
            assert_compact(saved[f"lower_block_weight_{valley}"], independent_weights(ham, vectors))
            np.testing.assert_array_equal(valley_calls, np.concatenate([path, contour_kpts]))


def test_legacy_detail_call_reconstructs_checked_grid_weights(tmp_path, monkeypatch):
    fixture = detail_fixture()
    args, hams, kpts, _, _, _, systems = fixture
    paths = [ham.geom.high_symmetry_path(args.points_per_segment)[0] for ham in hams]
    isolate_detail_save(monkeypatch)
    calls = track_streaming_hamiltonians(monkeypatch, hams)

    save_detail(tmp_path, fixture, use_vectors=False)

    with np.load(tmp_path / "spectrum.npz", allow_pickle=False) as saved:
        for valley, ham, (energies, vectors), path, valley_calls in zip(
                ("K", "Kp"), hams, systems, paths, calls):
            np.testing.assert_array_equal(saved[f"evals_{valley}_meV"], energies)
            assert_compact(saved[f"lower_block_weight_{valley}"], independent_weights(ham, vectors))
            np.testing.assert_array_equal(valley_calls, np.concatenate([kpts, path]))


def test_legacy_detail_call_rejects_mismatched_fixed_potential_energies(tmp_path, monkeypatch):
    fixture = detail_fixture()
    isolate_detail_save(monkeypatch)
    changed_energies = fixture[-1][0][0] + .05

    with pytest.raises(ValueError, match="(?i)(energies|eigenvalue|spectrum).*(match|differ)|match.*(energies|eigenvalue|spectrum)"):
        save_detail(tmp_path, fixture, use_vectors=False, evals_K=changed_energies)


def test_run_point_passes_existing_grid_vectors_to_detail_save(tmp_path, monkeypatch):
    solutions, valley_systems, captured = [], [], []
    original_scf = workflow.run_scf_reference
    original_diagonalize = workflow.TDBGContinuumHamiltonian.diagonalize

    def track_scf(*args, **kwargs):
        result, metadata = original_scf(*args, **kwargs)
        solutions.append(result)
        return result, metadata

    def track_diagonalize(ham, *args, **kwargs):
        result = original_diagonalize(ham, *args, **kwargs)
        if ham.params.valley == -1:
            valley_systems.append(result)
        return result

    def capture(*args, **kwargs):
        captured.append(kwargs)
        return None

    def stop_after_scf(*args, **kwargs):
        raise RuntimeError("stop after checking SCF detail inputs")

    monkeypatch.setattr(workflow, "run_scf_reference", track_scf)
    monkeypatch.setattr(workflow.TDBGContinuumHamiltonian, "diagonalize", track_diagonalize)
    monkeypatch.setattr(workflow, "save_scf_detail_outputs", capture)
    monkeypatch.setattr(workflow, "solve_stoner_fixed_nu", stop_after_scf)
    row = workflow.run_point(dict(n_cm2=1e11, D_Vnm=.02, label="projection_inputs"),
                             small_args(), detail_dir=tmp_path)

    assert row["scf_status"] == "ok", row.get("error")
    assert len(solutions) == 1 and len(valley_systems) == 1 and len(captured) == 1
    assert captured[0].get("evecs_K") is solutions[0].evecs
    assert captured[0].get("evecs_Kp") is valley_systems[0][1]
