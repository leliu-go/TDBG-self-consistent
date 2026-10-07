"""First-order Stoner stationarity does not certify a local minimum."""
from types import SimpleNamespace

import numpy as np
import pytest

from tdbg_scf.stoner.dos import FlavorBandTable, build_flavor_band_table
from tdbg_scf.stoner.energy import stoner_energy_meV_per_cell
from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu


def _constant_optimizer(monkeypatch, values):
    monkeypatch.setattr("scipy.optimize.minimize", lambda *a, **kw:
        SimpleNamespace(x=np.array(values, copy=True), success=True, message="stationary"))


def test_real_slsqp_stationary_quartic_band_saddle_is_rejected():
    nodes = np.linspace(0., 1., 1001)
    conduction = np.diff(nodes**4)/np.diff(nodes)
    energies = np.r_[np.full(1000, -10.), conduction][None]
    table = build_neutrality_referenced_band_table(energies, [.001], 1.)
    tables = [table]*4
    params = StonerParams(u0_meV_A2=.9, max_iter=2, n_random_seeds=0)
    result = solve_stoner_fixed_nu(1.002, tables, params, 1.)
    candidate = next(item for item in result.local_minima if item["seed_name"] == "paramagnetic:slsqp")
    np.testing.assert_allclose(candidate["nu_f"], np.full(4, .2505), atol=1e-12)
    assert candidate["stationarity"]["optimization_valid"]
    redistributed = candidate["nu_f"] + np.array([.0004, -.0004, 0., 0.])
    gain = candidate["energy_meV_per_cell"]-stoner_energy_meV_per_cell(redistributed, tables, params, 1.)
    assert gain == pytest.approx(1.44e-7, rel=1e-6)
    assert gain > params.tol_energy_meV
    assert not candidate["success"]
    assert not candidate["local_stability"]["valid"]


def test_coupled_hund_descent_is_detected_when_every_pair_transfer_is_stable(monkeypatch):
    table = build_flavor_band_table([[0., 0.]], [1.], 1.)
    tables, values = [table]*4, np.full(4, .25)
    params = StonerParams(u0_meV_A2=-1., JH_meV_A2=.75, n_random_seeds=0)
    baseline = stoner_energy_meV_per_cell(values, tables, params, 1.)
    for receiver in range(4):
        for donor in range(receiver):
            pair = values+.1*(np.eye(4)[receiver]-np.eye(4)[donor])
            assert stoner_energy_meV_per_cell(pair, tables, params, 1.) > baseline
    coupled = values+np.array([.1, .1, -.1, -.1])
    assert stoner_energy_meV_per_cell(coupled, tables, params, 1.) < baseline-.009
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., tables, params, 1.)
    assert not result.success
    assert all(not item["success"] for item in result.local_minima)


def test_true_kinetic_kink_minimum_survives_negative_interaction_curvature(monkeypatch):
    table = FlavorBandTable(np.array([0., .25, 1.]), np.array([0., 0., 10.]),
                            np.array([0., 0., 7.5]), 0., 1.)
    values = np.full(4, .25)
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=.9, n_random_seeds=0), 1.)
    assert result.success
    np.testing.assert_allclose(result.nu_f, values)


def test_carrier_boundary_minimum_survives_negative_interaction_curvature(monkeypatch):
    table = build_flavor_band_table([[0.]], [1.], 1.)
    values = np.array([1., 0., 0., 0.])
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=.9, n_random_seeds=0), 1.)
    assert result.success
    np.testing.assert_allclose(result.nu_f, values)


def test_degenerate_flat_zero_interaction_minimum_is_retained(monkeypatch):
    table = build_flavor_band_table([[0., 0.]], [1.], 1.)
    values = np.full(4, .25)
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=0., n_random_seeds=0), 1.)
    assert result.success and result.energy_meV_per_cell == 0.


def test_a_free_flavor_face_can_be_unstable_while_other_flavors_are_at_kinks(monkeypatch):
    free = build_flavor_band_table([[0.]], [1.], 1.)
    kink = FlavorBandTable(np.array([0., .25, 1.]), np.array([-10., -10., 10.]),
                           np.array([0., -2.5, 5.]), 0., 1.)
    values = np.full(4, .25)
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [free, free, kink, kink],
        StonerParams(u0_meV_A2=.9, n_random_seeds=0), 1.)
    assert not result.success


def test_real_four_layer_hole_corner_obtains_a_valid_stoner_minimum(monkeypatch):
    import argparse
    from examples import run_full_band_stoner as workflow

    parser = argparse.ArgumentParser()
    workflow.add_hamiltonian_args(parser)
    workflow.add_full_scf_args(parser)
    workflow.add_stoner_args(parser)
    workflow.add_analysis_args(parser)
    args = parser.parse_args([
        "--theta-deg", "1.35", "--cutoff", "2", "--grid-n1", "6", "--grid-n2", "6",
        "--eps-perp", "4", "--full-max-iter", "100", "--full-tol-meV", "1e-4",
        "--stoner-random-seeds", "20",
    ])
    captured = []
    solve = workflow.solve_stoner_fixed_nu

    def record_solve(*a, **kw):
        result = solve(*a, **kw)
        captured.append(result)
        return result

    monkeypatch.setattr(workflow, "solve_stoner_fixed_nu", record_solve)
    row = workflow.run_point({"n_index": 0, "D_index": 0, "n_cm2": -6e12,
                              "D_Vnm": -1.2, "label": "hole_corner"}, args)
    assert row["scf_valid"]
    assert row["stoner_success"], row.get("error", row)
    assert row["stoner_dos_valid"]
    best = next(candidate for candidate in captured[0].local_minima
                if candidate["seed_name"] == captured[0].seed_name)
    assert best["success"] and best["stationarity"]["optimization_valid"]
    assert best["local_stability"]["valid"]
    assert best["local_stability"]["max_energy_decrease_meV"] == 0.
    for candidate in captured[0].local_minima:
        if "refinement_iterations" in candidate:
            assert candidate["total_iterations"] <= args.stoner_max_iter
            assert candidate["total_iterations"] == (
                candidate["optimizer_iterations"]+candidate["refinement_iterations"])


@pytest.mark.parametrize("remaining, expected_success", [(0, False), (1, True)])
def test_exact_kink_refinement_uses_only_remaining_optimizer_budget(monkeypatch, remaining, expected_success):
    table = FlavorBandTable(np.array([0., .25, 1.]), np.array([0., 0., 10.]),
                            np.array([0., 0., 7.5]), 0., 1.)
    near_kink = np.array([.24999, .25001, .25, .25])
    monkeypatch.setattr("scipy.optimize.minimize", lambda *a, **kw:
        SimpleNamespace(x=near_kink.copy(), success=True, message="near kink", nit=1))
    result = solve_stoner_fixed_nu(1., [table]*4,
        StonerParams(u0_meV_A2=.9, max_iter=1+remaining, n_random_seeds=0), 1.)
    assert result.success is expected_success
    raw = next(item for item in result.local_minima if item["seed_name"] == "paramagnetic:slsqp")
    assert not raw["success"] and not raw["stationarity"]["optimization_valid"]
    if expected_success:
        np.testing.assert_allclose(result.nu_f, np.full(4, .25), atol=1e-12)
        refined = next(item for item in result.local_minima if item["seed_name"] == result.seed_name)
        assert refined["optimizer_iterations"] == 1
        assert refined["refinement_iterations"] == 1
        assert refined["total_iterations"] <= 1+remaining
        assert refined["stationarity"]["optimization_valid"] and refined["local_stability"]["valid"]
