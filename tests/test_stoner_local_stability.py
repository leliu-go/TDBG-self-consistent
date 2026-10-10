"""First-order Stoner stationarity does not certify a local minimum."""
from types import SimpleNamespace

import numpy as np
import pytest

from tdbg_scf.stoner.dos import FlavorBandTable, build_flavor_band_table
from tdbg_scf.stoner.energy import stoner_energy_meV_per_cell
from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu


def _constant_optimizer(monkeypatch, values, bounds=None):
    bounds = bounds or [(0., 1.)]*4
    remaining, raw = float(np.sum(values)), []
    for i in range(3):
        lo = max(bounds[i][0], remaining-sum(b[1] for b in bounds[i+1:]))
        hi = min(bounds[i][1], remaining-sum(b[0] for b in bounds[i+1:]))
        raw.append((values[i]-lo)/(hi-lo) if hi > lo else 0.)
        remaining -= values[i]
    monkeypatch.setattr("scipy.optimize.differential_evolution", lambda *a, **kw:
        SimpleNamespace(x=np.array(raw), success=True, message="stationary", nit=1, nfev=64))


def test_stationary_quartic_band_saddle_is_rejected(monkeypatch):
    nodes = np.linspace(0., 1., 1001)
    conduction = np.diff(nodes**4)/np.diff(nodes)
    energies = np.r_[np.full(1000, -10.), conduction][None]
    table = build_neutrality_referenced_band_table(energies, [.001], 1.)
    tables = [table]*4
    params = StonerParams(u0_meV_A2=.9, max_iter=0)
    _constant_optimizer(monkeypatch, np.full(4, .2505), [(-1., 1.)]*4)
    result = solve_stoner_fixed_nu(1.002, tables, params, 1.)
    candidate = next(item for item in result.local_minima if item["seed_name"] == "de_seed_0:refined")
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
    params = StonerParams(u0_meV_A2=-1., JH_meV_A2=.75, max_iter=0)
    baseline = stoner_energy_meV_per_cell(values, tables, params, 1.)
    for receiver in range(4):
        for donor in range(receiver):
            pair = values+.1*(np.eye(4)[receiver]-np.eye(4)[donor])
            assert stoner_energy_meV_per_cell(pair, tables, params, 1.) > baseline
    coupled = values+np.array([.1, .1, -.1, -.1])
    assert stoner_energy_meV_per_cell(coupled, tables, params, 1.) < baseline-.009
    _constant_optimizer(monkeypatch, values, [(0., 2.)]*4)
    result = solve_stoner_fixed_nu(1., tables, params, 1.)
    assert not result.success
    assert all(not item["success"] for item in result.local_minima)


def test_true_kinetic_kink_minimum_survives_negative_interaction_curvature(monkeypatch):
    table = FlavorBandTable(np.array([0., .25, 1.]), np.array([0., 0., 10.]),
                            np.array([0., 0., 7.5]), 0., 1.)
    values = np.full(4, .25)
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=.9), 1.)
    assert result.success
    np.testing.assert_allclose(result.nu_f, values)


def test_carrier_boundary_minimum_survives_negative_interaction_curvature(monkeypatch):
    table = build_flavor_band_table([[0.]], [1.], 1.)
    values = np.array([1., 0., 0., 0.])
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=.9), 1.)
    assert result.success
    np.testing.assert_allclose(result.nu_f, values)


def test_degenerate_flat_zero_interaction_minimum_is_retained(monkeypatch):
    table = build_flavor_band_table([[0., 0.]], [1.], 1.)
    values = np.full(4, .25)
    _constant_optimizer(monkeypatch, values, [(0., 2.)]*4)
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=0.), 1.)
    assert result.success and result.energy_meV_per_cell == 0.


def test_a_free_flavor_face_can_be_unstable_while_other_flavors_are_at_kinks(monkeypatch):
    free = build_flavor_band_table([[0.]], [1.], 1.)
    kink = FlavorBandTable(np.array([0., .25, 1.]), np.array([-10., -10., 10.]),
                           np.array([0., -2.5, 5.]), 0., 1.)
    values = np.full(4, .25)
    _constant_optimizer(monkeypatch, values)
    result = solve_stoner_fixed_nu(1., [free, free, kink, kink],
        StonerParams(u0_meV_A2=.9, max_iter=0), 1.)
    assert not result.success


def test_resolved_saddle_descent_near_a_segment_end_is_detected(monkeypatch):
    """An 80% probe misses a >tol decrease inside the same kinetic piece."""
    from tdbg_scf.stoner.energy import stoner_local_stability, stoner_stationarity_intervals
    free = FlavorBandTable(np.array([0., .5]), np.zeros(2), np.zeros(2), 0., .5)
    kink = FlavorBandTable(np.array([0., .25, .5]), np.array([-10., -10., 10.]),
                           np.array([2.5, 0., 2.5]), 0., .5)
    tables, values = [free, free, kink, kink], np.full(4, .25)
    params = StonerParams(u0_meV_A2=2e-7, max_iter=0, tol_energy_meV=1e-8)
    baseline = stoner_energy_meV_per_cell(values, tables, params, 1.)
    probe = np.array([.4975, .0025, .25, .25])
    assert baseline-stoner_energy_meV_per_cell(probe, tables, params, 1.) == pytest.approx(1.225125e-8)
    assert stoner_stationarity_intervals(values, tables, params, 1.)['physical_valid']
    check = stoner_local_stability(values, tables, params, 1.)
    assert not check['valid']
    assert check['max_energy_decrease_meV'] > params.tol_energy_meV
    _constant_optimizer(monkeypatch, values, [(0., .5)]*4)
    assert not solve_stoner_fixed_nu(1., tables, params, 1.).success


@pytest.mark.parametrize("remaining, expected_success", [(0, False), (1, True)])
def test_exact_kink_refinement_respects_its_independent_budget(monkeypatch, remaining, expected_success):
    table = FlavorBandTable(np.array([0., .25, 1.]), np.array([0., 0., 10.]),
                            np.array([0., 0., 7.5]), 0., 1.)
    near_kink = np.array([.24999, .25001, .25, .25])
    _constant_optimizer(monkeypatch, near_kink)
    result = solve_stoner_fixed_nu(1., [table]*4,
        StonerParams(u0_meV_A2=.9, max_iter=remaining), 1.)
    assert result.success is expected_success
    raw = next(item for item in result.local_minima if item["seed_name"] == "de_seed_0:raw")
    assert not raw["success"] and not raw["stationarity"]["optimization_valid"]
    if expected_success:
        np.testing.assert_allclose(result.nu_f, np.full(4, .25), atol=1e-12)
        refined = next(item for item in result.local_minima if item["seed_name"] == result.seed_name)
        assert refined["optimizer_iterations"] == 1
        assert refined["refinement_iterations"] == 1
        assert refined["total_iterations"] <= 1+remaining
        assert refined["stationarity"]["optimization_valid"] and refined["local_stability"]["valid"]


def test_saved_results_preserve_the_selected_validation_and_rejected_attempt(monkeypatch):
    from tdbg_scf.stoner.io import result_to_dict
    from tdbg_scf.stoner.full_band import stoner_result_to_dict

    table = FlavorBandTable(np.array([0., .25, 1.]), np.array([0., 0., 10.]),
                            np.array([0., 0., 7.5]), 0., 1.)
    values = np.array([.24999, .25001, .25, .25])
    _constant_optimizer(monkeypatch, values)
    solved = solve_stoner_fixed_nu(1., [table]*4,
        StonerParams(u0_meV_A2=.9, max_iter=2), 1.)
    for saved in (result_to_dict(solved), stoner_result_to_dict(solved, [table]*4)):
        check = saved["minimum_validation"]
        assert check["stationarity"]["optimization_valid"]
        assert check["local_stability"]["valid"]
        assert check["refinement_iterations"] == 1
        assert check["total_iterations"] == 2
    saved = stoner_result_to_dict(solved, [table]*4)
    raw = next(item for item in saved["local_minima"] if item["seed_name"] == "de_seed_0:raw")
    assert not raw["success"]
    assert not raw["stationarity"]["optimization_valid"]
    assert raw["message"] == "stationary"
