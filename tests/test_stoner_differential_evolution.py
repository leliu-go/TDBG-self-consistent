"""Charge-exact global search and independently validated Stoner candidates."""
from types import SimpleNamespace

import numpy as np
import pytest

from tdbg_scf.stoner.dos import FlavorBandTable
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu
import tdbg_scf.stoner.solver as solver


def kink_table():
    return FlavorBandTable(np.array([0., .25, 1.]), np.array([0., 0., 10.]),
                           np.array([0., 0., 7.5]), 0., 1.)


def encode(values, bounds):
    """Independent inverse used to present known feasible candidates to DE."""
    remaining = float(np.sum(values))
    coordinates = []
    for i in range(3):
        lo = max(bounds[i][0], remaining - sum(b[1] for b in bounds[i+1:]))
        hi = min(bounds[i][1], remaining - sum(b[0] for b in bounds[i+1:]))
        coordinates.append((values[i]-lo)/(hi-lo) if hi > lo else 0.)
        remaining -= values[i]
    return np.array(coordinates)


def test_default_solver_runs_two_independent_three_dimensional_searches(monkeypatch):
    calls = []
    def de(objective, bounds, **kwargs):
        calls.append((bounds, kwargs))
        x = encode([.24999, .25001, .25, .25], [(0., 1.)]*4)
        return SimpleNamespace(x=x, fun=objective(x), success=False,
                               message="generation budget", nit=150, nfev=9600)
    monkeypatch.setattr("scipy.optimize.differential_evolution", de)
    table = kink_table()
    result = solve_stoner_fixed_nu(1., [table]*4, StonerParams(u0_meV_A2=.9), 1.)
    assert len(calls) == 2
    assert [c[1]["seed"] for c in calls] == [0, 1]
    for bounds, options in calls:
        assert bounds == [(0., 1.)]*3
        assert options["init"] == "sobol" and options["polish"] is False
        assert options["maxiter"] == 150 and options["popsize"] == 12
        assert options["workers"] == 1
    assert result.success
    np.testing.assert_allclose(result.nu_f, [.25]*4, atol=1e-12)
    check = result.minimum_validation
    assert check["method"] == "differential_evolution_3d_two_seeds"
    assert check["de_seeds"] == [0, 1]
    assert check["refinement_iterations"] == 1
    assert check["stationarity"]["physical_valid"]
    assert check["local_stability"]["valid"]
    assert not check["global_minimum_certified"]
    assert check["independent_runs_agree"]
    assert check["independent_run_energy_spread_meV"] == pytest.approx(0.)
    assert len(result.local_minima) == 4
    assert all(not c["de_converged"] for c in result.local_minima)
    from tdbg_scf.stoner.full_band import stoner_result_to_dict
    from tdbg_scf.stoner.io import result_to_dict
    for saved in (stoner_result_to_dict(result, [table]*4), result_to_dict(result)):
        assert {c["de_seed"] for c in saved["local_minima"]} == {0, 1}
        assert all(c["de_function_evaluations"] == 9600 for c in saved["local_minima"])


@pytest.mark.parametrize("total", [-8., -6., -4., 0., 4., 6., 8.])
def test_cube_mapping_covers_charge_plane_including_compensated_fillings(total):
    assert hasattr(solver, "_fillings_from_unit_cube"), "Need a feasible three-dimensional map"
    bounds = [(-2., 2.)]*4
    rng = np.random.default_rng(12)
    points = [np.array([x, y, z], float) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    points.extend(rng.random((60, 3)))
    for coordinates in points:
        values = solver._fillings_from_unit_cube(coordinates, bounds, total)
        assert values.sum() == pytest.approx(total, abs=2e-12)
        assert np.all(values >= -2.-1e-12) and np.all(values <= 2.+1e-12)
        np.testing.assert_allclose(solver._fillings_from_unit_cube(encode(values, bounds), bounds, total),
                                   values, atol=2e-12)
    if total == 0.:
        np.testing.assert_allclose(solver._fillings_from_unit_cube([1., 1., 0.], bounds, total),
                                   [2., 2., -2., -2.], atol=1e-12)


def test_cube_mapping_handles_asymmetric_band_capacities():
    assert hasattr(solver, "_fillings_from_unit_cube")
    bounds = [(-.3, .5), (-1., .8), (0., .1), (-.2, 1.)]
    target = np.array([-.1, -.5, .06, .9])
    np.testing.assert_allclose(solver._fillings_from_unit_cube(encode(target, bounds), bounds, target.sum()),
                               target, atol=1e-12)


@pytest.mark.parametrize("bad", [[np.nan, .5, .5], [-.1, .5, .5], [.5]*4])
def test_invalid_de_coordinates_never_become_an_accepted_candidate(monkeypatch, bad):
    monkeypatch.setattr("scipy.optimize.differential_evolution", lambda *a, **kw:
        SimpleNamespace(x=np.array(bad), success=True, message="fake success", nit=1, nfev=1))
    result = solve_stoner_fixed_nu(1., [kink_table()]*4, StonerParams(u0_meV_A2=.9), 1.)
    assert not result.success
    assert np.isfinite(result.energy_meV_per_cell)  # a diagnostic filling only


def test_python_cli_forwards_new_de_budget_and_base_seed():
    import argparse
    from examples.run_full_band_stoner import add_stoner_args, stoner_params_from_args
    parser = argparse.ArgumentParser()
    add_stoner_args(parser)
    args = parser.parse_args(["--stoner-de-max-iter", "17", "--stoner-de-popsize", "7",
                             "--stoner-de-tol", "1e-7", "--stoner-seed", "23"])
    params = stoner_params_from_args(args)
    assert (params.de_max_iter, params.de_popsize, params.de_tol, params.seed) == (17, 7, 1e-7, 23)


@pytest.mark.parametrize("seed,expected", [(23, [23, 24]), (2**32-1, [2**32-1, 0])])
def test_custom_de_controls_reach_both_searches(monkeypatch, seed, expected):
    observed = []
    def de(*args, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(x=encode([.25]*4, [(0., 1.)]*4), success=True,
                               message="success", nit=1, nfev=64)
    monkeypatch.setattr("scipy.optimize.differential_evolution", de)
    params = StonerParams(u0_meV_A2=.9, de_max_iter=17, de_popsize=7, de_tol=1e-7, seed=seed)
    solved = solve_stoner_fixed_nu(1., [kink_table()]*4, params, 1.)
    assert solved.success
    assert [c["seed"] for c in observed] == expected
    assert all((c["maxiter"], c["popsize"], c["tol"]) == (17, 7, 1e-7) for c in observed)


@pytest.mark.parametrize("changes", [{"de_max_iter": np.int64(1), "de_popsize": np.int64(1)},
                                    {"de_tol": np.float32(1e-8)}])
def test_accepted_numpy_parameter_scalars_can_be_saved_as_json(changes):
    import json
    from tdbg_scf.stoner.io import result_to_dict
    from tdbg_scf.stoner.full_band import stoner_result_to_dict
    params = StonerParams(u0_meV_A2=.9, **changes)
    table = kink_table()
    solved = solve_stoner_fixed_nu(1., [table]*4, params, 1.)
    assert solved.success
    for saved in (result_to_dict(solved), stoner_result_to_dict(solved, [table]*4)):
        assert json.loads(json.dumps(saved))["minimum_validation"]["de_seeds"] == [0, 1]


def test_a_lower_unvalidated_final_candidate_is_not_hidden_by_a_higher_minimum(monkeypatch):
    from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table
    from tdbg_scf.stoner.energy import stoner_energy_meV_per_cell
    table = build_neutrality_referenced_band_table(np.array([[-100., -50., 0., 10.]]), [1.], 1.)
    bad, good = np.array([1.001, .199, 0., 0.]), np.array([1.2, 0., 0., 0.])
    outputs = iter([bad, good])
    def de(*args, **kwargs):
        return SimpleNamespace(x=encode(next(outputs), [(-2., 2.)]*4), success=True,
                               message="success", nit=1, nfev=64)
    monkeypatch.setattr("scipy.optimize.differential_evolution", de)
    assert hasattr(StonerParams(u0_meV_A2=9.), "de_max_iter")
    params = StonerParams(u0_meV_A2=9., max_iter=0)
    result = solve_stoner_fixed_nu(1.2, [table]*4, params, 1.)
    assert not result.success
    np.testing.assert_allclose(result.nu_f, bad, atol=1e-12)
    assert result.energy_meV_per_cell == pytest.approx(stoner_energy_meV_per_cell(bad, [table]*4, params, 1.))
    assert result.minimum_validation["reason"] == "lower_energy_candidate_unvalidated"


@pytest.mark.parametrize("kwargs", [{"de_max_iter": 0}, {"de_popsize": 0}, {"de_tol": np.nan},
                                     {"de_tol": -1.}, {"max_iter": -1}, {"seed": -1}])
def test_de_parameters_reject_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        StonerParams(u0_meV_A2=0., **kwargs)
