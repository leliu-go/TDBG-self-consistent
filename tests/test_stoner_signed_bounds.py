"""Signed flavor search, charge conservation and genuine spectral equilibrium."""
import argparse

import numpy as np
import pytest

from tdbg_scf.stoner.full_band import (
    build_neutrality_referenced_band_table, stoner_mean_field_fermi_level,
)
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import StonerResult, make_stoner_seeds, solve_stoner_fixed_nu


def table(levels):
    return build_neutrality_referenced_band_table(np.array([levels], float), np.ones(1), 1.)


def candidate(values):
    values = np.array(values, float)
    return StonerResult(float(values.sum()), values, 0., True, "analytic", [])


def test_default_flavor_window_is_minus_two_to_two():
    params = StonerParams(u0_meV_A2=0.)
    assert params.flavor_nu_min == -2.
    assert params.flavor_nu_max == 2.


@pytest.mark.parametrize("lo,hi", [(np.nan,2.),(-2.,np.inf),(2.,-2.),(1.,1.)])
def test_flavor_window_must_be_finite_and_ordered(lo, hi):
    with pytest.raises(ValueError, match="[Bb]ound|[Ff]inite|flavor"):
        StonerParams(u0_meV_A2=0., flavor_nu_min=lo, flavor_nu_max=hi)


def test_zero_total_has_compensated_seeds_without_random_starts():
    starts = make_stoner_seeds(0., [(-2.,2.)]*4, np.random.default_rng(0), 0)
    assert any(np.min(x) < 0 < np.max(x) for _, x in starts)
    for _, x in starts:
        assert x.sum() == pytest.approx(0., abs=1e-12)
        assert np.max(np.abs(x)) <= 2.


def test_neutral_solver_finds_analytic_compensated_spin_state():
    # Inside |nu_f|<=1 K=|nu_f|, with a prohibitive remote-band cost beyond
    # one. The global minimum has two +1, two -1, parallel spin in valleys:
    # K=4, E_U=-20, E_Hund=-8, hence E=-24 meV/cell.
    bands = [table([-100.,-1.,1.,100.])]*4
    params = StonerParams(u0_meV_A2=10., JH_meV_A2=2., n_random_seeds=0)
    solved = solve_stoner_fixed_nu(0., bands, params, 1.)
    assert solved.success
    assert solved.energy_meV_per_cell == pytest.approx(-24., abs=1e-5)
    np.testing.assert_allclose(np.abs(solved.nu_f), 1., atol=1e-6)
    assert solved.nu_f.sum() == pytest.approx(0., abs=1e-10)
    level = stoner_mean_field_fermi_level(solved, bands, params, 1.)
    assert level["valid"] and not level["filling_bound_hit"]


def test_bound_clipped_minimum_has_kkt_but_no_physical_common_mu():
    bands = [table([0.]*6)]*4  # true capacity +/-3, user window +/-2
    level = stoner_mean_field_fermi_level(candidate([2.,2.,-2.,-2.]), bands,
        StonerParams(u0_meV_A2=10., JH_meV_A2=2.), 1.)
    assert level["optimization_valid"]
    assert level["filling_bound_hit"] and level["artificial_bound_hit"]
    assert level["filling_bound_limited"]
    assert not level["valid"]
    assert level["reason"] == "filling_bound_limited"
    assert np.isnan(level["mu_common_meV"])


def test_touching_user_bound_with_real_spectral_gap_keeps_physical_dos():
    bands = [table([-100.,-50.,-1.,1.,50.,100.])]*4
    level = stoner_mean_field_fermi_level(candidate([2.,2.,-2.,-2.]), bands,
        StonerParams(u0_meV_A2=40.), 1.)
    assert level["filling_bound_hit"] and level["artificial_bound_hit"]
    assert not level["filling_bound_limited"]
    assert level["valid"] and np.isfinite(level["mu_common_meV"])


def test_real_band_capacity_is_not_an_artificial_bound():
    bands = [table([0.]*4)]*4
    level = stoner_mean_field_fermi_level(candidate([2.,2.,-2.,-2.]), bands,
        StonerParams(u0_meV_A2=3.), 1.)
    assert level["valid"]
    assert not level["artificial_bound_hit"]
    assert level["flavor_bounds_nu"] == [[-2.,2.]]*4


def test_custom_window_intersects_capacity_without_changing_spectral_intervals():
    from tdbg_scf.stoner.dos import flavor_filling_bounds
    bands = [table([-10.,10.])]*4
    assert flavor_filling_bounds(bands, -2.,2.) == [(-1.,1.)]*4
    assert flavor_filling_bounds(bands, -.25,.75) == [(-.25,.75)]*4
    assert bands[0].mu_interval_of_nu(0.) == (-10.,10.)


def test_total_outside_four_flavor_window_is_rejected():
    with pytest.raises(ValueError, match="filling bounds"):
        solve_stoner_fixed_nu(8.1, [table([0.]*6)]*4,
            StonerParams(u0_meV_A2=0., n_random_seeds=0), 1.)


@pytest.mark.parametrize("total", [-6.,6.])
def test_original_total_scan_endpoints_are_feasible_and_bounded(total):
    solved = solve_stoner_fixed_nu(total, [table([0.]*6)]*4,
        StonerParams(u0_meV_A2=0., n_random_seeds=0), 1.)
    assert solved.success
    assert solved.nu_f.sum() == pytest.approx(total, abs=1e-10)
    assert np.max(np.abs(solved.nu_f)) <= 2. + 1e-10
    assert solved.flavor_bounds_nu == [(-2.,2.)]*4


def test_python_cli_forwards_custom_per_flavor_window():
    from examples.run_full_band_stoner import add_stoner_args, stoner_params_from_args
    parser = argparse.ArgumentParser()
    add_stoner_args(parser)
    args = parser.parse_args(["--stoner-flavor-nu-min=-1.5", "--stoner-flavor-nu-max=2.5"])
    params = stoner_params_from_args(args)
    assert (params.flavor_nu_min, params.flavor_nu_max) == (-1.5,2.5)


def test_live_map_preserves_bound_diagnostics_and_missing_cells():
    from tdbg_scf.live_map import scalar_grids
    rows = [{"n_index":0, "D_index":0, "status":"ok", "stoner_success":True,
             "stoner_dos_valid":False, "stoner_flavor_nu_min":-2.,
             "stoner_flavor_nu_max":2., "stoner_filling_bound_hit":True,
             "stoner_artificial_bound_hit":True, "stoner_filling_bound_limited":True}]
    grids = scalar_grids(rows, (2,1), 4)
    for key in ("stoner_filling_bound_hit", "stoner_artificial_bound_hit", "stoner_filling_bound_limited"):
        assert grids[key][0,0] == 1.
        assert np.isnan(grids[key][1,0])
    assert grids["stoner_flavor_nu_min"][0,0] == -2.


def test_both_serializers_save_effective_flavor_bounds():
    from tdbg_scf.stoner.io import result_to_dict
    from tdbg_scf.stoner.full_band import stoner_result_to_dict
    bands = [table([-10.,10.])]*4
    solved = solve_stoner_fixed_nu(0., bands, StonerParams(u0_meV_A2=0., n_random_seeds=0), 1.)
    assert result_to_dict(solved)["flavor_bounds_nu"] == [(-1.,1.)]*4
    assert stoner_result_to_dict(solved,bands)["flavor_bounds_nu"] == [(-1.,1.)]*4
