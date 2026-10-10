"""Independent counterexamples for four-flavor optimization and occupations."""
import numpy as np
import pytest
from types import SimpleNamespace

from tdbg_scf.stoner.dos import FlavorBandTable, build_flavor_band_table
from tdbg_scf.stoner.full_band import (
    build_neutrality_referenced_band_table, build_full_band_flavor_tables,
    explicit_flavor_dos_at_mu, layer_filling_from_nu,
    stoner_mu_by_flavor,
)
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu


def quadratic_tables():
    nu = np.linspace(0., 1., 1001)
    return [FlavorBandTable(nu, c*nu, .5*c*nu**2, 0., 1.) for c in (1., 4., 1., 4.)]


def test_budget_limit_does_not_validate_a_nonstationary_candidate(monkeypatch):
    monkeypatch.setattr('scipy.optimize.differential_evolution', lambda *a, **k:
        SimpleNamespace(x=np.array([.25, 1./3., .5]), success=False,
                        message='generation limit', nit=1, nfev=64))
    tables = quadratic_tables()
    result = solve_stoner_fixed_nu(1., tables,
        StonerParams(u0_meV_A2=0., max_iter=0, de_max_iter=1), 1.)
    # The exact constrained minimum is (.4,.1,.4,.1), E=.2.
    # A candidate with no refinement budget must still pass stationarity.
    assert result.energy_meV_per_cell > .2 + 1e-3
    assert not result.success
    assert all(not r['success'] for r in result.local_minima)


def test_accepted_optimizer_result_solves_an_independent_quadratic_problem():
    result = solve_stoner_fixed_nu(1., quadratic_tables(),
        StonerParams(u0_meV_A2=0.), 1.)
    assert result.success
    np.testing.assert_allclose(result.nu_f, [.4,.1,.4,.1], atol=2e-3, rtol=0)
    assert result.energy_meV_per_cell == pytest.approx(.2, abs=2e-6)


@pytest.mark.parametrize('bad', [np.array([.1]*4), np.array([np.nan,.2,.3,.5]), np.array([-1.,1.,.5,.5])])
def test_optimizer_success_flag_does_not_accept_an_invalid_raw_candidate(monkeypatch, bad):
    monkeypatch.setattr('scipy.optimize.differential_evolution', lambda *a, **k:
        SimpleNamespace(x=bad.copy(), success=True, message='fake success'))
    result = solve_stoner_fixed_nu(1., quadratic_tables(),
        StonerParams(u0_meV_A2=0.), 1.)
    assert not result.success
    assert np.isfinite(result.energy_meV_per_cell)


@pytest.mark.parametrize('nu, expected', [(-.5,-10.), (0.,0.), (.5,10.)])
def test_signed_table_mu_matches_the_kinetic_slope_or_gap_interval(nu, expected):
    table = build_neutrality_referenced_band_table(np.array([[-10.,10.]]), np.ones(1), 1.)
    assert table.mu_of_nu(nu) == pytest.approx(expected)
    if nu != 0:
        eps = 1e-6
        derivative = (table.kinetic_of_nu(nu+eps)-table.kinetic_of_nu(nu-eps))/(2*eps)
        assert table.mu_of_nu(nu) == pytest.approx(derivative)


def test_search_bounds_do_not_change_the_neutral_gap_mu_interval():
    from tdbg_scf.stoner.dos import flavor_filling_bounds
    table = build_neutrality_referenced_band_table(np.array([[-10.,10.]]), np.ones(1), 1.)
    assert flavor_filling_bounds([table]*4, -.5,.5) == [(-.5,.5)]*4
    assert table.mu_of_nu(0.) == pytest.approx(0.)


def test_partial_state_dos_uses_its_energy_instead_of_a_spurious_gap_mu():
    energies = np.array([[-10.,10.]])
    tables = build_full_band_flavor_tables(energies, energies, np.ones(1), 1.)
    result = SimpleNamespace(nu_f=np.full(4,.5))
    mu = stoner_mu_by_flavor(result, tables)
    np.testing.assert_allclose(mu, np.full(4,10.), atol=1e-12, rtol=0)
    assert explicit_flavor_dos_at_mu(energies, energies, np.ones(1), mu, 1.) == pytest.approx(4/np.sqrt(2*np.pi))


@pytest.mark.parametrize('nu', [-.5,0.,.5])
def test_degenerate_layer_occupations_are_invariant_under_unitary_basis_rotation(nu):
    energy = np.zeros((1,2))
    original = np.eye(2,dtype=complex)[None]
    rotated = np.array([[1.,1.],[-1.,1.]], dtype=complex)[None]/np.sqrt(2)
    left = layer_filling_from_nu(energy, original, np.ones(1), np.eye(2), 1., nu)
    right = layer_filling_from_nu(energy, rotated, np.ones(1), np.eye(2), 1., nu)
    np.testing.assert_allclose(left, [nu/2,nu/2], atol=1e-12, rtol=0)
    np.testing.assert_allclose(left, right, atol=1e-12, rtol=0)


def test_zero_weight_states_do_not_change_the_band_table_or_mu():
    energies = np.array([[100.,200.],[-10.,10.]])
    table = build_neutrality_referenced_band_table(energies, np.array([0.,1.]), 1.)
    assert np.all(np.diff(table.nu_grid) > 0.)
    assert table.mu_of_nu(.5) == pytest.approx(10.)


def test_large_uniform_mesh_preserves_neutral_gap_mu_despite_cumulative_roundoff():
    levels = np.r_[np.arange(-125.,0.), np.arange(1.,126.)]
    energies = np.tile(levels, (2304,1))
    weights = np.full(2304, 1./2304)
    table = build_neutrality_referenced_band_table(energies, weights, 1.)
    assert table.mu_of_nu(0.) == pytest.approx(0., abs=1e-8)
    assert table.nu_max == pytest.approx(125., abs=1e-11)


@pytest.mark.parametrize('builder', [build_flavor_band_table, build_neutrality_referenced_band_table])
@pytest.mark.parametrize('weights,area', [([0.],1.),([np.nan],1.),([1.],np.nan),([1.],-1.)])
def test_invalid_table_measure_is_rejected(builder, weights, area):
    with pytest.raises(ValueError):
        builder(np.array([[-10.,10.]]), np.asarray(weights), area)


@pytest.mark.parametrize('changes', [dict(u0_meV_A2=np.nan),dict(JH_meV_A2=np.inf),dict(temperature_K=1.)])
def test_invalid_or_unimplemented_stoner_parameters_are_rejected(changes):
    values = dict(u0_meV_A2=0.)
    values.update(changes)
    with pytest.raises(ValueError):
        solve_stoner_fixed_nu(1., quadratic_tables(), StonerParams(**values), 1.)


def test_nonfinite_cell_area_cannot_produce_successful_stoner_energy():
    with pytest.raises(ValueError):
        solve_stoner_fixed_nu(1., quadratic_tables(), StonerParams(u0_meV_A2=0.), np.nan)
