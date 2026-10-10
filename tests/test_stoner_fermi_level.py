"""A single Stoner Fermi level must reproduce all four occupations."""
import numpy as np
import pytest

from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import StonerResult


def result(values):
    return StonerResult(float(np.sum(values)), np.asarray(values, dtype=float), 0., True, "analytic", [])


def test_gapped_flavors_share_common_mu_with_partially_filled_flavor():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    table = build_neutrality_referenced_band_table(np.array([[-100., -50., 0., 10.]]), [1.], 1.)
    solved = result([1., .2, 0., 0.])
    tables = [table]*4
    level = stoner_mean_field_fermi_level(solved, tables, StonerParams(u0_meV_A2=9., JH_meV_A2=0.), 1.)
    assert level["valid"] is True
    assert level["mu_common_meV"] == pytest.approx(9.)
    np.testing.assert_allclose(level["self_energy_f_meV"], [1.8, 9., 10.8, 10.8])
    np.testing.assert_allclose(level["mu_f_meV"], [7.2, 0., -1.8, -1.8])
    np.testing.assert_allclose(np.asarray(level["mu_f_meV"])+level["self_energy_f_meV"], 9.)


def test_mean_field_self_energy_matches_independent_energy_derivative_with_hund():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    from tdbg_scf.stoner.energy import stoner_energy_meV_per_cell
    table = build_neutrality_referenced_band_table(np.array([[0., 0.]]), [1.], 1.)
    values = np.array([.2, .1, -.1, -.2])
    params = StonerParams(u0_meV_A2=3., JH_meV_A2=2.)
    level = stoner_mean_field_fermi_level(result(values), [table]*4, params, 1.)
    h = 1e-5
    gradient = []
    for f in range(4):
        delta = np.eye(4)[f]*h
        gradient.append((stoner_energy_meV_per_cell(values+delta, [table]*4, params, 1.)
                         - stoner_energy_meV_per_cell(values-delta, [table]*4, params, 1.))/(2*h))
    np.testing.assert_allclose(level["self_energy_f_meV"], gradient, atol=1e-10)


def test_arbitrary_flavor_fillings_are_not_a_physical_common_fermi_level():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    table = build_neutrality_referenced_band_table(np.array([[-1., 1.]]), [1.], 1.)
    solved = result([.2, 0., 0., 0.])
    tables = [table]*4
    level = stoner_mean_field_fermi_level(solved, tables, StonerParams(u0_meV_A2=100., JH_meV_A2=0.), 1.)
    assert level["valid"] is False
    assert level["interval_mismatch_meV"] > 0


@pytest.mark.parametrize("nu,expected", [(-.5,(-10.,-10.)),(0.,(-10.,10.)),(.5,(10.,10.))])
def test_discrete_kinetic_subgradient_is_single_state_or_gap(nu, expected):
    table = build_neutrality_referenced_band_table(np.array([[-10.,10.]]), [1.], 1.)
    assert table.mu_interval_of_nu(nu) == pytest.approx(expected)


def test_saved_stoner_dos_uses_shifted_flavors_instead_of_scf_curve(tmp_path):
    from examples import run_full_band_stoner as workflow
    from test_block_pipeline import small_args
    energies = np.array([[-100.,-50.,0.,10.]])
    centers = np.array([-1.,0.,1.])
    mu_f = np.array([7.2,0.,-1.8,-1.8])
    ham = workflow.TDBGContinuumHamiltonian(workflow.params_from_args(small_args(), 1))
    workflow.save_dos_figure(tmp_path, energies, energies, np.ones(1), 0., mu_f,
                             small_args(), ham, dos_data=(centers, np.full(3, 999.)))
    expected = np.zeros(3)
    for mu in mu_f:
        expected += np.sum(np.exp(-.5*(energies[...,None]-mu-centers)**2)/np.sqrt(2*np.pi), axis=(0,1))
    with np.load(tmp_path/"full_band_stoner_dos.npz", allow_pickle=False) as cache:
        np.testing.assert_allclose(cache["dos"], expected, atol=1e-12)
        assert cache["dos"][1] == pytest.approx(.564758048588, rel=1e-10)


def test_existing_stoner_reference_accepts_gapped_common_fermi_level():
    from tdbg_scf.susceptibility.stoner_reference import make_stoner_reference
    table = build_neutrality_referenced_band_table(np.array([[-100.,-50.,0.,10.]]), [1.], 1.)
    solved = result([1.,.2,0.,0.])
    tables = [table]*4
    reference = make_stoner_reference(solved,tables,StonerParams(u0_meV_A2=9.,JH_meV_A2=0.),1.)
    assert reference.reference_valid is True
    assert reference.mu_eq_meV == pytest.approx(9.)
    np.testing.assert_allclose(reference.nu_f_reconstructed_common_mu,solved.nu_f,atol=1e-6)
