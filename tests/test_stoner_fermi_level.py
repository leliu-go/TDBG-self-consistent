"""Analytic checks of optimization stationarity and physical Fermi levels."""
from types import SimpleNamespace

import numpy as np
import pytest

from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table, restrict_tables_to_carrier_sector
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import StonerResult, solve_stoner_fixed_nu


def _result(values):
    values = np.asarray(values, dtype=float)
    return StonerResult(float(values.sum()), values, 0., True, "analytic", [])


def _table(energies):
    return build_neutrality_referenced_band_table(np.asarray([energies], dtype=float), [1.], 1.)


def test_candidate_selection_rejects_successful_nonstationary_lower_energy(monkeypatch):
    table = _table([-100., -50., 0., 10.])
    tables = restrict_tables_to_carrier_sector([table]*4, 1.2)
    params = StonerParams(u0_meV_A2=9., n_random_seeds=0)
    bad, good = np.array([1.001, .199, 0., 0.]), np.array([1.2, 0., 0., 0.])
    outputs = iter([bad])
    monkeypatch.setattr("scipy.optimize.minimize", lambda *a, **kw:
        SimpleNamespace(x=next(outputs, good).copy(), success=True, message="success"))
    result = solve_stoner_fixed_nu(1.2, tables, params, 1.)
    assert result.success
    np.testing.assert_allclose(result.nu_f, good, atol=1e-12)
    assert result.energy_meV_per_cell == pytest.approx(2.)
    assert all(not item["success"] for item in result.local_minima
               if item["seed_name"].endswith(":slsqp") and np.allclose(item["nu_f"], bad))


def test_gapped_flavors_share_the_partially_filled_flavors_fermi_level():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    table = _table([-100., -50., 0., 10.])
    result = _result([1., .2, 0., 0.])
    tables = restrict_tables_to_carrier_sector([table]*4, result.nu_total)
    level = stoner_mean_field_fermi_level(result, tables, StonerParams(u0_meV_A2=9.), 1.)
    assert level["valid"] and level["optimization_valid"]
    assert level["mu_common_meV"] == pytest.approx(9.)
    np.testing.assert_allclose(level["self_energy_f_meV"], [1.8, 9., 10.8, 10.8])
    np.testing.assert_allclose(level["mu_f_meV"], [7.2, 0., -1.8, -1.8])


def test_common_mu_and_flavor_shifts_obey_particle_hole_reflection():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    values = np.array([1., .2, 0., 0.])
    levels = []
    for table, filling in ((_table([-100., -50., 0., 10.]), values),
                           (_table([-10., 0., 50., 100.]), -values)):
        tables = restrict_tables_to_carrier_sector([table]*4, float(filling.sum()))
        levels.append(stoner_mean_field_fermi_level(_result(filling), tables, StonerParams(u0_meV_A2=9.), 1.))
    assert all(level["valid"] for level in levels)
    assert levels[1]["mu_common_meV"] == pytest.approx(-levels[0]["mu_common_meV"])
    np.testing.assert_allclose(levels[1]["mu_f_meV"], -np.asarray(levels[0]["mu_f_meV"]))


def test_mean_field_self_energy_matches_the_interaction_energy_derivative():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    from tdbg_scf.stoner.energy import stoner_energy_meV_per_cell
    table = _table([0., 0.])
    values = np.array([.2, .1, -.1, -.2])
    params = StonerParams(u0_meV_A2=3., JH_meV_A2=2.)
    level = stoner_mean_field_fermi_level(_result(values), [table]*4, params, 1.)
    h = 1e-5
    derivative = [(stoner_energy_meV_per_cell(values+np.eye(4)[f]*h, [table]*4, params, 1.)
                  - stoner_energy_meV_per_cell(values-np.eye(4)[f]*h, [table]*4, params, 1.))/(2*h)
                  for f in range(4)]
    np.testing.assert_allclose(level["self_energy_f_meV"], derivative, atol=1e-10)


def test_carrier_constrained_kkt_cannot_manufacture_a_physical_common_mu():
    from tdbg_scf.stoner.full_band import stoner_mean_field_fermi_level
    table = _table([-1., 1.])
    tables = restrict_tables_to_carrier_sector([table]*4, .2)
    level = stoner_mean_field_fermi_level(_result([.2, 0., 0., 0.]), tables, StonerParams(u0_meV_A2=100.), 1.)
    assert level["optimization_valid"] and not level["valid"]
    assert level["reason"] == "carrier_constraint_incompatible"
    assert np.isnan(level["mu_common_meV"]) and np.isnan(level["mu_f_meV"]).all()


def test_finite_q_reference_accepts_a_common_mu_inside_flavor_gaps():
    from tdbg_scf.susceptibility.stoner_reference import make_stoner_reference
    table = _table([-100., -50., 0., 10.])
    result = _result([1., .2, 0., 0.])
    tables = restrict_tables_to_carrier_sector([table]*4, result.nu_total)
    ref = make_stoner_reference(result, tables, StonerParams(u0_meV_A2=9.), 1.)
    assert ref.reference_valid
    assert ref.mu_eq_meV == pytest.approx(9.)
    np.testing.assert_allclose(ref.nu_f_reconstructed_common_mu, result.nu_f, atol=1e-6)


@pytest.mark.parametrize("width", [0., -1., np.nan, np.inf])
def test_layer_dos_rejects_an_invalid_gaussian_width(width):
    from tdbg_scf.stoner.full_band import layer_dos_at_mu_gaussian
    with pytest.raises(ValueError, match="sigma_meV"):
        layer_dos_at_mu_gaussian(np.array([[-1., 1.]]), np.eye(2)[None], [1.], np.eye(2), 0., width)
