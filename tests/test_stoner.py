import numpy as np


def _flat_table():
    from tdbg_scf.stoner.dos import build_flavor_band_table

    energies = np.array([[0.0], [1.0], [2.0], [3.0]])
    weights = np.full(4, 0.25)
    return build_flavor_band_table(energies, weights, A_M_A2=1.0)


def test_flavor_band_table_interpolates_capacity_and_kinetic_energy():
    table = _flat_table()

    assert table.nu_min == 0.0
    assert table.nu_max == 1.0
    assert table.mu_of_nu(0.5) == 1.0
    assert table.kinetic_of_nu(1.0) == 1.5


def test_large_stoner_repulsion_prefers_single_flavor_at_nu_one():
    from tdbg_scf.stoner.params import StonerParams
    from tdbg_scf.stoner.solver import solve_stoner_fixed_nu

    tables = [_flat_table() for _ in range(4)]
    result = solve_stoner_fixed_nu(
        1.0,
        tables,
        StonerParams(u0_meV_A2=100.0, JH_meV_A2=0.0, n_random_seeds=4, seed=1),
        A_M_A2=1.0,
    )

    assert result.success
    assert np.isclose(np.sum(result.nu_f), 1.0)
    assert np.max(result.nu_f) > 0.99
    assert result.flavor_polarization > 0.99


def test_hund_coupling_aligns_spin_polarization_across_valleys():
    from tdbg_scf.stoner.params import StonerParams
    from tdbg_scf.stoner.solver import solve_stoner_fixed_nu

    tables = [_flat_table() for _ in range(4)]
    result = solve_stoner_fixed_nu(
        2.0,
        tables,
        StonerParams(u0_meV_A2=0.0, JH_meV_A2=100.0, n_random_seeds=4, seed=2),
        A_M_A2=1.0,
    )

    assert result.success
    assert np.isclose(np.sum(result.nu_f), 2.0)
    assert abs(result.spin_polarization) > 1.9
    assert abs(result.valley_polarization) < 1e-6


def test_stoner_result_reports_normalized_flavor_order_parameters():
    from tdbg_scf.stoner.solver import StonerResult

    result = StonerResult(
        nu_total=2.0,
        nu_f=np.array([1.0, 0.25, 0.5, 0.25]),
        energy_meV_per_cell=0.0,
        success=True,
        seed_name="test",
        local_minima=[],
    )

    assert result.spin_polarization == 0.5
    assert result.valley_polarization == 1.0
    assert result.spin_valley_polarization == 0.5
    assert result.spin_polarization_norm == 0.25
    assert result.valley_polarization_norm == 0.5
    assert result.spin_valley_polarization_norm == 0.25


def test_stoner_result_normalized_order_parameters_use_signed_total_filling():
    from tdbg_scf.stoner.solver import StonerResult

    result = StonerResult(
        nu_total=-2.0,
        nu_f=np.array([-1.0, -0.25, -0.5, -0.25]),
        energy_meV_per_cell=0.0,
        success=True,
        seed_name="test",
        local_minima=[],
    )

    assert result.spin_polarization == 0.5
    assert result.valley_polarization == 1.0
    assert result.spin_valley_polarization == 0.5
    assert result.spin_polarization_norm == 0.25
    assert result.valley_polarization_norm == 0.5
    assert result.spin_valley_polarization_norm == 0.25


def test_stoner_result_normalized_order_parameters_are_nan_at_neutrality():
    from tdbg_scf.stoner.solver import StonerResult

    result = StonerResult(
        nu_total=0.0,
        nu_f=np.zeros(4),
        energy_meV_per_cell=0.0,
        success=True,
        seed_name="test",
        local_minima=[],
    )

    assert np.isnan(result.spin_polarization_norm)
    assert np.isnan(result.valley_polarization_norm)
    assert np.isnan(result.spin_valley_polarization_norm)
