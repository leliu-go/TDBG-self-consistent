import numpy as np


def test_neutrality_referenced_table_supports_signed_fillings():
    from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table

    energies = np.array([[-2.0, 2.0], [-1.0, 3.0]])
    weights = np.array([0.5, 0.5])
    table = build_neutrality_referenced_band_table(energies, weights, A_M_A2=1.0)

    assert np.isclose(table.nu_min, -1.0)
    assert np.isclose(table.nu_max, 1.0)
    assert np.isclose(table.kinetic_of_nu(0.0), 0.0)
    assert table.mu_of_nu(-0.5) < table.mu_of_nu(0.5)


def test_full_band_flavor_tables_follow_explicit_flavor_order():
    from tdbg_scf.stoner.full_band import build_full_band_flavor_tables

    evals_K = np.array([[-2.0, 2.0], [-1.0, 3.0]])
    evals_Kp = evals_K + 10.0
    weights = np.array([0.5, 0.5])
    tables = build_full_band_flavor_tables(evals_K, evals_Kp, weights, A_M_A2=1.0)

    assert len(tables) == 4
    assert np.isclose(tables[0].mu_of_nu(0.0), tables[2].mu_of_nu(0.0))
    assert np.isclose(tables[1].mu_of_nu(0.0), tables[3].mu_of_nu(0.0))
    assert tables[1].mu_of_nu(0.0) > tables[0].mu_of_nu(0.0)


def test_signed_stoner_minimization_handles_hole_filling():
    from tdbg_scf.stoner.full_band import build_neutrality_referenced_band_table
    from tdbg_scf.stoner.params import StonerParams
    from tdbg_scf.stoner.solver import solve_stoner_fixed_nu

    energies = np.array([[-2.0, 2.0], [-1.0, 3.0]])
    weights = np.array([0.5, 0.5])
    table = build_neutrality_referenced_band_table(energies, weights, A_M_A2=1.0)
    result = solve_stoner_fixed_nu(
        -0.5,
        [table, table, table, table],
        StonerParams(u0_meV_A2=100.0, JH_meV_A2=0.0, n_random_seeds=4, seed=0),
        A_M_A2=1.0,
    )

    assert result.success
    assert np.isclose(np.sum(result.nu_f), -0.5)
    assert np.min(result.nu_f) < -0.49


def test_carrier_sector_restriction_forbids_compensated_electron_hole_pairs():
    from tdbg_scf.stoner.full_band import (
        build_neutrality_referenced_band_table,
        restrict_tables_to_carrier_sector,
    )
    from tdbg_scf.stoner.params import StonerParams
    from tdbg_scf.stoner.solver import solve_stoner_fixed_nu

    energies = np.array([[-2.0, 2.0], [-1.0, 3.0]])
    weights = np.array([0.5, 0.5])
    table = build_neutrality_referenced_band_table(energies, weights, A_M_A2=1.0)
    tables = restrict_tables_to_carrier_sector([table, table, table, table], nu_total=0.5)
    result = solve_stoner_fixed_nu(
        0.5,
        tables,
        StonerParams(u0_meV_A2=100.0, JH_meV_A2=0.0, n_random_seeds=6, seed=0),
        A_M_A2=1.0,
    )

    assert result.success
    assert np.isclose(np.sum(result.nu_f), 0.5)
    assert np.all(result.nu_f >= -1e-10)
    assert np.max(result.nu_f) <= 0.5 + 1e-10


def test_layer_filling_from_nu_uses_full_band_layer_weights():
    from tdbg_scf.stoner.full_band import layer_filling_from_nu

    evals = np.array([[0.0, 2.0]])
    evecs = np.tile(np.eye(2, dtype=np.complex128), (1, 1, 1))
    weights = np.array([1.0])
    layer_masks = np.array([[1.0, 0.0], [0.0, 1.0]])

    neutral = layer_filling_from_nu(evals, evecs, weights, layer_masks, A_M_A2=1.0, nu=0.0)
    electron = layer_filling_from_nu(evals, evecs, weights, layer_masks, A_M_A2=1.0, nu=1.0)
    hole = layer_filling_from_nu(evals, evecs, weights, layer_masks, A_M_A2=1.0, nu=-1.0)

    assert np.allclose(neutral, [0.5, -0.5])
    assert np.allclose(electron, [0.5, 0.5])
    assert np.allclose(hole, [-0.5, -0.5])
    assert np.isclose(np.sum(neutral), 0.0)
    assert np.isclose(np.sum(electron), 1.0)
    assert np.isclose(np.sum(hole), -1.0)


def test_stoner_layer_dos_sums_to_total_flavor_dos():
    from tdbg_scf.density import dos_at_mu_gaussian
    from tdbg_scf.stoner.full_band import stoner_layer_dos_at_mu

    evals_K = np.array([[0.0, 2.0]])
    evals_Kp = np.array([[1.0, 3.0]])
    evecs = np.tile(np.eye(2, dtype=np.complex128), (1, 1, 1))
    weights = np.array([1.0])
    layer_masks = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu_f = np.array([0.0, 1.0, 0.0, 1.0])

    layer_dos = stoner_layer_dos_at_mu(
        evals_K,
        evecs,
        evals_Kp,
        evecs,
        weights,
        layer_masks,
        mu_f,
        sigma_meV=1.0,
    )
    total = (
        dos_at_mu_gaussian(evals_K, weights, mu_f[0], sigma_meV=1.0, degeneracy=1)
        + dos_at_mu_gaussian(evals_Kp, weights, mu_f[1], sigma_meV=1.0, degeneracy=1)
        + dos_at_mu_gaussian(evals_K, weights, mu_f[2], sigma_meV=1.0, degeneracy=1)
        + dos_at_mu_gaussian(evals_Kp, weights, mu_f[3], sigma_meV=1.0, degeneracy=1)
    )

    assert np.isclose(np.sum(layer_dos), total)
    assert np.all(layer_dos > 0.0)
