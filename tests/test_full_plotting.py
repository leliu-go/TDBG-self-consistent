import numpy as np


def test_full_path_calculation_keeps_every_continuum_band():
    from tdbg_scf import TDBGContinuumHamiltonian, TDBGParameters
    from tdbg_scf.plotting import compute_full_bands_along_path

    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=1))
    U = np.array([3., 1., -1., -3.])
    distance, energies, ticks, labels = compute_full_bands_along_path(ham, U, points_per_segment=2)
    kpts, expected_distance, expected_ticks, expected_labels = ham.geom.high_symmetry_path(2)
    expected, _ = ham.diagonalize(kpts, U)
    assert energies.shape == (len(distance), 8*(2*ham.params.cutoff+1)**2)
    np.testing.assert_allclose(energies, expected)
    np.testing.assert_allclose(distance, expected_distance)
    assert ticks == expected_ticks and labels == expected_labels
