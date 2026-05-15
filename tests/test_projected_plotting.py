import numpy as np

from tdbg_scf import TDBGContinuumHamiltonian, TDBGParameters
from tdbg_scf.plotting import compute_projected_bands_along_path


def test_compute_projected_bands_along_path_uses_final_potential():
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    U_ref = np.zeros(4)
    U_final = np.array([3.0, 1.0, -1.0, -3.0])

    dist, bands_ref, ticks, labels = compute_projected_bands_along_path(
        ham,
        U_ref_meV=U_ref,
        U_meV=U_ref,
        mu_ref_meV=0.0,
        n_active=4,
        points_per_segment=2,
    )
    dist_final, bands_final, ticks_final, labels_final = compute_projected_bands_along_path(
        ham,
        U_ref_meV=U_ref,
        U_meV=U_final,
        mu_ref_meV=0.0,
        n_active=4,
        points_per_segment=2,
    )

    assert bands_ref.shape == (len(dist), 4)
    assert bands_final.shape == bands_ref.shape
    assert np.array_equal(dist_final, dist)
    assert ticks_final == ticks
    assert labels_final == labels
    assert not np.allclose(bands_final, bands_ref)


def test_compute_projected_bands_along_path_supports_overlap_selection():
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))

    dist, bands, ticks, labels = compute_projected_bands_along_path(
        ham,
        U_ref_meV=np.zeros(4),
        U_meV=np.zeros(4),
        mu_ref_meV=0.0,
        n_active=4,
        selection="overlap",
        points_per_segment=2,
    )

    assert bands.shape == (len(dist), 4)
    assert ticks == [0, 2, 4, 6]
    assert labels == [r"$K_s$", r"$\Gamma_s$", r"$M_s$", r"$K'_s$"]
