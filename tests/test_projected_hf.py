import numpy as np
import pytest


def test_density_matrix_has_target_filling_and_is_hermitian():
    from tdbg_scf.projected_hf.density_matrix import density_matrices_from_eigensystems

    evals = np.array(
        [
            [[-1.0, 2.0], [0.0, 3.0]],
            [[-1.0, 2.0], [0.0, 3.0]],
            [[-1.0, 2.0], [0.0, 3.0]],
            [[-1.0, 2.0], [0.0, 3.0]],
        ]
    )
    evecs = np.tile(np.eye(2, dtype=np.complex128), (4, 2, 1, 1))
    weights = np.array([0.5, 0.5])

    mu, density = density_matrices_from_eigensystems(
        evals,
        evecs,
        weights,
        A_M_A2=2.0,
        nu_target=0.0,
        n_ref_per_flavor=1.0,
        kBT_meV=0.0,
    )

    assert -1.0 < mu < 2.0
    assert density.P.shape == (4, 2, 2, 2)
    assert np.isclose(np.sum(density.nu_f), 0.0)
    assert np.allclose(density.P, np.swapaxes(density.P.conj(), -1, -2))


def test_contact_projected_self_energy_is_hermitian():
    from tdbg_scf.projected_hf.self_energy import contact_projected_self_energy

    P = np.zeros((4, 2, 2, 2), dtype=np.complex128)
    P[:, :, 0, 0] = 1.0
    sigma = contact_projected_self_energy(P, g_contact_meV=3.0)

    assert sigma.shape == P.shape
    assert np.allclose(sigma, np.swapaxes(sigma.conj(), -1, -2))
    assert np.all(np.real(sigma[:, :, 0, 0]) < 0.0)


def test_coulomb_formfactor_exchange_is_explicitly_not_implemented():
    from tdbg_scf.projected_hf.self_energy import build_exchange_self_energy

    with pytest.raises(NotImplementedError):
        build_exchange_self_energy("coulomb_formfactor", None, None)


def test_projected_hf_solver_smoke_exchange_none():
    from tdbg_scf.projected_hf.flavor_model import FlavorProjectedModel
    from tdbg_scf.projected_hf.params import ProjectedHFParams
    from tdbg_scf.projected_hf.solver import ProjectedHFSolver

    kpts = np.zeros((2, 2))
    weights = np.array([0.5, 0.5])
    layer_mats = np.zeros((2, 4, 2, 2), dtype=np.complex128)
    for ik in range(2):
        for layer in range(4):
            layer_mats[ik, layer] = 0.25 * np.eye(2)
    models = [
        FlavorProjectedModel(
            flavor_index=f,
            energies0_meV=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            layer_mats=layer_mats,
            U_ref_meV=np.zeros(4),
            kpts=kpts,
            weights=weights,
            remote_density_per_k_layer=np.zeros((2, 4)),
        )
        for f in range(4)
    ]

    solver = ProjectedHFSolver(models, params=ProjectedHFParams(exchange_model="none", max_iter=2))
    result = solver.solve_fixed_nu_D(nu_total=0.0, D_Vnm=0.0)

    assert result.converged
    assert np.isclose(np.sum(result.nu_f), 0.0)
    assert result.U_meV.shape == (4,)
