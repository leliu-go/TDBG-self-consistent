import numpy as np


class _Geom:
    b1m = np.array([1.0, 0.0])
    b2m = np.array([0.0, 2.0])


def test_commensurate_q_points_and_folded_indices_follow_grid_steps():
    from tdbg_scf.susceptibility.qmesh import QPoint, folded_indices_for_q, make_commensurate_q_points

    qpts = make_commensurate_q_points(_Geom(), (4, 3), max_abs_step=1)
    assert len(qpts) == 9
    assert any(q.is_gamma for q in qpts)
    assert any(q.dq1 == 1 and q.dq2 == -1 and np.allclose(q.qvec_Ainv, [0.25, -2.0 / 3.0]) for q in qpts)

    folded = folded_indices_for_q(QPoint(1, -1, np.array([0.25, -2.0 / 3.0])), (4, 3))
    assert folded.tolist()[:6] == [5, 3, 4, 8, 6, 7]


def test_stoner_self_energy_shifts_match_interaction_derivative():
    from tdbg_scf.stoner.params import StonerParams
    from tdbg_scf.susceptibility.stoner_reference import stoner_self_energy_shifts_meV

    nu = np.array([1.0, 0.5, 0.2, 0.1])
    params = StonerParams(u0_meV_A2=10.0, JH_meV_A2=4.0)

    raw = stoner_self_energy_shifts_meV(nu, params, A_M_A2=2.0, remove_common=False)
    assert np.allclose(raw, [3.2, 4.9, 8.8, 10.1])

    shifted = stoner_self_energy_shifts_meV(nu, params, A_M_A2=2.0, remove_common=True)
    assert np.allclose(shifted, raw - np.mean(raw))


def test_vertex_models_use_generalized_stoner_kernel():
    from tdbg_scf.susceptibility.vertex import generalized_stoner_lambda, make_valley_vertex_models

    models = make_valley_vertex_models(u_cell_meV=2.0, J_cell_meV=0.3, hund_transverse_factor=2.0)
    assert np.allclose(models["su4_diag"].gamma, [[2.0, 0.0], [0.0, 2.0]])
    assert np.allclose(models["su2_hund_factor2"].gamma, [[2.0, 0.6], [0.6, 2.0]])
    assert np.allclose(models["legacy_offdiag_J"].gamma, [[2.0, 0.3], [0.3, 2.0]])

    lam, vec = generalized_stoner_lambda(np.diag([0.5, 0.25]), models["su4_diag"].gamma)
    assert np.isclose(lam, 1.0)
    assert vec.shape == (2,)


def test_lindhard_static_ratio_uses_fermi_derivative_for_degenerate_denominator():
    from tdbg_scf.susceptibility.bubble import lindhard_static_ratio

    ratio = lindhard_static_ratio(
        np.array([0.0]),
        np.array([0.0]),
        mu_meV=0.0,
        kBT_meV=0.5,
        denom_tol_meV=1e-6,
    )

    assert ratio.shape == (1, 1)
    assert np.allclose(ratio, [[0.5]])


def test_fermi_surface_jdos_q_counts_same_energy_pairs():
    from tdbg_scf.susceptibility.bubble import compute_fermi_surface_jdos_q
    from tdbg_scf.susceptibility.qmesh import QPoint

    evals = np.array([[0.0], [5.0], [0.0]])
    weights = np.ones(3) / 3.0

    connected = compute_fermi_surface_jdos_q(
        q=QPoint(1, 0, np.array([0.1, 0.0])),
        grid_shape=(3, 1),
        weights=weights,
        A_M_A2=6.0,
        evals_by_flavor_meV=[evals],
        mu_by_flavor_meV=[0.0],
        sigma_meV=0.5,
        energy_window_meV=2.0,
        max_bands_per_k=4,
    )
    disconnected = compute_fermi_surface_jdos_q(
        q=QPoint(0, 0, np.zeros(2)),
        grid_shape=(3, 1),
        weights=weights,
        A_M_A2=6.0,
        evals_by_flavor_meV=[evals + 5.0],
        mu_by_flavor_meV=[0.0],
        sigma_meV=0.5,
        energy_window_meV=2.0,
        max_bands_per_k=4,
    )

    assert connected["jdos_total_cell_meV_inv2"] > 0.0
    assert disconnected["jdos_total_cell_meV_inv2"] == 0.0


def test_minimal_transverse_chi_q_accumulates_two_valleys():
    from tdbg_scf.susceptibility.bubble import compute_transverse_chi_q
    from tdbg_scf.susceptibility.params import SusceptibilityParams
    from tdbg_scf.susceptibility.qmesh import QPoint
    from tdbg_scf.susceptibility.stoner_reference import StonerReference

    evals = np.array([[0.0]])
    evecs = np.ones((1, 1, 1), dtype=np.complex128)
    ref = StonerReference(
        nu_f=np.zeros(4),
        mu_f_meV=np.zeros(4),
        sigma_f_meV=np.zeros(4),
        mu_common_meV=0.0,
        mu_common_spread_meV=0.0,
        u_cell_meV=2.0,
        J_cell_meV=0.0,
    )

    result = compute_transverse_chi_q(
        q=QPoint(0, 0, np.zeros(2)),
        grid_shape=(1, 1),
        weights=np.array([1.0]),
        A_M_A2=1.0,
        evals_K_meV=evals,
        evecs_K=evecs,
        evals_Kp_meV=evals,
        evecs_Kp=evecs,
        ref=ref,
        params=SusceptibilityParams(kBT_meV=0.5, include_layer_matrix=False),
    )

    assert result.chi_K_cell_meV_inv == 0.5
    assert result.chi_Kp_cell_meV_inv == 0.5
    assert result.chi_total_cell_meV_inv == 1.0
    assert result.lambda_u == 2.0
    assert np.isclose(result.lambda_u_plus_hund, 1.0)
    assert result.chi_K_plus_cell_meV_inv == 0.5
    assert result.chi_K_minus_cell_meV_inv == 0.5
    assert np.isclose(result.lambda_su4_diag_selected, 1.0)
    assert result.selected_vertex_model == "su4_diag"
    assert result.selected_spin_flip_direction in {"plus", "minus"}


def test_q_scan_summary_detects_finite_q_winner():
    from tdbg_scf.susceptibility.bubble import ChiQResult
    from tdbg_scf.susceptibility.qmesh import QPoint
    from tdbg_scf.susceptibility.workflow import summarize_q_scan

    results = [
        ChiQResult(
            QPoint(0, 0, np.zeros(2)),
            1.0,
            0.5,
            0.5,
            1.0,
            1.0,
            lambda_su4_diag_selected=1.0,
        ),
        ChiQResult(
            QPoint(1, 0, np.array([0.1, 0.0])),
            1.2,
            0.8,
            0.4,
            1.2,
            1.25,
            lambda_su4_diag_selected=1.25,
            su4_diag_spin_flip_direction="plus",
            extra={"jdos_total_cell_meV_inv2": 0.7},
        ),
        ChiQResult(
            QPoint(0, 1, np.array([0.0, 0.2])),
            0.8,
            0.4,
            0.4,
            0.8,
            0.8,
            lambda_su4_diag_selected=0.8,
            extra={"jdos_total_cell_meV_inv2": 1.4},
        ),
    ]

    summary = summarize_q_scan(results, finite_q_tol=1e-3)

    assert summary["gamma_lambda_su4_diag_selected"] == 1.0
    assert summary["qstar_su4_diag_dq1"] == 1
    assert summary["finite_q_wins_su4_diag"] is True
    assert np.isclose(summary["finite_q_ratio_su4_diag"], 1.25)
    assert summary["qstar_jdos_total_dq2"] == 1
    assert np.isclose(summary["qstar_jdos_total_cell_meV_inv2"], 1.4)
