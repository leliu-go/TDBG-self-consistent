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


def test_first_mbz_reduction_deduplicates_reciprocal_equivalent_q_points():
    from tdbg_scf.susceptibility.qmesh import make_commensurate_q_points, reduce_q_to_first_mbz

    reduced = reduce_q_to_first_mbz(np.array([1.25, 0.0]), _Geom())
    assert np.allclose(reduced.q_mbz_Ainv, [0.25, 0.0])
    assert reduced.reciprocal_shift_m == -1
    assert reduced.reciprocal_shift_n == 0

    qpts = make_commensurate_q_points(_Geom(), (2, 1), first_mbz_only=True)
    canonical = {(round(q.qx_mbz_Ainv, 12), round(q.qy_mbz_Ainv, 12)) for q in qpts}
    assert len(canonical) == len(qpts)


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


def test_flavor_mu_lindhard_does_not_silently_use_derivative_for_nonequilibrium_degeneracy():
    from tdbg_scf.susceptibility.bubble import lindhard_static_ratio_flavor_mu

    ratio, diag = lindhard_static_ratio_flavor_mu(
        np.array([0.0]),
        np.array([0.0]),
        mu_initial_meV=-1.0,
        mu_final_meV=1.0,
        kBT_meV=0.5,
        denom_tol_meV=1e-6,
        return_diagnostics=True,
    )

    assert diag["regulated_nonequilibrium_pairs"] == 1
    assert not np.allclose(ratio, [[0.5]])


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


def test_soft_transverse_channel_follows_spin_polarization_sign():
    from tdbg_scf.susceptibility.stoner_reference import soft_transverse_channel

    assert soft_transverse_channel(np.array([1.0, 0.5, 0.1, 0.2])) == "minus"
    assert soft_transverse_channel(np.array([0.1, 0.2, 1.0, 0.5])) == "plus"
    assert soft_transverse_channel(np.array([0.5, 0.5, 0.5, 0.5])) == "degenerate"


def test_common_mu_reference_reconstructs_fillings_from_stoner_tables():
    from tdbg_scf.stoner.dos import FlavorBandTable
    from tdbg_scf.stoner.params import StonerParams
    from tdbg_scf.stoner.solver import StonerResult
    from tdbg_scf.susceptibility.stoner_reference import make_stoner_reference

    table = FlavorBandTable(
        nu_grid=np.array([0.0, 1.0]),
        energy_grid_meV=np.array([0.0, 1.0]),
        kinetic_grid_meV=np.array([0.0, 0.5]),
        nu_min=0.0,
        nu_max=1.0,
    )
    result = StonerResult(
        nu_total=2.0,
        nu_f=np.array([0.5, 0.5, 0.5, 0.5]),
        energy_meV_per_cell=0.0,
        success=True,
        seed_name="toy",
        local_minima=[],
    )

    ref = make_stoner_reference(result, [table, table, table, table], StonerParams(u0_meV_A2=0.0, JH_meV_A2=0.0), A_M_A2=1.0)

    assert np.allclose(ref.nu_f_reconstructed_common_mu, result.nu_f)
    assert ref.max_abs_nu_f_mismatch < 1e-12
    assert ref.reference_valid is True


def test_spinflip_nesting_peak_follows_known_q_shift():
    from tdbg_scf.susceptibility.bubble import compute_spinflip_nesting_q
    from tdbg_scf.susceptibility.params import SusceptibilityParams
    from tdbg_scf.susceptibility.qmesh import QPoint

    evals_i = np.array([[0.0], [5.0], [5.0]])
    evals_f = np.array([[5.0], [0.0], [5.0]])
    evecs = np.ones((3, 1, 1), dtype=np.complex128)
    weights = np.ones(3) / 3.0

    peak = compute_spinflip_nesting_q(
        q=QPoint(1, 0, np.array([0.1, 0.0])),
        grid_shape=(3, 1),
        weights=weights,
        A_M_A2=3.0,
        evals_i=evals_i,
        evecs_i=evecs,
        evals_f=evals_f,
        evecs_f=evecs,
        mu_i_meV=0.0,
        mu_f_meV=0.0,
        params=SusceptibilityParams(nesting_sigma_meV=0.5),
        folded_mode=True,
    )
    offpeak = compute_spinflip_nesting_q(
        q=QPoint(0, 0, np.zeros(2)),
        grid_shape=(3, 1),
        weights=weights,
        A_M_A2=3.0,
        evals_i=evals_i,
        evecs_i=evecs,
        evals_f=evals_f,
        evecs_f=evecs,
        mu_i_meV=0.0,
        mu_f_meV=0.0,
        params=SusceptibilityParams(nesting_sigma_meV=0.5),
        folded_mode=True,
    )

    assert peak > offpeak


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
        params=SusceptibilityParams(kBT_meV=0.5, include_layer_matrix=False, occupation_mode="flavor_mu"),
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


def test_equilibrium_common_mu_matches_shifted_self_energy_gauge():
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
        sigma_full_f_meV=np.full(4, 10.0),
        sigma_shifted_f_meV=np.zeros(4),
        mu_eq_meV=10.0,
        soft_transverse_channel="degenerate",
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
        params=SusceptibilityParams(kBT_meV=0.5, include_layer_matrix=False, occupation_mode="equilibrium_common_mu"),
    )

    assert result.chi_K_plus_cell_meV_inv == 0.5
    assert result.chi_K_minus_cell_meV_inv == 0.5
    assert result.chi_total_cell_meV_inv == 1.0


def test_vhs_diagnostics_uses_equilibrium_mu_in_shifted_self_energy_gauge():
    from tdbg_scf.susceptibility.stoner_reference import StonerReference
    from tdbg_scf.susceptibility.vhs import robust_vhs_diagnostics

    ref = StonerReference(
        nu_f=np.zeros(4),
        mu_f_meV=np.zeros(4),
        sigma_f_meV=np.zeros(4),
        mu_common_meV=0.0,
        mu_common_spread_meV=0.0,
        u_cell_meV=2.0,
        J_cell_meV=0.0,
        sigma_full_f_meV=np.full(4, 10.0),
        sigma_shifted_f_meV=np.zeros(4),
        mu_eq_meV=10.0,
    )

    class _State:
        stoner_reference = ref
        evals_K_meV = np.array([[0.0]])
        evals_Kp_meV = np.array([[0.0]])
        weights = np.array([1.0])

    diag = robust_vhs_diagnostics(_State(), sigma_meV=1.0)

    assert diag["DOS_active_flavor_EF"] > 0.3
    assert diag["DOS_total_EF"] > 1.0


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
            extra={"lambda_su4_diag_soft": 1.0, "soft_transverse_channel": "minus"},
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
            extra={"jdos_total_cell_meV_inv2": 0.7, "lambda_su4_diag_soft": 1.25, "soft_transverse_channel": "minus"},
        ),
        ChiQResult(
            QPoint(0, 1, np.array([0.0, 0.2])),
            0.8,
            0.4,
            0.4,
            0.8,
            0.8,
            lambda_su4_diag_selected=0.8,
            extra={"jdos_total_cell_meV_inv2": 1.4, "lambda_su4_diag_soft": 0.8, "soft_transverse_channel": "minus"},
        ),
    ]

    summary = summarize_q_scan(results, finite_q_tol=1e-3)

    assert summary["gamma_lambda_su4_diag_selected"] == 1.0
    assert summary["qstar_su4_diag_dq1"] == 1
    assert summary["finite_q_wins_su4_diag"] is True
    assert np.isclose(summary["finite_q_ratio_su4_diag"], 1.25)
    assert summary["qstar_jdos_total_dq2"] == 1
    assert np.isclose(summary["qstar_jdos_total_cell_meV_inv2"], 1.4)
    assert summary["soft_transverse_channel"] == "minus"
    assert summary["lambda_gamma_su4_soft"] == 1.0
    assert summary["lambda_qstar_su4_soft"] == 1.25
    assert np.isclose(summary["finite_q_delta_normalized"], 0.25)
    assert summary["finite_q_status"] == "finite_Q_candidate"
