from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tdbg_scf import (
    FullSCFConfig,
    FullSCFSolver,
    ProjectedSCFConfig,
    ProjectedSCFSolver,
    TDBGContinuumHamiltonian,
    TDBGParameters,
    linear_potential_from_D,
    make_uniform_mbz_grid,
)
from tdbg_scf.filling import moire_cell_area_A2_from_weights
from tdbg_scf.stoner.full_band import (
    build_full_band_flavor_tables,
    density_cm2_to_nu_total,
    restrict_tables_to_carrier_sector,
    stoner_result_to_dict,
)
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu

from .bubble import ChiQResult, compute_transverse_chi_q
from .params import SusceptibilityParams
from .qmesh import make_commensurate_q_points
from .stoner_reference import StonerReference, make_stoner_reference


@dataclass
class SinglePointStonerState:
    """Everything needed for chi(Q) at one (n, D) point."""

    n_cm2: float
    D_Vnm: float
    nu_total: float
    grid_shape: tuple[int, int]
    kpts: np.ndarray
    weights: np.ndarray
    A_M_A2: float
    U_meV: np.ndarray
    scf_result: object
    stoner_result: object
    stoner_params: StonerParams
    stoner_reference: StonerReference
    tables: list
    ham_K: TDBGContinuumHamiltonian
    ham_Kp: TDBGContinuumHamiltonian
    evals_K_meV: np.ndarray
    evecs_K: np.ndarray
    evals_Kp_meV: np.ndarray
    evecs_Kp: np.ndarray
    layer_masks: np.ndarray
    scf_metadata: dict | None = None

    def metadata(self) -> dict:
        out = {
            "n_cm2": float(self.n_cm2),
            "D_Vnm": float(self.D_Vnm),
            "nu_total": float(self.nu_total),
            "A_M_A2": float(self.A_M_A2),
            "grid_n1": int(self.grid_shape[0]),
            "grid_n2": int(self.grid_shape[1]),
            "U1_meV": float(self.U_meV[0]),
            "U2_meV": float(self.U_meV[1]),
            "U3_meV": float(self.U_meV[2]),
            "U4_meV": float(self.U_meV[3]),
            "scf_mu_meV": float(getattr(self.scf_result, "mu_meV", np.nan)),
            "scf_converged": bool(getattr(self.scf_result, "converged", False)),
            "scf_residual_meV": float(getattr(self.scf_result, "residual_meV", np.nan)),
            "scf_iterations": int(getattr(self.scf_result, "iterations", 0)),
        }
        if self.scf_metadata:
            out.update(self.scf_metadata)
        out.update(self.stoner_reference.to_record())
        stoner_dict = stoner_result_to_dict(self.stoner_result, self.tables)
        out.update({f"stoner_{k}": v for k, v in stoner_dict.items() if k != "local_minima"})
        return out


def _make_params(
    *,
    theta_deg: float,
    cutoff: int,
    valley: int,
    omega_meV: float,
    wAA: float,
    wAB: float,
    sublattice_Z_meV: float,
    a_cc_A: float,
    gamma0_meV: float,
    gamma1_meV: float,
    gamma3_meV: float,
    gamma4_meV: float,
) -> TDBGParameters:
    return TDBGParameters(
        theta_deg=float(theta_deg),
        cutoff=int(cutoff),
        valley=int(valley),
        omega_meV=float(omega_meV),
        wAA=float(wAA),
        wAB=float(wAB),
        sublattice_Z_meV=float(sublattice_Z_meV),
        a_cc_A=float(a_cc_A),
        gamma0_meV=float(gamma0_meV),
        gamma1_meV=float(gamma1_meV),
        gamma3_meV=float(gamma3_meV),
        gamma4_meV=float(gamma4_meV),
    )


def _mixer_kwargs(mixer: str, alpha: float, anderson_beta: float, anderson_memory: int) -> dict:
    if mixer == "linear":
        return {"alpha": float(alpha)}
    return {
        "beta": float(anderson_beta),
        "memory": int(anderson_memory),
        "fallback_alpha": float(alpha),
    }


def _projected_reference_from_legacy(initial_U: str | None, projected_reference_mode: str) -> str:
    if initial_U is None or projected_reference_mode != "uploaded_D":
        return projected_reference_mode
    mapping = {"uploaded": "uploaded_D", "bare": "bare_D", "zero": "zero"}
    return mapping.get(str(initial_U), projected_reference_mode)


def _supplied_projector_U(D_Vnm: float, projected_reference_mode: str) -> np.ndarray | None:
    if projected_reference_mode == "full_scf":
        return None
    if projected_reference_mode == "zero":
        return np.zeros(4, dtype=float)
    strength = "bare" if projected_reference_mode == "bare_D" else "uploaded"
    U = linear_potential_from_D(float(D_Vnm), strength=strength)
    U -= np.mean(U)
    return U


def _run_scf_reference(
    *,
    scf_source: str,
    ham: TDBGContinuumHamiltonian,
    kpts: np.ndarray,
    weights: np.ndarray,
    grid_shape: tuple[int, int],
    n_cm2: float,
    D_Vnm: float,
    kBT_meV: float,
    eps_perp: float,
    d_layer_nm: float,
    D_sign: float,
    mixer: str,
    alpha: float,
    anderson_beta: float,
    anderson_memory: int,
    full_max_iter: int,
    full_tol_meV: float,
    initial_U: str | None,
    n_active: int,
    selection: str,
    projector_refreshes: int,
    projected_reference_mode: str,
    projected_max_iter: int,
    projected_tol_meV: float,
) -> tuple[object, dict]:
    mix_kwargs = _mixer_kwargs(mixer, alpha, anderson_beta, anderson_memory)
    if scf_source == "full":
        cfg = FullSCFConfig(
            target_density_cm2=float(n_cm2),
            D_Vnm=float(D_Vnm),
            degeneracy=4,
            kBT_meV=float(kBT_meV),
            max_iter=int(full_max_iter),
            tol_meV=float(full_tol_meV),
            eps_perp=float(eps_perp),
            d_layer_nm=float(d_layer_nm),
            D_sign=float(D_sign),
            mixer=str(mixer),
            mixer_kwargs=mix_kwargs,
            initial_U=str(initial_U or "uploaded"),
            keep_eigensystem=False,
        )
        result = FullSCFSolver(ham, kpts, weights).solve(cfg)
        return result, {
            "scf_source": "full",
            "scf_reference_mode": "full",
            "full_max_iter": int(full_max_iter),
            "full_tol_meV": float(full_tol_meV),
        }

    if scf_source != "projected_8band":
        raise ValueError("scf_source must be 'projected_8band' or 'full'")

    projected_reference_mode = _projected_reference_from_legacy(initial_U, projected_reference_mode)
    supplied_U = _supplied_projector_U(D_Vnm, projected_reference_mode)
    reference_mode = "supplied_U"
    full_cfg = None
    if projected_reference_mode == "full_scf":
        reference_mode = "full_scf"
        full_cfg = FullSCFConfig(
            target_density_cm2=float(n_cm2),
            D_Vnm=float(D_Vnm),
            degeneracy=4,
            kBT_meV=float(kBT_meV),
            max_iter=int(full_max_iter),
            tol_meV=float(full_tol_meV),
            eps_perp=float(eps_perp),
            d_layer_nm=float(d_layer_nm),
            D_sign=float(D_sign),
            mixer=str(mixer),
            mixer_kwargs=mix_kwargs,
            initial_U=str(initial_U or "uploaded"),
            keep_eigensystem=False,
        )

    cfg = ProjectedSCFConfig(
        target_density_cm2=float(n_cm2),
        D_Vnm=float(D_Vnm),
        n_active=int(n_active),
        degeneracy=4,
        kBT_meV=float(kBT_meV),
        max_iter=int(projected_max_iter),
        tol_meV=float(projected_tol_meV),
        eps_perp=float(eps_perp),
        d_layer_nm=float(d_layer_nm),
        D_sign=float(D_sign),
        mixer=str(mixer),
        mixer_kwargs=mix_kwargs,
        selection=str(selection),
        grid_shape=grid_shape,
        projector_refreshes=int(projector_refreshes),
        reference_mode=reference_mode,
        supplied_U_ref_meV=supplied_U,
        full_scf_config=full_cfg,
    )
    result = ProjectedSCFSolver(ham, kpts, weights, grid_shape=grid_shape).solve(cfg)
    return result, {
        "scf_source": "projected_8band",
        "scf_reference_mode": projected_reference_mode,
        "n_active": int(n_active),
        "selection": str(selection),
        "projector_refreshes": int(projector_refreshes),
        "projected_max_iter": int(projected_max_iter),
        "projected_tol_meV": float(projected_tol_meV),
    }


def solve_full_scf_stoner_state(
    *,
    n_cm2: float,
    D_Vnm: float,
    theta_deg: float = 1.35,
    cutoff: int = 1,
    grid_n1: int = 9,
    grid_n2: int = 9,
    omega_meV: float = 100.0,
    wAA: float = 0.8,
    wAB: float = 1.0,
    sublattice_Z_meV: float = 15.0,
    a_cc_A: float = 1.420,
    gamma0_meV: float = 2610.0,
    gamma1_meV: float = 361.0,
    gamma3_meV: float = 283.0,
    gamma4_meV: float = 138.0,
    scf_source: str = "projected_8band",
    scf_valley: int = 1,
    full_scf_valley: int | None = None,
    n_active: int = 8,
    selection: str = "overlap",
    projector_refreshes: int = 0,
    projected_reference_mode: str = "uploaded_D",
    kBT_meV: float | None = None,
    scf_kBT_meV: float | None = None,
    projected_max_iter: int = 100,
    projected_tol_meV: float = 1e-4,
    scf_max_iter: int | None = None,
    scf_tol_meV: float | None = None,
    full_max_iter: int = 60,
    full_tol_meV: float = 1e-4,
    eps_perp: float = 3.0,
    d_layer_nm: float = 0.335,
    D_sign: float = -1.0,
    mixer: str = "anderson",
    alpha: float = 0.06,
    anderson_beta: float = 0.5,
    anderson_memory: int = 6,
    initial_U: str | None = None,
    stoner_u0_meV_A2: float = 7.9e4,
    stoner_JH_meV_A2: float = 2.4e4,
    stoner_temperature_K: float = 0.0,
    stoner_n_random_seeds: int = 20,
    stoner_seed: int = 0,
) -> SinglePointStonerState:
    """Run SCF reference, diagonalize full K/Kp bands, and solve Stoner."""

    params_K = _make_params(
        theta_deg=theta_deg,
        cutoff=cutoff,
        valley=+1,
        omega_meV=omega_meV,
        wAA=wAA,
        wAB=wAB,
        sublattice_Z_meV=sublattice_Z_meV,
        a_cc_A=a_cc_A,
        gamma0_meV=gamma0_meV,
        gamma1_meV=gamma1_meV,
        gamma3_meV=gamma3_meV,
        gamma4_meV=gamma4_meV,
    )
    params_Kp = _make_params(
        theta_deg=theta_deg,
        cutoff=cutoff,
        valley=-1,
        omega_meV=omega_meV,
        wAA=wAA,
        wAB=wAB,
        sublattice_Z_meV=sublattice_Z_meV,
        a_cc_A=a_cc_A,
        gamma0_meV=gamma0_meV,
        gamma1_meV=gamma1_meV,
        gamma3_meV=gamma3_meV,
        gamma4_meV=gamma4_meV,
    )

    ham_K = TDBGContinuumHamiltonian(params_K)
    ham_Kp = TDBGContinuumHamiltonian(params_Kp)
    valley_for_scf = int(full_scf_valley if full_scf_valley is not None else scf_valley)
    ham_scf = ham_K if valley_for_scf == 1 else ham_Kp
    kpts, weights = make_uniform_mbz_grid(ham_scf.geom, int(grid_n1), int(grid_n2))
    A_M_A2 = moire_cell_area_A2_from_weights(weights)

    scf_temperature = float(kBT_meV if kBT_meV is not None else (scf_kBT_meV if scf_kBT_meV is not None else 0.2))
    if scf_max_iter is not None:
        if scf_source == "full":
            full_max_iter = int(scf_max_iter)
        else:
            projected_max_iter = int(scf_max_iter)
    if scf_tol_meV is not None:
        if scf_source == "full":
            full_tol_meV = float(scf_tol_meV)
        else:
            projected_tol_meV = float(scf_tol_meV)

    scf, scf_meta = _run_scf_reference(
        scf_source=str(scf_source),
        ham=ham_scf,
        kpts=kpts,
        weights=weights,
        grid_shape=(int(grid_n1), int(grid_n2)),
        n_cm2=float(n_cm2),
        D_Vnm=float(D_Vnm),
        kBT_meV=scf_temperature,
        eps_perp=float(eps_perp),
        d_layer_nm=float(d_layer_nm),
        D_sign=float(D_sign),
        mixer=str(mixer),
        alpha=float(alpha),
        anderson_beta=float(anderson_beta),
        anderson_memory=int(anderson_memory),
        full_max_iter=int(full_max_iter),
        full_tol_meV=float(full_tol_meV),
        initial_U=initial_U,
        n_active=int(n_active),
        selection=str(selection),
        projector_refreshes=int(projector_refreshes),
        projected_reference_mode=str(projected_reference_mode),
        projected_max_iter=int(projected_max_iter),
        projected_tol_meV=float(projected_tol_meV),
    )
    scf_meta.update(
        {
            "scf_valley": int(valley_for_scf),
            "scf_kBT_meV": float(scf_temperature),
            "eps_perp": float(eps_perp),
            "d_layer_nm": float(d_layer_nm),
            "D_sign": float(D_sign),
            "mixer": str(mixer),
        }
    )
    U = np.asarray(scf.U_meV, dtype=float)

    evals_K, evecs_K = ham_K.diagonalize(kpts, U)
    evals_Kp, evecs_Kp = ham_Kp.diagonalize(kpts, U)
    layer_masks = ham_K.layer_projectors_diagonal()

    nu_total = density_cm2_to_nu_total(float(n_cm2), A_M_A2)
    tables = restrict_tables_to_carrier_sector(
        build_full_band_flavor_tables(evals_K, evals_Kp, weights, A_M_A2),
        nu_total,
    )
    stoner_params = StonerParams(
        u0_meV_A2=float(stoner_u0_meV_A2),
        JH_meV_A2=float(stoner_JH_meV_A2),
        temperature_K=float(stoner_temperature_K),
        n_random_seeds=int(stoner_n_random_seeds),
        seed=int(stoner_seed),
    )
    stoner = solve_stoner_fixed_nu(nu_total, tables, stoner_params, A_M_A2)
    ref = make_stoner_reference(stoner, tables, stoner_params, A_M_A2)

    return SinglePointStonerState(
        n_cm2=float(n_cm2),
        D_Vnm=float(D_Vnm),
        nu_total=float(nu_total),
        grid_shape=(int(grid_n1), int(grid_n2)),
        kpts=kpts,
        weights=weights,
        A_M_A2=float(A_M_A2),
        U_meV=U,
        scf_result=scf,
        stoner_result=stoner,
        stoner_params=stoner_params,
        stoner_reference=ref,
        tables=tables,
        ham_K=ham_K,
        ham_Kp=ham_Kp,
        evals_K_meV=evals_K,
        evecs_K=evecs_K,
        evals_Kp_meV=evals_Kp,
        evecs_Kp=evecs_Kp,
        layer_masks=layer_masks,
        scf_metadata=scf_meta,
    )


def scan_q_for_state(
    state: SinglePointStonerState,
    susc_params: SusceptibilityParams,
    q_stride: int = 1,
    max_abs_q_step: int | None = None,
    include_gamma: bool = True,
) -> list[ChiQResult]:
    qpts = make_commensurate_q_points(
        state.ham_K.geom,
        state.grid_shape,
        q_stride=int(q_stride),
        max_abs_step=max_abs_q_step,
        include_gamma=include_gamma,
    )

    results: list[ChiQResult] = []
    for q in qpts:
        if susc_params.q_mode == "unfolded_diagonalize":
            kqpts = state.kpts + q.qvec_Ainv[None, :]
            evals_K_q, evecs_K_q = state.ham_K.diagonalize(kqpts, state.U_meV)
            evals_Kp_q, evecs_Kp_q = state.ham_Kp.diagonalize(kqpts, state.U_meV)
        else:
            evals_K_q = evecs_K_q = evals_Kp_q = evecs_Kp_q = None

        results.append(
            compute_transverse_chi_q(
                q=q,
                grid_shape=state.grid_shape,
                weights=state.weights,
                A_M_A2=state.A_M_A2,
                evals_K_meV=state.evals_K_meV,
                evecs_K=state.evecs_K,
                evals_Kp_meV=state.evals_Kp_meV,
                evecs_Kp=state.evecs_Kp,
                ref=state.stoner_reference,
                params=susc_params,
                layer_masks=state.layer_masks if susc_params.include_layer_matrix else None,
                evals_K_q_meV=evals_K_q,
                evecs_K_q=evecs_K_q,
                evals_Kp_q_meV=evals_Kp_q,
                evecs_Kp_q=evecs_Kp_q,
                plane_wave_G_indices=state.ham_K.lattice.G_indices,
            )
        )
    return results


def q_results_to_dataframe(results: list[ChiQResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_record() for r in results])


def summarize_q_scan(results: list[ChiQResult], finite_q_tol: float = 1e-3) -> dict:
    if not results:
        raise ValueError("empty q scan")

    gamma = [r for r in results if r.q.is_gamma]
    gamma_res = gamma[0] if gamma else min(results, key=lambda r: r.q.q_norm_Ainv)
    nonzero = [r for r in results if not r.q.is_gamma]

    out = {
        "gamma_dq1": int(gamma_res.q.dq1),
        "gamma_dq2": int(gamma_res.q.dq2),
        "gamma_chi_total_cell_meV_inv": float(gamma_res.chi_total_cell_meV_inv),
    }
    models = [
        "su4_diag",
        "su2_hund_factor2",
        "legacy_scalar_total",
        "legacy_offdiag_J",
    ]
    first_model_summary = None
    for model in models:
        attr = f"lambda_{model}_selected"
        available = [r for r in results if getattr(r, attr, None) is not None]
        if not available:
            continue
        gamma_model = gamma_res if getattr(gamma_res, attr, None) is not None else min(available, key=lambda r: r.q.q_norm_Ainv)
        nonzero_available = [r for r in available if not r.q.is_gamma]
        max_all = max(available, key=lambda r: float(getattr(r, attr)))
        max_nonzero = max(nonzero_available, key=lambda r: float(getattr(r, attr))) if nonzero_available else max_all
        gamma_lambda = float(getattr(gamma_model, attr))
        nonzero_lambda = float(getattr(max_nonzero, attr))
        ratio = float(nonzero_lambda / gamma_lambda) if abs(gamma_lambda) > 1e-14 else np.inf
        finite_q_wins = bool(nonzero_available and nonzero_lambda > gamma_lambda * (1.0 + float(finite_q_tol)))
        prefix = model
        out[f"gamma_lambda_{prefix}_selected"] = gamma_lambda
        out[f"qstar_nonzero_lambda_{prefix}_selected"] = nonzero_lambda
        out[f"finite_q_ratio_{prefix}"] = ratio
        out[f"finite_q_wins_{prefix}"] = finite_q_wins
        out[f"qstar_{prefix}_dq1"] = int(max_nonzero.q.dq1)
        out[f"qstar_{prefix}_dq2"] = int(max_nonzero.q.dq2)
        out[f"qstar_{prefix}_qx_Ainv"] = float(max_nonzero.q.qx_Ainv)
        out[f"qstar_{prefix}_qy_Ainv"] = float(max_nonzero.q.qy_Ainv)
        out[f"qstar_{prefix}_qnorm_Ainv"] = float(max_nonzero.q.q_norm_Ainv)
        out[f"qstar_{prefix}_spin_flip_direction"] = str(
            getattr(max_nonzero, f"{model}_spin_flip_direction", None)
            or getattr(max_nonzero, "selected_spin_flip_direction", "")
        )
        overlap = getattr(max_nonzero, f"{model}_layer_dipole_overlap", None)
        ratio_layer = getattr(max_nonzero, f"{model}_layer_dipole_ratio", None)
        if overlap is not None:
            out[f"qstar_{prefix}_layer_dipole_overlap"] = float(overlap)
        if ratio_layer is not None:
            out[f"qstar_{prefix}_layer_dipole_ratio"] = float(ratio_layer)
        if first_model_summary is None:
            first_model_summary = (ratio, finite_q_wins, max_nonzero, nonzero_lambda, gamma_lambda)

    if first_model_summary is not None:
        ratio, finite_q_wins, max_nonzero, nonzero_lambda, gamma_lambda = first_model_summary
        out["qstar_nonzero_dq1"] = int(max_nonzero.q.dq1)
        out["qstar_nonzero_dq2"] = int(max_nonzero.q.dq2)
        out["qstar_nonzero_norm_Ainv"] = float(max_nonzero.q.q_norm_Ainv)
        out["qstar_nonzero_lambda_u_plus_hund"] = float(nonzero_lambda)
        out["gamma_lambda_u_plus_hund"] = float(gamma_lambda)
        out["finite_q_lambda_ratio"] = float(ratio)
        out["finite_q_wins"] = bool(finite_q_wins)

    jdos_key = "jdos_total_cell_meV_inv2"
    jdos_available = [r for r in results if jdos_key in r.extra]
    if jdos_available:
        gamma_jdos = float(gamma_res.extra.get(jdos_key, 0.0))
        nonzero_jdos = [r for r in jdos_available if not r.q.is_gamma]
        max_jdos = max(nonzero_jdos if nonzero_jdos else jdos_available, key=lambda r: float(r.extra[jdos_key]))
        qstar_jdos = float(max_jdos.extra[jdos_key])
        ratio = float(qstar_jdos / gamma_jdos) if abs(gamma_jdos) > 1e-30 else np.inf
        out["gamma_jdos_total_cell_meV_inv2"] = gamma_jdos
        out["qstar_jdos_total_cell_meV_inv2"] = qstar_jdos
        out["finite_q_jdos_ratio"] = ratio
        out["finite_q_jdos_wins"] = bool(nonzero_jdos and qstar_jdos > gamma_jdos * (1.0 + float(finite_q_tol)))
        out["qstar_jdos_total_dq1"] = int(max_jdos.q.dq1)
        out["qstar_jdos_total_dq2"] = int(max_jdos.q.dq2)
        out["qstar_jdos_total_qx_Ainv"] = float(max_jdos.q.qx_Ainv)
        out["qstar_jdos_total_qy_Ainv"] = float(max_jdos.q.qy_Ainv)
        out["qstar_jdos_total_qnorm_Ainv"] = float(max_jdos.q.q_norm_Ainv)
    return out


def write_single_point_outputs(
    out_dir: Path,
    state: SinglePointStonerState,
    q_results: list[ChiQResult],
    summary: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qdf = q_results_to_dataframe(q_results)
    qdf.to_csv(out_dir / "susceptibility_qmap.csv", index=False)

    np.savez(
        out_dir / "susceptibility_qmap.npz",
        qmap=qdf.to_records(index=False),
        U_meV=np.asarray(state.U_meV, dtype=float),
        nu_f=np.asarray(state.stoner_result.nu_f, dtype=float),
        mu_f_meV=np.asarray(state.stoner_reference.mu_f_meV, dtype=float),
        sigma_f_meV=np.asarray(state.stoner_reference.sigma_f_meV, dtype=float),
    )
    payload = {"state": state.metadata(), "summary": summary}
    (out_dir / "susceptibility_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
