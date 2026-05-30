#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdbg_scf.susceptibility import (
    SusceptibilityParams,
    q_results_to_dataframe,
    scan_q_for_state,
    solve_full_scf_stoner_state,
    summarize_q_scan,
)
from tdbg_scf.susceptibility.workflow import write_single_point_outputs


def add_scf_reference_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--scf-source", choices=["projected_8band", "full"], default="projected_8band")
    ap.add_argument("--scf-valley", type=int, choices=[-1, 1], default=1)
    ap.add_argument("--full-scf-valley", type=int, choices=[-1, 1], default=None)
    ap.add_argument("--n-active", type=int, default=8)
    ap.add_argument("--selection", choices=["closest", "contiguous", "overlap"], default="overlap")
    ap.add_argument("--projector-refreshes", type=int, default=0)
    ap.add_argument("--projected-reference-mode", choices=["uploaded_D", "bare_D", "zero", "full_scf"], default="uploaded_D")
    ap.add_argument("--max-iter", "--projected-max-iter", dest="projected_max_iter", type=int, default=100)
    ap.add_argument("--tol-meV", "--projected-tol-meV", dest="projected_tol_meV", type=float, default=1e-4)
    ap.add_argument("--scf-max-iter", dest="legacy_scf_max_iter", type=int, default=None)
    ap.add_argument("--scf-tol-meV", dest="legacy_scf_tol_meV", type=float, default=None)
    ap.add_argument("--kBT-meV", "--scf-kBT-meV", dest="kBT_meV", type=float, default=0.2)
    ap.add_argument("--eps-perp", type=float, default=3.0)
    ap.add_argument("--d-layer-nm", type=float, default=0.335)
    ap.add_argument("--D-sign", type=float, default=-1.0)
    ap.add_argument("--full-max-iter", type=int, default=60)
    ap.add_argument("--full-tol-meV", type=float, default=1e-4)
    ap.add_argument("--mixer", choices=["linear", "anderson"], default="anderson")
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--anderson-beta", type=float, default=0.5)
    ap.add_argument("--anderson-memory", type=int, default=6)
    ap.add_argument("--initial-U", choices=["uploaded", "bare", "zero"], default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Finite-Q transverse susceptibility on a Stoner-after TDBG state")
    ap.add_argument("--n-cm2", type=float, required=True)
    ap.add_argument("--D-Vnm", type=float, required=True)
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1)
    ap.add_argument("--grid-n1", type=int, default=9)
    ap.add_argument("--grid-n2", type=int, default=9)
    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0)
    ap.add_argument("--a-cc-A", type=float, default=1.420)
    ap.add_argument("--gamma0-meV", type=float, default=2610.0)
    ap.add_argument("--gamma1-meV", type=float, default=361.0)
    ap.add_argument("--gamma3-meV", type=float, default=283.0)
    ap.add_argument("--gamma4-meV", type=float, default=138.0)
    add_scf_reference_args(ap)
    ap.add_argument("--stoner-u0-meV-A2", type=float, default=7.9e4)
    ap.add_argument("--stoner-JH-meV-A2", type=float, default=2.4e4)
    ap.add_argument("--stoner-n-random-seeds", type=int, default=20)
    ap.add_argument("--stoner-seed", type=int, default=0)
    ap.add_argument("--chi-kBT-meV", type=float, default=0.05)
    ap.add_argument("--energy-window-meV", type=float, default=30.0)
    ap.add_argument("--max-bands-per-k", type=int, default=24)
    ap.add_argument("--main-vertex-model", choices=["su4_diag", "su2_hund_factor2"], default="su4_diag")
    ap.add_argument("--also-run-hund-factor2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hund-transverse-factor", type=float, default=2.0)
    ap.add_argument("--occupation-mode", choices=["flavor_mu", "common_mu"], default="flavor_mu")
    ap.add_argument("--spin-flip-mode", choices=["plus", "minus", "both_pm"], default="both_pm")
    ap.add_argument("--legacy-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--q-mode", choices=["folded_grid", "unfolded_diagonalize", "folded_grid_with_G_shift"], default="folded_grid")
    ap.add_argument("--q-stride", type=int, default=1)
    ap.add_argument("--max-abs-q-step", type=int, default=None)
    ap.add_argument("--no-layer-matrix", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    state = solve_full_scf_stoner_state(
        n_cm2=args.n_cm2,
        D_Vnm=args.D_Vnm,
        theta_deg=args.theta_deg,
        cutoff=args.cutoff,
        grid_n1=args.grid_n1,
        grid_n2=args.grid_n2,
        omega_meV=args.omega,
        wAA=args.wAA,
        wAB=args.wAB,
        sublattice_Z_meV=args.Z,
        a_cc_A=args.a_cc_A,
        gamma0_meV=args.gamma0_meV,
        gamma1_meV=args.gamma1_meV,
        gamma3_meV=args.gamma3_meV,
        gamma4_meV=args.gamma4_meV,
        scf_source=args.scf_source,
        scf_valley=args.scf_valley,
        full_scf_valley=args.full_scf_valley,
        n_active=args.n_active,
        selection=args.selection,
        projector_refreshes=args.projector_refreshes,
        projected_reference_mode=args.projected_reference_mode,
        kBT_meV=args.kBT_meV,
        projected_max_iter=args.projected_max_iter,
        projected_tol_meV=args.projected_tol_meV,
        scf_max_iter=args.legacy_scf_max_iter,
        scf_tol_meV=args.legacy_scf_tol_meV,
        full_max_iter=args.full_max_iter,
        full_tol_meV=args.full_tol_meV,
        eps_perp=args.eps_perp,
        d_layer_nm=args.d_layer_nm,
        D_sign=args.D_sign,
        mixer=args.mixer,
        alpha=args.alpha,
        anderson_beta=args.anderson_beta,
        anderson_memory=args.anderson_memory,
        initial_U=args.initial_U,
        stoner_u0_meV_A2=args.stoner_u0_meV_A2,
        stoner_JH_meV_A2=args.stoner_JH_meV_A2,
        stoner_n_random_seeds=args.stoner_n_random_seeds,
        stoner_seed=args.stoner_seed,
    )
    chi_params = SusceptibilityParams(
        kBT_meV=args.chi_kBT_meV,
        energy_window_meV=args.energy_window_meV,
        max_bands_per_k=args.max_bands_per_k,
        q_mode=args.q_mode,
        main_vertex_model=args.main_vertex_model,
        also_run_hund_factor2=args.also_run_hund_factor2,
        hund_transverse_factor=args.hund_transverse_factor,
        occupation_mode=args.occupation_mode,
        spin_flip_mode=args.spin_flip_mode,
        legacy_diagnostics=args.legacy_diagnostics,
        include_layer_matrix=not args.no_layer_matrix,
    )
    q_results = scan_q_for_state(
        state,
        chi_params,
        q_stride=args.q_stride,
        max_abs_q_step=args.max_abs_q_step,
        include_gamma=True,
    )
    summary = summarize_q_scan(q_results, finite_q_tol=chi_params.finite_q_tol)
    write_single_point_outputs(args.out, state, q_results, summary)

    qdf = q_results_to_dataframe(q_results)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(qdf["qx_Ainv"], qdf["qy_Ainv"], c=qdf["lambda_u_plus_hund"], s=36)
    ax.set_xlabel(r"$Q_x$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$Q_y$ [$\AA^{-1}$]")
    ax.set_title("Stoner-after transverse susceptibility")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\lambda_{U+J}(Q)$")
    fig.savefig(args.out / "susceptibility_qmap.png", dpi=180)
    plt.close(fig)

    print("Saved:", args.out)
    print(summary)


if __name__ == "__main__":
    main()
