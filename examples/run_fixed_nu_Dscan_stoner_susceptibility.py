#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
for _path in (SCRIPT_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_nd_mapping_stoner_susceptibility import add_bool_optional_arg, add_scf_reference_args, write_json

from tdbg_scf.susceptibility import (
    SusceptibilityParams,
    q_results_to_dataframe,
    scan_q_for_state,
    solve_full_scf_stoner_state,
    summarize_q_scan,
)
from tdbg_scf.susceptibility.dscan import (
    D_point_label,
    build_D_values,
    build_fixed_nu_D_points,
    load_completed_D_values,
    parse_float_list,
)
from tdbg_scf.susceptibility.vhs import robust_vhs_diagnostics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fixed-filling D scan of Stoner-after finite-Q susceptibility.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--nu-total", type=float, default=None)
    group.add_argument("--n-cm2", type=float, default=None)
    ap.add_argument("--D-min", type=float, default=None)
    ap.add_argument("--D-max", type=float, default=None)
    ap.add_argument("--D-count", type=int, default=None)
    ap.add_argument("--D-values", type=str, default=None)
    ap.add_argument("--selected-D-values", type=str, default=None)
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1)
    ap.add_argument("--grid-n1", type=int, default=11)
    ap.add_argument("--grid-n2", type=int, default=11)
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
    ap.add_argument("--stoner-temperature-K", type=float, default=0.0)
    ap.add_argument("--stoner-n-random-seeds", type=int, default=20)
    ap.add_argument("--stoner-seed", type=int, default=0)
    ap.add_argument("--chi-kBT-meV", type=float, default=0.05)
    ap.add_argument("--energy-window-meV", type=float, default=30.0)
    ap.add_argument("--max-bands-per-k", type=int, default=24)
    ap.add_argument("--main-vertex-model", choices=["su4_diag", "su2_hund_factor2"], default="su4_diag")
    add_bool_optional_arg(ap, "also-run-hund-factor2", default=True)
    ap.add_argument("--hund-transverse-factor", type=float, default=2.0)
    ap.add_argument("--occupation-mode", choices=["equilibrium_common_mu", "flavor_mu", "common_mu"], default="equilibrium_common_mu")
    ap.add_argument("--spin-flip-mode", choices=["plus", "minus", "both_pm"], default="both_pm")
    add_bool_optional_arg(ap, "legacy-diagnostics", default=True)
    ap.add_argument("--q-mode", choices=["folded_grid", "unfolded_diagonalize", "folded_grid_with_G_shift"], default="folded_grid_with_G_shift")
    ap.add_argument("--q-stride", type=int, default=1)
    ap.add_argument("--max-abs-q-step", type=int, default=None)
    ap.add_argument("--dos-sigma-meV", type=float, default=1.0)
    ap.add_argument("--no-layer-matrix", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    return ap


def run_point(point, args: argparse.Namespace, save_detail: bool) -> dict:
    t0 = time.time()
    state = solve_full_scf_stoner_state(
        n_cm2=point.n_cm2,
        D_Vnm=point.D_Vnm,
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
        stoner_temperature_K=args.stoner_temperature_K,
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
        nesting_sigma_meV=args.dos_sigma_meV,
    )
    q_results = scan_q_for_state(state, chi_params, q_stride=args.q_stride, max_abs_q_step=args.max_abs_q_step, include_gamma=True)
    summary = summarize_q_scan(q_results, finite_q_tol=chi_params.finite_q_tol)
    rec = {
        "D_index": int(point.D_index),
        "D_Vnm": float(point.D_Vnm),
        "n_cm2": float(point.n_cm2),
        "nu_total": float(point.nu_total),
        "D_sign": float(args.D_sign),
        "chi_status": "ok",
        "chi_runtime_s": float(time.time() - t0),
    }
    rec.update(state.metadata())
    rec.update(summary)
    rec.update(robust_vhs_diagnostics(state, sigma_meV=args.dos_sigma_meV))
    if not bool(rec.get("reference_valid", False)):
        rec["finite_q_status"] = "invalid_reference"
    if save_detail:
        point_dir = args.out / "D_points" / D_point_label(point.D_Vnm)
        point_dir.mkdir(parents=True, exist_ok=True)
        q_results_to_dataframe(q_results).to_csv(point_dir / "susceptibility_qmap.csv", index=False)
        write_json(point_dir / "state_metadata.json", {"state": state.metadata(), "summary": summary})
    return rec


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    D_values = build_D_values(args.D_min, args.D_max, args.D_count, args.D_values)
    points = build_fixed_nu_D_points(
        nu_total=args.nu_total,
        n_cm2=args.n_cm2,
        D_values=D_values,
        theta_deg=args.theta_deg,
        a_cc_A=args.a_cc_A,
        valley=args.scf_valley,
    )
    selected = {round(x, 10) for x in parse_float_list(args.selected_D_values)}
    summary_csv = args.out / "fixed_nu_Dscan_summary.csv"
    done = load_completed_D_values(summary_csv) if args.resume else set()
    records = []
    if args.resume and summary_csv.exists():
        records.extend(pd.read_csv(summary_csv).to_dict("records"))
    write_json(args.out / "fixed_nu_Dscan_metadata.json", vars(args))
    for index, point in enumerate(points, start=1):
        if round(point.D_Vnm, 10) in done:
            print(f"[{index}/{len(points)}] skip D={point.D_Vnm:.6g}")
            continue
        try:
            rec = run_point(point, args, save_detail=round(point.D_Vnm, 10) in selected)
        except Exception as exc:
            rec = {
                "D_index": int(point.D_index),
                "D_Vnm": float(point.D_Vnm),
                "n_cm2": float(point.n_cm2),
                "nu_total": float(point.nu_total),
                "chi_status": "error",
                "chi_error": repr(exc),
            }
        records.append(rec)
        pd.DataFrame(records).sort_values("D_index", kind="stable").to_csv(summary_csv, index=False)
        print(f"[{index}/{len(points)}] {rec['chi_status']} D={point.D_Vnm:.6g}")
    print("Saved:", summary_csv)


if __name__ == "__main__":
    main()

