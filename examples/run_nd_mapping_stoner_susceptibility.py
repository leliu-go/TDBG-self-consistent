#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdbg_scf.susceptibility import (
    SusceptibilityParams,
    scan_q_for_state,
    solve_full_scf_stoner_state,
    summarize_q_scan,
)


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
    ap = argparse.ArgumentParser(description="Add finite-Q Stoner-after susceptibility diagnostics to an n-D map")
    ap.add_argument("--input-map", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nu-min", type=float, default=None)
    ap.add_argument("--nu-max", type=float, default=None)
    ap.add_argument("--D-min", type=float, default=None)
    ap.add_argument("--D-max", type=float, default=None)
    ap.add_argument("--spin-min", type=float, default=None, help="Filter by existing spin_polarization_norm")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--row-stride", type=int, default=1)
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
    ap.add_argument("--max-workers", type=str, default="auto", help="'auto' or an integer process count.")
    return ap


def filter_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.nu_min is not None:
        out = out[out["nu_total"] >= float(args.nu_min)]
    if args.nu_max is not None:
        out = out[out["nu_total"] <= float(args.nu_max)]
    if args.D_min is not None:
        out = out[out["D_Vnm"] >= float(args.D_min)]
    if args.D_max is not None:
        out = out[out["D_Vnm"] <= float(args.D_max)]
    if args.spin_min is not None and "spin_polarization_norm" in out.columns:
        out = out[out["spin_polarization_norm"] >= float(args.spin_min)]
    if int(args.row_stride) > 1:
        out = out.iloc[:: int(args.row_stride)]
    if args.max_points is not None:
        out = out.head(int(args.max_points))
    return out.reset_index(drop=True)


def row_key(row: pd.Series) -> tuple:
    if "n_index" in row and "D_index" in row:
        return (int(row["n_index"]), int(row["D_index"]))
    return (round(float(row["n_cm2"]), 3), round(float(row["D_Vnm"]), 6))


def resolve_workers(value: str) -> int:
    if str(value).lower() == "auto":
        return max(1, (os.cpu_count() or 2) - 2)
    workers = int(value)
    if workers < 1:
        raise ValueError("--max-workers must be >= 1 or auto")
    return workers


def run_chi_task(row_data: dict, args: argparse.Namespace) -> dict:
    t0 = time.time()
    n_cm2 = float(row_data["n_cm2"])
    D_Vnm = float(row_data["D_Vnm"])

    try:
        state = solve_full_scf_stoner_state(
            n_cm2=n_cm2,
            D_Vnm=D_Vnm,
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
        rec = dict(row_data)
        rec.update(state.metadata())
        rec.update(summarize_q_scan(q_results, finite_q_tol=chi_params.finite_q_tol))
        rec["chi_status"] = "ok"
        rec["chi_runtime_s"] = time.time() - t0
    except Exception as exc:
        rec = dict(row_data)
        rec["chi_status"] = "error"
        rec["chi_error"] = repr(exc)
        rec["chi_runtime_s"] = time.time() - t0
    return rec


def save_records(path: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    sort_cols = [c for c in ("n_index", "D_index") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")
    df.to_csv(path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input_map)
    todo = filter_rows(source, args)
    workers = resolve_workers(args.max_workers)

    done_keys = set()
    if args.resume and args.out_csv.exists():
        old = pd.read_csv(args.out_csv)
        for _, row in old.iterrows():
            done_keys.add(row_key(row))
        records = old.to_dict("records")
    else:
        records = []

    pending: list[dict] = []
    for count, row in todo.iterrows():
        key = row_key(row)
        if key in done_keys:
            print(f"skip existing {key}", flush=True)
            continue
        row_data = row.to_dict()
        row_data["_chi_input_order"] = int(count)
        pending.append(row_data)

    print(f"pending points: {len(pending)}; workers: {workers}", flush=True)
    if workers == 1:
        for index, row_data in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] n={float(row_data['n_cm2']):.6e} cm^-2 "
                f"D={float(row_data['D_Vnm']):.4f} V/nm",
                flush=True,
            )
            rec = run_chi_task(row_data, args)
            records.append(rec)
            save_records(args.out_csv, records)
            print(f"[{index}/{len(pending)}] {rec.get('chi_status')} runtime={rec.get('chi_runtime_s', 0.0):.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_chi_task, row_data, args) for row_data in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                rec = future.result()
                records.append(rec)
                save_records(args.out_csv, records)
                print(
                    f"[{index}/{len(pending)}] {rec.get('chi_status')} "
                    f"n={float(rec['n_cm2']):.6e} D={float(rec['D_Vnm']):.4f} "
                    f"runtime={rec.get('chi_runtime_s', 0.0):.1f}s",
                    flush=True,
                )

    print("saved", args.out_csv)


if __name__ == "__main__":
    main()
