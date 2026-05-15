#!/usr/bin/env python3
"""Run the projected eight-band self-consistent Hartree calculation for ABBA TDBG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from tdbg_scf import (
    TDBGParameters, TDBGContinuumHamiltonian, make_uniform_mbz_grid,
    ProjectedSCFConfig, ProjectedSCFSolver, FullSCFConfig,
)
from tdbg_scf.plotting import (
    compute_full_bands_along_path, compute_projected_bands_along_path,
    plot_bands, plot_layer_profile, save_dos_plot,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1, help="Plane-wave square cutoff. Use 1 for quick tests, 2-4 for convergence.")
    ap.add_argument("--valley", type=int, default=1)
    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0)
    ap.add_argument("--n-cm2", type=float, default=0.0e12)
    ap.add_argument("--D-Vnm", type=float, default=0.9)
    ap.add_argument("--grid-n1", type=int, default=24)
    ap.add_argument("--grid-n2", type=int, default=24)
    ap.add_argument("--n-active", type=int, default=8)
    ap.add_argument("--selection", choices=["closest", "contiguous", "overlap"], default="overlap")
    ap.add_argument("--projector-refreshes", type=int, default=0)
    ap.add_argument("--degeneracy", type=int, default=4)
    ap.add_argument("--kBT-meV", type=float, default=0.2)
    ap.add_argument("--eps-perp", type=float, default=4.0)
    ap.add_argument("--D-sign", type=float, default=-1.0, help="Default -1 matches uploaded TDBG D convention; +1 uses D=e(nb-nt)/(2eps0).")
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--tol-meV", type=float, default=1e-4)
    ap.add_argument("--mixer", choices=["linear", "anderson"], default="anderson")
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--reference-mode", choices=["full_scf", "supplied_U"], default="full_scf")
    ap.add_argument("--U-ref", type=float, nargs=4, default=None, help="Four layer potentials in meV if reference-mode=supplied_U")
    ap.add_argument("--points-per-segment", type=int, default=60)
    ap.add_argument("--out", type=str, default="outputs/projected_8band")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = TDBGParameters(
        theta_deg=args.theta_deg, cutoff=args.cutoff, valley=args.valley,
        omega_meV=args.omega, wAA=args.wAA, wAB=args.wAB, sublattice_Z_meV=args.Z,
    )
    ham = TDBGContinuumHamiltonian(params)
    kpts, weights = make_uniform_mbz_grid(ham.geom, args.grid_n1, args.grid_n2)
    mixer_kwargs = {"alpha": args.alpha} if args.mixer == "linear" else {"beta": 0.5, "memory": 6, "fallback_alpha": args.alpha}

    full_cfg = FullSCFConfig(
        target_density_cm2=args.n_cm2,
        D_Vnm=args.D_Vnm,
        degeneracy=args.degeneracy,
        kBT_meV=args.kBT_meV,
        eps_perp=args.eps_perp,
        D_sign=args.D_sign,
        max_iter=max(20, min(args.max_iter, 60)),
        tol_meV=max(args.tol_meV, 1e-4),
        mixer=args.mixer,
        mixer_kwargs=mixer_kwargs,
        keep_eigensystem=False,
    )
    cfg = ProjectedSCFConfig(
        target_density_cm2=args.n_cm2,
        D_Vnm=args.D_Vnm,
        n_active=args.n_active,
        degeneracy=args.degeneracy,
        kBT_meV=args.kBT_meV,
        eps_perp=args.eps_perp,
        D_sign=args.D_sign,
        max_iter=args.max_iter,
        tol_meV=args.tol_meV,
        mixer=args.mixer,
        mixer_kwargs=mixer_kwargs,
        selection=args.selection,
        grid_shape=(args.grid_n1, args.grid_n2),
        projector_refreshes=args.projector_refreshes,
        reference_mode=args.reference_mode,
        supplied_U_ref_meV=np.array(args.U_ref, dtype=float) if args.U_ref is not None else None,
        full_scf_config=full_cfg,
    )
    result = ProjectedSCFSolver(ham, kpts, weights, grid_shape=(args.grid_n1, args.grid_n2)).solve(cfg)

    pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)
    pd.DataFrame({"layer": [1, 2, 3, 4], "U_meV": result.U_meV, "n_cm2": result.n_layer_cm2}).to_csv(out / "layers.csv", index=False)
    summary = {
        "mode": "projected_8band_scf",
        "theta_deg": args.theta_deg,
        "cutoff": args.cutoff,
        "full_dim": ham.dim,
        "nG": ham.siteN,
        "n_active": args.n_active,
        "selection": args.selection,
        "grid_n1": args.grid_n1,
        "grid_n2": args.grid_n2,
        "target_density_cm2": args.n_cm2,
        "D_Vnm": args.D_Vnm,
        "mu_meV": result.mu_meV,
        "U_ref_meV": result.model.U_ref_meV.tolist(),
        "mu_ref_meV": result.model.mu_ref_meV,
        "U_scf_meV": result.U_meV.tolist(),
        "n_layer_cm2": result.n_layer_cm2.tolist(),
        "isolation_gap_meV": result.model.isolation_gap_meV,
        "selection_diagnostics": result.model.selection_diagnostics or {},
        "max_abs_delta_U_meV": float(np.max(np.abs(result.U_meV - result.model.U_ref_meV))),
        "converged": result.converged,
        "residual_meV": result.residual_meV,
        "iterations": result.iterations,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out / "projected_8band_model.npz",
             U_ref_meV=result.model.U_ref_meV,
             U_scf_meV=result.U_meV,
             mu_ref_meV=result.model.mu_ref_meV,
             mu_scf_meV=result.mu_meV,
             miniband_energies0_meV=result.model.energies0_meV,
             final_evals_meV=result.evals,
             layer_mats=result.model.layer_mats,
             remote_density_per_k_layer=result.model.remote_density_per_k_layer,
             selected_indices=result.model.selected_indices,
             weights=weights,
             kpts=kpts)

    # Plot full-Hamiltonian comparisons plus the projected model at the final SCF potential.
    dist, ref_bands, ticks, labels = compute_full_bands_along_path(ham, result.model.U_ref_meV, points_per_segment=args.points_per_segment)
    plot_bands(dist, ref_bands, ticks, labels, mu_meV=result.model.mu_ref_meV, n_show=12,
               title="Full TDBG reference", out=str(out / "reference_full_bands.png"))
    dist, final_full_bands, ticks, labels = compute_full_bands_along_path(ham, result.U_meV, points_per_segment=args.points_per_segment)
    plot_bands(dist, final_full_bands, ticks, labels, mu_meV=result.mu_meV, n_show=12,
               title="Full Hamiltonian at projected-SCF U", out=str(out / "final_full_bands_at_projected_U.png"))
    dist, projected_bands, ticks, labels = compute_projected_bands_along_path(
        ham,
        U_ref_meV=result.model.U_ref_meV,
        U_meV=result.U_meV,
        mu_ref_meV=result.model.mu_ref_meV,
        n_active=args.n_active,
        selection=args.selection,
        points_per_segment=args.points_per_segment,
    )
    plot_bands(dist, projected_bands, ticks, labels, mu_meV=result.mu_meV, n_show=args.n_active,
               title=f"Projected {args.n_active}-band at projected-SCF U", out=str(out / "projected_8band_bands.png"))
    plot_layer_profile(result.U_meV, result.n_layer_cm2, out=str(out / "layer_profile.png"))
    save_dos_plot(result.evals, weights, result.mu_meV, args.degeneracy, out=str(out / "projected_dos.png"))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
