#!/usr/bin/env python3
"""Run the full, non-projected self-consistent Hartree calculation for ABBA TDBG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian, make_uniform_mbz_grid, FullSCFConfig, FullSCFSolver
from tdbg_scf.plotting import compute_full_bands_along_path, plot_bands, plot_layer_profile, save_dos_plot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1, help="Plane-wave square cutoff. Use 1 for quick tests, 2-4 for convergence.")
    ap.add_argument("--valley", type=int, default=1)
    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0, help="Sublattice onsite asymmetry in meV.")
    ap.add_argument("--n-cm2", type=float, default=0.0)
    ap.add_argument("--D-Vnm", type=float, default=0.0)
    ap.add_argument("--grid-n1", type=int, default=7)
    ap.add_argument("--grid-n2", type=int, default=7)
    ap.add_argument("--degeneracy", type=int, default=4)
    ap.add_argument("--kBT-meV", type=float, default=0.2)
    ap.add_argument("--eps-perp", type=float, default=4.0)
    ap.add_argument("--D-sign", type=float, default=-1.0, help="Default -1 matches uploaded TDBG D convention; +1 uses D=e(nb-nt)/(2eps0).")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--tol-meV", type=float, default=1e-4)
    ap.add_argument("--mixer", choices=["linear", "anderson"], default="anderson")
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--points-per-segment", type=int, default=60)
    ap.add_argument("--out", type=str, default="outputs/full_scf")
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
    cfg = FullSCFConfig(
        target_density_cm2=args.n_cm2,
        D_Vnm=args.D_Vnm,
        degeneracy=args.degeneracy,
        kBT_meV=args.kBT_meV,
        eps_perp=args.eps_perp,
        D_sign=args.D_sign,
        max_iter=args.max_iter,
        tol_meV=args.tol_meV,
        mixer=args.mixer,
        mixer_kwargs=mixer_kwargs,
        keep_eigensystem=True,
    )
    result = FullSCFSolver(ham, kpts, weights).solve(cfg)

    pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)
    pd.DataFrame({"layer": [1, 2, 3, 4], "U_meV": result.U_meV, "n_cm2": result.n_layer_cm2}).to_csv(out / "layers.csv", index=False)
    summary = {
        "mode": "full_scf",
        "theta_deg": args.theta_deg,
        "cutoff": args.cutoff,
        "dim": ham.dim,
        "nG": ham.siteN,
        "grid_n1": args.grid_n1,
        "grid_n2": args.grid_n2,
        "target_density_cm2": args.n_cm2,
        "D_Vnm": args.D_Vnm,
        "mu_meV": result.mu_meV,
        "U_meV": result.U_meV.tolist(),
        "n_layer_cm2": result.n_layer_cm2.tolist(),
        "converged": result.converged,
        "residual_meV": result.residual_meV,
        "iterations": result.iterations,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out / "full_scf_result.npz", U_meV=result.U_meV, mu_meV=result.mu_meV,
             n_layer_a2=result.n_layer_a2, evals=result.evals, weights=weights, kpts=kpts)

    dist, bands, ticks, labels = compute_full_bands_along_path(ham, result.U_meV, points_per_segment=args.points_per_segment)
    plot_bands(dist, bands, ticks, labels, mu_meV=result.mu_meV, n_show=12,
               title="Full TDBG SCF", out=str(out / "full_scf_bands.png"))
    plot_layer_profile(result.U_meV, result.n_layer_cm2, out=str(out / "layer_profile.png"))
    save_dos_plot(result.evals, weights, result.mu_meV, args.degeneracy, out=str(out / "dos.png"))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
