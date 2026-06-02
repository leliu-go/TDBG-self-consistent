#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
for _path in (SCRIPT_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_full_band_stoner import (
    FLAVOR_NAMES,
    full_band_dos_summary,
    layer_dos_polarizations,
    save_detail_outputs,
)

from tdbg_scf.susceptibility import (
    SusceptibilityParams,
    q_results_to_dataframe,
    scan_q_for_state,
    solve_full_scf_stoner_state,
    summarize_q_scan,
)
from tdbg_scf.susceptibility.workflow import write_single_point_outputs
from tdbg_scf.filling import filling_to_density_cm2
from tdbg_scf.stoner.full_band import (
    explicit_flavor_dos_at_mu,
    layer_polarizations,
    stoner_flavor_layer_dos_at_mu,
    stoner_flavor_layer_fillings,
    stoner_mu_by_flavor,
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


def add_bool_optional_arg(ap: argparse.ArgumentParser, name: str, *, default: bool) -> None:
    dest = name.replace("-", "_")
    option = f"--{name}"
    no_option = f"--no-{name}"
    if hasattr(argparse, "BooleanOptionalAction"):
        ap.add_argument(option, action=argparse.BooleanOptionalAction, default=default)
        return
    group = ap.add_mutually_exclusive_group()
    group.add_argument(option, dest=dest, action="store_true")
    group.add_argument(no_option, dest=dest, action="store_false")
    ap.set_defaults(**{dest: bool(default)})


def add_stoner_detail_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dos-sigma-meV", type=float, default=1.0)
    ap.add_argument("--dos-emin-meV", type=float, default=-80.0)
    ap.add_argument("--dos-emax-meV", type=float, default=80.0)
    ap.add_argument("--dos-bins", type=int, default=240)
    ap.add_argument("--points-per-segment", type=int, default=60)
    ap.add_argument("--n-show-full", type=int, default=12)
    ap.add_argument("--contour-n1", type=int, default=0, help="0 uses the SCF grid for contour plots.")
    ap.add_argument("--contour-n2", type=int, default=0, help="0 uses the SCF grid for contour plots.")
    ap.add_argument("--contour-band-count", type=int, default=6)
    ap.add_argument("--no-stoner-detail-outputs", action="store_true")


def stoner_detail_row(state, args: argparse.Namespace) -> dict:
    scf = state.scf_result
    stoner = state.stoner_result
    mu_f = stoner_mu_by_flavor(stoner, state.tables)
    stoner_flavor_nu_layer = stoner_flavor_layer_fillings(
        state.evals_K_meV,
        state.evecs_K,
        state.evals_Kp_meV,
        state.evecs_Kp,
        state.weights,
        state.layer_masks,
        state.A_M_A2,
        stoner.nu_f,
    )
    stoner_nu_layer = np.sum(stoner_flavor_nu_layer, axis=0)
    stoner_flavor_n_layer_cm2 = np.asarray(
        [[filling_to_density_cm2(x, state.A_M_A2) for x in flavor_layer] for flavor_layer in stoner_flavor_nu_layer],
        dtype=float,
    )
    stoner_n_layer_cm2 = np.asarray([filling_to_density_cm2(x, state.A_M_A2) for x in stoner_nu_layer], dtype=float)
    stoner_flavor_layer_dos = stoner_flavor_layer_dos_at_mu(
        state.evals_K_meV,
        state.evecs_K,
        state.evals_Kp_meV,
        state.evecs_Kp,
        state.weights,
        state.layer_masks,
        mu_f,
        args.dos_sigma_meV,
    )
    stoner_layer_dos = np.sum(stoner_flavor_layer_dos, axis=0)
    n_layer_cm2 = np.asarray(scf.n_layer_cm2, dtype=float)
    full_dos = full_band_dos_summary(
        state.evals_K_meV,
        state.evals_Kp_meV,
        state.weights,
        n_cm2=state.n_cm2,
        projected_mu_meV=scf.mu_meV,
        sigma_meV=args.dos_sigma_meV,
        kBT_meV=args.kBT_meV,
    )
    meta = state.metadata()
    row = {
        "status": "ok",
        "label": "single_point",
        "scf_source": meta.get("scf_source", ""),
        "scf_reference_mode": meta.get("scf_reference_mode", ""),
        "n_index": 0,
        "D_index": 0,
        "n_cm2": float(state.n_cm2),
        "nu_total": float(state.nu_total),
        "D_Vnm": float(state.D_Vnm),
        "A_M_A2": float(state.A_M_A2),
        "scf_mu_meV": float(scf.mu_meV),
        "scf_converged": bool(scf.converged),
        "scf_residual_meV": float(scf.residual_meV),
        "scf_iterations": int(scf.iterations),
        "full_mu_meV": full_dos["full_mu_meV"],
        "full_converged": bool(scf.converged),
        "full_residual_meV": float(scf.residual_meV),
        "full_iterations": int(scf.iterations),
        "stoner_success": bool(stoner.success),
        "stoner_seed": stoner.seed_name,
        "stoner_energy_meV_per_cell": float(stoner.energy_meV_per_cell),
        "nu_K_up": float(stoner.nu_f[0]),
        "nu_Kp_up": float(stoner.nu_f[1]),
        "nu_K_down": float(stoner.nu_f[2]),
        "nu_Kp_down": float(stoner.nu_f[3]),
        "mu_K_up_meV": float(mu_f[0]),
        "mu_Kp_up_meV": float(mu_f[1]),
        "mu_K_down_meV": float(mu_f[2]),
        "mu_Kp_down_meV": float(mu_f[3]),
        "spin_polarization": stoner.spin_polarization,
        "spin_polarization_norm": stoner.spin_polarization_norm,
        "valley_polarization": stoner.valley_polarization,
        "valley_polarization_norm": stoner.valley_polarization_norm,
        "spin_valley_polarization": stoner.spin_valley_polarization,
        "spin_valley_polarization_norm": stoner.spin_valley_polarization_norm,
        "flavor_polarization": stoner.flavor_polarization,
        "dos_mu_scf": full_dos["dos_mu_full_at_full_mu"],
        "dos_mu_full_scf": full_dos["dos_mu_full_at_full_mu"],
        "dos_mu_full_at_full_mu": full_dos["dos_mu_full_at_full_mu"],
        "dos_mu_full_at_projected_mu": full_dos["dos_mu_full_at_projected_mu"],
        "dos_mu_stoner": explicit_flavor_dos_at_mu(
            state.evals_K_meV,
            state.evals_Kp_meV,
            state.weights,
            mu_f,
            args.dos_sigma_meV,
        ),
        "U1_meV": float(scf.U_meV[0]),
        "U2_meV": float(scf.U_meV[1]),
        "U3_meV": float(scf.U_meV[2]),
        "U4_meV": float(scf.U_meV[3]),
        "n1_cm2": float(n_layer_cm2[0]),
        "n2_cm2": float(n_layer_cm2[1]),
        "n3_cm2": float(n_layer_cm2[2]),
        "n4_cm2": float(n_layer_cm2[3]),
        "stoner_n1_cm2": float(stoner_n_layer_cm2[0]),
        "stoner_n2_cm2": float(stoner_n_layer_cm2[1]),
        "stoner_n3_cm2": float(stoner_n_layer_cm2[2]),
        "stoner_n4_cm2": float(stoner_n_layer_cm2[3]),
        "stoner_n_total_layer_cm2": float(np.sum(stoner_n_layer_cm2)),
        "stoner_layer_dos_L1": float(stoner_layer_dos[0]),
        "stoner_layer_dos_L2": float(stoner_layer_dos[1]),
        "stoner_layer_dos_L3": float(stoner_layer_dos[2]),
        "stoner_layer_dos_L4": float(stoner_layer_dos[3]),
    }
    row.update(layer_polarizations(n_layer_cm2))
    row.update({f"stoner_{key}": value for key, value in layer_polarizations(stoner_n_layer_cm2).items()})
    row.update(
        {
            "stoner_layer_dos_total": float(np.sum(stoner_layer_dos)),
            "stoner_layer_dos_top_bottom": float((stoner_layer_dos[0] + stoner_layer_dos[1]) - (stoner_layer_dos[2] + stoner_layer_dos[3])),
            "stoner_layer_dos_outer_inner": float((stoner_layer_dos[0] + stoner_layer_dos[3]) - (stoner_layer_dos[1] + stoner_layer_dos[2])),
            "stoner_layer_dos_dipole": float(1.5 * stoner_layer_dos[0] + 0.5 * stoner_layer_dos[1] - 0.5 * stoner_layer_dos[2] - 1.5 * stoner_layer_dos[3]),
        }
    )
    for flavor_index, flavor_name in enumerate(FLAVOR_NAMES):
        flavor_n = stoner_flavor_n_layer_cm2[flavor_index]
        flavor_dos = stoner_flavor_layer_dos[flavor_index]
        flavor_prefix = f"stoner_{flavor_name}"
        for layer_index in range(4):
            row[f"{flavor_prefix}_n{layer_index + 1}_cm2"] = float(flavor_n[layer_index])
            row[f"{flavor_prefix}_layer_dos_L{layer_index + 1}"] = float(flavor_dos[layer_index])
        for key, value in layer_polarizations(flavor_n).items():
            row[f"{flavor_prefix}_{key}"] = value
        row[f"{flavor_prefix}_layer_dos_total"] = float(np.sum(flavor_dos))
        for key, value in layer_dos_polarizations(flavor_dos).items():
            row[f"{flavor_prefix}_{key}"] = value
    return row


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
    ap.add_argument("--jdos-sigma-meV", type=float, default=1.0)
    ap.add_argument("--no-jdos", action="store_true")
    ap.add_argument("--main-vertex-model", choices=["su4_diag", "su2_hund_factor2"], default="su4_diag")
    add_bool_optional_arg(ap, "also-run-hund-factor2", default=True)
    ap.add_argument("--hund-transverse-factor", type=float, default=2.0)
    ap.add_argument("--occupation-mode", choices=["flavor_mu", "common_mu"], default="flavor_mu")
    ap.add_argument("--spin-flip-mode", choices=["plus", "minus", "both_pm"], default="both_pm")
    add_bool_optional_arg(ap, "legacy-diagnostics", default=True)
    ap.add_argument("--q-mode", choices=["folded_grid", "unfolded_diagonalize", "folded_grid_with_G_shift"], default="folded_grid")
    ap.add_argument("--q-stride", type=int, default=1)
    ap.add_argument("--max-abs-q-step", type=int, default=None)
    ap.add_argument("--no-layer-matrix", action="store_true")
    add_stoner_detail_args(ap)
    ap.add_argument("--out", type=Path, required=True)
    return ap


def save_jdos_qmap_figure(out: Path, qdf) -> None:
    key = "jdos_total_cell_meV_inv2"
    if key not in qdf.columns:
        return
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(qdf["qx_Ainv"], qdf["qy_Ainv"], c=qdf[key], s=36)
    if "is_gamma" in qdf.columns:
        nonzero = qdf[~qdf["is_gamma"].astype(bool)]
    else:
        nonzero = qdf
    if len(nonzero):
        imax = nonzero[key].astype(float).idxmax()
        ax.scatter([qdf.loc[imax, "qx_Ainv"]], [qdf.loc[imax, "qy_Ainv"]], marker="x", s=70, color="tab:red", lw=1.6)
    ax.set_xlabel(r"$Q_x$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$Q_y$ [$\AA^{-1}$]")
    ax.set_title("Fermi-surface JDOS / nesting function")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$N(Q)$ [cell meV$^{-2}$]")
    fig.savefig(out / "jdos_qmap.png", dpi=180)
    plt.close(fig)


def q_vector_from_summary(summary: dict, prefix: str) -> np.ndarray | None:
    qx = summary.get(f"qstar_{prefix}_qx_Ainv")
    qy = summary.get(f"qstar_{prefix}_qy_Ainv")
    if qx is None or qy is None:
        return None
    qvec = np.asarray([float(qx), float(qy)], dtype=float)
    if not np.all(np.isfinite(qvec)):
        return None
    return qvec


def contour_crosses_zero(z: np.ndarray) -> bool:
    vals = np.asarray(z, dtype=float)
    vals = vals[np.isfinite(vals)]
    return bool(vals.size and float(np.min(vals)) <= 0.0 <= float(np.max(vals)))


def plot_shifted_fermi_contour_panels(
    out: Path,
    filename: str,
    kpts: np.ndarray,
    panels: list[tuple[str, np.ndarray, float]],
    band_indices: np.ndarray,
    qvec_Ainv: np.ndarray,
    title: str,
) -> None:
    qvec = np.asarray(qvec_Ainv, dtype=float)
    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.5 * nrows), squeeze=False)
    x = np.asarray(kpts[:, 0], dtype=float)
    y = np.asarray(kpts[:, 1], dtype=float)
    shifts = [
        (np.zeros(2), "k", "-", "FS"),
        (qvec, "tab:red", "--", "FS + Q*"),
        (-qvec, "tab:blue", ":", "FS - Q*"),
    ]
    for ax, (name, evals, mu) in zip(axes.ravel(), panels):
        for band in band_indices:
            z = np.asarray(evals[:, int(band)], dtype=float) - float(mu)
            if not contour_crosses_zero(z):
                continue
            for shift, color, linestyle, _label in shifts:
                ax.tricontour(
                    x + float(shift[0]),
                    y + float(shift[1]),
                    z,
                    levels=[0.0],
                    colors=color,
                    linewidths=0.9,
                    linestyles=linestyle,
                )
        ax.quiver(
            [0.0],
            [0.0],
            [float(qvec[0])],
            [float(qvec[1])],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            color="0.25",
        )
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$k_x$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$k_y$ [$\AA^{-1}$]")
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=color, lw=1.1, linestyle=linestyle, label=label) for _shift, color, linestyle, label in shifts]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / filename, dpi=220)
    plt.close(fig)


def save_qstar_contour_overlays(out: Path, summary: dict, args: argparse.Namespace) -> None:
    before_path = out / "full_band_contours_before_stoner.npz"
    after_path = out / "full_band_contours_stoner_flavors.npz"
    if not before_path.exists() or not after_path.exists():
        return

    qvec_chi = q_vector_from_summary(summary, args.main_vertex_model)
    qvec_jdos = q_vector_from_summary(summary, "jdos_total")
    with np.load(before_path) as before:
        before_kpts = before["kpts"]
        before_band_indices = before["band_indices"]
        before_panels = [
            ("K before Stoner", before["evals_K_meV"], float(before["full_mu_meV"])),
            ("Kp before Stoner", before["evals_Kp_meV"], float(before["full_mu_meV"])),
        ]
        if qvec_chi is not None:
            plot_shifted_fermi_contour_panels(
                out,
                "fermi_contours_before_stoner_with_chi_Qstar.png",
                before_kpts,
                before_panels,
                before_band_indices,
                qvec_chi,
                f"Before Stoner contours shifted by {args.main_vertex_model} Q*",
            )

    with np.load(after_path) as after:
        after_kpts = after["kpts"]
        after_band_indices = after["band_indices"]
        mu_f = after["mu_f_meV"]
        after_panels = [
            ("K up after Stoner", after["evals_K_meV"], float(mu_f[0])),
            ("Kp up after Stoner", after["evals_Kp_meV"], float(mu_f[1])),
            ("K down after Stoner", after["evals_K_meV"], float(mu_f[2])),
            ("Kp down after Stoner", after["evals_Kp_meV"], float(mu_f[3])),
        ]
        if qvec_chi is not None:
            plot_shifted_fermi_contour_panels(
                out,
                "fermi_contours_after_stoner_with_chi_Qstar.png",
                after_kpts,
                after_panels,
                after_band_indices,
                qvec_chi,
                f"After Stoner contours shifted by {args.main_vertex_model} Q*",
            )
        if qvec_jdos is not None:
            plot_shifted_fermi_contour_panels(
                out,
                "fermi_contours_after_stoner_with_jdos_Qstar.png",
                after_kpts,
                after_panels,
                after_band_indices,
                qvec_jdos,
                "After Stoner contours shifted by JDOS Q*",
            )


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
        include_jdos=not args.no_jdos,
        jdos_sigma_meV=args.jdos_sigma_meV,
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
    save_jdos_qmap_figure(args.out, qdf)

    if not args.no_stoner_detail_outputs:
        row = stoner_detail_row(state, args)
        save_detail_outputs(
            args.out,
            row,
            args,
            state.ham_K,
            state.ham_Kp,
            state.kpts,
            state.weights,
            state.scf_result,
            state.evals_K_meV,
            state.evals_Kp_meV,
            state.evecs_K,
            state.evecs_Kp,
            state.stoner_result,
            state.tables,
        )
        save_qstar_contour_overlays(args.out, summary, args)

    print("Saved:", args.out)
    print(summary)


if __name__ == "__main__":
    main()
