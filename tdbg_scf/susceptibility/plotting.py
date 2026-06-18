from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MAP_SPECS: tuple[tuple[str, str], ...] = (
    ("finite_q_ratio_su4_diag", "Finite-Q ratio, SU4 diagonal"),
    ("finite_q_wins_su4_diag", "Finite-Q wins, SU4 diagonal"),
    ("lambda_Qstar_su4_diag", "Lambda at best nonzero Q, SU4 diagonal"),
    ("delta_lambda_Qstar_Gamma_su4_diag", "Lambda(Q*) - Lambda(Gamma), SU4 diagonal"),
    ("log10_finite_q_ratio_su4_diag_clipped", "log10 finite-Q ratio, SU4 diagonal, clipped"),
    ("robust_finite_q_su4_diag", "Robust finite-Q flag, SU4 diagonal"),
    ("gamma_lambda_su4_diag_selected", "Gamma lambda, SU4 diagonal"),
    ("qstar_nonzero_lambda_su4_diag_selected", "Best nonzero-Q lambda, SU4 diagonal"),
    ("qstar_su4_diag_qnorm_Ainv", "Best nonzero-Q |Q|, SU4 diagonal"),
    ("qstar_su4_diag_qx_Ainv", "Best nonzero-Q Qx, SU4 diagonal"),
    ("qstar_su4_diag_qy_Ainv", "Best nonzero-Q Qy, SU4 diagonal"),
    ("qstar_su4_diag_layer_dipole_overlap", "Best-Q layer dipole overlap, SU4 diagonal"),
    ("qstar_su4_diag_layer_dipole_ratio", "Best-Q layer dipole ratio, SU4 diagonal"),
    ("finite_q_ratio_su2_hund_factor2", "Finite-Q ratio, SU2 Hund factor 2"),
    ("finite_q_wins_su2_hund_factor2", "Finite-Q wins, SU2 Hund factor 2"),
    ("gamma_lambda_su2_hund_factor2_selected", "Gamma lambda, SU2 Hund factor 2"),
    ("qstar_nonzero_lambda_su2_hund_factor2_selected", "Best nonzero-Q lambda, SU2 Hund factor 2"),
    ("qstar_su2_hund_factor2_qnorm_Ainv", "Best nonzero-Q |Q|, SU2 Hund factor 2"),
    ("qstar_su2_hund_factor2_layer_dipole_overlap", "Best-Q layer dipole overlap, SU2 Hund factor 2"),
    ("qstar_su2_hund_factor2_layer_dipole_ratio", "Best-Q layer dipole ratio, SU2 Hund factor 2"),
    ("finite_q_lambda_ratio", "Finite-Q ratio, default model"),
    ("finite_q_wins", "Finite-Q wins, default model"),
    ("qstar_nonzero_norm_Ainv", "Best nonzero-Q |Q|, default model"),
    ("gamma_lambda_u_plus_hund", "Gamma lambda, default model"),
    ("qstar_nonzero_lambda_u_plus_hund", "Best nonzero-Q lambda, default model"),
    ("chi_runtime_s", "Runtime per point (s)"),
    ("scf_iterations", "SCF iterations"),
    ("scf_residual_meV", "SCF residual (meV)"),
)


def _finite_q_map_specs(plot_keys: Iterable[str] | None = None) -> list[tuple[str, str]]:
    if plot_keys is None:
        return list(DEFAULT_MAP_SPECS)
    titles = dict(DEFAULT_MAP_SPECS)
    return [(key, titles.get(key, key)) for key in plot_keys]


def _numeric_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(float)
    if values.dtype == object:
        lowered = values.astype(str).str.lower()
        if lowered.isin(["true", "false", "1", "0"]).all():
            return lowered.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    return pd.to_numeric(values, errors="coerce")


def add_derived_finite_q_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ratio_key = "finite_q_ratio_su4_diag"
    gamma_key = "gamma_lambda_su4_diag_selected"
    qstar_key = "qstar_nonzero_lambda_su4_diag_selected"
    if {ratio_key, gamma_key, qstar_key}.issubset(out.columns):
        ratio = pd.to_numeric(out[ratio_key], errors="coerce")
        gamma = pd.to_numeric(out[gamma_key], errors="coerce")
        qstar = pd.to_numeric(out[qstar_key], errors="coerce")
        out["lambda_Qstar_su4_diag"] = qstar
        out["delta_lambda_Qstar_Gamma_su4_diag"] = qstar - gamma
        positive_ratio = ratio.where(ratio > 0.0)
        out["log10_finite_q_ratio_su4_diag_clipped"] = np.log10(np.clip(positive_ratio, 0.1, 10.0))
        out["robust_finite_q_su4_diag"] = ((qstar > 1.0) & (ratio > 1.05)).astype(float)
    return out


def _axis_values(df: pd.DataFrame, index_col: str, value_col: str) -> list[float]:
    grouped = df[[index_col, value_col]].dropna().drop_duplicates()
    if grouped.empty:
        return []
    values = []
    for _, row in grouped.sort_values(index_col, kind="stable").iterrows():
        values.append(float(row[value_col]))
    return values


def _map_array(df: pd.DataFrame, key: str) -> tuple[np.ndarray, list[float], list[float]]:
    if {"n_index", "D_index", "n_cm2", "D_Vnm"}.issubset(df.columns):
        n_values = _axis_values(df, "n_index", "n_cm2")
        d_values = _axis_values(df, "D_index", "D_Vnm")
        if not n_values or not d_values:
            raise ValueError("could not infer n/D axes")
        arr = np.full((len(n_values), len(d_values)), np.nan, dtype=float)
        for _, row in df.iterrows():
            ni = int(row["n_index"])
            Di = int(row["D_index"])
            if 0 <= ni < arr.shape[0] and 0 <= Di < arr.shape[1]:
                arr[ni, Di] = float(row[key])
        return arr, n_values, d_values

    if {"n_cm2", "D_Vnm"}.issubset(df.columns):
        n_values = sorted(float(x) for x in df["n_cm2"].dropna().unique())
        d_values = sorted(float(x) for x in df["D_Vnm"].dropna().unique())
        n_lookup = {value: i for i, value in enumerate(n_values)}
        d_lookup = {value: i for i, value in enumerate(d_values)}
        arr = np.full((len(n_values), len(d_values)), np.nan, dtype=float)
        for _, row in df.iterrows():
            arr[n_lookup[float(row["n_cm2"])], d_lookup[float(row["D_Vnm"])]] = float(row[key])
        return arr, n_values, d_values

    raise ValueError("CSV must contain n_cm2 and D_Vnm columns")


def plot_nd_map(df: pd.DataFrame, key: str, out_path: Path, title: str) -> bool:
    if key not in df.columns:
        return False

    plot_df = df.copy()
    if "chi_status" in plot_df.columns:
        plot_df = plot_df[plot_df["chi_status"] == "ok"].copy()
    if plot_df.empty:
        return False

    plot_df[key] = _numeric_series(plot_df[key])
    plot_df = plot_df.dropna(subset=[key])
    if plot_df.empty:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr, n_values, d_values = _map_array(plot_df, key)
    d_min, d_max = min(d_values), max(d_values)
    n_min, n_max = min(n_values) / 1.0e12, max(n_values) / 1.0e12
    if d_min == d_max:
        d_min -= 0.5
        d_max += 0.5
    if n_min == n_max:
        n_min -= 0.5
        n_max += 0.5

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[d_min, d_max, n_min, n_max],
        interpolation="nearest",
    )
    ax.set_xlabel("D (V/nm)")
    ax.set_ylabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def plot_finite_q_outputs(
    csv_path: Path,
    out_dir: Path | None = None,
    plot_keys: Iterable[str] | None = None,
) -> dict:
    csv_path = Path(csv_path)
    if out_dir is None:
        out_dir = csv_path.parent / "figures"
    out_dir = Path(out_dir)
    df = add_derived_finite_q_columns(pd.read_csv(csv_path))

    generated: list[str] = []
    skipped: list[str] = []
    for key, title in _finite_q_map_specs(plot_keys):
        out_path = out_dir / f"nd_map_{key}.png"
        if plot_nd_map(df, key, out_path, title):
            generated.append(str(out_path))
        else:
            skipped.append(key)

    return {
        "csv": str(csv_path),
        "figures_dir": str(out_dir),
        "row_count": int(len(df)),
        "generated_count": int(len(generated)),
        "generated": generated,
        "skipped": skipped,
    }


def _save_pdf_png(fig, out_base: Path) -> list[str]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".pdf", ".png"):
        path = out_base.with_suffix(suffix)
        fig.savefig(path, dpi=220)
        paths.append(str(path))
    return paths


def plot_fixed_nu_Dscan_overview(scan_csv: Path, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(scan_csv)
    if "chi_status" in df.columns:
        df = df[df["chi_status"] == "ok"].copy()
    df = df.sort_values("D_Vnm", kind="stable")
    out_dir = Path(out_dir)

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 7.0), sharex=True, constrained_layout=True)
    D = pd.to_numeric(df["D_Vnm"], errors="coerce")
    if "stoner_spin_polarization_norm" in df.columns:
        spin = pd.to_numeric(df["stoner_spin_polarization_norm"], errors="coerce")
    else:
        spin = pd.to_numeric(df.get("spin_polarization_norm", np.nan), errors="coerce")
    axes[0].plot(D, spin, marker="o", lw=1.1)
    axes[0].set_ylabel("spin pol.")

    if "DOS_active_flavor_EF" in df.columns:
        axes[1].plot(D, pd.to_numeric(df["DOS_active_flavor_EF"], errors="coerce"), marker="o", lw=1.1)
    axes[1].set_ylabel("active DOS(EF)")

    if "finite_q_delta_normalized" in df.columns:
        axes[2].plot(D, pd.to_numeric(df["finite_q_delta_normalized"], errors="coerce"), marker="o", lw=1.1)
    axes[2].axhline(0.0, color="0.5", lw=0.8, ls="--")
    axes[2].set_ylabel(r"$\Delta_{\rm fQ}$")
    axes[2].set_xlabel("D (V/nm)")
    paths = _save_pdf_png(fig, out_dir / "fig_fixed_nu_Dscan_overview")
    plt.close(fig)
    return {"figure": "fig_fixed_nu_Dscan_overview", "paths": paths}


def plot_representative_qmaps(points_root: Path, representative_D: Iterable[float], out_dir: Path, key: str = "lambda_su4_diag_soft") -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .dscan import D_point_label

    reps = [float(x) for x in representative_D]
    if not reps:
        return {"figure": "fig_representative_Q_maps", "paths": [], "skipped": "no representative D values"}
    fig, axes = plt.subplots(1, len(reps), figsize=(4.0 * len(reps), 3.4), squeeze=False, constrained_layout=True)
    generated = 0
    for ax, D in zip(axes.ravel(), reps):
        csv = Path(points_root) / D_point_label(D) / "susceptibility_qmap.csv"
        if not csv.exists():
            ax.set_axis_off()
            continue
        qdf = pd.read_csv(csv)
        if key not in qdf.columns:
            ax.set_axis_off()
            continue
        gamma = qdf[qdf.get("is_gamma", False).astype(bool)]
        gamma_value = float(gamma.iloc[0][key]) if len(gamma) else float(pd.to_numeric(qdf[key], errors="coerce").max())
        values = pd.to_numeric(qdf[key], errors="coerce") / gamma_value - 1.0 if abs(gamma_value) > 1e-14 else pd.to_numeric(qdf[key], errors="coerce")
        xkey = "qx_mbz_Ainv" if "qx_mbz_Ainv" in qdf.columns else "qx_Ainv"
        ykey = "qy_mbz_Ainv" if "qy_mbz_Ainv" in qdf.columns else "qy_Ainv"
        sc = ax.scatter(qdf[xkey], qdf[ykey], c=values, s=28, cmap="coolwarm")
        ax.scatter([0.0], [0.0], marker="+", color="k", s=60)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"D={D:.4g}")
        fig.colorbar(sc, ax=ax, shrink=0.8)
        generated += 1
    paths = _save_pdf_png(fig, Path(out_dir) / "fig_representative_Q_maps") if generated else []
    plt.close(fig)
    return {"figure": "fig_representative_Q_maps", "paths": paths}


def plot_q_linecuts(*args, **kwargs) -> dict:
    return {"figure": "fig_Q_linecuts_through_transition", "paths": [], "skipped": "linecut extraction not requested"}


def plot_vhs_spinflip_nesting(*args, **kwargs) -> dict:
    return {"figure": "fig_VHS_spinflip_nesting", "paths": [], "skipped": "representative nesting detail not requested"}


def plot_convergence_summary(*args, **kwargs) -> dict:
    return {"figure": "convergence_summary", "paths": [], "skipped": "no convergence CSV supplied"}
