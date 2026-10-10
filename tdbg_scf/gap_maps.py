"""Portable n-D gap grids and figures shared by both Hartree workflows."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .band_gaps import (
    GAP_DEFINITION, GAP_FILLINGS, GAP_SPECTRUM_SOURCE,
    STONER_GAP_FILLINGS, STONER_GAP_NUMERIC_KEYS, stoner_gap_analysis_metadata,
)
from .output_metadata import displacement_metadata
from .output_metadata import hamiltonian_npz_metadata, hamiltonian_metadata
from .scan_axes import density_axis_metadata
from .atomic_io import atomic_npz, atomic_output


def _axis_edges(values: np.ndarray, single_half_width: float) -> np.ndarray:
    if values.size == 1:
        return np.array([values[0] - single_half_width, values[0] + single_half_width])
    midpoints = (values[1:] + values[:-1]) / 2
    return np.r_[values[0] - (values[1] - values[0]) / 2,
                 midpoints, values[-1] + (values[-1] - values[-2]) / 2]


def save_energy_gap_maps(out: Path, rows: list[dict], n_values: list[float],
                         d_values: list[float], ham, D_sign: float = -1.0, *, plot_figures=True) -> None:
    """Save complete grids; physical overlap is 0, failed/missing points NaN.

    All energy arrays have shape (n_count, D_count), in the supplied axis
    order. Figures sort only their display axes and use actual coordinates,
    including nonuniform grids. SCF convergence is recorded separately from
    successful spectral analysis, retaining the existing diagnostic policy.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    axes = [np.asarray(n_values, dtype=float), np.asarray(d_values, dtype=float)]
    for values in axes:
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Gap map axes must be nonempty finite 1-D arrays")
        if np.unique(values).size != values.size:
            raise ValueError("Gap map axes must have unique values")
    n_axis, d_axis = axes
    shape = (n_axis.size, d_axis.size)
    numeric_keys = [key for label in GAP_FILLINGS for key in (
        f"gap_{label}_meV", f"indirect_gap_{label}_meV",
        f"gap_{label}_lower_edge_meV", f"gap_{label}_upper_edge_meV",
    )]
    grids = {key: np.full(shape, np.nan) for key in numeric_keys}
    valid = np.zeros(shape, dtype=bool)
    converged = np.zeros(shape, dtype=bool)
    k_counts = np.full(shape, -1, dtype=int)
    for row in rows:
        if row.get("status") != "ok":
            continue
        i, j = int(row["n_index"]), int(row["D_index"])
        if row.get("D_sign", D_sign) != D_sign:
            raise ValueError("Point D_sign differs from the gap map convention")
        converged[i, j] = bool(row.get("scf_converged", row.get("converged", False)))
        # Old accepted Hartree checkpoints can predate gap analysis. Preserve
        # their valid Hartree stage while representing absent gaps as no data.
        if not all(key in row for key in numeric_keys):
            continue
        for key in numeric_keys:
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"Successful point has a nonfinite {key}")
            grids[key][i, j] = value
        valid[i, j] = True
        converged[i, j] = bool(row.get("scf_converged", row.get("converged", False)))
        k_counts[i, j] = int(row.get("gap_kpoint_count", -1))

    direction = displacement_metadata(D_sign)
    atomic_npz(
        out / "energy_gap_maps.npz",
        **hamiltonian_npz_metadata(ham), **direction, **grids,
        **density_axis_metadata(n_axis, ham.geom), D_values_Vnm=d_axis,
        gap_valid=valid, scf_converged=converged, gap_kpoint_count=k_counts,
        gap_spectrum_source=GAP_SPECTRUM_SOURCE, gap_definition=GAP_DEFINITION,
        gap_sampling="finite_2D_MBZ_grid", gap_spin_valley_degeneracy=4,
    )
    if not plot_figures:
        return

    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    i_order, j_order = np.argsort(n_axis), np.argsort(d_axis)
    n_edges = _axis_edges(n_axis[i_order] / 1e12, .1)
    d_edges = _axis_edges(d_axis[j_order], .05)
    displayed = [grids[f"gap_{label}_meV"] for label in GAP_FILLINGS]
    vmax = max((float(grid[valid].max()) for grid in displayed if valid.any()), default=0)
    vmax = vmax if vmax > 0 else 1.0
    direction_text = direction["D_positive_direction"].replace("_", " ")
    unconverged = int(np.count_nonzero(valid & ~converged))
    for label, nu in GAP_FILLINGS.items():
        fig, ax = plt.subplots(figsize=(6.2, 4.7), layout="constrained")
        grid = grids[f"gap_{label}_meV"][np.ix_(i_order, j_order)]
        im = ax.pcolormesh(d_edges, n_edges, grid, shading="flat", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel(f"D (V/nm; positive: {direction_text})")
        ax.set_ylabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
        nu_text = f"+{nu}" if nu > 0 else str(nu)
        ax.set_title(rf"Hartree indirect gap at $\nu={nu_text}$" + "\n" + hamiltonian_metadata(ham)['stacking_top_to_bottom']
                     + f" (physical {hamiltonian_npz_metadata(ham)['stacking_top_to_bottom']})")
        fig.colorbar(im, ax=ax, label="Gap (meV)")
        note = "Finite MBZ mesh at final SCF U; before Stoner"
        if unconverged:
            note += f"; {unconverged} SCF points unconverged"
        ax.text(.5, -.23, note, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        with atomic_output(fig_dir / f"nd_map_gap_{label}_meV.png") as temporary:
            fig.savefig(temporary, format="png", dpi=220)
        plt.close(fig)


def save_stoner_energy_gap_maps(out: Path, rows: list[dict], n_values: list[float],
                                d_values: list[float], ham, D_sign: float = -1.0, *,
                                plot_figures=True) -> None:
    """Save post-Stoner integer reference gaps, masking unavailable solutions.

    Reference gaps use each (n,D) point's own frozen Hartree U and Stoner
    self-energies. Equilibrium gaps at nu_ref are read at the matching scan
    filling; gaps elsewhere do not substitute for re-solving at nu_ref.
    """
    import json

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    axes = [np.asarray(n_values, dtype=float), np.asarray(d_values, dtype=float)]
    for values in axes:
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Stoner gap map axes must be nonempty finite 1-D arrays")
        if np.unique(values).size != values.size:
            raise ValueError("Stoner gap map axes must have unique values")
    n_axis, d_axis = axes
    shape = (n_axis.size, d_axis.size)
    grids = {key: np.full(shape, np.nan) for key in STONER_GAP_NUMERIC_KEYS}
    valid = np.zeros(shape, dtype=bool)
    converged = np.zeros(shape, dtype=bool)
    k_counts = np.full(shape, -1, dtype=int)
    for row in rows:
        if row.get("status") != "ok" or row.get("stoner_gap_valid", False) != True:
            continue
        i, j = int(row["n_index"]), int(row["D_index"])
        if not (0 <= i < shape[0] and 0 <= j < shape[1]):
            raise ValueError("Stoner gap point index is outside the map axes")
        if row.get("D_sign", D_sign) != D_sign:
            raise ValueError("Point D_sign differs from the Stoner gap map convention")
        for key, grid in grids.items():
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"Valid Stoner gap point has a nonfinite {key}")
            grid[i, j] = value
        valid[i, j] = True
        converged[i, j] = bool(row.get("scf_converged", False))
        k_counts[i, j] = int(row.get("stoner_gap_kpoint_count", -1))
    metadata = stoner_gap_analysis_metadata()
    metadata["stoner_gap_filling_labels_json"] = json.dumps(metadata.pop("stoner_gap_filling_labels"), sort_keys=True)
    direction = displacement_metadata(D_sign)
    atomic_npz(out / "stoner_energy_gap_maps.npz", **hamiltonian_npz_metadata(ham),
               **direction, **grids, **density_axis_metadata(n_axis, ham.geom),
               D_values_Vnm=d_axis, stoner_gap_valid=valid, scf_converged=converged,
               stoner_gap_kpoint_count=k_counts, **metadata)
    if not plot_figures:
        return

    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    i_order, j_order = np.argsort(n_axis), np.argsort(d_axis)
    n_edges = _axis_edges(n_axis[i_order] / 1e12, .1)
    d_edges = _axis_edges(d_axis[j_order], .05)
    vmax = max((float(grids[f"stoner_gap_{label}_meV"][valid].max())
                for label in STONER_GAP_FILLINGS if valid.any()), default=0.) or 1.
    for label, nu in STONER_GAP_FILLINGS.items():
        fig, ax = plt.subplots(figsize=(6.2, 4.7), layout="constrained")
        grid = grids[f"stoner_gap_{label}_meV"][np.ix_(i_order, j_order)]
        ax.set_facecolor("#eeeeee")
        im = ax.pcolormesh(d_edges, n_edges, np.ma.masked_invalid(grid), shading="flat",
                           cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("D (V/nm; positive: " + direction["D_positive_direction"].replace("_", " ") + ")")
        ax.set_ylabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
        ax.set_title(rf"Stoner spectrum indirect gap at $\nu_{{ref}}={nu:+d}$" + "\n" + hamiltonian_metadata(ham)['stacking_top_to_bottom'])
        fig.colorbar(im, ax=ax, label="Gap (meV)")
        ax.text(.5, -.23, "Frozen U and self-energies of each scan point; finite MBZ mesh\n"
                "Equilibrium integer gap: use the row at that same filling", transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
        with atomic_output(fig_dir / f"nd_map_stoner_gap_{label}_meV.png") as temporary:
            fig.savefig(temporary, format="png", dpi=220)
        plt.close(fig)
