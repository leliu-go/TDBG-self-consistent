"""Render supplied band and contour weights without recomputing physics.

All energies are already relative to the stage's physical chemical potential.
The common color is the probability on the lower bilayer, layers 3 and 4.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


_FLAVORS = {
    "before_stoner": ("K", "Kp"),
    "after_stoner": ("K_up", "Kp_up", "K_down", "Kp_down"),
}
_LABELS = {
    "K": r"$K$", "Kp": r"$K^\prime$",
    "K_up": r"$K\uparrow$", "Kp_up": r"$K^\prime\uparrow$",
    "K_down": r"$K\downarrow$", "Kp_down": r"$K^\prime\downarrow$",
    "Gamma": r"$\Gamma$", "Kprime": r"$K^\prime$", "M": r"$M$",
}
_ENERGY_WINDOW = (-100.0, 100.0)
_OUTPUT_NAMES = ("block_resolved_bands_and_contours.pdf",
                 "block_resolved_bands_and_contours.png")


def _probabilities(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1e-7) or np.any(values > 1.0 + 1e-7):
        raise ValueError("lower_block probabilities must be finite and lie in [0, 1]")
    # Remove only small numerical excursions; never alter the supplied arrays.
    return np.clip(values, 0.0, 1.0)


def _band_segments(data):
    dist = np.asarray(data["dist"], dtype=float)
    energies = np.asarray(data["energies_meV"], dtype=float)
    weights = _probabilities(data["weights_lower_block"])
    ticks = np.asarray(data["ticks"])
    labels = list(data["labels"])
    if (energies.ndim != 2 or dist.shape != (len(energies),) or len(dist) < 2
            or weights.shape != energies.shape or ticks.shape != (4,) or len(labels) != 4):
        raise ValueError("bands require matching (Nk, Nb) energies/weights, distances and four ticks/labels")
    if (not np.isfinite(energies).all() or not np.isfinite(dist).all()
            or np.any(np.diff(dist) < 0) or dist[-1] <= dist[0]):
        raise ValueError("band energies must be finite and path distances must increase")
    if not np.isfinite(ticks).all() or np.any(ticks != ticks.astype(int)):
        raise ValueError("symmetry ticks must be integers")
    ticks = ticks.astype(int)
    if np.any(ticks < 0) or np.any(ticks >= len(dist)):
        raise ValueError("symmetry ticks must index the supplied band path")

    # Band-major order retains every band with a segment in the plotted window.
    xy = np.stack([np.broadcast_to(dist, energies.T.shape), energies.T], axis=-1)
    segments = np.stack([xy[:, :-1], xy[:, 1:]], axis=2).reshape(-1, 2, 2)
    colors = (0.5 * (weights[:-1] + weights[1:])).T.ravel()
    energy_ends = segments[:, :, 1]
    visible = ((energy_ends.min(axis=1) <= _ENERGY_WINDOW[1])
               & (energy_ends.max(axis=1) >= _ENERGY_WINDOW[0]))
    return dist, ticks, labels, segments[visible], colors[visible]


def _contour_segments(components):
    segments, colors = [], []
    for xy, weights in components:
        xy = np.asarray(xy, dtype=float)
        weights = _probabilities(weights)
        if xy.ndim != 2 or xy.shape[1] != 2 or weights.shape != (len(xy),):
            raise ValueError("each contour requires (N, 2) coordinates and N probabilities")
        if not np.isfinite(xy).all():
            raise ValueError("contour coordinates must be finite")
        if len(xy) >= 2:
            # Components are segmented before concatenation, so there is no bridge.
            segments.append(np.stack([xy[:-1], xy[1:]], axis=1))
            colors.append(0.5 * (weights[:-1] + weights[1:]))
    if not segments:
        return np.empty((0, 2, 2)), np.empty(0)
    return np.concatenate(segments), np.concatenate(colors)


def make_block_stage_figure(context, bands_by_name, lines_by_name, boundary, *, point_info=None):
    """Return a reusable Figure; the caller owns closing it after saving.

    ``bands_by_name`` contains ``dist``, ``energies_meV``,
    ``weights_lower_block``, ``ticks`` and ``labels`` for each valley/flavor.
    ``lines_by_name`` contains independent ``(xy, weights_lower_block)`` tuples.
    No spectra, weights or energies are read or computed here. In particular,
    ``context['mu_meV']`` labels the panels and is never subtracted again.
    """
    stage = context["stage"]
    if stage not in _FLAVORS:
        raise ValueError("block figures require before_stoner or after_stoner")
    names = _FLAVORS[stage]
    if set(bands_by_name) != set(names) or set(lines_by_name) - set(names):
        raise ValueError("band and contour flavor names must match the physical stage")
    mu = float(context["mu_meV"])
    theta = float(context["theta_deg"])
    if not np.isfinite(mu) or not np.isfinite(theta):
        raise ValueError("the stage chemical potential and twist angle must be finite")
    stacking = str(context["stacking_top_to_bottom"])
    upper, lower = stacking.split("-", 1)
    boundary = np.asarray(boundary, dtype=float)
    if (boundary.shape != (7, 2) or not np.isfinite(boundary).all()
            or not np.allclose(boundary[0], boundary[-1])):
        raise ValueError("the first MBZ boundary must be a finite closed (7, 2) hexagon")
    extent = 1.2 * float(np.max(np.abs(boundary)))
    if extent <= 0:
        raise ValueError("the first MBZ boundary must have positive extent")
    band_data = {name: _band_segments(bands_by_name[name]) for name in names}
    contour_data = {name: _contour_segments(lines_by_name.get(name, [])) for name in names}

    cmap = LinearSegmentedColormap.from_list(
        "lower_block_probability", [(0.0, "#2454a4"), (0.5, "#827889"), (1.0, "#bc2939")])
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    height = 4.0 * len(names) + 1.5
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": .65,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(len(names), 2, figsize=(8.3, height), squeeze=False)
        fig.subplots_adjust(left=.105, right=.965, top=1.0-1.20/height,
                            bottom=1.6/height, wspace=.34, hspace=.35)
        stage_title = "Before Stoner" if stage == "before_stoner" else "After Stoner"
        fig.suptitle(stage_title + "  |  " + stacking.replace("-", "–")
                     + r"  |  $\theta = " + format(theta, ".6g") + r"^\circ$",
                     y=1.0-.18/height, fontsize=14, va="top")
        info = []
        point_info = point_info or {}
        for key, label, unit in (("nu_total", r"$\nu_{\rm total}$", ""),
                                 ("D_Vnm", r"$D$", " V/nm"), ("cutoff", "$N$", "")):
            if point_info.get(key) is not None:
                info.append(label + " = " + format(float(point_info[key]), ".6g") + unit)
        if info:
            fig.text(.535, 1.0-.54/height, "    ".join(info), ha="center", va="top", fontsize=11)
        mu_title = r"$\mu = " + format(mu, ".6g") + "$ meV"

        for row, name in enumerate(names):
            band_ax, contour_ax = axes[row]
            band_ax.set_gid("bands_" + name)
            contour_ax.set_gid("fermi_" + name)
            dist, ticks, labels, segments, colors = band_data[name]
            for tick in ticks:
                band_ax.axvline(dist[tick], color=".85", linewidth=.55, zorder=0)
            band_ax.axhline(0.0, color=".7", linewidth=.65, linestyle="--", zorder=1)
            if len(segments):
                collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=.85, zorder=2)
                collection.set_array(colors)
                band_ax.add_collection(collection)
            band_ax.set_xlim(dist[0], dist[-1])
            band_ax.set_ylim(*_ENERGY_WINDOW)
            band_ax.set_xticks(dist[ticks])
            band_ax.set_xticklabels([_LABELS.get(str(label), str(label)) for label in labels])
            band_ax.set_ylabel(r"$E-\mu$ (meV)")
            band_ax.set_title(_LABELS[name] + " bands; " + mu_title, fontsize=11, pad=7)
            if row == len(names)-1:
                band_ax.set_xlabel("High-symmetry path")

            segments, colors = contour_data[name]
            if len(segments):
                collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.25, zorder=2)
                collection.set_array(colors)
                contour_ax.add_collection(collection)
            else:
                message = ("Contour unavailable: mesh below 2 x 2"
                           if context.get("contour_status") == "unavailable_insufficient_mesh"
                           else r"No $E-\mu=0$ contours")
                contour_ax.text(.5, .5, message, ha="center", va="center",
                                transform=contour_ax.transAxes, fontsize=10, color=".35")
            contour_ax.plot(boundary[:, 0], boundary[:, 1], color=".25", linewidth=.75, zorder=3)
            contour_ax.set_xlim(-extent, extent)
            contour_ax.set_ylim(-extent, extent)
            contour_ax.set_aspect("equal", adjustable="box")
            contour_ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)", labelpad=2)
            contour_ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)", labelpad=2)
            contour_ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
            contour_ax.set_title(_LABELS[name] + r" contours; " + mu_title, fontsize=11, pad=7)

        color_ax = fig.add_axes([.20, .95/height, .60, .14/height])
        color_ax.set_gid("block_weight_colorbar")
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                               orientation="horizontal", ticks=[0.0, .25, .5, .75, 1.0])
        colorbar.set_label(r"$w_{\rm lower}$: 0 = upper " + upper + "; 1 = lower " + lower, fontsize=11)
        fig.text(.105, .12/height,
                 "Color: probability on lower " + lower + " (layers 3–4). Raw path weights: full-basis projection.\n"
                 "Contour weights: C3 interpolated plotting field; components are drawn independently.",
                 fontsize=8.5, ha="left", va="bottom", linespacing=1.4, color=".25")
    return fig


def render_block_stage(folder, context, bands_by_name, lines_by_name, boundary, *, point_info=None):
    """Save the stage PDF/PNG and return figure records separately from TXT data."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig = make_block_stage_figure(context, bands_by_name, lines_by_name, boundary, point_info=point_info)
    try:
        records = []
        # PDF font embedding is selected at save time, after the build context.
        with plt.rc_context({"pdf.fonttype": 42}):
            for filename in _OUTPUT_NAMES:
                path = folder / filename
                fig.savefig(path, dpi=200, facecolor="white")
                records.append({"filename": filename, "bytes": path.stat().st_size})
        return records
    finally:
        plt.close(fig)
