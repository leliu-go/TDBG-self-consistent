"""Full-size scalar snapshots and a compact preview of available scan data."""
from __future__ import annotations

import time

import numpy as np

from .atomic_io import atomic_output
from .output_metadata import axis_edges as _axis_edges


CORE_FIELDS = (
    "dos_mu_scf", "dos_mu_stoner", "full_mu_meV", "stoner_energy_meV_per_cell",
    "P_top_bottom_cm2", "P_outer_inner_cm2", "P_dipole_cm2",
    "stoner_P_top_bottom_cm2", "stoner_P_outer_inner_cm2", "stoner_P_dipole_cm2",
    "spin_polarization", "valley_polarization", "spin_valley_polarization", "flavor_polarization",
    "stoner_flavor_nu_min", "stoner_flavor_nu_max", "stoner_filling_bound_hit",
    "stoner_artificial_bound_hit", "stoner_filling_bound_limited",
    *[f"stoner_{flavor}_filling_bound_hit" for flavor in ("K_up", "Kp_up", "K_down", "Kp_down")],
    "scf_residual_meV", "scf_iterations",
)
SCF_CORE_FIELDS = tuple(key for key in CORE_FIELDS if key.startswith(("gap_", "scf_", "P_"))
                        or key in ("dos_mu_scf", "full_mu_meV"))


def complete_rows(rows, n_values, d_values):
    points = {(int(row["n_index"]), int(row["D_index"])): row for row in rows}
    return [points.get((i, j), {"n_index": i, "D_index": j, "n_cm2": float(n),
            "D_Vnm": float(d), "status": "pending", "scf_status": "pending",
            "scf_converged": False, "stoner_success": False, "detail_saved": False})
            for i, n in enumerate(n_values) for j, d in enumerate(d_values)]


def scalar_grids(rows, shape, n_layers, excluded=()):
    fields = set(CORE_FIELDS)
    for layer in range(1, n_layers + 1):
        fields.update((f"U{layer}_meV", f"n{layer}_cm2", f"stoner_n{layer}_cm2",
                       f"scf_layer_dos_L{layer}", f"stoner_layer_dos_L{layer}"))
    excluded = set(excluded) | {"n_index", "D_index", "n_cm2", "D_Vnm", "runtime_s"}
    for row in rows:
        fields.update(key for key, value in row.items() if key not in excluded
                      and isinstance(value, (int, float, np.number))
                      and not isinstance(value, (bool, np.bool_)))
    fields.difference_update(excluded)
    grids = {key: np.full(shape, np.nan) for key in sorted(fields)}
    masks = {key: np.zeros(shape, dtype=bool) for key in
             ("point_finished", "point_ok", "stoner_valid", "stoner_dos_valid")}
    for row in rows:
        index = int(row["n_index"]), int(row["D_index"])
        masks["point_finished"][index] = row.get("status") in ("ok", "failed")
        valid = row.get("status") == "ok"
        masks["point_ok"][index] = valid
        masks["stoner_valid"][index] = valid and bool(row.get("stoner_success", False))
        masks["stoner_dos_valid"][index] = valid and bool(row.get("stoner_dos_valid", False))
        if valid:
            for key, grid in grids.items():
                value = row.get(key)
                if value is not None:
                    grid[index] = float(value)
    return {**grids, **masks}


class LivePreview:
    def __init__(self, out, n_values, d_values, interval_s):
        self.path = out / "nd_map_live.png"
        self.n, self.d = np.asarray(n_values), np.asarray(d_values)
        self.interval = float(interval_s)
        self.last = -float("inf")

    def update(self, rows, *, force=False):
        if self.interval == 0:
            return
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        import matplotlib.pyplot as plt
        panels = [
            ("dos_mu_scf", "Hartree DOS at mu", "scf_status"),
            ("dos_mu_stoner", "Stoner DOS at mu", "status"),
            ("spin_polarization", "Spin polarization", "status"),
            ("P_top_bottom_cm2", "Hartree layer polarization (cm^-2)", "scf_status"),
            ("stoner_P_top_bottom_cm2", "Stoner layer polarization (cm^-2)", "status"),
            ("valley_polarization", "Valley polarization", "status"),
        ]
        i_order, j_order = np.argsort(self.n), np.argsort(self.d)
        n_edges = _axis_edges(self.n[i_order] / 1e12, .1)
        d_edges = _axis_edges(self.d[j_order], .05)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        try:
            for ax, (key, title, status_key) in zip(axes.flat, panels):
                grid = np.full((len(self.n), len(self.d)), np.nan)
                for row in rows:
                    if row.get(status_key) == "ok" and row.get(key) is not None:
                        grid[int(row["n_index"]), int(row["D_index"])] = float(row[key])
                ax.set_facecolor("#eeeeee")
                im = ax.pcolormesh(d_edges, n_edges, np.ma.masked_invalid(grid[np.ix_(i_order, j_order)]), shading="flat")
                ax.set(title=title, xlabel="D (V/nm)", ylabel="n (10^12 cm^-2)")
                fig.colorbar(im, ax=ax)
            finished = sum(row.get("status") in ("ok", "failed") for row in rows)
            fig.suptitle(f"n-D scan: {finished}/{len(self.n) * len(self.d)} returned points; blank cells have no data")
            with atomic_output(self.path) as temporary:
                fig.savefig(temporary, format="png", dpi=140)
        finally:
            plt.close(fig)
        self.last = now
