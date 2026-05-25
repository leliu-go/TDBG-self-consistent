"""Single-flavor cumulative band tables for flavor minimization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlavorBandTable:
    """Sorted single-flavor band table."""

    nu_grid: np.ndarray
    energy_grid_meV: np.ndarray
    kinetic_grid_meV: np.ndarray
    nu_min: float
    nu_max: float

    def mu_of_nu(self, nu: float) -> float:
        value = float(nu)
        if value < self.nu_min - 1e-12 or value > self.nu_max + 1e-12:
            raise ValueError(f"nu={value} outside [{self.nu_min}, {self.nu_max}]")
        return float(np.interp(value, self.nu_grid, self.energy_grid_meV))

    def kinetic_of_nu(self, nu: float) -> float:
        value = float(nu)
        if value < self.nu_min - 1e-12 or value > self.nu_max + 1e-12:
            raise ValueError(f"nu={value} outside [{self.nu_min}, {self.nu_max}]")
        return float(np.interp(value, self.nu_grid, self.kinetic_grid_meV))


def build_flavor_band_table(
    energies_meV: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    energy_zero_meV: float = 0.0,
) -> FlavorBandTable:
    """Build cumulative filling and kinetic-energy tables for one flavor."""

    e = np.asarray(energies_meV, dtype=float)
    w = np.asarray(weights, dtype=float)
    if e.size == 0:
        raise ValueError("energies_meV must not be empty")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1D array")

    if e.ndim == 1:
        if e.shape[0] != w.shape[0]:
            raise ValueError("1D energies must have the same length as weights")
        state_e = e - float(energy_zero_meV)
        state_w = w
    elif e.shape[0] == w.shape[0]:
        state_e = e.reshape(-1) - float(energy_zero_meV)
        state_w = np.repeat(w, e.shape[1])
    elif e.shape[-1] == w.shape[0]:
        moved = np.moveaxis(e, -1, 0)
        state_e = moved.reshape(-1) - float(energy_zero_meV)
        state_w = np.repeat(w, int(np.prod(moved.shape[1:])))
    else:
        raise ValueError("energies shape must contain an axis matching weights")

    dnu = float(A_M_A2) * state_w
    if np.any(dnu < 0.0):
        raise ValueError("weights must be non-negative")

    order = np.argsort(state_e, kind="mergesort")
    es = state_e[order]
    ws = dnu[order]

    nu_grid = np.concatenate([[0.0], np.cumsum(ws)])
    kinetic_grid = np.concatenate([[0.0], np.cumsum(ws * es)])
    energy_grid = np.concatenate([[es[0]], es])

    return FlavorBandTable(
        nu_grid=nu_grid,
        energy_grid_meV=energy_grid,
        kinetic_grid_meV=kinetic_grid,
        nu_min=0.0,
        nu_max=float(nu_grid[-1]),
    )
