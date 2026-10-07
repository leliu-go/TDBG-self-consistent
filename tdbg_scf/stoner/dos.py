"""Single-flavor cumulative band tables for flavor minimization."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class FlavorBandTable:
    """Sorted single-flavor band table."""

    nu_grid: np.ndarray
    energy_grid_meV: np.ndarray
    kinetic_grid_meV: np.ndarray
    nu_min: float
    nu_max: float

    def __post_init__(self) -> None:
        self.nu_grid = np.asarray(self.nu_grid, dtype=float)
        self.energy_grid_meV = np.asarray(self.energy_grid_meV, dtype=float)
        self.kinetic_grid_meV = np.asarray(self.kinetic_grid_meV, dtype=float)
        if (self.nu_grid.ndim != 1 or self.nu_grid.size < 2
                or self.energy_grid_meV.shape != self.nu_grid.shape
                or self.kinetic_grid_meV.shape != self.nu_grid.shape
                or not np.isfinite([self.nu_min, self.nu_max]).all()
                or not np.isfinite(self.nu_grid).all()
                or not np.isfinite(self.energy_grid_meV).all()
                or not np.isfinite(self.kinetic_grid_meV).all()
                or np.any(np.diff(self.nu_grid) <= 0)
                or self.nu_min > self.nu_max
                or self.nu_min < self.nu_grid[0] - 1e-10
                or self.nu_max > self.nu_grid[-1] + 1e-10):
            raise ValueError("band table needs finite arrays, increasing filling nodes, and valid bounds")

    def mu_of_nu(self, nu: float) -> float:
        """T=0 kinetic derivative; use the gap midpoint at a filling node.

        Within an interval the partially occupied state's energy is the
        slope of cumulative kinetic energy, not an interpolation between
        two different eigenvalues. At a node any chemical potential between
        the adjacent slopes is allowed; the midpoint fixes that convention.
        """
        lo, hi = self.mu_interval_of_nu(nu)
        if np.isneginf(lo):
            return hi
        if np.isposinf(hi):
            return lo
        return float(.5 * (lo + hi))

    def mu_interval_of_nu(self, nu: float, node_tolerance: float = 0.) -> tuple[float, float]:
        """Physical T=0 subgradient of K, before optimization-window constraints.

        A partly occupied level fixes mu; at a filled-state node any mu in
        the neighboring gap produces that filling. Restricted optimization
        bounds must not extend this physical interval into a remote band.
        """
        value = float(nu)
        if not np.isfinite(value) or value < self.nu_min - 1e-12 or value > self.nu_max + 1e-12:
            raise ValueError(f"nu={value} outside [{self.nu_min}, {self.nu_max}]")
        if not np.isfinite(node_tolerance) or node_tolerance < 0:
            raise ValueError("node_tolerance must be finite and non-negative")
        value = float(np.clip(value, self.nu_min, self.nu_max))
        index = int(np.searchsorted(self.nu_grid, value, side="left"))
        def slope(interval):
            return float((self.kinetic_grid_meV[interval+1]-self.kinetic_grid_meV[interval])
                         /(self.nu_grid[interval+1]-self.nu_grid[interval]))
        # Cumulative sums on large meshes must not turn an exact neutrality
        # node into a partially filled valence/conduction interval.
        roundoff = 32*np.finfo(float).eps*max(1.,abs(self.nu_grid[0]),abs(self.nu_grid[-1]))
        for node in (index, index - 1):
            if not 0 <= node < len(self.nu_grid):
                continue
            adjacent = [self.nu_grid[i+1]-self.nu_grid[i]
                        for i in range(max(0,node-1),min(len(self.nu_grid)-1,node+1))]
            tolerance = min(max(roundoff, node_tolerance), .01 * float(np.min(adjacent)))
            if abs(value - self.nu_grid[node]) <= tolerance:
                if node == 0:
                    return -float("inf"), slope(0)
                if node == len(self.nu_grid)-1:
                    return slope(node-1), float("inf")
                left, right = slope(node-1), slope(node)
                return min(left,right), max(left,right)
        energy = slope(max(0,min(index-1,len(self.nu_grid)-2)))
        return energy, energy

    def kinetic_of_nu(self, nu: float) -> float:
        value = float(nu)
        if not np.isfinite(value) or value < self.nu_min - 1e-12 or value > self.nu_max + 1e-12:
            raise ValueError(f"nu={value} outside [{self.nu_min}, {self.nu_max}]")
        return float(np.interp(value, self.nu_grid, self.kinetic_grid_meV))


def flavor_filling_bounds(tables, nu_min: float, nu_max: float) -> list[tuple[float, float]]:
    """Intersect a finite per-flavor search window with spectral capacity.

    Do not truncate the kinetic tables: common-mu validation still needs
    the physical states on both sides of an artificial search boundary.
    """
    if len(tables) != 4:
        raise ValueError("Need exactly four flavor DOS tables")
    if not np.isfinite([nu_min, nu_max]).all() or nu_min >= nu_max:
        raise ValueError("flavor filling bounds must be finite and ordered")
    bounds = [(max(float(table.nu_min), float(nu_min)),
               min(float(table.nu_max), float(nu_max))) for table in tables]
    if any(lo > hi for lo, hi in bounds):
        raise ValueError("flavor filling bounds have no available band capacity")
    return bounds


def _sorted_weighted_states(energies_meV, weights, A_M_A2, energy_zero_meV=0.0):
    """Validate a real spectrum/measure and discard zero-capacity states."""
    if np.iscomplexobj(energies_meV):
        raise ValueError("energies_meV must be real")
    e = np.asarray(energies_meV, dtype=float)
    w = np.asarray(weights, dtype=float)
    if e.size == 0 or not np.isfinite(e).all():
        raise ValueError("energies_meV must be non-empty and finite")
    if (w.ndim != 1 or w.size == 0 or not np.isfinite(w).all()
            or np.any(w < 0) or not np.any(w > 0)):
        raise ValueError("weights must be finite, non-negative, and include a positive weight")
    if not np.isfinite(A_M_A2) or A_M_A2 <= 0 or not np.isfinite(energy_zero_meV):
        raise ValueError("A_M_A2 must be finite and positive; energy_zero_meV must be finite")

    if e.ndim == 1:
        if e.shape[0] != w.shape[0]:
            raise ValueError("1D energies must have the same length as weights")
        state_e = e - float(energy_zero_meV)
        state_w = w
    elif e.ndim >= 2 and e.shape[0] == w.shape[0]:
        state_e = e.reshape(-1) - float(energy_zero_meV)
        state_w = np.repeat(w, int(np.prod(e.shape[1:])))
    elif e.ndim >= 2 and e.shape[-1] == w.shape[0]:
        moved = np.moveaxis(e, -1, 0)
        state_e = moved.reshape(-1) - float(energy_zero_meV)
        state_w = np.repeat(w, int(np.prod(moved.shape[1:])))
    else:
        raise ValueError("energies shape must contain an axis matching weights")

    dnu = float(A_M_A2) * state_w
    positive = dnu > 0
    if not np.any(positive):
        raise ValueError("spectrum must have positive state capacity")
    state_e, dnu = state_e[positive], dnu[positive]
    order = np.argsort(state_e, kind="mergesort")
    es = state_e[order]
    return es, dnu[order]


def _stable_cumsum(values):
    """Compensated block totals, without relying on Windows longdouble."""
    values = np.asarray(values, dtype=float)
    result = np.empty(values.size, dtype=float)
    carry, correction = 0., 0.
    for start in range(0, values.size, 256):
        stop = min(start+256, values.size)
        block = values[start:stop]
        result[start:stop] = carry + np.cumsum(block)
        increment = math.fsum(block) - correction
        new_carry = carry + increment
        correction = (new_carry-carry) - increment
        carry = new_carry
        result[stop-1] = carry
    return result


def _cumulative_integrals(es, ws):
    # Uniform meshes have exact integer state counts; repeated floating-point
    # additions must not move the neutrality node into a neighboring band.
    cumulative_weight = (ws[0]*np.arange(1,len(ws)+1,dtype=float)
                         if np.all(ws == ws[0]) else _stable_cumsum(ws))
    nu = np.r_[0., cumulative_weight]
    kinetic = np.r_[0., _stable_cumsum(ws*es)]
    return nu, kinetic


def build_flavor_band_table(
    energies_meV: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    energy_zero_meV: float = 0.0,
) -> FlavorBandTable:
    """Build cumulative filling and kinetic-energy tables for one flavor."""
    es, ws = _sorted_weighted_states(energies_meV, weights, A_M_A2, energy_zero_meV)
    nu, kinetic = _cumulative_integrals(es, ws)

    nu_grid = np.asarray(nu, dtype=float)
    kinetic_grid = np.asarray(kinetic, dtype=float)
    energy_grid = np.concatenate([[es[0]], es])

    return FlavorBandTable(
        nu_grid=nu_grid,
        energy_grid_meV=energy_grid,
        kinetic_grid_meV=kinetic_grid,
        nu_min=0.0,
        nu_max=float(nu_grid[-1]),
    )
