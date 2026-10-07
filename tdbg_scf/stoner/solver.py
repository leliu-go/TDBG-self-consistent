"""Fixed-filling minimization for the four-flavor phenomenological model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dos import FlavorBandTable, flavor_filling_bounds
from .energy import (stoner_energy_meV_per_cell, stoner_stationarity_intervals,
                     stoner_local_stability, stoner_feasible_descent)
from .params import StonerParams


@dataclass
class StonerResult:
    nu_total: float
    nu_f: np.ndarray
    energy_meV_per_cell: float
    success: bool
    seed_name: str
    local_minima: list[dict]
    mu_f_meV: np.ndarray | None = None
    mean_field_fermi_level: dict | None = None
    flavor_bounds_nu: list[tuple[float, float]] | None = None

    def _spin_polarization_signed(self) -> float:
        return float((self.nu_f[0] + self.nu_f[1]) - (self.nu_f[2] + self.nu_f[3]))

    def _valley_polarization_signed(self) -> float:
        return float((self.nu_f[0] + self.nu_f[2]) - (self.nu_f[1] + self.nu_f[3]))

    def _spin_valley_polarization_signed(self) -> float:
        return float((self.nu_f[0] + self.nu_f[3]) - (self.nu_f[1] + self.nu_f[2]))

    @property
    def spin_polarization(self) -> float:
        return abs(self._spin_polarization_signed())

    @property
    def valley_polarization(self) -> float:
        return abs(self._valley_polarization_signed())

    @property
    def spin_valley_polarization(self) -> float:
        return abs(self._spin_valley_polarization_signed())

    def _normalize_by_total_filling(self, value: float) -> float:
        total = abs(float(self.nu_total))
        if abs(total) < 1e-12:
            return float("nan")
        return float(value) / total

    @property
    def spin_polarization_norm(self) -> float:
        return self._normalize_by_total_filling(self.spin_polarization)

    @property
    def valley_polarization_norm(self) -> float:
        return self._normalize_by_total_filling(self.valley_polarization)

    @property
    def spin_valley_polarization_norm(self) -> float:
        return self._normalize_by_total_filling(self.spin_valley_polarization)

    @property
    def flavor_polarization(self) -> float:
        return float(np.max(self.nu_f) - np.min(self.nu_f))


def _clip_to_bounds_and_total(x, bounds, total):
    values = np.asarray(x, dtype=float).copy()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    target = float(total)
    values = np.clip(values, lo, hi)

    for _ in range(64):
        residual = target - float(np.sum(values))
        if abs(residual) < 1e-12:
            break
        if residual > 0.0:
            room = hi - values
            mask = room > 1e-13
            if not np.any(mask):
                break
            share = residual * room / float(np.sum(room))
            values += np.where(mask, np.minimum(share, room), 0.0)
        else:
            room = values - lo
            mask = room > 1e-13
            if not np.any(mask):
                break
            share = (-residual) * room / float(np.sum(room))
            values -= np.where(mask, np.minimum(share, room), 0.0)
    return np.clip(values, lo, hi)


def _fill_order(total: float, bounds, order) -> np.ndarray:
    x = _clip_to_bounds_and_total(np.zeros(4), bounds, 0.0)
    remaining = float(total) - float(np.sum(x))
    for index in order:
        if abs(remaining) <= 1e-12:
            break
        if remaining > 0.0:
            room = float(bounds[index][1]) - x[index]
            delta = min(room, remaining)
        else:
            room = x[index] - float(bounds[index][0])
            delta = -min(room, -remaining)
        x[index] += delta
        remaining -= delta
    return _clip_to_bounds_and_total(x, bounds, total)


def make_stoner_seeds(nu_total: float, bounds, rng, n_random: int) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = []

    def add(name: str, raw) -> None:
        x = _clip_to_bounds_and_total(raw, bounds, nu_total)
        if abs(float(np.sum(x)) - float(nu_total)) < 1e-8:
            if not any(np.allclose(x, old, atol=1e-10, rtol=0.0) for _, old in seeds):
                seeds.append((name, x))

    add("paramagnetic", np.full(4, float(nu_total) / 4.0))
    add("spin_up", _fill_order(nu_total, bounds, [0, 1, 2, 3]))
    add("spin_down", _fill_order(nu_total, bounds, [2, 3, 0, 1]))
    add("valley_K", _fill_order(nu_total, bounds, [0, 2, 1, 3]))
    add("valley_Kp", _fill_order(nu_total, bounds, [1, 3, 0, 2]))
    for flavor in range(4):
        order = [flavor] + [i for i in range(4) if i != flavor]
        add(f"single_flavor_{flavor}", _fill_order(nu_total, bounds, order))

    # Same-sign starts all collapse to zero at neutrality and miss
    # compensated phases. Explicitly break spin, valley and spin-valley
    # symmetry in both directions, including fully polarized corners.
    center = _clip_to_bounds_and_total(np.full(4, float(nu_total)/4.), bounds, nu_total)
    patterns = (("spin", [1.,1.,-1.,-1.]), ("valley", [1.,-1.,1.,-1.]),
                ("spin_valley", [1.,-1.,-1.,1.]))
    full_scale = max(hi-lo for lo,hi in bounds)
    for scale in (.5, 1., full_scale):
        for name, pattern in patterns:
            for sign in (-1, 1):
                add(f"compensated_{name}_{sign}_{scale:g}", center+sign*scale*np.asarray(pattern))
    for flavor in range(4):
        pattern = np.full(4, -1./3.)
        pattern[flavor] = 1.
        for sign in (-1, 1):
            add(f"compensated_single_{flavor}_{sign}", center+sign*pattern)

    lower, upper = np.asarray(bounds).T
    for index in range(int(n_random)):
        add(f"random_{index}", rng.uniform(lower, upper))
    return seeds


def solve_stoner_fixed_nu(
    nu_total: float,
    tables: list[FlavorBandTable],
    params: StonerParams,
    A_M_A2: float,
) -> StonerResult:
    if len(tables) != 4:
        raise ValueError("Need exactly four flavor DOS tables")
    if not np.isfinite(nu_total) or not np.isfinite(A_M_A2) or A_M_A2 <= 0:
        raise ValueError("nu_total must be finite and A_M_A2 finite and positive")

    bounds = flavor_filling_bounds(tables, params.flavor_nu_min, params.flavor_nu_max)
    total_min = sum(lo for lo, _ in bounds)
    total_max = sum(hi for _, hi in bounds)
    if nu_total < total_min - 1e-12 or nu_total > total_max + 1e-12:
        raise ValueError("nu_total outside combined flavor filling bounds and band capacity")

    rng = np.random.default_rng(params.seed)
    seeds = make_stoner_seeds(nu_total, bounds, rng, params.n_random_seeds)

    def objective(x):
        return stoner_energy_meV_per_cell(np.asarray(x, dtype=float), tables, params, A_M_A2)

    def is_feasible(values):
        return bool(values.shape == (4,) and np.isfinite(values).all()
                    and abs(float(np.sum(values)) - nu_total) < 1e-8
                    and all(lo - 1e-8 <= value <= hi + 1e-8
                            for value, (lo, hi) in zip(values, bounds)))

    minima: list[dict] = []
    for name, x in seeds:
        minima.append(
            {
                "seed_name": name,
                "nu_f": x,
                "energy_meV_per_cell": objective(x),
                "success": False,
                "message": "unoptimized seed (diagnostic only)",
            }
        )

    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    if minimize is not None:
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - float(nu_total)},)
        for name, x0 in seeds:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": int(params.max_iter), "ftol": float(params.tol_energy_meV)},
            )
            raw = np.asarray(res.x, dtype=float)
            feasible = is_feasible(raw)
            # Only repair floating-point constraint drift. Projection must not
            # turn an infeasible optimizer output into an accepted solution.
            x = _clip_to_bounds_and_total(raw, bounds, nu_total) if feasible else raw
            energy = objective(x) if feasible else float("inf")
            stationarity = (stoner_stationarity_intervals(x, tables, params, A_M_A2)
                            if feasible and np.isfinite(energy) else None)
            valid = bool(res.success and feasible and np.isfinite(energy)
                         and stationarity["optimization_valid"])
            stability = stoner_local_stability(x, tables, params, A_M_A2) if valid else None
            valid = bool(valid and stability["valid"])
            optimizer_iterations = max(0, int(getattr(res, "nit", params.max_iter)))
            minima.append(
                {
                    "seed_name": f"{name}:slsqp",
                    "nu_f": x,
                    "energy_meV_per_cell": energy,
                    "success": valid,
                    "stationarity": stationarity,
                    "local_stability": stability,
                    "optimizer_iterations": optimizer_iterations,
                    "message": str(res.message) if feasible else f"{res.message}; invalid filling/energy",
                }
            )
            # Smooth SLSQP may stop beside a kinetic kink or at a saddle.
            # Refine only feasible optimizer outputs with unused iterations;
            # each exact feasible segment step consumes one remaining iterate.
            # Preserve the rejected SLSQP attempt as a separate diagnostic.
            remaining = max(0, int(params.max_iter)-optimizer_iterations)
            if valid or not feasible or not np.isfinite(energy) or remaining == 0:
                continue
            refined, refinement_iterations = x.copy(), 0
            refined_energy, refined_stationarity, refined_stability = energy, stationarity, stability
            refined_valid = False
            for _ in range(remaining):
                descent = stoner_feasible_descent(refined, tables, params, A_M_A2)
                if descent is None:
                    break
                next_values = _clip_to_bounds_and_total(descent, bounds, nu_total)
                next_energy = objective(next_values) if is_feasible(next_values) else float("inf")
                if not np.isfinite(next_energy) or next_energy >= refined_energy:
                    break
                refined = next_values
                refinement_iterations += 1
                refined_energy = next_energy
                refined_stationarity = stoner_stationarity_intervals(refined, tables, params, A_M_A2)
                refined_stability = (stoner_local_stability(refined, tables, params, A_M_A2)
                                     if refined_stationarity["optimization_valid"] else None)
                refined_valid = bool(refined_stability and refined_stability["valid"])
                if refined_valid:
                    break
            if refinement_iterations:
                minima.append({
                    "seed_name": f"{name}:refined",
                    "nu_f": refined,
                    "energy_meV_per_cell": refined_energy,
                    "success": refined_valid,
                    "stationarity": refined_stationarity,
                    "local_stability": refined_stability,
                    "optimizer_iterations": optimizer_iterations,
                    "refinement_iterations": refinement_iterations,
                    "total_iterations": optimizer_iterations+refinement_iterations,
                    "energy_decrease_meV": energy-refined_energy,
                    "message": "exact feasible redistribution refinement",
                })

    minima.sort(key=lambda item: float(item["energy_meV_per_cell"]) if np.isfinite(item["energy_meV_per_cell"]) else float("inf"))
    accepted = [candidate for candidate in minima if candidate["success"]]
    # If every solve failed, retain a diagnostic candidate with success=False.
    # Raw seeds and failed optimizer attempts never certify a minimum.
    best = accepted[0] if accepted else minima[0]
    return StonerResult(
        nu_total=float(nu_total),
        nu_f=np.asarray(best["nu_f"], dtype=float),
        energy_meV_per_cell=float(best["energy_meV_per_cell"]),
        success=bool(best["success"]),
        seed_name=str(best["seed_name"]),
        local_minima=minima,
        flavor_bounds_nu=bounds,
    )
