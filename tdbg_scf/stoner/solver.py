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
    minimum_validation: dict | None = None
    active_occupation_record: dict | None = None

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
        # Compensated electron-hole states can have |P|/|nu_total| > 1.
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


def _fillings_from_unit_cube(coordinates, bounds, total):
    """Surjective 3D parametrization of the bounded fixed-charge plane.

    Each interval reserves the full attainable charge range of the remaining
    flavors. Thus every feasible filling has a cube preimage, including faces,
    corners and electron-hole compensated states; no charge penalty is used.
    """
    raw = np.asarray(coordinates, dtype=float)
    if raw.shape != (3,) or not np.isfinite(raw).all() or np.any((raw < 0.) | (raw > 1.)):
        raise ValueError("DE coordinates must be three finite values in [0, 1]")
    values = np.empty(4)
    remaining = float(total)
    for index in range(3):
        lo = max(bounds[index][0], remaining-sum(b[1] for b in bounds[index+1:]))
        hi = min(bounds[index][1], remaining-sum(b[0] for b in bounds[index+1:]))
        if lo > hi + 1e-12:
            raise ValueError("Empty feasible fixed-charge interval")
        values[index] = lo+float(raw[index])*max(0., hi-lo)
        remaining -= values[index]
    values[3] = remaining
    # Repair only floating-point drift at a face, never invalid DE coordinates.
    return _clip_to_bounds_and_total(values, bounds, total)


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

    def objective(x):
        return stoner_energy_meV_per_cell(np.asarray(x, dtype=float), tables, params, A_M_A2)

    def is_feasible(values):
        return bool(values.shape == (4,) and np.isfinite(values).all()
                    and abs(float(np.sum(values)) - nu_total) < 1e-8
                    and all(lo - 1e-8 <= value <= hi + 1e-8
                            for value, (lo, hi) in zip(values, bounds)))

    def validation(x, energy):
        stationarity = (stoner_stationarity_intervals(x, tables, params, A_M_A2)
                        if is_feasible(x) and np.isfinite(energy) else None)
        stability = (stoner_local_stability(x, tables, params, A_M_A2)
                     if stationarity and stationarity["optimization_valid"] else None)
        return stationarity, stability, bool(stability and stability["valid"])

    from scipy.optimize import differential_evolution

    # SciPy 1.10's integer seed is uint32. Consecutive modulo seeds remain
    # distinct, reproducible and independent of the n-D scheduling order.
    seeds = [int(params.seed) % (2**32), (int(params.seed)+1) % (2**32)]
    minima, refined_candidates = [], []
    for seed in seeds:
        name = f"de_seed_{seed}"
        res = differential_evolution(
            lambda raw: objective(_fillings_from_unit_cube(raw, bounds, nu_total)),
            [(0., 1.)]*3, seed=seed, init="sobol", polish=False,
            maxiter=params.de_max_iter, popsize=params.de_popsize,
            tol=params.de_tol, atol=params.tol_energy_meV,
            workers=1, updating="immediate",
        )
        try:
            x = _fillings_from_unit_cube(res.x, bounds, nu_total)
            feasible = is_feasible(x)
        except (ValueError, TypeError):
            x, feasible = np.full(4, np.nan), False
        energy = objective(x) if feasible else float("inf")
        stationarity, stability, _ = validation(x, energy)
        search = {"de_seed": seed, "de_converged": bool(res.success),
                  "optimizer_iterations": max(0, int(getattr(res, "nit", 0))),
                  "de_function_evaluations": max(0, int(getattr(res, "nfev", 0))),
                  "de_message": str(res.message)}
        minima.append({"seed_name": name+":raw", "nu_f": x.copy(),
                       "energy_meV_per_cell": energy, "success": False,
                       "stationarity": stationarity, "local_stability": stability,
                       "message": str(res.message), "stage": "raw", **search})
        refined, refined_energy, steps = x.copy(), energy, 0
        if feasible and np.isfinite(energy):
            # Always refine, even if the first-order check already passes.
            # Follow feasible segments to occupation nodes / quadratic minima
            # until no exact descent remains or the independent budget ends.
            for _ in range(params.max_iter):
                descent = stoner_feasible_descent(refined, tables, params, A_M_A2)
                if descent is None:
                    break
                next_values = _clip_to_bounds_and_total(descent, bounds, nu_total)
                next_energy = objective(next_values) if is_feasible(next_values) else float("inf")
                if not np.isfinite(next_energy) or next_energy >= refined_energy:
                    break
                refined, refined_energy = next_values, next_energy
                steps += 1
        stationarity, stability, valid = validation(refined, refined_energy)
        budget_exhausted = bool(steps == params.max_iter and feasible
                                and stoner_feasible_descent(refined, tables, params, A_M_A2) is not None)
        candidate = {"seed_name": name+":refined", "nu_f": refined,
                     "energy_meV_per_cell": refined_energy, "success": valid,
                     "stationarity": stationarity, "local_stability": stability,
                     "refinement_iterations": steps,
                     "refinement_budget_exhausted": budget_exhausted,
                     "total_iterations": search["optimizer_iterations"]+steps,
                     "energy_decrease_meV": energy-refined_energy if feasible else 0.,
                     "message": "exact occupation-node refinement" if feasible else "invalid DE candidate",
                     "stage": "refined", **search}
        minima.append(candidate)
        refined_candidates.append(candidate)

    finite = sorted((c for c in refined_candidates if np.isfinite(c["energy_meV_per_cell"])),
                    key=lambda c: c["energy_meV_per_cell"])
    accepted = [c for c in finite if c["success"]]
    energy_tolerance = max(params.tol_energy_meV, 64*np.finfo(float).eps*
                           max([1.]+[abs(c["energy_meV_per_cell"]) for c in finite]))
    reason = "validated_minimum"
    if accepted:
        best = accepted[0]
        if finite[0]["energy_meV_per_cell"] < best["energy_meV_per_cell"]-energy_tolerance:
            best, reason = finite[0], "lower_energy_candidate_unvalidated"
    elif finite:
        best, reason = finite[0], "no_validated_minimum"
    else:
        # Keep a charge-conserving diagnostic only, never accept a raw seed.
        x = _fillings_from_unit_cube([.5]*3, bounds, nu_total)
        best = {"seed_name": "diagnostic_only", "nu_f": x, "success": False,
                "energy_meV_per_cell": objective(x), "message": "invalid DE outputs"}
        minima.append(best)
        reason = "invalid_de_candidates"
    spread = (float(finite[-1]["energy_meV_per_cell"]-finite[0]["energy_meV_per_cell"])
              if len(finite) == 2 else None)
    minima.sort(key=lambda c: c["energy_meV_per_cell"])
    return StonerResult(
        nu_total=float(nu_total),
        nu_f=np.asarray(best["nu_f"], dtype=float),
        energy_meV_per_cell=float(best["energy_meV_per_cell"]),
        success=bool(best["success"]),
        seed_name=str(best["seed_name"]),
        local_minima=minima,
        flavor_bounds_nu=bounds,
        minimum_validation={
            "method": "differential_evolution_3d_two_seeds",
            "reason": reason,
            "de_seeds": seeds,
            "de_max_iter": int(params.de_max_iter),
            "de_popsize": int(params.de_popsize),
            "de_tol": float(params.de_tol),
            "de_converged": best.get("de_converged", False),
            "de_function_evaluations": best.get("de_function_evaluations", 0),
            "independent_runs_agree": bool(len(accepted) == 2 and spread <= energy_tolerance),
            "independent_run_energy_spread_meV": spread,
            "stationarity": best.get("stationarity"),
            "local_stability": best.get("local_stability"),
            "optimizer_iterations": best.get("optimizer_iterations", 0),
            "refinement_iterations": best.get("refinement_iterations", 0),
            "refinement_budget_exhausted": best.get("refinement_budget_exhausted", False),
            "total_iterations": (best.get("optimizer_iterations", 0)
                                 + best.get("refinement_iterations", 0)),
            "global_minimum_certified": False,
        },
    )
