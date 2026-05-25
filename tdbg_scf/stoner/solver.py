"""Fixed-filling minimization for the four-flavor phenomenological model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dos import FlavorBandTable
from .energy import stoner_energy_meV_per_cell
from .params import StonerParams


@dataclass
class StonerResult:
    nu_total: float
    nu_f: np.ndarray
    energy_meV_per_cell: float
    success: bool
    seed_name: str
    local_minima: list[dict]

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

    for index in range(int(n_random)):
        add(f"random_{index}", rng.dirichlet(np.ones(4)) * float(nu_total))
    return seeds


def solve_stoner_fixed_nu(
    nu_total: float,
    tables: list[FlavorBandTable],
    params: StonerParams,
    A_M_A2: float,
) -> StonerResult:
    if len(tables) != 4:
        raise ValueError("Need exactly four flavor DOS tables")

    bounds = [(table.nu_min, table.nu_max) for table in tables]
    total_min = sum(lo for lo, _ in bounds)
    total_max = sum(hi for _, hi in bounds)
    if nu_total < total_min - 1e-12 or nu_total > total_max + 1e-12:
        raise ValueError("nu_total outside available band capacity")

    rng = np.random.default_rng(params.seed)
    seeds = make_stoner_seeds(nu_total, bounds, rng, params.n_random_seeds)

    def objective(x):
        return stoner_energy_meV_per_cell(np.asarray(x, dtype=float), tables, params, A_M_A2)

    minima: list[dict] = []
    for name, x in seeds:
        minima.append(
            {
                "seed_name": name,
                "nu_f": x,
                "energy_meV_per_cell": objective(x),
                "success": True,
                "message": "seed",
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
            x = _clip_to_bounds_and_total(res.x, bounds, nu_total)
            minima.append(
                {
                    "seed_name": f"{name}:slsqp",
                    "nu_f": x,
                    "energy_meV_per_cell": objective(x),
                    "success": bool(res.success),
                    "message": str(res.message),
                }
            )

    minima.sort(key=lambda item: float(item["energy_meV_per_cell"]))
    best = minima[0]
    return StonerResult(
        nu_total=float(nu_total),
        nu_f=np.asarray(best["nu_f"], dtype=float),
        energy_meV_per_cell=float(best["energy_meV_per_cell"]),
        success=bool(best["success"]),
        seed_name=str(best["seed_name"]),
        local_minima=minima,
    )
