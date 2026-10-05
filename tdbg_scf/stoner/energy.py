"""Energy functional for the four-flavor phenomenological model."""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from itertools import product

from .dos import FlavorBandTable
from .params import StonerParams


@lru_cache(maxsize=64)
def _redistribution_directions(u: float, J: float) -> tuple:
    """Quadratic stationary directions on every four-flavor sign face.

    Normalize donated and received filling separately to one. On each sign
    face this leaves a compact product of simplices; its quadratic extrema
    occur at stationary points or smaller faces, all enumerated here. Pair
    transfers alone miss a coupled Hund instability.
    """
    spin_K, spin_Kp = np.array([1., 0., -1., 0.]), np.array([0., 1., 0., -1.])
    H = u*(np.ones((4, 4))-np.eye(4))-J*(np.outer(spin_K, spin_Kp)+np.outer(spin_Kp, spin_K))
    H /= max(1., float(np.max(np.abs(H))))
    directions = []
    for signs_tuple in product((-1, 0, 1), repeat=4):
        signs = np.asarray(signs_tuple)
        if not np.any(signs > 0) or not np.any(signs < 0):
            continue
        indices = np.flatnonzero(signs)
        A = np.array([signs[indices] > 0, signs[indices] < 0], dtype=float)
        local_H = H[np.ix_(indices, indices)]
        kkt = np.block([[local_H, A.T], [A, np.zeros((2, 2))]])
        rhs = np.r_[np.zeros(len(indices)), 1., -1.]
        solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
        local = solution[:len(indices)]
        if (np.max(np.abs(kkt@solution-rhs)) > 1e-10
                or np.any(local*signs[indices] < -1e-12)):
            continue
        direction = np.zeros(4)
        direction[indices] = local
        direction[np.abs(direction) < 1e-12] = 0.
        positive, negative = direction > 0, direction < 0
        if not np.any(positive) or not np.any(negative):
            continue
        direction[positive] /= np.sum(direction[positive])
        direction[negative] /= -np.sum(direction[negative])
        if not any(np.allclose(direction, old, atol=1e-12, rtol=0.) for old in directions):
            directions.append(direction)
    return tuple(tuple(direction) for direction in directions)


def _local_kinetic_pieces(values, tables):
    """One-sided kinetic slopes and nearest nodes, including carrier bounds."""
    slopes, endpoints = np.zeros((4, 2)), np.repeat(values[:, None], 2, axis=1)
    for f, (table, value) in enumerate(zip(tables, values)):
        grid, kinetic = table.nu_grid, table.kinetic_grid_meV
        left = int(np.searchsorted(grid, value, side="left"))-1
        right = int(np.searchsorted(grid, value, side="right"))-1
        if value > table.nu_min and left >= 0:
            endpoints[f, 0] = max(float(grid[left]), table.nu_min)
            slopes[f, 0] = (kinetic[left+1]-kinetic[left])/(grid[left+1]-grid[left])
        if value < table.nu_max and right < len(grid)-1:
            endpoints[f, 1] = min(float(grid[right+1]), table.nu_max)
            slopes[f, 1] = (kinetic[right+1]-kinetic[right])/(grid[right+1]-grid[right])
    return slopes, endpoints


def stoner_feasible_descent(nu_f, tables, params, A_M_A2):
    """Choose an exact quadratic step among redistribution directions.

    Each segment ends at its first kinetic node or carrier bound.
    Kinetic slopes are constant on each signed segment, so its total energy
    is an exact quadratic. Test its endpoints and, for positive curvature,
    its interior stationary point. Coupled Hund directions supplement pair
    transfers. A step is an optimization iterate, not an accepted minimum;
    callers must separately check feasibility, KKT and local stability.
    """
    values = np.asarray(nu_f, dtype=float)
    u, J = params.u0_meV_A2/A_M_A2, params.JH_meV_A2/A_M_A2
    m_K, m_Kp = values[0]-values[2], values[1]-values[3]
    sigma = u*(np.sum(values)-values)+J*np.array([-m_Kp, -m_K, m_Kp, m_K])
    spin_K, spin_Kp = np.array([1., 0., -1., 0.]), np.array([0., 1., 0., -1.])
    H = u*(np.ones((4, 4))-np.eye(4))-J*(np.outer(spin_K, spin_Kp)+np.outer(spin_Kp, spin_K))
    slopes, endpoints = _local_kinetic_pieces(values, tables)
    room = np.abs(endpoints-values[:, None])
    baseline = stoner_energy_meV_per_cell(values, tables, params, A_M_A2)
    best_gain, best_probe = 0., None
    for raw_direction in _redistribution_directions(float(u), float(J)):
        direction = np.asarray(raw_direction)
        active = np.flatnonzero(direction)
        sides = (direction[active] > 0).astype(int)
        distances = room[active, sides]
        if np.any(distances <= 0.):
            continue
        limits = distances/np.abs(direction[active])
        limit = float(np.min(limits))
        slope = float((slopes[active, sides]+sigma[active])@direction[active])
        curvature = float(direction@H@direction)
        step = limit if curvature <= 0. else float(np.clip(-slope/curvature, 0., limit))
        if step <= 0.:
            continue
        predicted_gain = -step*(slope+.5*step*curvature)
        roundoff = 64*np.finfo(float).eps*max(1., abs(baseline))
        if predicted_gain < best_gain-roundoff:
            continue
        probe = values+step*direction
        if step == limit:
            # Resolve the limiting kink exactly rather than stopping beside
            # it again. Other flavors keep the conserved redistribution.
            limiting = limits == limit
            probe[active[limiting]] = endpoints[active[limiting], sides[limiting]]
        energy = stoner_energy_meV_per_cell(probe, tables, params, A_M_A2)
        gain = baseline-energy
        roundoff = 64*np.finfo(float).eps*max(1., abs(baseline), abs(energy))
        if gain > roundoff and gain > best_gain:
            best_gain, best_probe = float(gain), probe
    return best_probe


def stoner_local_stability(nu_f, tables, params, A_M_A2) -> dict:
    """Reject resolved local descent inside the current kinetic pieces.

    Kinetic energy is piecewise linear and the interaction is quadratic.
    A negative Hessian eigenvalue alone must not reject a kink or a bound:
    only feasible directions with nonpositive one-sided slope are probed,
    without crossing the next kinetic node. Actual energy is evaluated and
    a decrease must exceed the configured energy tolerance and roundoff.
    This is a local acceptance check, not a global-minimum certificate.
    """
    values = np.asarray(nu_f, dtype=float)
    if values.shape != (4,) or len(tables) != 4 or not np.isfinite(values).all():
        raise ValueError("local stability requires four finite flavor fillings")
    u, J = params.u0_meV_A2/A_M_A2, params.JH_meV_A2/A_M_A2
    m_K, m_Kp = values[0]-values[2], values[1]-values[3]
    sigma = u*(np.sum(values)-values)+J*np.array([-m_Kp, -m_K, m_Kp, m_K])
    spin_K, spin_Kp = np.array([1., 0., -1., 0.]), np.array([0., 1., 0., -1.])
    H = u*(np.ones((4, 4))-np.eye(4))-J*(np.outer(spin_K, spin_Kp)+np.outer(spin_Kp, spin_K))
    slopes, endpoints = _local_kinetic_pieces(values, tables)
    room = np.abs(endpoints-values[:, None])
    baseline = stoner_energy_meV_per_cell(values, tables, params, A_M_A2)
    best_gain, best_probe, probe_count = 0., None, 0
    for raw_direction in _redistribution_directions(float(u), float(J)):
        direction = np.asarray(raw_direction)
        active = np.flatnonzero(direction)
        sides = (direction[active] > 0).astype(int)
        distances = room[active, sides]
        if np.any(distances <= 0.):
            continue
        gradient = slopes[active, sides]+sigma[active]
        directional_slope = float(gradient@direction[active])
        slope_roundoff = 128*np.finfo(float).eps*max(1., float(np.max(np.abs(gradient))))
        if directional_slope > slope_roundoff:
            # Positive one-sided slope makes a true kink/boundary minimum
            # even when the interaction has negative curvature farther away.
            continue
        limit = float(np.min(distances/np.abs(direction[active])))
        curvature = float(direction@H@direction)
        step = .8*limit
        if curvature > 0.:
            step = min(step, max(0., -directional_slope/curvature))
        if step <= 0.:
            continue
        probe = values+step*direction
        energy = stoner_energy_meV_per_cell(probe, tables, params, A_M_A2)
        probe_count += 1
        gain = baseline-energy
        threshold = max(float(params.tol_energy_meV),
                        64*np.finfo(float).eps*max(1., abs(baseline), abs(energy)))
        if gain > threshold and gain > best_gain:
            best_gain, best_probe = float(gain), probe
    return {"valid": best_probe is None, "energy_tolerance_meV": float(params.tol_energy_meV),
            "max_energy_decrease_meV": best_gain, "probe_count": probe_count,
            "descent_nu_f": None if best_probe is None else best_probe.tolist()}


def stoner_stationarity_intervals(nu_f, tables, params, A_M_A2,
                                  mu_tolerance_meV: float = 1e-4) -> dict:
    """Physical and bound-constrained subgradient intervals at one candidate."""
    values = np.asarray(nu_f, dtype=float)
    if (values.shape != (4,) or len(tables) != 4 or not np.isfinite(values).all()
            or not np.isfinite(A_M_A2) or A_M_A2 <= 0
            or not np.isfinite(mu_tolerance_meV) or mu_tolerance_meV < 0):
        raise ValueError("stationarity requires finite fillings, positive area and valid tolerance")
    u, J = params.u0_meV_A2/A_M_A2, params.JH_meV_A2/A_M_A2
    m_K, m_Kp = values[0]-values[2], values[1]-values[3]
    sigma = u*(np.sum(values)-values) + J*np.array([-m_Kp,-m_K,m_Kp,m_K])
    # Nonsmooth SLSQP can stop just beside an exact occupation node while
    # its energy is within ftol. Resolve that node only within 1e-6 filling
    # (also capped at 1% of a state interval by the table), and record this
    # occupation precision separately from the exact total-filling check.
    filling_tolerance = 1e-6
    intervals = np.asarray([table.mu_interval_of_nu(value,node_tolerance=filling_tolerance)
                            for table,value in zip(tables,values)]) + sigma[:,None]
    constrained = intervals.copy()
    for f,(table,value) in enumerate(zip(tables,values)):
        if abs(value-table.nu_min) <= filling_tolerance:
            constrained[f,0] = -np.inf
        if abs(value-table.nu_max) <= filling_tolerance:
            constrained[f,1] = np.inf
    constraint_mismatch = max(0.,float(np.max(constrained[:,0])-np.min(constrained[:,1])))
    lo,hi = float(np.max(intervals[:,0])),float(np.min(intervals[:,1]))
    mismatch = max(0.,lo-hi)
    finite = bool(np.isfinite(sigma).all())
    return {"self_energy_f_meV": sigma.tolist(),
            "optimization_valid": finite and constraint_mismatch <= mu_tolerance_meV,
            "constraint_mismatch_meV": constraint_mismatch,
            "physical_valid": finite and mismatch <= mu_tolerance_meV,
            "filling_tolerance_nu": filling_tolerance,
            "interval_mismatch_meV": mismatch,
            "mu_interval_lower_meV": lo,"mu_interval_upper_meV": hi,
            "mu_tolerance_meV": float(mu_tolerance_meV)}


def stoner_energy_meV_per_cell(
    nu_f: np.ndarray,
    tables: list[FlavorBandTable],
    params: StonerParams,
    A_M_A2: float,
) -> float:
    values = np.asarray(nu_f, dtype=float)
    if values.shape != (4,):
        raise ValueError("nu_f must have shape (4,)")
    if len(tables) != 4:
        raise ValueError("tables must contain four flavor tables")

    kinetic = 0.0
    for index in range(4):
        kinetic += tables[index].kinetic_of_nu(float(values[index]))

    u_cell = float(params.u0_meV_A2) / float(A_M_A2)
    J_cell = float(params.JH_meV_A2) / float(A_M_A2)

    total = float(np.sum(values))
    offdiag = total * total - float(np.dot(values, values))
    E_u = 0.5 * u_cell * offdiag

    m_K = values[0] - values[2]
    m_Kp = values[1] - values[3]
    E_hund = -J_cell * m_K * m_Kp

    return float(kinetic + E_u + E_hund)
