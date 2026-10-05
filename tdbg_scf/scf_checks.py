"""Common acceptance checks for SCF results before physical analysis."""
from __future__ import annotations

from typing import Any

import numpy as np


def scf_convergence_diagnostics(result: Any, target_density_cm2: float,
                                tol_meV: float) -> dict[str, Any]:
    """Report solver, residual, charge and finite-value acceptance separately.

    The charge tolerance matches the map acceptance used by the full-band
    workflow. A solver flag alone never makes a state suitable for analysis.
    """
    U = np.asarray(result.U_meV, dtype=float)
    density = np.asarray(result.n_layer_cm2, dtype=float)
    residual = float(result.residual_meV)
    target = float(target_density_cm2)
    tolerance = float(tol_meV)
    charge_error = abs(float(np.sum(density) - target))
    charge_tolerance = max(1e4, abs(target) * 1e-8)
    values_finite = bool(
        U.ndim == density.ndim == 1
        and U.size > 0 and U.shape == density.shape
        and np.all(np.isfinite(U)) and np.all(np.isfinite(density))
        and np.isfinite(float(result.mu_meV))
    )
    residual_valid = bool(np.isfinite(residual) and np.isfinite(tolerance)
                          and tolerance > 0 and 0 <= residual < tolerance)
    charge_valid = bool(np.isfinite(target) and np.isfinite(charge_error)
                        and abs(charge_error) < charge_tolerance)
    solver_converged = bool(result.converged)
    valid = bool(solver_converged and residual_valid and charge_valid and values_finite)
    return {
        "scf_valid": valid,
        "scf_solver_converged": solver_converged,
        "scf_converged": valid,
        "scf_residual_meV": residual,
        "scf_tolerance_meV": tolerance,
        "scf_charge_error_cm2": charge_error,
        "scf_charge_tolerance_cm2": charge_tolerance,
        "scf_values_finite": values_finite,
        "scf_residual_valid": residual_valid,
        "scf_charge_valid": charge_valid,
    }


def require_converged_scf(result: Any, target_density_cm2: float,
                          tol_meV: float) -> dict[str, Any]:
    """Return acceptance diagnostics, or reject a state before DOS/gap/bands."""
    diagnostic = scf_convergence_diagnostics(result, target_density_cm2, tol_meV)
    if not diagnostic["scf_valid"]:
        raise RuntimeError(
            "SCF did not converge to an acceptable state: "
            f"solver_converged={diagnostic['scf_solver_converged']}, "
            f"residual={diagnostic['scf_residual_meV']:.6g} meV "
            f"(tol={diagnostic['scf_tolerance_meV']:.6g}), "
            f"charge_error={diagnostic['scf_charge_error_cm2']:.6g} cm^-2 "
            f"(tol={diagnostic['scf_charge_tolerance_cm2']:.6g}), "
            f"finite_values={diagnostic['scf_values_finite']}"
        )
    return diagnostic
