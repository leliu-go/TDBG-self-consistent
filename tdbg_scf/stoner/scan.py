"""Four-flavor Stoner scans on converged full-band layer-Hartree references."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..continuum import TDBGParameters, TDBGContinuumHamiltonian
from ..correlated_io import ensure_output_dir, write_json
from ..filling import filling_to_density_cm2, moire_cell_area_A2_from_weights
from ..lattice import make_uniform_mbz_grid
from ..scf_checks import require_converged_scf, scf_convergence_diagnostics
from ..solver_full import FullSCFConfig, FullSCFSolver
from .full_band import (build_full_band_flavor_tables,
                        stoner_mean_field_fermi_level, stoner_result_to_dict)
from .params import StonerParams
from .solver import solve_stoner_fixed_nu


def run_stoner_scan(config) -> list:
    """Run every (nu, D) through full layer Hartree before flavor minimization.

    The config contains ``hamiltonian``, ``grid``, ``full_scf``, ``stoner``
    and ``shared.nu_grid/D_Vnm_grid``. Precomputed tables cannot certify a
    self-consistent reference and are not accepted by this scan entry point.
    """
    if config.get("method", "stoner") != "stoner":
        raise ValueError("method must be stoner")
    if any(key in config for key in ("tables", "band_source", "A_M_A2")):
        raise ValueError("correlated scans require full-band layer Hartree; precomputed tables are unsupported")

    shared = config.get("shared", {})
    out = ensure_output_dir(Path(shared["output_dir"]) / "stoner") if shared.get("output_dir") else None
    if out is not None:
        for name in ("results.json", "partial_results.json", "error.json"):
            (out / name).unlink(missing_ok=True)
    parameters = TDBGParameters(**config.get("hamiltonian", {}))
    ham_K = TDBGContinuumHamiltonian(replace(parameters, valley=+1))
    ham_Kp = TDBGContinuumHamiltonian(replace(parameters, valley=-1))
    grid = config.get("grid", {})
    n1, n2 = grid.get("n1", 9), grid.get("n2", 9)
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in (n1, n2)):
        raise ValueError("grid.n1 and grid.n2 must be positive integers")
    kpts, weights = make_uniform_mbz_grid(ham_K.geom, n1=n1, n2=n2)
    A_M_A2 = moire_cell_area_A2_from_weights(weights)
    hartree_parameters = dict(config.get("full_scf", {}))
    if hartree_parameters.get("degeneracy", 4) != 4:
        raise ValueError("four-flavor scans require full_scf.degeneracy=4")
    params = StonerParams(**config.get("stoner", {}))
    full_solver = FullSCFSolver(ham_K, kpts, weights)
    results, records = [], []
    for D in shared.get("D_Vnm_grid", [0.]):
        for nu in shared.get("nu_grid", []):
            stage, state = "full_hartree", None
            try:
                target_cm2 = filling_to_density_cm2(float(nu), A_M_A2)
                state_config = FullSCFConfig(**{**hartree_parameters, "target_density_cm2": target_cm2,
                                               "D_Vnm": float(D), "keep_eigensystem": True})
                state = full_solver.solve(state_config)
                diagnostics = require_converged_scf(state, target_cm2, state_config.tol_meV)
                stage = "stoner"
                # Reuse the complete K eigenpairs at the accepted potential.
                evals_Kp, _ = ham_Kp.diagonalize(kpts, state.U_meV)
                tables = build_full_band_flavor_tables(state.evals, evals_Kp, weights, A_M_A2)
                result = solve_stoner_fixed_nu(float(nu), tables, params, A_M_A2)
                if not result.success:
                    raise RuntimeError(f"Stoner did not converge at nu={nu}, D={D}")
                level = stoner_mean_field_fermi_level(result, tables, params, A_M_A2)
                if not level["valid"]:
                    raise RuntimeError(f"Stoner has no physical common Fermi level at nu={nu}, D={D}: {level['reason']}")
                result.mu_f_meV = np.asarray(level["mu_f_meV"], dtype=float)
                result.mean_field_fermi_level = level
                results.append(result)
                records.append({**stoner_result_to_dict(result, tables), "D_Vnm": float(D),
                                "target_density_cm2": target_cm2, "A_M_A2": A_M_A2,
                                "scf": {**diagnostics, "source": "full", "band_count": ham_K.dim,
                                        "iterations": state.iterations, "mu_meV": state.mu_meV,
                                        "U_meV": state.U_meV.tolist(), "n_layer_cm2": state.n_layer_cm2.tolist()}})
            except Exception as exc:
                if out is not None:
                    error = {"stage": stage, "nu_total": float(nu), "D_Vnm": float(D),
                             "error": str(exc), "completed_points": len(records)}
                    if state is not None:
                        error["scf"] = {**scf_convergence_diagnostics(state, target_cm2, state_config.tol_meV),
                                        "source": "full", "band_count": ham_K.dim,
                                        "iterations": state.iterations, "mu_meV": state.mu_meV,
                                        "U_meV": state.U_meV.tolist(), "n_layer_cm2": state.n_layer_cm2.tolist()}
                    write_json(out / "error.json", error)
                    if records:
                        write_json(out / "partial_results.json", records)
                raise
    if out is not None:
        write_json(out / "results.json", records)
    return results
