"""Provenance for the existing four-layer ABBA model and scan plots."""
from __future__ import annotations
from dataclasses import asdict
import argparse
from .continuum import TDBGParameters
import json
import numpy as np


def hamiltonian_metadata(ham) -> dict:
    # This names the existing bonds; it does not permute the historical basis.
    return {"model": "ABBA", "stacking": "BA-AB", "stacking_top_to_bottom": "BA-AB",
            "n_layers": 4, "hamiltonian_dimension": int(ham.dim),
            "layer_order": "L1,L2,L3,L4: top to bottom",
            "bernal_AB_convention": "B_upper_A_lower_dimer",
            "upper_dimer": "A1-B2", "lower_dimer": "B3-A4", **asdict(ham.params)}


def hamiltonian_npz_metadata(ham) -> dict:
    result = hamiltonian_metadata(ham)
    return {**result, "hamiltonian_json": json.dumps(result, sort_keys=True),
            "hamiltonian_parameters_json": json.dumps(asdict(ham.params), sort_keys=True)}


def displacement_metadata(D_sign: float = -1.0) -> dict:
    """Signed external displacement D/epsilon_0, in V/nm.

    Positive z runs from top to bottom and electron onsite U=-e*phi.
    E_down=(1/e)*dU/dz. In the neutral unscreened slab the Gauss update
    gives E_down=D_sign*D/eps_perp; screened local fields need not retain
    the imposed direction in every interlayer region.
    """
    if D_sign not in (-1.0, 1.0):
        raise ValueError("D_sign must be -1 or +1")
    return {
        "D_sign": float(D_sign),
        "D_positive_direction": "top_to_bottom" if D_sign == 1 else "bottom_to_top",
        "z_positive_direction": "top_to_bottom",
        "layer_potential_energy_convention": "U=-e*phi",
        "D_units_convention": "external_displacement_divided_by_epsilon0_V_per_nm",
    }


def axis_edges(values, single_half_width=.5) -> np.ndarray:
    """Place plot-cell edges halfway between explicitly supplied axis values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scan axis must be a finite non-empty vector")
    if values.size == 1:
        if not np.isfinite(single_half_width) or single_half_width <= 0:
            raise ValueError("single-axis half width must be positive")
        return np.array([values[0] - single_half_width, values[0] + single_half_width])
    if not np.all(np.diff(values) > 0):
        raise ValueError("scan axis values must be strictly increasing")
    middle = .5 * (values[:-1] + values[1:])
    return np.r_[values[0] - .5*(values[1]-values[0]), middle,
                 values[-1] + .5*(values[-1]-values[-2])]


def parameters_from_args(args: argparse.Namespace, valley: int) -> TDBGParameters:
    return TDBGParameters(
        theta_deg=args.theta_deg,
        cutoff=args.cutoff,
        valley=valley,
        omega_meV=args.omega,
        wAA=args.wAA,
        wAB=args.wAB,
        a_cc_A=args.a_cc_A,
        gamma0_meV=args.gamma0_meV,
        gamma1_meV=args.gamma1_meV,
        gamma3_meV=args.gamma3_meV,
        gamma4_meV=args.gamma4_meV,
        sublattice_Z_meV=args.Z,
    )


D_SIGN_HELP = "Positive D: -1 (default) bottom to top; +1 top to bottom. Layers are ordered top to bottom."


def hamiltonian_metadata_matches(saved, expected):
    """Compare all physical fields; allow only newly added descriptive fields to be absent."""
    from dataclasses import fields
    from .atomic_io import canonical_json
    if not isinstance(saved, dict):
        return False
    required = {field.name for field in fields(TDBGParameters)} | {
        "model", "n_layers", "hamiltonian_dimension", "layer_order"}
    if not required.issubset(saved):
        return False
    return all(canonical_json(saved[key]) == canonical_json(value)
               for key, value in expected.items() if key in saved)
