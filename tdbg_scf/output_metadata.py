"""Provenance for the existing four-layer ABBA model and scan plots."""
from dataclasses import asdict
import json
import numpy as np


def hamiltonian_metadata(ham) -> dict:
    return {"model": "ABBA", "n_layers": 4, "hamiltonian_dimension": int(ham.dim),
            "layer_order": "L1,L2,L3,L4: top to bottom", **asdict(ham.params)}


def hamiltonian_npz_metadata(ham) -> dict:
    result = hamiltonian_metadata(ham)
    return {**result, "hamiltonian_parameters_json": json.dumps(asdict(ham.params), sort_keys=True)}


def displacement_metadata(D_sign) -> dict:
    return {"D_sign": float(D_sign)}


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
