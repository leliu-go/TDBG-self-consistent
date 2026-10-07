"""Resolve total filling or density scan grids using the Hamiltonian geometry."""
from __future__ import annotations

import argparse

import numpy as np

from .filling import moire_cell_area_A2_from_geometry


FILLING_DEFINITION = "nu=n_cm2*A_M_A2*1e-16; total spin/valley carriers relative to neutrality"


def add_density_axis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nu-values", type=str, default=None, help="Explicit total fillings per moire cell, comma-separated.")
    parser.add_argument("--nu-min", type=float, default=None, help="Filling minimum; default -6.")
    parser.add_argument("--nu-max", type=float, default=None, help="Filling maximum; default +6.")
    parser.add_argument("--nu-count", type=int, default=None, help="Number of filling points; default 61 (step 0.2).")
    parser.add_argument("--n-values-cm2", type=str, default=None, help="Explicit density grid instead of filling; comma-separated.")
    parser.add_argument("--n-min-cm2", type=float, default=None, help="Density minimum instead of filling, in cm^-2.")
    parser.add_argument("--n-max-cm2", type=float, default=None, help="Density maximum instead of filling, in cm^-2.")
    parser.add_argument("--n-count", type=int, default=None, help="Number of points in an explicitly selected density grid.")


def density_axis_metadata(n_values, geom) -> dict:
    """Portable conversion metadata for either scan unit, without diagonalization."""
    area = moire_cell_area_A2_from_geometry(geom)
    n = np.asarray(n_values, dtype=float)
    return {"n_values_cm2": n.tolist(), "nu_values": (n * 1e-16 * area).tolist(),
            "A_M_A2": area, "density_per_nu_cm2": 1e16 / area,
            "filling_definition": FILLING_DEFINITION}


def resolve_density_axis(args: argparse.Namespace, geom,
                         density_defaults=(-6e12, 6e12, 61)) -> tuple[list[float], dict]:
    """Default to nu=-6..6/61; explicit density input selects density units.

    Filling includes all spin and valley flavors: n=nu/A_M, with no extra
    factor of four. A one-point range uses its midpoint, matching map axes.
    """
    nu_names = ("nu_values", "nu_min", "nu_max", "nu_count")
    n_names = ("n_values_cm2", "n_min_cm2", "n_max_cm2", "n_count")
    have_nu = any(getattr(args, name, None) is not None for name in nu_names)
    have_n = any(getattr(args, name, None) is not None for name in n_names)
    if have_nu and have_n:
        raise ValueError("Choose either filling (--nu-*) or density (--n-*), not both")
    names, defaults = (n_names, density_defaults) if have_n else (nu_names, (-6., 6., 61))
    explicit, *bounds = [getattr(args, name, None) for name in names]
    if explicit is not None:
        if any(value is not None for value in bounds):
            raise ValueError("Explicit scan values cannot be combined with limits or count")
        values = np.asarray([float(x.strip()) for x in explicit.split(",") if x.strip()])
    else:
        lo, hi, count = [fallback if value is None else value for value, fallback in zip(bounds, defaults)]
        if not np.isfinite([lo, hi]).all() or lo > hi or count < 1:
            raise ValueError("Scan limits must be finite and ordered; count must be >= 1")
        values = np.array([.5 * lo + .5 * hi]) if count == 1 else np.linspace(lo, hi, count)
    if values.size == 0 or not np.isfinite(values).all() or np.unique(values).size != values.size:
        raise ValueError("Scan values must be nonempty, finite and unique")
    area = moire_cell_area_A2_from_geometry(geom)
    n = values if have_n else values * (1e16 / area)
    if not np.isfinite(n).all():
        raise ValueError("Converted densities must be finite")
    metadata = density_axis_metadata(n, geom)
    metadata["scan_axis"] = "n_cm2" if have_n else "nu"
    if not have_n:
        metadata["nu_values"] = values.tolist()
    return n.tolist(), metadata
