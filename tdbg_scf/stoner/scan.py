"""Small grid runner for the four-flavor phenomenological solver."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..correlated_io import ensure_output_dir
from ..filling import moire_cell_area_A2_from_weights
from .dos import build_flavor_band_table
from .io import write_results_json
from .params import StonerParams
from .solver import solve_stoner_fixed_nu


def _load_tables_from_npz(path: str | Path, A_M_A2: float | None = None) -> tuple[list, float]:
    data = np.load(path, allow_pickle=False)
    if "weights" not in data:
        raise ValueError("Stoner NPZ input must contain weights")
    weights = np.asarray(data["weights"], dtype=float)
    area = moire_cell_area_A2_from_weights(weights) if A_M_A2 is None else float(A_M_A2)
    if "energies_meV" in data:
        energies = np.asarray(data["energies_meV"], dtype=float)
    elif "energies0_meV" in data:
        energies = np.asarray(data["energies0_meV"], dtype=float)
    else:
        raise ValueError("Stoner NPZ input must contain energies_meV or energies0_meV")

    if energies.ndim == 2:
        tables = [build_flavor_band_table(energies, weights, area) for _ in range(4)]
    elif energies.ndim == 3 and energies.shape[0] == 4:
        tables = [build_flavor_band_table(energies[f], weights, area) for f in range(4)]
    else:
        raise ValueError("energies must have shape (Nk,Nb) or (4,Nk,Nb)")
    return tables, area


def run_stoner_scan(config) -> list:
    """Run a scan over ``shared.nu_grid`` using prebuilt flavor tables."""

    shared = config.get("shared", {})
    tables = config.get("tables")
    A_M_A2 = config.get("A_M_A2", shared.get("A_M_A2"))
    if tables is None:
        band_source = config.get("band_source", {})
        source_path = band_source.get("precomputed_npz_path") or band_source.get("projected_model_path")
        if source_path is None:
            raise ValueError("config must provide tables or band_source.precomputed_npz_path")
        tables, A_M_A2 = _load_tables_from_npz(source_path, A_M_A2)
    else:
        A_M_A2 = float(A_M_A2)
    params = StonerParams(**config.get("stoner", {}))
    results = [solve_stoner_fixed_nu(float(nu), tables, params, float(A_M_A2)) for nu in shared.get("nu_grid", [])]
    if shared.get("output_dir"):
        out = ensure_output_dir(Path(shared["output_dir"]) / "stoner")
        write_results_json(out / "results.json", results)
    return results
