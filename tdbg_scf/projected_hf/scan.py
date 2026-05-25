"""Grid runner for projected flavor calculations."""
from __future__ import annotations

from pathlib import Path

from ..correlated_io import ensure_output_dir
from .io import load_flavor_models_npz, write_results_json
from .params import ProjectedHFParams
from .solver import ProjectedHFSolver


def run_projected_hf_scan(config, models=None) -> list:
    shared = config.get("shared", {})
    if models is None:
        band_source = config.get("band_source", {})
        source_path = band_source.get("projected_model_path") or band_source.get("precomputed_npz_path")
        if source_path is None:
            raise ValueError("config must provide models or band_source.projected_model_path")
        models = load_flavor_models_npz(source_path)
    params = ProjectedHFParams(**config.get("projected_hf", {}))
    solver = ProjectedHFSolver(models, params=params)
    results = []
    for D in shared.get("D_Vnm_grid", [0.0]):
        for nu in shared.get("nu_grid", [0.0]):
            results.append(solver.solve_fixed_nu_D(float(nu), float(D)))
    if shared.get("output_dir"):
        out = ensure_output_dir(Path(shared["output_dir"]) / "projected_hf")
        write_results_json(out / "results.json", results)
    return results


def run_projected_scan(config, models=None) -> list:
    return run_projected_hf_scan(config, models=models)
