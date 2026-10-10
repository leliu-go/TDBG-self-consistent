"""Serialization helpers for four-flavor scan results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .active_polarization import active_result_fields


def result_to_dict(result) -> dict:
    return {
        **active_result_fields(result),
        "nu_total": result.nu_total,
        "nu_f": result.nu_f.tolist(),
        "flavor_bounds_nu": result.flavor_bounds_nu,
        "energy_meV_per_cell": result.energy_meV_per_cell,
        "success": result.success,
        "seed_name": result.seed_name,
        "minimum_validation": getattr(result, "minimum_validation", None),
        "local_minima": [{**candidate, "nu_f": np.asarray(candidate["nu_f"]).tolist()}
                         for candidate in result.local_minima],
        "spin_polarization": result.spin_polarization,
        "valley_polarization": result.valley_polarization,
        "flavor_polarization": result.flavor_polarization,
    }


def write_results_json(path: str | Path, results) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([result_to_dict(r) for r in results], indent=2), encoding="utf-8")
