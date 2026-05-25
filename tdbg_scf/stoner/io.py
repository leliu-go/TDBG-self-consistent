"""Serialization helpers for four-flavor scan results."""
from __future__ import annotations

import json
from pathlib import Path


def result_to_dict(result) -> dict:
    return {
        "nu_total": result.nu_total,
        "nu_f": result.nu_f.tolist(),
        "energy_meV_per_cell": result.energy_meV_per_cell,
        "success": result.success,
        "seed_name": result.seed_name,
        "spin_polarization": result.spin_polarization,
        "valley_polarization": result.valley_polarization,
        "flavor_polarization": result.flavor_polarization,
    }


def write_results_json(path: str | Path, results) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([result_to_dict(r) for r in results], indent=2), encoding="utf-8")
