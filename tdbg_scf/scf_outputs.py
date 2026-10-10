"""Views of existing SCF point data, without solving or diagonalizing again."""
from __future__ import annotations

import numpy as np

from .output_metadata import hamiltonian_metadata
from .output_metadata import displacement_metadata


SCF_FIELDS = (
    "label", "n_index", "D_index", "n_cm2", "D_Vnm", "nu_total", "A_M_A2",
    "eps_perp", "d_layer_nm", "full_mu_meV", "dos_mu_scf",
    "dos_mu_full_scf", "dos_mu_full_at_full_mu",
    "P_top_bottom_cm2", "P_outer_inner_cm2", "P_dipole_cm2",
    "D_sign", "D_positive_direction", "z_positive_direction", "electron_potential_convention", "layer_potential_energy_convention", "D_units_convention",
)


def scf_point_record(row: dict, ham) -> dict:
    """The baseline SCF stage remains valid if later Stoner processing fails."""
    keys = set(SCF_FIELDS)
    keys.update(key for key in row if key.startswith(("scf_", "gap_", "indirect_gap_")))
    keys.update(f"{prefix}{layer}_meV" if prefix == "U" else f"n{layer}_cm2"
                for layer in range(1, len(ham.layer_indices) + 1) for prefix in ("U", "n"))
    record = {key: value for key, value in row.items() if key in keys}
    record.update(hamiltonian_metadata(ham))
    record.update(displacement_metadata(row.get("D_sign", -1)))
    record["status"] = row.get("scf_status", row.get("status", "failed"))
    record["stage"] = "scf_before_stoner"
    record["source_point_status"] = row.get("status", "failed")
    if row.get("error"):
        record["source_error"] = row["error"]
    return record


def scf_scalar_grids(rows: list[dict], n_count: int, D_count: int) -> dict:
    """Numeric fields have (n,D) shape; failed stages stay NaN, never zero."""
    shape = (n_count, D_count)
    valid = np.zeros(shape, dtype=bool)
    converged = np.zeros(shape, dtype=bool)
    # Cell area is fixed by the Hamiltonian and saved once as axis metadata.
    excluded = {"n_index", "D_index", "n_cm2", "D_Vnm", "n_layers", "valley", "cutoff", "D_sign", "A_M_A2"}
    fields = set()
    for row in rows:
        for key, value in row.items():
            if key not in excluded and isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                if key in SCF_FIELDS or key.startswith(("scf_", "gap_", "indirect_gap_", "U")) or (key.startswith("n") and key.endswith("_cm2")):
                    fields.add(key)
    arrays = {key: np.full(shape, np.nan) for key in sorted(fields)}
    for row in rows:
        i, j = int(row["n_index"]), int(row["D_index"])
        if row.get("status") != "ok":
            continue
        valid[i, j] = True
        converged[i, j] = bool(row.get("scf_converged", False))
        for key, arr in arrays.items():
            value = row.get(key, np.nan)
            if value is not None:
                arr[i, j] = float(value)
    return {**arrays, "scf_valid": valid, "scf_converged": converged}
