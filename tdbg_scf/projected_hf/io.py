"""Serialization helpers for projected flavor calculations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .flavor_model import FlavorProjectedModel


def load_flavor_models_npz(path: str | Path) -> list[FlavorProjectedModel]:
    data = np.load(path, allow_pickle=False)
    required = {"energies0_meV", "layer_mats", "U_ref_meV", "kpts", "weights"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"projected NPZ input is missing keys: {missing}")

    energies = np.asarray(data["energies0_meV"], dtype=float)
    layer_mats = np.asarray(data["layer_mats"], dtype=np.complex128)
    U_ref = np.asarray(data["U_ref_meV"], dtype=float)
    kpts = np.asarray(data["kpts"], dtype=float)
    weights = np.asarray(data["weights"], dtype=float)
    remote = np.asarray(data["remote_density_per_k_layer"], dtype=float) if "remote_density_per_k_layer" in data else None

    if energies.ndim == 2:
        energies = np.repeat(energies[None, :, :], 4, axis=0)
    if layer_mats.ndim == 4:
        layer_mats = np.repeat(layer_mats[None, :, :, :, :], 4, axis=0)
    if U_ref.ndim == 1:
        U_ref = np.repeat(U_ref[None, :], 4, axis=0)
    if remote is None:
        remote = np.zeros((4, energies.shape[1], 4), dtype=float)
    elif remote.ndim == 2:
        remote = np.repeat(remote[None, :, :], 4, axis=0)

    if energies.shape[0] != 4 or layer_mats.shape[0] != 4 or U_ref.shape[0] != 4 or remote.shape[0] != 4:
        raise ValueError("flavor-resolved arrays must have first dimension 4")

    return [
        FlavorProjectedModel(
            flavor_index=f,
            energies0_meV=energies[f],
            layer_mats=layer_mats[f],
            U_ref_meV=U_ref[f],
            kpts=kpts,
            weights=weights,
            remote_density_per_k_layer=remote[f],
        )
        for f in range(4)
    ]


def result_to_dict(result) -> dict:
    return {
        "nu_total": result.nu_total,
        "D_Vnm": result.D_Vnm,
        "seed_name": result.seed_name,
        "nu_f": result.nu_f.tolist(),
        "U_meV": result.U_meV.tolist(),
        "layer_density_a2": result.layer_density_a2.tolist(),
        "mu_meV": result.mu_meV,
        "energy_meV_per_cell": result.energy_meV_per_cell,
        "converged": result.converged,
        "iterations": result.iterations,
        "message": result.message,
        "spin_polarization": result.spin_polarization,
        "valley_polarization": result.valley_polarization,
        "flavor_polarization": result.flavor_polarization,
    }


def write_results_json(path: str | Path, results) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([result_to_dict(r) for r in results], indent=2), encoding="utf-8")
