"""Projected self-energy models."""
from __future__ import annotations

import numpy as np


def _hermitian(arr: np.ndarray) -> np.ndarray:
    return 0.5 * (arr + np.swapaxes(arr.conj(), -1, -2))


def contact_projected_self_energy(P: np.ndarray, g_contact_meV: float) -> np.ndarray:
    density = np.asarray(P, dtype=np.complex128)
    return _hermitian(-float(g_contact_meV) * density)


def build_exchange_self_energy(exchange_model: str | None, density, params):
    model = "none" if exchange_model is None else str(exchange_model)
    if model == "none":
        return None
    if model == "contact_projected":
        if density is None:
            raise ValueError("density is required for contact_projected exchange")
        g = getattr(params, "g_contact_meV", 0.0) if params is not None else 0.0
        return contact_projected_self_energy(density.P, g)
    if model == "coulomb_formfactor":
        raise NotImplementedError("coulomb_formfactor exchange requires active-state form factors")
    raise ValueError(f"unknown exchange_model={model!r}")
