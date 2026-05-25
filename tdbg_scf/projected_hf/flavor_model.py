"""Data container for one explicit spin/valley flavor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlavorProjectedModel:
    flavor_index: int
    energies0_meV: np.ndarray
    layer_mats: np.ndarray
    U_ref_meV: np.ndarray
    kpts: np.ndarray
    weights: np.ndarray
    remote_density_per_k_layer: np.ndarray | None = None
    active_vecs: np.ndarray | None = None

    @property
    def n_k(self) -> int:
        return int(self.energies0_meV.shape[0])

    @property
    def n_band(self) -> int:
        return int(self.energies0_meV.shape[1])

    def h0_at_U(self, U_meV: np.ndarray) -> np.ndarray:
        delta_U = np.asarray(U_meV, dtype=float) - np.asarray(self.U_ref_meV, dtype=float)
        out = np.zeros((self.n_k, self.n_band, self.n_band), dtype=np.complex128)
        for ik in range(self.n_k):
            h = np.diag(np.asarray(self.energies0_meV[ik], dtype=float)).astype(np.complex128)
            for layer in range(4):
                h = h + delta_U[layer] * self.layer_mats[ik, layer]
            out[ik] = 0.5 * (h + h.conj().T)
        return out
