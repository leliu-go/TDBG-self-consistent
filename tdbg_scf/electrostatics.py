"""Layer-resolved electrostatics for a four-layer TDBG slab."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import gate_densities_from_n_D_a2, gauss_coeff_meV_per_a2


@dataclass
class LayerElectrostatics:
    n_layers: int = 4
    d_layer_nm: float = 0.335
    eps_perp: float = 4.0
    D_sign: float = -1.0
    # D_sign=-1 matches the uploaded ABBATDBG2.py convention: positive D corresponds
    # to U=(+3/2,+1/2,-1/2,-3/2) Zk in the absence of layer screening.
    # Set D_sign=+1 to use the RMG paper convention D=e(n_b-n_t)/(2 eps0) directly.

    def update_U_from_density(self, n_layer_a2: np.ndarray, n_bottom_gate_a2: float) -> np.ndarray:
        """
        Gauss-law layer-potential update.

        Convention:
            layer 1 is the top layer in the TDBG basis;
            layer n_layers is the bottom layer.
        The formula is written by accumulating charge from the bottom gate upward.
        The returned U has zero average gauge.
        """
        n = np.asarray(n_layer_a2, dtype=float)
        if n.shape != (self.n_layers,):
            raise ValueError(f"n_layer_a2 must have shape ({self.n_layers},)")

        # Work from bottom to top. Let U[layer] be top-to-bottom in arrays.
        coeff = gauss_coeff_meV_per_a2(self.d_layer_nm, self.eps_perp)
        U_bottom_to_top = np.zeros(self.n_layers, dtype=float)
        cumulative = 0.0
        n_bottom_to_top = n[::-1]
        for m in range(self.n_layers - 1):
            cumulative += n_bottom_to_top[m]
            sheet_below = n_bottom_gate_a2 + cumulative
            dU = -coeff * sheet_below
            U_bottom_to_top[m + 1] = U_bottom_to_top[m] + dU
        U_top_to_bottom = U_bottom_to_top[::-1]
        U_top_to_bottom -= np.mean(U_top_to_bottom)
        return U_top_to_bottom

    def gate_densities(self, n_total_a2: float, D_Vnm: float) -> tuple[float, float]:
        return gate_densities_from_n_D_a2(n_total_a2, self.D_sign * D_Vnm)


def linear_potential_from_D(D_Vnm: float, strength: str = "uploaded", D_sign: float = -1.0) -> np.ndarray:
    """
    Initial layer-potential profile for TDBG.

    ``uploaded`` uses the relation in ABBATDBG2.py: D = 4 Zk / 330 V/nm,
    and U = (3/2, 1/2, -1/2, -3/2) Zk.
    The input displacement is converted to the corresponding uploaded-sign
    profile with ``D_profile = -D_sign * D_Vnm`` so that the default
    ``D_sign=-1`` preserves the uploaded convention, while ``D_sign=+1``
    follows the direct gate convention.
    """
    D_profile_Vnm = -float(D_sign) * float(D_Vnm)
    if strength == "uploaded":
        Zk = 330.0 * D_profile_Vnm / 4.0
        return np.array([1.5, 0.5, -0.5, -1.5], dtype=float) * Zk
    if strength == "bare":
        # Potential drop e D d between adjacent graphene layers in meV.
        step = 1000.0 * 0.335 * D_profile_Vnm
        return np.array([1.5, 0.5, -0.5, -1.5], dtype=float) * step
    raise ValueError("strength must be 'uploaded' or 'bare'")
