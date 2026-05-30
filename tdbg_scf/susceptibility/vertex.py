from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VertexSpec:
    name: str
    gamma: np.ndarray
    description: str


def hermitize(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.complex128)
    return 0.5 * (arr + arr.conj().T)


def psd_sqrt(a: np.ndarray, clip_tol: float = 1e-12) -> np.ndarray:
    """Hermitian PSD square root with small negative eigenvalues clipped."""

    h = hermitize(a)
    vals, vecs = np.linalg.eigh(h)
    vals = np.where(vals > float(clip_tol), vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def make_valley_vertex_models(
    u_cell_meV: float,
    J_cell_meV: float,
    include_legacy: bool = True,
    hund_transverse_factor: float = 2.0,
) -> dict[str, VertexSpec]:
    """Return valley-pair transverse spin vertices in (K, K') basis."""

    u = float(u_cell_meV)
    J = float(J_cell_meV)
    f = float(hund_transverse_factor)
    models: dict[str, VertexSpec] = {
        "su4_diag": VertexSpec(
            name="su4_diag",
            gamma=np.array([[u, 0.0], [0.0, u]], dtype=float),
            description="Conservative valley-diagonal transverse vertex from the u_cell flavor-polarization term.",
        ),
        "su2_hund_factor2": VertexSpec(
            name="su2_hund_factor2",
            gamma=np.array([[u, f * J], [f * J, u]], dtype=float),
            description="Optional SU(2)-rotated transverse Hund extension; f=2 matches -J*mK*mKp with m=2Sz.",
        ),
    }
    if include_legacy:
        models["legacy_offdiag_J"] = VertexSpec(
            name="legacy_offdiag_J",
            gamma=np.array([[u, J], [J, u]], dtype=float),
            description="Legacy phenomenological off-diagonal J vertex from the first implementation; not main.",
        )
    return models


def generalized_stoner_lambda(
    chi_valley: np.ndarray,
    gamma: np.ndarray,
    clip_chi: bool = True,
    clip_gamma: bool = False,
) -> tuple[float, np.ndarray]:
    """Return largest generalized Stoner eigenvalue and eigenvector."""

    chi = hermitize(np.asarray(chi_valley, dtype=np.complex128))
    gam = hermitize(np.asarray(gamma, dtype=np.complex128))
    if clip_chi:
        sqrt_chi = psd_sqrt(chi)
    else:
        vals, vecs = np.linalg.eigh(chi)
        sqrt_chi = (vecs * np.sqrt(vals.astype(np.complex128))) @ vecs.conj().T

    if clip_gamma:
        sqrt_gam = psd_sqrt(gam)
        gam = sqrt_gam @ sqrt_gam

    kernel = hermitize(sqrt_chi @ gam @ sqrt_chi)
    vals, vecs = np.linalg.eigh(kernel)
    return float(np.real(vals[-1])), np.asarray(vecs[:, -1], dtype=np.complex128)


def legacy_scalar_total_lambda(chi_K: float, chi_Kp: float, u_cell_meV: float) -> float:
    return float(u_cell_meV) * float(chi_K + chi_Kp)
