"""Moiré lattice utilities for ABBA twisted double bilayer graphene."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlaneWaveLattice:
    """Square plane-wave cutoff used in the uploaded ABBATDBG2.py code."""

    cutoff: int
    G_indices: np.ndarray       # shape (nG, 2), integer pairs (i,j)
    inv: dict[tuple[int, int], int]

    @property
    def nG(self) -> int:
        return int(self.G_indices.shape[0])


@dataclass(frozen=True)
class MoireGeometry:
    """Geometry and high-symmetry path for TDBG in Angstrom units."""

    theta_deg: float = 1.35
    a_cc_A: float = 1.420
    valley: int = +1

    def __post_init__(self) -> None:
        if self.valley not in (-1, +1):
            raise ValueError("valley must be +1 or -1")

    @property
    def theta_rad(self) -> float:
        return np.deg2rad(self.theta_deg)

    @property
    def b1m(self) -> np.ndarray:
        th = self.theta_rad
        d = self.a_cc_A
        return 8.0 * np.pi * np.sin(th / 2.0) / (3.0 * d) * np.array([0.5, -np.sqrt(3.0) / 2.0])

    @property
    def b2m(self) -> np.ndarray:
        th = self.theta_rad
        d = self.a_cc_A
        return 8.0 * np.pi * np.sin(th / 2.0) / (3.0 * d) * np.array([0.5, np.sqrt(3.0) / 2.0])

    @property
    def qb(self) -> np.ndarray:
        th = self.theta_rad
        d = self.a_cc_A
        return 8.0 * np.pi * np.sin(th / 2.0) / (3.0 * np.sqrt(3.0) * d) * np.array([0.0, -1.0])

    @property
    def K1(self) -> np.ndarray:
        th = self.theta_rad
        d = self.a_cc_A
        return 8.0 * np.pi * np.sin(th / 2.0) / (3.0 * np.sqrt(3.0) * d) * np.array([-np.sqrt(3.0) / 2.0, 0.5])

    @property
    def K2(self) -> np.ndarray:
        th = self.theta_rad
        d = self.a_cc_A
        return 8.0 * np.pi * np.sin(th / 2.0) / (3.0 * np.sqrt(3.0) * d) * np.array([-np.sqrt(3.0) / 2.0, -0.5])

    @property
    def mBZ_area_A2_inv(self) -> float:
        """Area of the parallelogram spanned by b1m and b2m, in A^{-2}."""
        b1 = self.b1m
        b2 = self.b2m
        return abs(b1[0] * b2[1] - b1[1] * b2[0])

    def high_symmetry_path(self, points_per_segment: int = 80) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
        """
        Return path matching the uploaded script: Ks -> Gamma_s -> M_s -> K'_s.
        The returned k-points are in Angstrom^{-1}.
        """
        kD = -self.qb[1]
        KptoG = np.linspace(0.5, 0.0, points_per_segment, endpoint=False)
        GtoM = np.linspace(0.0, np.sqrt(3.0) / 2.0, points_per_segment, endpoint=False)
        MtoK = np.linspace(0.0, 0.5, points_per_segment + 1, endpoint=True)

        kpts = []
        for k in KptoG:
            kpts.append([np.sqrt(3.0) * k * kD, k * kD])
        for k in GtoM:
            kpts.append([k * kD, 0.0])
        for k in MtoK:
            kpts.append([np.sqrt(3.0) / 2.0 * kD, -k * kD])
        kpts = np.asarray(kpts, dtype=float)

        dist = np.zeros(len(kpts), dtype=float)
        if len(kpts) > 1:
            dist[1:] = np.cumsum(np.linalg.norm(np.diff(kpts, axis=0), axis=1))
        ticks = [0, len(KptoG), len(KptoG) + len(GtoM), len(kpts) - 1]
        labels = [r"$K_s$", r"$\Gamma_s$", r"$M_s$", r"$K'_s$"]
        return kpts, dist, ticks, labels


def make_plane_wave_lattice(cutoff: int) -> PlaneWaveLattice:
    pairs = []
    inv: dict[tuple[int, int], int] = {}
    count = 0
    for i in range(-cutoff, cutoff + 1):
        for j in range(-cutoff, cutoff + 1):
            pairs.append((i, j))
            inv[(i, j)] = count
            count += 1
    return PlaneWaveLattice(cutoff=cutoff, G_indices=np.asarray(pairs, dtype=int), inv=inv)


def make_uniform_mbz_grid(geom: MoireGeometry, n1: int = 9, n2: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform midpoint grid over one moiré reciprocal-lattice parallelogram.

    Returns
    -------
    kpts : (n1*n2, 2) array in Angstrom^{-1}
    weights : integration weights for integral d^2k/(2pi)^2, in Angstrom^{-2}
    """
    b1 = geom.b1m
    b2 = geom.b2m
    kpts = []
    for i in range(n1):
        u = (i + 0.5) / n1 - 0.5
        for j in range(n2):
            v = (j + 0.5) / n2 - 0.5
            kpts.append(u * b1 + v * b2)
    kpts = np.asarray(kpts, dtype=float)
    area = geom.mBZ_area_A2_inv
    weight = area / (n1 * n2) / (2.0 * np.pi) ** 2
    return kpts, np.full(kpts.shape[0], weight, dtype=float)
