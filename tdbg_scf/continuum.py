"""Continuum Hamiltonian for ABBA twisted double bilayer graphene.

The implementation follows the structure of the uploaded ``ABBATDBG2.py``:
- square plane-wave cutoff (i,j) in [-N,N]^2;
- one AB bilayer rotated by -theta/2 and one AB bilayer rotated by +theta/2;
- moiré tunneling matrices T_qb, T_qtr, T_qtl between the two middle layers;
- layer order (1,2,3,4) = top-to-bottom in the uploaded code convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .lattice import MoireGeometry, PlaneWaveLattice, make_plane_wave_lattice


@dataclass
class TDBGParameters:
    theta_deg: float = 1.35
    cutoff: int = 2
    valley: int = +1
    omega_meV: float = 100.0
    wAA: float = 0.8
    wAB: float = 1.0
    a_cc_A: float = 1.420
    gamma0_meV: float = 2610.0
    gamma1_meV: float = 361.0
    gamma3_meV: float = 283.0
    gamma4_meV: float = 138.0
    sublattice_Z_meV: float = 15.0

    @property
    def hv_meVA(self) -> float:
        return 1.5 * self.a_cc_A * self.gamma0_meV

    @property
    def hv2_meVA(self) -> float:
        # Same symbol as uploaded code: hv2 = 1.5*d*283, assigned to gamma3-like terms.
        return 1.5 * self.a_cc_A * self.gamma3_meV

    @property
    def hv3_meVA(self) -> float:
        # Same symbol as uploaded code: hv3 = 1.5*d*138, electron-hole asymmetry terms.
        return 1.5 * self.a_cc_A * self.gamma4_meV


class TDBGContinuumHamiltonian:
    """ABBA TDBG continuum Hamiltonian with layer-dependent potentials U_l.

    Basis order for each plane-wave site of the upper bilayer:
        (A1, B1, A2, B2)
    and for each plane-wave site of the lower bilayer:
        (A3, B3, A4, B4).
    The total basis is all upper-bilayer sites followed by all lower-bilayer sites.
    """

    def __init__(self, params: TDBGParameters):
        self.params = params
        self.geom = MoireGeometry(theta_deg=params.theta_deg, a_cc_A=params.a_cc_A, valley=params.valley)
        self.lattice: PlaneWaveLattice = make_plane_wave_lattice(params.cutoff)
        self.siteN = self.lattice.nG
        self.dim = 8 * self.siteN
        self._init_tunneling_matrices()
        self.layer_indices = self._build_layer_indices()

    @property
    def degeneracy_default(self) -> int:
        # spin x valley, if both valleys are treated as degenerate.
        return 4

    def _init_tunneling_matrices(self) -> None:
        p = self.params
        I = 1j
        valley = p.valley
        ei120 = np.cos(2 * np.pi / 3) + valley * I * np.sin(2 * np.pi / 3)
        ei240 = np.cos(2 * np.pi / 3) - valley * I * np.sin(2 * np.pi / 3)
        self.Tqb = p.omega_meV * np.array([[p.wAA, p.wAB], [p.wAB, p.wAA]], dtype=np.complex128)
        self.Tqtr = p.omega_meV * np.array([[p.wAA, p.wAB * ei240], [p.wAB * ei120, p.wAA]], dtype=np.complex128)
        self.Tqtl = p.omega_meV * np.array([[p.wAA, p.wAB * ei120], [p.wAB * ei240, p.wAA]], dtype=np.complex128)
        self.TqbD = self.Tqb.conj().T
        self.TqtrD = self.Tqtr.conj().T
        self.TqtlD = self.Tqtl.conj().T

    def _upper_index(self, g: int, orb: int) -> int:
        return 4 * g + orb

    def _lower_index(self, g: int, orb: int) -> int:
        return 4 * (self.siteN + g) + orb

    def _build_layer_indices(self) -> list[np.ndarray]:
        indices: list[list[int]] = [[], [], [], []]
        for g in range(self.siteN):
            # layer 1: A1 B1
            indices[0].extend([self._upper_index(g, 0), self._upper_index(g, 1)])
            # layer 2: A2 B2
            indices[1].extend([self._upper_index(g, 2), self._upper_index(g, 3)])
            # layer 3: A3 B3
            indices[2].extend([self._lower_index(g, 0), self._lower_index(g, 1)])
            # layer 4: A4 B4
            indices[3].extend([self._lower_index(g, 2), self._lower_index(g, 3)])
        return [np.asarray(x, dtype=int) for x in indices]

    def layer_projectors_diagonal(self) -> np.ndarray:
        """Return P_l diagonal masks, shape (4, dim)."""
        masks = np.zeros((4, self.dim), dtype=float)
        for l, idx in enumerate(self.layer_indices):
            masks[l, idx] = 1.0
        return masks

    def layer_potential_from_Zk(self, Zk_meV: float) -> np.ndarray:
        """Linear layer profile used by the uploaded script: (3/2,1/2,-1/2,-3/2) Zk."""
        return np.array([1.5, 0.5, -0.5, -1.5], dtype=float) * Zk_meV

    def hamiltonian(self, kx: float, ky: float, U_layer_meV: np.ndarray | None = None) -> np.ndarray:
        """Build the full continuum Hamiltonian at momentum (kx,ky), in meV."""
        p = self.params
        geom = self.geom
        valley = p.valley
        I = 1j
        U = np.zeros(4, dtype=float) if U_layer_meV is None else np.asarray(U_layer_meV, dtype=float)
        if U.shape != (4,):
            raise ValueError("U_layer_meV must have shape (4,)")

        H = np.zeros((self.dim, self.dim), dtype=np.complex128)
        b1m = geom.b1m
        b2m = geom.b2m
        K1 = geom.K1
        K2 = geom.K2
        th = geom.theta_rad
        Z = p.sublattice_Z_meV

        # Upper bilayer: rotated by -theta/2 in the uploaded code convention.
        for g, (ix, iy) in enumerate(self.lattice.G_indices):
            ax = kx - valley * K1[0] + ix * b1m[0] + iy * b2m[0]
            ay = ky - valley * K1[1] + ix * b1m[1] + iy * b2m[1]
            qx = np.cos(th / 2.0) * ax + np.sin(th / 2.0) * ay
            qy = -np.sin(th / 2.0) * ax + np.cos(th / 2.0) * ay
            A1, B1, A2, B2 = [self._upper_index(g, o) for o in range(4)]

            H[A1, A1] = U[0] + Z
            H[B1, B1] = U[0]
            H[A2, A2] = U[1]
            H[B2, B2] = U[1] + Z

            km = valley * qx - I * qy
            kp = valley * qx + I * qy

            H[A1, B1] = -p.hv_meVA * km
            H[B1, A1] = -p.hv_meVA * kp
            H[A2, B2] = -p.hv_meVA * km
            H[B2, A2] = -p.hv_meVA * kp

            H[B1, A2] = p.hv2_meVA * km
            H[A2, B1] = p.hv2_meVA * kp
            H[A1, A2] = p.hv3_meVA * kp
            H[A2, A1] = p.hv3_meVA * km
            H[B1, B2] = p.hv3_meVA * kp
            H[B2, B1] = p.hv3_meVA * km
            H[A1, B2] = p.gamma1_meV
            H[B2, A1] = p.gamma1_meV

            # Moire coupling from lower layer of upper bilayer (A2,B2) to upper layer of lower bilayer (A3,B3).
            self._add_tunnel_upper_to_lower(H, g, ix, iy)

        # Lower bilayer: rotated by +theta/2 in the uploaded code convention.
        for g, (ix, iy) in enumerate(self.lattice.G_indices):
            ax = kx - valley * K2[0] + ix * b1m[0] + iy * b2m[0]
            ay = ky - valley * K2[1] + ix * b1m[1] + iy * b2m[1]
            qx = np.cos(th / 2.0) * ax - np.sin(th / 2.0) * ay
            qy = np.sin(th / 2.0) * ax + np.cos(th / 2.0) * ay
            A3, B3, A4, B4 = [self._lower_index(g, o) for o in range(4)]

            H[A3, A3] = U[2]
            H[B3, B3] = U[2] + Z
            H[A4, A4] = U[3] + Z
            H[B4, B4] = U[3]

            km = valley * qx - I * qy
            kp = valley * qx + I * qy

            H[A3, B3] = -p.hv_meVA * km
            H[B3, A3] = -p.hv_meVA * kp
            H[A4, B4] = -p.hv_meVA * km
            H[B4, A4] = -p.hv_meVA * kp

            H[A3, B4] = p.hv2_meVA * kp
            H[B4, A3] = p.hv2_meVA * km
            H[A3, A4] = p.hv3_meVA * km
            H[A4, A3] = p.hv3_meVA * kp
            H[B3, B4] = p.hv3_meVA * km
            H[B4, B3] = p.hv3_meVA * kp
            H[B3, A4] = p.gamma1_meV
            H[A4, B3] = p.gamma1_meV

            # Add Hermitian counterparts of moire coupling. This mirrors the uploaded code.
            self._add_tunnel_lower_to_upper(H, g, ix, iy)

        # The two tunnel loops fill Hermitian entries explicitly. Numerical symmetrization removes tiny asymmetry.
        H = 0.5 * (H + H.conj().T)
        return H

    def _add_tunnel_block(self, H: np.ndarray, lower_g: int, upper_g: int, T: np.ndarray) -> None:
        """Add T from upper middle layer (A2,B2) to lower middle layer (A3,B3)."""
        A2 = self._upper_index(upper_g, 2)
        B2 = self._upper_index(upper_g, 3)
        A3 = self._lower_index(lower_g, 0)
        B3 = self._lower_index(lower_g, 1)
        H[A3, A2] = T[0, 0]
        H[A3, B2] = T[0, 1]
        H[B3, A2] = T[1, 0]
        H[B3, B2] = T[1, 1]

    def _add_tunnel_upper_to_lower(self, H: np.ndarray, upper_g: int, ix: int, iy: int) -> None:
        p = self.params
        c = p.cutoff
        valley = p.valley
        # qb: same integer G
        self._add_tunnel_block(H, upper_g, upper_g, self.Tqb)
        # qtr: lower G = (ix, iy - valley)
        if iy != -valley * c:
            lower_g = self.lattice.inv[(ix, iy - valley)]
            self._add_tunnel_block(H, lower_g, upper_g, self.Tqtr)
        # qtl: lower G = (ix + valley, iy)
        if ix != valley * c:
            lower_g = self.lattice.inv[(ix + valley, iy)]
            self._add_tunnel_block(H, lower_g, upper_g, self.Tqtl)

    def _add_tunnel_lower_to_upper(self, H: np.ndarray, lower_g: int, ix: int, iy: int) -> None:
        p = self.params
        c = p.cutoff
        valley = p.valley
        # qb dagger: upper same G
        self._add_tunnel_block_dagger(H, lower_g, lower_g, self.TqbD)
        if iy != valley * c:
            upper_g = self.lattice.inv[(ix, iy + valley)]
            self._add_tunnel_block_dagger(H, lower_g, upper_g, self.TqtrD)
        if ix != -valley * c:
            upper_g = self.lattice.inv[(ix - valley, iy)]
            self._add_tunnel_block_dagger(H, lower_g, upper_g, self.TqtlD)

    def _add_tunnel_block_dagger(self, H: np.ndarray, lower_g: int, upper_g: int, Tdag: np.ndarray) -> None:
        """Add dagger block from lower middle layer to upper middle layer."""
        A3 = self._lower_index(lower_g, 0)
        B3 = self._lower_index(lower_g, 1)
        A2 = self._upper_index(upper_g, 2)
        B2 = self._upper_index(upper_g, 3)
        H[A2, A3] = Tdag[0, 0]
        H[A2, B3] = Tdag[0, 1]
        H[B2, A3] = Tdag[1, 0]
        H[B2, B3] = Tdag[1, 1]

    def diagonalize(self, kpts: np.ndarray, U_layer_meV: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        evals = np.empty((len(kpts), self.dim), dtype=float)
        evecs = np.empty((len(kpts), self.dim, self.dim), dtype=np.complex128)
        for ik, (kx, ky) in enumerate(kpts):
            e, v = np.linalg.eigh(self.hamiltonian(float(kx), float(ky), U_layer_meV))
            evals[ik] = e
            evecs[ik] = v
        return evals, evecs
