"""Full non-projected self-consistent Hartree solver for TDBG."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import cm2_to_a2, a2_to_cm2
from .continuum import TDBGContinuumHamiltonian
from .electrostatics import LayerElectrostatics, linear_potential_from_D
from .density import find_mu_for_density, full_layer_density
from .mixing import make_mixer


@dataclass
class FullSCFResult:
    U_meV: np.ndarray
    mu_meV: float
    n_layer_a2: np.ndarray
    converged: bool
    residual_meV: float
    iterations: int
    history: list[dict]
    evals: np.ndarray | None = None
    evecs: np.ndarray | None = None

    @property
    def n_layer_cm2(self) -> np.ndarray:
        return np.array([a2_to_cm2(x) for x in self.n_layer_a2])


@dataclass
class FullSCFConfig:
    target_density_cm2: float = 0.0
    D_Vnm: float = 0.0
    degeneracy: int = 4
    kBT_meV: float = 0.1
    max_iter: int = 80
    tol_meV: float = 1e-5
    eps_perp: float = 4.0
    d_layer_nm: float = 0.335
    D_sign: float = -1.0
    mixer: str = "anderson"
    mixer_kwargs: dict | None = None
    initial_U: str = "uploaded"  # uploaded, bare, zero
    keep_eigensystem: bool = True


class FullSCFSolver:
    def __init__(self, ham: TDBGContinuumHamiltonian, kpts: np.ndarray, weights: np.ndarray):
        self.ham = ham
        self.kpts = np.asarray(kpts, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.layer_masks = ham.layer_projectors_diagonal()

    def solve(self, config: FullSCFConfig) -> FullSCFResult:
        electro = LayerElectrostatics(n_layers=4, d_layer_nm=config.d_layer_nm, eps_perp=config.eps_perp, D_sign=config.D_sign)
        target_a2 = cm2_to_a2(config.target_density_cm2)
        _, n_bottom_gate_a2 = electro.gate_densities(target_a2, config.D_Vnm)
        if config.initial_U == "zero":
            U = np.zeros(4, dtype=float)
        else:
            U = linear_potential_from_D(config.D_Vnm, strength=config.initial_U)
            U -= np.mean(U)
        mixer = make_mixer(config.mixer, **(config.mixer_kwargs or {}))

        history: list[dict] = []
        converged = False
        residual = np.inf
        mu = 0.0
        n_layer = np.zeros(4, dtype=float)
        evals = None
        evecs = None

        for it in range(config.max_iter):
            evals, evecs = self.ham.diagonalize(self.kpts, U)
            mu = find_mu_for_density(evals, self.weights, target_a2,
                                     degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
            n_layer = full_layer_density(mu, evals, evecs, self.weights, self.layer_masks,
                                         degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
            U_new = electro.update_U_from_density(n_layer, n_bottom_gate_a2)
            residual = float(np.max(np.abs(U_new - U)))
            history.append({
                "iteration": it,
                "residual_meV": residual,
                "mu_meV": float(mu),
                "U1_meV": float(U[0]),
                "U2_meV": float(U[1]),
                "U3_meV": float(U[2]),
                "U4_meV": float(U[3]),
                "n1_cm2": float(a2_to_cm2(n_layer[0])),
                "n2_cm2": float(a2_to_cm2(n_layer[1])),
                "n3_cm2": float(a2_to_cm2(n_layer[2])),
                "n4_cm2": float(a2_to_cm2(n_layer[3])),
            })
            if residual < config.tol_meV:
                U = U_new
                converged = True
                break
            U = mixer.update(U, U_new)

        # Recompute final eigenstates if last step changed U.
        evals_final, evecs_final = self.ham.diagonalize(self.kpts, U)
        mu_final = find_mu_for_density(evals_final, self.weights, target_a2,
                                       degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
        n_layer_final = full_layer_density(mu_final, evals_final, evecs_final, self.weights, self.layer_masks,
                                           degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
        U_check = electro.update_U_from_density(n_layer_final, n_bottom_gate_a2)
        residual_final = float(np.max(np.abs(U_check - U)))

        return FullSCFResult(
            U_meV=U,
            mu_meV=float(mu_final),
            n_layer_a2=n_layer_final,
            converged=bool(residual_final < config.tol_meV),
            residual_meV=residual_final,
            iterations=len(history),
            history=history,
            evals=evals_final if config.keep_eigensystem else None,
            evecs=evecs_final if config.keep_eigensystem else None,
        )
