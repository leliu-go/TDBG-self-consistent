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

    def __post_init__(self) -> None:
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.tol_meV) or self.tol_meV <= 0.:
            raise ValueError("tol_meV must be finite and positive")


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
            U = linear_potential_from_D(config.D_Vnm, strength=config.initial_U, D_sign=config.D_sign)
            U -= np.mean(U)
        mixer = make_mixer(config.mixer, **(config.mixer_kwargs or {}))

        history: list[dict] = []
        converged = False
        residual = np.inf
        mu = 0.0
        n_layer = np.zeros(4, dtype=float)
        evals = None
        evecs = None
        evaluated_U = None

        def evaluate(U_state):
            # diagonalize returns every one of the 8(2N+1)^2 continuum bands;
            # all of them enter the layer density and half-state subtraction.
            energies, vectors = self.ham.diagonalize(self.kpts, U_state)
            chemical_potential = find_mu_for_density(
                energies, self.weights, target_a2,
                degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
            density = full_layer_density(
                chemical_potential, energies, vectors, self.weights, self.layer_masks,
                degeneracy=config.degeneracy, kBT_meV=config.kBT_meV,
                target_density_a2=target_a2)
            U_response = electro.update_U_from_density(density, n_bottom_gate_a2)
            defect = float(np.max(np.abs(U_response-U_state)))
            return energies, vectors, chemical_potential, density, U_response, defect

        def acceptable(U_state, chemical_potential, density, defect):
            charge_error = abs(float(np.sum(density)-target_a2))*1e16
            charge_tolerance = max(1e4, abs(config.target_density_cm2)*1e-8)
            return bool(np.isfinite(U_state).all() and np.isfinite(chemical_potential)
                        and np.isfinite(density).all() and np.isfinite(defect)
                        and 0. <= defect < config.tol_meV
                        and charge_error < charge_tolerance)

        for it in range(config.max_iter):
            evals, evecs, mu, n_layer, U_new, residual = evaluate(U)
            evaluated_U = U.copy()
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
            if acceptable(U, mu, n_layer, residual):
                # Return exactly the potential whose eigenstates and density
                # passed the residual test. One more unmixed response can
                # increase the residual of a stiff Hartree map.
                converged = True
                break
            if not np.isfinite(U_new).all():
                break
            U = mixer.update(U, U_new)

        # A final mixed step needs evaluation; a state already tested at U
        # reuses its complete eigenpairs instead of diagonalizing again.
        if not np.array_equal(U, evaluated_U):
            evals, evecs, mu, n_layer, _, residual = evaluate(U)
        converged = acceptable(U, mu, n_layer, residual)

        return FullSCFResult(
            U_meV=U,
            mu_meV=float(mu),
            n_layer_a2=n_layer,
            converged=converged,
            residual_meV=residual,
            iterations=len(history),
            history=history,
            evals=evals if config.keep_eigensystem else None,
            evecs=evecs if config.keep_eigensystem else None,
        )
