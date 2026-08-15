"""Self-consistent flavor-resolved projected Hartree-Fock solver."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..electrostatics import LayerElectrostatics
from ..filling import filling_to_density_a2, moire_cell_area_A2_from_weights
from .density_matrix import ProjectedDensity, density_matrices_from_eigensystems
from .energy import band_energy_meV_per_cell
from .flavor_model import FlavorProjectedModel
from .params import ProjectedHFParams
from .seeds import flavor_filling_seeds
from .self_energy import build_exchange_self_energy


@dataclass
class ProjectedHFResult:
    converged: bool
    nu_total: float
    D_Vnm: float
    seed_name: str
    nu_f: np.ndarray
    U_meV: np.ndarray
    layer_density_a2: np.ndarray
    mu_meV: float
    density: ProjectedDensity
    evals_meV: np.ndarray
    evecs: np.ndarray
    self_energy_meV: np.ndarray
    energy_meV_per_cell: float
    iterations: int
    message: str
    local_minima: list[dict]

    @property
    def spin_polarization(self) -> float:
        return float((self.nu_f[0] + self.nu_f[1]) - (self.nu_f[2] + self.nu_f[3]))

    @property
    def valley_polarization(self) -> float:
        return float((self.nu_f[0] + self.nu_f[2]) - (self.nu_f[1] + self.nu_f[3]))

    @property
    def flavor_polarization(self) -> float:
        return float(np.max(self.nu_f) - np.min(self.nu_f))


class ProjectedHFSolver:
    def __init__(self, models: list[FlavorProjectedModel], params: ProjectedHFParams | None = None):
        if len(models) != 4:
            raise ValueError("models must contain four explicit flavors")
        self.models = models
        self.params = params or ProjectedHFParams()
        first = models[0]
        self.weights = np.asarray(first.weights, dtype=float)
        self.A_M_A2 = moire_cell_area_A2_from_weights(self.weights)
        self.n_k = first.n_k
        self.n_band = first.n_band
        for model in models:
            if model.n_k != self.n_k or model.n_band != self.n_band:
                raise ValueError("all flavor models must use the same k and band dimensions")
            if not np.allclose(model.weights, self.weights):
                raise ValueError("all flavor models must use the same weights")

    def _reference_count(self) -> float:
        if self.params.n_ref_per_flavor is not None:
            return float(self.params.n_ref_per_flavor)
        if self.params.filling_reference == "conduction_only":
            return 0.0
        return 0.5 * float(self.n_band)

    def _electrostatics(self) -> LayerElectrostatics:
        return LayerElectrostatics(
            n_layers=4,
            d_layer_nm=float(self.params.d_layer_nm),
            eps_perp=float(self.params.eps_perp),
            D_sign=float(self.params.D_sign),
        )

    def _initial_layer_potential(self, nu_total: float, D_Vnm: float) -> np.ndarray:
        if not self.params.uniform_layer_hartree:
            z = (np.arange(4, dtype=float) - 1.5) * float(self.params.d_layer_nm)
            U = float(self.params.D_sign) * float(D_Vnm) * z * 1000.0
            return U - np.mean(U)
        electro = self._electrostatics()
        _, n_bottom = electro.gate_densities(filling_to_density_a2(float(nu_total), self.A_M_A2), float(D_Vnm))
        return electro.update_U_from_density(np.zeros(4), n_bottom)

    def _diagonalize(self, U_meV: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        evals = np.zeros((4, self.n_k, self.n_band), dtype=float)
        evecs = np.zeros((4, self.n_k, self.n_band, self.n_band), dtype=np.complex128)
        for f, model in enumerate(self.models):
            h0 = model.h0_at_U(U_meV)
            for ik in range(self.n_k):
                h = h0[ik] + sigma[f, ik]
                h = 0.5 * (h + h.conj().T)
                vals, vecs = np.linalg.eigh(h)
                evals[f, ik] = vals
                evecs[f, ik] = vecs
        return evals, evecs

    def _layer_density_a2(self, density: ProjectedDensity, n_ref_per_flavor: float) -> np.ndarray:
        n_layer = np.zeros(4, dtype=float)
        ref_fraction = float(n_ref_per_flavor) / float(self.n_band) if self.n_band else 0.0
        for f, model in enumerate(self.models):
            remote = model.remote_density_per_k_layer
            if remote is None:
                remote = np.zeros((self.n_k, 4), dtype=float)
            for ik in range(self.n_k):
                P_fk = density.P[f, ik]
                for layer in range(4):
                    projector = model.layer_mats[ik, layer]
                    active = np.real(np.trace(projector @ P_fk))
                    reference = ref_fraction * np.real(np.trace(projector))
                    n_layer[layer] += self.weights[ik] * (active - reference + remote[ik, layer])
        return n_layer

    def _hartree_update(self, layer_density_a2: np.ndarray, nu_total: float, D_Vnm: float) -> np.ndarray:
        if not self.params.uniform_layer_hartree:
            return self._initial_layer_potential(nu_total, D_Vnm)
        electro = self._electrostatics()
        n_total_a2 = filling_to_density_a2(float(nu_total), self.A_M_A2)
        _, n_bottom = electro.gate_densities(n_total_a2, float(D_Vnm))
        return electro.update_U_from_density(layer_density_a2, n_bottom)

    def _seed_self_energy(self, flavor_nu: np.ndarray) -> np.ndarray:
        centered = np.asarray(flavor_nu, dtype=float) - float(np.mean(flavor_nu))
        sigma = np.zeros((4, self.n_k, self.n_band, self.n_band), dtype=np.complex128)
        for f in range(4):
            sigma[f] = -float(self.params.seed_bias_meV) * centered[f] * np.eye(self.n_band)
        return sigma

    def solve_fixed_nu_D(
        self,
        nu_total: float,
        D_Vnm: float,
        initial_U_meV: np.ndarray | None = None,
    ) -> ProjectedHFResult:
        seeds = flavor_filling_seeds(
            float(nu_total),
            n_random=int(self.params.n_random_seeds),
            seed=int(self.params.seed),
        )
        results = [self._solve_from_seed(name, seed, nu_total, D_Vnm, initial_U_meV) for name, seed in seeds]
        results.sort(key=lambda item: (not item.converged, item.energy_meV_per_cell))
        best = results[0]
        minima = [
            {
                "seed_name": r.seed_name,
                "energy_meV_per_cell": r.energy_meV_per_cell,
                "converged": r.converged,
                "iterations": r.iterations,
                "nu_f": r.nu_f.tolist(),
            }
            for r in results
        ]
        best.local_minima[:] = minima
        return best

    def _solve_from_seed(
        self,
        seed_name: str,
        seed_nu_f: np.ndarray,
        nu_total: float,
        D_Vnm: float,
        initial_U_meV: np.ndarray | None,
    ) -> ProjectedHFResult:
        U = self._initial_layer_potential(nu_total, D_Vnm)
        if initial_U_meV is not None:
            U = U + np.asarray(initial_U_meV, dtype=float)
        sigma = self._seed_self_energy(seed_nu_f)
        density = None
        evals = None
        evecs = None
        layer_density = np.zeros(4, dtype=float)
        converged = False
        message = "max_iter reached"
        kBT = float(self.params.kBT_meV)
        n_ref = self._reference_count()

        for iteration in range(1, int(self.params.max_iter) + 1):
            evals, evecs = self._diagonalize(U, sigma)
            mu, density = density_matrices_from_eigensystems(
                evals,
                evecs,
                self.weights,
                self.A_M_A2,
                float(nu_total),
                n_ref,
                kBT_meV=kBT,
            )

            layer_density = self._layer_density_a2(density, n_ref)
            new_U = self._hartree_update(layer_density, nu_total, D_Vnm)
            new_sigma = build_exchange_self_energy(self.params.exchange_model, density, self.params)
            if new_sigma is None:
                new_sigma = np.zeros_like(sigma)
            else:
                new_sigma = np.asarray(new_sigma, dtype=np.complex128)

            sigma_delta = float(np.max(np.abs(new_sigma - sigma))) if sigma.size else 0.0
            U_delta = float(np.max(np.abs(new_U - U)))
            mix = float(self.params.mixing)
            sigma = (1.0 - mix) * sigma + mix * new_sigma
            U = (1.0 - mix) * U + mix * new_U

            if sigma_delta < float(self.params.tol_self_energy_meV) and U_delta < float(self.params.tol_U_meV):
                converged = True
                message = "converged"
                break
        else:
            iteration = int(self.params.max_iter)
            mu = float("nan")

        assert density is not None and evals is not None and evecs is not None
        energy = band_energy_meV_per_cell(evals, density.occ, self.weights, self.A_M_A2)
        return ProjectedHFResult(
            converged=converged,
            nu_total=float(nu_total),
            D_Vnm=float(D_Vnm),
            seed_name=str(seed_name),
            nu_f=np.asarray(density.nu_f, dtype=float),
            U_meV=np.asarray(U, dtype=float),
            layer_density_a2=np.asarray(layer_density, dtype=float),
            mu_meV=float(density.mu_meV),
            density=density,
            evals_meV=evals,
            evecs=evecs,
            self_energy_meV=sigma,
            energy_meV_per_cell=float(energy),
            iterations=int(iteration),
            message=message,
            local_minima=[],
        )
