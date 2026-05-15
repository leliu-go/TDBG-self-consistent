"""Eight-band projected self-consistent Hartree solver for TDBG."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

from .constants import cm2_to_a2, a2_to_cm2
from .continuum import TDBGContinuumHamiltonian
from .electrostatics import LayerElectrostatics
from .density import fermi, find_mu_for_density
from .mixing import make_mixer
from .solver_full import FullSCFConfig, FullSCFSolver, FullSCFResult


@dataclass
class ProjectedModel:
    """Frozen eight-band model data at a reference layer potential."""
    energies0_meV: np.ndarray       # (Nk, n_active)
    layer_mats: np.ndarray          # (Nk, L=4, n_active, n_active)
    remote_density_per_k_layer: np.ndarray  # (Nk, L), without k weights or degeneracy
    U_ref_meV: np.ndarray
    mu_ref_meV: float
    selected_indices: np.ndarray    # (Nk, n_active)
    isolation_gap_meV: float
    selection_diagnostics: dict | None = None

    @property
    def n_active(self) -> int:
        return int(self.energies0_meV.shape[1])


def _select_active_indices(evals_k: np.ndarray, mu: float, n_active: int, mode: str = "closest") -> np.ndarray:
    dim = len(evals_k)
    if n_active >= dim:
        return np.arange(dim, dtype=int)
    if mode == "closest":
        idx = np.argsort(np.abs(evals_k - mu))[:n_active]
        return np.sort(idx)
    if mode == "contiguous":
        below = int(np.searchsorted(evals_k, mu, side="right"))
        start = below - n_active // 2
        start = max(0, min(start, dim - n_active))
        return np.arange(start, start + n_active, dtype=int)
    raise ValueError("mode must be 'closest', 'contiguous', or 'overlap'")


def _selection_isolation_gap(evals_k: np.ndarray, idx: np.ndarray) -> float:
    active = np.zeros(len(evals_k), dtype=bool)
    active[idx] = True
    outside = np.where(~active)[0]
    if outside.size == 0:
        return np.inf
    gaps = np.abs(evals_k[outside, None] - evals_k[idx][None, :])
    return float(np.min(gaps))


def _initial_overlap_seed(evals: np.ndarray, mu: float, n_active: int) -> int:
    best_seed = 0
    best_gap = -np.inf
    for ik, evals_k in enumerate(evals):
        idx = _select_active_indices(evals_k, mu, n_active, mode="closest")
        gap = _selection_isolation_gap(evals_k, idx)
        if gap > best_gap:
            best_gap = gap
            best_seed = ik
    return best_seed


def _select_by_overlap(prev_evecs: np.ndarray, prev_idx: np.ndarray,
                       next_evecs: np.ndarray, n_active: int) -> tuple[np.ndarray, float, float]:
    prev_subspace = prev_evecs[:, prev_idx]
    scores = np.sum(np.abs(next_evecs.conj().T @ prev_subspace) ** 2, axis=1)
    order = np.argsort(scores)[::-1]
    idx = np.sort(order[:n_active])
    if n_active >= len(scores):
        score_gap = np.inf
    else:
        score_gap = float(scores[order[n_active - 1]] - scores[order[n_active]])
    min_kept_score = float(scores[order[n_active - 1]])
    return idx, score_gap, min_kept_score


def _selection_diagnostics(selected: np.ndarray, mode: str,
                           seed_index: int | None = None,
                           tracked_points: int = 0,
                           ambiguous_overlap_points: int = 0,
                           min_overlap_score_gap: float | None = None,
                           min_kept_overlap_score: float | None = None) -> dict:
    windows = {tuple(row.tolist()) for row in selected}
    diagnostics = {
        "selection": mode,
        "unique_selected_windows": len(windows),
        "selected_first_min": int(selected[:, 0].min()),
        "selected_first_max": int(selected[:, 0].max()),
        "selected_last_min": int(selected[:, -1].min()),
        "selected_last_max": int(selected[:, -1].max()),
    }
    if seed_index is not None:
        diagnostics["overlap_seed_index"] = int(seed_index)
    if mode == "overlap":
        diagnostics.update({
            "tracked_points": int(tracked_points),
            "ambiguous_overlap_points": int(ambiguous_overlap_points),
            "min_overlap_score_gap": float(min_overlap_score_gap if min_overlap_score_gap is not None else 0.0),
            "min_kept_overlap_score": float(min_kept_overlap_score if min_kept_overlap_score is not None else 0.0),
        })
    return diagnostics


def select_active_indices(evals: np.ndarray, evecs: np.ndarray, mu: float, n_active: int,
                          mode: str = "closest", grid_shape: tuple[int, int] | None = None,
                          seed_index: int | None = None,
                          ambiguity_tol: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Select active bands at each k-point.

    ``overlap`` tracks the active subspace by wavefunction overlap. With
    ``grid_shape`` it propagates over the 2D uniform grid; otherwise it tracks
    along the supplied k-point order.
    """
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=np.complex128)
    Nk, dim = evals.shape
    if n_active >= dim:
        selected = np.tile(np.arange(dim, dtype=int), (Nk, 1))
        return selected, _selection_diagnostics(selected, mode)
    if mode in {"closest", "contiguous"}:
        selected = np.asarray([
            _select_active_indices(evals_k, mu, n_active, mode=mode)
            for evals_k in evals
        ], dtype=int)
        return selected, _selection_diagnostics(selected, mode)
    if mode != "overlap":
        raise ValueError("mode must be 'closest', 'contiguous', or 'overlap'")

    if seed_index is None:
        seed_index = _initial_overlap_seed(evals, mu, n_active)
    if seed_index < 0 or seed_index >= Nk:
        raise ValueError(f"seed_index={seed_index} is outside [0, {Nk})")
    selected = np.full((Nk, n_active), -1, dtype=int)
    selected[seed_index] = _select_active_indices(evals[seed_index], mu, n_active, mode="closest")
    min_score_gap = np.inf
    min_kept_score = np.inf
    ambiguous = 0
    tracked = 0

    def track_neighbor(prev: int, nxt: int) -> None:
        nonlocal min_score_gap, min_kept_score, ambiguous, tracked
        idx, score_gap, kept_score = _select_by_overlap(evecs[prev], selected[prev], evecs[nxt], n_active)
        selected[nxt] = idx
        min_score_gap = min(min_score_gap, score_gap)
        min_kept_score = min(min_kept_score, kept_score)
        if score_gap <= ambiguity_tol:
            ambiguous += 1
        tracked += 1

    if grid_shape is None:
        for ik in range(seed_index + 1, Nk):
            track_neighbor(ik - 1, ik)
        for ik in range(seed_index - 1, -1, -1):
            track_neighbor(ik + 1, ik)
    else:
        n1, n2 = grid_shape
        if n1 * n2 != Nk:
            raise ValueError(f"grid_shape={grid_shape} is incompatible with {Nk} k-points")
        visited = np.zeros(Nk, dtype=bool)
        visited[seed_index] = True
        queue = [seed_index]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            i, j = divmod(current, n2)
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * n2 + j)
            if i + 1 < n1:
                neighbors.append((i + 1) * n2 + j)
            if j > 0:
                neighbors.append(i * n2 + j - 1)
            if j + 1 < n2:
                neighbors.append(i * n2 + j + 1)
            for nxt in neighbors:
                if visited[nxt]:
                    continue
                track_neighbor(current, nxt)
                visited[nxt] = True
                queue.append(nxt)

    return selected, _selection_diagnostics(
        selected,
        mode,
        seed_index=seed_index,
        tracked_points=tracked,
        ambiguous_overlap_points=ambiguous,
        min_overlap_score_gap=0.0 if not np.isfinite(min_score_gap) else min_score_gap,
        min_kept_overlap_score=0.0 if not np.isfinite(min_kept_score) else min_kept_score,
    )


def build_projected_model(ham: TDBGContinuumHamiltonian,
                          kpts: np.ndarray,
                          weights: np.ndarray,
                          U_ref_meV: np.ndarray,
                          target_density_cm2: float = 0.0,
                          n_active: int = 8,
                          degeneracy: int = 4,
                          kBT_meV: float = 0.1,
                          selection: str = "closest",
                          grid_shape: tuple[int, int] | None = None) -> ProjectedModel:
    """Diagonalize the full TDBG Hamiltonian and project onto low-energy minibands."""
    target_a2 = cm2_to_a2(target_density_cm2)
    evals, evecs = ham.diagonalize(kpts, U_ref_meV)
    mu_ref = find_mu_for_density(evals, weights, target_a2, degeneracy=degeneracy, kBT_meV=kBT_meV)
    layer_masks = ham.layer_projectors_diagonal()
    Nk, dim = evals.shape
    L = layer_masks.shape[0]
    selected = np.zeros((Nk, n_active), dtype=int)
    energies0 = np.zeros((Nk, n_active), dtype=float)
    layer_mats = np.zeros((Nk, L, n_active, n_active), dtype=np.complex128)
    remote_density_per_k_layer = np.zeros((Nk, L), dtype=float)
    min_iso_gap = np.inf

    occ_ref = fermi(evals - mu_ref, kBT_meV)
    selected, diagnostics = select_active_indices(
        evals,
        evecs,
        mu_ref,
        n_active,
        mode=selection,
        grid_shape=grid_shape,
    )
    for ik in range(Nk):
        idx = selected[ik]
        energies0[ik] = evals[ik, idx]
        V = evecs[ik]  # columns full eigenvectors
        Va = V[:, idx]
        active_mask = np.zeros(dim, dtype=bool)
        active_mask[idx] = True
        # Isolation diagnostic: nearest band outside the selected window at each k.
        outside = np.where(~active_mask)[0]
        if outside.size > 0:
            gaps = np.abs(evals[ik, outside, None] - evals[ik, idx][None, :])
            min_iso_gap = min(min_iso_gap, float(np.min(gaps)))
        for l in range(L):
            mask = layer_masks[l]
            layer_mats[ik, l] = Va.conj().T @ (mask[:, None] * Va)
            W_all_l = np.sum(np.abs(V) ** 2 * mask[:, None], axis=0)
            remote_density_per_k_layer[ik, l] = np.sum((occ_ref[ik, outside] - 0.5) * W_all_l[outside])

    if not np.isfinite(min_iso_gap):
        min_iso_gap = 0.0

    return ProjectedModel(
        energies0_meV=energies0,
        layer_mats=layer_mats,
        remote_density_per_k_layer=remote_density_per_k_layer,
        U_ref_meV=np.asarray(U_ref_meV, dtype=float).copy(),
        mu_ref_meV=float(mu_ref),
        selected_indices=selected,
        isolation_gap_meV=float(min_iso_gap),
        selection_diagnostics=diagnostics,
    )


def build_projected_model_at_mu(ham: TDBGContinuumHamiltonian,
                                kpts: np.ndarray,
                                U_ref_meV: np.ndarray,
                                mu_ref_meV: float,
                                n_active: int = 8,
                                selection: str = "closest") -> ProjectedModel:
    """Project the full Hamiltonian at arbitrary k-points using a supplied reference mu."""
    evals, evecs = ham.diagonalize(kpts, U_ref_meV)
    layer_masks = ham.layer_projectors_diagonal()
    Nk, dim = evals.shape
    L = layer_masks.shape[0]
    selected = np.zeros((Nk, n_active), dtype=int)
    energies0 = np.zeros((Nk, n_active), dtype=float)
    layer_mats = np.zeros((Nk, L, n_active, n_active), dtype=np.complex128)
    min_iso_gap = np.inf

    selected, diagnostics = select_active_indices(
        evals,
        evecs,
        mu_ref_meV,
        n_active,
        mode=selection,
    )
    for ik in range(Nk):
        idx = selected[ik]
        energies0[ik] = evals[ik, idx]
        V = evecs[ik]
        Va = V[:, idx]
        active_mask = np.zeros(dim, dtype=bool)
        active_mask[idx] = True
        outside = np.where(~active_mask)[0]
        if outside.size > 0:
            gaps = np.abs(evals[ik, outside, None] - evals[ik, idx][None, :])
            min_iso_gap = min(min_iso_gap, float(np.min(gaps)))
        for l in range(L):
            mask = layer_masks[l]
            layer_mats[ik, l] = Va.conj().T @ (mask[:, None] * Va)

    if not np.isfinite(min_iso_gap):
        min_iso_gap = 0.0

    return ProjectedModel(
        energies0_meV=energies0,
        layer_mats=layer_mats,
        remote_density_per_k_layer=np.zeros((Nk, L), dtype=float),
        U_ref_meV=np.asarray(U_ref_meV, dtype=float).copy(),
        mu_ref_meV=float(mu_ref_meV),
        selected_indices=selected,
        isolation_gap_meV=float(min_iso_gap),
        selection_diagnostics=diagnostics,
    )


def projected_eigensystem(model: ProjectedModel, U_meV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize H_eff(k;U) for all k."""
    U = np.asarray(U_meV, dtype=float)
    dU = U - model.U_ref_meV
    Nk, nb = model.energies0_meV.shape
    evals = np.empty((Nk, nb), dtype=float)
    evecs = np.empty((Nk, nb, nb), dtype=np.complex128)
    for ik in range(Nk):
        H = np.diag(model.energies0_meV[ik]).astype(np.complex128)
        H += np.einsum("l,lab->ab", dU, model.layer_mats[ik], optimize=True)
        H = 0.5 * (H + H.conj().T)
        e, v = np.linalg.eigh(H)
        evals[ik] = e
        evecs[ik] = v
    return evals, evecs


def projected_total_density(mu: float, evals: np.ndarray, weights: np.ndarray,
                            model: ProjectedModel, degeneracy: int, kBT_meV: float) -> float:
    occ = fermi(evals - mu, kBT_meV)
    active_trace_total = np.real(np.trace(np.sum(model.layer_mats, axis=1), axis1=1, axis2=2))  # (Nk,)
    # For a complete active subspace, active_trace_total = n_active, up to numerical noise.
    active = np.sum(weights[:, None] * (occ - 0.5), axis=(0, 1))
    # Remote density summed over layers, per k, then weighted.
    remote = np.sum(weights[:, None] * model.remote_density_per_k_layer, axis=(0, 1))
    return float(degeneracy * (active + remote))


def find_projected_mu(evals: np.ndarray, weights: np.ndarray, model: ProjectedModel,
                      target_density_a2: float, degeneracy: int = 4, kBT_meV: float = 0.1) -> float:
    lo = float(np.min(evals) - 500.0)
    hi = float(np.max(evals) + 500.0)

    def fn(mu: float) -> float:
        return projected_total_density(mu, evals, weights, model, degeneracy, kBT_meV) - target_density_a2

    flo = fn(lo)
    fhi = fn(hi)
    if flo > 0 or fhi < 0:
        raise RuntimeError(
            f"Projected mu bracket failed: f(lo)={flo:.3e}, f(hi)={fhi:.3e}. "
            "Increase n_active or use a full model reference."
        )
    return float(brentq(fn, lo, hi, maxiter=200, xtol=1e-10))


def projected_layer_density(mu: float, evals: np.ndarray, evecs: np.ndarray, weights: np.ndarray,
                            model: ProjectedModel, degeneracy: int = 4, kBT_meV: float = 0.1) -> np.ndarray:
    occ = fermi(evals - mu, kBT_meV)
    Nk, nb = evals.shape
    L = model.layer_mats.shape[1]
    n_layer = np.zeros(L, dtype=float)
    for ik in range(Nk):
        C = evecs[ik]  # columns are effective eigenvectors in active miniband basis
        for l in range(L):
            M = C.conj().T @ model.layer_mats[ik, l] @ C
            W = np.real(np.diag(M))
            trace_active = np.real(np.trace(model.layer_mats[ik, l]))
            active_part = np.sum(occ[ik] * W) - 0.5 * trace_active
            n_layer[l] += weights[ik] * (active_part + model.remote_density_per_k_layer[ik, l])
    return degeneracy * n_layer


@dataclass
class ProjectedSCFResult:
    U_meV: np.ndarray
    mu_meV: float
    n_layer_a2: np.ndarray
    converged: bool
    residual_meV: float
    iterations: int
    history: list[dict]
    model: ProjectedModel
    evals: np.ndarray
    evecs: np.ndarray

    @property
    def n_layer_cm2(self) -> np.ndarray:
        return np.array([a2_to_cm2(x) for x in self.n_layer_a2])


@dataclass
class ProjectedSCFConfig:
    target_density_cm2: float = 0.0
    D_Vnm: float = 0.0
    n_active: int = 8
    degeneracy: int = 4
    kBT_meV: float = 0.1
    max_iter: int = 100
    tol_meV: float = 1e-5
    eps_perp: float = 4.0
    d_layer_nm: float = 0.335
    D_sign: float = -1.0
    mixer: str = "anderson"
    mixer_kwargs: dict | None = None
    selection: str = "overlap"
    grid_shape: tuple[int, int] | None = None
    projector_refreshes: int = 0
    reference_mode: str = "full_scf"  # full_scf or supplied_U
    supplied_U_ref_meV: np.ndarray | None = None
    full_scf_config: FullSCFConfig | None = None


class ProjectedSCFSolver:
    def __init__(self, ham: TDBGContinuumHamiltonian, kpts: np.ndarray, weights: np.ndarray,
                 grid_shape: tuple[int, int] | None = None):
        self.ham = ham
        self.kpts = np.asarray(kpts, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.grid_shape = grid_shape

    def _reference_U(self, config: ProjectedSCFConfig) -> np.ndarray:
        if config.reference_mode == "supplied_U":
            if config.supplied_U_ref_meV is None:
                raise ValueError("supplied_U_ref_meV must be provided when reference_mode='supplied_U'")
            U = np.asarray(config.supplied_U_ref_meV, dtype=float).copy()
            U -= np.mean(U)
            return U
        if config.reference_mode == "full_scf":
            full_cfg = config.full_scf_config or FullSCFConfig(
                target_density_cm2=config.target_density_cm2,
                D_Vnm=config.D_Vnm,
                degeneracy=config.degeneracy,
                kBT_meV=config.kBT_meV,
                max_iter=60,
                tol_meV=max(config.tol_meV, 1e-4),
                eps_perp=config.eps_perp,
                d_layer_nm=config.d_layer_nm,
                D_sign=config.D_sign,
                mixer=config.mixer,
                mixer_kwargs=config.mixer_kwargs,
                keep_eigensystem=False,
            )
            full = FullSCFSolver(self.ham, self.kpts, self.weights).solve(full_cfg)
            return full.U_meV
        raise ValueError("reference_mode must be 'full_scf' or 'supplied_U'")

    def solve(self, config: ProjectedSCFConfig) -> ProjectedSCFResult:
        U_ref = self._reference_U(config)
        result = None
        for refresh in range(config.projector_refreshes + 1):
            model = build_projected_model(
                self.ham, self.kpts, self.weights, U_ref,
                target_density_cm2=config.target_density_cm2,
                n_active=config.n_active,
                degeneracy=config.degeneracy,
                kBT_meV=config.kBT_meV,
                selection=config.selection,
                grid_shape=config.grid_shape or self.grid_shape,
            )
            result = self._solve_fixed_projector(model, config)
            U_ref = result.U_meV.copy()
        assert result is not None
        return result

    def _solve_fixed_projector(self, model: ProjectedModel, config: ProjectedSCFConfig) -> ProjectedSCFResult:
        target_a2 = cm2_to_a2(config.target_density_cm2)
        electro = LayerElectrostatics(n_layers=4, d_layer_nm=config.d_layer_nm, eps_perp=config.eps_perp, D_sign=config.D_sign)
        _, n_bottom_gate_a2 = electro.gate_densities(target_a2, config.D_Vnm)
        mixer = make_mixer(config.mixer, **(config.mixer_kwargs or {}))
        U = model.U_ref_meV.copy()
        U -= np.mean(U)
        history: list[dict] = []
        converged = False
        residual = np.inf

        for it in range(config.max_iter):
            evals, evecs = projected_eigensystem(model, U)
            mu = find_projected_mu(evals, self.weights, model, target_a2,
                                   degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
            n_layer = projected_layer_density(mu, evals, evecs, self.weights, model,
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

        evals, evecs = projected_eigensystem(model, U)
        mu = find_projected_mu(evals, self.weights, model, target_a2,
                               degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
        n_layer = projected_layer_density(mu, evals, evecs, self.weights, model,
                                          degeneracy=config.degeneracy, kBT_meV=config.kBT_meV)
        U_check = electro.update_U_from_density(n_layer, n_bottom_gate_a2)
        residual = float(np.max(np.abs(U_check - U)))
        return ProjectedSCFResult(
            U_meV=U,
            mu_meV=float(mu),
            n_layer_a2=n_layer,
            converged=bool(residual < config.tol_meV),
            residual_meV=residual,
            iterations=len(history),
            history=history,
            model=model,
            evals=evals,
            evecs=evecs,
        )
