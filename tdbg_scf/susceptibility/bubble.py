from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import SusceptibilityParams
from .qmesh import QPoint, folded_indices_and_shifts_for_q, folded_indices_for_q
from .stoner_reference import StonerReference
from .vertex import generalized_stoner_lambda, legacy_scalar_total_lambda, make_valley_vertex_models


@dataclass
class ChiQResult:
    q: QPoint
    chi_total_cell_meV_inv: float
    chi_K_cell_meV_inv: float
    chi_Kp_cell_meV_inv: float
    lambda_u: float
    lambda_u_plus_hund: float
    chi_layer_cell_meV_inv: np.ndarray | None = None
    chi_s0_cell_meV_inv: float | None = None
    chi_sD_cell_meV_inv: float | None = None
    layer_leading_eig_cell_meV_inv: float | None = None
    layer_dipole_overlap: float | None = None
    chi_K_plus_cell_meV_inv: float | None = None
    chi_Kp_plus_cell_meV_inv: float | None = None
    chi_K_minus_cell_meV_inv: float | None = None
    chi_Kp_minus_cell_meV_inv: float | None = None
    lambda_su4_diag_plus: float | None = None
    lambda_su4_diag_minus: float | None = None
    lambda_su4_diag_selected: float | None = None
    lambda_su2_hund_factor2_plus: float | None = None
    lambda_su2_hund_factor2_minus: float | None = None
    lambda_su2_hund_factor2_selected: float | None = None
    lambda_legacy_scalar_total_plus: float | None = None
    lambda_legacy_scalar_total_minus: float | None = None
    lambda_legacy_scalar_total_selected: float | None = None
    lambda_legacy_offdiag_J_plus: float | None = None
    lambda_legacy_offdiag_J_minus: float | None = None
    lambda_legacy_offdiag_J_selected: float | None = None
    selected_vertex_model: str | None = None
    selected_spin_flip_direction: str | None = None
    selected_lambda_raw: float | None = None
    selected_valley_eigenvector_K_real: float | None = None
    selected_valley_eigenvector_K_imag: float | None = None
    selected_valley_eigenvector_Kp_real: float | None = None
    selected_valley_eigenvector_Kp_imag: float | None = None
    selected_chi_s0: float | None = None
    selected_chi_sD: float | None = None
    selected_layer_dipole_overlap: float | None = None
    selected_layer_dipole_ratio: float | None = None
    su4_diag_spin_flip_direction: str | None = None
    su4_diag_layer_dipole_overlap: float | None = None
    su4_diag_layer_dipole_ratio: float | None = None
    su2_hund_factor2_spin_flip_direction: str | None = None
    su2_hund_factor2_layer_dipole_overlap: float | None = None
    su2_hund_factor2_layer_dipole_ratio: float | None = None
    extra: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        out = self.q.to_record()
        out.update(
            {
                "chi_total_cell_meV_inv": float(self.chi_total_cell_meV_inv),
                "chi_K_cell_meV_inv": float(self.chi_K_cell_meV_inv),
                "chi_Kp_cell_meV_inv": float(self.chi_Kp_cell_meV_inv),
                "lambda_u": float(self.lambda_u),
                "lambda_u_plus_hund": float(self.lambda_u_plus_hund),
            }
        )
        if self.chi_s0_cell_meV_inv is not None:
            out["chi_s0_cell_meV_inv"] = float(self.chi_s0_cell_meV_inv)
        if self.chi_sD_cell_meV_inv is not None:
            out["chi_sD_cell_meV_inv"] = float(self.chi_sD_cell_meV_inv)
        if self.layer_leading_eig_cell_meV_inv is not None:
            out["layer_leading_eig_cell_meV_inv"] = float(self.layer_leading_eig_cell_meV_inv)
        if self.layer_dipole_overlap is not None:
            out["layer_dipole_overlap"] = float(self.layer_dipole_overlap)
        if self.chi_layer_cell_meV_inv is not None:
            mat = np.asarray(self.chi_layer_cell_meV_inv)
            for l in range(mat.shape[0]):
                for lp in range(mat.shape[1]):
                    out[f"chi_layer_L{l + 1}_L{lp + 1}_cell_meV_inv"] = float(np.real(mat[l, lp]))
        for name in (
            "chi_K_plus_cell_meV_inv",
            "chi_Kp_plus_cell_meV_inv",
            "chi_K_minus_cell_meV_inv",
            "chi_Kp_minus_cell_meV_inv",
            "lambda_su4_diag_plus",
            "lambda_su4_diag_minus",
            "lambda_su4_diag_selected",
            "lambda_su2_hund_factor2_plus",
            "lambda_su2_hund_factor2_minus",
            "lambda_su2_hund_factor2_selected",
            "lambda_legacy_scalar_total_plus",
            "lambda_legacy_scalar_total_minus",
            "lambda_legacy_scalar_total_selected",
            "lambda_legacy_offdiag_J_plus",
            "lambda_legacy_offdiag_J_minus",
            "lambda_legacy_offdiag_J_selected",
            "selected_lambda_raw",
            "selected_valley_eigenvector_K_real",
            "selected_valley_eigenvector_K_imag",
            "selected_valley_eigenvector_Kp_real",
            "selected_valley_eigenvector_Kp_imag",
            "selected_chi_s0",
            "selected_chi_sD",
            "selected_layer_dipole_overlap",
            "selected_layer_dipole_ratio",
            "su4_diag_layer_dipole_overlap",
            "su4_diag_layer_dipole_ratio",
            "su2_hund_factor2_layer_dipole_overlap",
            "su2_hund_factor2_layer_dipole_ratio",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        for name in (
            "selected_vertex_model",
            "selected_spin_flip_direction",
            "su4_diag_spin_flip_direction",
            "su2_hund_factor2_spin_flip_direction",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out


def fermi_occ(E_minus_mu_meV: np.ndarray, kBT_meV: float) -> np.ndarray:
    T = max(float(kBT_meV), 1e-12)
    x = np.clip(np.asarray(E_minus_mu_meV, dtype=float) / T, -100.0, 100.0)
    return 1.0 / (np.exp(x) + 1.0)


def minus_fermi_derivative(E_minus_mu_meV: np.ndarray, kBT_meV: float) -> np.ndarray:
    T = max(float(kBT_meV), 1e-12)
    x = np.clip(np.asarray(E_minus_mu_meV, dtype=float) / (2.0 * T), -50.0, 50.0)
    return 1.0 / (4.0 * T * np.cosh(x) ** 2)


def gaussian_delta(E_minus_mu_meV: np.ndarray, sigma_meV: float) -> np.ndarray:
    sigma = float(sigma_meV)
    if sigma <= 0.0:
        raise ValueError("sigma_meV must be positive")
    x = np.asarray(E_minus_mu_meV, dtype=float) / sigma
    return np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sigma)


def lindhard_static_ratio(
    E_initial_meV: np.ndarray,
    E_final_meV: np.ndarray,
    mu_meV: float,
    kBT_meV: float,
    denom_tol_meV: float,
) -> np.ndarray:
    """Return (f_i - f_f) / (E_f - E_i), shape (n_final, n_initial)."""

    Ei = np.asarray(E_initial_meV, dtype=float)
    Ef = np.asarray(E_final_meV, dtype=float)
    fi = fermi_occ(Ei - float(mu_meV), kBT_meV)
    ff = fermi_occ(Ef - float(mu_meV), kBT_meV)

    den = Ef[:, None] - Ei[None, :]
    num = fi[None, :] - ff[:, None]
    midpoint = 0.5 * (Ef[:, None] + Ei[None, :])
    limiting = minus_fermi_derivative(midpoint - float(mu_meV), kBT_meV)
    mask = np.abs(den) > float(denom_tol_meV)
    out = np.array(limiting, dtype=float, copy=True)
    out[mask] = num[mask] / den[mask]
    return out


def lindhard_static_ratio_flavor_mu(
    E_initial_meV: np.ndarray,
    E_final_meV: np.ndarray,
    mu_initial_meV: float,
    mu_final_meV: float,
    kBT_meV: float,
    denom_tol_meV: float,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Return flavor-resolved (f_i - f_f)/(E_f - E_i)."""

    Ei = np.asarray(E_initial_meV, dtype=float)
    Ef = np.asarray(E_final_meV, dtype=float)
    mui = float(mu_initial_meV)
    muf = float(mu_final_meV)
    fi = fermi_occ(Ei - mui, kBT_meV)
    ff = fermi_occ(Ef - muf, kBT_meV)
    den = Ef[:, None] - Ei[None, :]
    num = fi[None, :] - ff[:, None]
    midpoint = 0.5 * (Ef[:, None] + Ei[None, :])
    limiting_mu = 0.5 * (mui + muf)
    limiting = minus_fermi_derivative(midpoint - limiting_mu, kBT_meV)
    mask = np.abs(den) > float(denom_tol_meV)
    out = np.array(limiting, dtype=float, copy=True)
    out[mask] = num[mask] / den[mask]
    small = ~mask
    noneq = small & (np.abs(num) > 1e-10) & (abs(mui - muf) > 1e-10)
    if np.any(noneq):
        sign = np.sign(den[noneq])
        sign = np.where(sign == 0.0, np.sign(num[noneq]), sign)
        sign = np.where(sign == 0.0, 1.0, sign)
        out[noneq] = num[noneq] / (sign * float(denom_tol_meV))
    diag = {
        "regulated_nonequilibrium_pairs": int(np.count_nonzero(noneq)),
        "regulated_nonequilibrium_abs_weight": float(np.sum(np.abs(num[noneq]))),
    }
    if return_diagnostics:
        return out, diag
    return out


def band_indices_near_mu(
    energies_meV: np.ndarray,
    mu_meV: float,
    energy_window_meV: float | None,
    max_bands: int | None,
) -> np.ndarray:
    """Select bands for the bubble sum at one k point."""

    e = np.asarray(energies_meV, dtype=float)
    if e.ndim != 1:
        raise ValueError("energies_meV must be 1D")

    distance = np.abs(e - float(mu_meV))
    order = np.argsort(distance)
    if energy_window_meV is None:
        idx = np.arange(e.size, dtype=int)
    else:
        idx = np.where(distance <= float(energy_window_meV))[0]
        if idx.size == 0:
            keep = 1 if max_bands is None else max(1, min(int(max_bands), e.size))
            idx = order[:keep]

    if max_bands is not None and idx.size > int(max_bands):
        local_order = np.argsort(distance[idx])
        idx = idx[local_order[: int(max_bands)]]
    return np.sort(idx.astype(int))


def _jdos_band_indices_near_mu(
    energies_meV: np.ndarray,
    mu_meV: float,
    energy_window_meV: float | None,
    max_bands: int | None,
) -> np.ndarray:
    e = np.asarray(energies_meV, dtype=float)
    if e.ndim != 1:
        raise ValueError("energies_meV must be 1D")
    distance = np.abs(e - float(mu_meV))
    if energy_window_meV is None:
        idx = np.arange(e.size, dtype=int)
    else:
        idx = np.where(distance <= float(energy_window_meV))[0]
    if max_bands is not None and idx.size > int(max_bands):
        order = np.argsort(distance[idx])
        idx = idx[order[: int(max_bands)]]
    return np.sort(idx.astype(int))


def _safe_flavor_label(name: str, index: int) -> str:
    label = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return label or f"flavor_{index}"


def compute_fermi_surface_jdos_q(
    q: QPoint,
    grid_shape: tuple[int, int],
    weights: np.ndarray,
    A_M_A2: float,
    evals_by_flavor_meV: list[np.ndarray] | tuple[np.ndarray, ...],
    mu_by_flavor_meV: list[float] | tuple[float, ...] | np.ndarray,
    sigma_meV: float,
    energy_window_meV: float | None,
    max_bands_per_k: int | None,
    evals_q_by_flavor_meV: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    flavor_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Gaussian-broadened FS autocorrelation/JDOS at one Q.

    N(Q) = sum_{k,n,m} delta_sigma(E_n(k)-mu) delta_sigma(E_m(k+Q)-mu).
    The result is reported per moire cell in meV^-2.
    """

    weights_arr = np.asarray(weights, dtype=float)
    Nk = int(np.prod(np.asarray(grid_shape, dtype=int)))
    if weights_arr.shape != (Nk,):
        raise ValueError("weights must have shape (grid_n1 * grid_n2,)")
    if len(evals_by_flavor_meV) != len(mu_by_flavor_meV):
        raise ValueError("evals_by_flavor_meV and mu_by_flavor_meV must have the same length")
    if evals_q_by_flavor_meV is not None and len(evals_q_by_flavor_meV) != len(evals_by_flavor_meV):
        raise ValueError("evals_q_by_flavor_meV must match evals_by_flavor_meV")

    folded_mode = evals_q_by_flavor_meV is None
    kq_indices = folded_indices_for_q(q, grid_shape) if folded_mode else np.arange(Nk, dtype=int)
    names = flavor_names if flavor_names is not None else tuple(f"flavor_{i}" for i in range(len(evals_by_flavor_meV)))

    total = 0.0
    out: dict[str, float] = {}
    for flavor_index, (evals_i_all, mu) in enumerate(zip(evals_by_flavor_meV, mu_by_flavor_meV)):
        evals_i_all = np.asarray(evals_i_all, dtype=float)
        evals_f_all = evals_i_all if folded_mode else np.asarray(evals_q_by_flavor_meV[flavor_index], dtype=float)
        if evals_i_all.ndim != 2 or evals_f_all.ndim != 2:
            raise ValueError("flavor eval arrays must have shape (Nk, n_bands)")
        if evals_i_all.shape[0] != Nk or evals_f_all.shape[0] != Nk:
            raise ValueError("flavor eval arrays must use the same k grid as weights")

        flavor_value = 0.0
        for ik in range(Nk):
            ikq = int(kq_indices[ik])
            Ei_all = evals_i_all[ik]
            Ef_all = evals_f_all[ikq]
            idx_i = _jdos_band_indices_near_mu(Ei_all, float(mu), energy_window_meV, max_bands_per_k)
            idx_f = _jdos_band_indices_near_mu(Ef_all, float(mu), energy_window_meV, max_bands_per_k)
            if idx_i.size == 0 or idx_f.size == 0:
                continue
            di = gaussian_delta(Ei_all[idx_i] - float(mu), sigma_meV)
            df = gaussian_delta(Ef_all[idx_f] - float(mu), sigma_meV)
            flavor_value += float(A_M_A2) * float(weights_arr[ik]) * float(np.sum(di) * np.sum(df))

        label = _safe_flavor_label(str(names[flavor_index]), flavor_index)
        out[f"jdos_{label}_cell_meV_inv2"] = float(flavor_value)
        total += float(flavor_value)

    out["jdos_total_cell_meV_inv2"] = float(total)
    return out


def compute_spinflip_nesting_q(
    q: QPoint,
    grid_shape: tuple[int, int],
    weights: np.ndarray,
    A_M_A2: float,
    evals_i: np.ndarray,
    evecs_i: np.ndarray,
    evals_f: np.ndarray,
    evecs_f: np.ndarray,
    mu_i_meV: float,
    mu_f_meV: float,
    params: SusceptibilityParams,
    folded_mode: bool,
    folded_with_G_shift: bool = False,
    G_indices: np.ndarray | None = None,
) -> float:
    Nk = evals_i.shape[0]
    if folded_mode and folded_with_G_shift:
        kq_indices, g_shifts = folded_indices_and_shifts_for_q(q, grid_shape)
    else:
        kq_indices = folded_indices_for_q(q, grid_shape) if folded_mode else np.arange(Nk, dtype=int)
        g_shifts = np.zeros((Nk, 2), dtype=int)

    total = 0.0
    for ik in range(Nk):
        ikq = int(kq_indices[ik])
        Ei_all = np.asarray(evals_i[ik], dtype=float)
        Ef_all = np.asarray(evals_f[ikq], dtype=float)
        idx_i = _jdos_band_indices_near_mu(Ei_all, mu_i_meV, params.energy_window_meV, params.max_bands_per_k)
        idx_f = _jdos_band_indices_near_mu(Ef_all, mu_f_meV, params.energy_window_meV, params.max_bands_per_k)
        if idx_i.size == 0 or idx_f.size == 0:
            continue
        Vi = evecs_i[ik][:, idx_i]
        Vf_all = evecs_f[ikq]
        if folded_with_G_shift:
            if G_indices is None and np.any(g_shifts[ik] != 0):
                raise ValueError("folded_grid_with_G_shift requires plane-wave G indices")
            if G_indices is not None:
                Vf_all = _shift_plane_wave_evecs(Vf_all, G_indices, g_shifts[ik])
        Vf = Vf_all[:, idx_f]
        lam = _pair_form_factors_total(Vf, Vi)
        di = gaussian_delta(Ei_all[idx_i] - float(mu_i_meV), params.nesting_sigma_meV)
        df = gaussian_delta(Ef_all[idx_f] - float(mu_f_meV), params.nesting_sigma_meV)
        total += float(A_M_A2) * float(weights[ik]) * float(np.sum((df[:, None] * di[None, :]) * np.abs(lam) ** 2))
    return float(total)


def _pair_form_factors_total(Vf: np.ndarray, Vi: np.ndarray) -> np.ndarray:
    return Vf.conj().T @ Vi


def _pair_form_factors_layers(Vf: np.ndarray, Vi: np.ndarray, layer_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(layer_masks, dtype=float)
    out = np.empty((masks.shape[0], Vf.shape[1], Vi.shape[1]), dtype=np.complex128)
    for l in range(masks.shape[0]):
        out[l] = Vf.conj().T @ (masks[l, :, None] * Vi)
    return out


def _shift_plane_wave_evecs(evecs: np.ndarray, G_indices: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Represent a folded final eigenvector in the unfolded k+Q plane-wave basis."""

    g_pairs = np.asarray(G_indices, dtype=int)
    shift_tuple = (int(shift[0]), int(shift[1]))
    inv = {tuple(pair): idx for idx, pair in enumerate(g_pairs.tolist())}
    nG = int(g_pairs.shape[0])
    out = np.zeros_like(evecs)
    for target_g, pair in enumerate(g_pairs.tolist()):
        source_g = inv.get((int(pair[0]) + shift_tuple[0], int(pair[1]) + shift_tuple[1]))
        if source_g is None:
            continue
        for orb in range(4):
            out[4 * target_g + orb, :] = evecs[4 * source_g + orb, :]
            out[4 * (nG + target_g) + orb, :] = evecs[4 * (nG + source_g) + orb, :]
    return out


def _accumulate_one_valley_pair(
    evals_i: np.ndarray,
    evecs_i: np.ndarray,
    evals_f: np.ndarray,
    evecs_f: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    mu_i_meV: float,
    mu_f_meV: float,
    q: QPoint,
    grid_shape: tuple[int, int],
    layer_masks: np.ndarray | None,
    params: SusceptibilityParams,
    folded_mode: bool,
    folded_with_G_shift: bool = False,
    G_indices: np.ndarray | None = None,
) -> tuple[float, np.ndarray | None, dict[str, float]]:
    """Bubble for one valley spin pair: down(k) -> up(k+Q)."""

    Nk = evals_i.shape[0]
    if evals_f.shape[0] != Nk:
        raise ValueError("initial and final eval arrays must have the same Nk")
    if weights.shape != (Nk,):
        raise ValueError("weights must have shape (Nk,)")

    if folded_mode and folded_with_G_shift:
        kq_indices, g_shifts = folded_indices_and_shifts_for_q(q, grid_shape)
    else:
        kq_indices = folded_indices_for_q(q, grid_shape) if folded_mode else np.arange(Nk, dtype=int)
        g_shifts = np.zeros((Nk, 2), dtype=int)
    chi = 0.0
    diagnostics = {
        "regulated_nonequilibrium_pairs": 0,
        "regulated_nonequilibrium_abs_weight": 0.0,
    }
    chi_layer = None
    if layer_masks is not None and params.include_layer_matrix:
        chi_layer = np.zeros((int(layer_masks.shape[0]), int(layer_masks.shape[0])), dtype=np.complex128)

    for ik in range(Nk):
        ikq = int(kq_indices[ik])
        Ei_all = np.asarray(evals_i[ik], dtype=float)
        Ef_all = np.asarray(evals_f[ikq], dtype=float)
        idx_i = band_indices_near_mu(Ei_all, mu_i_meV, params.energy_window_meV, params.max_bands_per_k)
        idx_f = band_indices_near_mu(Ef_all, mu_f_meV, params.energy_window_meV, params.max_bands_per_k)

        Ei = Ei_all[idx_i]
        Ef = Ef_all[idx_f]
        R, diag = lindhard_static_ratio_flavor_mu(
            Ei,
            Ef,
            mu_i_meV,
            mu_f_meV,
            params.kBT_meV,
            params.denom_tol_meV,
            return_diagnostics=True,
        )
        diagnostics["regulated_nonequilibrium_pairs"] += int(diag["regulated_nonequilibrium_pairs"])
        diagnostics["regulated_nonequilibrium_abs_weight"] += float(diag["regulated_nonequilibrium_abs_weight"])
        Vi = evecs_i[ik][:, idx_i]
        Vf_all = evecs_f[ikq]
        if folded_with_G_shift:
            if G_indices is None and np.any(g_shifts[ik] != 0):
                raise ValueError("folded_grid_with_G_shift requires plane-wave G indices")
            if G_indices is not None:
                Vf_all = _shift_plane_wave_evecs(Vf_all, G_indices, g_shifts[ik])
        Vf = Vf_all[:, idx_f]

        lam = _pair_form_factors_total(Vf, Vi)
        prefactor = float(A_M_A2) * float(weights[ik])
        chi += prefactor * float(np.real(np.sum(R * np.abs(lam) ** 2)))

        if chi_layer is not None:
            lam_l = _pair_form_factors_layers(Vf, Vi, layer_masks)
            chi_layer += prefactor * np.einsum("mn,lmn,pmn->lp", R, lam_l, lam_l.conj(), optimize=True)

    if chi_layer is not None and params.hermitize_layer_matrix:
        chi_layer = 0.5 * (chi_layer + chi_layer.conj().T)
    return float(np.real(chi)), chi_layer, diagnostics


def _lambda_u_plus_hund(chi_K: float, chi_Kp: float, ref: StonerReference) -> float:
    vals = np.asarray([max(float(chi_K), 0.0), max(float(chi_Kp), 0.0)], dtype=float)
    sqrt_chi = np.sqrt(vals)
    gamma = np.asarray(
        [[float(ref.u_cell_meV), float(ref.J_cell_meV)], [float(ref.J_cell_meV), float(ref.u_cell_meV)]],
        dtype=float,
    )
    matrix = sqrt_chi[:, None] * gamma * sqrt_chi[None, :]
    return float(np.linalg.eigvalsh(matrix)[-1])


def _layer_diagnostics(chi_layer: np.ndarray, params: SusceptibilityParams) -> tuple[float, float, float, float]:
    mat = np.real(0.5 * (chi_layer + chi_layer.conj().T))
    L = mat.shape[0]
    one = np.ones(L, dtype=float)
    zeta = np.asarray(params.layer_dipole_zeta, dtype=float)
    if zeta.shape != (L,):
        raise ValueError(f"layer_dipole_zeta has shape {zeta.shape}; expected {(L,)}")

    chi_s0 = float(one @ mat @ one)
    chi_sD = float(zeta @ mat @ zeta)
    evals, evecs = np.linalg.eigh(mat)
    leading = float(evals[-1])
    v = np.asarray(evecs[:, -1], dtype=np.complex128)
    z = zeta.astype(np.complex128)
    denom = float(np.vdot(v, v).real * np.vdot(z, z).real)
    overlap = 0.0 if denom <= 0.0 else float(abs(np.vdot(v, z)) ** 2 / denom)
    return chi_s0, chi_sD, leading, overlap


def _spin_flip_enabled(params: SusceptibilityParams, direction: str) -> bool:
    return str(params.spin_flip_mode) in {direction, "both_pm"}


def _model_selected(
    plus_value: float | None,
    minus_value: float | None,
) -> tuple[float, str]:
    values = []
    if plus_value is not None:
        values.append((float(plus_value), "plus"))
    if minus_value is not None:
        values.append((float(minus_value), "minus"))
    if not values:
        return float("nan"), "none"
    return max(values, key=lambda item: item[0])


def _selected_layer_diagnostics(
    layer_K: np.ndarray | None,
    layer_Kp: np.ndarray | None,
    valley_vec: np.ndarray,
    params: SusceptibilityParams,
) -> tuple[float | None, float | None, float | None, float | None, np.ndarray | None]:
    if layer_K is None or layer_Kp is None:
        return None, None, None, None, None
    v = np.asarray(valley_vec, dtype=np.complex128)
    norm = float(np.vdot(v, v).real)
    if norm <= 0.0:
        weights = np.array([0.5, 0.5], dtype=float)
    else:
        weights = np.abs(v) ** 2 / norm
    mat = float(weights[0]) * layer_K + float(weights[1]) * layer_Kp
    if params.hermitize_layer_matrix:
        mat = 0.5 * (mat + mat.conj().T)
    chi_s0, chi_sD, _, overlap = _layer_diagnostics(mat, params)
    ratio = float(chi_sD / max(abs(chi_s0), 1e-30))
    return chi_s0, chi_sD, overlap, ratio, mat


def compute_transverse_chi_q(
    q: QPoint,
    grid_shape: tuple[int, int],
    weights: np.ndarray,
    A_M_A2: float,
    evals_K_meV: np.ndarray,
    evecs_K: np.ndarray,
    evals_Kp_meV: np.ndarray,
    evecs_Kp: np.ndarray,
    ref: StonerReference,
    params: SusceptibilityParams,
    layer_masks: np.ndarray | None = None,
    evals_K_q_meV: np.ndarray | None = None,
    evecs_K_q: np.ndarray | None = None,
    evals_Kp_q_meV: np.ndarray | None = None,
    evecs_Kp_q: np.ndarray | None = None,
    plane_wave_G_indices: np.ndarray | None = None,
) -> ChiQResult:
    """Compute generalized Stoner-after chi(Q) for both transverse channels."""

    if params.q_mode not in {"folded_grid", "unfolded_diagonalize", "folded_grid_with_G_shift"}:
        raise ValueError("params.q_mode must be 'folded_grid', 'folded_grid_with_G_shift', or 'unfolded_diagonalize'")
    if params.occupation_mode not in {"flavor_mu", "common_mu", "equilibrium_common_mu"}:
        raise ValueError("params.occupation_mode must be 'flavor_mu', 'common_mu', or 'equilibrium_common_mu'")
    if params.spin_flip_mode not in {"plus", "minus", "both_pm"}:
        raise ValueError("params.spin_flip_mode must be 'plus', 'minus', or 'both_pm'")

    sigma = np.asarray(ref.sigma_shifted_f_meV, dtype=float)
    mu_f = np.asarray(ref.mu_f_meV, dtype=float)
    mu_bar = mu_f + sigma
    if params.occupation_mode == "common_mu":
        mu_bar = np.full(4, float(ref.mu_common_meV), dtype=float)
    if params.occupation_mode == "equilibrium_common_mu":
        mu_bar = np.full(4, float(ref.mu_eq_meV), dtype=float)
    folded_mode = params.q_mode in {"folded_grid", "folded_grid_with_G_shift"}
    folded_with_G_shift = params.q_mode == "folded_grid_with_G_shift"

    if folded_mode:
        evals_K_up_q = evals_K_meV + sigma[0]
        evals_K_down_q = evals_K_meV + sigma[2]
        evecs_K_f = evecs_K
        evals_Kp_up_q = evals_Kp_meV + sigma[1]
        evals_Kp_down_q = evals_Kp_meV + sigma[3]
        evecs_Kp_f = evecs_Kp
    else:
        if evals_K_q_meV is None or evecs_K_q is None or evals_Kp_q_meV is None or evecs_Kp_q is None:
            raise ValueError("unfolded_diagonalize mode requires k+Q eigensystems")
        evals_K_up_q = evals_K_q_meV + sigma[0]
        evals_K_down_q = evals_K_q_meV + sigma[2]
        evecs_K_f = evecs_K_q
        evals_Kp_up_q = evals_Kp_q_meV + sigma[1]
        evals_Kp_down_q = evals_Kp_q_meV + sigma[3]
        evecs_Kp_f = evecs_Kp_q

    weights_arr = np.asarray(weights, dtype=float)
    chi_K_plus = chi_Kp_plus = chi_K_minus = chi_Kp_minus = None
    layer_K_plus = layer_Kp_plus = layer_K_minus = layer_Kp_minus = None
    extra: dict = {}

    if params.include_jdos:
        mu_base = np.asarray(mu_bar, dtype=float) - sigma
        evals_by_flavor = (evals_K_meV, evals_Kp_meV, evals_K_meV, evals_Kp_meV)
        if folded_mode:
            evals_q_by_flavor = None
        else:
            evals_q_by_flavor = (evals_K_q_meV, evals_Kp_q_meV, evals_K_q_meV, evals_Kp_q_meV)
        extra.update(
            compute_fermi_surface_jdos_q(
                q=q,
                grid_shape=grid_shape,
                weights=weights_arr,
                A_M_A2=float(A_M_A2),
                evals_by_flavor_meV=evals_by_flavor,
                mu_by_flavor_meV=mu_base,
                sigma_meV=float(params.jdos_sigma_meV),
                energy_window_meV=params.energy_window_meV,
                max_bands_per_k=params.max_bands_per_k,
                evals_q_by_flavor_meV=evals_q_by_flavor,
                flavor_names=ref.flavor_names,
            )
        )

    if _spin_flip_enabled(params, "plus"):
        chi_K_plus, layer_K_plus, diag_K_plus = _accumulate_one_valley_pair(
            evals_i=evals_K_meV + sigma[2],
            evecs_i=evecs_K,
            evals_f=evals_K_up_q,
            evecs_f=evecs_K_f,
            weights=weights_arr,
            A_M_A2=float(A_M_A2),
            mu_i_meV=float(mu_bar[2]),
            mu_f_meV=float(mu_bar[0]),
            q=q,
            grid_shape=grid_shape,
            layer_masks=layer_masks,
            params=params,
            folded_mode=folded_mode,
            folded_with_G_shift=folded_with_G_shift,
            G_indices=plane_wave_G_indices,
        )
        chi_Kp_plus, layer_Kp_plus, diag_Kp_plus = _accumulate_one_valley_pair(
            evals_i=evals_Kp_meV + sigma[3],
            evecs_i=evecs_Kp,
            evals_f=evals_Kp_up_q,
            evecs_f=evecs_Kp_f,
            weights=weights_arr,
            A_M_A2=float(A_M_A2),
            mu_i_meV=float(mu_bar[3]),
            mu_f_meV=float(mu_bar[1]),
            q=q,
            grid_shape=grid_shape,
            layer_masks=layer_masks,
            params=params,
            folded_mode=folded_mode,
            folded_with_G_shift=folded_with_G_shift,
            G_indices=plane_wave_G_indices,
        )
        extra["flavor_mu_regulated_pairs_plus"] = int(diag_K_plus["regulated_nonequilibrium_pairs"] + diag_Kp_plus["regulated_nonequilibrium_pairs"])
        extra["flavor_mu_regulated_abs_weight_plus"] = float(
            diag_K_plus["regulated_nonequilibrium_abs_weight"] + diag_Kp_plus["regulated_nonequilibrium_abs_weight"]
        )

    if _spin_flip_enabled(params, "minus"):
        chi_K_minus, layer_K_minus, diag_K_minus = _accumulate_one_valley_pair(
            evals_i=evals_K_meV + sigma[0],
            evecs_i=evecs_K,
            evals_f=evals_K_down_q,
            evecs_f=evecs_K_f,
            weights=weights_arr,
            A_M_A2=float(A_M_A2),
            mu_i_meV=float(mu_bar[0]),
            mu_f_meV=float(mu_bar[2]),
            q=q,
            grid_shape=grid_shape,
            layer_masks=layer_masks,
            params=params,
            folded_mode=folded_mode,
            folded_with_G_shift=folded_with_G_shift,
            G_indices=plane_wave_G_indices,
        )
        chi_Kp_minus, layer_Kp_minus, diag_Kp_minus = _accumulate_one_valley_pair(
            evals_i=evals_Kp_meV + sigma[1],
            evecs_i=evecs_Kp,
            evals_f=evals_Kp_down_q,
            evecs_f=evecs_Kp_f,
            weights=weights_arr,
            A_M_A2=float(A_M_A2),
            mu_i_meV=float(mu_bar[1]),
            mu_f_meV=float(mu_bar[3]),
            q=q,
            grid_shape=grid_shape,
            layer_masks=layer_masks,
            params=params,
            folded_mode=folded_mode,
            folded_with_G_shift=folded_with_G_shift,
            G_indices=plane_wave_G_indices,
        )
        extra["flavor_mu_regulated_pairs_minus"] = int(diag_K_minus["regulated_nonequilibrium_pairs"] + diag_Kp_minus["regulated_nonequilibrium_pairs"])
        extra["flavor_mu_regulated_abs_weight_minus"] = float(
            diag_K_minus["regulated_nonequilibrium_abs_weight"] + diag_Kp_minus["regulated_nonequilibrium_abs_weight"]
        )

    if params.include_spinflip_nesting:
        nesting_plus = nesting_minus = None
        if _spin_flip_enabled(params, "plus"):
            nesting_plus = compute_spinflip_nesting_q(
                q,
                grid_shape,
                weights_arr,
                float(A_M_A2),
                evals_K_meV + sigma[2],
                evecs_K,
                evals_K_up_q,
                evecs_K_f,
                float(mu_bar[2]),
                float(mu_bar[0]),
                params,
                folded_mode,
                folded_with_G_shift,
                plane_wave_G_indices,
            ) + compute_spinflip_nesting_q(
                q,
                grid_shape,
                weights_arr,
                float(A_M_A2),
                evals_Kp_meV + sigma[3],
                evecs_Kp,
                evals_Kp_up_q,
                evecs_Kp_f,
                float(mu_bar[3]),
                float(mu_bar[1]),
                params,
                folded_mode,
                folded_with_G_shift,
                plane_wave_G_indices,
            )
            extra["spinflip_nesting_plus"] = float(nesting_plus)
        if _spin_flip_enabled(params, "minus"):
            nesting_minus = compute_spinflip_nesting_q(
                q,
                grid_shape,
                weights_arr,
                float(A_M_A2),
                evals_K_meV + sigma[0],
                evecs_K,
                evals_K_down_q,
                evecs_K_f,
                float(mu_bar[0]),
                float(mu_bar[2]),
                params,
                folded_mode,
                folded_with_G_shift,
                plane_wave_G_indices,
            ) + compute_spinflip_nesting_q(
                q,
                grid_shape,
                weights_arr,
                float(A_M_A2),
                evals_Kp_meV + sigma[1],
                evecs_Kp,
                evals_Kp_down_q,
                evecs_Kp_f,
                float(mu_bar[1]),
                float(mu_bar[3]),
                params,
                folded_mode,
                folded_with_G_shift,
                plane_wave_G_indices,
            )
            extra["spinflip_nesting_minus"] = float(nesting_minus)

    models = make_valley_vertex_models(
        ref.u_cell_meV,
        ref.J_cell_meV,
        include_legacy=bool(params.legacy_diagnostics),
        hund_transverse_factor=float(params.hund_transverse_factor),
    )
    if not params.also_run_hund_factor2:
        models.pop("su2_hund_factor2", None)

    model_payload: dict[str, dict] = {}
    for model_name, spec in models.items():
        plus_lam = minus_lam = None
        plus_vec = minus_vec = None
        if chi_K_plus is not None and chi_Kp_plus is not None:
            if model_name == "legacy_scalar_total":
                plus_lam = legacy_scalar_total_lambda(chi_K_plus, chi_Kp_plus, ref.u_cell_meV)
                plus_vec = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
            else:
                plus_lam, plus_vec = generalized_stoner_lambda(np.diag([chi_K_plus, chi_Kp_plus]), spec.gamma)
        if chi_K_minus is not None and chi_Kp_minus is not None:
            if model_name == "legacy_scalar_total":
                minus_lam = legacy_scalar_total_lambda(chi_K_minus, chi_Kp_minus, ref.u_cell_meV)
                minus_vec = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
            else:
                minus_lam, minus_vec = generalized_stoner_lambda(np.diag([chi_K_minus, chi_Kp_minus]), spec.gamma)
        selected, direction = _model_selected(plus_lam, minus_lam)
        soft_channel = str(ref.soft_transverse_channel)
        if soft_channel == "plus":
            soft_lam = plus_lam
        elif soft_channel == "minus":
            soft_lam = minus_lam
        else:
            soft_lam = selected
        vec = plus_vec if direction == "plus" else minus_vec
        layer_pair = (layer_K_plus, layer_Kp_plus) if direction == "plus" else (layer_K_minus, layer_Kp_minus)
        chi_s0_model, chi_sD_model, od_model, ratio_model, _ = _selected_layer_diagnostics(
            layer_pair[0], layer_pair[1], vec if vec is not None else np.ones(2), params
        )
        model_payload[model_name] = {
            "plus": plus_lam,
            "minus": minus_lam,
            "selected": selected,
            "soft": soft_lam,
            "direction": direction,
            "vec": vec if vec is not None else np.ones(2, dtype=np.complex128),
            "chi_s0": chi_s0_model,
            "chi_sD": chi_sD_model,
            "layer_dipole_overlap": od_model,
            "layer_dipole_ratio": ratio_model,
        }
        extra[f"{model_name}_spin_flip_direction"] = direction
        extra[f"lambda_{model_name}_soft"] = float(soft_lam) if soft_lam is not None else np.nan
        if chi_s0_model is not None:
            extra[f"{model_name}_chi_s0"] = float(chi_s0_model)
            extra[f"{model_name}_chi_sD"] = float(chi_sD_model)
            extra[f"{model_name}_layer_dipole_overlap"] = float(od_model)
            extra[f"{model_name}_layer_dipole_ratio"] = float(ratio_model)

    if params.legacy_diagnostics:
        for direction, k_chi, kp_chi in (
            ("plus", chi_K_plus, chi_Kp_plus),
            ("minus", chi_K_minus, chi_Kp_minus),
        ):
            if k_chi is not None and kp_chi is not None:
                extra[f"lambda_legacy_scalar_total_{direction}"] = legacy_scalar_total_lambda(k_chi, kp_chi, ref.u_cell_meV)
        plus_legacy = extra.get("lambda_legacy_scalar_total_plus")
        minus_legacy = extra.get("lambda_legacy_scalar_total_minus")
        extra["lambda_legacy_scalar_total_selected"], extra["legacy_scalar_total_spin_flip_direction"] = _model_selected(
            plus_legacy, minus_legacy
        )

    selected_name = str(params.main_vertex_model)
    if selected_name not in model_payload:
        selected_name = "su4_diag"
    selected_payload = model_payload[selected_name]
    extra["soft_transverse_channel"] = str(ref.soft_transverse_channel)
    extra["channel_switch_warning"] = bool(
        ref.soft_transverse_channel in {"plus", "minus"} and selected_payload["direction"] != ref.soft_transverse_channel
    )
    if params.include_spinflip_nesting:
        if ref.soft_transverse_channel == "plus":
            extra["spinflip_nesting_soft"] = float(extra.get("spinflip_nesting_plus", np.nan))
        elif ref.soft_transverse_channel == "minus":
            extra["spinflip_nesting_soft"] = float(extra.get("spinflip_nesting_minus", np.nan))
        else:
            extra["spinflip_nesting_soft"] = float(max(extra.get("spinflip_nesting_plus", np.nan), extra.get("spinflip_nesting_minus", np.nan)))
    selected_vec = np.asarray(selected_payload["vec"], dtype=np.complex128)
    chi_layer = None
    chi_s0 = chi_sD = leading = overlap = None
    if selected_payload["direction"] == "plus":
        _, _, _, _, chi_layer = _selected_layer_diagnostics(layer_K_plus, layer_Kp_plus, selected_vec, params)
    elif selected_payload["direction"] == "minus":
        _, _, _, _, chi_layer = _selected_layer_diagnostics(layer_K_minus, layer_Kp_minus, selected_vec, params)
    if chi_layer is not None:
        chi_s0, chi_sD, leading, overlap = _layer_diagnostics(chi_layer, params)

    chi_K_legacy = float(chi_K_plus if chi_K_plus is not None else (chi_K_minus if chi_K_minus is not None else 0.0))
    chi_Kp_legacy = float(chi_Kp_plus if chi_Kp_plus is not None else (chi_Kp_minus if chi_Kp_minus is not None else 0.0))
    chi_total = float(chi_K_legacy + chi_Kp_legacy)
    legacy_offdiag_plus = model_payload.get("legacy_offdiag_J", {}).get("plus")
    legacy_offdiag_minus = model_payload.get("legacy_offdiag_J", {}).get("minus")

    return ChiQResult(
        q=q,
        chi_total_cell_meV_inv=chi_total,
        chi_K_cell_meV_inv=chi_K_legacy,
        chi_Kp_cell_meV_inv=chi_Kp_legacy,
        lambda_u=float(ref.u_cell_meV * chi_total),
        lambda_u_plus_hund=float(legacy_offdiag_plus if legacy_offdiag_plus is not None else _lambda_u_plus_hund(chi_K_legacy, chi_Kp_legacy, ref)),
        chi_layer_cell_meV_inv=chi_layer,
        chi_s0_cell_meV_inv=chi_s0,
        chi_sD_cell_meV_inv=chi_sD,
        layer_leading_eig_cell_meV_inv=leading,
        layer_dipole_overlap=overlap,
        chi_K_plus_cell_meV_inv=chi_K_plus,
        chi_Kp_plus_cell_meV_inv=chi_Kp_plus,
        chi_K_minus_cell_meV_inv=chi_K_minus,
        chi_Kp_minus_cell_meV_inv=chi_Kp_minus,
        lambda_su4_diag_plus=model_payload.get("su4_diag", {}).get("plus"),
        lambda_su4_diag_minus=model_payload.get("su4_diag", {}).get("minus"),
        lambda_su4_diag_selected=model_payload.get("su4_diag", {}).get("selected"),
        lambda_su2_hund_factor2_plus=model_payload.get("su2_hund_factor2", {}).get("plus"),
        lambda_su2_hund_factor2_minus=model_payload.get("su2_hund_factor2", {}).get("minus"),
        lambda_su2_hund_factor2_selected=model_payload.get("su2_hund_factor2", {}).get("selected"),
        lambda_legacy_scalar_total_plus=extra.get("lambda_legacy_scalar_total_plus"),
        lambda_legacy_scalar_total_minus=extra.get("lambda_legacy_scalar_total_minus"),
        lambda_legacy_scalar_total_selected=extra.get("lambda_legacy_scalar_total_selected"),
        lambda_legacy_offdiag_J_plus=legacy_offdiag_plus,
        lambda_legacy_offdiag_J_minus=legacy_offdiag_minus,
        lambda_legacy_offdiag_J_selected=model_payload.get("legacy_offdiag_J", {}).get("selected"),
        selected_vertex_model=selected_name,
        selected_spin_flip_direction=selected_payload["direction"],
        selected_lambda_raw=selected_payload["selected"],
        selected_valley_eigenvector_K_real=float(np.real(selected_vec[0])),
        selected_valley_eigenvector_K_imag=float(np.imag(selected_vec[0])),
        selected_valley_eigenvector_Kp_real=float(np.real(selected_vec[1])),
        selected_valley_eigenvector_Kp_imag=float(np.imag(selected_vec[1])),
        selected_chi_s0=selected_payload["chi_s0"],
        selected_chi_sD=selected_payload["chi_sD"],
        selected_layer_dipole_overlap=selected_payload["layer_dipole_overlap"],
        selected_layer_dipole_ratio=selected_payload["layer_dipole_ratio"],
        su4_diag_spin_flip_direction=model_payload.get("su4_diag", {}).get("direction"),
        su4_diag_layer_dipole_overlap=model_payload.get("su4_diag", {}).get("layer_dipole_overlap"),
        su4_diag_layer_dipole_ratio=model_payload.get("su4_diag", {}).get("layer_dipole_ratio"),
        su2_hund_factor2_spin_flip_direction=model_payload.get("su2_hund_factor2", {}).get("direction"),
        su2_hund_factor2_layer_dipole_overlap=model_payload.get("su2_hund_factor2", {}).get("layer_dipole_overlap"),
        su2_hund_factor2_layer_dipole_ratio=model_payload.get("su2_hund_factor2", {}).get("layer_dipole_ratio"),
        extra=extra,
    )
