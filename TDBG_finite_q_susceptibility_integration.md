# TDBG self-consistent + Stoner 后的 finite-Q transverse spin susceptibility 集成方案

这份文档是给 Codex 用的集成说明。目标是在不改变现有 self-consistent continuum model、projected/full SCF、以及现有 `tdbg_scf.stoner` 功能的前提下，新增一套 **Stoner 后参考态的 finite-Q transverse spin susceptibility** 计算。

核心用途：在你现有的 `q=0` Stoner spin-polarized dome 上，检查 halo / boundary 附近是否存在 `Q != 0` 的 transverse spin soft mode，也就是 SDW / spiral / skyrmion-like texture 的线性前驱。

---

## 1. 物理对象和代码对象的对应

### 1.1 已有代码里的参考态

你已有的流程大致是：

```text
continuum Hamiltonian H0(k; U_l^SC)
        ↓
self-consistent Hartree layer potentials U_l^SC(n, D)
        ↓
full-band Stoner solver: minimize E[nu_f]
        ↓
q=0 spin/flavor-polarized solution nu_f = (nu_K_up, nu_Kp_up, nu_K_down, nu_Kp_down)
```

这里 Stoner 是 `Q=0` 的意思是：它只允许 flavor occupation 不同，

```text
< c†_{k f} c_{k f} > differs between flavors,
```

但没有允许

```text
< c†_{k+Q, up} c_{k, down} > != 0.
```

所以新增计算就是在 Stoner 后的 reference state 上，测试这个被漏掉的 finite-Q transverse channel。

---

### 1.2 Stoner 后 single-particle energy shift

现有 `tdbg_scf/stoner/energy.py` 里的 interaction energy 是：

```python
u = np.asarray(nu_f)
u_cell = params.u0_meV_A2 / A_M_A2
J_cell = params.JH_meV_A2 / A_M_A2
total = np.sum(nu)
E_u = 0.5 * u_cell * (total * total - np.dot(nu, nu))
m_K = nu[0] - nu[2]
m_Kp = nu[1] - nu[3]
E_hund = -J_cell * m_K * m_Kp
```

因此 Stoner 后 flavor energy shift 可取 interaction derivative：

```text
Sigma_f = dE_int / dnu_f.
```

显式地：

```text
Sigma_0 = u_cell * (nu_total - nu_0) - J_cell * (nu_1 - nu_3)
Sigma_1 = u_cell * (nu_total - nu_1) - J_cell * (nu_0 - nu_2)
Sigma_2 = u_cell * (nu_total - nu_2) + J_cell * (nu_1 - nu_3)
Sigma_3 = u_cell * (nu_total - nu_3) + J_cell * (nu_0 - nu_2)
```

其中 flavor order 沿用现有代码：

```text
0: K_up
1: Kp_up
2: K_down
3: Kp_down
```

公共常数 shift 不影响 susceptibility denominator，所以代码默认减去 `mean(Sigma_f)`，只保留 flavor splitting。

Stoner 后的 band energy 是：

```text
E_{K,up}(k,n)   = eps_K(k,n)  + Sigma_0
E_{Kp,up}(k,n)  = eps_Kp(k,n) + Sigma_1
E_{K,down}(k,n) = eps_K(k,n)  + Sigma_2
E_{Kp,down}(k,n)= eps_Kp(k,n)+ Sigma_3
```

如果当前 Stoner term 只是 flavor scalar shift，波函数仍然用 SCF 后的 continuum eigenvectors；变化只在 energy 和 occupation。

---

### 1.3 finite-Q transverse susceptibility

对每个 valley，计算 spin-down 到 spin-up 的 transverse bubble：

```text
chi^{+-}_tau(Q)
= sum_{k,m,n} W_k A_M |<u_{tau,m,k+Q}|u_{tau,n,k}>|^2
  * [ f(E_{tau,down,n,k} - mu) - f(E_{tau,up,m,k+Q} - mu) ]
  / [ E_{tau,up,m,k+Q} - E_{tau,down,n,k} ].
```

总 susceptibility：

```text
chi_total(Q) = chi_K(Q) + chi_Kp(Q).
```

这里：

```text
k     = single-particle moire Bloch momentum
Q     = spin-density/order-parameter momentum
W_k   = existing uniform MBZ integration weight, unit A^{-2}
A_M   = moire unit-cell area, so A_M * W_k gives filling weight
chi   = per moire cell per meV
```

如果 `Q=0`，它回到 Stoner transverse channel 的 local curvature；如果 `Q != 0` 的 chi 更大，则说明 uniform Stoner reference 对 finite-Q mode 更软。

---

### 1.4 layer-resolved optional diagnostic

为了后面和 layer polarization / layer dipole 连接，顺手保留 layer matrix：

```text
Lambda_l(k,Q) = <u_{m,k+Q}| P_l |u_{n,k}>

chi_ll'(Q) = sum_{tau,k,m,n} W_k A_M
             Lambda_l(k,Q) Lambda^*_{l'}(k,Q)
             * Lindhard_ratio.
```

总 spin channel 满足：

```text
chi_total(Q) = sum_{l,l'} chi_ll'(Q),
```

因为 `sum_l P_l = Identity`。

layer dipole vector 沿用现有 `layer_polarizations` 里的 convention：

```text
zeta = (1.5, 0.5, -0.5, -1.5)
```

可输出：

```text
chi_s0 = 1^T chi_layer 1
chi_sD = zeta^T chi_layer zeta
O_D    = leading_eigenvector 与 zeta 的 normalized overlap
```

---

## 2. 推荐新增文件结构

只新增文件，不改旧功能：

```text
tdbg_scf/
  susceptibility/
    __init__.py
    params.py
    qmesh.py
    stoner_reference.py
    bubble.py
    workflow.py
examples/
  run_stoner_susceptibility_single.py
  run_nd_mapping_stoner_susceptibility.py
```

可选：如果你希望顶层 `from tdbg_scf import ...` 能直接导入这些函数，再轻微追加 `tdbg_scf/__init__.py`；但不是必须。为了最小侵入，本方案不要求改顶层 `__init__.py`。

---

## 3. 新增文件：`tdbg_scf/susceptibility/__init__.py`

```python
"""Finite-Q transverse spin susceptibility on top of Stoner reference states."""

from .params import SusceptibilityParams
from .qmesh import QPoint, make_commensurate_q_points, folded_indices_for_q
from .stoner_reference import StonerReference, make_stoner_reference, stoner_self_energy_shifts_meV
from .bubble import ChiQResult, compute_transverse_chi_q
from .workflow import (
    SinglePointStonerState,
    solve_full_scf_stoner_state,
    scan_q_for_state,
    summarize_q_scan,
    q_results_to_dataframe,
)

__all__ = [
    "SusceptibilityParams",
    "QPoint",
    "make_commensurate_q_points",
    "folded_indices_for_q",
    "StonerReference",
    "make_stoner_reference",
    "stoner_self_energy_shifts_meV",
    "ChiQResult",
    "compute_transverse_chi_q",
    "SinglePointStonerState",
    "solve_full_scf_stoner_state",
    "scan_q_for_state",
    "summarize_q_scan",
    "q_results_to_dataframe",
]
```

---

## 4. 新增文件：`tdbg_scf/susceptibility/params.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SusceptibilityParams:
    """Numerical controls for finite-Q transverse susceptibility.

    The susceptibility is computed per moire unit cell per meV.
    The default layer-dipole vector follows the existing repository convention
    used in stoner.full_band.layer_polarizations:
        P_dipole = 1.5*n1 + 0.5*n2 - 0.5*n3 - 1.5*n4.
    """

    kBT_meV: float = 0.05
    denom_tol_meV: float = 1e-8

    # Restrict the Lindhard sum to bands near the common Stoner chemical potential.
    # None means all bands. 20-40 meV is a good first pass for the flat-band problem.
    energy_window_meV: float | None = 30.0
    max_bands_per_k: int | None = 24

    # Fast mode uses the same periodic MBZ mesh and folds k+Q back to the mesh.
    # Safe mode diagonalizes at the unfolded k+Q point; slower but better for checks.
    q_mode: str = "folded_grid"  # folded_grid or unfolded_diagonalize

    include_layer_matrix: bool = True
    hermitize_layer_matrix: bool = True
    layer_dipole_zeta: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5)

    # Finite-Q decision threshold: finite-Q wins if lambda(Q*) > lambda(Gamma)*(1 + tol).
    finite_q_tol: float = 1e-3
```

---

## 5. 新增文件：`tdbg_scf/susceptibility/qmesh.py`

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QPoint:
    """A Q point commensurate with the uniform MBZ grid.

    dq1, dq2 are integer shifts on the k-grid:
        Q = dq1/n1 * b1m + dq2/n2 * b2m.
    """

    dq1: int
    dq2: int
    qvec_Ainv: np.ndarray

    @property
    def qx_Ainv(self) -> float:
        return float(self.qvec_Ainv[0])

    @property
    def qy_Ainv(self) -> float:
        return float(self.qvec_Ainv[1])

    @property
    def q_norm_Ainv(self) -> float:
        return float(np.linalg.norm(self.qvec_Ainv))

    @property
    def is_gamma(self) -> bool:
        return int(self.dq1) == 0 and int(self.dq2) == 0

    def to_record(self) -> dict:
        return {
            "dq1": int(self.dq1),
            "dq2": int(self.dq2),
            "qx_Ainv": self.qx_Ainv,
            "qy_Ainv": self.qy_Ainv,
            "q_norm_Ainv": self.q_norm_Ainv,
            "is_gamma": bool(self.is_gamma),
        }


def _centered_integer_steps(n: int, stride: int = 1, max_abs_step: int | None = None) -> list[int]:
    if n <= 0:
        raise ValueError("grid size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    # For n=15 gives -7,...,+7; for n=16 gives -8,...,+7.
    start = -(n // 2)
    stop = n - n // 2
    steps = [x for x in range(start, stop) if x % stride == 0]

    if max_abs_step is not None:
        steps = [x for x in steps if abs(x) <= int(max_abs_step)]

    if 0 not in steps:
        steps.append(0)
    return sorted(set(int(x) for x in steps))


def make_commensurate_q_points(
    geom,
    grid_shape: tuple[int, int],
    q_stride: int = 1,
    max_abs_step: int | None = None,
    include_gamma: bool = True,
) -> list[QPoint]:
    """Build Q points commensurate with make_uniform_mbz_grid.

    Parameters
    ----------
    geom:
        TDBG MoireGeometry object. Needs b1m and b2m.
    grid_shape:
        Same (n1, n2) used for make_uniform_mbz_grid.
    q_stride:
        Use every q_stride-th commensurate Q step. For quick scans use 2 or 3.
    max_abs_step:
        Optional cutoff in integer grid steps. Useful for small-Q scans around Gamma.
    include_gamma:
        Whether to keep Q=(0,0).
    """
    n1, n2 = [int(x) for x in grid_shape]
    steps1 = _centered_integer_steps(n1, stride=q_stride, max_abs_step=max_abs_step)
    steps2 = _centered_integer_steps(n2, stride=q_stride, max_abs_step=max_abs_step)

    qpts: list[QPoint] = []
    b1 = np.asarray(geom.b1m, dtype=float)
    b2 = np.asarray(geom.b2m, dtype=float)
    for dq1 in steps1:
        for dq2 in steps2:
            if not include_gamma and dq1 == 0 and dq2 == 0:
                continue
            qvec = (float(dq1) / n1) * b1 + (float(dq2) / n2) * b2
            qpts.append(QPoint(dq1=int(dq1), dq2=int(dq2), qvec_Ainv=qvec))

    qpts.sort(key=lambda q: (q.q_norm_Ainv, q.dq1, q.dq2))
    return qpts


def folded_index_for_q(ik: int, q: QPoint, grid_shape: tuple[int, int]) -> int:
    """Return the grid index of k_i + Q folded back to the uniform MBZ mesh."""
    n1, n2 = [int(x) for x in grid_shape]
    i, j = divmod(int(ik), n2)
    ip = (i + int(q.dq1)) % n1
    jp = (j + int(q.dq2)) % n2
    return ip * n2 + jp


def folded_indices_for_q(q: QPoint, grid_shape: tuple[int, int]) -> np.ndarray:
    n1, n2 = [int(x) for x in grid_shape]
    Nk = n1 * n2
    return np.asarray([folded_index_for_q(ik, q, grid_shape) for ik in range(Nk)], dtype=int)
```

---

## 6. 新增文件：`tdbg_scf/susceptibility/stoner_reference.py`

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tdbg_scf.stoner.full_band import stoner_mu_by_flavor
from tdbg_scf.stoner.params import StonerParams


FLAVOR_NAMES = ("K_up", "Kp_up", "K_down", "Kp_down")


@dataclass
class StonerReference:
    """Stoner-after reference data used by finite-Q susceptibility."""

    nu_f: np.ndarray
    mu_f_meV: np.ndarray
    sigma_f_meV: np.ndarray
    mu_common_meV: float
    mu_common_spread_meV: float
    u_cell_meV: float
    J_cell_meV: float
    flavor_names: tuple[str, str, str, str] = FLAVOR_NAMES

    def to_record(self) -> dict:
        out = {
            "mu_common_meV": float(self.mu_common_meV),
            "mu_common_spread_meV": float(self.mu_common_spread_meV),
            "u_cell_meV": float(self.u_cell_meV),
            "J_cell_meV": float(self.J_cell_meV),
        }
        for i, name in enumerate(self.flavor_names):
            out[f"nu_{name}"] = float(self.nu_f[i])
            out[f"mu_{name}_meV"] = float(self.mu_f_meV[i])
            out[f"sigma_{name}_meV"] = float(self.sigma_f_meV[i])
            out[f"mu_plus_sigma_{name}_meV"] = float(self.mu_f_meV[i] + self.sigma_f_meV[i])
        return out


def stoner_self_energy_shifts_meV(
    nu_f: np.ndarray,
    params: StonerParams,
    A_M_A2: float,
    remove_common: bool = True,
) -> np.ndarray:
    """Return flavor-dependent Stoner shifts Sigma_f = dE_int/dnu_f.

    The common part is irrelevant for particle-hole denominators, so by default
    it is removed. The flavor splittings are unchanged.
    """
    nu = np.asarray(nu_f, dtype=float)
    if nu.shape != (4,):
        raise ValueError("nu_f must have shape (4,)")

    u_cell = float(params.u0_meV_A2) / float(A_M_A2)
    J_cell = float(params.JH_meV_A2) / float(A_M_A2)
    total = float(np.sum(nu))

    # d/dnu_i of 0.5*u*(total^2 - sum_i nu_i^2)
    sigma = u_cell * (total - nu)

    # d/dnu_i of -J*(nu0-nu2)*(nu1-nu3)
    m_K = float(nu[0] - nu[2])
    m_Kp = float(nu[1] - nu[3])
    sigma[0] += -J_cell * m_Kp
    sigma[2] += +J_cell * m_Kp
    sigma[1] += -J_cell * m_K
    sigma[3] += +J_cell * m_K

    if remove_common:
        sigma = sigma - float(np.mean(sigma))
    return np.asarray(sigma, dtype=float)


def make_stoner_reference(
    result,
    tables,
    params: StonerParams,
    A_M_A2: float,
    remove_common_shift: bool = True,
) -> StonerReference:
    """Convert an existing StonerResult into a reference state for chi(Q)."""
    nu_f = np.asarray(result.nu_f, dtype=float)
    mu_f = stoner_mu_by_flavor(result, tables)
    sigma = stoner_self_energy_shifts_meV(nu_f, params, A_M_A2, remove_common=remove_common_shift)

    # In an exactly self-consistent Stoner minimum, mu_f + Sigma_f should be common.
    mu_eff = mu_f + sigma
    mu_common = float(np.mean(mu_eff))
    spread = float(np.max(mu_eff) - np.min(mu_eff))

    return StonerReference(
        nu_f=nu_f,
        mu_f_meV=np.asarray(mu_f, dtype=float),
        sigma_f_meV=np.asarray(sigma, dtype=float),
        mu_common_meV=mu_common,
        mu_common_spread_meV=spread,
        u_cell_meV=float(params.u0_meV_A2) / float(A_M_A2),
        J_cell_meV=float(params.JH_meV_A2) / float(A_M_A2),
    )
```

---

## 7. 新增文件：`tdbg_scf/susceptibility/bubble.py`

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import SusceptibilityParams
from .qmesh import QPoint, folded_indices_for_q
from .stoner_reference import StonerReference


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
                    out[f"chi_layer_L{l+1}_L{lp+1}_cell_meV_inv"] = float(np.real(mat[l, lp]))
        return out


def fermi_occ(E_minus_mu_meV: np.ndarray, kBT_meV: float) -> np.ndarray:
    T = max(float(kBT_meV), 1e-12)
    x = np.clip(np.asarray(E_minus_mu_meV, dtype=float) / T, -100.0, 100.0)
    return 1.0 / (np.exp(x) + 1.0)


def minus_fermi_derivative(E_minus_mu_meV: np.ndarray, kBT_meV: float) -> np.ndarray:
    T = max(float(kBT_meV), 1e-12)
    x = np.clip(np.asarray(E_minus_mu_meV, dtype=float) / (2.0 * T), -50.0, 50.0)
    return 1.0 / (4.0 * T * np.cosh(x) ** 2)


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
    return np.where(np.abs(den) > float(denom_tol_meV), num / den, limiting)


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
            # Always keep at least one band near mu so the code does not silently drop a k point.
            keep = 1 if max_bands is None else max(1, min(int(max_bands), e.size))
            idx = order[:keep]

    if max_bands is not None and idx.size > int(max_bands):
        local_order = np.argsort(distance[idx])
        idx = idx[local_order[: int(max_bands)]]

    return np.sort(idx.astype(int))


def _pair_form_factors_total(Vf: np.ndarray, Vi: np.ndarray) -> np.ndarray:
    # columns are eigenvectors; result shape (n_final, n_initial)
    return Vf.conj().T @ Vi


def _pair_form_factors_layers(Vf: np.ndarray, Vi: np.ndarray, layer_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(layer_masks, dtype=float)
    L = masks.shape[0]
    out = np.empty((L, Vf.shape[1], Vi.shape[1]), dtype=np.complex128)
    for l in range(L):
        out[l] = Vf.conj().T @ (masks[l, :, None] * Vi)
    return out


def _accumulate_one_valley_pair(
    evals_i: np.ndarray,
    evecs_i: np.ndarray,
    evals_f: np.ndarray,
    evecs_f: np.ndarray,
    weights: np.ndarray,
    A_M_A2: float,
    mu_meV: float,
    q: QPoint,
    grid_shape: tuple[int, int],
    layer_masks: np.ndarray | None,
    params: SusceptibilityParams,
    folded_mode: bool,
) -> tuple[float, np.ndarray | None]:
    """Bubble for one valley spin pair: down(k) -> up(k+Q)."""
    Nk = evals_i.shape[0]
    if evals_f.shape[0] != Nk:
        raise ValueError("initial and final eval arrays must have the same Nk")
    if weights.shape != (Nk,):
        raise ValueError("weights must have shape (Nk,)")

    if folded_mode:
        kq_indices = folded_indices_for_q(q, grid_shape)
    else:
        kq_indices = np.arange(Nk, dtype=int)

    chi = 0.0
    chi_layer = None
    if layer_masks is not None and params.include_layer_matrix:
        L = int(layer_masks.shape[0])
        chi_layer = np.zeros((L, L), dtype=np.complex128)

    for ik in range(Nk):
        ikq = int(kq_indices[ik])
        Ei_all = np.asarray(evals_i[ik], dtype=float)
        Ef_all = np.asarray(evals_f[ikq], dtype=float)

        idx_i = band_indices_near_mu(Ei_all, mu_meV, params.energy_window_meV, params.max_bands_per_k)
        idx_f = band_indices_near_mu(Ef_all, mu_meV, params.energy_window_meV, params.max_bands_per_k)

        Ei = Ei_all[idx_i]
        Ef = Ef_all[idx_f]
        R = lindhard_static_ratio(Ei, Ef, mu_meV, params.kBT_meV, params.denom_tol_meV)

        Vi = evecs_i[ik][:, idx_i]
        Vf = evecs_f[ikq][:, idx_f]

        Lam = _pair_form_factors_total(Vf, Vi)
        prefactor = float(A_M_A2) * float(weights[ik])
        chi += prefactor * float(np.real(np.sum(R * np.abs(Lam) ** 2)))

        if chi_layer is not None:
            Lam_l = _pair_form_factors_layers(Vf, Vi, layer_masks)
            chi_layer += prefactor * np.einsum("mn,lmn,pmn->lp", R, Lam_l, Lam_l.conj(), optimize=True)

    if chi_layer is not None and params.hermitize_layer_matrix:
        chi_layer = 0.5 * (chi_layer + chi_layer.conj().T)
    return float(np.real(chi)), chi_layer


def _lambda_u_plus_hund(chi_K: float, chi_Kp: float, ref: StonerReference) -> float:
    """Leading eigenvalue of sqrt(chi) Gamma sqrt(chi) in valley spin-pair space."""
    vals = np.asarray([max(float(chi_K), 0.0), max(float(chi_Kp), 0.0)], dtype=float)
    sqrt_chi = np.sqrt(vals)
    Gamma = np.asarray(
        [[float(ref.u_cell_meV), float(ref.J_cell_meV)], [float(ref.J_cell_meV), float(ref.u_cell_meV)]],
        dtype=float,
    )
    M = sqrt_chi[:, None] * Gamma * sqrt_chi[None, :]
    return float(np.linalg.eigvalsh(M)[-1])


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
    overlap = 0.0 if denom <= 0 else float(abs(np.vdot(v, z)) ** 2 / denom)
    return chi_s0, chi_sD, leading, overlap


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
    # Only used in unfolded_diagonalize mode. These should already include the k+Q eigensystem.
    evals_K_q_meV: np.ndarray | None = None,
    evecs_K_q: np.ndarray | None = None,
    evals_Kp_q_meV: np.ndarray | None = None,
    evecs_Kp_q: np.ndarray | None = None,
) -> ChiQResult:
    """Compute chi^{+-}(Q) on the Stoner-after reference.

    Flavor convention:
        K_up=0, Kp_up=1, K_down=2, Kp_down=3.
    Transverse S^+ bubble uses down -> up within the same valley:
        K:  flavor 2 -> 0
        Kp: flavor 3 -> 1
    """
    sigma = np.asarray(ref.sigma_f_meV, dtype=float)
    mu = float(ref.mu_common_meV)
    folded_mode = params.q_mode == "folded_grid"
    if params.q_mode not in {"folded_grid", "unfolded_diagonalize"}:
        raise ValueError("params.q_mode must be 'folded_grid' or 'unfolded_diagonalize'")

    if folded_mode:
        evals_K_f = evals_K_meV + sigma[0]
        evecs_K_f = evecs_K
        evals_Kp_f = evals_Kp_meV + sigma[1]
        evecs_Kp_f = evecs_Kp
    else:
        if evals_K_q_meV is None or evecs_K_q is None or evals_Kp_q_meV is None or evecs_Kp_q is None:
            raise ValueError("unfolded_diagonalize mode requires k+Q eigensystems")
        evals_K_f = evals_K_q_meV + sigma[0]
        evecs_K_f = evecs_K_q
        evals_Kp_f = evals_Kp_q_meV + sigma[1]
        evecs_Kp_f = evecs_Kp_q

    # Initial down-spin energies always live at k.
    evals_K_i = evals_K_meV + sigma[2]
    evals_Kp_i = evals_Kp_meV + sigma[3]

    chi_K, layer_K = _accumulate_one_valley_pair(
        evals_i=evals_K_i,
        evecs_i=evecs_K,
        evals_f=evals_K_f,
        evecs_f=evecs_K_f,
        weights=weights,
        A_M_A2=A_M_A2,
        mu_meV=mu,
        q=q,
        grid_shape=grid_shape,
        layer_masks=layer_masks,
        params=params,
        folded_mode=folded_mode,
    )
    chi_Kp, layer_Kp = _accumulate_one_valley_pair(
        evals_i=evals_Kp_i,
        evecs_i=evecs_Kp,
        evals_f=evals_Kp_f,
        evecs_f=evecs_Kp_f,
        weights=weights,
        A_M_A2=A_M_A2,
        mu_meV=mu,
        q=q,
        grid_shape=grid_shape,
        layer_masks=layer_masks,
        params=params,
        folded_mode=folded_mode,
    )

    chi_total = float(chi_K + chi_Kp)
    lambda_u = float(ref.u_cell_meV * chi_total)
    lambda_uh = _lambda_u_plus_hund(chi_K, chi_Kp, ref)

    chi_layer = None
    chi_s0 = chi_sD = leading = overlap = None
    if layer_K is not None and layer_Kp is not None:
        chi_layer = layer_K + layer_Kp
        if params.hermitize_layer_matrix:
            chi_layer = 0.5 * (chi_layer + chi_layer.conj().T)
        chi_s0, chi_sD, leading, overlap = _layer_diagnostics(chi_layer, params)

    return ChiQResult(
        q=q,
        chi_total_cell_meV_inv=chi_total,
        chi_K_cell_meV_inv=float(chi_K),
        chi_Kp_cell_meV_inv=float(chi_Kp),
        lambda_u=lambda_u,
        lambda_u_plus_hund=lambda_uh,
        chi_layer_cell_meV_inv=chi_layer,
        chi_s0_cell_meV_inv=chi_s0,
        chi_sD_cell_meV_inv=chi_sD,
        layer_leading_eig_cell_meV_inv=leading,
        layer_dipole_overlap=overlap,
    )
```

---

## 8. 新增文件：`tdbg_scf/susceptibility/workflow.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from tdbg_scf import TDBGContinuumHamiltonian, TDBGParameters, FullSCFConfig, FullSCFSolver, make_uniform_mbz_grid
from tdbg_scf.filling import moire_cell_area_A2_from_weights
from tdbg_scf.stoner.full_band import (
    build_full_band_flavor_tables,
    density_cm2_to_nu_total,
    restrict_tables_to_carrier_sector,
    stoner_result_to_dict,
)
from tdbg_scf.stoner.params import StonerParams
from tdbg_scf.stoner.solver import solve_stoner_fixed_nu

from .bubble import ChiQResult, compute_transverse_chi_q
from .params import SusceptibilityParams
from .qmesh import make_commensurate_q_points
from .stoner_reference import StonerReference, make_stoner_reference


@dataclass
class SinglePointStonerState:
    """Everything needed for chi(Q) at one (n, D) point."""

    n_cm2: float
    D_Vnm: float
    nu_total: float
    grid_shape: tuple[int, int]
    kpts: np.ndarray
    weights: np.ndarray
    A_M_A2: float
    U_meV: np.ndarray
    scf_result: object
    stoner_result: object
    stoner_params: StonerParams
    stoner_reference: StonerReference
    tables: list
    ham_K: TDBGContinuumHamiltonian
    ham_Kp: TDBGContinuumHamiltonian
    evals_K_meV: np.ndarray
    evecs_K: np.ndarray
    evals_Kp_meV: np.ndarray
    evecs_Kp: np.ndarray
    layer_masks: np.ndarray

    def metadata(self) -> dict:
        out = {
            "n_cm2": float(self.n_cm2),
            "D_Vnm": float(self.D_Vnm),
            "nu_total": float(self.nu_total),
            "A_M_A2": float(self.A_M_A2),
            "grid_n1": int(self.grid_shape[0]),
            "grid_n2": int(self.grid_shape[1]),
            "U1_meV": float(self.U_meV[0]),
            "U2_meV": float(self.U_meV[1]),
            "U3_meV": float(self.U_meV[2]),
            "U4_meV": float(self.U_meV[3]),
            "scf_mu_meV": float(getattr(self.scf_result, "mu_meV", np.nan)),
            "scf_converged": bool(getattr(self.scf_result, "converged", False)),
            "scf_residual_meV": float(getattr(self.scf_result, "residual_meV", np.nan)),
        }
        out.update(self.stoner_reference.to_record())
        out.update({f"stoner_{k}": v for k, v in stoner_result_to_dict(self.stoner_result, self.tables).items() if k != "local_minima"})
        return out


def _make_params(
    *,
    theta_deg: float,
    cutoff: int,
    valley: int,
    omega_meV: float,
    wAA: float,
    wAB: float,
    sublattice_Z_meV: float,
    a_cc_A: float,
    gamma0_meV: float,
    gamma1_meV: float,
    gamma3_meV: float,
    gamma4_meV: float,
) -> TDBGParameters:
    return TDBGParameters(
        theta_deg=float(theta_deg),
        cutoff=int(cutoff),
        valley=int(valley),
        omega_meV=float(omega_meV),
        wAA=float(wAA),
        wAB=float(wAB),
        sublattice_Z_meV=float(sublattice_Z_meV),
        a_cc_A=float(a_cc_A),
        gamma0_meV=float(gamma0_meV),
        gamma1_meV=float(gamma1_meV),
        gamma3_meV=float(gamma3_meV),
        gamma4_meV=float(gamma4_meV),
    )


def solve_full_scf_stoner_state(
    *,
    n_cm2: float,
    D_Vnm: float,
    theta_deg: float = 1.35,
    cutoff: int = 1,
    grid_n1: int = 9,
    grid_n2: int = 9,
    omega_meV: float = 100.0,
    wAA: float = 0.8,
    wAB: float = 1.0,
    sublattice_Z_meV: float = 15.0,
    a_cc_A: float = 1.420,
    gamma0_meV: float = 2610.0,
    gamma1_meV: float = 361.0,
    gamma3_meV: float = 283.0,
    gamma4_meV: float = 138.0,
    scf_kBT_meV: float = 0.1,
    scf_max_iter: int = 80,
    scf_tol_meV: float = 1e-5,
    eps_perp: float = 6.0,
    d_layer_nm: float = 0.335,
    D_sign: float = -1.0,
    mixer: str = "anderson",
    initial_U: str = "uploaded",
    stoner_u0_meV_A2: float = 7.9e4,
    stoner_JH_meV_A2: float = 2.4e4,
    stoner_temperature_K: float = 0.0,
    stoner_n_random_seeds: int = 20,
    stoner_seed: int = 0,
) -> SinglePointStonerState:
    """Run full SCF at one point, diagonalize K/Kp bands, and solve existing Stoner model.

    This deliberately does not modify existing SCF/Stoner code.
    """
    params_K = _make_params(
        theta_deg=theta_deg,
        cutoff=cutoff,
        valley=+1,
        omega_meV=omega_meV,
        wAA=wAA,
        wAB=wAB,
        sublattice_Z_meV=sublattice_Z_meV,
        a_cc_A=a_cc_A,
        gamma0_meV=gamma0_meV,
        gamma1_meV=gamma1_meV,
        gamma3_meV=gamma3_meV,
        gamma4_meV=gamma4_meV,
    )
    params_Kp = _make_params(
        theta_deg=theta_deg,
        cutoff=cutoff,
        valley=-1,
        omega_meV=omega_meV,
        wAA=wAA,
        wAB=wAB,
        sublattice_Z_meV=sublattice_Z_meV,
        a_cc_A=a_cc_A,
        gamma0_meV=gamma0_meV,
        gamma1_meV=gamma1_meV,
        gamma3_meV=gamma3_meV,
        gamma4_meV=gamma4_meV,
    )

    ham_K = TDBGContinuumHamiltonian(params_K)
    ham_Kp = TDBGContinuumHamiltonian(params_Kp)
    kpts, weights = make_uniform_mbz_grid(ham_K.geom, int(grid_n1), int(grid_n2))
    A_M_A2 = moire_cell_area_A2_from_weights(weights)

    scf_cfg = FullSCFConfig(
        target_density_cm2=float(n_cm2),
        D_Vnm=float(D_Vnm),
        degeneracy=4,
        kBT_meV=float(scf_kBT_meV),
        max_iter=int(scf_max_iter),
        tol_meV=float(scf_tol_meV),
        eps_perp=float(eps_perp),
        d_layer_nm=float(d_layer_nm),
        D_sign=float(D_sign),
        mixer=str(mixer),
        initial_U=str(initial_U),
        keep_eigensystem=False,
    )
    scf = FullSCFSolver(ham_K, kpts, weights).solve(scf_cfg)
    U = np.asarray(scf.U_meV, dtype=float)

    evals_K, evecs_K = ham_K.diagonalize(kpts, U)
    evals_Kp, evecs_Kp = ham_Kp.diagonalize(kpts, U)
    layer_masks = ham_K.layer_projectors_diagonal()

    nu_total = density_cm2_to_nu_total(float(n_cm2), A_M_A2)
    tables = build_full_band_flavor_tables(evals_K, evals_Kp, weights, A_M_A2)
    tables = restrict_tables_to_carrier_sector(tables, nu_total)

    stoner_params = StonerParams(
        u0_meV_A2=float(stoner_u0_meV_A2),
        JH_meV_A2=float(stoner_JH_meV_A2),
        temperature_K=float(stoner_temperature_K),
        n_random_seeds=int(stoner_n_random_seeds),
        seed=int(stoner_seed),
    )
    stoner = solve_stoner_fixed_nu(nu_total, tables, stoner_params, A_M_A2)
    ref = make_stoner_reference(stoner, tables, stoner_params, A_M_A2)

    return SinglePointStonerState(
        n_cm2=float(n_cm2),
        D_Vnm=float(D_Vnm),
        nu_total=float(nu_total),
        grid_shape=(int(grid_n1), int(grid_n2)),
        kpts=kpts,
        weights=weights,
        A_M_A2=float(A_M_A2),
        U_meV=U,
        scf_result=scf,
        stoner_result=stoner,
        stoner_params=stoner_params,
        stoner_reference=ref,
        tables=tables,
        ham_K=ham_K,
        ham_Kp=ham_Kp,
        evals_K_meV=evals_K,
        evecs_K=evecs_K,
        evals_Kp_meV=evals_Kp,
        evecs_Kp=evecs_Kp,
        layer_masks=layer_masks,
    )


def scan_q_for_state(
    state: SinglePointStonerState,
    susc_params: SusceptibilityParams,
    q_stride: int = 1,
    max_abs_q_step: int | None = None,
    include_gamma: bool = True,
) -> list[ChiQResult]:
    qpts = make_commensurate_q_points(
        state.ham_K.geom,
        state.grid_shape,
        q_stride=int(q_stride),
        max_abs_step=max_abs_q_step,
        include_gamma=include_gamma,
    )

    results: list[ChiQResult] = []
    for q in qpts:
        if susc_params.q_mode == "unfolded_diagonalize":
            kqpts = state.kpts + q.qvec_Ainv[None, :]
            evals_K_q, evecs_K_q = state.ham_K.diagonalize(kqpts, state.U_meV)
            evals_Kp_q, evecs_Kp_q = state.ham_Kp.diagonalize(kqpts, state.U_meV)
        else:
            evals_K_q = evecs_K_q = evals_Kp_q = evecs_Kp_q = None

        res = compute_transverse_chi_q(
            q=q,
            grid_shape=state.grid_shape,
            weights=state.weights,
            A_M_A2=state.A_M_A2,
            evals_K_meV=state.evals_K_meV,
            evecs_K=state.evecs_K,
            evals_Kp_meV=state.evals_Kp_meV,
            evecs_Kp=state.evecs_Kp,
            ref=state.stoner_reference,
            params=susc_params,
            layer_masks=state.layer_masks if susc_params.include_layer_matrix else None,
            evals_K_q_meV=evals_K_q,
            evecs_K_q=evecs_K_q,
            evals_Kp_q_meV=evals_Kp_q,
            evecs_Kp_q=evecs_Kp_q,
        )
        results.append(res)
    return results


def q_results_to_dataframe(results: list[ChiQResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_record() for r in results])


def summarize_q_scan(results: list[ChiQResult], finite_q_tol: float = 1e-3) -> dict:
    if not results:
        raise ValueError("empty q scan")

    gamma = [r for r in results if r.q.is_gamma]
    gamma_res = gamma[0] if gamma else min(results, key=lambda r: r.q.q_norm_Ainv)

    nonzero = [r for r in results if not r.q.is_gamma]
    max_all = max(results, key=lambda r: r.lambda_u_plus_hund)
    max_nonzero = max(nonzero, key=lambda r: r.lambda_u_plus_hund) if nonzero else max_all

    gamma_lambda = float(gamma_res.lambda_u_plus_hund)
    nonzero_lambda = float(max_nonzero.lambda_u_plus_hund)
    ratio = float(nonzero_lambda / gamma_lambda) if abs(gamma_lambda) > 1e-14 else np.inf
    finite_q_wins = bool(nonzero and nonzero_lambda > gamma_lambda * (1.0 + float(finite_q_tol)))

    out = {
        "gamma_dq1": int(gamma_res.q.dq1),
        "gamma_dq2": int(gamma_res.q.dq2),
        "gamma_chi_total_cell_meV_inv": float(gamma_res.chi_total_cell_meV_inv),
        "gamma_lambda_u": float(gamma_res.lambda_u),
        "gamma_lambda_u_plus_hund": float(gamma_res.lambda_u_plus_hund),
        "qstar_all_dq1": int(max_all.q.dq1),
        "qstar_all_dq2": int(max_all.q.dq2),
        "qstar_all_norm_Ainv": float(max_all.q.q_norm_Ainv),
        "qstar_all_chi_total_cell_meV_inv": float(max_all.chi_total_cell_meV_inv),
        "qstar_all_lambda_u": float(max_all.lambda_u),
        "qstar_all_lambda_u_plus_hund": float(max_all.lambda_u_plus_hund),
        "qstar_nonzero_dq1": int(max_nonzero.q.dq1),
        "qstar_nonzero_dq2": int(max_nonzero.q.dq2),
        "qstar_nonzero_norm_Ainv": float(max_nonzero.q.q_norm_Ainv),
        "qstar_nonzero_chi_total_cell_meV_inv": float(max_nonzero.chi_total_cell_meV_inv),
        "qstar_nonzero_lambda_u": float(max_nonzero.lambda_u),
        "qstar_nonzero_lambda_u_plus_hund": float(max_nonzero.lambda_u_plus_hund),
        "finite_q_lambda_ratio": ratio,
        "finite_q_delta_lambda": float(nonzero_lambda - gamma_lambda),
        "finite_q_wins": finite_q_wins,
    }
    if max_nonzero.layer_dipole_overlap is not None:
        out["qstar_nonzero_layer_dipole_overlap"] = float(max_nonzero.layer_dipole_overlap)
        out["qstar_nonzero_chi_sD_cell_meV_inv"] = float(max_nonzero.chi_sD_cell_meV_inv)
        out["qstar_nonzero_chi_s0_cell_meV_inv"] = float(max_nonzero.chi_s0_cell_meV_inv)
    return out


def write_single_point_outputs(out_dir: Path, state: SinglePointStonerState, q_results: list[ChiQResult], summary: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qdf = q_results_to_dataframe(q_results)
    qdf.to_csv(out_dir / "susceptibility_qmap.csv", index=False)

    payload = {
        "state": state.metadata(),
        "summary": summary,
    }
    (out_dir / "susceptibility_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

---

## 9. 新增例子：`examples/run_stoner_susceptibility_single.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdbg_scf.susceptibility import (
    SusceptibilityParams,
    solve_full_scf_stoner_state,
    scan_q_for_state,
    summarize_q_scan,
    q_results_to_dataframe,
)
from tdbg_scf.susceptibility.workflow import write_single_point_outputs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Finite-Q transverse susceptibility on a Stoner-after TDBG state")

    ap.add_argument("--n-cm2", type=float, required=True)
    ap.add_argument("--D-Vnm", type=float, required=True)
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1)
    ap.add_argument("--grid-n1", type=int, default=9)
    ap.add_argument("--grid-n2", type=int, default=9)

    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0)
    ap.add_argument("--a-cc-A", type=float, default=1.420)
    ap.add_argument("--gamma0-meV", type=float, default=2610.0)
    ap.add_argument("--gamma1-meV", type=float, default=361.0)
    ap.add_argument("--gamma3-meV", type=float, default=283.0)
    ap.add_argument("--gamma4-meV", type=float, default=138.0)

    ap.add_argument("--eps-perp", type=float, default=6.0)
    ap.add_argument("--d-layer-nm", type=float, default=0.335)
    ap.add_argument("--D-sign", type=float, default=-1.0)
    ap.add_argument("--scf-kBT-meV", type=float, default=0.1)
    ap.add_argument("--scf-max-iter", type=int, default=80)
    ap.add_argument("--scf-tol-meV", type=float, default=1e-5)
    ap.add_argument("--mixer", type=str, default="anderson")
    ap.add_argument("--initial-U", type=str, default="uploaded")

    ap.add_argument("--stoner-u0-meV-A2", type=float, default=7.9e4)
    ap.add_argument("--stoner-JH-meV-A2", type=float, default=2.4e4)
    ap.add_argument("--stoner-n-random-seeds", type=int, default=20)
    ap.add_argument("--stoner-seed", type=int, default=0)

    ap.add_argument("--chi-kBT-meV", type=float, default=0.05)
    ap.add_argument("--energy-window-meV", type=float, default=30.0)
    ap.add_argument("--max-bands-per-k", type=int, default=24)
    ap.add_argument("--q-mode", choices=["folded_grid", "unfolded_diagonalize"], default="folded_grid")
    ap.add_argument("--q-stride", type=int, default=1)
    ap.add_argument("--max-abs-q-step", type=int, default=None)
    ap.add_argument("--no-layer-matrix", action="store_true")

    ap.add_argument("--out", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    state = solve_full_scf_stoner_state(
        n_cm2=args.n_cm2,
        D_Vnm=args.D_Vnm,
        theta_deg=args.theta_deg,
        cutoff=args.cutoff,
        grid_n1=args.grid_n1,
        grid_n2=args.grid_n2,
        omega_meV=args.omega,
        wAA=args.wAA,
        wAB=args.wAB,
        sublattice_Z_meV=args.Z,
        a_cc_A=args.a_cc_A,
        gamma0_meV=args.gamma0_meV,
        gamma1_meV=args.gamma1_meV,
        gamma3_meV=args.gamma3_meV,
        gamma4_meV=args.gamma4_meV,
        scf_kBT_meV=args.scf_kBT_meV,
        scf_max_iter=args.scf_max_iter,
        scf_tol_meV=args.scf_tol_meV,
        eps_perp=args.eps_perp,
        d_layer_nm=args.d_layer_nm,
        D_sign=args.D_sign,
        mixer=args.mixer,
        initial_U=args.initial_U,
        stoner_u0_meV_A2=args.stoner_u0_meV_A2,
        stoner_JH_meV_A2=args.stoner_JH_meV_A2,
        stoner_n_random_seeds=args.stoner_n_random_seeds,
        stoner_seed=args.stoner_seed,
    )

    chi_params = SusceptibilityParams(
        kBT_meV=args.chi_kBT_meV,
        energy_window_meV=args.energy_window_meV,
        max_bands_per_k=args.max_bands_per_k,
        q_mode=args.q_mode,
        include_layer_matrix=not args.no_layer_matrix,
    )
    q_results = scan_q_for_state(
        state,
        chi_params,
        q_stride=args.q_stride,
        max_abs_q_step=args.max_abs_q_step,
        include_gamma=True,
    )
    summary = summarize_q_scan(q_results, finite_q_tol=chi_params.finite_q_tol)
    write_single_point_outputs(args.out, state, q_results, summary)

    # Quick q-map plot.
    qdf = q_results_to_dataframe(q_results)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(qdf["qx_Ainv"], qdf["qy_Ainv"], c=qdf["lambda_u_plus_hund"], s=36)
    ax.set_xlabel(r"$Q_x$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$Q_y$ [$\AA^{-1}$]")
    ax.set_title("Stoner-after transverse susceptibility")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\lambda_{U+J}(Q)$")
    fig.savefig(args.out / "susceptibility_qmap.png", dpi=180)
    plt.close(fig)

    print("Saved:", args.out)
    print(summary)


if __name__ == "__main__":
    main()
```

### 单点运行示例

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 9 --grid-n2 9 \
  --n-cm2 2.0e12 \
  --D-Vnm 0.50 \
  --stoner-u0-meV-A2 7.9e4 \
  --stoner-JH-meV-A2 2.4e4 \
  --chi-kBT-meV 0.05 \
  --energy-window-meV 30 \
  --max-bands-per-k 24 \
  --q-stride 1 \
  --out outputs/chi_single_n2_D0p5
```

更严格但慢的 gauge check：

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 7 --grid-n2 7 \
  --n-cm2 2.0e12 \
  --D-Vnm 0.50 \
  --q-mode unfolded_diagonalize \
  --max-abs-q-step 2 \
  --out outputs/chi_single_unfolded_check
```

---

## 10. 新增例子：`examples/run_nd_mapping_stoner_susceptibility.py`

这个脚本读已有的 `nd_map.csv`，对筛选后的 `(n,D)` 点逐点重算 SCF + Stoner + finite-Q susceptibility。这样不需要改现有 mapping 脚本。第一次建议只扫 halo 附近的少量点。

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdbg_scf.susceptibility import (
    SusceptibilityParams,
    solve_full_scf_stoner_state,
    scan_q_for_state,
    summarize_q_scan,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Add finite-Q Stoner-after susceptibility diagnostics to an n-D map")
    ap.add_argument("--input-map", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")

    # Filtering controls. Defaults focus on conduction half-filling-ish region but are intentionally broad.
    ap.add_argument("--nu-min", type=float, default=None)
    ap.add_argument("--nu-max", type=float, default=None)
    ap.add_argument("--D-min", type=float, default=None)
    ap.add_argument("--D-max", type=float, default=None)
    ap.add_argument("--spin-min", type=float, default=None, help="Filter by existing spin_polarization_norm")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--row-stride", type=int, default=1)

    # Model / numerical controls.
    ap.add_argument("--theta-deg", type=float, default=1.35)
    ap.add_argument("--cutoff", type=int, default=1)
    ap.add_argument("--grid-n1", type=int, default=9)
    ap.add_argument("--grid-n2", type=int, default=9)
    ap.add_argument("--omega", type=float, default=100.0)
    ap.add_argument("--wAA", type=float, default=0.8)
    ap.add_argument("--wAB", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=15.0)
    ap.add_argument("--eps-perp", type=float, default=6.0)
    ap.add_argument("--D-sign", type=float, default=-1.0)
    ap.add_argument("--scf-kBT-meV", type=float, default=0.1)
    ap.add_argument("--scf-max-iter", type=int, default=80)
    ap.add_argument("--scf-tol-meV", type=float, default=1e-5)

    ap.add_argument("--stoner-u0-meV-A2", type=float, default=7.9e4)
    ap.add_argument("--stoner-JH-meV-A2", type=float, default=2.4e4)

    ap.add_argument("--chi-kBT-meV", type=float, default=0.05)
    ap.add_argument("--energy-window-meV", type=float, default=30.0)
    ap.add_argument("--max-bands-per-k", type=int, default=24)
    ap.add_argument("--q-mode", choices=["folded_grid", "unfolded_diagonalize"], default="folded_grid")
    ap.add_argument("--q-stride", type=int, default=1)
    ap.add_argument("--max-abs-q-step", type=int, default=None)
    ap.add_argument("--no-layer-matrix", action="store_true")
    return ap


def filter_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    if args.nu_min is not None:
        out = out[out["nu_total"] >= float(args.nu_min)]
    if args.nu_max is not None:
        out = out[out["nu_total"] <= float(args.nu_max)]
    if args.D_min is not None:
        out = out[out["D_Vnm"] >= float(args.D_min)]
    if args.D_max is not None:
        out = out[out["D_Vnm"] <= float(args.D_max)]
    if args.spin_min is not None and "spin_polarization_norm" in out.columns:
        out = out[out["spin_polarization_norm"] >= float(args.spin_min)]
    if int(args.row_stride) > 1:
        out = out.iloc[:: int(args.row_stride)]
    if args.max_points is not None:
        out = out.head(int(args.max_points))
    return out.reset_index(drop=True)


def row_key(row: pd.Series) -> tuple:
    if "n_index" in row and "D_index" in row:
        return (int(row["n_index"]), int(row["D_index"]))
    return (round(float(row["n_cm2"]), 3), round(float(row["D_Vnm"]), 6))


def main() -> None:
    args = build_parser().parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.input_map)
    todo = filter_rows(source, args)

    done_keys = set()
    if args.resume and args.out_csv.exists():
        old = pd.read_csv(args.out_csv)
        for _, r in old.iterrows():
            done_keys.add(row_key(r))
        records = old.to_dict("records")
    else:
        records = []

    chi_params = SusceptibilityParams(
        kBT_meV=args.chi_kBT_meV,
        energy_window_meV=args.energy_window_meV,
        max_bands_per_k=args.max_bands_per_k,
        q_mode=args.q_mode,
        include_layer_matrix=not args.no_layer_matrix,
    )

    for count, row in todo.iterrows():
        key = row_key(row)
        if key in done_keys:
            print(f"skip existing {key}")
            continue

        t0 = time.time()
        n_cm2 = float(row["n_cm2"])
        D_Vnm = float(row["D_Vnm"])
        print(f"[{count+1}/{len(todo)}] n={n_cm2:.6e} cm^-2 D={D_Vnm:.4f} V/nm")

        try:
            state = solve_full_scf_stoner_state(
                n_cm2=n_cm2,
                D_Vnm=D_Vnm,
                theta_deg=args.theta_deg,
                cutoff=args.cutoff,
                grid_n1=args.grid_n1,
                grid_n2=args.grid_n2,
                omega_meV=args.omega,
                wAA=args.wAA,
                wAB=args.wAB,
                sublattice_Z_meV=args.Z,
                eps_perp=args.eps_perp,
                D_sign=args.D_sign,
                scf_kBT_meV=args.scf_kBT_meV,
                scf_max_iter=args.scf_max_iter,
                scf_tol_meV=args.scf_tol_meV,
                stoner_u0_meV_A2=args.stoner_u0_meV_A2,
                stoner_JH_meV_A2=args.stoner_JH_meV_A2,
            )
            q_results = scan_q_for_state(
                state,
                chi_params,
                q_stride=args.q_stride,
                max_abs_q_step=args.max_abs_q_step,
                include_gamma=True,
            )
            summary = summarize_q_scan(q_results, finite_q_tol=chi_params.finite_q_tol)

            rec = row.to_dict()
            rec.update(state.metadata())
            rec.update(summary)
            rec["chi_status"] = "ok"
            rec["chi_runtime_s"] = time.time() - t0
        except Exception as exc:
            rec = row.to_dict()
            rec["chi_status"] = "error"
            rec["chi_error"] = repr(exc)
            rec["chi_runtime_s"] = time.time() - t0
            print("ERROR", repr(exc))

        records.append(rec)
        pd.DataFrame(records).to_csv(args.out_csv, index=False)

    print("saved", args.out_csv)


if __name__ == "__main__":
    main()
```

### n-D map 运行示例

先只取 halo 附近少量点，避免一上来全图太慢：

```bash
PYTHONPATH=. python examples/run_nd_mapping_stoner_susceptibility.py \
  --input-map /path/to/nd_map.csv \
  --out-csv outputs/nd_map_with_finite_q_chi.csv \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 9 --grid-n2 9 \
  --nu-min 1.0 --nu-max 2.5 \
  --D-min 0.25 --D-max 0.65 \
  --spin-min 0.5 \
  --max-points 40 \
  --q-stride 1 \
  --energy-window-meV 30 \
  --max-bands-per-k 24 \
  --resume
```

如果先做快速小-Q 判断：

```bash
PYTHONPATH=. python examples/run_nd_mapping_stoner_susceptibility.py \
  --input-map /path/to/nd_map.csv \
  --out-csv outputs/nd_map_with_small_q_chi.csv \
  --nu-min 1.0 --nu-max 2.5 \
  --D-min 0.25 --D-max 0.65 \
  --spin-min 0.5 \
  --grid-n1 9 --grid-n2 9 \
  --max-abs-q-step 3 \
  --q-stride 1 \
  --max-points 80 \
  --resume
```

---

## 11. 输出列怎么看

### 11.1 单点 qmap

`outputs/.../susceptibility_qmap.csv` 每一行是一个 Q 点：

```text
dq1, dq2, qx_Ainv, qy_Ainv, q_norm_Ainv
chi_total_cell_meV_inv
chi_K_cell_meV_inv
chi_Kp_cell_meV_inv
lambda_u
lambda_u_plus_hund
chi_s0_cell_meV_inv
chi_sD_cell_meV_inv
layer_dipole_overlap
chi_layer_L1_L1_cell_meV_inv ... chi_layer_L4_L4_cell_meV_inv
```

最重要的先看：

```text
lambda_u_plus_hund(Q)
```

如果你想先不信 Hund 的 transverse extension，就看：

```text
lambda_u(Q)
```

两者的 Q* 如果一致，结论更稳。

### 11.2 单点 summary

`outputs/.../susceptibility_summary.json` 里：

```text
gamma_lambda_u_plus_hund
qstar_nonzero_lambda_u_plus_hund
finite_q_lambda_ratio
finite_q_delta_lambda
finite_q_wins
qstar_nonzero_dq1, qstar_nonzero_dq2
qstar_nonzero_layer_dipole_overlap
```

判断逻辑：

```text
finite_q_wins = True
```

表示：

```text
lambda(Q* != 0) > lambda(Gamma) * (1 + tol)
```

这是 “Stoner 后 q=0 polarized reference 对 finite-Q transverse mode 更软” 的直接数值诊断。

---

## 12. 必做 sanity checks

### 12.1 q=0 normalization

在 `Q=0`：

```text
lambda(Gamma) = U_eff * chi(Gamma)
```

不一定严格等于 1，因为：

1. 你现在的 Stoner 是 phenomenological flavor filling model；
2. transverse SU(2) Goldstone 在这个 minimal density-functional 里不一定精确满足 Ward identity；
3. 代码用了有限温度、有限 band window、有限 k mesh。

所以第一版不要用绝对值过度解释，先看：

```text
finite_q_lambda_ratio = lambda(Q* != 0) / lambda(Gamma)
```

这个 ratio 对整体 normalization 不敏感。

### 12.2 folded-grid vs unfolded-diagonalize

`folded_grid` 很快，但在 MBZ 边界涉及 Bloch gauge / plane-wave G-index folding。初筛可以用它。

对文章图或关键点，必须抽几个点用：

```bash
--q-mode unfolded_diagonalize
```

做 check。如果 Q* 位置和 finite-Q tendency 在两个模式下都存在，才比较稳。

### 12.3 band window convergence

建议至少做：

```text
energy_window_meV = 20, 30, 50
max_bands_per_k  = 12, 24, 40
```

如果 Q* 稳定，说明不是 band truncation artifact。

### 12.4 k mesh convergence

建议：

```text
7x7 -> 9x9 -> 11x11 -> 15x15
```

初始可以 cutoff=1；关键图再 cutoff=2。

### 12.5 symmetry check

在没有显式破坏 C3 的数值设置下，应该检查：

```text
chi(Q) approximately equals chi(C3 Q)
chi(Q) approximately equals chi(-Q)
```

有限 cutoff 和有限 mesh 会有偏差，但大的不对称通常说明 q folding / gauge 有问题。

---

## 13. 推荐的文章级 workflow

### 第一阶段：快速定位

1. 选 3 个点：dome center、halo boundary、normal outside。
2. 用 `folded_grid` + `9x9` + `cutoff=1` 做完整 Q map。
3. 看 `lambda(Q)` 的最大值是否从 Gamma 移到 finite Q。

### 第二阶段：halo line scan

在已有 `nd_map.csv` 里筛选：

```text
1.0 < nu_total < 2.5
0.25 < D < 0.65
spin_polarization_norm > 0.5
```

输出：

```text
finite_q_lambda_ratio(nu,D)
qstar_nonzero_norm_Ainv(nu,D)
qstar_nonzero_layer_dipole_overlap(nu,D)
```

### 第三阶段：严格验证

对最强的 5-10 个 halo 点：

```text
q_mode = unfolded_diagonalize
k grid = 11x11 or 15x15
cutoff = 2
energy window convergence
```

### 第四阶段：HF seed

如果 `Q* != 0` 且 `layer_dipole_overlap` 在 halo 增强，下一步再用 Q* 作为 seed 做 finite-Q variational HF / unrestricted HF。

---

## 14. 需要特别小心的两个理论点

### 14.1 这里不是在 claim skyrmion ground state

这个新增模块只能说明：

```text
Stoner 后的 q=0 reference 在 quadratic level 对 finite-Q transverse mode 变软。
```

它能支持 SDW / spiral / skyrmion-like texture 的前驱不稳定性，但不能单独证明最终基态就是 skyrmion lattice。若要强 claim skyrmion，需要后续比较 single-Q、double-Q、triple-Q 的 HF energy。

### 14.2 Hund transverse vertex 是 phenomenological extension

`lambda_u_plus_hund` 用的是一个自然但 phenomenological 的 valley-pair transverse vertex：

```text
Gamma = [[U_cell, J_cell],
         [J_cell, U_cell]]
```

如果不想引入这个假设，可以只报告：

```text
lambda_u = U_cell * (chi_K + chi_Kp)
```

我建议两个都输出。若两个 Q* 一致，文章里更容易防守。

---

## 15. Codex 执行清单

请 Codex 按以下顺序集成：

1. 新建 `tdbg_scf/susceptibility/` 文件夹。
2. 添加上面 5 个模块：`__init__.py`, `params.py`, `qmesh.py`, `stoner_reference.py`, `bubble.py`, `workflow.py`。
3. 添加两个 examples。
4. 不要改动现有 `tdbg_scf/stoner`、`solver_full.py`、`solver_projected.py`。
5. 运行：

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 5 --grid-n2 5 \
  --n-cm2 2.0e12 \
  --D-Vnm 0.50 \
  --q-stride 2 \
  --max-abs-q-step 2 \
  --out outputs/chi_smoke_test
```

6. 检查输出文件是否存在：

```text
outputs/chi_smoke_test/susceptibility_qmap.csv
outputs/chi_smoke_test/susceptibility_summary.json
outputs/chi_smoke_test/susceptibility_qmap.png
```

7. 再做一个 unfolded check：

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 5 --grid-n2 5 \
  --n-cm2 2.0e12 \
  --D-Vnm 0.50 \
  --q-mode unfolded_diagonalize \
  --max-abs-q-step 1 \
  --out outputs/chi_unfolded_smoke_test
```

8. 如果 smoke test 通过，再跑 halo n-D mapping 子集。
