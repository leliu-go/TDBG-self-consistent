# Complete integration note: Stoner-after generalized finite-Q transverse susceptibility for TDBG

**Status:** this document replaces the earlier finite-Q susceptibility integration note and all follow-up patches. Give **only this file** to Codex. Do not separately apply the older follow-up documents.

**Main correction relative to the first implementation:** the bare bubble `chi0(Q)` is not the final Stoner criterion. The paper-level diagnostic must use a generalized Stoner/RPA eigenvalue

```text
lambda_max(Q) = largest eigenvalue of a Hermitianized interaction-dressed kernel.
```

The previously used scalar quantities such as

```text
lambda_u = u_cell * (chi_K + chi_Kp)
lambda_u_plus_hund with Gamma = [[u_cell, J_cell], [J_cell, u_cell]]
```

should be retained only as **legacy diagnostics**, not as the main paper-level criterion.

---

## 0. Executive summary for Codex

Implement a **Stoner-after finite-Q transverse susceptibility** module with the following defaults:

```text
reference_state          = Stoner-after q=0 flavor-polarized state
occupation_mode          = flavor_mu
spin_flip_mode           = both_pm
main_vertex_model        = su4_diag
optional_vertex_model    = su2_hund_factor2
legacy_vertex_models     = legacy_scalar_total, legacy_offdiag_J
q_mode_for_screening     = folded_grid, but only for quick scan
q_mode_for_paper_points  = unfolded_diagonalize or folded_grid_with_G_shift
main_output              = lambda_selected_norm and finite_q_lambda_ratio_selected
```

Interpretation:

```text
finite_q_wins = True
```

means:

```text
max_{Q != 0} lambda_selected(Q) > lambda_selected(Q=0) * (1 + tol)
```

This means the existing q=0 Stoner reference state is softer to a finite-Q transverse spin mode than to the uniform Q=0 transverse mode at the quadratic/RPA level. It does **not** by itself prove a skyrmion ground state; it identifies the leading soft mode to seed finite-Q HF or variational calculations.

---

## 1. Clarify the two lambda formulas

Two equivalent Hermitian kernels may appear:

```text
K_Gamma = Gamma^{1/2} chi0 Gamma^{1/2}
K_chi   = chi0^{1/2} Gamma chi0^{1/2}
```

They are **not physically different** if both `Gamma` and `chi0` are Hermitian positive semidefinite. They have the same nonzero eigenvalues because they are `A A†` and `A† A` with

```text
A = Gamma^{1/2} chi0^{1/2}.
```

Therefore,

```text
lambda_max(Gamma^{1/2} chi0 Gamma^{1/2})
=
lambda_max(chi0^{1/2} Gamma chi0^{1/2})
```

under the assumptions above.

The underlying RPA/Stoner criterion is really

```text
det[I - Gamma chi0(Q)] = 0.
```

Because `Gamma chi0` is not necessarily Hermitian, use either of the Hermitianized forms above for stable numerical eigenvalues. In this project, implement:

```python
kernel = sqrt_chi @ Gamma @ sqrt_chi
lambda_max = eigvalsh(kernel)[-1]
```

because `chi0` is a small matrix in valley-pair space and is easy to diagonalize safely.

If `chi0` has small negative eigenvalues from numerical noise, Hermitize it and clip eigenvalues below zero to zero before taking the square root.

---

## 2. Existing Stoner functional in the current code

The current four-flavor Stoner energy functional is

```text
E[nu_f] = E_kin[nu_f]
        + 0.5 * u_cell * (nu_total^2 - sum_f nu_f^2)
        - J_cell * m_K * m_Kp
```

with

```text
u_cell = u0_meV_A2 / A_M_A2
J_cell = JH_meV_A2 / A_M_A2
m_K  = nu_K_up  - nu_K_down
m_Kp = nu_Kp_up - nu_Kp_down
```

Flavor convention:

```text
0 = K_up
1 = Kp_up
2 = K_down
3 = Kp_down
```

The first finite-Q implementation already used this derivative correctly for the Stoner-after self-energy. Keep that part, but also store more diagnostics.

---

## 3. Stoner-after reference state

### 3.1 Self-energy from the current Stoner energy

Use

```text
Sigma_f = dE_int / dnu_f
```

where `E_int` excludes the kinetic energy. Explicitly:

```text
Sigma_0 = u_cell * (nu_total - nu_0) - J_cell * (nu_1 - nu_3)
Sigma_1 = u_cell * (nu_total - nu_1) - J_cell * (nu_0 - nu_2)
Sigma_2 = u_cell * (nu_total - nu_2) + J_cell * (nu_1 - nu_3)
Sigma_3 = u_cell * (nu_total - nu_3) + J_cell * (nu_0 - nu_2)
```

Store both versions:

```text
sigma_full_f_meV      = the derivative above
sigma_shifted_f_meV   = sigma_full_f_meV - mean(sigma_full_f_meV)
```

Use `sigma_shifted_f_meV` in particle-hole denominators because a common shift cancels. Store `sigma_full_f_meV` for reproducibility and validation.

### 3.2 Chemical potentials and occupations

The kinetic flavor chemical potentials from the Stoner solver are

```text
mu_f_kin = dE_kin / dnu_f.
```

At an interior Stoner minimum, the electrochemical quantities

```text
mu_eff_f = mu_f_kin + sigma_full_f
```

should be equal for all active flavors, up to numerical tolerance. But if a flavor is pinned at a band-capacity boundary, `mu_eff_f` may not be equal. Therefore:

```text
DEFAULT: occupation_mode = flavor_mu
```

For the bare bubble numerator, use flavor-resolved occupations:

```text
f_i = f(eps_initial + sigma_initial_shifted - mu_initial_kin_shifted)
f_f = f(eps_final   + sigma_final_shifted   - mu_final_kin_shifted)
```

Equivalently, since common shifts cancel, implement as

```text
Ebar_f(k,n) = eps_tau(k,n) + sigma_shifted_f
mubar_f     = mu_f_kin + sigma_shifted_f
f_f(k,n)    = f(Ebar_f(k,n) - mubar_f)
```

The denominator must still use Stoner-after quasiparticle energies:

```text
DeltaE = [eps_final + sigma_shifted_final]
       - [eps_initial + sigma_shifted_initial]
```

Also compute the diagnostic spread:

```text
mu_eff_spread_meV = max(mu_f_kin + sigma_full_f) - min(mu_f_kin + sigma_full_f)
```

If `mu_eff_spread_meV` is large, warn that a common-mu Stoner-after bubble may be unreliable; the flavor-resolved occupation mode remains the safer default.

---

## 4. Finite-Q transverse bubbles

For each valley and spin-flip direction, compute:

### 4.1 Plus channel: down -> up

```text
chi_K_plus(Q):   K_down(k,n)  -> K_up(k+Q,m)
chi_Kp_plus(Q):  Kp_down(k,n) -> Kp_up(k+Q,m)
```

### 4.2 Minus channel: up -> down

```text
chi_K_minus(Q):  K_up(k,n)   -> K_down(k+Q,m)
chi_Kp_minus(Q): Kp_up(k,n)  -> Kp_down(k+Q,m)
```

Because the Stoner-after state is spin-polarized, `chi_plus` and `chi_minus` may not be identical. Compute both and let the final selected lambda use the larger generalized Stoner eigenvalue.

For a channel `a -> b`:

```text
chi_tau^{a->b}(Q)
= sum_{k,m,n} A_M * w_k * |Lambda_tau_mn(k,Q)|^2
  * [f(E_{tau,a,n,k} - mu_a) - f(E_{tau,b,m,k+Q} - mu_b)]
  / [E_{tau,b,m,k+Q} - E_{tau,a,n,k}]
```

where

```text
Lambda_tau_mn(k,Q) = <u_tau,m,k+Q | u_tau,n,k>
```

or, for layer-resolved diagnostics,

```text
Lambda_tau,l,mn(k,Q) = <u_tau,m,k+Q | P_l | u_tau,n,k>.
```

Use the finite-temperature derivative limit when the denominator is below `denom_tol_meV`.

---

## 5. Interaction vertex models

The finite-Q Stoner criterion needs an interaction vertex `Gamma`. The existing code only defines a **longitudinal flavor-filling functional**, not a unique transverse spin-flip Hamiltonian. Therefore, implement several vertex models and make the main interpretation robust.

### 5.1 Recommended baseline: `su4_diag`

This is the most conservative vertex that follows directly from the SU(4)-like `u_cell` flavor-polarization term:

```text
Gamma_su4_diag = [[u_cell, 0],
                  [0,      u_cell]]
```

This acts in the two-dimensional valley-pair space:

```text
M = (M_K, M_Kp)
```

where `M_K = <S_K^+(Q)>` or `<S_K^-(Q)>` depending on the channel.

The corresponding generalized Stoner score is

```text
lambda_su4_diag(Q) = eigmax[ sqrt(chi_valley) Gamma_su4_diag sqrt(chi_valley) ]
```

with

```text
chi_valley = diag(chi_K, chi_Kp)
```

for the simplest valley-diagonal bubble.

Do **not** replace this by `u_cell * (chi_K + chi_Kp)` as the main criterion. The scalar total version assumes an additional valley-coherent interaction channel not directly implied by the current energy functional.

### 5.2 Optional SU(2)-rotated Hund extension: `su2_hund_factor2`

The code contains a longitudinal Hund-like term

```text
E_Hund = -J_cell * m_K * m_Kp
```

with

```text
m_tau = nu_tau_up - nu_tau_down = 2 S_tau^z.
```

If we impose spin-rotational invariance, the longitudinal term is extended as

```text
-J_cell * m_K * m_Kp
= -4 J_cell S_K^z S_Kp^z
--> -4 J_cell S_K · S_Kp.
```

Using

```text
S_K · S_Kp = S_K^z S_Kp^z
           + 0.5 * (S_K^+ S_Kp^- + S_K^- S_Kp^+),
```

the transverse part becomes

```text
-2 J_cell * (S_K^+ S_Kp^- + S_K^- S_Kp^+).
```

Therefore, with the convention

```text
S^+ = c_up^dagger c_down,
S^z = (n_up - n_down)/2,
```

the SU(2)-rotated transverse Hund vertex is

```text
Gamma_su2_hund_factor2 = [[u_cell, 2*J_cell],
                          [2*J_cell, u_cell]]
```

This **factor 2 is not already present in the current code**. It is an optional SU(2)-symmetric extension of the existing longitudinal term. It should not be hidden as a default assumption. Output it separately and label it clearly.

### 5.3 Legacy diagnostics only

Keep these only for continuity with the first implementation:

```text
legacy_scalar_total:
    lambda = u_cell * (chi_K + chi_Kp)

legacy_offdiag_J:
    Gamma = [[u_cell, J_cell],
             [J_cell, u_cell]]
```

Do not use these as the main criterion in paper-level plots.

### 5.4 Suggested main reporting strategy

For main figures, use:

```text
main_vertex_model = su4_diag
```

For robustness / Supplementary:

```text
compare su4_diag vs su2_hund_factor2.
```

If finite-Q wins in both models and the same `Q*` appears, the conclusion is robust. If finite-Q only wins with `su2_hund_factor2`, phrase the result as dependent on the SU(2)-rotated Hund extension.

---

## 6. Generalized Stoner eigenvalue implementation

Create a new file:

```text
tdbg_scf/susceptibility/vertex.py
```

with the following core functions.

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VertexSpec:
    name: str
    gamma: np.ndarray
    description: str


def hermitize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def psd_sqrt(a: np.ndarray, clip_tol: float = 1e-12) -> np.ndarray:
    """Hermitian PSD square root with small negative eigenvalues clipped."""
    h = hermitize(a)
    vals, vecs = np.linalg.eigh(h)
    vals = np.where(vals > clip_tol, vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def make_valley_vertex_models(
    u_cell_meV: float,
    J_cell_meV: float,
    include_legacy: bool = True,
    hund_transverse_factor: float = 2.0,
) -> dict[str, VertexSpec]:
    """Return valley-pair transverse spin vertices.

    Valley-pair basis: (K spin-flip pair, K' spin-flip pair).

    su4_diag:
        Direct conservative vertex from the SU(4)-like u term.

    su2_hund_factor2:
        Optional spin-rotational extension of the code's longitudinal
        -J*m_K*m_Kp term. The default factor 2 follows from
        m_tau = 2 S_tau^z and S dot S = SzSz + 1/2(S+S- + S-S+).

    legacy models:
        Kept only to compare with early outputs.
    """
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
    """Return largest generalized Stoner eigenvalue and eigenvector.

    Uses the Hermitian kernel sqrt(chi) Gamma sqrt(chi).
    chi_valley and gamma should have the same dimension.
    """
    chi = hermitize(np.asarray(chi_valley, dtype=np.complex128))
    gam = hermitize(np.asarray(gamma, dtype=np.complex128))

    if clip_chi:
        sqrt_chi = psd_sqrt(chi)
    else:
        vals, vecs = np.linalg.eigh(chi)
        sqrt_chi = (vecs * np.sqrt(vals.astype(np.complex128))) @ vecs.conj().T

    if clip_gamma:
        gam = psd_sqrt(gam) @ psd_sqrt(gam)

    kernel = hermitize(sqrt_chi @ gam @ sqrt_chi)
    vals, vecs = np.linalg.eigh(kernel)
    return float(np.real(vals[-1])), vecs[:, -1]


def legacy_scalar_total_lambda(chi_K: float, chi_Kp: float, u_cell_meV: float) -> float:
    return float(u_cell_meV) * float(chi_K + chi_Kp)
```

---

## 7. Required changes to result objects

Replace the old `ChiQResult` fields with a richer version. Keep old names only as legacy fields if needed.

Recommended fields:

```text
q metadata:
    dq1, dq2, qx_Ainv, qy_Ainv, q_norm_Ainv, is_gamma

bare bubbles:
    chi_K_plus
    chi_Kp_plus
    chi_K_minus
    chi_Kp_minus

main matrix Stoner scores:
    lambda_su4_diag_plus
    lambda_su4_diag_minus
    lambda_su4_diag_selected

optional Hund-extension scores:
    lambda_su2_hund_factor2_plus
    lambda_su2_hund_factor2_minus
    lambda_su2_hund_factor2_selected

legacy diagnostics:
    lambda_legacy_scalar_total_plus
    lambda_legacy_scalar_total_minus
    lambda_legacy_offdiag_J_plus
    lambda_legacy_offdiag_J_minus

selected mode metadata:
    selected_vertex_model
    selected_spin_flip_direction     # plus or minus
    selected_lambda_raw
    selected_valley_eigenvector_K_real/imag
    selected_valley_eigenvector_Kp_real/imag

layer diagnostics aligned to selected mode:
    selected_chi_s0
    selected_chi_sD
    selected_layer_dipole_overlap
```

---

## 8. How to align layer diagnostics with the selected Stoner mode

The first implementation computed a valley-summed layer matrix and took its leading eigenvector. That is useful, but it may not correspond to the **same mode** that maximizes the generalized Stoner eigenvalue.

Instead, after selecting a vertex and spin-flip direction, obtain the valley eigenvector

```text
v = (v_K, v_Kp)
```

from the selected generalized Stoner kernel. Then construct the selected layer matrix as

```text
chi_layer_selected = |v_K|^2  * chi_layer_K_selected_direction
                   + |v_Kp|^2 * chi_layer_Kp_selected_direction
```

If you later implement intervalley off-diagonal bubbles, include cross terms. For the current valley-diagonal bubble, the weighted sum above is sufficient.

Then compute:

```text
one  = (1, 1, 1, 1)
zeta = (1.5, 0.5, -0.5, -1.5)  # or repository convention

chi_s0 = one.T  @ chi_layer_selected @ one
chi_sD = zeta.T @ chi_layer_selected @ zeta
O_D    = |eigvec_layer_leading^† zeta|^2 / (||eigvec||^2 ||zeta||^2)
```

Also output a simpler ratio:

```text
layer_dipole_ratio = chi_sD / max(abs(chi_s0), tiny)
```

This makes it easier to compare along the n-D map.

---

## 9. Occupation and denominator implementation details

Add an internal helper:

```python
def shifted_energy_and_mu(eps_tau, sigma_shifted_f, mu_kin_f):
    Ebar = eps_tau + sigma_shifted_f
    mubar = mu_kin_f + sigma_shifted_f
    return Ebar, mubar
```

For transition `f_i -> f_f`:

```text
Ei_bar, mui_bar = shifted_energy_and_mu(eps_initial, sigma_i, mu_i_kin)
Ef_bar, muf_bar = shifted_energy_and_mu(eps_final,   sigma_f, mu_f_kin)

num = f(Ei_bar - mui_bar) - f(Ef_bar - muf_bar)
den = Ef_bar - Ei_bar
ratio = num / den
```

Do not use a single common `mu` as the default. Keep `common_mu` as an optional comparison mode:

```text
occupation_mode = common_mu
```

but make `flavor_mu` default.

---

## 10. Q folding and form factor requirements

### 10.1 Quick mode

For screening maps, `folded_grid` is acceptable if the goal is only to quickly locate candidate halo points.

### 10.2 Paper-level mode

For final figures or claims about Q*, use one of:

```text
q_mode = unfolded_diagonalize
```

or implement a corrected folded mode:

```text
q_mode = folded_grid_with_G_shift
```

If `k + Q` is folded back as

```text
k + Q = k_folded + G_fold
```

then the plane-wave components of the final state must be shifted consistently before evaluating

```text
<u(k+Q)|P_l|u(k)>.
```

A naive overlap

```text
<u(k_folded)|u(k)>
```

can produce gauge/folding artifacts near the MBZ boundary.

Mandatory checks:

```text
chi(Q) ≈ chi(-Q)
chi(Q) ≈ chi(C3 Q) when C3 is not explicitly broken
folded_grid candidate Q* survives unfolded_diagonalize on selected points
```

---

## 11. Summary logic

Implement summary for each single point:

```text
gamma_lambda_su4_diag_selected
qstar_nonzero_lambda_su4_diag_selected
finite_q_ratio_su4_diag
finite_q_wins_su4_diag

same for su2_hund_factor2
same for legacy diagnostics
```

Define:

```text
finite_q_ratio_model = max_{Q != 0} lambda_model_selected(Q) / lambda_model_selected(Q=0)
finite_q_wins_model  = finite_q_ratio_model > 1 + finite_q_tol
```

For the main paper-level conclusion, use:

```text
finite_q_wins_su4_diag
```

and discuss whether it remains true under

```text
su2_hund_factor2.
```

---

## 12. n-D mapping outputs

For each `(n,D)` row, append:

```text
# Reference validation
mu_eff_spread_meV
sigma_K_up_meV, sigma_Kp_up_meV, sigma_K_down_meV, sigma_Kp_down_meV
nu_K_up, nu_Kp_up, nu_K_down, nu_Kp_down

# Main generalized Stoner result
gamma_lambda_su4_diag_selected
qstar_nonzero_lambda_su4_diag_selected
finite_q_ratio_su4_diag
finite_q_wins_su4_diag
qstar_su4_diag_dq1, qstar_su4_diag_dq2
qstar_su4_diag_qx_Ainv, qstar_su4_diag_qy_Ainv, qstar_su4_diag_qnorm_Ainv
qstar_su4_diag_spin_flip_direction
qstar_su4_diag_layer_dipole_overlap
qstar_su4_diag_layer_dipole_ratio

# Optional SU(2) Hund extension
finite_q_ratio_su2_hund_factor2
finite_q_wins_su2_hund_factor2
qstar_su2_hund_factor2_dq1, qstar_su2_hund_factor2_dq2
qstar_su2_hund_factor2_spin_flip_direction
qstar_su2_hund_factor2_layer_dipole_overlap

# Legacy diagnostics
finite_q_ratio_legacy_scalar_total
finite_q_ratio_legacy_offdiag_J
```

---

## 13. CLI updates

Add these CLI options to both single-point and n-D scripts:

```text
--main-vertex-model {su4_diag,su2_hund_factor2}
--also-run-hund-factor2
--hund-transverse-factor 2.0
--occupation-mode {flavor_mu,common_mu}
--spin-flip-mode {plus,minus,both_pm}
--q-mode {folded_grid,unfolded_diagonalize,folded_grid_with_G_shift}
--legacy-diagnostics
```

Defaults:

```text
main_vertex_model      = su4_diag
also_run_hund_factor2  = True
hund_transverse_factor = 2.0
occupation_mode        = flavor_mu
spin_flip_mode         = both_pm
legacy_diagnostics     = True
```

---

## 14. Minimal smoke tests

### 14.1 Single-point smoke test

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 7 --grid-n2 7 \
  --n-cm2 2.0e12 \
  --D-Vnm 0.50 \
  --stoner-u0-meV-A2 7.9e4 \
  --stoner-JH-meV-A2 2.4e4 \
  --chi-kBT-meV 0.05 \
  --energy-window-meV 30 \
  --max-bands-per-k 24 \
  --q-stride 1 \
  --main-vertex-model su4_diag \
  --also-run-hund-factor2 \
  --occupation-mode flavor_mu \
  --spin-flip-mode both_pm \
  --out outputs/chi_single_complete_v3
```

### 14.2 Unfolded check on candidate Q region

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 7 --grid-n2 7 \
  --n-cm2 2.0e12 \
  --D-Vnm 0.50 \
  --q-mode unfolded_diagonalize \
  --max-abs-q-step 2 \
  --main-vertex-model su4_diag \
  --also-run-hund-factor2 \
  --out outputs/chi_single_complete_v3_unfolded
```

### 14.3 n-D halo subset

```bash
PYTHONPATH=. python examples/run_nd_mapping_stoner_susceptibility.py \
  --input-map /path/to/nd_map.csv \
  --out-csv outputs/nd_map_with_complete_finiteQ_stoner.csv \
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
  --main-vertex-model su4_diag \
  --also-run-hund-factor2 \
  --occupation-mode flavor_mu \
  --spin-flip-mode both_pm \
  --resume
```

---

## 15. Required sanity checks before using in a paper

### 15.1 Reproduce occupations

Using `mu_f_kin`, check that the Stoner occupations are reproduced from the flavor-resolved band tables. Output maximum error.

```text
max_abs_nu_reconstruction_error < tolerance
```

### 15.2 Stoner reference consistency

Output:

```text
mu_eff_spread_meV
```

Small spread means the solution is interior. Large spread likely means at least one flavor is at a filling bound; use `flavor_mu` occupation mode.

### 15.3 Q symmetry

Check:

```text
lambda(Q) ≈ lambda(-Q)
```

and, when applicable,

```text
lambda(Q) ≈ lambda(C3 Q).
```

### 15.4 Convergence

Run at least:

```text
k grid: 7x7, 9x9, 11x11
energy window: 20, 30, 50 meV
max bands per k: 12, 24, 40
q mode: folded_grid candidate vs unfolded_diagonalize validation
vertex model: su4_diag vs su2_hund_factor2
```

### 15.5 Interpret ratios more than absolute values

Because the current Stoner functional is phenomenological and may not obey the exact transverse Ward identity, do not overinterpret absolute `lambda = 1`. Use primarily:

```text
finite_q_ratio = max_{Q != 0} lambda(Q) / lambda(Q=0)
```

and the movement of `Q*` along the halo.

---

## 16. How to phrase the result in the paper

Safe wording:

```text
We compute the Stoner-after transverse finite-momentum susceptibility by expanding around the self-consistent q=0 flavor-polarized Stoner state. The leading eigenvalue of the generalized Stoner kernel identifies the softest transverse spin mode. Near the experimentally relevant halo boundary, the maximum shifts from Q=0 to finite Q, indicating that the uniform Stoner state becomes susceptible to a finite-momentum spin-density texture.
```

If the result holds only with the Hund extension:

```text
When the longitudinal Hund term in our phenomenological Stoner functional is promoted to a spin-rotationally invariant transverse vertex, the halo boundary develops an enhanced finite-Q transverse mode. We therefore treat this as a model-dependent indication rather than a standalone proof of a skyrmion ground state.
```

If `su4_diag` already shows finite-Q winning:

```text
The finite-Q softening is already present in the conservative valley-diagonal Stoner vertex and is further enhanced by the optional spin-rotational Hund extension.
```

---

## 17. Notes on references and provenance

1. The existing repository Stoner functional contains the longitudinal four-flavor energy `0.5*u*(nu_total^2 - sum nu_f^2) - J*m_K*m_Kp`. It does not contain a transverse Hund spin-flip term.

2. The factor `2*J` in `su2_hund_factor2` is not copied from existing code. It follows from the algebraic SU(2) extension of the existing longitudinal term using `m = n_up - n_down = 2 S^z` and `S_a · S_b = S_a^z S_b^z + 1/2(S_a^+S_b^- + S_a^-S_b^+)`.

3. Standard multiorbital Kanamori/Hund Hamiltonians include spin-flip Hund terms, but this project should not pretend the current phenomenological Stoner code already implemented them. Treat `su2_hund_factor2` as an optional physically motivated extension and report robustness against `su4_diag`.

---

## 18. Codex checklist

1. Replace old main criterion with matrix generalized Stoner eigenvalues.
2. Add `vertex.py` with `su4_diag`, `su2_hund_factor2`, and legacy models.
3. Change default occupation mode to `flavor_mu`.
4. Compute both `plus` and `minus` spin-flip bubbles.
5. Select the maximum lambda across spin-flip direction for each vertex model.
6. Keep old scalar lambda columns only as legacy diagnostics.
7. Align layer-dipole diagnostics with the selected generalized Stoner eigenmode.
8. Add CLI switches listed above.
9. Add reference validation columns to all outputs.
10. Mark `unfolded_diagonalize` as required for paper-level candidate points.

