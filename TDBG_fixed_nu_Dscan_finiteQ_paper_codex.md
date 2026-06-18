# Codex task: audit and upgrade the current TDBG finite-Q calculation for a fixed-filling displacement-field scan

> Repository: `https://github.com/leliu-go/TDBG-self-consistent`  
> Audit target: current `master`, especially `tdbg_scf/susceptibility/` and the susceptibility example scripts.  
> Scientific goal: at a fixed experimental filling \(\nu=\nu_{\rm exp}\), scan displacement field \(D\) through the van-Hove/transport transition and determine whether the uniform \(Q=0\) Stoner-polarized reference state develops a **resolved finite-\(Q\) transverse spin soft mode** consistent with a spiral/helical SDW.

This is an additive, backward-compatible upgrade. Do not remove or change the existing SCF, Stoner, or legacy susceptibility outputs. Add corrected paper-level defaults, diagnostics, a fixed-\(\nu\) \(D\)-scan workflow, tests, and publication plots.

---

## 1. Scope and claim

The current calculation probes the intravalley transverse order parameter

\[
M_{\tau}^{+}(\mathbf Q)
=\sum_{\mathbf k,m,n}
\Lambda^{\tau}_{mn}(\mathbf k,\mathbf Q)
\langle c^{\dagger}_{\tau m,\mathbf k+\mathbf Q,\uparrow}
        c_{\tau n,\mathbf k,\downarrow}\rangle,
\]

and its \(S^-\) counterpart. It therefore tests a **moiré-scale, intravalley spiral/helical SDW instability** of a uniform spin-polarized reference state. It does not test an atomic-scale intervalley SDW at momentum \(K-K'\), and it does not by itself prove that the reconstructed state is insulating.

The paper-level claim supported by this module must be:

> A generalized Stoner linear-stability calculation reveals that the uniform spin-polarized state becomes anomalously soft at a nonzero moiré momentum near the VHS, providing a microscopic instability toward a spiral/helical SDW.

Do not claim that susceptibility alone proves the nonlinear SDW ground state, an SDW gap, or a skyrmion lattice.

---

## 2. Audit of the current implementation

### 2.1 Parts that are conceptually correct and should be retained

1. `stoner_reference.py` evaluates

   \[
   \Sigma_f=\partial E_{\rm int}/\partial\nu_f
   \]

   for the existing four-flavor Stoner energy. The implemented derivatives of the \(u_0\) and longitudinal Hund terms are consistent with the current Stoner functional.

2. Since the existing Stoner term is a flavor-dependent scalar energy shift, using the self-consistent continuum eigenvectors together with spin/flavor-shifted eigenvalues is internally consistent with that model. The Stoner term changes energies and occupations but not layer/sublattice wavefunctions.

3. The code evaluates both circular transverse channels:

   - `plus`: down \(\rightarrow\) up, \(S^+\);
   - `minus`: up \(\rightarrow\) down, \(S^-\).

4. The matrix generalized-Stoner score based on the Hermitian kernel

   \[
   K(\mathbf Q)=\chi^0(\mathbf Q)^{1/2}\,\Gamma\,\chi^0(\mathbf Q)^{1/2}
   \]

   is a valid numerical representation of the nonzero spectrum of \(\Gamma\chi^0\). Keep `su4_diag` as the primary conservative vertex. Keep the factor-2 Hund extension and all legacy scores only as robustness diagnostics.

5. `folded_grid_with_G_shift` and `unfolded_diagonalize` are the correct directions for handling Bloch/plane-wave folding. Retain both.

6. Layer-resolved matrix elements are useful as mode-character diagnostics, but are secondary to the finite-\(Q\) instability itself.

### 2.2 Mandatory corrections before paper-level use

#### A. Use a genuine equilibrium Stoner-after reference for the main result

The current default is:

```python
occupation_mode = "flavor_mu"
```

and the bubble uses separate initial and final flavor chemical potentials. This reproduces the optimized flavor fillings, but it is not a valid equilibrium Kubo reference whenever

\[
\mu_f^{\rm kin}+\Sigma_f
\]

is not common across the flavors connected by the spin-flip operator. It is especially dangerous in the near-degenerate denominator branch, where the current code substitutes a Fermi derivative even when the two flavor occupations are controlled by different chemical potentials. That can generate an artificial large contribution.

For the **main calculation**, use one common electrochemical potential for the post-Stoner quasiparticle Hamiltonian:

\[
E_{f n\mathbf k}^{\rm St}=\epsilon_{\tau n\mathbf k}+\Sigma_f,
\]

\[
f_{f n\mathbf k}=f\!\left(E_{f n\mathbf k}^{\rm St}-\mu_{\rm eq}\right).
\]

Requirements:

1. Add `occupation_mode="equilibrium_common_mu"` and make it the paper/default mode.
2. Solve \(\mu_{\rm eq}\) using the same filling convention and carrier-sector bookkeeping as the existing Stoner tables. Do not manually count the full valence sea with a new convention.
3. Use the same numerical temperature in the Stoner reference and the susceptibility whenever possible. If the susceptibility temperature is used only as a regulator, explicitly record the reconstructed filling mismatch.
4. Reconstruct all four flavor fillings from \(E_f^{\rm St}\) and \(\mu_{\rm eq}\), and save

   ```text
   nu_f_target
   nu_f_reconstructed_common_mu
   max_abs_nu_f_mismatch
   total_nu_mismatch
   mu_eff_spread_meV
   reference_valid
   reference_warning
   ```

5. A point is valid for the main paper result only if the total filling is reproduced and either:
   - the flavor fillings are reproduced within tolerance; or
   - a flavor lies at a rigorously identified empty/full KKT bound and the common-\(\mu\) occupation is nevertheless identical.
6. Keep `flavor_mu` only as a named diagnostic. Never use it for the main figure without showing that `mu_eff_spread_meV` is negligible.

Suggested parameters:

```python
occupation_mode: str = "equilibrium_common_mu"
mu_eff_spread_warn_meV: float = 0.02
mu_eff_spread_fail_meV: float = 0.10
filling_mismatch_warn: float = 2e-3
filling_mismatch_fail: float = 1e-2
```

The numerical values must remain configurable and be tested against mesh/temperature convergence.

#### B. Fix the small-denominator logic for flavor-resolved diagnostic mode

For `equilibrium_common_mu`, the derivative limit is correct:

\[
\frac{f(E_i)-f(E_f)}{E_f-E_i}\rightarrow -f'(E).
\]

For `flavor_mu`, if \(\mu_i\neq\mu_f\), the numerator need not vanish when \(E_i=E_f\), so replacing the ratio by a derivative is mathematically wrong. In diagnostic `flavor_mu` mode:

- either reject/flag pairs with `abs(Ef-Ei)<tol` and `abs(fi-ff)` not small;
- or evaluate a regulated complex denominator and retain the real principal-value part;
- record the number and total weight of regulated non-equilibrium pairs.

Do not silently use the common-\(\mu\) derivative formula.

#### C. Change the paper/default folding mode

Current default:

```python
q_mode = "folded_grid"
```

must not be the paper default because plain folded overlaps can be wrong when \(\mathbf k+\mathbf Q\) crosses the moiré BZ boundary.

Use:

```python
q_mode = "folded_grid_with_G_shift"
```

for broad scans, and use:

```python
q_mode = "unfolded_diagonalize"
```

for the representative \(D\) points and final verification.

Keep plain `folded_grid` only as a legacy/debug mode and label its outputs accordingly.

#### D. Canonicalize the \(Q\) mesh to the first moiré Brillouin zone

The present commensurate mesh spans a centered reciprocal-lattice parallelogram. Add:

1. Wigner-Seitz reduction of each \(\mathbf Q\) by choosing the reciprocal-lattice equivalent vector of minimum norm.
2. A `first_mbz_only=True` option.
3. Deduplication of reciprocal-equivalent points.
4. Stored fields:

   ```text
   qx_raw_Ainv, qy_raw_Ainv
   qx_mbz_Ainv, qy_mbz_Ainv
   q_norm_mbz_Ainv
   reciprocal_shift_m, reciprocal_shift_n
   ```

5. Draw the hexagonal first-MBZ boundary on all paper q maps.

A generic implementation can enumerate \(m,n\in[-2,2]\) and minimize

\[
|\mathbf Q+m\mathbf b_1+n\mathbf b_2|.
\]

#### E. Do not classify every nonzero grid point as a resolved finite-\(Q\) mode

The current summary compares \(\Gamma\) with every point for which `is_gamma=False`, using `finite_q_tol=1e-3`. This is too permissive for a publication claim.

Add the following diagnostics:

```text
q_resolution_Ainv
qstar_shell_index
qstar_is_first_nonzero_shell
qstar_distance_from_gamma_in_grid_units
```

A finite-\(Q\) candidate at the first nonzero q shell is **unresolved**, not automatically an SDW mode. It becomes resolved only if it remains at a nonzero physical \(|Q^*|\) under at least one finer q/k mesh or an unfolded local-q refinement.

Replace a binary `finite_q_wins` main criterion by a status:

```text
uniform_Q0
finite_Q_candidate
finite_Q_resolved
numerically_ambiguous
invalid_reference
```

Keep all old booleans for backward compatibility.

#### F. Track one physical transverse branch instead of taking a pointwise envelope

The current selected score can choose the larger of `plus` and `minus` independently at every \(Q\). That can splice two circular branches and create artificial cusps or a false change of \(Q^*\).

Add a paper-level `soft_transverse_channel`:

```python
m_spin = (nu_K_up + nu_Kp_up) - (nu_K_down + nu_Kp_down)
if m_spin > spin_channel_tol:
    soft_channel = "minus"  # up -> down, S-
elif m_spin < -spin_channel_tol:
    soft_channel = "plus"   # down -> up, S+
else:
    soft_channel = "degenerate"
```

Then verify that this channel is the one whose \(Q=0\) Stoner score is closest to the expected Goldstone-normalized value. Save both branch-resolved data and the old envelope, but use the continuously tracked soft branch for the main D scan.

Required output columns:

```text
lambda_su4_plus_Q
lambda_su4_minus_Q
lambda_su4_soft_Q
soft_transverse_channel
channel_switch_warning
```

Near an unpolarized point, show plus/minus separately rather than forcing a branch.

#### G. Replace the generic same-flavor JDOS by a spin-flip nesting diagnostic for the mechanism plot

The present JDOS is a same-flavor Fermi-surface autocorrelation. It is not the phase space appearing in the transverse spin bubble.

Add the form-factor-weighted spin-flip nesting function

\[
\mathcal N^{+-}(\mathbf Q)=A_M\sum_{\tau,\mathbf k,m,n}w_{\mathbf k}
|\Lambda^{\tau}_{mn}(\mathbf k,\mathbf Q)|^2
\delta_\eta(E_{\tau\downarrow n\mathbf k}-\mu_{\rm eq})
\delta_\eta(E_{\tau\uparrow m,\mathbf k+\mathbf Q}-\mu_{\rm eq}),
\]

and analogously \(\mathcal N^{-+}\). Use the same soft circular channel selected above.

Output:

```text
spinflip_nesting_plus
spinflip_nesting_minus
spinflip_nesting_soft
qstar_spinflip_nesting_x/y/norm
qstar_chi_minus_qstar_nesting_distance_Ainv
```

Keep the old JDOS only as a generic diagnostic.

#### H. Correct the automatically generated single-point q map

The current example script colors `susceptibility_qmap.png` with legacy `lambda_u_plus_hund`. Change the default plotting key to the paper-level soft-channel SU4 result:

```python
plot_key = "lambda_su4_diag_soft"
```

or the closest implemented column name. The command-line option `--plot-key` should allow all models/channels. The title and colorbar must explicitly identify the chosen vertex and circular channel.

#### I. Do not use `lambda > 1` alone to draw the SDW region

The current longitudinal filling-only Stoner functional is not a fully SU(2)-covariant HF functional, so the transverse \(Q=0\) Ward/Goldstone identity is not guaranteed numerically. For the main result, calibrate each D point by the same soft branch at \(\Gamma\):

\[
\widetilde\lambda(\mathbf Q;D)
=\frac{\lambda_{\rm soft}(\mathbf Q;D)}
       {\lambda_{\rm soft}(\Gamma;D)},
\]

\[
\Delta_{\rm fQ}(D)
=\max_{\mathbf Q\neq 0}\widetilde\lambda(\mathbf Q;D)-1.
\]

Also save the raw difference

\[
\Delta\lambda_{\rm raw}(D)=
\lambda_{\rm soft}(\mathbf Q^*;D)-\lambda_{\rm soft}(\Gamma;D).
\]

Use \(\Delta_{\rm fQ}\) as the main finite-\(Q\) tendency. Show raw \(\lambda\) values in the Supplement. Do not label a phase solely from `qstar > 1`.

#### J. Estimate numerical uncertainty instead of using `finite_q_tol=1e-3`

For each paper-critical D point, estimate

```text
error_kmesh
error_qmesh
error_band_window
error_fold_method
error_symmetry
finite_q_numerical_error
```

with

```text
finite_q_numerical_error = max(all available errors)
```

Classify a resolved finite-Q mode only when

\[
\Delta_{\rm fQ} > \max(\Delta_{\rm min}, 2\,\sigma_{\rm num}),
\]

where a preliminary configurable value such as `delta_min=0.02` may be used for screening, but the final classification must be based on measured convergence errors.

---

## 3. Paper-level physical quantities

For each fixed-\(\nu\), fixed-\(D\) Stoner-after state calculate:

### 3.1 Main SU4-diagonal generalized Stoner score

For each circular channel \(c\in\{+,-\}\), form

\[
\chi_c^0(\mathbf Q)=
\begin{pmatrix}
\chi_{K,c}^0(\mathbf Q)&0\\
0&\chi_{K',c}^0(\mathbf Q)
\end{pmatrix},
\qquad
\Gamma_{\rm SU4}=u_{\rm cell}I_2,
\]

and

\[
\lambda_c(\mathbf Q)
=\lambda_{\max}\!\left[
\chi_c^{0\,1/2}(\mathbf Q)
\Gamma_{\rm SU4}
\chi_c^{0\,1/2}(\mathbf Q)
\right].
\]

Use the physically tracked soft circular channel for the main result.

### 3.2 Main finite-Q metrics

At every D:

\[
\lambda_\Gamma(D)=\lambda_{\rm soft}(\Gamma;D),
\]

\[
\lambda_{\rm fQ}(D)=
\max_{\mathbf Q\neq0}\lambda_{\rm soft}(\mathbf Q;D),
\]

\[
R_{\rm fQ}(D)=\frac{\lambda_{\rm fQ}(D)}{\lambda_\Gamma(D)},
\]

\[
\Delta_{\rm fQ}(D)=R_{\rm fQ}(D)-1,
\]

\[
\mathbf Q^*(D)=\arg\max_{\mathbf Q\neq0}
\lambda_{\rm soft}(\mathbf Q;D),
\]

\[
L_{\rm SDW}(D)=\frac{2\pi}{|\mathbf Q^*(D)|}.
\]

Store \(L_{\rm SDW}\) in nm and suppress it when \(Q^*\) is unresolved.

### 3.3 VHS diagnostics

Add `tdbg_scf/susceptibility/vhs.py` with two levels of diagnostics.

#### Required robust level

At every D save:

```text
DOS_total_EF
DOS_active_flavor_EF
D_at_max_DOS_on_scan
```

Use the same broadening and band window across the D scan.

#### Preferred saddle-point level

For each active Stoner-after band close to \(\mu_{\rm eq}\):

1. Use the periodic uniform k mesh.
2. Fit a local 2D quadratic surface over a 3x3 or 5x5 neighborhood in fractional reciprocal coordinates.
3. Identify a saddle when the fitted Hessian determinant is negative and the fitted gradient is small.
4. Select the saddle closest to \(\mu_{\rm eq}\).
5. Save:

   ```text
   vhs_status
   vhs_flavor
   vhs_band_index
   vhs_kx_Ainv, vhs_ky_Ainv
   E_vhs_meV
   delta_E_vhs_meV = mu_eq_meV - E_vhs_meV
   vhs_fit_residual_meV
   ```

Do not force a saddle result when the mesh/fit is unreliable. In that case use `vhs_status="unresolved"` and retain the DOS diagnostic.

### 3.4 Optional layer character

For the selected soft finite-Q eigenmode output:

```text
qstar_layer_dipole_overlap
qstar_layer_dipole_ratio
qstar_chi_s0
qstar_chi_sD
```

This is supporting evidence for a layer-dipolar SDW mode, not the criterion defining the SDW candidate.

---

## 4. New fixed-filling D-scan workflow

Add:

```text
examples/run_fixed_nu_Dscan_stoner_susceptibility.py
examples/plot_fixed_nu_Dscan_stoner_susceptibility.py
```

and reusable functions in:

```text
tdbg_scf/susceptibility/dscan.py
```

### 4.1 CLI

The run script must support:

```text
--nu-total
--D-min
--D-max
--D-count
--D-values                 # optional comma-separated override
--theta-deg
--D-sign
--scf-source
--cutoff
--grid-n1 --grid-n2
--q-mode
--q-stride
--energy-window-meV
--max-bands-per-k
--chi-kBT-meV
--stoner-temperature-K
--main-vertex-model
--occupation-mode
--out
--resume
--selected-D-values        # save detailed q maps for these points
```

Require exactly one of `--nu-total` and `--n-cm2`. If `--nu-total` is supplied, convert to density using the repository's existing moiré-cell-area convention.

### 4.2 Per-D workflow

For every D, independently run:

```text
self-consistent continuum electrostatics at fixed nu
    -> q=0 Stoner minimization
    -> equilibrium Stoner-after reference validation
    -> full finite-Q transverse scan
    -> Q* refinement/validation
    -> VHS and spin-flip nesting diagnostics
```

Do not reuse a band structure or a Stoner splitting from another D. Reusing the previous converged layer potential as an optional SCF initial guess is allowed, but the final solution must satisfy the original convergence criterion.

The D scan must run in both ascending and descending order for a small subset around the transition. Save any hysteresis or branch disagreement rather than silently averaging it.

### 4.3 Two-pass strategy

#### Pass 1: screening

Recommended initial settings:

```text
cutoff = 1 or 2
k mesh = 11x11 or 15x15
q_mode = folded_grid_with_G_shift
q_stride = 1
main_vertex_model = su4_diag
occupation_mode = equilibrium_common_mu
spin_flip_mode = both_pm
```

Use a moderate D spacing to locate:

```text
D_SP      # rapid change of q=0 spin polarization
D_VHS     # DOS maximum or delta_E_vhs crossing
D_fQ      # finite-Q tendency becomes positive
```

#### Pass 2: paper refinement

For 5-9 D values centered on the transition:

```text
cutoff >= 2
k/q mesh convergence: at least two meshes
q_mode = unfolded_diagonalize for representative points
energy-window convergence
local D spacing finer than the screening scan
```

If \(Q^*\) sits on the first nonzero shell, increase the mesh or add an unfolded local-q patch around the coarse maximum.

### 4.4 Output layout

```text
out/
  fixed_nu_Dscan_summary.csv
  fixed_nu_Dscan_metadata.json
  convergence_summary.csv
  figures/
  D_points/
    D_m0p5000/
      state_metadata.json
      susceptibility_qmap.csv
      susceptibility_qmap.npz
      qmap_soft_su4.pdf
      spinflip_nesting_qmap.pdf
      fermi_surface_qstar_overlay.pdf
    ...
```

Required summary columns:

```text
nu_total
n_cm2
D_Vnm
D_sign
scf_converged
scf_residual_meV
stoner_converged
spin_polarization
spin_polarization_norm
valley_polarization
nu_K_up, nu_Kp_up, nu_K_down, nu_Kp_down
sigma_K_up_meV, sigma_Kp_up_meV, sigma_K_down_meV, sigma_Kp_down_meV
mu_eq_meV
mu_eff_spread_meV
max_abs_nu_f_mismatch
total_nu_mismatch
reference_valid
soft_transverse_channel
lambda_gamma_su4_soft
lambda_qstar_su4_soft
finite_q_ratio_su4_soft
finite_q_delta_normalized
finite_q_delta_raw
qstar_qx_mbz_Ainv
qstar_qy_mbz_Ainv
qstar_qnorm_mbz_Ainv
qstar_shell_index
qstar_is_first_nonzero_shell
L_SDW_nm
finite_q_status
finite_q_numerical_error
DOS_total_EF
DOS_active_flavor_EF
E_vhs_meV
delta_E_vhs_meV
vhs_status
spinflip_nesting_at_qstar
qstar_nesting_qnorm_Ainv
qstar_chi_minus_qstar_nesting_distance_Ainv
qstar_layer_dipole_overlap
qstar_layer_dipole_ratio
q_minus_symmetry_error
C3_symmetry_error
folded_unfolded_error
```

Keep existing/legacy output columns too.

---

## 5. Figures required for the paper

The central calculation figure should be a **fixed-\(\nu\), varying-\(D\)** multi-panel figure. It should visually demonstrate that a nonzero-\(Q\) transverse mode appears only near/after the experimental critical D and near the VHS.

### Figure 1: fixed-filling D-scan overview

Create `fig_fixed_nu_Dscan_overview.pdf` with aligned x axes.

#### Panel (a): q=0 reference and VHS

Plot versus D:

- normalized spin polarization of the q=0 Stoner state;
- active-flavor DOS at the Fermi level;
- optionally `delta_E_vhs_meV` on a secondary axis or separate subpanel.

Mark:

```text
D_SP   = q=0 spin-polarization boundary/crossover
D_VHS  = closest approach to the VHS
D_fQ   = onset of a resolved finite-Q mode
```

Use symbols rather than assuming these three fields are identical.

#### Panel (b): main finite-Q criterion

Plot versus D:

\[
R_{\rm fQ}(D)=\lambda_{\rm fQ}/\lambda_\Gamma
\]

and/or

\[
\Delta_{\rm fQ}(D)=R_{\rm fQ}-1.
\]

Include a horizontal zero line and a numerical-uncertainty band.

Shading convention:

- blue: `uniform_Q0`;
- green: `finite_Q_resolved`;
- gray: `finite_Q_candidate` or `numerically_ambiguous`;
- hatched/red warning: invalid Stoner-after reference.

Do not use `lambda > 1` alone for the shading.

#### Panel (c): ordering wavevector

Plot versus D:

- \(|Q^*|\) in \(\mathrm{\AA}^{-1}\) or normalized by a moiré reciprocal scale;
- \(L_{\rm SDW}=2\pi/|Q^*|\) in nm on a secondary axis.

Only draw a solid value for `finite_Q_resolved`. Use open symbols for unresolved first-shell candidates.

#### Panel (d), optional but recommended: layer character

Plot the selected mode's layer-dipole overlap or ratio versus D and align it with the finite-Q onset.

### Figure 2: representative momentum-space evolution through the transition

Create `fig_representative_Q_maps.pdf`. Choose at least three D values from the actual calculated scan:

```text
D1: clearly inside the uniform spin-polarized metal
D2: near the VHS/critical point
D3: inside the resolved finite-Q candidate region
D4: optional control beyond the region
```

For each D plot the same quantity and same normalization:

\[
\delta\widetilde\lambda(\mathbf Q;D)
=\frac{\lambda_{\rm soft}(\mathbf Q;D)}
       {\lambda_{\rm soft}(\Gamma;D)}-1.
\]

Requirements:

- show the first moiré BZ boundary;
- mark \(\Gamma\), \(\mathbf Q^*\), and \(-\mathbf Q^*\);
- use a common color scale centered at zero;
- report D, spin polarization, \(\delta E_{\rm VHS}\), \(|Q^*|\), and soft circular channel in each title;
- show raw and symmetry-averaged maps in the Supplement if symmetrization is used.

Expected visual narrative:

```text
D1: maximum at Gamma
D2: broad finite-Q ridge/incipient peaks near VHS
D3: clear nonzero-Q pair or C3-related star of maxima
```

A ring that collapses toward Gamma with mesh refinement is not sufficient.

### Figure 3: q-line cuts

Create `fig_Q_linecuts_through_transition.pdf`.

For the same representative D values, plot along the line from \(\Gamma\) through \(Q^*\):

\[
\delta\widetilde\lambda(q)=
\widetilde\lambda(q)-1.
\]

This makes the movement of the quadratic maximum away from \(\Gamma\) unambiguous. Add error bars or shaded convergence differences where available.

For a small-Q mode, optionally fit

\[
1-\widetilde\lambda(q)=a_2q^2+a_4q^4
\]

near \(\Gamma\). A sign change of \(a_2\) is a useful spin-stiffness diagnostic, but only report it if the fit is stable over multiple meshes and fit windows.

### Figure 4: microscopic VHS/nesting mechanism

Create `fig_VHS_spinflip_nesting.pdf` for D2 or D3.

Panels:

1. Stoner-after majority and minority Fermi contours for the active valley/flavor.
2. The final-spin contour shifted by \(-Q^*\), overlaid on the initial-spin contour.
3. Mark resolved saddle/VHS points and draw the \(Q^*\) connecting vectors.
4. Plot the form-factor-weighted spin-flip nesting map \(\mathcal N_{\rm soft}(Q)\).
5. Mark both \(Q^*_{\chi}\) and \(Q^*_{\rm nesting}\).

This figure is the mechanism evidence that the finite-Q susceptibility is enhanced by VHS-related particle-hole phase space rather than by an arbitrary numerical maximum.

### Optional Figure 5: local \(\nu-D\) context map

After the fixed-\(\nu\) result is established, optionally calculate a narrow \(\nu-D\) window around the experimental line and plot

\[
\Delta_{\rm fQ}(\nu,D).
\]

Overlay the fixed-\(\nu\) cut and q=0 Stoner boundary. This is supplementary context, not a replacement for the fixed-\(\nu\) D scan.

---

## 6. Plotting implementation requirements

Add publication plotting functions to `tdbg_scf/susceptibility/plotting.py` or a new `paper_plotting.py`:

```python
plot_fixed_nu_Dscan_overview(...)
plot_representative_qmaps(...)
plot_q_linecuts(...)
plot_vhs_spinflip_nesting(...)
plot_convergence_summary(...)
```

Requirements:

- save both PDF and PNG;
- use physical \(D\) sign and label the repository `D_sign` convention in metadata;
- no hard-coded experimental critical D;
- select representative D values from the computed statuses or explicit CLI input;
- never plot legacy `lambda_u_plus_hund` unless explicitly requested;
- default to SU4-diagonal, tracked-soft-channel, common-\(\mu\) data;
- include a small footer or metadata JSON with vertex, occupation mode, q mode, k mesh, q mesh, cutoff, temperature, band window, and git commit hash when available.

---

## 7. Numerical validation and tests

Extend `tests/test_susceptibility.py` and add integration tests.

### 7.1 Required unit tests

1. **Equilibrium common-\(\mu\) reconstruction**
   - Construct toy spin-split bands with a known common Fermi level.
   - Verify total and flavor fillings.

2. **Flavor-\(\mu\) small-denominator protection**
   - Use \(E_i=E_f\), \(\mu_i\neq\mu_f\).
   - Verify the code does not use the equilibrium derivative limit silently.

3. **Soft-channel selection**
   - Positive magnetization selects `minus`.
   - Negative magnetization selects `plus`.
   - Near-zero magnetization returns `degenerate`.

4. **First-MBZ reduction and deduplication**
   - Reciprocal-equivalent Q points map to the same canonical vector.

5. **Spin-flip nesting**
   - A toy pair of shifted initial/final Fermi surfaces peaks at the known Q.

6. **Summary status**
   - A first-shell maximum is `finite_Q_candidate` until refinement confirms it.

### 7.2 Required physical regression checks

Add a lightweight `tests/test_susceptibility_physics.py` or a documented validation script for:

1. \(\chi(Q)\approx\chi(-Q)\) at zero external field.
2. Approximate C3 symmetry when the model settings preserve C3.
3. `folded_grid_with_G_shift` agrees with `unfolded_diagonalize` at selected Q points.
4. \(Q^*\) and \(\Delta_{\rm fQ}\) converge with k/q mesh.
5. Results converge with energy window and band count.
6. A simple isotropic parabolic toy band does not falsely produce a resolved finite-Q maximum due only to grid discretization.

Do not make the full expensive convergence suite part of the default fast unit-test run; provide a marker such as `pytest -m slow`.

---

## 8. Acceptance criteria for a paper-level finite-Q result

A D point may be labeled `finite_Q_resolved` only when all applicable conditions hold:

1. SCF and q=0 Stoner calculations converged.
2. Equilibrium common-\(\mu\) reference is valid.
3. The same physical circular branch is tracked from \(\Gamma\) to \(Q^*\).
4. \(\Delta_{\rm fQ}\) exceeds the measured numerical uncertainty.
5. \(Q^*\) is stable under a finer q/k mesh or local unfolded refinement.
6. \(\chi(Q)\approx\chi(-Q)\); C3-related peaks appear when symmetry requires them.
7. `folded_grid_with_G_shift` and `unfolded_diagonalize` agree within tolerance at representative points.
8. The result is stable to reasonable temperature/broadening and band-window changes.
9. The finite-Q onset occurs near a reproducible DOS/VHS feature if the manuscript attributes it to the VHS.
10. The spin-flip nesting function supports the same \(Q^*\) or explains any difference through form factors/energy denominators.

If one or more conditions fail, save the result but mark it as candidate/ambiguous rather than discarding it.

---

## 9. Suggested commands

### 9.1 Coarse fixed-\(\nu\) scan

```bash
PYTHONPATH=. python examples/run_fixed_nu_Dscan_stoner_susceptibility.py \
  --nu-total <EXPERIMENTAL_NU> \
  --D-min <D_MIN> --D-max <D_MAX> --D-count 31 \
  --theta-deg 1.35 \
  --cutoff 1 \
  --grid-n1 11 --grid-n2 11 \
  --q-mode folded_grid_with_G_shift \
  --q-stride 1 \
  --occupation-mode equilibrium_common_mu \
  --main-vertex-model su4_diag \
  --chi-kBT-meV 0.05 \
  --energy-window-meV 30 \
  --max-bands-per-k 24 \
  --out outputs/fixed_nu_Dscan_coarse \
  --resume
```

### 9.2 Refined transition scan

```bash
PYTHONPATH=. python examples/run_fixed_nu_Dscan_stoner_susceptibility.py \
  --nu-total <EXPERIMENTAL_NU> \
  --D-values <D1,D2,D3,D4,D5,D6,D7> \
  --theta-deg 1.35 \
  --cutoff 2 \
  --grid-n1 15 --grid-n2 15 \
  --q-mode folded_grid_with_G_shift \
  --occupation-mode equilibrium_common_mu \
  --main-vertex-model su4_diag \
  --selected-D-values <D_BEFORE,D_CRITICAL,D_AFTER> \
  --out outputs/fixed_nu_Dscan_refined \
  --resume
```

### 9.3 Unfolded verification at representative D

Reuse or extend the single-point script:

```bash
PYTHONPATH=. python examples/run_stoner_susceptibility_single.py \
  --nu-total <EXPERIMENTAL_NU> \
  --D-Vnm <D_SELECTED> \
  --cutoff 2 \
  --grid-n1 15 --grid-n2 15 \
  --q-mode unfolded_diagonalize \
  --occupation-mode equilibrium_common_mu \
  --main-vertex-model su4_diag \
  --plot-key lambda_su4_diag_soft \
  --out outputs/unfolded_D_selected
```

### 9.4 Generate paper figures

```bash
PYTHONPATH=. python examples/plot_fixed_nu_Dscan_stoner_susceptibility.py \
  --scan-csv outputs/fixed_nu_Dscan_refined/fixed_nu_Dscan_summary.csv \
  --points-root outputs/fixed_nu_Dscan_refined/D_points \
  --representative-D <D_BEFORE,D_NEAR_VHS,D_FINITE_Q> \
  --out outputs/fixed_nu_Dscan_refined/figures
```

---

## 10. Manuscript-safe interpretation

If the calculation succeeds, the strongest defensible statement is:

> At fixed filling, the generalized transverse Stoner kernel of the self-consistent, uniformly spin-polarized reference state evolves from a maximum at \(Q=0\) to a resolved maximum at \(Q^*\neq0\) as the displacement field tunes the Fermi level through the van Hove region. The same \(Q^*\) connects high-weight initial- and final-spin Fermi-surface patches, supporting a VHS-assisted spiral-SDW instability.

Do not write:

> The susceptibility calculation proves an insulating SDW ground state.

To establish the insulating character, a later calculation must add a finite-\(Q\) order parameter and show self-consistent band reconstruction/gap opening at selected D points.

---

## 11. Codex completion checklist

- [ ] Preserve all existing SCF/Stoner APIs and legacy columns.
- [ ] Add equilibrium common-\(\mu\) post-Stoner reference and validity checks.
- [ ] Fix the non-equilibrium small-denominator branch.
- [ ] Make `folded_grid_with_G_shift` the broad-scan default.
- [ ] Add first-MBZ Q reduction/deduplication.
- [ ] Track a single physical soft circular channel across Q and D.
- [ ] Add form-factor-weighted spin-flip nesting.
- [ ] Replace the example's legacy q-map color field by the paper-level SU4 soft-channel score.
- [ ] Add fixed-\(\nu\), varying-D scan and resume support.
- [ ] Add VHS diagnostics.
- [ ] Add robust finite-Q status and numerical uncertainty.
- [ ] Generate the four requested publication figures.
- [ ] Add unit and slow physical-regression tests.
- [ ] Write a short `docs/finite_q_fixed_nu_Dscan.md` explaining CLI, outputs, and claim limitations.
