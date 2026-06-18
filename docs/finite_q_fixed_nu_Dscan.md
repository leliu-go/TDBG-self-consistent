# Fixed-filling finite-Q D scan

This workflow tests a moire-scale intravalley transverse spin soft mode of the
uniform Stoner-polarized reference state. It does not by itself prove a
nonlinear SDW ground state, an insulating reconstructed spectrum, or an
atomic-scale intervalley SDW.

## Main workflow

Run a fixed-filling displacement-field scan with:

```powershell
python -u examples\run_fixed_nu_Dscan_stoner_susceptibility.py `
  --nu-total 2.0 `
  --D-min -0.8 `
  --D-max 0.8 `
  --D-count 21 `
  --theta-deg 1.35 `
  --cutoff 2 `
  --grid-n1 15 `
  --grid-n2 15 `
  --q-mode folded_grid_with_G_shift `
  --q-stride 1 `
  --occupation-mode equilibrium_common_mu `
  --main-vertex-model su4_diag `
  --selected-D-values -0.4,0.0,0.4 `
  --out outputs\fixed_nu_Dscan `
  --resume
```

Provide exactly one of `--nu-total` and `--n-cm2`. When `--nu-total` is used,
the script converts it to density using the repository moire-cell convention
for the requested twist angle.

Each D point independently runs:

```text
SCF electrostatics -> full-band Stoner -> common-mu reference validation -> finite-Q scan
```

Selected D values also save full q maps in:

```text
out/D_points/D_.../susceptibility_qmap.csv
out/D_points/D_.../state_metadata.json
```

## Paper-level defaults

The paper-level defaults are:

```text
occupation_mode = equilibrium_common_mu
q_mode = folded_grid_with_G_shift
main_vertex_model = su4_diag
first_mbz_only = true
soft transverse channel tracked from the sign of spin polarization
```

The fixed-filling paper workflow intentionally rejects the older diagnostic
reference modes and plain folded q mode. Use older scripts only for reproducing
old exploratory data, not for the fixed-nu paper-level scan.

## Main columns

The main finite-Q fields are:

```text
lambda_gamma_su4_soft
lambda_qstar_su4_soft
finite_q_ratio_su4_soft
finite_q_delta_normalized
finite_q_delta_raw
qstar_qx_mbz_Ainv
qstar_qy_mbz_Ainv
qstar_qnorm_mbz_Ainv
qstar_is_first_nonzero_shell
finite_q_status
```

`finite_q_status` can be:

```text
uniform_Q0
finite_Q_candidate
finite_Q_resolved
numerically_ambiguous
invalid_reference
```

A first-shell maximum is only a candidate, not a resolved SDW mode.

## Mechanism diagnostics

The old same-flavor JDOS is retained as a generic diagnostic. The mechanism
diagnostic for the transverse instability is:

```text
spinflip_nesting_plus
spinflip_nesting_minus
spinflip_nesting_soft
qstar_spinflip_nesting_qx_Ainv
qstar_spinflip_nesting_qy_Ainv
qstar_chi_minus_qstar_nesting_distance_Ainv
```

The robust VHS level currently records DOS at EF and marks saddle extraction as
`vhs_status="unresolved"` unless a later saddle-fit refinement is added.

## Plotting

Generate overview figures with:

```powershell
python -u examples\plot_fixed_nu_Dscan_stoner_susceptibility.py `
  --scan-csv outputs\fixed_nu_Dscan\fixed_nu_Dscan_summary.csv `
  --points-root outputs\fixed_nu_Dscan\D_points `
  --representative-D -0.4,0.0,0.4 `
  --out outputs\fixed_nu_Dscan\figures
```

The plotter saves both PDF and PNG where implemented.
