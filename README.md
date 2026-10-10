# TDBG self-consistent continuum and Stoner workflows

This repository contains a Python implementation of an ABBA twisted double
bilayer graphene (TDBG) continuum model, full continuum self-consistent layer-Hartree and a phenomenological full-band Stoner
post-processing workflow.

The most reliable parts of the code are:

- the ABBA continuum Hamiltonian and band-structure utilities;
- full continuum self-consistent layer-Hartree calculation;
- full-band four-flavor Stoner calculation after self-consistency;
- n-D mapping and single-point output generation for the Stoner workflow.

The finite-Q susceptibility code is currently experimental. It is kept in the
repository for development, but it should not be used as a final physical result
until the finite-Q formulation and diagnostics have been rechecked.

[Current changes and preserved parameters](docs/tdbg_parity_update.md).

Windows deployment and full Hartree commands: [server_full_hartree_stoner_cli.md](server_full_hartree_stoner_cli.md). Numerical audit: [docs/full_hartree_numerical_audit.md](docs/full_hartree_numerical_audit.md).

The Stoner search uses signed per-flavor fillings with default bounds
`-2 <= nu_f <= 2` and conserved total filling. Electron and hole flavors can
coexist, including at neutrality. Physical Stoner DOS still requires a common
interacting chemical potential; artificially clipped states are flagged.

The full-band n-D map driver saves each completed point before worker return,
updates missing-aware scalar maps as points finish and supports
`--resume --out <original_run_directory>`. It displays elapsed time and ETA.
Windows setup/launch scripts are in `scripts/`; default results go to
`D:\LeLiu\data\tdbg_self_consistent`. See the deployment document for commands.

## Code layout

```text
tdbg_scf/continuum.py              ABBA TDBG continuum Hamiltonian
tdbg_scf/lattice.py                moire reciprocal lattice and k grids
tdbg_scf/electrostatics.py         displacement-field convention and Gauss-law update
tdbg_scf/density.py                Fermi functions, density, layer weights, DOS
tdbg_scf/solver_full.py            full continuum self-consistent Hartree solver
tdbg_scf/stoner/                  phenomenological full-band Stoner model
tdbg_scf/susceptibility/          experimental finite-Q susceptibility code
examples/                         runnable scripts for SCF, Stoner, and mappings
docs/full_hartree_numerical_audit.md
                                  numerical conventions, checks, and changes
```

## Physical model

### Continuum Hamiltonian

The single-particle Hamiltonian follows the standard continuum-model philosophy
for graphene moire systems, starting from the Bistritzer-MacDonald type
continuum model for twisted bilayers and using the TDBG AB-BA/ABBA conventions
discussed in the TDBG literature [1-3].

The legacy model identifier and result-folder prefix remain ABBA. With
AB = B_upper-A_lower and BA = A_upper-B_lower, the actual strong dimers below
are BA-AB from top to bottom. This clarifies the label; the Hamiltonian matrix
and every physical parameter are unchanged. The layer order used everywhere is

```text
L1, L2, L3, L4 = top to bottom.
```

For each plane-wave momentum `G`, the upper bilayer basis is

```text
(A1, B1, A2, B2),
```

and the lower bilayer basis is

```text
(A3, B3, A4, B4).
```

The full basis places all upper-bilayer plane-wave states first, followed by all
lower-bilayer plane-wave states. With plane-wave cutoff `N`,

$$
N_G=(2N+1)^2,
\qquad
\dim H=8N_G .
$$

The ABBA convention appears in the intrabilayer dimer couplings:

```text
upper bilayer: A1 <-> B2
lower bilayer: B3 <-> A4
```

The four layer potentials are

$$
U=(U_1,U_2,U_3,U_4),
\qquad
\sum_l U_l=0 .
$$

They are always reported in this zero-average gauge.

### Displacement-field convention

The layer coordinate is positive from top to bottom; electron potential energy
is U=-e*phi. Input D is external displacement divided by epsilon0, in V/nm.
Before screening, the potential step is

```text
U[l+1]-U[l] = D_sign * 1000 * d_layer_nm * D / eps_perp  (meV).
```

The original D_sign=-1 is retained: D>0 points bottom to top and U1>U2>U3>U4.
D_sign=+1 makes positive D point top to bottom. Potentials have zero mean.
Actual d_layer_nm, eps_perp and D_sign are saved in config_used.json.
The standalone Hartree entry retains eps_perp=4; Stoner/Windows retains
its original eps_perp=3. Pass an explicit value when comparing these entries.

## Self-consistent Hartree calculation

The self-consistent part of the code solves only a layer-resolved Hartree
electrostatic problem. It does not include moire-periodic Hartree modulation and
does not include exchange.

The idea is related to self-consistent Hartree and electrostatic treatments of
moire graphene and rhombohedral graphite systems, where screening can strongly
reshape flat or surface bands [4-7].

This repository uses a simpler four-layer capacitor/Gauss-law update:

1. Choose $U=(U_1,U_2,U_3,U_4)$.
2. Diagonalize the continuum Hamiltonian $H(k;U)$.
3. Find $\mu$ such that the target density $n$ is reached.
4. Compute layer densities $n_l$ from eigenvector layer weights.
5. Update $U_l$ from the vertical Gauss-law electrostatics.
6. Subtract the average of $U_l$.
7. Mix old and new $U_l$ with linear or Anderson mixing.
8. Iterate until

$$
\max_l|U_{\mathrm{new},l}-U_l|<\mathrm{tolerance}.
$$

The full density convention is neutrality referenced:

$$
n=g\sum_{k,b}w_k\left[f(E_{k,b}-\mu)-\frac{1}{2}\right].
$$

Layer densities are computed as

$$
n_l=g\sum_{k,b}w_k
\left[f(E_{k,b}-\mu)-\frac{1}{2}\right]
\langle u_{k,b}|P_l|u_{k,b}\rangle .
$$

Here `g=4` is normally spin times valley degeneracy for the SCF step.

### Full continuum SCF

`tdbg_scf/solver_full.py` diagonalizes the full continuum Hamiltonian at every
iteration. This is the most direct self-consistent calculation, but it is more
expensive.

Quick test:

```powershell
python examples\run_full_scf.py `
  --theta-deg 1.35 `
  --cutoff 1 `
  --D-Vnm 0.2 `
  --n-cm2 0 `
  --grid-n1 5 `
  --grid-n2 5 `
  --max-iter 30 `
  --tol-meV 1e-3 `
  --out outputs\full_quick
```

### Full-only numerical workflow

All SCF and Stoner entry points now use full continuum diagonalization at every
Hartree iteration. The projected Hartree solver, projected-HF scaffolding and
projection CLI options have been removed. A full calculation includes all
`8(2N+1)^2` eigenstates of the chosen plane-wave cutoff, including their layer
weights and neutrality reference.

The returned potential must pass the full fixed-point residual, density and
finite-value checks before DOS, bands or Stoner analysis. A failed map point is
recorded as failed. Its diagnostic residual and charge error are retained;
physical outputs are skipped. Full convergence does not by itself establish
convergence in plane-wave cutoff or k-mesh resolution.

## Full-band Stoner model

The Stoner model is a phenomenological post-processing calculation after the
self-consistent Hartree potential has been found. It does not feed back into the
Hartree self-consistency. The phenomenological itinerant-ferromagnet picture is
motivated by the Young-lab R3G and R2G works [8,9]. When comparing to
hBN-aligned rhombohedral graphene devices, the hBN alignment orientation should
be treated as an additional control of the moire perturbation strength [10].

The four flavors are ordered as

```text
(K_up, Kp_up, K_down, Kp_down).
```

For each `(n,D)` point:

```text
1. Run full continuum self-consistent Hartree to obtain U_scf.
2. Diagonalize the full continuum Hamiltonian at U_scf in K and K'.
3. Build neutrality-referenced full-band filling tables for K and K'.
4. Intersect signed flavor bounds [-2,2] with full-spectrum capacity.
5. Run two independent 3D DE searches, occupation-node refinement, common-mu
   and stability checks; compare accepted energies at fixed nu_total.
6. Save flavor fillings, polarizations, DOS, layer DOS, band structure, and contours.
```

The Stoner filling variables are

$$
\nu_f=(\nu_{K\uparrow},\nu_{K'\uparrow},
\nu_{K\downarrow},\nu_{K'\downarrow}),
\qquad
\sum_f\nu_f=\nu_{\mathrm{total}} .
$$

The energy functional is

$$
E[\nu_f]
=\sum_f K_f(\nu_f)
+\frac{1}{2}u_{\mathrm{cell}}
\left[\left(\sum_f\nu_f\right)^2-\sum_f\nu_f^2\right]
-J_{\mathrm{cell}}m_Km_{K'} .
$$

where

$$
m_K=\nu_{K\uparrow}-\nu_{K\downarrow},
\qquad
m_{K'}=\nu_{K'\uparrow}-\nu_{K'\downarrow},
$$

$$
u_{\mathrm{cell}}=\frac{U_0}{A_M},
\qquad
J_{\mathrm{cell}}=\frac{J_H}{A_M}.
$$

The typical parameters used in the current calculations are

$$
U_0=7.9\times10^4\ \mathrm{meV\,A^2},
\qquad
J_H=2.4\times10^4\ \mathrm{meV\,A^2}.
$$

The solver uses two independent 3D differential-evolution runs with seeds
seed and seed+1, followed by exact occupation-node refinement. Feasible
piecewise-quadratic segments include their endpoints. States must pass common
Fermi-level stationarity and local stability checks. Rejected lower-energy
candidates are retained in diagnostics. This stochastic search does not certify
a mathematical global minimum. Legacy random-seed counts are accepted for old
configurations but ignored by DE.

Single-point full-band Stoner run:

```powershell
python examples\run_full_band_stoner.py `
  --n-cm2 1.8e12 `
  --D-Vnm -0.4 `
  --theta-deg 1.35 `
  --cutoff 2 `
  --grid-n1 24 `
  --grid-n2 24 `
  --full-max-iter 60 `
  --full-tol-meV 1e-4 `
  --u0-meV-A2 7.9e4 `
  --JH-meV-A2 2.4e4 `
  --out outputs\full_band_stoner_single
```

n-D map:

```powershell
python examples\run_nd_mapping_full_band_stoner.py `
  --theta-deg 1.35 `
  --cutoff 2 `
  --n-min-cm2=-6e12 `
  --n-max-cm2 6e12 `
  --n-count 61 `
  --D-min-Vnm=-1.2 `
  --D-max-Vnm 1.2 `
  --D-count 121 `
  --grid-n1 24 `
  --grid-n2 24 `
  --full-max-iter 60 `
  --full-tol-meV 1e-4 `
  --u0-meV-A2 7.9e4 `
  --JH-meV-A2 2.4e4 `
  --max-workers 40 `
  --out outputs\nd_mapping_full_band_stoner
```

## Polarization definitions

The Stoner polarizations stored by the code are absolute values:

$$
P_{\mathrm{spin}}
=\left|(\nu_{K\uparrow}+\nu_{K'\uparrow})
-(\nu_{K\downarrow}+\nu_{K'\downarrow})\right|,
$$

$$
P_{\mathrm{valley}}
=\left|(\nu_{K\uparrow}+\nu_{K\downarrow})
-(\nu_{K'\uparrow}+\nu_{K'\downarrow})\right|,
$$

$$
P_{\mathrm{spin-valley}}
=\left|(\nu_{K\uparrow}+\nu_{K'\downarrow})
-(\nu_{K'\uparrow}+\nu_{K\downarrow})\right|.
$$

Legacy *_polarization_norm columns divide by |nu_total| and may exceed one
for compensated states. Default active-band maps instead use
*_polarization_active_norm, with P_a=abs(sum_f q_a,f*(e_f-h_f))/sum_f(e_f+h_f).
Here e_f and h_f count central conduction electrons and central valence vacancies
after filling the full spectrum; hole labels are those of the missing electrons.
The generators in flavor order are (+,+,-,-), (+,-,+,-), (+,-,-,+) for spin,
valley and spin-valley. Values lie in [0,1]; no active carriers gives NaN.
Remote filling is recorded separately. This observable selection does not
project the Hamiltonian or the Stoner kinetic energy.

Layer-density polarizations use `L1,L2,L3,L4 = top to bottom`:

$$
P_{\mathrm{top-bottom}}=(n_1+n_2)-(n_3+n_4),
$$

$$
P_{\mathrm{outer-inner}}=(n_1+n_4)-(n_2+n_3),
$$

$$
P_{\mathrm{dipole}}
=1.5\,n_1+0.5\,n_2-0.5\,n_3-1.5\,n_4 .
$$

The same combinations are used for layer DOS. Additionally,
scf_layer_dos_lower_block_fraction and stoner_layer_dos_lower_block_fraction
are (DOS_L3+DOS_L4)/DOS_all: one denotes the lower bilayer, zero the upper.
Per-state block weights sum both sublattices and all plane waves on L3-L4.

## Outputs

Common outputs include:

```text
summary.json / run_summary.json
config_used.json
layers.csv
history.csv
nd_map.csv
*.npz cache files
band-structure figures
DOS figures
layer-polarization figures
fermi-contour figures
```

The `scf/` subdirectory contains the accepted Hartree stage before Stoner:
`nd_map.csv`, `nd_map.json`, numeric map arrays, layer DOS, layer density,
potentials, band and DOS figures for detail points. This stage is retained if
the subsequent Stoner optimizer fails. It reuses the same calculation.

Stoner DOS and band figures require a physically consistent common mean-field
Fermi level. A constrained filling-window optimum can be valid while no such
Fermi level exists; these spectral fields are then NaN, not zero, with the
reason recorded. `stoner_dos_valid` distinguishes this case. See the numerical
audit for the finite-grid occupation and kinetic-energy conventions.

For detail points, full_band_stoner_cache.npz stores compressed full spectra,
compact lower-bilayer probabilities, U_scf, flavor fillings, chemical potentials,
layer densities and layer DOS. Complete eigenvectors are used in RAM and are
not saved by this workflow. Details export Origin TXT with Long Name/Units/
Comments rows and block-colored PDF/PNG plots for both stages.

Hartree gaps at nu=0,+/-4 and post-Stoner gaps at integer reference fillings
-6 through +6 have independent numeric maps. Missing/invalid values are NaN;
a validated overlap gives zero gap. Gaps use the fixed Hamiltonian at each
point's accepted Hartree potential and exchange shifts.

Use --reuse-scf-run for a fresh Stoner-only pass on saved Hartree potentials,
--update-stoner-in-place to archive and update the same folder, and
--restart-stoner-in-place for another generation. --resume --out continues an
interrupted generation. See the Windows deployment document.

## Finite-Q susceptibility status

Finite-Q code exists under

```text
tdbg_scf/susceptibility/
examples/run_stoner_susceptibility_single.py
examples/run_nd_mapping_stoner_susceptibility.py
examples/run_fixed_nu_Dscan_stoner_susceptibility.py
```

However, this part is not finalized. The present finite-Q output should be
treated as a diagnostic/prototype only. In particular:

- the finite-Q ratio can be misleading when the Gamma-point denominator is small;
- the channel selection and soft-mode definition still need careful validation;
- the current implementation should not be used as a paper-level conclusion;
- any finite-Q result should be rederived and benchmarked before being quoted.

The current reliable paper workflow should therefore be:

```text
continuum Hamiltonian -> SCF Hartree -> full-band Stoner -> DOS/layer/flavor analysis
```

not finite-Q.

## Installation

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run scripts from the repository root. On Windows PowerShell, the examples above
work directly. On Linux/macOS, replace backslashes with slashes and use the
usual shell continuation syntax.

For large server runs, set BLAS thread counts to one before using process-level
parallelism:

```powershell
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
```

## Notes for manuscript writing

For a more detailed note about the basis, layer order, displacement-field sign,
self-consistency loop, Stoner model, and polarization definitions, see:

```text
docs/TDBG_model_and_workflow_note.md
docs/TDBG_model_and_workflow_note.docx
docs/TDBG_model_and_workflow_note.pdf
```

The exact parameters used for each numerical run should be taken from that run's
saved `config_used.json` or `run_summary.json`, not from this README.

## References

[1] R. Bistritzer and A. H. MacDonald, "Moire bands in twisted double-layer
graphene," PNAS 108, 12233 (2011),
https://doi.org/10.1073/pnas.1108174108.

[2] M. Koshino, "Band structure and topological properties of twisted double
bilayer graphenes," Phys. Rev. B 99, 235406 (2019),
https://doi.org/10.1103/PhysRevB.99.235406.

[3] N. R. Chebrolu, B. L. Chittari, and J. Jung, "Flat bands in twisted double
bilayer graphene," Phys. Rev. B 99, 235417 (2019),
https://doi.org/10.1103/PhysRevB.99.235417.

[4] F. Guinea and N. R. Walet, "Electrostatic effects, band distortions and
superconductivity in twisted graphene bilayers," PNAS 115, 13174 (2018),
https://doi.org/10.1073/pnas.1810947115.

[5] T. Cea, N. R. Walet, and F. Guinea, "Electronic band structure and pinning of
Fermi energy to van Hove singularities in twisted bilayer graphene: a self
consistent approach," arXiv:1906.10570 (2019),
https://arxiv.org/abs/1906.10570.

[6] Y. Guo, O. I. Sheekey, T. Arp, K. Kolar, T. Charpentier, L. Holleis,
B. Foutty, A. Keough, M. Kang-Chou, M. E. Huber, T. Taniguchi, K. Watanabe,
C. Lewandowski, and A. F. Young, "Flat band surface state superconductivity in
thick rhombohedral graphene," arXiv:2511.17423 (2025),
https://arxiv.org/abs/2511.17423.

[7] K. Kolar, A. F. Young, and C. Lewandowski, "Electrostatically stabilized
surface flat bands in rhombohedral graphite at zero displacement field,"
arXiv:2605.24080 (2026), https://arxiv.org/abs/2605.24080.

[8] H. Zhou, T. Xie, A. Ghazaryan, T. Holder, J. R. Ehrets, E. M. Spanton,
T. Taniguchi, K. Watanabe, E. Berg, M. Serbyn, and A. F. Young, "Half- and
quarter-metals in rhombohedral trilayer graphene," Nature 598, 429 (2021),
https://doi.org/10.1038/s41586-021-03938-w.

[9] H. Zhou, L. Holleis, Y. Saito, L. Cohen, W. Huynh, C. L. Patterson,
F. Yang, T. Taniguchi, K. Watanabe, and A. F. Young, "Isospin magnetism and
spin-polarized superconductivity in Bernal bilayer graphene," Science 375, 774
(2022), https://doi.org/10.1126/science.abm8386.

[10] M. Uzan, W. Zhi, M. Bocarsly, J. Dong, S. Dutta, N. Auerbach,
N. S. Kander, M. Labendik, Y. Myasoedov, M. E. Huber, K. Watanabe,
T. Taniguchi, D. E. Parker, and E. Zeldov, "hBN alignment orientation controls
moire strength in rhombohedral graphene," arXiv:2507.20647 (2025),
https://arxiv.org/abs/2507.20647.
