# TDBG numerical and output parity implementation plan

**Goal:** Apply the user-approved audit fixes and current 2+3 public workflow features while preserving every existing TDBG physical parameter and Hamiltonian convention.

**Architecture:** Keep the four-layer continuum, Hartree solver, electrostatics, fractional-occupation density routines and original entry-point defaults. Port the shared Stoner optimizer and analysis/output modules with explicit four-layer adapters. Preserve resumable transactional point publication and require source compatibility; older scans may be reused as fixed Hartree references through a new Stoner-only run.

**Tech stack:** Python >=3.8, NumPy, SciPy 1.10-compatible DE, Matplotlib, pandas, Windows PowerShell.

## Invariants

- Preserve all original dataclass and CLI physical defaults, including the independent Hartree epsilon=4 and server/Stoner epsilon=3; never silently harmonize them.
- Preserve gamma4=138 meV, Z=15 meV, original tunnelling, U0/JH, grid and scan defaults, D_sign=-1, and ABBA legacy basis.
- Metadata identifies actual dimers A1-B2 and B3-A4, physical top-to-bottom BA-AB in the fixed B_upper-A_lower=AB convention.
- No eight-band Hamiltonian truncation. Active-band and block weights are observables of full eigenstates.
- No complete eigenvectors saved by the main detailed workflow. Keep compact lower-bilayer weights and compressed spectra; old full-vector cache reading remains supported where useful.
- Publish only after user authorization; the user authorized GitHub publication and removal of the local audit checkout on 2026-10-10. Do not change finite-Q physics.

## Execution checklist

- [x] Lock original physical defaults and continuum matrix through regression tests; record baseline hashes.
- [x] Reproduce endpoint stability leak and missing two-seed DE diagnostics with failing tests, then port energy/solver and algorithm configuration.
- [x] Add active-carrier normalization, integer gap analysis, lower-bilayer projector weights and physical metadata; verify charge, degeneracies, bounds and formulas.
- [x] Integrate compressed compact detailed outputs, Origin headers/bands/contours/DOS and block plots into the single-point workflow; retain target-aware T=0 density handling.
- [x] Integrate gap/active maps, Hartree-only records, fixed-Hartree Stoner replay, transactional in-place restart and resume into the nD workflow.
- [x] Update the Windows launcher and documentation for DE/replay while preserving original defaults. Check argument forwarding and inherited saved physical inputs.
- [x] Run focused numerical and integration tests, the relevant full suite, Windows preview checks, and a real tiny end-to-end detailed/replay/resume smoke run.
- [x] Inspect diff for parameter drift and forbidden full-vector writes; report scope, evidence and remaining limitations.

## Verification approach

Tests must fail for the observed missing behavior before production changes. Reuse the 2+3 numerical regression tests only where their contracts are model-independent; adapt layer tests to four layers. Retain all existing TDBG full-Hartree, finite-grid fractional occupation and correlated scan acceptance tests. Compare Hamiltonian/density/electrostatic source hashes to the baseline at completion. Final smoke results go under the workspace temporary directory.

## Final evidence

- Relevant regression suite: 337 passed in 886.06 seconds on 2026-10-10. Finite-Q susceptibility tests were excluded because finite-Q physics is outside this change.
- Native Windows PowerShell launch, fixed-Hartree replay, in-place restart, interrupted archive recovery and resume passed, including a real two-point replay with zero Hartree iterations.
- Continuum, density, electrostatics, mixing, full Hartree solver and standalone Hartree entry-point files retain their original SHA-256 hashes. Original physical CLI and Windows defaults remain unchanged.
- All 62 production Python files parse under Python 3.8 syntax rules; numerical tests ran with Python 3.11.7, NumPy 1.24.4 and SciPy 1.10.1.
- Real detailed outputs contain both Hartree and Stoner Origin tables and block plots, with four layer rows and no complete eigenvectors in the compact cache. A representative plot was visually inspected.
- Read-only independent review findings were reproduced and fixed; follow-up review reported no outstanding actionable findings.
