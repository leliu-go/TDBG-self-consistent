# Full continuum Hartree migration and numerical audit

This change updates the original four-layer `leliu-go/TDBG-self-consistent` repository. The model remains the original ABBA continuum Hamiltonian, with layers L1 through L4 ordered top to bottom. The strong dimer pairs are A1–B2 and B3–A4; the moire tunneling connects L2 and L3. The source of the Hamiltonian, lattice, constants and electrostatics is preserved.

## Full Hartree

Every iteration diagonalizes the full `8(2N+1)^2` Hamiltonian at every k point, finds the density-matched chemical potential, computes all four layer densities using all eigenvectors, and applies the original Gauss-law electrostatics. The half-state-per-orbital neutrality subtraction and spin/valley degeneracy are unchanged. There is no active-band projection, frozen remote-band density or projected-HF path.

The result includes eigenpairs at the returned potential. They are reused for the chosen SCF valley. The other valley is diagonalized once for full-band postprocessing. A potential is accepted only if the actual fixed-point residual is below tolerance, the total layer density matches the requested density, and the quantities are finite. Failed single points/maps retain diagnostics and do not produce physical DOS or Stoner results from an invalid Hartree potential.

The default `D_sign=-1` is retained: the internal gate field is `D_gate=D_sign*D_input=e(n_b-n_t)/(2 epsilon0)`. With no layer charge and positive input D, the electron energies satisfy U1>U2>U3>U4; adjacent drops are `1000*d_layer_nm*D_input/eps_perp` meV. Initial uploaded potentials are starting guesses, not the self-consistent electrostatics.

## Anderson damping

For residual `r=F(x)-x`, history differences `Delta X`, `Delta R` and least-squares coefficient `gamma`, the damped Type-II update is

```text
x_next = x + beta*r - (Delta X + beta*Delta R) @ gamma
```

The earlier code multiplied the entire history correction by beta a second time. This has been corrected, keeping the existing default beta=0.5 and memory=6. The first/fallback step uses the specified linear mixing alpha. Zero-average potential gauge remains enforced.

The formula follows Walker and Ni, Eq. (1.2), and the damping section of Walker's implementation note:

- [Walker and Ni, SIAM J. Numer. Anal. 49, 1715–1735 (2011)](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf)
- [Walker, Anderson Acceleration: Algorithms and Implementations, pp. 5–6](https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf)

The old residual test also took an extra unmixed `U=F(U)` step after passing tolerance. A stiff fixed-point map can magnify the remaining error in that step. The solver now returns the tested potential. Tests compare with the exact solution of a stiff linear problem and independently recompute the electrostatic residual from the returned eigenstates.

## Density, Stoner and spectral consistency

At zero temperature on a finite k mesh, a target density may cut through a degenerate Fermi-level group. A chemical-potential root alone cannot select the required fractional occupation with a step distribution. The solver therefore distributes the needed occupation equally within that group, using integration weights, and checks total charge. Finite-temperature Fermi occupations retain their original physical definition.

The cumulative Stoner kinetic table is piecewise linear. Its derivative is the energy of the partially occupied state, with an interval of allowed chemical potentials at a kink. Linear interpolation of endpoint energies is not that derivative. Optimization and common mean-field Fermi-level checks use consistent slopes and intervals. Initial seeds are not marked successful solely for being finite. A candidate must pass filling, bounds, constrained stationarity and local energy-stability checks.

SLSQP can report success at a symmetric stationary saddle, or stop near a kinetic kink with a nonzero stationarity residual. The solver therefore checks feasible redistribution directions, including coupled Hund directions, against the actual piecewise-quadratic energy. When necessary, refinement minimizes that energy along feasible segments ending at the next kinetic kink or filling bound. It uses only the remaining per-seed iteration budget, retains rejected optimizer attempts in diagnostics, and applies the same strict final checks. Kink, boundary and flat minima remain admissible. The selected state is the lowest-energy accepted local minimum among the attempted seeds; this is not a certificate of the global minimum.

The Stoner calculation remains the original phenomenological four-flavor carrier-sector model, after Hartree, without feedback into Hartree. The four flavors are `(K_up, Kp_up, K_down, Kp_down)`. Spectral DOS and bands require a common Fermi level after applying the interaction shifts. If carrier-sector constraints prevent a physical common level, the valid occupation/energy result is retained, while spectral fields are NaN and `stoner_dos_valid=false` records the reason.

Pure Hartree data are saved in `scf/` before Stoner so they remain available if that later stage fails. The Stoner DOS figure now plots the shifted mean-field spectra relative to the common interacting Fermi level alongside the Hartree DOS. Total DOS equals the sum of flavor and layer contributions by construction and is checked in regression tests.

Rerunning a single point clears only that workflow's named generated files, so an earlier accepted spectrum cannot survive a rejected Hartree calculation. This also applies to the existing susceptibility single-point entry point and map worker transport failures. Mapping always replaces its scalar cache, including an empty successful-row table when all points fail. A plot whose quantity is absent is removed. A failed configured scan saves a current error and, if available, completed records from that run as partial results; it cannot retain an earlier complete results file. Unrelated files are preserved.

## Verification scope

Analytical tests cover raw tunneling Hermiticity (before any symmetrization), both-valley time reversal, uniform potential shifts, the dimer-only limit, electrostatic units and sign, Anderson damping, charge conservation and Stoner stationarity. Integration tests cover full-only CLI controls, removal of projected modules, failure gates, full eigensystem reuse and stage-specific outputs.

Verification on Windows, 2026-10-06, with Python 3.11.7, NumPy 1.26.4, SciPy 1.11.4 and one native math thread per process:

- The complete test suite passed: 139 tests in 31.08 seconds. Failure-output and Stoner saddle/kink regressions were reproduced before their fixes. Independent specification and code-quality reviews approved the final implementation; the final quality review independently passed 88 focused tests.
- A native full Hartree/Stoner point at theta=1.2 degrees, N=1, 4x4 k mesh, n=0 and D=0 passed in eight Hartree iterations, with residual 1.5504e-5 meV and charge error 0.00551 cm^-2.
- A native two-worker map at theta=1.2 degrees, N=1, 4x4 mesh, n in {-4e11, 0, +4e11} cm^-2 and D in {-0.2, +0.2} V/nm passed all six points. Maximum Hartree residual was 6.3115e-5 meV; maximum charge error was 0.0201 cm^-2. CSV/NPZ data and layer/flavor DOS sums were checked.
- At theta=1.35 degrees, N=2, 6x6 mesh, eps_perp=4, all four server corners n=+/-6e12 cm^-2, D=+/-1.2 V/nm and the n=D=0 point passed Hartree, Stoner stationarity, local-stability and common-Fermi-level/DOS checks. Hartree took 6-11 iterations; maximum residual was 2.7003e-5 meV. The selected corner states each needed three exact refinement steps after SLSQP, within the original 200-step budget.
- Native Hartree-only and configured correlated Stoner commands passed. A deliberately limited one-iteration SCF command exited with failure and saved diagnostics instead of physical results. An accepted-to-rejected Gamma-only susceptibility rerun exited with failure, removed its earlier spectra and saved the current SCF diagnostic. This checks workflow integration, not finite-Q physics.

These are numerical implementation checks on small meshes, not cutoff or k-mesh convergence studies for publication. No complete 48x48 production n-D map was run as part of this audit.

`Full` refers to all states in the chosen continuum plane-wave basis. Accuracy still requires convergence in cutoff N and k-grid, and examination of potentially competing Hartree solutions. This audit does not add exchange or moire-periodic Hartree terms, nor does it certify the experimental finite-Q susceptibility formulation.
