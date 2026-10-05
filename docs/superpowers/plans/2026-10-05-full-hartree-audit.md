# Full Hartree migration and numerical audit

The user requests removal of the eight-band projection from TDBG-self-consistent, full continuum layer-Hartree self-consistency throughout, and correction of numerical errors including Anderson mixing.

Baseline: master cacff761aa1708a477a01ee8a86d5f50d7d1d124. Work in an isolated checkout. Preserve the four-layer ABBA Hamiltonian, lattice, physical parameters, gate convention, and layer-density reference. Do not import the five-layer Hamiltonian.

- [x] Record baseline tests and source hashes; check damping against Walker and Ni and reproduce confirmed bugs with regression tests.
- [x] Correct damped Anderson Type-II, stop at the tested potential, validate charge and final full residual, and audit chemical-potential and Stoner numerics.
- [x] Delete projected Hartree/HF modules and their dedicated entry points; remove projected options, configurations, plotting helpers, and retired tests.
- [x] Route single-point, n-D mapping, correlated CLI and existing susceptibility workflows through converged full Hartree. Reuse the returned full eigensystem. Preserve physical outputs and report failed stages explicitly.
- [x] Update documentation and server commands for full-only calculation, documenting actual fixes and model limits.
- [x] Run analytical, numerical, CLI and map verification; perform independent specification and code-quality reviews and resolve findings.

Publication procedure: publish the verified files as a fast-forward commit on the original repository master, then verify its parent, changed paths and remote file hashes against the tested local manifest.

Anderson convention: r = F(x)-x, gamma minimizes ||r - Delta R gamma||, x_next = x + beta r - (Delta X + beta Delta R) gamma. Source: Walker and Ni, SIAM J. Numer. Anal. 49 (2011), Eq. (1.2), and Walker, Anderson Acceleration: Algorithms and Implementations, damping section.

Full means all 8(2N+1)^2 states of the truncated continuum basis at every Hartree iteration; plane-wave and k-mesh convergence remain numerical convergence questions. The Hartree model includes layer electrostatics and does not add spatial Hartree modulation or exchange. Existing phenomenological Stoner remains a subsequent calculation.

Validation record (2026-10-06): baseline 64 tests passed; final migrated suite 139 tests passed. Native single-point, two-worker six-point mapping, five N=2 server representative points, Hartree-only, configured Stoner and Gamma-only susceptibility glue commands were exercised. Specification and code-quality reviews approved after stale-output, Stoner saddle and kinetic-kink regressions were reproduced and fixed. The final independent quality review passed 88 focused tests. Detailed evidence and model limits are recorded in `docs/full_hartree_numerical_audit.md`.
