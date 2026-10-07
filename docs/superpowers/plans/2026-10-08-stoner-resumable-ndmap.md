# Signed Stoner fillings and resumable TDBG n-D scans

**Goal:** Apply the already approved 2+3 behavior to four-layer ABBA TDBG: each flavor defaults to -2 <= nu_f <= 2, with sum(nu_f)=nu; save every completed point, restore interrupted scans and publish live missing-aware maps and timing.

**Architecture:** Preserve the full continuum Hartree solver, four-layer Hamiltonian, interaction functional and existing local-stability refinement. Intersect each search window with real spectrum capacity. Keep full-spectrum chemical-potential intervals separate from artificial optimization boundaries. Adapt the tested 2+3 atomic writes, OS locks, attempt fencing, checkpoint verification and progress modules; keep existing TDBG observables and density CLI inputs. Add an optional filling axis and a Windows venv launcher.

**Scope:** No finite-Q work, no projection, no new Hamiltonian convention, no production map reruns. A resume skips only validated completed point records; an unfinished point restarts its computation. Legacy runs lack ordinary-point checkpoints and cannot recover RAM-only results.

**Tech stack:** Python 3.8+, NumPy/SciPy, pandas, Matplotlib, Windows PowerShell.

- [x] Add failing analytical tests for compensated neutrality, capacity/window intersection, common-mu rejection at artificial bounds and CLI bounds.
- [x] Update `tdbg_scf/stoner/{params,dos,solver,energy,full_band,io,scan}.py` and both Stoner workflow callers. Preserve local-stability and exact redistribution refinement; exercise existing regression tests.
- [x] Add failing scan tests for point durability, NaN pending cells, resume settings and skipped points, corrupt records, directory locks and timing.
- [x] Adapt `atomic_io.py`, `scan_checkpoint.py`, `live_map.py`, `run_progress.py`, `run_storage.py`, `worker_diagnostics.py` and `scan_axes.py` from the verified 2+3 implementation. Integrate them into the TDBG map driver without five-layer/gap dependencies.
- [x] Add PowerShell launcher and venv setup, retain TDBG physical parameter defaults, validate forwarding and saved-argument restoration with native PowerShell.
- [x] Verify a tiny real serial/parallel scan and resume with Python 3.8; run the repository tests, inspect the diff and document exact deployment/resume commands.

## Verification

Use `D:\Code\twisted2+3\tmp\venv_py38\Scripts\python.exe -m pytest` from this repository. Observe new tests fail before implementation. Checkpoint tests must exercise interruption after a real point and before its next point, filesystem checksums and worker-before-parent commit. Integration scans use cutoff=0 and a 2x2 k-grid, rather than an expensive production map. Published arrays must have the complete scan shape with NaN data and false validity masks for pending cells. Failed transport must not erase a committed physical result.

## Completed validation

- CPython 3.8.20, NumPy 1.24.4, SciPy 1.10.1: **182 tests passed in 74.40 seconds**.
- Independent review reproduced a retry-transport corner case that could hide earlier valid Hartree maps; a failing regression was added and the issue was fixed. Newer worker commits retain priority, and transport failures leave checkpoint files untouched.
- A real parent process was terminated with os._exit, then resumed with two Windows workers. The original completed checkpoint was byte-identical and retained its modification timestamp.
- Native PowerShell successfully ran a tiny real scan, then restored it without recomputing the completed point.
- Live preview was rendered and inspected; NaN pending cells remain blank.
- pip check and git diff --check passed. No production-sized scan was run.
