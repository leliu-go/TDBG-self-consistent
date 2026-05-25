# Full Band Stoner Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runnable local and server examples that run full SCF at each `(n,D)` point, then run a four-flavor Stoner model on the resulting full continuum bands.

**Architecture:** Keep the physics glue in `tdbg_scf/stoner/full_band.py`, then keep the two examples thin. Full SCF remains owned by `tdbg_scf/solver_full.py`; the new scripts consume `U_scf`, full-band eigensystems, and layer densities without changing the existing SCF implementation.

**Tech Stack:** Python, NumPy, SciPy, Matplotlib, Pandas, existing `tdbg_scf` solvers and Stoner module.

---

### Task 1: Signed Full-Band Stoner Tables

**Files:**
- Create: `tdbg_scf/stoner/full_band.py`
- Modify: `tdbg_scf/stoner/solver.py`
- Test: `tests/test_full_band_stoner.py`

- [ ] Write tests for neutrality-referenced signed DOS tables and signed Stoner minimization.
- [ ] Verify the tests fail before implementation.
- [ ] Implement full-band table helpers and make Stoner seeds work for negative total filling.
- [ ] Verify the focused tests pass.

### Task 2: Local Single-Point Example

**Files:**
- Create: `examples/run_full_band_stoner.py`
- Test: `tests/test_full_band_stoner_scripts.py`

- [ ] Write a CLI help smoke test.
- [ ] Implement a single-point example that runs full SCF, diagonalizes both valleys at `U_scf`, solves Stoner, and writes JSON/CSV/PNG/NPZ outputs.
- [ ] Verify the script help test and a tiny runtime smoke pass.

### Task 3: Server n-D Mapping Example

**Files:**
- Create: `examples/run_nd_mapping_full_band_stoner.py`
- Test: `tests/test_full_band_stoner_scripts.py`

- [ ] Extend CLI help tests to cover server mapping options.
- [ ] Implement parallel `(n,D)` scan with full SCF per point and Stoner post-processing.
- [ ] Save scalar maps, detail mesh outputs, full bands, DOS, layer profile, and flavor polarization figures.
- [ ] Run full test suite and a small mapping smoke.
