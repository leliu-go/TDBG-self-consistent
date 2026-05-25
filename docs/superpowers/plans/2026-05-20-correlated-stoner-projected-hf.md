# Correlated Stoner Projected HF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent Stoner and projected-HF modules with a shared CLI while keeping existing TDBG Hartree solvers working.

**Architecture:** Shared `flavors` and `filling` helpers provide explicit spin/valley flavor conventions and filling-density conversion. `tdbg_scf/stoner` implements a complete phenomenological Stoner solver. `tdbg_scf/projected_hf` implements explicit-flavor projected-HF scaffolding with `none` and `contact_projected` exchange; true Coulomb form-factor HF raises `NotImplementedError`.

**Tech Stack:** Python, NumPy, SciPy optimizer, existing TDBG continuum/projected Hartree modules, pytest.

---

### Task 1: Shared Helpers

**Files:**
- Create: `tdbg_scf/flavors.py`
- Create: `tdbg_scf/filling.py`
- Test: `tests/test_correlated_shared.py`

- [ ] Write tests for flavor names and filling-density round trips.
- [ ] Implement `Flavor`, `FLAVORS`, `flavor_names`.
- [ ] Implement moire-cell area and filling conversion helpers.
- [ ] Run `pytest tests/test_correlated_shared.py -q`.

### Task 2: Stoner Core

**Files:**
- Create: `tdbg_scf/stoner/__init__.py`
- Create: `tdbg_scf/stoner/params.py`
- Create: `tdbg_scf/stoner/dos.py`
- Create: `tdbg_scf/stoner/energy.py`
- Create: `tdbg_scf/stoner/solver.py`
- Create: `tdbg_scf/stoner/scan.py`
- Create: `tdbg_scf/stoner/io.py`
- Test: `tests/test_stoner.py`

- [ ] Write tests for DOS table capacity, zero-interaction behavior, large-`u0` flavor polarization, and Hund alignment.
- [ ] Implement DOS cumulative filling/kinetic table.
- [ ] Implement Stoner energy functional.
- [ ] Implement fixed-filling multi-seed SLSQP solver and result diagnostics.
- [ ] Implement lightweight scan and IO helpers.
- [ ] Run `pytest tests/test_stoner.py -q`.

### Task 3: Projected-HF Scaffolding

**Files:**
- Create: `tdbg_scf/projected_hf/__init__.py`
- Create: `tdbg_scf/projected_hf/params.py`
- Create: `tdbg_scf/projected_hf/flavor_model.py`
- Create: `tdbg_scf/projected_hf/density_matrix.py`
- Create: `tdbg_scf/projected_hf/self_energy.py`
- Create: `tdbg_scf/projected_hf/energy.py`
- Create: `tdbg_scf/projected_hf/seeds.py`
- Create: `tdbg_scf/projected_hf/solver.py`
- Create: `tdbg_scf/projected_hf/scan.py`
- Create: `tdbg_scf/projected_hf/io.py`
- Create: `tdbg_scf/projected_hf/form_factors.py`
- Create: `tdbg_scf/projected_hf/coulomb.py`
- Test: `tests/test_projected_hf.py`

- [ ] Write tests for density matrix Hermiticity/filling, contact self-energy Hermiticity, Coulomb form-factor not-implemented behavior, and a tiny solver smoke run.
- [ ] Implement flavor projected model containers and construction from existing projected models for both valleys.
- [ ] Implement common chemical potential and density matrix builders.
- [ ] Implement `none` and `contact_projected` self-energy models.
- [ ] Implement projected-HF solver with multi-seed selection and approximate energy.
- [ ] Run `pytest tests/test_projected_hf.py -q`.

### Task 4: Unified Config and CLI

**Files:**
- Create: `tdbg_scf/correlated_config.py`
- Create: `tdbg_scf/correlated_io.py`
- Create: `tdbg_scf/correlated_cli.py`
- Create: `configs/correlated_example.yaml`
- Test: `tests/test_correlated_cli.py`
- Test: `tests/test_module_independence.py`

- [ ] Write tests for module independence and CLI method dispatch.
- [ ] Implement JSON/YAML config loading with YAML optional and JSON fallback.
- [ ] Implement `--method stoner|projected-hf|both` dispatch.
- [ ] Implement comparison summary for `both`.
- [ ] Run CLI smoke tests.

### Task 5: Verification

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m tdbg_scf.correlated_cli --help`.
- [ ] Run Stoner smoke command.
- [ ] Run projected-HF `exchange_model=none` smoke command.
- [ ] Run projected-HF `exchange_model=contact_projected` smoke command.
