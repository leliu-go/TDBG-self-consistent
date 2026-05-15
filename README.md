# ABBA TDBG continuum model with full and projected self-consistent Hartree solvers

This project rewrites the uploaded `ABBATDBG2.py` into a reusable Python package and adds two electrostatic self-consistency routes:

1. **Full non-projected SCF**: diagonalizes the full `8*N_G` continuum Hamiltonian at every iteration.
2. **Projected eight-band SCF**: diagonalizes the full Hamiltonian at a reference potential, selects low-energy minibands near the Fermi level, projects the layer-potential operators into this subspace, and performs the self-consistency in the reduced eight-band space.

The code keeps the uploaded model conventions: square plane-wave cutoff, layer order `(A1,B1,A2,B2)` for the upper bilayer and `(A3,B3,A4,B4)` for the lower bilayer, three moire tunneling matrices `Tqb`, `Tqtr`, `Tqtl`, and the high-symmetry path `K_s -> Gamma_s -> M_s -> K'_s`.

## Install

```bash
pip install -r requirements.txt
```

Run from the project root with `PYTHONPATH=.`.

## Quick full-SCF test

```bash
PYTHONPATH=. python examples/run_full_scf.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --D-Vnm 0.2 \
  --n-cm2 0 \
  --grid-n1 5 --grid-n2 5 \
  --max-iter 30 \
  --tol-meV 1e-3 \
  --out outputs/full_quick
```

`cutoff=1` is only for quick debugging. Increase to `cutoff=2` or larger for convergence checks.

## Quick projected eight-band SCF test

```bash
PYTHONPATH=. python examples/run_projected_8band_scf.py \
  --theta-deg 1.35 \
  --cutoff 1 \
  --D-Vnm 0.2 \
  --n-cm2 0 \
  --grid-n1 7 --grid-n2 7 \
  --n-active 8 \
  --max-iter 40 \
  --tol-meV 1e-3 \
  --out outputs/proj8_quick
```

A more serious run might use:

```bash
PYTHONPATH=. python examples/run_projected_8band_scf.py \
  --theta-deg 1.35 \
  --cutoff 2 \
  --D-Vnm 0.2 \
  --n-cm2 0 \
  --grid-n1 15 --grid-n2 15 \
  --n-active 8 \
  --projector-refreshes 1 \
  --max-iter 100 \
  --tol-meV 1e-4 \
  --out outputs/proj8_cut2
```

## Outputs

Both scripts produce:

- `summary.json`
- `layers.csv`
- `history.csv`
- layer-potential profile plot
- band-structure plots
- DOS plot
- `.npz` data files for further analysis

For the projected solver, `projected_8band_model.npz` contains the quantities needed to reuse the eight-band model:

- `miniband_energies0_meV`
- `layer_mats` = projected layer-potential matrices `Lambda_l(k)`
- `U_ref_meV`
- `U_scf_meV`
- `remote_density_per_k_layer`
- `selected_indices`

The projected Hamiltonian is

```text
H8(k;U) = diag(epsilon0(k)) + sum_l [U_l - U_l_ref] Lambda_l(k).
```

## Notes

- Density units inside the code are Angstrom^{-2}; command-line density is in cm^{-2}.
- Hamiltonian momentum units are Angstrom^{-1}, matching the uploaded script where `hv` has units meV Angstrom.
- The electrostatic update uses a uniform layer Hartree potential. It does not include moire-periodic Hartree components.
- The four layer potentials are gauge-fixed by subtracting their average.

## Displacement-field sign

The default command-line convention is chosen to match the uploaded `ABBATDBG2.py`: a positive `--D-Vnm` corresponds, before self-consistent screening, to

```text
U = (+3/2, +1/2, -1/2, -3/2) Zk,
D = 4 Zk / 330  V/nm.
```

Internally this is implemented with `--D-sign -1`. Use `--D-sign +1` if you want the direct convention `D = e(n_b-n_t)/(2 eps0)` with the same top-to-bottom layer order.
