# Windows PowerShell: full Hartree followed by Stoner

Run in the repository root. The original model is four-layer ABBA. All bands of the chosen cutoff N are included in every Hartree iteration. Projection controls have been removed.

## Deploy

```powershell
git clone https://github.com/leliu-go/TDBG-self-consistent.git
cd TDBG-self-consistent
python -m pip install -r requirements.txt

$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
```

For an existing checkout, use `git pull --ff-only origin master` before running.

## One point

```powershell
python -u examples\run_full_band_stoner.py `
  --theta-deg 1.35 --cutoff 2 `
  --n-cm2 0 --D-Vnm 0 `
  --grid-n1 48 --grid-n2 48 `
  --kBT-meV 0.2 --eps-perp 4 --D-sign=-1 `
  --full-max-iter 100 --full-tol-meV 1e-4 `
  --mixer anderson --alpha 0.06 --anderson-beta 0.5 --anderson-memory 6 `
  --u0-meV-A2 7.9e4 --JH-meV-A2 2.4e4 `
  --out outputs\full_hartree_single
```

## n-D map

The example below retains the original server density range. Change the grid, angle and output directory as required.

```powershell
python -u examples\run_nd_mapping_full_band_stoner.py `
  --theta-deg 1.35 --cutoff 2 `
  --n-min-cm2=-6e12 --n-max-cm2 6e12 --n-count 121 `
  --D-min-Vnm=-1.2 --D-max-Vnm 1.2 --D-count 121 `
  --grid-n1 48 --grid-n2 48 `
  --kBT-meV 0.2 --eps-perp 4 --d-layer-nm 0.335 --D-sign=-1 `
  --full-max-iter 100 --full-tol-meV 1e-4 `
  --mixer anderson --alpha 0.06 --anderson-beta 0.5 --anderson-memory 6 `
  --u0-meV-A2 7.9e4 --JH-meV-A2 2.4e4 `
  --max-workers 40 `
  --out outputs\ndmap_full_hartree
```

`scf/nd_map.csv` and `scf/nd_map_scalars.npz` save pure Hartree DOS, layer DOS, layer density, polarization and convergence diagnostics. Main `nd_map.csv` additionally contains Stoner quantities. Detail points also contain both stages' band/DOS figures and caches. No second Hartree calculation is run to export the pure stage.

Check `run_summary.json`, failed rows and `stoner_dos_valid`. A failed Hartree stage does not generate physical analysis. A valid Hartree stage is retained if Stoner fails. An optimum that lacks a common interacting Fermi level retains occupation/energy data, but its Stoner spectral fields are NaN. The command exits nonzero if points fail or spectral validation fails.

Full Hartree is more expensive than the retired fixed eight-band approximation because each iteration rediagonalizes the full Hamiltonian. Use a small run first and verify cutoff/k-mesh convergence for the final physical quantities.
