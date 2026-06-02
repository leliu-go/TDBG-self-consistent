# Server CLI: finite-Q susceptibility n-D mapping

Run from the repository root:

```powershell
cd D:\Code\self-consistent-code\tdbg_scf_project\tdbg_scf_project

$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

python -u examples\run_nd_mapping_stoner_susceptibility.py `
  --out D:\LeLiu\data\TDBG_self_consistent\finite_q_susceptibility_U0_7p9e4_JH_2p4e4_dense_more `
  --resume `
  --theta-deg 1.35 `
  --cutoff 2 `
  --omega 100.0 `
  --wAA 0.8 `
  --wAB 1.0 `
  --Z 15.0 `
  --a-cc-A 1.420 `
  --gamma0-meV 2610.0 `
  --gamma1-meV 361.0 `
  --gamma3-meV 283.0 `
  --gamma4-meV 138.0 `
  --n-min-cm2=-6e12 `
  --n-max-cm2 6e12 `
  --n-count 121 `
  --D-min-Vnm=-1.2 `
  --D-max-Vnm 1.2 `
  --D-count 121 `
  --grid-n1 48 `
  --grid-n2 48 `
  --scf-source projected_8band `
  --scf-valley 1 `
  --projected-reference-mode uploaded_D `
  --n-active 8 `
  --selection overlap `
  --projector-refreshes 0 `
  --kBT-meV 0.2 `
  --eps-perp 4.0 `
  --d-layer-nm 0.335 `
  --D-sign=-1.0 `
  --max-iter 100 `
  --tol-meV 1e-4 `
  --mixer anderson `
  --alpha 0.06 `
  --anderson-beta 0.5 `
  --anderson-memory 6 `
  --stoner-u0-meV-A2 7.9e4 `
  --stoner-JH-meV-A2 2.4e4 `
  --stoner-n-random-seeds 20 `
  --stoner-seed 0 `
  --chi-kBT-meV 0.05 `
  --energy-window-meV 50 `
  --max-bands-per-k 24 `
  --main-vertex-model su4_diag `
  --also-run-hund-factor2 `
  --hund-transverse-factor 2.0 `
  --occupation-mode flavor_mu `
  --spin-flip-mode both_pm `
  --q-mode folded_grid_with_G_shift `
  --q-stride 4 `
  --max-workers 40
```

Smoke test version:

```powershell
cd D:\Code\self-consistent-code\tdbg_scf_project\tdbg_scf_project

$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

python -u examples\run_nd_mapping_stoner_susceptibility.py `
  --out D:\LeLiu\data\TDBG_self_consistent\finite_q_susceptibility_U0_7p9e4_JH_2p4e4_smoke `
  --theta-deg 1.35 `
  --cutoff 2 `
  --omega 100.0 `
  --wAA 0.8 `
  --wAB 1.0 `
  --Z 15.0 `
  --a-cc-A 1.420 `
  --gamma0-meV 2610.0 `
  --gamma1-meV 361.0 `
  --gamma3-meV 283.0 `
  --gamma4-meV 138.0 `
  --n-min-cm2=-6e12 `
  --n-max-cm2 6e12 `
  --n-count 121 `
  --D-min-Vnm=-1.2 `
  --D-max-Vnm 1.2 `
  --D-count 121 `
  --grid-n1 48 `
  --grid-n2 48 `
  --scf-source projected_8band `
  --scf-valley 1 `
  --projected-reference-mode uploaded_D `
  --n-active 8 `
  --selection overlap `
  --projector-refreshes 0 `
  --kBT-meV 0.2 `
  --eps-perp 4.0 `
  --d-layer-nm 0.335 `
  --D-sign=-1.0 `
  --max-iter 100 `
  --tol-meV 1e-4 `
  --mixer anderson `
  --alpha 0.06 `
  --anderson-beta 0.5 `
  --anderson-memory 6 `
  --stoner-u0-meV-A2 7.9e4 `
  --stoner-JH-meV-A2 2.4e4 `
  --stoner-n-random-seeds 20 `
  --stoner-seed 0 `
  --chi-kBT-meV 0.05 `
  --energy-window-meV 30 `
  --max-bands-per-k 24 `
  --main-vertex-model su4_diag `
  --also-run-hund-factor2 `
  --hund-transverse-factor 2.0 `
  --occupation-mode flavor_mu `
  --spin-flip-mode both_pm `
  --q-mode folded_grid_with_G_shift `
  --q-stride 1 `
  --max-abs-q-step 4 `
  --max-workers 4 `
  --max-points 4
```

Finite-Q controls:

- `--grid-n1 48 --grid-n2 48`: k-point grid for projected SCF, full-continuum diagonalization, Stoner, and the folded finite-Q scan. If omitted, the script default is `9 x 9`.
- `--q-stride 1`: use every commensurate q-grid step.
- `--max-abs-q-step 4`: restrict to `|dq1| <= 4` and `|dq2| <= 4`; with `q-stride 1`, this gives 81 Q points.
- `--q-mode folded_grid_with_G_shift`: folded MBZ Q scan with plane-wave reciprocal shift correction.
- `--chi-kBT-meV 0.05`: Fermi smearing used only in the susceptibility bubble.
- `--energy-window-meV 30`: only keep bands near each flavor chemical potential for the bubble.
- `--max-bands-per-k 24`: max bands kept per k point in the bubble.

Output files:

- `finite_q_susceptibility.csv`: main incremental result table. `--resume` skips points already present here.
- `config_used.json`: exact command, resolved args, platform, worker count, and thread environment.
- `run_summary.json`: updated while running; contains status, point counts, failures, and runtime.
- `source_grid.csv`: full generated or loaded n-D grid before filters.
- `todo_grid.csv`: grid after filters such as `--max-points`, `--row-stride`, `--nu-min`, or `--D-min`.
