# Windows PowerShell: signed Stoner and resumable full-Hartree maps

The four-layer Hamiltonian remains unchanged. Its legacy identifier is ABBA; its explicit top-to-bottom dimers give BA-AB (A1-B2 and B3-A4). Every Hartree iteration uses the full continuum Hamiltonian of the chosen cutoff. The Stoner search now permits electron and hole flavors simultaneously, with conserved total filling:

\[
-2\leq\nu_f\leq2,\qquad\sum_{f=1}^4\nu_f=\nu.
\]

The search window intersects actual spectrum capacity; it does not truncate the full-band kinetic tables or introduce an active-band projection. Bounds are configurable with `-StonerFlavorNuMin` and `-StonerFlavorNuMax` (Python: `--stoner-flavor-nu-min`, `--stoner-flavor-nu-max`).

## Update and create the environment

From the existing server checkout:

```powershell
Set-Location D:\LeLiu\code\TDBG-self-consistent
git pull --ff-only origin master
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_venv_windows.ps1
```

The setup script uses the existing 64-bit Python at `C:\Users\liule\AppData\Local\Programs\Python\Python38\python.exe`, creates `.venv` and installs pinned dependencies. If Python is elsewhere, pass `-PythonBin 'C:\path\to\python.exe'`. The environment is created on the server and excluded from Git. The launcher automatically uses `.venv\Scripts\python.exe` and limits BLAS to one thread per worker.

## Start a scan

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ndmap_windows.ps1 -Workers 20
```

PowerShell defaults: theta=1.35 degrees, N=2, 36x36 k mesh, total nu=-6...6 at 61 points, D=-0.8...0.8 V/nm at 41 points and 10 detailed points. The filling grid includes nu=0,+/-4. Density uses the current twist-angle moire area, `n_cm2=nu*1e16/A_M_A2`, including all four flavors. The Hamiltonian, Z=15 meV, U0=79000 meV A^2, JH=24000 meV A^2 and eps_perp=3 retain native TDBG physical defaults. The Hartree iteration limit is 100; pass `-EpsPerp 4` to reproduce the earlier server examples' explicitly selected screening value.

For another angle or worker count:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ndmap_windows.ps1 -ThetaDeg 1.2 -Workers 40
```

`-Preview` prints arguments and RAM estimates without computing. `-Out 'D:\path\to\new_run'` selects an exact directory. A new run refuses to overwrite an existing result directory.

Default result layout:

```text
D:\LeLiu\data\tdbg_self_consistent\ABBA_theta1p35_N2_ndmap_<timestamp>\
```

Startup displays `Results:` and a locally calibrated ETA range. Each returned point displays completed/total count, elapsed time, remaining time and expected finish time (UTC+8). ETA uses successful point timings; rapid failures do not imply the calculation is nearly complete.

## Resume after interruption

Use the exact original directory printed as `Results:`:

```powershell
$run = 'D:\LeLiu\data\tdbg_self_consistent\ABBA_theta1p35_N2_ndmap_<original_timestamp>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ndmap_windows.ps1 -Resume -Out $run -Workers 20
```

Replace `<original_timestamp>` with the actual folder name. Original Hamiltonian, angle, mesh, axes, Hartree and Stoner parameters are restored automatically. Verified completed points are skipped. Missing, failed, damaged or incomplete points are computed again. An interrupted point restarts from its beginning, rather than resuming individual SCF iterations.

Worker count and live-image interval may change on resume. Scientific parameters, flavor bounds and Python source must match the original scan; incompatible resume requests stop before replacing data. An OS lock prevents concurrent writers and releases after process death.

Each worker writes a flushed, atomic checkpoint **before** returning. A scan manifest, per-point checksums and selected-detail checksums protect recovery. Attempt IDs prevent workers left from a previous process from overwriting a resumed scan. Runs made before checkpoint support cannot recover ordinary points that existed only in RAM.

## Saved and live results

After each returned point, these files are atomically replaced:

- `nd_map.csv`, `nd_map.json`, `nd_map_scalars.npz`: full-size axes, scalar maps and validity masks.
- `scf/nd_map.csv`, `scf/nd_map.json`, `scf/nd_map_scalars.npz`: pure Hartree DOS, layer DOS, densities, polarization, potentials and convergence diagnostics.
- `run_summary.json`: timing, progress, failures and bound-limited Stoner counts.
- `nd_map_live.png`: Hartree/Stoner DOS and polarization preview, refreshed when points return and the 30-second interval has elapsed, plus final completion.

`-LivePlotIntervalSeconds 0` disables the live image; scalar data still update every point. Uncomputed cells have `status=pending`, NaN scalar values and false validity masks. They are never represented as zero DOS or zero polarization. Full per-observable figures are written at completion or handled interruption. Detailed band/DOS/layer-DOS/contour outputs are saved only at the selected 10 points. Full eigenvectors stay in RAM. Live snapshots overwrite their preceding versions, so a complete map is not accumulated for every point.

A valid Hartree stage is retained if Stoner fails, including when a resumed retry suffers process-transport failure before writing any replacement. Stoner DOS requires a physical common interacting chemical potential, independently of artificial filling-bound normal cones. If the user window clips the state and prevents a common chemical potential, `stoner_filling_bound_limited=true`; the reason is saved and spectral observables remain NaN. Hitting a bound alone does not invalidate a genuine common chemical potential.

## Direct Python

Existing explicit density inputs remain supported. Choose either `--n-*` or `--nu-*` scan inputs. The direct Python map driver retains D=-1.2...1.2/121 and the existing Hartree defaults; PowerShell explicitly supplies the server defaults above.

```powershell
.\.venv\Scripts\python.exe -u examples\run_nd_mapping_full_band_stoner.py --theta-deg 1.35 --cutoff 2 --nu-min=-6 --nu-max=6 --nu-count 61 --D-min-Vnm=-0.8 --D-max-Vnm=0.8 --D-count 41 --grid-n1 36 --grid-n2 36 --max-workers 20 --detail-count 10
```

Single point:

```powershell
.\.venv\Scripts\python.exe -u examples\run_full_band_stoner.py --theta-deg 1.35 --cutoff 2 --n-cm2 0 --D-Vnm 0 --grid-n1 36 --grid-n2 36 --kBT-meV 0.2 --eps-perp 3 --full-max-iter 100 --stoner-flavor-nu-min=-2 --stoner-flavor-nu-max=2 --out outputs\full_hartree_single
```


## Two-seed DE and compact outputs

The search now uses two independent 3D differential-evolution runs, exact
occupation-node refinement, common-mu and local-stability validation, then
energy comparison. The defaults are StonerDeMaxIter=150, StonerDePopsize=12,
StonerDeTol=1e-8, StonerMaxIter=200 and StonerSeed=0. The two seeds are Seed and
Seed+1. Legacy StonerRandomSeeds inputs are accepted but ignored by DE. This
stochastic search is not a mathematical global-minimum certificate.

All original physical defaults remain unchanged: gamma4=138 meV, Z=15 meV,
eps_perp=3 in this launcher, eps_perp=4 in standalone run_full_scf, and
D_sign=-1 (positive D points bottom to top).

Active spin, valley and spin-valley maps use *_polarization_active_norm in
[0,1]. Legacy *_polarization_norm columns remain labeled as ratios to absolute
net filling. The lower-bilayer DOS fraction is (DOS_L3+DOS_L4)/DOS_all.
Main detailed outputs save compressed spectra and compact block probabilities,
not complete eigenvectors. Detailed points include Origin TXT with separate
Long Name/Units/Comments rows, and colored band/contour PDF/PNG plots.

Each live update includes energy_gap_maps.npz for Hartree nu=0,+/-4 and
stoner_energy_gap_maps.npz for integer reference fillings -6...6. Each gap is
measured in the fixed spectrum at that point's converged Hartree potential and
exchange shifts; these are not independently self-consistent reference fillings.
Missing/invalid values are NaN, and an accepted band overlap gives zero.

## Stoner-only recomputation from saved Hartree

Keep the original Hartree inputs automatically and update the same folder:

```powershell
$run = 'D:\LeLiu\data\tdbg_self_consistent\ABBA_theta1p35_N2_ndmap_<actual_timestamp>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ndmap_windows.ps1 -ReuseScfRun $run -UpdateStonerInPlace -Workers 20
```

If that directory already contains a Stoner-only generation, start another:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ndmap_windows.ps1 -ReuseScfRun $run -RestartStonerInPlace -Workers 20 -StonerDeMaxIter 150 -StonerDePopsize 12 -StonerMaxIter 200
```

After interruption, continue the same generation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ndmap_windows.ps1 -Resume -Out $run -Workers 20
```

Replay uses zero Hartree iterations. Two full-valley diagonalizations at the
saved potential are still needed for Stoner tables, layer DOS and consistency
checks. The previous outputs are archived under _stoner_backups. The pure
Hartree map stays complete while pending Stoner cells are NaN. Archives retain
previous data and therefore add to total disk use. Only one writer may own a
result directory.

Ordinary resume rejects checkpoints with different calculation source. After
updating the solver, start a Stoner-only generation instead of mixing old and
new Stoner results. Every Hartree input and the saved Hamiltonian/density/
electrostatic code must still match; inconsistent saved references are rejected.

Origin: set leading # lines as the main header and the next three tabular rows
as Long Name, Units and Comments. C3 square and line files are plotting
interpolants; raw mesh spectra are also retained. If a valid grid has fewer than
two samples on either reciprocal axis, raw data remain available and the
interpolated contour is explicitly marked unavailable, not as a zero gap.
