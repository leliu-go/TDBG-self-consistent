[CmdletBinding()]
param(
    [string]$PythonBin = '',
    [string]$Out = '',
    [switch]$Resume,
    [string]$ReuseScfRun = '',
    [switch]$UpdateStonerInPlace,
    [switch]$RestartStonerInPlace,
    [string]$DataRoot = 'D:\LeLiu\data',
    [ValidateRange(1, 61)][int]$Workers = 20,
    [ValidateRange(0.001, 20)][double]$ThetaDeg = 1.35,
    [ValidateRange(0, 100)][int]$Cutoff = 2,
    [ValidateRange(2, 10000)][int]$GridN1 = 36,
    [ValidateRange(2, 10000)][int]$GridN2 = 36,
    [double]$NuMin = -6,
    [double]$NuMax = 6,
    [ValidateRange(1, 100000)][int]$NuCount = 61,
    [double]$DMinVnm = -0.8,
    [double]$DMaxVnm = 0.8,
    [ValidateRange(1, 100000)][int]$DCount = 41,
    [ValidateRange(0, 100000)][int]$DetailCount = 10,
    [ValidateRange(1, 100000)][int]$FullMaxIter = 100,
    [double]$KBtMeV = 0.2,
    [double]$EpsPerp = 3,
    [ValidateSet(-1, 1)][int]$DSign = -1,
    [double]$ZMeV = 15,
    [double]$U0MeVA2 = 79000,
    [double]$JHMeVA2 = 24000,
    [ValidateRange(0, 100000)][int]$StonerRandomSeeds = 20,
    [ValidateRange(1, 100000)][int]$StonerDeMaxIter = 150,
    [ValidateRange(1, 100000)][int]$StonerDePopsize = 12,
    [double]$StonerDeTol = 1e-8,
    [ValidateRange(0, 100000)][int]$StonerMaxIter = 200,
    [ValidateRange(0, 2147483647)][int]$StonerSeed = 0,
    [double]$StonerFlavorNuMin = -2,
    [double]$StonerFlavorNuMax = 2,
    [ValidateRange(2, 100000)][int]$DosBins = 240,
    [ValidateRange(2, 100000)][int]$PointsPerSegment = 60,
    [ValidateRange(0, 86400)][double]$LivePlotIntervalSeconds = 30,
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonBin)) { $PythonBin = Join-Path $repoRoot '.venv\Scripts\python.exe' }
$numberCulture = [System.Globalization.CultureInfo]::InvariantCulture
function Format-Number([double]$Value) { $Value.ToString($numberCulture) }
if (-not [string]::IsNullOrWhiteSpace($Out)) {
    if (-not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $repoRoot $Out }
    $Out = [System.IO.Path]::GetFullPath($Out)
}
$options = [ordered]@{
    Workers = '--max-workers'; ThetaDeg = '--theta-deg'; Cutoff = '--cutoff'
    GridN1 = '--grid-n1'; GridN2 = '--grid-n2'; NuMin = '--nu-min'; NuMax = '--nu-max'; NuCount = '--nu-count'
    DMinVnm = '--D-min-Vnm'; DMaxVnm = '--D-max-Vnm'; DCount = '--D-count'; DetailCount = '--detail-count'
    FullMaxIter = '--full-max-iter'; KBtMeV = '--kBT-meV'; EpsPerp = '--eps-perp'; DSign = '--D-sign'
    ZMeV = '--Z'; U0MeVA2 = '--u0-meV-A2'; JHMeVA2 = '--JH-meV-A2'; StonerRandomSeeds = '--stoner-random-seeds'
    StonerDeMaxIter = '--stoner-de-max-iter'; StonerDePopsize = '--stoner-de-popsize'; StonerDeTol = '--stoner-de-tol'
    StonerMaxIter = '--stoner-max-iter'; StonerSeed = '--stoner-seed'
    StonerFlavorNuMin = '--stoner-flavor-nu-min'; StonerFlavorNuMax = '--stoner-flavor-nu-max'
    DosBins = '--dos-bins'; PointsPerSegment = '--points-per-segment'; LivePlotIntervalSeconds = '--live-plot-interval-s'
}
$savedNames = [ordered]@{
    Workers = 'max_workers'; ThetaDeg = 'theta_deg'; Cutoff = 'cutoff'; GridN1 = 'grid_n1'; GridN2 = 'grid_n2'
    NuMin = 'nu_min'; NuMax = 'nu_max'; NuCount = 'nu_count'; DMinVnm = 'D_min_Vnm'; DMaxVnm = 'D_max_Vnm'
    DCount = 'D_count'; DetailCount = 'detail_count'; FullMaxIter = 'full_max_iter'; KBtMeV = 'kBT_meV'
    EpsPerp = 'eps_perp'; DSign = 'D_sign'; ZMeV = 'Z'; U0MeVA2 = 'u0_meV_A2'; JHMeVA2 = 'JH_meV_A2'
    StonerDeMaxIter = 'stoner_de_max_iter'; StonerDePopsize = 'stoner_de_popsize'; StonerDeTol = 'stoner_de_tol'
    StonerMaxIter = 'stoner_max_iter'; StonerSeed = 'stoner_seed'
    StonerRandomSeeds = 'stoner_random_seeds'; StonerFlavorNuMin = 'stoner_flavor_nu_min'
    StonerFlavorNuMax = 'stoner_flavor_nu_max'; DosBins = 'dos_bins'; PointsPerSegment = 'points_per_segment'
    LivePlotIntervalSeconds = 'live_plot_interval_s'
}
$isReplay = -not [string]::IsNullOrWhiteSpace($ReuseScfRun)
if (($UpdateStonerInPlace -or $RestartStonerInPlace) -and -not $isReplay -and -not $Resume) {
    throw '-UpdateStonerInPlace / -RestartStonerInPlace requires -ReuseScfRun.'
}
if ($RestartStonerInPlace -and $Resume) { throw 'Use -Resume without -RestartStonerInPlace to continue a generation.' }
if ($isReplay -and -not $Resume) {
    if (-not [System.IO.Path]::IsPathRooted($ReuseScfRun)) { $ReuseScfRun = Join-Path $repoRoot $ReuseScfRun }
    $ReuseScfRun = [System.IO.Path]::GetFullPath($ReuseScfRun)
    if ($RestartStonerInPlace) { $UpdateStonerInPlace = $true }
    if ($UpdateStonerInPlace) {
        if ([string]::IsNullOrWhiteSpace($Out)) { $Out = $ReuseScfRun }
        elseif ($Out -ne $ReuseScfRun) { throw 'In-place Stoner update requires the source directory as -Out.' }
    }
    $savedPath = Join-Path $ReuseScfRun 'config_used.json'
    if (-not (Test-Path -LiteralPath $savedPath)) { throw 'Saved Hartree source lacks config_used.json.' }
    $savedConfig = Get-Content -LiteralPath $savedPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $savedArguments = $savedConfig.args
    foreach ($name in $savedNames.Keys) {
        $savedValue = $savedArguments.($savedNames[$name])
        if (-not $PSBoundParameters.ContainsKey($name) -and $null -ne $savedValue) {
            if ($name -eq 'Workers' -and $savedValue -eq 'auto') {
                $savedValue = [math]::Min(61, [math]::Max(1, [Environment]::ProcessorCount - 2))
            }
            Set-Variable -Name $name -Value $savedValue -Scope Local
        }
    }
}
if ($Resume) {
    if ([string]::IsNullOrWhiteSpace($Out)) { throw '-Resume requires -Out pointing to the original run directory.' }
    $savedPath = Join-Path $Out 'config_used.json'
    if (Test-Path -LiteralPath $savedPath) {
        $savedConfig = Get-Content -LiteralPath $savedPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $savedArguments = $savedConfig.args
    } else {
        $savedPath = Join-Path $Out 'checkpoints\scan_manifest.json'
        if (-not (Test-Path -LiteralPath $savedPath)) { throw 'No checkpointed run exists in this directory.' }
        $savedConfig = Get-Content -LiteralPath $savedPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $savedArguments = $savedConfig.saved_args
    }
    foreach ($name in $savedNames.Keys) {
        $savedValue = $savedArguments.($savedNames[$name])
        if (-not $PSBoundParameters.ContainsKey($name) -and $null -ne $savedValue) {
            if ($name -eq 'Workers' -and $savedValue -eq 'auto') {
                $savedValue = [math]::Min(61, [math]::Max(1, [Environment]::ProcessorCount - 2))
            }
            Set-Variable -Name $name -Value $savedValue -Scope Local
        }
    }
}
if ($NuMin -gt $NuMax -or $DMinVnm -gt $DMaxVnm) { throw 'Minimum must be <= maximum.' }
if ([double]::IsNaN($StonerFlavorNuMin) -or [double]::IsInfinity($StonerFlavorNuMin) -or
    [double]::IsNaN($StonerFlavorNuMax) -or [double]::IsInfinity($StonerFlavorNuMax) -or
    $StonerFlavorNuMin -ge $StonerFlavorNuMax) { throw 'Flavor filling bounds must be finite and ordered.' }

[string[]]$pythonArguments = @('-u', 'examples\run_nd_mapping_full_band_stoner.py')
foreach ($name in $options.Keys) {
    if (($Resume -or $isReplay) -and -not $PSBoundParameters.ContainsKey($name)) { continue }
    $value = Get-Variable -Name $name -ValueOnly
    $argumentText = if ($value -is [double]) { Format-Number $value } else { [string]$value }
    if ($name -eq 'Workers') { $pythonArguments += @('--max-workers', $argumentText) }
    else { $pythonArguments += ($options[$name] + '=' + $argumentText) }
}
if ($Resume) { $pythonArguments += '--resume' }
if ($isReplay) { $pythonArguments += @('--reuse-scf-run', $ReuseScfRun) }
if ($UpdateStonerInPlace) { $pythonArguments += '--update-stoner-in-place' }
if ($RestartStonerInPlace) { $pythonArguments += '--restart-stoner-in-place' }
if (-not [string]::IsNullOrWhiteSpace($Out)) { $pythonArguments += @('--out', $Out) }
if ((-not $Resume -and -not $isReplay) -or $PSBoundParameters.ContainsKey('DataRoot')) { $pythonArguments += @('--data-root', $DataRoot) }

$threadEnvironment = [ordered]@{
    OMP_NUM_THREADS = '1'; OPENBLAS_NUM_THREADS = '1'; MKL_NUM_THREADS = '1'; NUMEXPR_NUM_THREADS = '1'
}
$dimension = 8 * [math]::Pow(2 * $Cutoff + 1, 2)
$eigenvectorsGB = 2 * $GridN1 * $GridN2 * $dimension * $dimension * 16 / 1e9
$totalPoints = $NuCount * $DCount
if (($Resume -or $isReplay) -and $null -ne $savedConfig.n_values_cm2 -and $null -ne $savedConfig.D_values_Vnm) {
    $totalPoints = $savedConfig.n_values_cm2.Count * $savedConfig.D_values_Vnm.Count
}
if ($Preview) {
    [ordered]@{
        python_executable = $PythonBin; python_arguments = $pythonArguments; thread_environment = $threadEnvironment
        output_layout = [ordered]@{
            data_root = $DataRoot; automatic_parent = (Join-Path $DataRoot 'tdbg_self_consistent')
            explicit_out = $Out; unique_run_folder = (-not ($Resume -or $UpdateStonerInPlace -or $RestartStonerInPlace)); resume = [bool]$Resume
            reuse_scf_run = $ReuseScfRun; update_stoner_in_place = [bool]($UpdateStonerInPlace -or $RestartStonerInPlace)
        }
        resource_estimates = [ordered]@{
            hamiltonian_dimension = $dimension; total_points = $totalPoints
            two_valley_eigenvectors_GB_per_worker = $eigenvectorsGB; all_workers_eigenvectors_GB = $Workers * $eigenvectorsGB
            detail_points = [math]::Min($totalPoints, $DetailCount); full_eigenvectors_saved = $false
            runtime_estimate = 'Calibrated at startup; elapsed time and ETA update from completed points.'
        }
    } | ConvertTo-Json -Depth 6
    exit 0
}
$pythonCommand = Get-Command -Name $PythonBin -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) { throw "Python not found at $PythonBin. Run scripts\setup_venv_windows.ps1 or pass -PythonBin." }
foreach ($entry in $threadEnvironment.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
$env:MPLBACKEND = 'Agg'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONFAULTHANDLER = '1'
Push-Location -LiteralPath $repoRoot
try {
    & $pythonCommand.Source @pythonArguments
    $runExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $runExitCode
