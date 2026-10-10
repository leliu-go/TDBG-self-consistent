[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Run,
    [ValidateRange(1,61)][int]$Workers = 5,
    [ValidateRange(3,10001)][int]$SquarePoints = 145,
    [ValidateRange(3,100001)][int]$DosPoints = 1001,
    [string]$PythonBin = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonBin)) {
    $PythonBin = Join-Path $repoRoot '.venv\Scripts\python.exe'
}
$pythonCommand = Get-Command -Name $PythonBin -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    throw "Python was not found at $PythonBin. Run scripts\setup_venv_windows.ps1 first."
}
$Run = (Resolve-Path -LiteralPath $Run).Path
foreach ($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')) {
    [Environment]::SetEnvironmentVariable($name,'1','Process')
}
$env:MPLBACKEND = 'Agg'
$env:PYTHONUNBUFFERED = '1'
Push-Location -LiteralPath $repoRoot
try {
    & $pythonCommand.Source -m tdbg_scf.origin_export --run $Run --workers $Workers --square-points $SquarePoints --dos-points $DosPoints
    $runExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runExitCode
