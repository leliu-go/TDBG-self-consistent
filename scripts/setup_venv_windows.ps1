[CmdletBinding()]
param(
    [string]$PythonBin = 'C:\Users\liule\AppData\Local\Programs\Python\Python38\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$requirements = Join-Path $repoRoot 'requirements-windows-py38.txt'

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $venvDir) {
        throw "Existing .venv is incomplete. Check $venvDir before recreating it."
    }
    $basePython = Get-Command -Name $PythonBin -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $basePython) {
        throw "Python was not found at $PythonBin. Pass -PythonBin with your existing python.exe path."
    }
    @'
import struct, sys
assert (3, 8) <= sys.version_info < (3, 12), "Python 3.8 through 3.11 required for the pinned dependencies"
assert struct.calcsize("P") == 8, "64-bit Python required"
'@ | & $basePython.Source -
    if ($LASTEXITCODE -ne 0) { throw 'The base Python version or architecture is unsupported.' }
    Write-Host "Creating virtual environment: $venvDir"
    & $basePython.Source -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

Write-Host "Installing pinned dependencies into: $venvPython"
& $venvPython -m pip install --disable-pip-version-check --only-binary=:all: -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the pip error above and rerun this script.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment dependency check failed.' }

Push-Location -LiteralPath $repoRoot
try {
    @'
import struct, sys, numpy, scipy, matplotlib, pandas, tdbg_scf
assert sys.prefix != sys.base_prefix, "Not a virtual environment"
assert (3, 8) <= sys.version_info < (3, 12)
assert struct.calcsize("P") == 8
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("NumPy/SciPy/Matplotlib/pandas:", numpy.__version__, scipy.__version__, matplotlib.__version__, pandas.__version__)
'@ | & $venvPython -
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment import check failed.' }
} finally {
    Pop-Location
}
Write-Host 'Environment ready. run_ndmap_windows.ps1 now uses this .venv automatically.'
