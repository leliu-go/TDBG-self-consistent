"""Native Windows argument forwarding and checkpoint-aware deployment."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
pytestmark = pytest.mark.skipif(not PS.exists(), reason="Native PowerShell required")


def preview(*args):
    result = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(ROOT / "scripts/run_ndmap_windows.ps1"), *args, "-Preview"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_windows_launcher_uses_venv_and_signed_bounds():
    result = preview("-Workers", "20")
    assert result["python_executable"].endswith(r".venv\Scripts\python.exe")
    argv = result["python_arguments"]
    assert "--stoner-flavor-nu-min=-2" in argv and "--stoner-flavor-nu-max=2" in argv
    assert argv[argv.index("--max-workers") + 1] == "20"
    assert result["resource_estimates"]["hamiltonian_dimension"] == 200
    assert result["output_layout"]["automatic_parent"] == r"D:\LeLiu\data\tdbg_self_consistent"
    assert set(result["thread_environment"].values()) == {"1"}


def test_windows_resume_does_not_inject_default_physics(tmp_path):
    (tmp_path / "config_used.json").write_text(json.dumps({"args": {
        "theta_deg": 1.15, "cutoff": 3, "grid_n1": 24, "grid_n2": 30,
        "max_workers": "auto", "stoner_flavor_nu_min": -1.5, "stoner_flavor_nu_max": 2.5}}))
    result = preview("-Resume", "-Out", str(tmp_path), "-Workers", "2")
    argv = result["python_arguments"]
    assert "--resume" in argv and "--theta-deg" not in argv
    assert not any(arg.startswith("--stoner-flavor-nu") for arg in argv)
    assert result["resource_estimates"]["hamiltonian_dimension"] == 392
    assert argv[argv.index("--max-workers") + 1] == "2"


def test_windows_launcher_preserves_custom_fillings():
    argv = preview("-StonerFlavorNuMin", "-1.5", "-StonerFlavorNuMax", "2.5")["python_arguments"]
    assert "--stoner-flavor-nu-min=-1.5" in argv and "--stoner-flavor-nu-max=2.5" in argv


def test_windows_launcher_keeps_tdbg_physical_defaults():
    argv = preview()["python_arguments"]
    assert "--eps-perp=3" in argv
    assert "--Z=15" in argv and "--theta-deg=1.35" in argv
    assert "--u0-meV-A2=79000" in argv and "--JH-meV-A2=24000" in argv


def test_windows_venv_setup_uses_correct_server_python_path():
    setup = ROOT / "scripts/setup_venv_windows.ps1"
    assert setup.is_file()
    text = setup.read_text()
    assert r"C:\Users\liule\AppData\Local\Programs\Python\Python38\python.exe" in text
    assert "tdbg_scf" in text and "tbtg_scf" not in text
    assert (ROOT / "requirements-windows-py38.txt").is_file()


def test_actual_powershell_run_and_resume_keep_completed_checkpoint(tmp_path):
    script = ROOT / "scripts/run_ndmap_windows.ps1"
    common = [str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    argv = [*common, "-PythonBin", sys.executable, "-Out", str(tmp_path), "-Workers", "1",
            "-Cutoff", "0", "-ZMeV", "0", "-GridN1", "2", "-GridN2", "2", "-KBtMeV", "2",
            "-NuMin", "0", "-NuMax", "0", "-NuCount", "1", "-DMinVnm", "0.02", "-DMaxVnm", "0.02",
            "-DCount", "1", "-DetailCount", "0", "-DosBins", "12", "-PointsPerSegment", "2",
            "-StonerRandomSeeds", "0", "-LivePlotIntervalSeconds", "0"]
    first = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "ETA initial:" in first.stdout and "elapsed" in first.stdout
    point = next((tmp_path / "checkpoints/points").glob("*.json"))
    contents, timestamp = point.read_bytes(), point.stat().st_mtime_ns
    second = subprocess.run([*common, "-PythonBin", sys.executable, "-Out", str(tmp_path), "-Resume",
                             "-Workers", "2"], capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "1 verified points reused; 0 points to compute" in second.stdout
    assert point.read_bytes() == contents and point.stat().st_mtime_ns == timestamp


def test_windows_launcher_forwards_de_controls():
    argv=preview('-StonerDeMaxIter','17','-StonerDePopsize','7','-StonerDeTol','1e-7',
                 '-StonerMaxIter','23','-StonerSeed','11')['python_arguments']
    assert '--stoner-de-max-iter=17' in argv and '--stoner-de-popsize=7' in argv
    assert '--stoner-max-iter=23' in argv and '--stoner-seed=11' in argv


def test_windows_replay_does_not_inject_default_physics(tmp_path):
    (tmp_path/'config_used.json').write_text(json.dumps({'args':{
        'theta_deg':1.15,'eps_perp':7.,'Z':9.,'cutoff':3,'grid_n1':24,'grid_n2':30}}))
    result=preview('-ReuseScfRun',str(tmp_path),'-RestartStonerInPlace','-Workers','2')
    argv=result['python_arguments']
    assert argv[argv.index('--reuse-scf-run')+1] == str(tmp_path)
    assert '--restart-stoner-in-place' in argv
    assert not any(arg.startswith(('--theta-deg','--eps-perp','--Z=')) for arg in argv)
    assert result['resource_estimates']['hamiltonian_dimension']==392
