import json
import subprocess
import sys


def test_correlated_cli_help_lists_methods():
    result = subprocess.run(
        [sys.executable, "-m", "tdbg_scf.correlated_cli", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "stoner" in result.stdout
    assert "projected-hf" in result.stdout
    assert "both" in result.stdout


def test_correlated_config_json_load_and_override(tmp_path):
    from tdbg_scf.correlated_config import load_correlated_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "method": "stoner",
                "shared": {"nu_grid": [1.0], "D_Vnm_grid": [0.0], "output_dir": str(tmp_path)},
                "stoner": {"u0_meV_A2": 1.0},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_correlated_config(config_path, nu_override=2.0, D_override=0.3)

    assert cfg["shared"]["nu_grid"] == [2.0]
    assert cfg["shared"]["D_Vnm_grid"] == [0.3]
