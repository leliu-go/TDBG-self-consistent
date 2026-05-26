import subprocess
import sys

import pytest


def test_full_band_stoner_defaults_use_reference_hund_parameters():
    import argparse

    from examples.run_full_band_stoner import add_stoner_args

    parser = argparse.ArgumentParser()
    add_stoner_args(parser)
    args = parser.parse_args([])

    assert args.u0_meV_A2 == pytest.approx(7.9e4)
    assert args.JH_meV_A2 == pytest.approx(2.4e4)


def test_full_band_dos_summary_recomputes_full_mu_from_full_bands():
    import numpy as np

    from examples.run_full_band_stoner import full_band_dos_summary

    evals_K = np.array([[-1.0, 1.0]])
    evals_Kp = np.array([[-1.0, 1.0]])
    weights = np.array([1.0])

    summary = full_band_dos_summary(
        evals_K,
        evals_Kp,
        weights,
        n_cm2=0.0,
        projected_mu_meV=10.0,
        sigma_meV=1.0,
        kBT_meV=0.2,
    )

    assert summary["full_mu_meV"] == pytest.approx(0.0, abs=1e-8)
    assert summary["dos_mu_full_at_full_mu"] > summary["dos_mu_full_at_projected_mu"]


def test_single_point_full_band_stoner_help():
    result = subprocess.run(
        [sys.executable, "examples/run_full_band_stoner.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--scf-source" in result.stdout
    assert "--n-active" in result.stdout
    assert "--u0-meV-A2" in result.stdout
    assert "--D-Vnm" in result.stdout


def test_nd_mapping_full_band_stoner_help():
    result = subprocess.run(
        [sys.executable, "examples/run_nd_mapping_full_band_stoner.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--scf-source" in result.stdout
    assert "--n-count" in result.stdout
    assert "--max-workers" in result.stdout
