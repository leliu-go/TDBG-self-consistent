import subprocess
import sys


def test_single_point_susceptibility_help():
    result = subprocess.run(
        [sys.executable, "examples/run_stoner_susceptibility_single.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--n-cm2" in result.stdout
    assert "--D-Vnm" in result.stdout
    assert "--q-mode" in result.stdout
    assert "--main-vertex-model" in result.stdout
    assert "--occupation-mode" in result.stdout
    assert "--spin-flip-mode" in result.stdout
    assert "--scf-source" in result.stdout
    assert "--n-active" in result.stdout
    assert "--selection" in result.stdout
    assert "--projected-reference-mode" in result.stdout


def test_nd_mapping_susceptibility_help():
    result = subprocess.run(
        [sys.executable, "examples/run_nd_mapping_stoner_susceptibility.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--input-map" in result.stdout
    assert "--nu-min" in result.stdout
    assert "--q-stride" in result.stdout
    assert "--max-workers" in result.stdout
    assert "--main-vertex-model" in result.stdout
    assert "--occupation-mode" in result.stdout
    assert "--scf-source" in result.stdout
    assert "--n-active" in result.stdout
    assert "--selection" in result.stdout
    assert "--projected-reference-mode" in result.stdout


def test_nd_mapping_susceptibility_resolves_worker_count():
    from examples.run_nd_mapping_stoner_susceptibility import resolve_workers

    assert resolve_workers("1") == 1
    assert resolve_workers("auto") >= 1


def test_single_point_susceptibility_defaults_to_projected_scf_flow(tmp_path):
    from examples.run_stoner_susceptibility_single import build_parser

    args = build_parser().parse_args(
        [
            "--n-cm2",
            "2e12",
            "--D-Vnm",
            "0.5",
            "--out",
            str(tmp_path / "chi"),
        ]
    )

    assert args.scf_source == "projected_8band"
    assert args.scf_valley == 1
    assert args.n_active == 8
    assert args.selection == "overlap"
    assert args.projected_reference_mode == "uploaded_D"
    assert args.kBT_meV == 0.2
    assert args.eps_perp == 3.0


def test_single_point_susceptibility_keeps_legacy_scf_iteration_alias_separate(tmp_path):
    from examples.run_stoner_susceptibility_single import build_parser

    args = build_parser().parse_args(
        [
            "--n-cm2",
            "2e12",
            "--D-Vnm",
            "0.5",
            "--out",
            str(tmp_path / "chi"),
            "--scf-source",
            "full",
            "--scf-max-iter",
            "17",
            "--scf-tol-meV",
            "2e-3",
        ]
    )

    assert args.legacy_scf_max_iter == 17
    assert args.legacy_scf_tol_meV == 2e-3
    assert args.projected_max_iter == 100
    assert args.projected_tol_meV == 1e-4
