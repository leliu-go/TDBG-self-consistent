import subprocess
import sys
import argparse


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
    assert "--n-min-cm2" in result.stdout
    assert "--D-min-Vnm" in result.stdout
    assert "--out" in result.stdout


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


def test_nd_mapping_susceptibility_can_build_direct_nd_grid(tmp_path):
    from examples.run_nd_mapping_stoner_susceptibility import build_parser, load_source_dataframe

    args = build_parser().parse_args(
        [
            "--out-csv",
            str(tmp_path / "chi.csv"),
            "--n-min-cm2=-1e12",
            "--n-max-cm2",
            "1e12",
            "--n-count",
            "3",
            "--D-min-Vnm=-0.2",
            "--D-max-Vnm",
            "0.2",
            "--D-count",
            "2",
        ]
    )

    source = load_source_dataframe(args)

    assert args.input_map is None
    assert list(source["n_index"]) == [0, 0, 1, 1, 2, 2]
    assert list(source["D_index"]) == [0, 1, 0, 1, 0, 1]
    assert list(source["n_cm2"]) == [-1e12, -1e12, 0.0, 0.0, 1e12, 1e12]
    assert list(source["D_Vnm"]) == [-0.2, 0.2, -0.2, 0.2, -0.2, 0.2]
    assert "nu_total" in source.columns


def test_nd_mapping_susceptibility_out_dir_sets_default_csv(tmp_path):
    from examples.run_nd_mapping_stoner_susceptibility import build_parser, resolve_output_layout

    out_dir = tmp_path / "finite_q_run"
    args = build_parser().parse_args(
        [
            "--out",
            str(out_dir),
            "--n-min-cm2=0",
            "--n-max-cm2",
            "0",
            "--n-count",
            "1",
            "--D-min-Vnm=0",
            "--D-max-Vnm",
            "0",
            "--D-count",
            "1",
        ]
    )

    resolved_out, resolved_csv = resolve_output_layout(args, stamp="fixed")

    assert resolved_out == out_dir
    assert resolved_csv == out_dir / "finite_q_susceptibility.csv"


def test_susceptibility_boolean_flags_work_without_boolean_optional_action(monkeypatch, tmp_path):
    from examples.run_nd_mapping_stoner_susceptibility import build_parser as build_nd_parser
    from examples.run_stoner_susceptibility_single import build_parser as build_single_parser

    monkeypatch.delattr(argparse, "BooleanOptionalAction", raising=False)

    nd_args = build_nd_parser().parse_args(
        [
            "--out-csv",
            str(tmp_path / "chi.csv"),
            "--n-min-cm2=0",
            "--n-max-cm2",
            "0",
            "--n-count",
            "1",
            "--D-min-Vnm=0",
            "--D-max-Vnm",
            "0",
            "--D-count",
            "1",
            "--no-also-run-hund-factor2",
            "--no-legacy-diagnostics",
        ]
    )
    single_args = build_single_parser().parse_args(
        [
            "--n-cm2",
            "0",
            "--D-Vnm",
            "0",
            "--out",
            str(tmp_path / "single"),
            "--no-also-run-hund-factor2",
            "--no-legacy-diagnostics",
        ]
    )

    assert nd_args.also_run_hund_factor2 is False
    assert nd_args.legacy_diagnostics is False
    assert single_args.also_run_hund_factor2 is False
    assert single_args.legacy_diagnostics is False
