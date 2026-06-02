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
    assert "--dos-sigma-meV" in result.stdout
    assert "--points-per-segment" in result.stdout
    assert "--contour-band-count" in result.stdout
    assert "--no-stoner-detail-outputs" in result.stdout
    assert "--jdos-sigma-meV" in result.stdout
    assert "--no-jdos" in result.stdout


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
    assert "--no-plots" in result.stdout


def test_plot_finite_q_susceptibility_help():
    result = subprocess.run(
        [sys.executable, "examples/plot_finite_q_susceptibility.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--run-dir" in result.stdout
    assert "--csv" in result.stdout
    assert "--out-dir" in result.stdout


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


def test_nd_mapping_susceptibility_progress_message_includes_pending_and_total_counts():
    from examples.run_nd_mapping_stoner_susceptibility import format_progress_message

    message = format_progress_message(
        completed_this_run=2,
        pending_total=10,
        recorded_total=5,
        todo_total=14,
        rec={"chi_status": "ok", "n_cm2": 1.7e12, "D_Vnm": 0.52, "chi_runtime_s": 12.3},
    )

    assert "[2/10]" in message
    assert "[recorded 5/14]" in message
    assert "ok" in message
    assert "n=1.700000e+12" in message
    assert "D=0.5200" in message


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


def test_plot_finite_q_susceptibility_generates_figures(tmp_path):
    import pandas as pd

    run_dir = tmp_path / "finite_q_run"
    run_dir.mkdir()
    csv_path = run_dir / "finite_q_susceptibility.csv"
    pd.DataFrame(
        [
            {
                "n_index": 0,
                "D_index": 0,
                "n_cm2": -1e12,
                "D_Vnm": -0.2,
                "chi_status": "ok",
                "finite_q_ratio_su4_diag": 1.1,
                "finite_q_wins_su4_diag": True,
                "gamma_lambda_su4_diag_selected": 0.8,
                "qstar_nonzero_lambda_su4_diag_selected": 1.2,
                "qstar_su4_diag_qnorm_Ainv": 0.01,
            },
            {
                "n_index": 0,
                "D_index": 1,
                "n_cm2": -1e12,
                "D_Vnm": 0.2,
                "chi_status": "ok",
                "finite_q_ratio_su4_diag": 0.9,
                "finite_q_wins_su4_diag": False,
                "gamma_lambda_su4_diag_selected": 1.0,
                "qstar_nonzero_lambda_su4_diag_selected": 0.9,
                "qstar_su4_diag_qnorm_Ainv": 0.02,
            },
            {
                "n_index": 1,
                "D_index": 0,
                "n_cm2": 1e12,
                "D_Vnm": -0.2,
                "chi_status": "ok",
                "finite_q_ratio_su4_diag": 1.3,
                "finite_q_wins_su4_diag": True,
                "gamma_lambda_su4_diag_selected": 1.1,
                "qstar_nonzero_lambda_su4_diag_selected": 1.43,
                "qstar_su4_diag_qnorm_Ainv": 0.03,
            },
            {
                "n_index": 1,
                "D_index": 1,
                "n_cm2": 1e12,
                "D_Vnm": 0.2,
                "chi_status": "ok",
                "finite_q_ratio_su4_diag": 0.8,
                "finite_q_wins_su4_diag": False,
                "gamma_lambda_su4_diag_selected": 1.2,
                "qstar_nonzero_lambda_su4_diag_selected": 0.96,
                "qstar_su4_diag_qnorm_Ainv": 0.04,
            },
        ]
    ).to_csv(csv_path, index=False)

    result = subprocess.run(
        [sys.executable, "examples/plot_finite_q_susceptibility.py", "--run-dir", str(run_dir)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (run_dir / "figures" / "nd_map_finite_q_ratio_su4_diag.png").exists()
    assert (run_dir / "figures" / "nd_map_finite_q_wins_su4_diag.png").exists()
    assert (run_dir / "figures" / "nd_map_lambda_Qstar_su4_diag.png").exists()
    assert (run_dir / "figures" / "nd_map_delta_lambda_Qstar_Gamma_su4_diag.png").exists()
    assert (run_dir / "figures" / "nd_map_log10_finite_q_ratio_su4_diag_clipped.png").exists()
    assert (run_dir / "figures" / "nd_map_robust_finite_q_su4_diag.png").exists()
    assert (run_dir / "plot_summary.json").exists()
