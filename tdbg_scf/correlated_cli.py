"""Command-line entry point for independent correlated solvers."""
from __future__ import annotations

import argparse

from .correlated_config import load_correlated_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stoner, projected-hf, or both correlated solvers.")
    parser.add_argument("--method", choices=("stoner", "projected-hf", "both"), default=None)
    parser.add_argument("--config", required=False)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--D", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is None:
        if args.method is None:
            return 0
        parser.error("--config is required when running a method")

    cfg = load_correlated_config(args.config, nu_override=args.nu, D_override=args.D)
    method = args.method or cfg.get("method", "stoner")
    if method == "stoner":
        from .stoner.scan import run_stoner_scan

        run_stoner_scan(cfg)
    elif method == "projected-hf":
        from .projected_hf.scan import run_projected_hf_scan

        run_projected_hf_scan(cfg)
    elif method == "both":
        from .projected_hf.scan import run_projected_hf_scan
        from .stoner.scan import run_stoner_scan

        run_stoner_scan(cfg)
        run_projected_hf_scan(cfg)
    else:
        parser.error(f"unknown method: {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
