#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdbg_scf.susceptibility import plot_finite_q_outputs


def parse_plot_keys(text: str | None) -> list[str] | None:
    if text is None or text.strip() == "":
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot finite-Q susceptibility n-D maps from an existing CSV.")
    ap.add_argument("--run-dir", type=Path, default=None, help="Directory containing finite_q_susceptibility.csv.")
    ap.add_argument("--csv", type=Path, default=None, help="Explicit finite_q_susceptibility.csv path.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Figure directory. Defaults to <run-dir>/figures.")
    ap.add_argument("--plot-keys", type=str, default=None, help="Comma-separated column names to plot. Defaults to standard finite-Q maps.")
    return ap


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.csv is None and args.run_dir is None:
        raise ValueError("provide --run-dir or --csv")
    csv_path = Path(args.csv) if args.csv is not None else Path(args.run_dir) / "finite_q_susceptibility.csv"
    out_dir = Path(args.out_dir) if args.out_dir is not None else csv_path.parent / "figures"
    return csv_path, out_dir


def main() -> None:
    args = build_parser().parse_args()
    csv_path, out_dir = resolve_paths(args)
    summary = plot_finite_q_outputs(csv_path, out_dir=out_dir, plot_keys=parse_plot_keys(args.plot_keys))
    summary_path = csv_path.parent / "plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
