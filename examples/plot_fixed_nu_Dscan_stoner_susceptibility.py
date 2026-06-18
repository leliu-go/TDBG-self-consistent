#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
for _path in (SCRIPT_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tdbg_scf.susceptibility import plot_fixed_nu_Dscan_overview, plot_representative_qmaps
from tdbg_scf.susceptibility.dscan import parse_float_list


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate paper-style figures from a fixed-nu D-scan finite-Q run.")
    ap.add_argument("--scan-csv", type=Path, required=True)
    ap.add_argument("--points-root", type=Path, default=None)
    ap.add_argument("--representative-D", type=str, default=None)
    ap.add_argument("--plot-key", type=str, default="lambda_su4_diag_soft")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [plot_fixed_nu_Dscan_overview(args.scan_csv, args.out)]
    reps = parse_float_list(args.representative_D)
    if args.points_root is not None and reps:
        summaries.append(plot_representative_qmaps(args.points_root, reps, args.out, key=args.plot_key))
    payload = {"figures": summaries}
    (args.out / "plot_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

