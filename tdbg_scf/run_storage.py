"""Shared result layout, provenance and compact-output resource estimates."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
from .atomic_io import atomic_json

DEFAULT_DATA_ROOT = Path(r"D:\LeLiu\data")
SGT = timezone(timedelta(hours=8), name="SGT")

ACTIVE_CARRIER_FIELDS = tuple("active_" + kind + "_nu_" + flavor
                              for kind in ("electron", "hole")
                              for flavor in ("K_up", "Kp_up", "K_down", "Kp_down"))


def active_carrier_scalar_fields(record=None):
    """Keep eight compact central-band populations in every durable point row."""
    if record is None:
        return dict.fromkeys(ACTIVE_CARRIER_FIELDS, float("nan"))
    counts = [np.asarray(record["active_" + kind + "_nu_f"], dtype=float)
              for kind in ("electron", "hole")]
    if any(value.shape != (4,) or not np.isfinite(value).all() or
           np.any(value < 0.) or np.any(value > 1. + 1e-8) for value in counts):
        raise ValueError("Need four finite central electron and hole populations in [0,1]")
    return dict(zip(ACTIVE_CARRIER_FIELDS, map(float, np.concatenate(counts))))


def now_sgt():
    return datetime.now(SGT)


def add_output_args(parser):
    parser.add_argument("--out", default=None, help="Exact result directory; overrides the automatic run layout.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Data root for the TDBG family and unique run folders.")


def default_output_dir(theta_deg, cutoff, workflow, data_root=None, *, now=None):
    root = Path(DEFAULT_DATA_ROOT if data_root is None else data_root).expanduser()
    angle = format(float(theta_deg), ".12g").replace("-", "m").replace(".", "p")
    stamp = (now or now_sgt()).strftime("%Y%m%d_%H%M%S_%f")
    return root / "tdbg_self_consistent" / f"ABBA_theta{angle}_N{cutoff}_{workflow}_{stamp}"


def create_output_dir(args, workflow):
    if getattr(args, "out", None):
        out = Path(args.out).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = default_output_dir(args.theta_deg, args.cutoff, workflow,
                                 getattr(args, "data_root", DEFAULT_DATA_ROOT)).resolve()
        while True:
            try:
                out.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                out = out.with_name(out.name + "_" + uuid.uuid4().hex[:8])
    args.out = str(out)
    return out


def detail_indices(n_values, d_values, count):
    """Deterministic farthest-point sampling in normalized physical n-D coordinates."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("detail count must be a nonnegative integer")
    n, d = np.asarray(n_values, dtype=float), np.asarray(d_values, dtype=float)
    if n.ndim != 1 or d.ndim != 1 or not np.isfinite(n).all() or not np.isfinite(d).all():
        raise ValueError("detail axes must be finite one-dimensional arrays")
    size = n.size * d.size
    if count == 0 or size == 0:
        return set()
    def normalized(axis):
        span = np.ptp(axis)
        return (axis - np.min(axis)) / span if span else np.full(axis.size, .5)
    x, y = np.meshgrid(normalized(n), normalized(d), indexing="ij")
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    first = int(np.argmin(np.sum((coordinates - .5) ** 2, axis=1)))
    distances = np.full(size, np.inf)
    chosen = []
    index = first
    for _ in range(min(count, size)):
        chosen.append(index)
        distances = np.minimum(distances, np.sum((coordinates - coordinates[index]) ** 2, axis=1))
        distances[chosen] = -1.
        index = int(np.argmax(distances))
    return {divmod(i, d.size) for i in chosen}


def explicit_contour_kpoint_count(args):
    """Match the detail runner: both positive contour axes select a new mesh."""
    n1, n2 = int(getattr(args, "contour_n1", 0)), int(getattr(args, "contour_n2", 0))
    return n1 * n2 if n1 > 0 and n2 > 0 else 0


def estimate_storage(dimension, n_layers, nk, total_points, detail_count, points_per_segment, dos_bins,
                     *, contour_nk=None):
    """Conservative uncompressed payload budget; RAM and saved bytes are separate."""
    dim, nk, total = int(dimension), int(nk), int(total_points)
    detail = min(max(0, int(detail_count)), total)
    contour = int(contour_nk) if contour_nk else nk
    path_points = 3 * int(points_per_segment) + 1
    # Budget repeated eigenvalue/contour spectra, band paths, plots and all stage formats.
    per_detail = (12 * nk * dim * 8 + 8 * path_points * dim * 8
                  + int(dos_bins) * (int(n_layers) + 8) * 8 + 12_000_000)
    # Each of two contour NPZs holds both valley spectra and two k coordinates.
    # Their SCF-mesh payload is already in the base budget.
    per_detail += 4 * max(0, contour - nk) * (dim + 1) * 8
    # Retain probabilities, not full complex eigenvectors, in each detail NPZ.
    per_detail += 2 * (nk + contour + path_points) * dim * 8
    # Both stages have six valley/flavor tables in total. Budget the scalar
    # and weighted band/XYZ companions at 24 bytes per formatted numeric field.
    # The default C3 square has 145-by-145 points. The final allowance covers
    # contour components, column metadata and DOS companions.
    origin_text_bytes = (6 * path_points * (4 + 3 * dim) * 24
                         + 6 * (contour + 145 ** 2) * 14 * 24 + 8_000_000)
    per_detail += origin_text_bytes + 8_000_000  # additional colored PDFs/PNGs
    scalar_bytes = total * 65_536 + 30_000_000
    return {
        "hamiltonian_dimension": dim,
        "scf_kpoint_count": nk,
        "contour_kpoint_count": contour,
        "two_valley_eigenvectors_GB_per_worker": 2 * nk * dim * dim * 16 / 1e9,
        "full_eigenvectors_saved": False,
        "compact_block_weights_saved_for_details": True,
        "origin_text_bytes_per_detail_estimate": origin_text_bytes,
        "detail_points": detail,
        "saved_results_estimate_GB": (scalar_bytes + detail * per_detail) / 1e9,
        "note": "Approximate uncompressed result budget including compact block weights and default Origin exports; contour complexity and plot size vary. Worker RAM is additional.",
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot save {type(value).__name__} as run metadata")


def write_run_config(out, args, ham, workflow, extra=None):
    from .band_gaps import gap_analysis_metadata, stoner_gap_analysis_metadata
    from .stoner.active_polarization import active_polarization_metadata
    from .output_metadata import displacement_metadata
    from .output_metadata import hamiltonian_metadata

    package = Path(__file__).resolve().parent
    source_files = ["continuum.py", "density.py", "electrostatics.py",
                    "mixing.py", "solver_full.py", "stoner/energy.py", "stoner/solver.py",
                    "band_gaps.py", "gap_maps.py", "stoner_update.py", "block_projection.py", "run_storage.py",
                    "origin_geometry.py", "origin_export.py", "origin_block_export.py", "block_plotting.py",
                    "stoner/active_polarization.py", "active_map.py", "active_reconstruction.py", "live_map.py"]
    record = {
        "created_at": now_sgt().isoformat(timespec="seconds"),
        "workflow": workflow,
        "command": list(sys.argv),
        "result_directory": str(Path(out).resolve()),
        "args": vars(args).copy(),
        "hamiltonian": hamiltonian_metadata(ham),
        "displacement": displacement_metadata(getattr(args, "D_sign", -1)),
        "analysis": {**gap_analysis_metadata(), **(stoner_gap_analysis_metadata()
                     if hasattr(args, "stoner_flavor_nu_min") else {}),
                     **({"active_polarization": {**active_polarization_metadata(),
                         "saved_carrier_fields": list(ACTIVE_CARRIER_FIELDS)}}
                        if hasattr(args, "stoner_flavor_nu_min") else {})},
        "platform": {"platform": platform.platform(), "python": sys.version,
                     "cpu_count": os.cpu_count(), "numpy": np.__version__},
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in source_files},
        "storage_policy": {"full_eigenvectors_saved": False, "npz_compressed": True,
                           "compact_block_weights_saved_for_details": True,
                           "detail_count": getattr(args, "detail_count", 1)},
        **(extra or {}),
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(out) / "config_used.json", record)
    return record
