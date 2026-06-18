from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tdbg_scf.filling import density_cm2_to_filling, filling_to_density_cm2
from tdbg_scf.lattice import MoireGeometry


@dataclass(frozen=True)
class FixedNuDPoint:
    D_index: int
    D_Vnm: float
    n_cm2: float
    nu_total: float


def parse_float_list(text: str | None) -> list[float]:
    if text is None or str(text).strip() == "":
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def moire_area_from_geometry(theta_deg: float, a_cc_A: float, valley: int = 1) -> float:
    geom = MoireGeometry(theta_deg=float(theta_deg), a_cc_A=float(a_cc_A), valley=int(valley))
    return float((2.0 * np.pi) ** 2 / geom.mBZ_area_A2_inv)


def density_for_fixed_nu(nu_total: float, theta_deg: float, a_cc_A: float, valley: int = 1) -> float:
    return filling_to_density_cm2(float(nu_total), moire_area_from_geometry(theta_deg, a_cc_A, valley))


def build_D_values(D_min: float | None, D_max: float | None, D_count: int | None, D_values: str | None) -> list[float]:
    explicit = parse_float_list(D_values)
    if explicit:
        return explicit
    if D_min is None or D_max is None or D_count is None:
        raise ValueError("provide --D-values or all of --D-min, --D-max, --D-count")
    if int(D_count) < 1:
        raise ValueError("--D-count must be >= 1")
    return [float(x) for x in np.linspace(float(D_min), float(D_max), int(D_count))]


def build_fixed_nu_D_points(
    *,
    nu_total: float | None,
    n_cm2: float | None,
    D_values: list[float],
    theta_deg: float,
    a_cc_A: float,
    valley: int = 1,
) -> list[FixedNuDPoint]:
    if (nu_total is None) == (n_cm2 is None):
        raise ValueError("provide exactly one of --nu-total and --n-cm2")
    A_M_A2 = moire_area_from_geometry(theta_deg, a_cc_A, valley)
    if nu_total is None:
        nu = density_cm2_to_filling(float(n_cm2), A_M_A2)
        density = float(n_cm2)
    else:
        nu = float(nu_total)
        density = filling_to_density_cm2(nu, A_M_A2)
    return [
        FixedNuDPoint(D_index=i, D_Vnm=float(D), n_cm2=float(density), nu_total=float(nu))
        for i, D in enumerate(D_values)
    ]


def D_point_label(D_Vnm: float) -> str:
    return f"D_{float(D_Vnm):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")


def load_completed_D_values(summary_csv: Path) -> set[float]:
    path = Path(summary_csv)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "D_Vnm" not in df.columns:
        return set()
    return {round(float(x), 10) for x in df["D_Vnm"].dropna()}

