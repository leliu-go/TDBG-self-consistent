"""Energy functional for the four-flavor phenomenological model."""
from __future__ import annotations

import numpy as np

from .dos import FlavorBandTable
from .params import StonerParams


def stoner_energy_meV_per_cell(
    nu_f: np.ndarray,
    tables: list[FlavorBandTable],
    params: StonerParams,
    A_M_A2: float,
) -> float:
    values = np.asarray(nu_f, dtype=float)
    if values.shape != (4,):
        raise ValueError("nu_f must have shape (4,)")
    if len(tables) != 4:
        raise ValueError("tables must contain four flavor tables")

    kinetic = 0.0
    for index in range(4):
        kinetic += tables[index].kinetic_of_nu(float(values[index]))

    u_cell = float(params.u0_meV_A2) / float(A_M_A2)
    J_cell = float(params.JH_meV_A2) / float(A_M_A2)

    total = float(np.sum(values))
    offdiag = total * total - float(np.dot(values, values))
    E_u = 0.5 * u_cell * offdiag

    m_K = values[0] - values[2]
    m_Kp = values[1] - values[3]
    E_hund = -J_cell * m_K * m_Kp

    return float(kinetic + E_u + E_hund)
