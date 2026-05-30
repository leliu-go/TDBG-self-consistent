from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPoint:
    """A Q point commensurate with the uniform MBZ grid."""

    dq1: int
    dq2: int
    qvec_Ainv: np.ndarray

    @property
    def qx_Ainv(self) -> float:
        return float(self.qvec_Ainv[0])

    @property
    def qy_Ainv(self) -> float:
        return float(self.qvec_Ainv[1])

    @property
    def q_norm_Ainv(self) -> float:
        return float(np.linalg.norm(self.qvec_Ainv))

    @property
    def is_gamma(self) -> bool:
        return int(self.dq1) == 0 and int(self.dq2) == 0

    def to_record(self) -> dict:
        return {
            "dq1": int(self.dq1),
            "dq2": int(self.dq2),
            "qx_Ainv": self.qx_Ainv,
            "qy_Ainv": self.qy_Ainv,
            "q_norm_Ainv": self.q_norm_Ainv,
            "is_gamma": bool(self.is_gamma),
        }


def _centered_integer_steps(n: int, stride: int = 1, max_abs_step: int | None = None) -> list[int]:
    if n <= 0:
        raise ValueError("grid size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    start = -(n // 2)
    stop = n - n // 2
    steps = [x for x in range(start, stop) if x % stride == 0]
    if max_abs_step is not None:
        steps = [x for x in steps if abs(x) <= int(max_abs_step)]
    if 0 not in steps:
        steps.append(0)
    return sorted(set(int(x) for x in steps))


def make_commensurate_q_points(
    geom,
    grid_shape: tuple[int, int],
    q_stride: int = 1,
    max_abs_step: int | None = None,
    include_gamma: bool = True,
) -> list[QPoint]:
    """Build Q points commensurate with make_uniform_mbz_grid."""

    n1, n2 = [int(x) for x in grid_shape]
    steps1 = _centered_integer_steps(n1, stride=q_stride, max_abs_step=max_abs_step)
    steps2 = _centered_integer_steps(n2, stride=q_stride, max_abs_step=max_abs_step)

    b1 = np.asarray(geom.b1m, dtype=float)
    b2 = np.asarray(geom.b2m, dtype=float)
    qpts: list[QPoint] = []
    for dq1 in steps1:
        for dq2 in steps2:
            if not include_gamma and dq1 == 0 and dq2 == 0:
                continue
            qvec = (float(dq1) / n1) * b1 + (float(dq2) / n2) * b2
            qpts.append(QPoint(dq1=int(dq1), dq2=int(dq2), qvec_Ainv=qvec))

    qpts.sort(key=lambda q: (q.q_norm_Ainv, q.dq1, q.dq2))
    return qpts


def folded_index_for_q(ik: int, q: QPoint, grid_shape: tuple[int, int]) -> int:
    """Return the grid index of k_i + Q folded back to the uniform MBZ mesh."""

    n1, n2 = [int(x) for x in grid_shape]
    i, j = divmod(int(ik), n2)
    ip = (i + int(q.dq1)) % n1
    jp = (j + int(q.dq2)) % n2
    return ip * n2 + jp


def folded_indices_for_q(q: QPoint, grid_shape: tuple[int, int]) -> np.ndarray:
    n1, n2 = [int(x) for x in grid_shape]
    return np.asarray([folded_index_for_q(ik, q, (n1, n2)) for ik in range(n1 * n2)], dtype=int)


def folded_indices_and_shifts_for_q(q: QPoint, grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return folded indices and integer reciprocal-lattice shifts for k+Q.

    If raw grid coordinate i+dq folds as ip + s*n1, the returned shift is s.
    Thus k+Q = k_folded + s1*b1m + s2*b2m.
    """

    n1, n2 = [int(x) for x in grid_shape]
    indices = []
    shifts = []
    for ik in range(n1 * n2):
        i, j = divmod(int(ik), n2)
        s1, ip = divmod(i + int(q.dq1), n1)
        s2, jp = divmod(j + int(q.dq2), n2)
        indices.append(ip * n2 + jp)
        shifts.append((s1, s2))
    return np.asarray(indices, dtype=int), np.asarray(shifts, dtype=int)
