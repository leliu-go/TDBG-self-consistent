from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPoint:
    """A Q point commensurate with the uniform MBZ grid."""

    dq1: int
    dq2: int
    qvec_Ainv: np.ndarray
    q_raw_Ainv: np.ndarray | None = None
    q_mbz_Ainv: np.ndarray | None = None
    reciprocal_shift_m: int = 0
    reciprocal_shift_n: int = 0
    q_resolution_Ainv: float | None = None
    shell_index: int | None = None

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
    def qx_raw_Ainv(self) -> float:
        vec = self.qvec_Ainv if self.q_raw_Ainv is None else self.q_raw_Ainv
        return float(vec[0])

    @property
    def qy_raw_Ainv(self) -> float:
        vec = self.qvec_Ainv if self.q_raw_Ainv is None else self.q_raw_Ainv
        return float(vec[1])

    @property
    def qx_mbz_Ainv(self) -> float:
        vec = self.qvec_Ainv if self.q_mbz_Ainv is None else self.q_mbz_Ainv
        return float(vec[0])

    @property
    def qy_mbz_Ainv(self) -> float:
        vec = self.qvec_Ainv if self.q_mbz_Ainv is None else self.q_mbz_Ainv
        return float(vec[1])

    @property
    def q_norm_mbz_Ainv(self) -> float:
        vec = self.qvec_Ainv if self.q_mbz_Ainv is None else self.q_mbz_Ainv
        return float(np.linalg.norm(vec))

    @property
    def is_gamma(self) -> bool:
        return self.q_norm_mbz_Ainv < 1e-14

    def to_record(self) -> dict:
        q_resolution = np.nan if self.q_resolution_Ainv is None else float(self.q_resolution_Ainv)
        shell_index = max(abs(int(self.dq1)), abs(int(self.dq2))) if self.shell_index is None else int(self.shell_index)
        return {
            "dq1": int(self.dq1),
            "dq2": int(self.dq2),
            "qx_Ainv": self.qx_Ainv,
            "qy_Ainv": self.qy_Ainv,
            "q_norm_Ainv": self.q_norm_Ainv,
            "qx_raw_Ainv": self.qx_raw_Ainv,
            "qy_raw_Ainv": self.qy_raw_Ainv,
            "qx_mbz_Ainv": self.qx_mbz_Ainv,
            "qy_mbz_Ainv": self.qy_mbz_Ainv,
            "q_norm_mbz_Ainv": self.q_norm_mbz_Ainv,
            "reciprocal_shift_m": int(self.reciprocal_shift_m),
            "reciprocal_shift_n": int(self.reciprocal_shift_n),
            "q_resolution_Ainv": q_resolution,
            "qstar_shell_index": shell_index,
            "qstar_distance_from_gamma_in_grid_units": float(np.hypot(int(self.dq1), int(self.dq2))),
            "is_gamma": bool(self.is_gamma),
        }


@dataclass(frozen=True)
class ReducedQ:
    q_raw_Ainv: np.ndarray
    q_mbz_Ainv: np.ndarray
    reciprocal_shift_m: int
    reciprocal_shift_n: int


def reduce_q_to_first_mbz(qvec_Ainv: np.ndarray, geom, search_range: int = 2) -> ReducedQ:
    raw = np.asarray(qvec_Ainv, dtype=float)
    b1 = np.asarray(geom.b1m, dtype=float)
    b2 = np.asarray(geom.b2m, dtype=float)
    best_vec = raw
    best_shift = (0, 0)
    best_norm = float(np.linalg.norm(raw))
    for m in range(-int(search_range), int(search_range) + 1):
        for n in range(-int(search_range), int(search_range) + 1):
            candidate = raw + float(m) * b1 + float(n) * b2
            norm = float(np.linalg.norm(candidate))
            if norm < best_norm - 1e-14:
                best_vec = candidate
                best_shift = (int(m), int(n))
                best_norm = norm
    return ReducedQ(q_raw_Ainv=raw, q_mbz_Ainv=np.asarray(best_vec, dtype=float), reciprocal_shift_m=best_shift[0], reciprocal_shift_n=best_shift[1])


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
    first_mbz_only: bool = False,
) -> list[QPoint]:
    """Build Q points commensurate with make_uniform_mbz_grid."""

    n1, n2 = [int(x) for x in grid_shape]
    steps1 = _centered_integer_steps(n1, stride=q_stride, max_abs_step=max_abs_step)
    steps2 = _centered_integer_steps(n2, stride=q_stride, max_abs_step=max_abs_step)

    b1 = np.asarray(geom.b1m, dtype=float)
    b2 = np.asarray(geom.b2m, dtype=float)
    qpts: list[QPoint] = []
    seen: set[tuple[int, int]] = set()
    for dq1 in steps1:
        for dq2 in steps2:
            if not include_gamma and dq1 == 0 and dq2 == 0:
                continue
            qvec = (float(dq1) / n1) * b1 + (float(dq2) / n2) * b2
            reduced = reduce_q_to_first_mbz(qvec, geom) if first_mbz_only else None
            q_mbz = qvec if reduced is None else reduced.q_mbz_Ainv
            key = (int(round(float(q_mbz[0]) * 1e12)), int(round(float(q_mbz[1]) * 1e12)))
            if first_mbz_only:
                if key in seen:
                    continue
                seen.add(key)
            shell = max(abs(int(dq1)), abs(int(dq2)))
            qpts.append(
                QPoint(
                    dq1=int(dq1),
                    dq2=int(dq2),
                    qvec_Ainv=q_mbz if first_mbz_only else qvec,
                    q_raw_Ainv=qvec,
                    q_mbz_Ainv=q_mbz,
                    reciprocal_shift_m=0 if reduced is None else reduced.reciprocal_shift_m,
                    reciprocal_shift_n=0 if reduced is None else reduced.reciprocal_shift_n,
                    shell_index=shell,
                )
            )

    qpts.sort(key=lambda q: (q.q_norm_mbz_Ainv, q.dq1, q.dq2))
    nonzero_norms = [q.q_norm_mbz_Ainv for q in qpts if not q.is_gamma]
    q_resolution = min(nonzero_norms) if nonzero_norms else 0.0
    return [
        QPoint(
            dq1=q.dq1,
            dq2=q.dq2,
            qvec_Ainv=q.qvec_Ainv,
            q_raw_Ainv=q.q_raw_Ainv,
            q_mbz_Ainv=q.q_mbz_Ainv,
            reciprocal_shift_m=q.reciprocal_shift_m,
            reciprocal_shift_n=q.reciprocal_shift_n,
            q_resolution_Ainv=q_resolution,
            shell_index=q.shell_index,
        )
        for q in qpts
    ]


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
