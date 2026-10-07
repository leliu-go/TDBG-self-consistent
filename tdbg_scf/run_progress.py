"""Measured initial estimates and completion-based progress for native CLI runs."""
from __future__ import annotations

from datetime import timedelta
import math
import time

import numpy as np

from .lattice import make_uniform_mbz_grid
from .run_storage import explicit_contour_kpoint_count, now_sgt


def duration_text(seconds):
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def initial_runtime_estimate(ham, args, total, workers, *, bands_only=False, reuse_scf=False):
    n1, n2 = int(getattr(args, "grid_n1", 1)), int(getattr(args, "grid_n2", 1))
    kpts, _ = make_uniform_mbz_grid(ham.geom, n1, n2)
    indices = np.unique(np.linspace(0, len(kpts) - 1, min(3, len(kpts))).astype(int))
    timings = []
    for index in indices:
        kx, ky = kpts[index]
        started = time.perf_counter()
        np.linalg.eigh(ham.hamiltonian(float(kx), float(ky), np.zeros(4)))
        timings.append(time.perf_counter() - started)
    per_k = max(float(np.median(timings)), 1e-9)
    limit = int(getattr(args, "full_max_iter", getattr(args, "max_iter", 60)))
    lo, hi = max(1, min(8, limit)), max(1, min(20, limit))
    if bands_only:
        lo = hi = 0
        nk = 3 * int(getattr(args, "points_per_segment", 60)) + 1
    else:
        nk = len(kpts)
    if reuse_scf:
        lo = hi = 0
    active = max(1, min(int(workers), int(total)))
    details = min(int(total), int(getattr(args, "detail_count", 1 if total == 1 else 10)))
    low = max(.001, float(total) / active * (nk * (2 if reuse_scf else lo + 1) * per_k * .75 + .1)
              + details / active * 2. + (5. if total > 1 else .5))
    high = max(low, float(total) / active * (nk * (hi + 2) * per_k * 2. + 1.5)
               + details / active * 15. + (45. if total > 1 else 2.))
    contour_nk = 0 if bands_only else explicit_contour_kpoint_count(args)
    contour_work = 2 * details * contour_nk
    # Detail meshes use eigvalsh in both valleys. Budget these extra solves
    # using the measured full diagonalization, with the same broad allowance.
    low += .5 * contour_work / active * per_k
    high += 2. * contour_work / active * per_k
    return {
        "low_s": low, "high_s": high, "sample_count": len(timings),
        "median_hamiltonian_and_eigh_seconds_per_k": per_k,
        "hartree_iteration_range": [lo, hi], "maximum_hartree_iterations": limit, "active_workers": active,
        "additional_contour_diagonalizations": contour_work,
        **({"full_valley_spectra_per_point": 2} if reuse_scf else {}),
        "note": ("Saved Hartree replay: two full-valley diagonalizations, zero Hartree iterations. Approximate parallel scaling; ETA updates from completed points."
                 if reuse_scf else "Approximate initial range from local k-point calibration; assumes the stated iteration range and parallel scaling. ETA updates from actual tasks, including postprocessing."),
    }


class RunProgress:
    def __init__(self, total, workers, initial, *, clock=None, now=None, resumed_completed=0):
        if int(total) < 1 or int(workers) < 1:
            raise ValueError("progress requires positive point and worker counts")
        self.total = int(total)
        self.workers = min(int(workers), self.total)
        self.initial = dict(initial)
        self.clock, self.now = clock or time.monotonic, now or now_sgt
        self.started = self.clock()
        if not 0 <= int(resumed_completed) <= self.total:
            raise ValueError("resumed count must be within the task count")
        self.resumed_completed = int(resumed_completed)
        self.completed = self.resumed_completed
        self.successful = 0
        self.failed = 0
        self.task_times = []

    def start_message(self):
        assumptions = self.initial.get("hartree_iteration_range")
        return (f"ETA initial: {duration_text(self.initial['low_s'])} - {duration_text(self.initial['high_s'])}; "
                f"{self.total - self.resumed_completed} remaining points, {self.workers} active workers. "
                + (f"Assumed Hartree iterations {assumptions[0]}-{assumptions[1]}. " if assumptions else "")
                + "Approximate; updates after completed tasks.")

    def snapshot(self):
        elapsed = max(0., self.clock() - self.started)
        remaining = self.total - self.completed
        if remaining == 0:
            eta, method = 0., "complete"
        elif self.completed > self.resumed_completed and not self.successful:
            eta, method = None, "awaiting_successful_point"
        elif self.successful >= self.workers and elapsed > 0:
            # A last partial batch has fewer working slots. Aggregate rate
            # cannot make its single remaining task finish 40 times faster.
            batches = math.ceil(remaining / self.workers)
            eta, method = batches * self.workers * elapsed / self.successful, "observed_throughput"
        elif self.task_times:
            eta = remaining * float(np.mean(self.task_times)) / min(self.workers, remaining)
            method = "observed_task_duration"
        else:
            eta = .5 * (self.initial["low_s"] + self.initial["high_s"])
            method = "calibration"
        return {"completed": self.completed, "total": self.total, "remaining": remaining,
                "resumed_points": self.resumed_completed,
                "successful_points": self.successful, "failed_points": self.failed,
                "elapsed_s": elapsed, "eta_s": eta, "method": method,
                "expected_completion_sgt": (None if eta is None else
                    (self.now() + timedelta(seconds=eta)).isoformat(timespec="seconds"))}

    def update(self, completed, label, status, task_runtime_s=None):
        if not self.completed <= int(completed) <= self.total:
            raise ValueError("completion count must increase within the task count")
        newly_completed = int(completed) - self.completed
        self.completed = int(completed)
        if status == "ok":
            self.successful += newly_completed
        else:
            self.failed += newly_completed
        if status == "ok" and task_runtime_s is not None and math.isfinite(float(task_runtime_s)) and float(task_runtime_s) > 0:
            self.task_times.append(float(task_runtime_s))
        state = self.snapshot()
        timing = ("ETA unavailable (no successful point yet); finish unknown" if state["eta_s"] is None else
                  f"ETA {duration_text(state['eta_s'])}; finish {state['expected_completion_sgt']} (SGT)")
        return (f"[{self.completed}/{self.total}] {label} {status}; elapsed {duration_text(state['elapsed_s'])}; "
                + timing)
