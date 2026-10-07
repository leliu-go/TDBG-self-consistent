"""Remaining time must account for parallel work and actual completion rate."""
from datetime import datetime, timezone

import pytest


def test_parallel_first_completion_uses_worker_task_time():
    from tdbg_scf.run_progress import RunProgress

    clock = [0.]
    progress = RunProgress(10, 2, {"low_s": 40., "high_s": 80.},
                           clock=lambda: clock[0], now=lambda: datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert "initial" in progress.start_message().lower()
    clock[0] = 10.
    message = progress.update(1, "point", "ok", task_runtime_s=10.)
    snapshot = progress.snapshot()
    assert snapshot["remaining"] == 9 and snapshot["eta_s"] == pytest.approx(45.)
    assert "1/10" in message and "ETA" in message and "finish" in message.lower()


def test_complete_batch_uses_observed_throughput_and_final_eta_is_zero():
    from tdbg_scf.run_progress import RunProgress

    clock = [0.]
    progress = RunProgress(10, 2, {"low_s": 40., "high_s": 80.}, clock=lambda: clock[0])
    clock[0] = 10.
    progress.update(2, "point", "ok", task_runtime_s=10.)
    assert progress.snapshot()["eta_s"] == pytest.approx(40.)
    assert progress.snapshot()["method"] == "observed_throughput"
    clock[0] = 50.
    progress.update(10, "last", "ok", task_runtime_s=10.)
    assert progress.snapshot()["eta_s"] == 0


def test_fast_failures_do_not_predict_successful_calculation_time():
    from tdbg_scf.run_progress import RunProgress

    clock = [0.]
    progress = RunProgress(9801, 40, {"low_s": 86400., "high_s": 172800.}, clock=lambda: clock[0])
    clock[0] = 11.
    message = progress.update(40, "broken", "failed", task_runtime_s=.01)
    state = progress.snapshot()
    assert state["eta_s"] is None
    assert state["expected_completion_sgt"] is None
    assert state["successful_points"] == 0 and state["failed_points"] == 40
    assert "ETA unavailable" in message
    clock[0] = 21.
    progress.update(41, "good", "ok", task_runtime_s=10.)
    assert progress.snapshot()["eta_s"] > 2000.


def test_final_partial_batch_uses_the_remaining_worker_slots():
    from tdbg_scf.run_progress import RunProgress

    clock = [0.]
    progress = RunProgress(41, 40, {"low_s": 20., "high_s": 40.}, clock=lambda: clock[0])
    clock[0] = 10.
    progress.update(40, "first_batch", "ok", task_runtime_s=10.)
    assert progress.snapshot()["remaining"] == 1
    assert progress.snapshot()["eta_s"] == pytest.approx(10.)


def test_initial_estimate_is_calibrated_and_bounded_by_iteration_limit():
    import argparse
    from tdbg_scf import TDBGContinuumHamiltonian, TDBGParameters
    from tdbg_scf.run_progress import initial_runtime_estimate

    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    args = argparse.Namespace(grid_n1=2, grid_n2=3, full_max_iter=3)
    estimate = initial_runtime_estimate(ham, args, total=10, workers=2)
    assert 0 < estimate["low_s"] <= estimate["high_s"]
    assert estimate["sample_count"] == 3
    assert estimate["hartree_iteration_range"] == [3, 3]
    assert "approximate" in estimate["note"].lower()


def test_initial_estimate_includes_explicit_contour_diagonalizations(monkeypatch):
    import argparse
    from tdbg_scf import TDBGContinuumHamiltonian, TDBGParameters
    from tdbg_scf import run_progress

    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    ticks = iter(i * .01 for i in range(12))
    monkeypatch.setattr(run_progress.time, "perf_counter", lambda: next(ticks))
    args = argparse.Namespace(grid_n1=2, grid_n2=3, full_max_iter=20, detail_count=10,
                              contour_n1=0, contour_n2=0)
    baseline = run_progress.initial_runtime_estimate(ham, args, total=9801, workers=40)
    args.contour_n1 = args.contour_n2 = 512
    larger = run_progress.initial_runtime_estimate(ham, args, total=9801, workers=40)
    assert larger["additional_contour_diagonalizations"] == 2 * 512 * 512 * 10
    assert larger["low_s"] > baseline["low_s"] + 600
    assert larger["high_s"] > baseline["high_s"] + 2500
