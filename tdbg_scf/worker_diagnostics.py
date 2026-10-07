"""Small process logs and native Python fault traces for server scans."""
from __future__ import annotations

from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import struct
import sys


_worker_log = None
_previous_fault_handler_enabled = False


def memory_snapshot() -> dict:
    """Windows counters are bytes; available commit is not physical RAM."""
    if sys.platform != "win32":
        return {}
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load_percent", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "commit_limit", "available_commit",
                    "total_process_virtual", "available_process_virtual", "extended_virtual")]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {f"{name}_bytes": int(getattr(status, name)) for name in (
            "total_physical", "available_physical", "commit_limit", "available_commit")}
    except Exception as exc:
        # Diagnostics must not turn an otherwise valid calculation into a failure.
        return {"memory_probe_error": repr(exc)}


def runtime_diagnostics() -> dict:
    return {
        "pid": os.getpid(), "python_executable": sys.executable,
        "python_version": sys.version, "python_bits": 8 * struct.calcsize("P"),
        "platform": sys.platform,
        "packages": {name: getattr(sys.modules.get(name), "__version__", None)
                     for name in ("numpy", "scipy", "matplotlib", "pandas")},
        "thread_env": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "memory": memory_snapshot(),
    }


def initialize_worker(directory: str) -> None:
    """Top-level spawn-safe initializer; retain the fault log's file handle."""
    global _worker_log, _previous_fault_handler_enabled
    if _worker_log is not None:
        close_worker_diagnostics()
    _previous_fault_handler_enabled = faulthandler.is_enabled()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    _worker_log = (path / f"worker_{os.getpid()}.log").open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_worker_log, all_threads=True)
    log_worker_event("worker_initialized", **runtime_diagnostics())


def log_worker_event(event: str, **fields) -> None:
    if _worker_log is not None:
        record = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "pid": os.getpid(), "event": event, **fields}
        _worker_log.write(json.dumps(record, ensure_ascii=True) + "\n")
        _worker_log.flush()


def close_worker_diagnostics() -> None:
    global _worker_log
    if _worker_log is not None:
        faulthandler.disable()
        _worker_log.close()
        _worker_log = None
        if _previous_fault_handler_enabled:
            faulthandler.enable()
