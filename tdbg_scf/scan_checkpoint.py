"""Immutable scan identity, durable worker results and crash-released locks."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .atomic_io import atomic_json, canonical_json
from .run_storage import now_sgt

CHECKPOINT_SCHEMA = 1
OPERATIONAL_ARGS = {"out", "data_root", "max_workers", "resume", "live_plot_interval_s", "restart_stoner_in_place"}


def digest(data):
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def source_hashes():
    package = Path(__file__).resolve().parent
    root = package.parent
    paths = sorted(package.rglob("*.py")) + [
        root / "examples/run_full_band_stoner.py", root / "examples/run_nd_mapping_full_band_stoner.py",
    ]
    return {path.relative_to(root).as_posix(): hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in paths}


def restore_resume_args(parser, args, argv):
    if not args.resume:
        return args
    if not args.out:
        raise ValueError("--resume requires --out pointing to the original result directory")
    manifest_path = Path(args.out).expanduser().resolve() / "checkpoints/scan_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("No resume manifest in this directory. Runs made before checkpoint support cannot recover RAM-only points.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Incompatible resume checkpoint version")
    saved = manifest["saved_args"]
    supplied = set()
    for token in argv:
        parsed = parser._parse_optional(token)
        if parsed is not None and parsed[0] is not None:
            supplied.add(parsed[0].dest)
    for name, value in saved.items():
        if name.startswith("_") or name in OPERATIONAL_ARGS:
            continue
        if name in supplied and canonical_json(getattr(args, name)) != canonical_json(value):
            raise ValueError("Incompatible resume parameter: " + name + " differs from the saved scan")
        setattr(args, name, value)
    for name in ("max_workers", "live_plot_interval_s"):
        if name not in supplied and name in saved:
            setattr(args, name, saved[name])
    return args


@contextmanager
def _file_lock(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    locked = False
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if not blocking:
                    raise RuntimeError("Another scan is already using this output directory") from exc
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(.05)
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


@contextmanager
def scan_directory_lock(out):
    """OS locks, rather than persistent PID files, recover after a hard crash."""
    with _file_lock(Path(out) / ".scan.lock"):
        yield


def _commit_lock(out):
    return _file_lock(Path(out) / ".checkpoint_commit.lock", blocking=True)


def begin_attempt(out, args):
    """Fence orphan workers before reading or scheduling this attempt's points."""
    attempt = uuid.uuid4().hex
    with _commit_lock(out):
        atomic_json(Path(out) / "checkpoints/active_attempt.json", {
            "attempt_id": attempt, "fingerprint": args._checkpoint_fingerprint,
        })
    args._checkpoint_attempt_id = attempt
    return attempt


def prepare_manifest(out, args, hamiltonian, tasks):
    public_args = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    inputs = {
        "args": {key: value for key, value in public_args.items() if key not in OPERATIONAL_ARGS},
        "hamiltonian": hamiltonian,
        "tasks": tasks,
        "source_sha256": source_hashes(),
    }
    fingerprint = digest(inputs)
    path = Path(out) / "checkpoints/scan_manifest.json"
    if args.resume:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != CHECKPOINT_SCHEMA or
                manifest.get("fingerprint") != fingerprint or digest(manifest.get("inputs")) != fingerprint):
            raise ValueError("Resume inputs/source mismatch: saved and current scan definitions differ")
    else:
        if (not getattr(args, "_inplace_update_prepared", False) and Path(out).exists()
                and any(child.name != ".scan.lock" for child in Path(out).iterdir())):
            raise ValueError("This result directory already contains a run. Use --resume --out or choose a new directory.")
        manifest = {"schema_version": CHECKPOINT_SCHEMA, "fingerprint": fingerprint,
                    "created_at": now_sgt().isoformat(), "inputs": inputs, "saved_args": public_args}
        atomic_json(path, manifest)
    args._checkpoint_fingerprint = fingerprint
    return fingerprint


def _artifacts(out, task, row):
    if not task.get("detail") or not row.get("detail_saved"):
        return {}
    directory = Path(out) / "details" / task["label"]
    return {path.relative_to(out).as_posix(): {"size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _checked_child(out, path):
    """Verify absolute containment before moving or deleting result directories."""
    root, child = Path(out).resolve(), Path(path).resolve()
    relative = child.relative_to(root)
    if not relative.parts:
        raise ValueError("Cannot move/delete the scan root")
    return child


def _publish_details(out, task, source):
    source = _checked_child(out, source)
    destination = _checked_child(out, Path(out) / "details" / task["label"])
    if not source.is_dir():
        return
    # Point configuration should name the public result directory after publication.
    for config in source.rglob("config_used.json"):
        data = json.loads(config.read_text(encoding="utf-8"))
        data["result_directory"] = str(destination / config.parent.relative_to(source))
        if isinstance(data.get("args"), dict):
            data["args"]["out"] = data["result_directory"]
        atomic_json(config, data)
    for file in source.rglob("*"):
        if file.is_file():
            with file.open("rb+") as stream:
                os.fsync(stream.fileno())
    backup = _checked_child(out, Path(out) / "checkpoints/retired_details" / uuid.uuid4().hex)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        os.replace(str(destination), str(backup))
    try:
        os.replace(str(source), str(destination))
    except OSError:
        if backup.exists() and not destination.exists():
            os.replace(str(backup), str(destination))
        raise
    if backup.exists():
        # Both paths were resolved and verified inside this explicitly named run.
        shutil.rmtree(_checked_child(out, backup))


def save_point(out, task, row, fingerprint, *, attempt_id=None, detail_source=None):
    """Only the current attempt can publish details and its durable point record."""
    out = Path(out)
    with _commit_lock(out):
        if attempt_id is not None:
            active = json.loads((out / "checkpoints/active_attempt.json").read_text(encoding="utf-8"))
            if active != {"attempt_id": attempt_id, "fingerprint": fingerprint}:
                return False
        if detail_source is not None:
            _publish_details(out, task, detail_source)
        payload = {"schema_version": CHECKPOINT_SCHEMA, "fingerprint": fingerprint,
                   "attempt_id": attempt_id, "completed_at": now_sgt().isoformat(), "point": row,
                   "artifacts": _artifacts(out, task, row)}
        atomic_json(out / "checkpoints/points" / (task["label"] + ".json"),
                    {**payload, "checksum": digest(payload)})
        error = out / "errors" / (task["label"] + ".json")
        if row.get("status") != "ok":
            atomic_json(error, row)
        elif error.exists():
            error.unlink()
    return True


def load_points(out, tasks, fingerprint, *, attempt_id=None):
    with _commit_lock(out):
        return _load_points(out, tasks, fingerprint, attempt_id=attempt_id)


def _load_points(out, tasks, fingerprint, *, attempt_id=None):
    rows, reusable, rejected = {}, set(), {}
    out = Path(out)
    for task in tasks:
        path = out / "checkpoints/points" / (task["label"] + ".json")
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("checkpoint must be a JSON object")
            checksum = data.pop("checksum")
            if (digest(data) != checksum or data["schema_version"] != CHECKPOINT_SCHEMA or
                    data["fingerprint"] != fingerprint):
                raise ValueError("checkpoint checksum/version/fingerprint mismatch")
            if attempt_id is not None and data.get("attempt_id") != attempt_id:
                continue
            row = data["point"]
            if not isinstance(row, dict) or not isinstance(data.get("artifacts"), dict):
                raise ValueError("checkpoint point/artifacts must be JSON objects")
            for key in ("label", "n_index", "D_index", "n_cm2", "D_Vnm"):
                if type(row[key]) is not type(task[key]) or row[key] != task[key]:
                    raise ValueError("checkpoint task identity mismatch: " + key)
            good = (row.get("status") == "ok" and row.get("scf_status") == "ok"
                    and row.get("scf_converged") is True and row.get("stoner_success") is True)
            if good and task.get("detail"):
                artifacts = data["artifacts"]
                expected = "details/" + task["label"] + "/"
                good = bool(row.get("detail_saved") and artifacts
                            and expected + "summary.json" in artifacts
                            and expected + "scf/summary.json" in artifacts)
                for relative, check in artifacts.items():
                    if not isinstance(check, dict):
                        raise ValueError("invalid detail artifact metadata")
                    if not relative.startswith(expected) or ".." in Path(relative).parts:
                        raise ValueError("invalid detail artifact path")
                    file = out / relative
                    if (not file.is_file() or file.stat().st_size != check["size"] or
                            hashlib.sha256(file.read_bytes()).hexdigest() != check["sha256"]):
                        good = False
                        break
                row["detail_dir"] = str(out / "details" / task["label"])
                if not good:
                    row["detail_saved"] = False
            rows[task["label"]] = row
            if good:
                reusable.add(task["label"])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            rejected[task["label"]] = str(exc)
    return rows, reusable, rejected
