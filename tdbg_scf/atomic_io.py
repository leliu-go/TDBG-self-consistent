"""Flush complete files before replacing published results on the same disk."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import numpy as np


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.stem + ".", suffix=path.suffix, dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        if os.name != "nt":
            directory = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Cannot serialize " + type(value).__name__)


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=json_default, ensure_ascii=True)


def atomic_json(path, data):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def atomic_csv(path, frame):
    with atomic_output(path) as temporary:
        # NaN denotes missing data; pandas' NaN-to-text conversion is intentional.
        with np.errstate(invalid="ignore"):
            frame.to_csv(temporary, index=False)


def atomic_npz(path, **payload):
    with atomic_output(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
