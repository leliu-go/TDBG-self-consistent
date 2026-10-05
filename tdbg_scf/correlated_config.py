"""Configuration loader for correlated calculation entry points."""
from __future__ import annotations

import copy
import json
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("YAML config files require PyYAML; use JSON otherwise") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if data is None else dict(data)


def load_correlated_config(path: str | Path, nu_override: float | None = None, D_override: float | None = None) -> dict:
    cfg_path = Path(path)
    suffix = cfg_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(cfg_path)
    else:
        raise ValueError("config path must end in .json, .yaml, or .yml")

    cfg = copy.deepcopy(data)
    if cfg.get("method", "stoner") != "stoner":
        raise ValueError("method must be 'stoner'; correlated scans use full-band layer Hartree")
    if any(str(key).startswith("projected") for key in cfg):
        raise ValueError("projection configuration is no longer supported; use full_scf")
    shared = cfg.setdefault("shared", {})
    if nu_override is not None:
        shared["nu_grid"] = [float(nu_override)]
    if D_override is not None:
        shared["D_Vnm_grid"] = [float(D_override)]
    return cfg
