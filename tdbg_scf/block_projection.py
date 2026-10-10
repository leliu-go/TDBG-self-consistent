"""Compact probabilities on layers 3--4 of the full continuum basis.

Zero denotes the upper bilayer (layers 1--2), and one denotes the lower
bilayer (layers 3--4). Both sublattices and every plane wave participate.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .atomic_io import atomic_npz
from .continuum import TDBGContinuumHamiltonian, TDBGParameters


ENERGY_ATOL_MEV = 1e-7
_EVECS_K_CHUNK = 4
_CACHE_FORMAT = 1


def _real_array(data, name):
    array = np.asarray(data)
    if array.dtype.kind not in "biuf":
        raise ValueError(name + " must contain real numerical values")
    return np.asarray(array, dtype=float)


def _lower_block_mask(layer_masks, orbital_count):
    masks = _real_array(layer_masks, "layer_masks")
    if (masks.shape != (4, orbital_count) or not np.isfinite(masks).all()
            or not np.all((masks == 0) | (masks == 1))
            or not np.all(masks.sum(axis=0) == 1)):
        raise ValueError("layer_masks must partition every orbital into four layers")
    return masks[2:].sum(axis=0)


def lower_block_weights_from_evecs(evecs, layer_masks):
    """Return normalized full-basis lower-bilayer probabilities, with bounded memory.

    ``evecs`` is orbital-by-band or k-by-orbital-by-band. A column need not be
    normalized: its lower-block norm is divided by its complete orbital norm.
    Small k chunks avoid allocating an all-grid magnitude-squared array.
    """
    states = np.asarray(evecs)
    if (states.ndim not in (2, 3) or not np.issubdtype(states.dtype, np.number)
            or any(size == 0 for size in states.shape)):
        raise ValueError("evecs must be a nonempty orbital-by-band or k-by-orbital-by-band array")
    single_k = states.ndim == 2
    if single_k:
        states = states[None, :, :]
    lower_block = _lower_block_mask(layer_masks, states.shape[1])
    result = np.empty((states.shape[0], states.shape[2]), dtype=float)
    for start in range(0, len(states), _EVECS_K_CHUNK):
        chunk = states[start:start + _EVECS_K_CHUNK]
        if not np.isfinite(chunk).all():
            raise ValueError("evecs must be finite")
        amplitude = np.abs(chunk).astype(float, copy=False)
        scale = amplitude.max(axis=1, keepdims=True)
        if not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError("each eigenvector must have a finite nonzero norm")
        # Scaling first also keeps probabilities invariant for very large or
        # very small (but finite) arbitrary column normalizations.
        amplitude /= scale
        np.square(amplitude, out=amplitude)
        norm = amplitude.sum(axis=1)
        lower_norm = np.einsum("o,kob->kb", lower_block, amplitude)
        result[start:start + len(chunk)] = np.clip(lower_norm / norm, 0, 1)
    return result[0] if single_k else result


def _finite_kpts(kpts, name="kpts"):
    points = _real_array(kpts, name)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError(name + " must be a nonempty finite (Nk, 2) array")
    return points


def diagonalize_block_weights(ham, kpts, U_meV, expected_evals=None):
    """Stream one full eigensystem per k at the fixed saved Hartree potential.

    If provided, every saved eigenvalue must agree within 1e-7 meV. A mismatch
    rejects the reconstruction instead of assigning weights from another model.
    Only compact eigenvalues and block probabilities are retained between k's.
    """
    if len(ham.layer_indices) != 4:
        raise ValueError("bilayer/bilayer probabilities require a four-layer Hamiltonian")
    points = _finite_kpts(kpts)
    potential = _real_array(U_meV, "U_meV")
    if potential.shape != (4,) or not np.isfinite(potential).all():
        raise ValueError("U_meV must be the saved finite four-layer potential")
    expected = None if expected_evals is None else _real_array(expected_evals, "saved eigenvalues")
    shape = (len(points), ham.dim)
    if expected is not None and (expected.shape != shape or not np.isfinite(expected).all()):
        raise ValueError("saved eigenvalues must have finite shape (Nk, full Hamiltonian dimension)")
    masks = ham.layer_projectors_diagonal()
    values = np.empty(shape, dtype=float)
    weights = np.empty(shape, dtype=float)
    for index, (kx, ky) in enumerate(points):
        energy, states = np.linalg.eigh(ham.hamiltonian(float(kx), float(ky), potential))
        if expected is not None and not np.allclose(energy, expected[index], rtol=0, atol=ENERGY_ATOL_MEV):
            difference = float(np.max(np.abs(energy - expected[index])))
            raise ValueError("saved spectrum does not match the fixed-U Hamiltonian at k index "
                             + str(index) + "; maximum error=" + str(difference) + " meV")
        values[index] = energy
        weights[index] = lower_block_weights_from_evecs(states, masks)
    return values, weights


def _finite_bands(data, name, point_count=None):
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(name + " must be a finite real band array")
    array = np.asarray(array, dtype=float)
    if (array.ndim != 2 or any(size == 0 for size in array.shape)
            or not np.isfinite(array).all()
            or (point_count is not None and len(array) != point_count)):
        raise ValueError(name + " must have finite shape (Nk, Nband)")
    return array


def _validated_weight(data, shape, name):
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(name + " weight must be a finite real probability")
    array = np.asarray(array, dtype=float)
    if (array.shape != shape or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 1)):
        raise ValueError(name + " weight must match its band shape and lie in [0, 1]")
    return array


def _saved_arrays(spectrum, contour, paths):
    """Validate all supplied probabilities before any cache can override them."""
    datasets, supplied = {}, {}
    try:
        for prefix, data in (("grid", spectrum), ("contour", contour)):
            points = _finite_kpts(data["kpts"], prefix + " kpts")
            for name in ("K", "Kp"):
                key = prefix + "_" + name
                bands = _finite_bands(data["evals_" + name + "_meV"], key + " eigenvalues", len(points))
                datasets[key] = {"bands": bands, "kpts": points}
                weight_field = "lower_block_weight_" + name
                if weight_field in data:
                    supplied[key] = _validated_weight(data[weight_field], bands.shape, key)
        for name in ("K", "Kp"):
            key = "path_" + name
            data = paths[name]
            bands = _finite_bands(data["bands"], key + " bands")
            datasets[key] = {"bands": bands}
            if "lower_block_weight" in data:
                supplied[key] = _validated_weight(data["lower_block_weight"], bands.shape, key)
    except KeyError as error:
        raise ValueError("saved block source is missing required field " + str(error)) from error
    return datasets, supplied


def _metadata(data, name):
    try:
        text = np.asarray(data["hamiltonian_json"]).item()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        metadata = json.loads(text)
    except (KeyError, ValueError, TypeError, UnicodeError) as error:
        raise ValueError(name + " requires scalar valid hamiltonian_json") from error
    if not isinstance(metadata, dict):
        raise ValueError(name + " hamiltonian_json must encode an object")
    return metadata


def _saved_hamiltonians(spectrum, contour, paths):
    metadata = _metadata(spectrum, "spectrum")
    parameter_names = {field.name for field in fields(TDBGParameters)}
    missing = parameter_names - set(metadata)
    if missing:
        raise ValueError("old saved data require complete Hamiltonian parameters; missing "
                         + ", ".join(sorted(missing)))
    # Saved metadata also contains derived conventions (n_layers, layer_order,
    # etc.); only dataclass construction fields enter the Hamiltonian.
    parameters = {name: metadata[name] for name in parameter_names}
    hams = {name: TDBGContinuumHamiltonian(TDBGParameters(**{**parameters, "valley": valley}))
            for name, valley in (("K", 1), ("Kp", -1))}
    if len(hams["K"].layer_indices) != 4 or metadata.get("n_layers", 4) != 4:
        raise ValueError("old block reconstruction requires a four-layer Hamiltonian")
    for name, data in (("contour", contour), ("path_K", paths["K"]), ("path_Kp", paths["Kp"])):
        if "hamiltonian_json" not in data:
            continue
        other = _metadata(data, name)
        if any(key in other and other[key] != metadata[key] for key in parameter_names - {"valley"}):
            raise ValueError(name + " Hamiltonian parameters do not match the saved spectrum")
    return hams


def _saved_path_points(ham, data, name):
    """Recover the exact saved high-symmetry sampling, with its valley geometry."""
    try:
        distance = _real_array(data["dist"], name + " dist")
        ticks = _real_array(data["ticks"], name + " ticks")
        bands = np.asarray(data["bands"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(name + " requires valid saved dist, ticks and bands") from error
    if (ticks.shape != (4,) or not np.issubdtype(ticks.dtype, np.number)
            or not np.isfinite(ticks).all() or not np.all(ticks == np.round(ticks))
            or ticks[0] != 0 or not np.all(np.diff(ticks) > 0)):
        raise ValueError(name + " ticks must describe the saved four-vertex path")
    points_per_segment = int(ticks[1] - ticks[0])
    points, expected_distance, expected_ticks, _ = ham.geom.high_symmetry_path(
        points_per_segment=points_per_segment)
    if (len(bands) != len(points) or distance.shape != expected_distance.shape
            or not np.isfinite(distance).all()
            or not np.array_equal(ticks, expected_ticks)
            or not np.allclose(distance, expected_distance, rtol=0, atol=1e-12)):
        raise ValueError(name + " dist, ticks or bands do not match the saved high-symmetry path")
    if "kpts" in data:
        points = _finite_kpts(data["kpts"], name + " kpts")
        measured = np.zeros(len(points))
        measured[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        if len(points) != len(bands) or not np.allclose(measured, distance, rtol=0, atol=1e-12):
            raise ValueError(name + " kpts do not match saved path distances")
    return points


def _hash_array(digest, label, value):
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + str(array.shape).encode("ascii") + b"\0")
    digest.update(memoryview(array).cast("B"))


def _source_fingerprint(spectrum, contour, paths, datasets):
    """Hash all source geometry/energies and supplied projections, not file mtimes."""
    digest = hashlib.sha256(b"full-basis-lower_block-probability-v1\0")
    source_fields = (
        ("spectrum", spectrum, ("U_meV", "kpts", "evals_K_meV", "evals_Kp_meV",
                                 "lower_block_weight_K", "lower_block_weight_Kp")),
        ("contour", contour, ("kpts", "evals_K_meV", "evals_Kp_meV",
                              "lower_block_weight_K", "lower_block_weight_Kp")),
        ("path_K", paths["K"], ("kpts", "bands", "dist", "ticks", "lower_block_weight")),
        ("path_Kp", paths["Kp"], ("kpts", "bands", "dist", "ticks", "lower_block_weight")),
    )
    for name, data, keys in source_fields:
        digest.update(name.encode("ascii") + b"\0")
        if "hamiltonian_json" in data:
            metadata = json.dumps(_metadata(data, name), sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest.update(metadata.encode("utf-8") + b"\0")
        for key in keys:
            if key in data:
                _hash_array(digest, name + "/" + key, data[key])
            else:
                digest.update((name + "/" + key + "/absent\0").encode("ascii"))
    # Include regenerated legacy path points as well as their saved distance,
    # ticks and bands. A change in the path implementation invalidates the cache.
    for name in ("path_K", "path_Kp"):
        _hash_array(digest, name + "/resolved_kpts", datasets[name]["kpts"])
    return digest.hexdigest()


def _matching_cache(path, fingerprint, datasets):
    if not path.is_file():
        return None
    try:
        # Buffer remote NPZ reads to avoid one network seek for each ZIP entry.
        with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as cache:
            if (int(cache["format_version"].item()) != _CACHE_FORMAT
                    or str(cache["source_fingerprint"].item()) != fingerprint):
                return None
            return {key: _validated_weight(cache[key], source["bands"].shape, "cached " + key)
                    for key, source in datasets.items()}
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        # An interrupted, outdated or invalid cache is expendable. Reconstruct
        # only after the saved source passes the same fixed-U checks again.
        return None


def get_saved_block_data(scf_dir, spectrum, contour, paths):
    """Use saved compact probabilities or reconstruct old full-basis results.

    Returns ``grid_K``, ``grid_Kp``, ``contour_K``, ``contour_Kp``, ``path_K``,
    ``path_Kp`` and ``source`` (``saved``, ``cache`` or ``reconstructed``).
    Fully supplied weights need only shape/finite/range validation; old results
    require complete saved parameters and U, and every full eigenvalue must
    reproduce before a compact, fingerprinted ``block_weights.npz`` is written.
    No SCF iteration or Stoner optimization runs here.
    """
    datasets, supplied = _saved_arrays(spectrum, contour, paths)
    if len(supplied) == len(datasets):
        return {**supplied, "source": "saved"}
    hams = _saved_hamiltonians(spectrum, contour, paths)
    if any(source["bands"].shape[1] != hams["K"].dim for source in datasets.values()):
        raise ValueError("old saved bands must match the full Hamiltonian dimension")
    if "U_meV" not in spectrum:
        raise ValueError("old saved data require the fixed saved U_meV")
    potential = _real_array(spectrum["U_meV"], "saved U_meV")
    if potential.shape != (4,) or not np.isfinite(potential).all():
        raise ValueError("saved U_meV must be a finite four-layer potential")
    for name in ("K", "Kp"):
        datasets["path_" + name]["kpts"] = _saved_path_points(hams[name], paths[name], "path_" + name)
    fingerprint = _source_fingerprint(spectrum, contour, paths, datasets)
    cache_path = Path(scf_dir) / "block_weights.npz"
    cached = _matching_cache(cache_path, fingerprint, datasets)
    if cached is not None:
        return {**cached, "source": "cache"}
    result = dict(supplied)
    for name in ("K", "Kp"):
        grid_key, contour_key, path_key = (prefix + "_" + name for prefix in ("grid", "contour", "path"))
        grid, contour_source = datasets[grid_key], datasets[contour_key]
        if grid_key not in result:
            _, result[grid_key] = diagonalize_block_weights(
                hams[name], grid["kpts"], potential, expected_evals=grid["bands"])
        if contour_key not in result:
            if (np.array_equal(grid["kpts"], contour_source["kpts"])
                    and np.array_equal(grid["bands"], contour_source["bands"])):
                result[contour_key] = result[grid_key]
            else:
                _, result[contour_key] = diagonalize_block_weights(
                    hams[name], contour_source["kpts"], potential, expected_evals=contour_source["bands"])
        if path_key not in result:
            source = datasets[path_key]
            _, result[path_key] = diagonalize_block_weights(
                hams[name], source["kpts"], potential, expected_evals=source["bands"])
    atomic_npz(cache_path, format_version=_CACHE_FORMAT, source_fingerprint=fingerprint, **result)
    return {**result, "source": "reconstructed"}
