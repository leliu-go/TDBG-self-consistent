"""Independent four-layer full-basis probability and legacy-cache checks."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian
from tdbg_scf.block_projection import lower_block_weights_from_evecs, diagonalize_block_weights


def test_every_plane_wave_and_sublattice_participates_in_lower_bilayer_probability():
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=1))
    rng = np.random.RandomState(17)
    states = rng.normal(size=(7, ham.dim, 3)) + 1j*rng.normal(size=(7, ham.dim, 3))
    expected = (np.abs(states[:,4*ham.siteN:])**2).sum(axis=1)/(np.abs(states)**2).sum(axis=1)
    original = states.copy()
    states *= np.array([1e-200, 1., 1e200])[None,None,:]
    result = lower_block_weights_from_evecs(states, ham.layer_projectors_diagonal())
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(lower_block_weights_from_evecs(original, ham.layer_projectors_diagonal()), expected)
    assert np.all((result >= 0) & (result <= 1))


def test_fixed_U_streaming_uses_complete_eigenvectors_and_checks_saved_energies(monkeypatch):
    ham = TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    k = np.array([[.003, -.007], [-.002, .001], [0., 0.]])
    U = np.array([7., 3., -4., -6.])
    energies, vectors = ham.diagonalize(k, U)
    expected = (np.abs(vectors[:,4:])**2).sum(axis=1)
    monkeypatch.setattr(ham, "diagonalize", lambda *a,**kw: pytest.fail("full-grid vectors allocated"))
    values, probability = diagonalize_block_weights(ham, k, U, expected_evals=energies)
    np.testing.assert_allclose(values, energies, atol=1e-12, rtol=0)
    np.testing.assert_allclose(probability, expected, atol=1e-14)
    with pytest.raises(ValueError, match="match.*fixed-U"):
        diagonalize_block_weights(ham, k, U, expected_evals=energies+.01)


def test_legacy_Origin_loader_skips_large_vectors_and_restores_complete_metadata(tmp_path, monkeypatch):
    from tdbg_scf.origin_export import _load
    from numpy.lib.npyio import NpzFile
    params = TDBGParameters(cutoff=0)
    path = tmp_path / "legacy.npz"
    np.savez(path, hamiltonian_parameters_json=json.dumps(asdict(params)),
             evals_K_meV=np.zeros((1,8)), evecs_K=np.eye(8)[None], evecs_Kp=np.eye(8)[None])
    get = NpzFile.__getitem__
    def checked(archive, key):
        assert not key.startswith("evecs_"), "legacy export loaded a full-vector member"
        return get(archive,key)
    monkeypatch.setattr(NpzFile,"__getitem__",checked)
    loaded = _load(path)
    metadata = json.loads(str(loaded["hamiltonian_json"].item()))
    assert metadata["stacking_top_to_bottom"] == "BA-AB" and metadata["n_layers"] == 4
    assert metadata["gamma4_meV"] == 138. and metadata["sublattice_Z_meV"] == 15.
    assert not any(key.startswith("evecs_") for key in loaded)


def test_lower_bilayer_DOS_fraction_matches_four_layer_sum():
    from examples.run_full_band_stoner import layer_dos_polarizations
    value = layer_dos_polarizations(np.array([1.,2.,3.,4.]))
    assert value["layer_dos_lower_block_fraction"] == pytest.approx(.7)
    assert np.isnan(layer_dos_polarizations(np.zeros(4))["layer_dos_lower_block_fraction"])
