"""Post-Stoner integer gaps use flavor bands, not discrete DOS level spacing."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from tdbg_scf import band_gaps


def post_gaps(K, Kp, sigma):
    calculate = getattr(band_gaps, "stoner_integer_band_gaps", None)
    assert callable(calculate), "Post-Stoner integer gap analysis is missing"
    return calculate(K, Kp, sigma)


def split_spectra():
    base = np.array([[-100., -40., -20., 0., 30., 50., 70., 100.]])
    return np.vstack([base, base + .5]), np.array([0., 2., 5., 9.])


@pytest.mark.parametrize("nu,expected", [
    (-6, 2.5), (-5, 3.5), (-4, 10.5), (-3, 1.5), (-2, 2.5), (-1, 3.5),
    (0, 20.5), (1, 1.5), (2, 2.5), (3, 3.5), (4, 10.5), (5, 1.5), (6, 2.5),
])
def test_integer_gaps_count_one_band_per_flavor(nu, expected):
    energy, sigma = split_spectra()
    label = "nu0" if nu == 0 else f"nu_{'plus' if nu > 0 else 'minus'}{abs(nu)}"
    gaps = post_gaps(energy, energy, sigma)
    assert gaps[f"stoner_gap_{label}_meV"] == pytest.approx(expected)
    assert gaps[f"stoner_indirect_gap_{label}_meV"] == pytest.approx(expected)
    assert gaps[f"stoner_gap_{label}_upper_edge_meV"] - gaps[f"stoner_gap_{label}_lower_edge_meV"] == pytest.approx(expected)


def test_unsplit_identical_flavors_have_no_odd_filling_gaps():
    energy, _ = split_spectra()
    gaps = post_gaps(energy, energy, np.zeros(4))
    before = band_gaps.four_flavor_band_gaps(energy, energy)
    for nu in (-6, -5, -3, -2, -1, 1, 2, 3, 5, 6):
        label = f"nu_{'plus' if nu > 0 else 'minus'}{abs(nu)}"
        assert gaps[f"stoner_gap_{label}_meV"] == 0
    for label in ("nu0", "nu_minus4", "nu_plus4"):
        assert gaps[f"stoner_gap_{label}_meV"] == before[f"gap_{label}_meV"]


def test_common_self_energy_shift_changes_edges_but_not_gaps():
    energy, sigma = split_spectra()
    original = post_gaps(energy, energy, sigma)
    shifted = post_gaps(energy, energy, sigma + 37.25)
    for key, value in original.items():
        expected = value + 37.25 if key.endswith("edge_meV") else value
        assert shifted[key] == pytest.approx(expected)


def test_positive_direct_flavor_gaps_can_have_zero_indirect_gap():
    energy, sigma = split_spectra()
    energy[1] = energy[0] + 3
    gaps = post_gaps(energy, energy, sigma)
    # The direct gap between the first two spin/valley conduction bands is 2;
    # its global indirect gap is -1. Sorted DOS levels would give a false gap.
    assert gaps["stoner_indirect_gap_nu_plus1_meV"] == -1
    assert gaps["stoner_gap_nu_plus1_meV"] == 0


def test_full_flavor_spectra_allow_reordering_far_from_neutral_band_indices():
    energy, _ = split_spectra()
    Kp = energy + 11
    sigma = np.array([-35., 8., 25., -3.])
    gaps = post_gaps(energy, Kp, sigma)
    combined = np.sort(np.concatenate([
        energy + sigma[0], Kp + sigma[1], energy + sigma[2], Kp + sigma[3],
    ], axis=1), axis=1)
    for nu in range(-6, 7):
        label = "nu0" if nu == 0 else f"nu_{'plus' if nu > 0 else 'minus'}{abs(nu)}"
        cut = 2 * energy.shape[1] + nu
        lower, upper = combined[:, cut - 1].max(), combined[:, cut].min()
        assert gaps[f"stoner_gap_{label}_lower_edge_meV"] == lower
        assert gaps[f"stoner_gap_{label}_upper_edge_meV"] == upper


@pytest.mark.parametrize("sigma", [np.zeros(3), np.zeros((2, 2)), [0, 1, np.nan, 3], [0, 1, 2, np.inf], [0, 1j, 2, 3]])
def test_invalid_self_energies_are_rejected(sigma):
    energy, _ = split_spectra()
    with pytest.raises(ValueError):
        post_gaps(energy, energy, sigma)


def test_combined_flavor_gaps_require_the_same_k_mesh():
    energy, sigma = split_spectra()
    with pytest.raises(ValueError, match="k|mesh|shape"):
        post_gaps(energy, energy[:1], sigma)
