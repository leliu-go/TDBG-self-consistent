"""Global filling gaps, physical conventions, and portable map exports."""
import numpy as np
import pytest



def spectra():
    return (
        np.array([[-20, -8, -1, 2, 7, 20], [-19, -6, 0, 3, 9, 21]], dtype=float),
        np.array([[-18, -7, -2, 4, 8, 22], [-17, -5, -.5, 2.5, 10, 23]], dtype=float),
    )


def test_filling_gaps_use_one_band_per_flavor_and_both_valleys():
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    gaps = four_flavor_band_gaps(*spectra())
    for label, expected, lower, upper in [
        ("nu0", 2, 0, 2), ("nu_plus4", 3, 4, 7), ("nu_minus4", 3, -5, -2)
    ]:
        assert gaps[f"gap_{label}_meV"] == expected
        assert gaps[f"indirect_gap_{label}_meV"] == expected
        assert gaps[f"gap_{label}_lower_edge_meV"] == lower
        assert gaps[f"gap_{label}_upper_edge_meV"] == upper


def test_positive_direct_gaps_do_not_hide_indirect_overlap():
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    energy = np.array([[-20, -10, 0, 1, 10, 20], [-19, -9, 3, 4, 11, 21]])
    assert np.all(energy[:, 3] - energy[:, 2] > 0)
    gaps = four_flavor_band_gaps(energy, energy)
    assert gaps["indirect_gap_nu0_meV"] == -2
    assert gaps["gap_nu0_meV"] == 0


def test_overlap_between_valleys_closes_the_global_gap():
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    energy = np.array([[-20, -10, -1, 1, 10, 20]])
    gaps = four_flavor_band_gaps(energy, energy + 3)
    assert gaps["indirect_gap_nu0_meV"] == -1
    assert gaps["gap_nu0_meV"] == 0


def test_touching_bands_have_zero_gap():
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    energy = np.array([[-20, -10, 0, 0, 10, 20]])
    assert four_flavor_band_gaps(energy, energy)["gap_nu0_meV"] == 0


def test_gaps_are_invariant_under_a_common_energy_shift():
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    k, kp = spectra()
    original = four_flavor_band_gaps(k, kp)
    shifted = four_flavor_band_gaps(k + 31.25, kp + 31.25)
    for label in ("nu0", "nu_plus4", "nu_minus4"):
        assert shifted[f"gap_{label}_meV"] == original[f"gap_{label}_meV"]
        assert shifted[f"gap_{label}_lower_edge_meV"] == original[f"gap_{label}_lower_edge_meV"] + 31.25


@pytest.mark.parametrize("bad", [
    np.zeros(6), np.zeros((0, 6)), np.zeros((1, 2)), np.zeros((1, 5)),
    np.array([[-3, -2, -1, np.nan, 2, 3]]),
    np.array([[-3, -2, -1, 1, 2, np.inf]]),
    np.array([[-3, -1, -2, 1, 2, 3]]),
])
def test_invalid_or_incomplete_spectra_are_not_reported_as_zero_gap(bad):
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    with pytest.raises(ValueError):
        four_flavor_band_gaps(bad, spectra()[1])


def test_valleys_must_have_the_same_complete_band_count():
    from tdbg_scf.band_gaps import four_flavor_band_gaps

    with pytest.raises(ValueError, match="band count"):
        four_flavor_band_gaps(np.zeros((1, 4)), np.zeros((1, 6)))
