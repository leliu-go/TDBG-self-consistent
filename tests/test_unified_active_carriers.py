import numpy as np
import pytest

from tdbg_scf.stoner.active_polarization import full_spectrum_active_occupations, POLARIZATION_FIELDS


def _count(nu, spectrum=None, weights=None):
    if spectrum is None:
        spectrum = np.array([[-100., -5., 5., 100.]])
        weights = [1.]
    return full_spectrum_active_occupations(spectrum, spectrum, weights, 1., nu)


@pytest.mark.parametrize('nu', [(0.,0.,1.,1.), (-1.,-1.,0.,0.)])
def test_symmetric_filled_valence_background_does_not_dilute_fully_spin_polarized_carriers(nu):
    record = _count(nu)
    np.testing.assert_allclose(list(record['polarizations'].values()), [1.,0.,0.], atol=1e-14)


def test_neutral_spin_polarization_includes_holes_with_opposite_electronic_sign():
    record = _count((-1.,-1.,1.,1.))
    assert 'active_electron_nu_f' in record
    assert record['active_electron_nu_f'] == [0.,0.,1.,1.]
    assert record['active_hole_nu_f'] == [1.,1.,0.,0.]
    assert record['polarizations'][POLARIZATION_FIELDS[0]] == 1.


def test_balanced_valence_holes_have_zero_polarization_even_when_active_manifold_is_empty():
    record = _count((-1.,-1.,-1.,-1.))
    np.testing.assert_allclose(list(record['polarizations'].values()), [0.,0.,0.], atol=1e-14)


def test_symmetric_reference_without_active_carriers_has_undefined_fraction():
    record = _count((0.,0.,0.,0.))
    assert all(np.isnan(value) for value in record['polarizations'].values())


def test_compensated_same_flavor_pockets_cannot_be_replaced_by_absolute_net_filling():
    spectrum = np.array([[-100.,-4.,-2.,100.], [-100.,2.,4.,100.]])
    record = _count((.25,.25,-.25,-.25), spectrum, [.5,.5])
    np.testing.assert_allclose(list(record['polarizations'].values()), [1./3.,0.,0.], atol=1e-14)
    np.testing.assert_allclose(record['active_electron_nu_f'], [.5,.5,.25,.25], atol=1e-14)
    np.testing.assert_allclose(record['active_hole_nu_f'], [.25,.25,.5,.5], atol=1e-14)


def test_band_overlap_at_neutrality_records_electron_and_hole_pockets_separately():
    spectrum = np.array([[-100.,-4.,-2.,100.], [-100.,2.,4.,100.]])
    record = _count((0.,0.,0.,0.), spectrum, [.5,.5])
    assert 'active_electron_nu_f' in record
    np.testing.assert_allclose(record['active_electron_nu_f'], [.5]*4, atol=1e-14)
    np.testing.assert_allclose(record['active_hole_nu_f'], [.5]*4, atol=1e-14)
    np.testing.assert_allclose(list(record['polarizations'].values()), [0.,0.,0.], atol=1e-14)


@pytest.mark.parametrize('weights', [np.ones(100)/100., np.array([.1,.2,.7])])
def test_complete_filled_valence_nodes_do_not_generate_phantom_minority_carriers(weights):
    nk=len(weights)
    spectrum=np.tile(np.linspace(-100.,100.,20),(nk,1))+np.arange(nk)[:,None]*.01
    record=_count((0.,0.,1.,1.),spectrum,weights)
    assert record['active_electron_nu_f'][:2]==[0.,0.]
    assert record['active_hole_nu_f']==[0.]*4
    assert record['polarizations'][POLARIZATION_FIELDS[0]]==1.


def test_tiny_genuine_minority_survives_full_spectrum_filling_around_neutrality():
    record=_count((1.,0.,1e-18,0.))
    assert record['active_electron_nu_f'][2] > 0.
    assert record['polarizations'][POLARIZATION_FIELDS[0]] < 1.
    assert record['polarizations'][POLARIZATION_FIELDS[2]] < 1.


def test_tiny_genuine_minority_hole_survives_full_spectrum_filling_around_neutrality():
    record=_count((-1.,0.,-1e-18,0.))
    assert record['active_hole_nu_f'][2] > 0.
    assert record['polarizations'][POLARIZATION_FIELDS[0]] < 1.
    assert record['polarizations'][POLARIZATION_FIELDS[2]] < 1.


def test_nonuniform_signed_boundary_distances_do_not_create_a_phantom_hole():
    spectrum=np.array([[-8.,4.,8.,10.],[-8.,-7.,0.,7.],[-6.,0.,2.,6.]])
    record=_count((1.,1.,1.,1.),spectrum,[1e-18,.25,.75])
    assert record['active_hole_nu_f']==[0.]*4


@pytest.mark.parametrize('nb',[2,4,6,8])
@pytest.mark.parametrize('sign',[-1.,1.])
def test_nonuniform_complete_spectrum_endpoints_are_full_or_empty(nb,sign):
    from tdbg_scf.stoner.active_polarization import count_flavor_active_carriers
    spectrum=np.tile(np.linspace(-100.,100.,nb),(3,1))
    electron,hole=count_flavor_active_carriers(spectrum,[.1,.2,.7],1.,sign*nb/2.)
    assert (electron,hole)==((1.,0.) if sign>0. else (0.,1.))
