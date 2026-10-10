import numpy as np
import pytest

from tdbg_scf.stoner.active_polarization import (
    FLAVORS, POLARIZATION_FIELDS, count_flavor_active_occupation,
    full_spectrum_active_occupations, polarizations_from_active_carriers,
    certified_active_occupations_from_row,
)


@pytest.mark.parametrize('nu,expected', [
    ([-1.,-1.,1.,1.],[1.,0.,0.]),
    ([1.,-1.,1.,-1.],[0.,1.,0.]),
    ([1.,-1.,-1.,1.],[0.,0.,1.]),
    ([0.,0.,0.,0.],[float("nan")]*3),
    ([1.,1.,1.,1.],[0.,0.,0.]),
])
def test_neutral_saturated_spin_counts_occupied_electrons(nu, expected):
    spectrum=np.array([[-100.,-5.,5.,100.]])
    record=full_spectrum_active_occupations(spectrum,spectrum,np.array([1.]),1.,nu)
    np.testing.assert_allclose(list(record['polarizations'].values()),expected,atol=1e-14)
    np.testing.assert_allclose(record['active_occupation_nu_f'],1.+np.asarray(nu),atol=1e-14)
    assert record['remote_reference_unchanged'] is True


def test_deep_balanced_bands_do_not_dilute_active_spin_polarization():
    for e in ([-100.,-5.,5.,100.],[-200.,-100.,-5.,5.,100.,200.]):
        spectrum=np.array([e])
        record=full_spectrum_active_occupations(spectrum,spectrum,[1.],1.,[-1.,-1.,1.,1.])
        assert record['polarizations']['spin_polarization_active_norm'] == 1.
        assert record['active_occupation_nu_f'] == [0.,0.,2.,2.]


def test_empty_active_manifold_has_balanced_holes():
    spectrum=np.array([[-100.,-5.,5.,100.]])
    record=full_spectrum_active_occupations(spectrum,spectrum,[1.],1.,[-1.,-1.,-1.,-1.])
    assert list(record['polarizations'].values()) == [0.,0.,0.]


def test_remote_overlap_requires_actual_band_occupations():
    e=np.array([[-10.,-3.,10.,12.],[-1.,1.,11.,13.]])
    assert count_flavor_active_occupation(e,[.5,.5],1.,-1.) == .5
    record=full_spectrum_active_occupations(e,e,[.5,.5],1.,[-1.,-1.,0.,0.])
    assert record['active_occupation_nu_f'] == [.5,.5,1.,1.]
    assert record['remote_net_filling_nu_f'] == [-.5,-.5,0.,0.]
    assert record['remote_reference_unchanged'] is False


def test_degenerate_active_remote_states_share_fractional_occupation():
    e=np.array([[-10.,-2.,2.,10.],[-2.,0.,4.,11.]])
    # target one occupied full-spectrum state per cell; the -2 pair shares
    # the remaining half state equally between an active and a remote state.
    assert count_flavor_active_occupation(e,[.5,.5],1.,-1.) == .25
    assert count_flavor_active_occupation(e[::-1],[.5,.5],1.,-1.) == .25


def test_only_actual_minority_occupation_prevents_exactly_one():
    values=polarizations_from_active_carriers([1.,0.,1e-18,0.],[0.,0.,0.,0.])
    assert values[POLARIZATION_FIELDS[0]] < 1.
    assert values[POLARIZATION_FIELDS[1]] == 1.
    with pytest.raises(ValueError):
        polarizations_from_active_carriers([-1.,0.,1.,0.],[0.,0.,0.,0.])
    with pytest.raises(ValueError):
        count_flavor_active_occupation([[-10.,-2.,2.,10.]],[.5],1.,0.)


def _row():
    return dict(status='ok',stoner_dos_valid=True,stoner_success=True,
        gap_spectrum_source='full_continuum_at_scf_U_before_stoner',
        indirect_gap_nu0_meV=2.,gap_nu0_lower_edge_meV=-1.,gap_nu0_upper_edge_meV=1.,
        indirect_gap_nu_minus4_meV=5.,gap_nu_minus4_lower_edge_meV=-10.,
        gap_nu_minus4_upper_edge_meV=-5.,
        indirect_gap_nu_plus4_meV=5.,gap_nu_plus4_lower_edge_meV=5.,
        gap_nu_plus4_upper_edge_meV=10.,
        nu_K_up=1.5,nu_Kp_up=1.5,nu_K_down=-.5,nu_Kp_down=-.5,
        mu_K_up_meV=12.,mu_Kp_up_meV=12.,mu_K_down_meV=-2.,mu_Kp_down_meV=-2.)


def test_isolated_outer_gaps_certify_capped_actual_active_occupation():
    result=certified_active_occupations_from_row(_row())
    assert result['active_occupation_nu_f'] == [2.,2.,.5,.5]
    assert result['remote_net_filling_nu_f'] == [.5,.5,0.,0.]
    assert result['polarizations'][POLARIZATION_FIELDS[0]] == 1.


def test_outer_overlap_requires_strict_flavor_edge_certificates():
    row=_row()
    row.update(indirect_gap_nu_minus4_meV=-2.,gap_nu_minus4_lower_edge_meV=-3.,
               gap_nu_minus4_upper_edge_meV=-5.)
    assert certified_active_occupations_from_row(row)['active_occupation_nu_f'] == [2.,2.,.5,.5]
    row['mu_K_down_meV']=-4.
    assert certified_active_occupations_from_row(row) is None
    row['mu_K_down_meV']=-3.
    assert certified_active_occupations_from_row(row) is None


def test_invalid_equilibrium_wrong_source_and_inconsistent_edges_not_accepted():
    row=_row(); row['stoner_dos_valid']=False
    assert certified_active_occupations_from_row(row) is None
    row=_row(); row['gap_spectrum_source']='shifted_stoner_spectrum'
    assert certified_active_occupations_from_row(row) is None
    row=_row(); row['indirect_gap_nu_plus4_meV']=9.
    with pytest.raises(ValueError,match='inconsistent'):
        certified_active_occupations_from_row(row)


def test_central_overlap_without_strict_mu_certificate_cannot_infer_carriers_from_occupation():
    row=_row()
    row.update(indirect_gap_nu0_meV=-2.,gap_nu0_lower_edge_meV=1.,gap_nu0_upper_edge_meV=-1.,
               mu_K_down_meV=0.,mu_Kp_down_meV=0.)
    assert certified_active_occupations_from_row(row) is None


def test_central_overlap_outside_flavor_mu_interval_can_still_certify_carriers():
    row=_row()
    row.update(indirect_gap_nu0_meV=-2.,gap_nu0_lower_edge_meV=1.,gap_nu0_upper_edge_meV=-1.)
    result=certified_active_occupations_from_row(row)
    assert result['active_electron_nu_f']==[1.,1.,0.,0.]
    assert result['active_hole_nu_f']==[0.,0.,.5,.5]


def test_strict_central_gap_chemical_potential_removes_aggregate_roundoff_carriers():
    from tdbg_scf.stoner.active_polarization import carriers_from_certified_occupations
    row=_row()
    row.update({'nu_'+f:0. for f in FLAVORS})
    row.update({'mu_'+f+'_meV':0. for f in FLAVORS})
    record=carriers_from_certified_occupations(row,[1.+2e-15,1.-2e-15,1.,1.],'test_certificate')
    assert record['active_electron_nu_f']==[0.]*4
    assert record['active_hole_nu_f']==[0.]*4


@pytest.mark.parametrize('population,mu', [(1.2,-2.),(.8,2.)])
def test_chemical_potential_carrier_certificate_rejects_contradictory_occupations(population,mu):
    from tdbg_scf.stoner.active_polarization import carriers_from_certified_occupations
    row=_row()
    row.update(indirect_gap_nu0_meV=-2.,gap_nu0_lower_edge_meV=1.,gap_nu0_upper_edge_meV=-1.)
    row['mu_K_up_meV']=mu
    with pytest.raises(ValueError,match='contradicts'):
        carriers_from_certified_occupations(row,[population,2.,.5,.5],'test_certificate')


@pytest.mark.parametrize('nu,expected_e,expected_h', [
    ([1e-18,0.,1.,1.],[1e-18,0.,1.,1.],[0.,0.,0.,0.]),
    ([-1e-18,0.,-1.,-1.],[0.,0.,0.,0.],[1e-18,0.,1.,1.]),
])
def test_remote_isolation_preserves_tiny_exact_flavor_filling(nu,expected_e,expected_h):
    row=_row()
    row.update({'nu_'+f:n for f,n in zip(FLAVORS,nu)})
    # Saved interpolated chemical potentials may lie inside the gap for a tiny
    # filling. The certified fixed population retains that real minority.
    row.update({'mu_'+f+'_meV':2.*np.sign(n) if abs(n)>.9 else 0. for f,n in zip(FLAVORS,nu)})
    result=certified_active_occupations_from_row(row)
    assert result['active_electron_nu_f']==expected_e
    assert result['active_hole_nu_f']==expected_h
    assert result['polarizations'][POLARIZATION_FIELDS[0]] < 1.


def test_rounded_full_central_population_does_not_certify_an_overlap():
    from tdbg_scf.stoner.active_polarization import carriers_from_certified_occupations
    row=_row()
    row.update(indirect_gap_nu0_meV=-4.,gap_nu0_lower_edge_meV=4.,gap_nu0_upper_edge_meV=0.,
        mu_K_up_meV=3.,mu_Kp_up_meV=3.,mu_K_down_meV=-6.,mu_Kp_down_meV=-6.)
    # Ns=2 can be rounded from 2-tiny_hole. It cannot certify h=0 while
    # chemical potentials lie inside the indirect central band overlap.
    assert carriers_from_certified_occupations(row,[2.,2.,.25,.25],'test_overlap') is None
