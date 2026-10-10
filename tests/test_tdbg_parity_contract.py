"""Regressions for the audit, without changing the original physical model."""
import argparse
import importlib.util
import numpy as np
import pytest
from tdbg_scf import TDBGParameters, TDBGContinuumHamiltonian
from tdbg_scf.stoner.dos import FlavorBandTable
from tdbg_scf.stoner.energy import stoner_local_stability, stoner_energy_meV_per_cell
from tdbg_scf.stoner.params import StonerParams


def test_original_hamiltonian_and_entrypoint_physical_defaults(monkeypatch):
    from examples import run_full_band_stoner as point, run_full_scf as hartree
    p = TDBGParameters()
    assert (p.theta_deg,p.cutoff,p.gamma0_meV,p.gamma1_meV,p.gamma3_meV,p.gamma4_meV,p.sublattice_Z_meV)==(1.35,2,2610.,361.,283.,138.,15.)
    parser=argparse.ArgumentParser()
    point.add_hamiltonian_args(parser); point.add_full_scf_args(parser); point.add_stoner_args(parser)
    args=parser.parse_args([])
    assert (args.eps_perp,args.d_layer_nm,args.D_sign,args.kBT_meV)==(3.,.335,-1.,.2)
    assert (args.u0_meV_A2,args.JH_meV_A2)==(79000.,24000.)
    class Captured(Exception): pass
    captured=[]
    def capture(parser,*a,**kw):
        captured.append(parser.get_default('eps_perp'))
        raise Captured
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',capture)
    with pytest.raises(Captured): hartree.main()
    assert captured == [4.]
    ham=TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    H=ham.hamiltonian(0.,0.,np.zeros(4))
    assert H[0,3] == pytest.approx(361.)
    assert H[5,6] == pytest.approx(361.)
    np.testing.assert_allclose(H,H.conj().T)


def test_endpoint_instability_above_tolerance_is_not_accepted():
    free=FlavorBandTable(np.array([0.,.5]),np.zeros(2),np.zeros(2),0.,.5)
    kink=FlavorBandTable(np.array([0.,.25,.5]),np.array([-10.,-10.,10.]),np.array([2.5,0.,2.5]),0.,.5)
    tables=[free,free,kink,kink]
    values=np.full(4,.25)
    p=StonerParams(u0_meV_A2=2e-7,tol_energy_meV=1e-8)
    gain=stoner_energy_meV_per_cell(values,tables,p,1.)-stoner_energy_meV_per_cell([.4975,.0025,.25,.25],tables,p,1.)
    assert gain > p.tol_energy_meV
    assert not stoner_local_stability(values,tables,p,1.)['valid']


def test_four_layer_block_weights_are_lower_bilayer_probabilities():
    assert importlib.util.find_spec('tdbg_scf.block_projection'), 'Missing four-layer block observables'
    from tdbg_scf.block_projection import lower_block_weights_from_evecs
    ham=TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    states=np.eye(8,dtype=complex)
    masks=ham.layer_projectors_diagonal()
    np.testing.assert_array_equal(lower_block_weights_from_evecs(states,masks),[0,0,0,0,1,1,1,1])
    states[:,0]=0.; states[0,0]=np.sqrt(.25); states[4,0]=np.sqrt(.75)
    assert lower_block_weights_from_evecs(states,masks)[0] == pytest.approx(.75)


def test_metadata_reports_physical_stack_and_displacement_direction():
    from tdbg_scf.output_metadata import hamiltonian_metadata,displacement_metadata
    ham=TDBGContinuumHamiltonian(TDBGParameters(cutoff=0))
    metadata=hamiltonian_metadata(ham)
    assert metadata.get('stacking_top_to_bottom') == 'BA-AB'
    assert metadata['n_layers']==4 and metadata['gamma4_meV']==138.
    assert displacement_metadata(-1).get('D_positive_direction') == 'bottom_to_top'
    assert displacement_metadata(1).get('D_positive_direction') == 'top_to_bottom'


def test_indirect_band_gaps_use_both_valleys():
    assert importlib.util.find_spec('tdbg_scf.band_gaps'), 'Missing integer gap observables'
    from tdbg_scf.band_gaps import four_flavor_band_gaps,stoner_integer_band_gaps
    energy=np.array([[-20.,-8.,-1.,2.,7.,20.],[-19.,-6.,0.,3.,9.,21.]])
    assert four_flavor_band_gaps(energy,energy)['gap_nu0_meV']==2.
    after=stoner_integer_band_gaps(energy,energy,[0.,0.,0.,0.])
    assert after['stoner_gap_nu_plus1_meV']==0.
