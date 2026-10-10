"""Reconstruct compact central electron/hole populations at saved Hartree U; never rerun SCF."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .atomic_io import atomic_json, canonical_json
from .band_gaps import four_flavor_band_gaps, GAP_FILLINGS
from .continuum import TDBGContinuumHamiltonian
from .output_metadata import parameters_from_args, hamiltonian_metadata, hamiltonian_metadata_matches
from .lattice import make_uniform_mbz_grid
from .scf_reuse import _source_config
from .scan_checkpoint import digest
from .stoner.active_polarization import FLAVORS, _occupation_record, full_spectrum_active_occupations


def _source_identity(run, row, config_sha=None):
    import tdbg_scf.stoner.active_polarization as counter
    return {**{key: row[key] for key in ('label','n_index','D_index','n_cm2','D_Vnm')},
        'source_scf_U_meV': [row['U'+str(i)+'_meV'] for i in range(1,5)],
        'source_nu_f': [row['nu_'+f] for f in FLAVORS],
        'A_M_A2': float(row['A_M_A2']),
        'source_config_sha256': config_sha or hashlib.sha256((Path(run)/'config_used.json').read_bytes()).hexdigest(),
        'occupation_counter_sha256': hashlib.sha256(Path(counter.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}


def load_cached_active_counts(run, row, directory, config_sha=None):
    path=Path(directory)/(row['label']+'.json')
    if not path.is_file():
        return None
    envelope=json.loads(path.read_text(encoding='utf-8'))
    checksum=envelope.pop('checksum',None)
    if checksum is None or checksum != digest(envelope):
        raise ValueError('Active occupation cache checksum mismatch: '+row['label'])
    expected=_source_identity(run,row,config_sha)
    if canonical_json(envelope.get('source')) != canonical_json(expected):
        raise ValueError('Active occupation cache source mismatch: '+row['label'])
    record = _occupation_record(envelope['active_electron_nu_f'], envelope['active_hole_nu_f'],
                                expected['source_nu_f'], envelope.get('method',
                                'reconstructed_full_spectrum_T0_active_electrons_and_holes'))
    saved = np.asarray(envelope['active_occupation_nu_f'], dtype=float)
    if (saved.shape != (4,) or not np.isfinite(saved).all() or
            not np.allclose(saved, record['active_occupation_nu_f'], rtol=0., atol=1e-8)):
        raise ValueError('Active carrier cache populations are inconsistent: ' + row['label'])
    return record


def _source_model(cfg):
    from argparse import Namespace
    args=Namespace(**cfg['args'])
    K=TDBGContinuumHamiltonian(parameters_from_args(args,1))
    Kp=TDBGContinuumHamiltonian(parameters_from_args(args,-1))
    source=K if int(cfg['hamiltonian']['valley'])==1 else Kp
    if not hamiltonian_metadata_matches(cfg['hamiltonian'], hamiltonian_metadata(source)):
        raise ValueError('Saved active-occupation Hamiltonian source mismatch')
    k,w=make_uniform_mbz_grid(source.geom,int(args.grid_n1),int(args.grid_n2))
    return K,Kp,k,w


def _validate_saved_gap_edges(row, K, Kp):
    gaps=four_flavor_band_gaps(K,Kp)
    saved=np.array([row[key] for key in gaps],dtype=float)
    calculated=np.array(list(gaps.values()),dtype=float)
    if not np.isfinite(saved).all() or not np.isfinite(calculated).all():
        raise ValueError('Saved and reconstructed band edges must be finite: '+row['label'])
    error=float(np.max(np.abs(saved-calculated)))
    if not np.isfinite(error) or error > 1e-7:
        raise ValueError('Reconstructed spectrum disagrees with saved band edges: '+row['label'])
    return error


def _reconstruct_worker(payload):
    run, cfg, row, directory, config_sha=payload
    start=time.monotonic()
    ham_K,ham_Kp,k,w=_source_model(cfg)
    U=np.array([row['U'+str(i)+'_meV'] for i in range(1,5)],dtype=float)
    if not np.isfinite(U).all() or not row['scf_converged']:
        raise ValueError('Cannot reconstruct an unconverged/nonfinite Hartree point')
    spectra=[]
    for ham in (ham_K,ham_Kp):
        energies=np.empty((len(k),ham.dim))
        for index,(kx,ky) in enumerate(k):
            energies[index]=np.linalg.eigvalsh(ham.hamiltonian(float(kx),float(ky),U))
        spectra.append(energies)
    error=_validate_saved_gap_edges(row,*spectra)
    nu=[row['nu_'+f] for f in FLAVORS]
    record=full_spectrum_active_occupations(*spectra,w,float(row['A_M_A2']),nu)
    envelope={'source':_source_identity(run,row,config_sha),
        'active_occupation_nu_f':record['active_occupation_nu_f'],
        'active_electron_nu_f':record['active_electron_nu_f'],
        'active_hole_nu_f':record['active_hole_nu_f'],
        'formula_version':'active_electron_minus_hole_v1',
        'method':'reconstructed_full_spectrum_T0_active_electrons_and_holes',
        'full_band_count_per_flavor':ham_K.dim,'kpoint_count':len(k),
        'saved_gap_max_difference_meV':error,'hartree_iterations_performed':0,
        'stoner_minimizations_performed':0,'runtime_s':time.monotonic()-start}
    envelope['checksum']=digest(envelope)
    atomic_json(Path(directory)/(row['label']+'.json'),envelope)
    return {'label':row['label'],'runtime_s':envelope['runtime_s'],'gap_error_meV':error}


def reconstruct_active_count_cache(run, rows, directory, workers=1):
    """Resume compact count reconstruction, verifying original physics sources."""
    cfg=_source_config(run)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    config_sha=hashlib.sha256((Path(run)/'config_used.json').read_bytes()).hexdigest()
    pending=[]; reused=0
    gap_keys=[key for label in GAP_FILLINGS for key in ('gap_'+label+'_meV',
        'indirect_gap_'+label+'_meV','gap_'+label+'_lower_edge_meV','gap_'+label+'_upper_edge_meV')]
    keys=['label','n_index','D_index','n_cm2','D_Vnm','A_M_A2','scf_converged']
    keys += ['U'+str(i)+'_meV' for i in range(1,5)] + ['nu_'+f for f in FLAVORS] + gap_keys
    for row in rows:
        if load_cached_active_counts(run,row,directory,config_sha) is not None:
            reused+=1
        else:
            pending.append({key:row[key] for key in keys})
    if not pending:
        return {'reconstructed_points':0,'reused_count_cache_points':reused,'cache_directory':str(directory)}
    active_workers=min(max(1,int(workers)),len(pending))
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    print('Reconstructing active electrons and holes at saved Hartree U: '+str(len(pending))+
          ' points; '+str(active_workers)+' workers; SCF iterations=0, Stoner searches=0.',flush=True)
    start=time.monotonic(); done=0; max_gap_error=0.
    with ProcessPoolExecutor(max_workers=active_workers) as pool:
        futures=[pool.submit(_reconstruct_worker,(str(run),cfg,row,str(directory),config_sha)) for row in pending]
        for future in as_completed(futures):
            result=future.result(); done+=1
            max_gap_error=max(max_gap_error,result['gap_error_meV'])
            elapsed=time.monotonic()-start
            eta=elapsed*(len(pending)-done)/done
            print('Active carriers ['+str(done)+'/'+str(len(pending))+'] '+result['label']+
                  '; elapsed '+str(round(elapsed))+' s; ETA '+str(round(eta))+' s',flush=True)
    return {'reconstructed_points':len(pending),'reused_count_cache_points':reused,
            'maximum_saved_gap_difference_meV':max_gap_error,'cache_directory':str(directory)}
