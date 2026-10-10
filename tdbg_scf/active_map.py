"""Append signed electron/hole active-band polarizations to an existing completed scan."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .atomic_io import atomic_json, atomic_npz, atomic_output, canonical_json
from .run_storage import ACTIVE_CARRIER_FIELDS
from .scan_checkpoint import scan_directory_lock, _commit_lock, digest
from .stoner.active_polarization import (
    FLAVORS, POLARIZATION_FIELDS, LABEL_SIGNS, REPLACED_CARRIER_FIELDS, certified_active_occupations_from_row,
    full_spectrum_active_occupations, undefined_polarizations, active_polarization_metadata,
    _occupation_record,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _saved_active_occupations(row):
    """Use exact populations persisted by ordinary full-spectrum calculations."""
    present = [field in row for field in ACTIVE_CARRIER_FIELDS]
    if not any(present):
        return None
    if not all(present):
        raise ValueError('Incomplete saved central electron and hole populations')
    counts = np.asarray([row[field] for field in ACTIVE_CARRIER_FIELDS], dtype=float).reshape(2, 4)
    if np.isnan(counts).all():
        return None  # Live/final tables pad legacy missing fields with NaN.
    return _occupation_record(*counts, [row['nu_' + flavor] for flavor in FLAVORS],
                              'saved_full_spectrum_T0_active_electrons_and_holes')


def _read_spectrum(root, row):
    label=row['label']
    if Path(label).name != label:
        raise ValueError('Unsafe saved point label')
    candidates=(root/'details'/label/'scf'/'spectrum.npz',
                root/'details'/label/'full_band_stoner_cache.npz')
    for path in candidates:
        if not path.is_file():
            continue
        from .scf_reuse import _source_config
        from .active_reconstruction import _source_model, _validate_saved_gap_edges
        from .output_metadata import hamiltonian_metadata
        K,Kp,expected_k,expected_w=_source_model(_source_config(root))
        with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as z:
            required=('evals_K_meV','evals_Kp_meV','weights','kpts','hamiltonian_json','gap_spectrum_source')
            potential='U_meV' if path.name=='spectrum.npz' else 'U_scf_meV'
            if any(key not in z for key in (*required,potential)):
                raise ValueError('Saved spectrum lacks source metadata: '+label)
            if str(z['gap_spectrum_source'].item()) != 'full_continuum_at_scf_U_before_stoner':
                raise ValueError('Saved spectrum source is not pre-Stoner: '+label)
            if path.name=='spectrum.npz' and ('stage' not in z or str(z['stage'].item()) != 'scf_before_stoner'):
                raise ValueError('Saved spectrum stage source mismatch: '+label)
            try:
                saved_ham=json.loads(str(z['hamiltonian_json'].item()))
            except (TypeError,ValueError):
                raise ValueError('Saved spectrum Hamiltonian source is malformed: '+label)
            if canonical_json(saved_ham) != canonical_json(hamiltonian_metadata(K)):
                raise ValueError('Saved spectrum Hamiltonian source mismatch: '+label)
            U=np.array([row['U'+str(i)+'_meV'] for i in range(1,5)],dtype=float)
            saved_U=np.asarray(z[potential],dtype=float)
            if (saved_U.shape != U.shape or not np.isfinite(saved_U).all() or
                    not np.isfinite(U).all() or not np.allclose(saved_U,U,rtol=0.,atol=1e-9)):
                raise ValueError('Saved spectrum Hartree potential source mismatch: '+label)
            for key,expected in (('kpts',expected_k),('weights',expected_w)):
                observed=np.asarray(z[key],dtype=float)
                if (observed.shape != expected.shape or not np.isfinite(observed).all() or
                        not np.allclose(observed,expected,rtol=1e-12,atol=1e-14)):
                    raise ValueError('Saved spectrum '+key+' source mismatch: '+label)
            spectra=[z['evals_K_meV'],z['evals_Kp_meV']]
            if any(e.shape != (len(expected_k),K.dim) for e in spectra):
                raise ValueError('Saved spectrum complete band/mesh shape source mismatch: '+label)
            _validate_saved_gap_edges(row,*spectra)
            nu=[row['nu_'+f] for f in FLAVORS]
            return full_spectrum_active_occupations(*spectra,expected_w,float(row['A_M_A2']),nu)
    raise ValueError('A full saved spectrum is required for point '+label+
                     '; use --reconstruct-missing to count active occupations at saved Hartree U')


def _saved_spectrum_labels(root):
    details=root/'details'
    if not details.is_dir():
        return set()
    return {path.name for path in details.iterdir() if path.is_dir() and
            ((path/'scf'/'spectrum.npz').is_file() or (path/'full_band_stoner_cache.npz').is_file())}


IDENTITY_FIELDS = ('label', 'n_index', 'D_index', 'n_cm2', 'D_Vnm')


def _validate_identity(expected, observed, keys=IDENTITY_FIELDS):
    for key in keys:
        if key not in expected or key not in observed:
            raise ValueError('Missing saved point identity field: ' + key)
        if key == 'label':
            matches = str(expected[key]) == str(observed[key])
        else:
            try:
                a, b = float(expected[key]), float(observed[key])
                matches = np.isfinite([a, b]).all() and a == b
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ValueError('Saved point identity mismatch: ' + key + ' for ' + str(expected['label']))


def _validate_grid_rows(config, rows):
    axes = [np.asarray(config[key], dtype=float) for key in ('n_values_cm2', 'D_values_Vnm')]
    if any(a.ndim != 1 or a.size == 0 or not np.isfinite(a).all() or
           np.unique(a).size != a.size for a in axes):
        raise ValueError('Invalid saved scan axes')
    occupied = set()
    for row in rows:
        label = row.get('label')
        if not isinstance(label, str) or not label or label in ('.', '..') or '/' in label or '\\' in label:
            raise ValueError('Unsafe saved point label')
        try:
            i, j = int(row['n_index']), int(row['D_index'])
            correct = (float(row['n_index']) == i and float(row['D_index']) == j and
                       0 <= i < len(axes[0]) and 0 <= j < len(axes[1]) and
                       float(row['n_cm2']) == axes[0][i] and float(row['D_Vnm']) == axes[1][j])
        except (KeyError, TypeError, ValueError, OverflowError):
            correct = False
        if not correct or (i, j) in occupied:
            raise ValueError('Saved point indices do not uniquely match scan axes')
        occupied.add((i, j))
    return tuple(len(a) for a in axes)


def _csv_with_new_columns(data, values, rows):
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    original = list(reader)
    if not original:
        raise ValueError('Empty saved map CSV')
    header = original[0]
    if len(header) != len(set(header)) or 'label' not in header:
        raise ValueError('Saved map CSV has invalid column labels')
    if any(len(r) != len(header) for r in original[1:]):
        raise ValueError('Saved CSV has an incomplete row')
    by_label = {row['label']: row for row in rows}
    label_index = header.index('label')
    labels = [r[label_index] for r in original[1:]]
    if len(labels) != len(set(labels)) or set(labels) != set(values):
        raise ValueError('CSV/JSON saved point identities differ')
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    new_header = [field for field in header if field not in REPLACED_CARRIER_FIELDS] + [field for field in POLARIZATION_FIELDS if field not in header]
    writer.writerow(new_header)
    for original_row in original[1:]:
        if len(original_row) != len(header):
            raise ValueError('Saved CSV has an incomplete row')
        row = dict(zip(header, original_row))
        _validate_identity(by_label[row['label']], row)
        row.update({field: (format(value, '.17g') if np.isfinite(value) else '')
                    for field, value in values[row['label']].items()})
        writer.writerow([row[field] for field in new_header])
    return output.getvalue().encode('utf-8')


def _updated_npz(data, rows, grids, config):
    by_label = {row['label']: row for row in rows}
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data), 'r') as old, zipfile.ZipFile(
            result, 'w', compression=zipfile.ZIP_DEFLATED) as new:
        for key in ('n_values_cm2', 'D_values_Vnm'):
            entry = key + '.npy'
            if entry in old.namelist():
                saved_axis = np.load(io.BytesIO(old.read(entry)), allow_pickle=False)
                if not np.array_equal(saved_axis, np.asarray(config[key], dtype=float)):
                    raise ValueError('Scalar NPZ axes do not match saved config: ' + key)
        record_data = None
        if 'rows.npy' in old.namelist():
            old_records = np.load(io.BytesIO(old.read('rows.npy')), allow_pickle=True)
            if old_records.dtype.names is None or 'label' not in old_records.dtype.names:
                raise ValueError('Scalar NPZ rows do not carry point labels')
            saved_labels = [str(r['label']) for r in old_records]
            if len(saved_labels) != len(set(saved_labels)) or set(saved_labels) != {r['label'] for r in rows if r.get('status') == 'ok'}:
                raise ValueError('Scalar NPZ/map point identity mismatch')
            identity_keys = tuple(key for key in IDENTITY_FIELDS if key in old_records.dtype.names)
            dtype = [(name, old_records.dtype.fields[name][0]) for name in old_records.dtype.names if name not in REPLACED_CARRIER_FIELDS]
            dtype += [(field, 'f8') for field in POLARIZATION_FIELDS
                      if field not in old_records.dtype.names]
            records = np.empty(old_records.shape, dtype=dtype)
            for name in old_records.dtype.names:
                if name in REPLACED_CARRIER_FIELDS:
                    continue
                records[name] = old_records[name]
            for index, original in enumerate(old_records):
                label = str(original['label'])
                _validate_identity(by_label[label], {key: original[key] for key in identity_keys}, identity_keys)
                for field in POLARIZATION_FIELDS:
                    records[field][index] = by_label[label][field]
            buffer = io.BytesIO()
            np.save(buffer, records, allow_pickle=True)
            record_data = buffer.getvalue()
        replaced = {'rows.npy'} if record_data is not None else set()
        replaced.update(field + '.npy' for field in (*POLARIZATION_FIELDS, *REPLACED_CARRIER_FIELDS))
        # Preserve every original non-row NPY payload byte for byte, including
        # NaNs, object metadata and floating-point roundoff.
        for entry in old.infolist():
            if entry.filename not in replaced:
                new.writestr(entry, old.read(entry.filename))
        if record_data is not None:
            new.writestr('rows.npy', record_data)
        for field in POLARIZATION_FIELDS:
            buffer = io.BytesIO()
            np.save(buffer, grids[field], allow_pickle=False)
            new.writestr(field + '.npy', buffer.getvalue())
    return result.getvalue()


def _plot_active_maps(root, config, grids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .gap_maps import _axis_edges
    n_values = np.asarray(config['n_values_cm2'], dtype=float)
    d_values = np.asarray(config['D_values_Vnm'], dtype=float)
    rows = json.loads((root / 'nd_map.json').read_text(encoding='utf-8'))
    nu_values = np.full(len(n_values), np.nan)
    for row in rows:
        if np.isfinite(float(row.get('nu_total', np.nan))):
            nu_values[int(row['n_index'])] = float(row['nu_total'])
    if not np.isfinite(nu_values).all():
        nu_values = n_values / 1e12
        ylabel = r'$n$ ($10^{12}$ cm$^{-2}$)'
    else:
        ylabel = r'$\nu$'
    i_order, j_order = np.argsort(nu_values), np.argsort(d_values)
    y_edges = _axis_edges(nu_values[i_order], .1)
    x_edges = _axis_edges(d_values[j_order], .05)
    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    titles = ('Spin active-band polarization', 'Valley active-band polarization',
              'Spin-valley active-band polarization')
    args = config.get('args', {})
    for field, title in zip(POLARIZATION_FIELDS, titles):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        im = ax.pcolormesh(x_edges, y_edges, grids[field][np.ix_(i_order, j_order)],
                           shading='flat', cmap='viridis', vmin=0., vmax=1.)
        ax.set_facecolor('#eeeeee')
        ax.set_xlabel('D (V/nm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title + '\n' + str(args.get('stacking', 'BA-AB')) +
                     '; theta=' + str(args.get('theta_deg', '')) + ' deg')
        fig.colorbar(im, ax=ax, label='Active carrier polarization (0-1)')
        with atomic_output(directory / ('nd_map_' + field + '.png')) as temporary:
            fig.savefig(temporary, format='png', dpi=220)
        plt.close(fig)


TRANSACTION_NAME='.active_polarization_transaction.json'


def _safe_update_path(root, name):
    relative=Path(name)
    allowed={'nd_map.csv','nd_map.json','nd_map_scalars.npz'}
    for prefix in ('active','carrier'):
        allowed.update({prefix+'_polarizations.npz',prefix+'_polarization_definition.json',
                        prefix+'_polarization_update.json',prefix+'_polarization_validation.json'})
    allowed.update('figures/nd_map_'+field+'.png' for field in (*POLARIZATION_FIELDS,*REPLACED_CARRIER_FIELDS))
    normalized=relative.as_posix()
    checkpoint=(len(relative.parts)==3 and relative.parts[:2]==('checkpoints','points')
                and relative.suffix=='.json')
    detailed=(len(relative.parts)==3 and relative.parts[0]=='details'
              and relative.parts[2]=='stoner_active_occupation.json')
    target=(root/relative).resolve()
    if (relative.is_absolute() or '..' in relative.parts or root not in target.parents or
            (normalized not in allowed and not checkpoint and not detailed)):
        raise ValueError('Unsafe active update journal output path: '+str(name))
    return target


def _recover_pending_update(root):
    """Rollback an interrupted publication before reading or updating its maps."""
    journal=root/TRANSACTION_NAME
    if not journal.is_file():
        return
    state=json.loads(journal.read_text(encoding='utf-8'))
    backup=(root/state['backup_directory']).resolve()
    if (state.get('schema_version') != 1 or (root/'_analysis_backups').resolve() not in backup.parents):
        raise ValueError('Invalid active update recovery backup')
    files=state['files']
    if len({item['path'] for item in files}) != len(files):
        raise ValueError('Duplicate active update recovery paths')
    payloads=[]
    for item in files:
        path=_safe_update_path(root,item['path'])
        data=(backup/item['path']).read_bytes() if item['existed'] else None
        if data is not None and _sha(data) != item['sha256']:
            raise ValueError('Active update recovery backup checksum mismatch')
        payloads.append((path,data))
    def restore(item):
        path,data=item
        if data is None:
            if path.is_file():
                path.unlink()
        else:
            with atomic_output(path) as temporary:
                temporary.write_bytes(data)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(restore,payloads))
    journal.unlink()


def _commit_prepared_update(root,backup,stage,original_payloads,obsolete,report,started):
    staged={p.relative_to(stage).as_posix():p for p in stage.rglob('*') if p.is_file()}
    paths=sorted(set(staged)|set(obsolete))
    for name in paths:
        _safe_update_path(root,name)
    def save_original(item):
        name,data=item
        target=backup/name; target.parent.mkdir(parents=True,exist_ok=True)
        with atomic_output(target) as temporary:
            temporary.write_bytes(data)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save_original,original_payloads.items()))
    state=dict(schema_version=1,backup_directory=backup.relative_to(root).as_posix(),files=[
        dict(path=name,existed=name in original_payloads,
             sha256=_sha(original_payloads[name]) if name in original_payloads else None) for name in paths])
    atomic_json(root/TRANSACTION_NAME,state)
    def publish(name):
        with atomic_output(root/name) as temporary:
            temporary.write_bytes(staged[name].read_bytes())
    try:
        checkpoints=[name for name in staged if name.startswith('checkpoints/')]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(publish,checkpoints))
        report_name='active_polarization_update.json'
        for name in sorted(set(staged)-set(checkpoints)-{report_name}):
            publish(name)
        for name in obsolete:
            if (root/name).is_file():
                (root/name).unlink()
        report['analysis_and_staging_runtime_s']=report['runtime_s']
        report['runtime_s']=time.monotonic()-started
        atomic_json(stage/report_name,report)
        publish(report_name)
    except BaseException:
        _recover_pending_update(root)
        raise
    (root/TRANSACTION_NAME).unlink()


def append_active_columns(run, *, plot_figures=True, reconstruct_missing=False, workers=1):
    root = Path(run).expanduser().resolve()
    started = time.monotonic()
    with scan_directory_lock(root), _commit_lock(root):
        _recover_pending_update(root)
        config = json.loads((root / 'config_used.json').read_text(encoding='utf-8'))
        names = ('nd_map.csv', 'nd_map.json', 'nd_map_scalars.npz')
        original = {name: (root / name).read_bytes() for name in names}
        rows = json.loads(original['nd_map.json'].decode('utf-8'))
        labels = [row['label'] for row in rows]
        if len(labels) != len(set(labels)):
            raise ValueError('Saved map contains duplicate points')
        shape = _validate_grid_rows(config, rows)
        available=_saved_spectrum_labels(root)
        details_root=root/'details'
        existing_details=({p.name for p in details_root.iterdir() if p.is_dir() and
                           (p/'stoner_active_occupation.json').is_file()}
                          if details_root.is_dir() else set())
        detail_labels=available|existing_details
        detail_records={}
        cache_dir=root/'.active_occupation_cache'
        reconstruction={}
        if reconstruct_missing:
            from .active_reconstruction import reconstruct_active_count_cache
            needed=[row for row in rows if row.get('status')=='ok' and row.get('stoner_dos_valid') is True
                    and _saved_active_occupations(row) is None
                    and certified_active_occupations_from_row(row) is None and row['label'] not in available]
            reconstruction=reconstruct_active_count_cache(root,needed,cache_dir,workers)
        config_sha=_sha((root/'config_used.json').read_bytes())
        grids = {field: np.full(shape, np.nan) for field in POLARIZATION_FIELDS}
        occupied = np.full((*shape, 4), np.nan)
        electrons = np.full((*shape, 4), np.nan)
        holes = np.full((*shape, 4), np.nan)
        active_net = np.full((*shape, 4), np.nan)
        remote_net = np.full((*shape, 4), np.nan)
        methods = Counter()
        invalid = zero = finite = 0
        detail_crosschecks = []
        values = {}
        for row in rows:
            i, j = int(row['n_index']), int(row['D_index'])
            if not (0 <= i < shape[0] and 0 <= j < shape[1]):
                raise ValueError('Point indices lie outside saved scan axes')
            if row.get('status') != 'ok' or row.get('stoner_dos_valid') is not True:
                invalid += 1
                metrics = undefined_polarizations()
                detail_record={key:[float('nan')]*4 for key in
                    ('active_electron_nu_f','active_hole_nu_f','active_carrier_nu_f',
                     'active_occupation_nu_f','active_net_filling_nu_f','remote_net_filling_nu_f')}
                detail_record.update(method='undefined_invalid_common_fermi_level',polarizations=metrics)
            else:
                record = _saved_active_occupations(row)
                if record is None:
                    record = certified_active_occupations_from_row(row)
                if record is None:
                    if row['label'] in available:
                        record=_read_spectrum(root,row)
                    else:
                        from .active_reconstruction import load_cached_active_counts
                        record=load_cached_active_counts(root,row,cache_dir,config_sha)
                        if record is None:
                            record=_read_spectrum(root,row)
                # Crosscheck certificates against a saved full spectrum, then
                # use that exact record for the map, sidecar and detailed file.
                if row['label'] in available and record['method'].startswith(('certified', 'saved_full_spectrum')):
                    exact = _read_spectrum(root, row)
                    error = max(float(np.max(np.abs(np.asarray(exact[key]) - record[key])))
                                for key in ('active_electron_nu_f','active_hole_nu_f','active_occupation_nu_f',
                                            'active_net_filling_nu_f','remote_net_filling_nu_f'))
                    if error > 1e-8:
                        raise ValueError('Saved spectrum contradicts active-occupation certificate: ' + row['label'])
                    detail_crosschecks.append({'label': row['label'], 'max_count_difference': error})
                    record = exact
                methods[record['method']] += 1
                metrics = record['polarizations']
                occupied[i,j] = record['active_occupation_nu_f']
                electrons[i,j] = record['active_electron_nu_f']
                holes[i,j] = record['active_hole_nu_f']
                active_net[i,j] = record['active_net_filling_nu_f']
                remote_net[i,j] = record['remote_net_filling_nu_f']
                if np.isfinite(list(metrics.values())).all():
                    finite += 1
                else:
                    zero += 1
                detail_record=record
            if row['label'] in detail_labels:
                detail_records[row['label']]={**detail_record,
                    'definition':active_polarization_metadata(),
                    'status':row.get('status'),'stoner_dos_valid':row.get('stoner_dos_valid') is True}
            for old_field in REPLACED_CARRIER_FIELDS:
                row.pop(old_field, None)
            row.update(metrics)
            values[row['label']] = metrics
            for field in POLARIZATION_FIELDS:
                grids[field][i,j] = metrics[field]
        csv_payload = _csv_with_new_columns(original['nd_map.csv'], values, rows)
        npz_payload = _updated_npz(original['nd_map_scalars.npz'], rows, grids, config)
        def prepare_checkpoint(row):
            path = root / 'checkpoints' / 'points' / (row['label'] + '.json')
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                return None
            envelope = json.loads(data.decode('utf-8'))
            checksum = envelope.pop('checksum', None)
            if checksum is None or checksum != digest(envelope):
                raise ValueError('Original checkpoint checksum mismatch: ' + row['label'])
            point = envelope['point']
            _validate_identity(row, point)
            source_keys={'status','scf_status','scf_converged','stoner_dos_valid','stoner_success',
                         'gap_spectrum_source','A_M_A2','nu_total'}
            if (row.get('status') == 'ok' and row.get('stoner_dos_valid') is True
                    and _saved_active_occupations(row) is not None):
                source_keys.update(ACTIVE_CARRIER_FIELDS)
            source_keys.update(config.get('hamiltonian',{}))
            source_keys.update(key for key in row if key.startswith(('nu_','mu_','gap_nu','indirect_gap_nu')))
            source_keys.update('U'+str(i)+'_meV' for i in range(1,5))
            for key in source_keys.intersection(row):
                expected,observed=row[key],point.get(key)
                invalid=(row.get('status')!='ok' or row.get('stoner_dos_valid') is not True)
                if key not in point and invalid and isinstance(expected,(int,float)) and np.isnan(expected):
                    continue  # exported NaN column absent from a failed raw point
                if key not in point or expected != observed:
                    if not (isinstance(expected,(int,float)) and isinstance(observed,(int,float))
                            and np.isnan(expected) and np.isnan(observed)):
                        raise ValueError('Checkpoint physics source mismatch: '+row['label']+' / '+key)
            for old_field in REPLACED_CARRIER_FIELDS:
                point.pop(old_field, None)
            point.update(values[row['label']])
            envelope['checksum'] = digest(envelope)
            return path, data, envelope
        # Validate all independent point files before publishing any changes.
        # Parallel reads avoid thousands of serial mapped-drive round trips.
        with ThreadPoolExecutor(max_workers=8) as pool:
            checkpoint_updates = [item for item in pool.map(prepare_checkpoint, rows)
                                  if item is not None]
        backup = root / '_analysis_backups' / ('active_polarization_' +
                  datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir(parents=True)
        legacy_names=['carrier_polarizations.npz','carrier_polarization_definition.json',
                      'carrier_polarization_update.json','carrier_polarization_validation.json']
        legacy_names += ['figures/nd_map_'+field+'.png' for field in REPLACED_CARRIER_FIELDS]
        legacy_payloads={name:(root/name).read_bytes() for name in legacy_names if (root/name).is_file()}
        stage=backup/'.staged'
        stage.mkdir()
        original_payloads=dict(original)
        original_payloads.update(legacy_payloads)
        existing_analysis=['active_polarizations.npz','active_polarization_definition.json',
                           'active_polarization_update.json','active_polarization_validation.json']
        existing_analysis += ['figures/nd_map_'+field+'.png' for field in POLARIZATION_FIELDS]
        for name in existing_analysis:
            if (root/name).is_file():
                original_payloads[name]=(root/name).read_bytes()
        for label,record in detail_records.items():
            relative='details/'+label+'/stoner_active_occupation.json'
            target=_safe_update_path(root,relative)
            if target.is_file():
                original_payloads[relative]=target.read_bytes()
            atomic_json(stage/relative,record)
        def stage_checkpoint(update):
            path,data,envelope=update
            relative=path.relative_to(root).as_posix()
            analysis='details/'+envelope['point']['label']+'/stoner_active_occupation.json'
            if analysis in envelope.get('artifacts', {}) and (stage/analysis).is_file():
                payload=(stage/analysis).read_bytes()
                envelope['artifacts'][analysis]={**envelope['artifacts'][analysis],
                    'size':len(payload),'sha256':_sha(payload)}
                envelope.pop('checksum', None)
                envelope['checksum']=digest(envelope)
            atomic_json(stage/relative,envelope)
            return relative,data
        with ThreadPoolExecutor(max_workers=8) as pool:
            original_payloads.update(pool.map(stage_checkpoint,checkpoint_updates))
        (stage/'nd_map.csv').write_bytes(csv_payload)
        atomic_json(stage/'nd_map.json',rows)
        (stage/'nd_map_scalars.npz').write_bytes(npz_payload)
        atomic_npz(stage / 'active_polarizations.npz', **grids,
                   active_electron_nu_f=electrons, active_hole_nu_f=holes,
                   active_carrier_nu_f=electrons+holes, active_occupation_nu_f=occupied,
                   active_net_filling_nu_f=active_net, remote_net_filling_nu_f=remote_net,
                   formula_version=np.array('active_electron_minus_hole_v1'),
                   flavor_order=np.array(FLAVORS), label_signs=LABEL_SIGNS,
                   n_values_cm2=np.asarray(config['n_values_cm2']),
                   D_values_Vnm=np.asarray(config['D_values_Vnm']))
        atomic_json(stage / 'active_polarization_definition.json',active_polarization_metadata())
        if plot_figures:
            _plot_active_maps(stage,config,grids)
        report = {
            'run_directory': str(root), 'fields_added': list(POLARIZATION_FIELDS),
            'formula_version': 'active_electron_minus_hole_v1',
            'total_points': len(rows), 'finite_points': finite,
            'invalid_equilibrium_points': invalid, 'zero_active_carrier_points': zero,
            'count_methods': dict(methods), 'saved_spectrum_crosschecks': detail_crosschecks,
            'backup_directory': str(backup), 'checkpoints_updated': len(checkpoint_updates),
            'detailed_records_updated': len(detail_records),
            'original_sha256': {name: _sha(payload) for name,payload in original.items()},
            'hartree_iterations_performed': 0, 'stoner_minimizations_performed': 0,
            'valley_spectrum_passes_performed': 2*int(reconstruction.get('reconstructed_points',0)),
            'diagonalizations_performed': (2*int(reconstruction.get('reconstructed_points',0))*
                int(config.get('args',{}).get('grid_n1',0))*int(config.get('args',{}).get('grid_n2',0))),
            'occupation_reconstruction':reconstruction, 'runtime_s': time.monotonic() - started,
        }
        atomic_json(stage / 'active_polarization_update.json',report)
        obsolete=list(legacy_payloads)
        if 'active_polarization_validation.json' in original_payloads:
            obsolete.append('active_polarization_validation.json')
        _commit_prepared_update(root,backup,stage,original_payloads,obsolete,report,started)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--reconstruct-missing', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(append_active_columns(args.run, plot_figures=not args.no_figures, reconstruct_missing=args.reconstruct_missing, workers=args.workers), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
