"""Export saved full Hartree/Stoner details as tab-separated Origin TXT."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from io import BytesIO
from .continuum import TDBGParameters, TDBGContinuumHamiltonian
from .output_metadata import hamiltonian_metadata
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from .atomic_io import atomic_json, atomic_output
from .lattice import MoireGeometry
from .origin_geometry import PeriodicBandSurface, UndersampledReciprocalGrid, first_mbz_boundary, square_mesh, zero_level_lines, pack_lines
from .block_projection import get_saved_block_data
from .block_plotting import render_block_stage
from .origin_block_export import band_block_table, surface_block_tables, block_dos_table

FLAVORS = ("K_up", "Kp_up", "K_down", "Kp_down")
FORMAT_VERSION = 1


def _load(path):
    # Old detailed caches contain multi-GB eigenvectors; never read those
    # arrays into RAM merely to export spectra. Buffer only compact files.
    path = Path(path)
    source = BytesIO(path.read_bytes()) if path.stat().st_size < 64*1024**2 else path
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if not key.startswith("evecs_")}
    if "hamiltonian_json" not in data and "hamiltonian_parameters_json" in data:
        parameters = json.loads(str(np.asarray(data["hamiltonian_parameters_json"]).item()))
        from dataclasses import fields
        names = {field.name for field in fields(TDBGParameters)}
        if not names.issubset(parameters):
            raise ValueError("Old Origin input requires complete saved Hamiltonian parameters")
        ham = TDBGContinuumHamiltonian(TDBGParameters(**{key:parameters[key] for key in names}))
        data["hamiltonian_json"] = np.asarray(json.dumps(hamiltonian_metadata(ham),sort_keys=True))
    return data


def _geometry(spectrum):
    hamiltonian = json.loads(str(spectrum["hamiltonian_json"].item()))
    return MoireGeometry(theta_deg=hamiltonian["theta_deg"], a_cc_A=hamiltonian["a_cc_A"]), hamiltonian


def gaussian_dos_window(energies, weights, mu_values, centers, sigma, block_size=512, *, state_weights=None):
    """Same full-spectrum Gaussian DOS with bounded working memory.

    Omit only states farther than 12 sigma from the complete requested window;
    each omitted kernel is below exp(-72)/(sqrt(2*pi)*sigma).
    """
    weights, centers = np.asarray(weights,float), np.asarray(centers,float)
    if (not np.isfinite(sigma) or sigma <= 0 or centers.ndim != 1
            or not len(centers) or not np.isfinite(centers).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or len(energies) != len(mu_values)):
        raise ValueError("invalid Gaussian DOS inputs")
    result = np.zeros_like(centers)
    normalization = 1/(np.sqrt(2*np.pi)*sigma)
    if state_weights is not None and len(state_weights) != len(energies):
        raise ValueError("DOS projections must match all flavors")
    for index, (bands, mu) in enumerate(zip(energies,mu_values)):
        bands = np.asarray(bands,float)
        if (bands.ndim != 2 or weights.shape != (len(bands),)
                or not np.isfinite(bands).all() or not np.isfinite(mu)):
            raise ValueError("DOS requires finite band arrays, weights and mu")
        flat = bands.ravel()-mu
        flat_weights = np.repeat(weights,bands.shape[1])
        if state_weights is not None:
            projection = np.asarray(state_weights[index], float)
            if (projection.shape != bands.shape or not np.isfinite(projection).all()
                    or np.any(projection < 0) or np.any(projection > 1)):
                raise ValueError("DOS state weights must match bands and lie between zero and one")
            flat_weights *= projection.ravel()
        keep = (flat >= centers.min()-12*sigma) & (flat <= centers.max()+12*sigma)
        flat, flat_weights = flat[keep],flat_weights[keep]
        for start in range(0,len(flat),block_size):
            kernel = (flat[start:start+block_size,None]-centers[None,:])/sigma
            np.square(kernel,out=kernel)
            kernel *= -.5
            np.exp(kernel,out=kernel)
            result += normalization * (flat_weights[start:start+block_size] @ kernel)
    return result


def _table_header(columns, context):
    """Separate Origin's file metadata from its three per-column label rows."""
    names, units = [], []
    suffix_units = (("_Ainv2_meVinv", "Angstrom^-2 meV^-1"),
                    ("_Ainv", "Angstrom^-1"), ("_meV", "meV"))
    for column in columns:
        name, unit = column, ""
        for suffix, candidate in suffix_units:
            if column.endswith(suffix):
                name, unit = column[:-len(suffix)], candidate
                break
        names.append(name)
        units.append(unit)
    comment = "mu="+format(float(context["mu_meV"]), ".12g")+" meV; stage="+context["stage"]
    if "flavor_or_valley" in context:
        comment += "; flavor="+str(context["flavor_or_valley"])
    comments = [comment]*len(columns)
    lines = []
    for key, value in context.items():
        if isinstance(value, (float, np.floating)):
            text = format(float(value), ".12g")
        elif isinstance(value, (dict, list, tuple, np.ndarray)):
            text = json.dumps(np.asarray(value).tolist() if isinstance(value, np.ndarray) else value,
                              ensure_ascii=True, sort_keys=True)
        else:
            text = str(value)
        lines.append("# "+key+" = "+text+"\n")
    # These rows must not start with '#': Origin assigns them to Long Name,
    # Units and Comments instead of treating them as one File Header block.
    lines.extend("\t".join(row)+"\n" for row in (names, units, comments))
    labels = {"column_header_rows": 3, "long_names": names, "units": units,
              "column_comments": comments}
    return "".join(lines), labels


def _write_table(folder, filename, values, columns, context, *, numeric_format="%.12g", **extra):
    path = folder / filename
    header, labels = _table_header(columns, {**context, **extra})
    with atomic_output(path) as temporary:
        # Larger buffering avoids thousands of small writes on mapped drives.
        with temporary.open("w",encoding="utf-8",newline="",buffering=1024*1024) as stream:
            stream.write(header)
            formats = ["%s"]+[numeric_format]*(len(columns)-1) if values.dtype.kind == "O" else numeric_format
            np.savetxt(stream,values,delimiter="\t",fmt=formats)
    return {"filename": filename, "rows": len(values), "columns": columns, "bytes": path.stat().st_size,
            **labels, **extra}


def _band_table(folder, name, path, context, offset):
    bands = np.asarray(path["bands"],float)
    dist, ticks = np.asarray(path["dist"],float),np.asarray(path["ticks"],int)
    if bands.ndim != 2 or dist.shape != (len(bands),) or ticks.shape != (4,):
        raise ValueError("saved bands require matching distances and four symmetry ticks")
    if not np.isfinite(bands).all() or not np.isfinite(dist).all() or np.any(ticks < 0) or np.any(ticks >= len(bands)):
        raise ValueError("saved band path contains invalid values or tick indices")
    labels = np.full(len(bands),"",dtype=object)
    labels[ticks] = ["K","Gamma","M","Kprime"]
    table = np.empty((len(bands),bands.shape[1]+2),dtype=object)
    table[:,0],table[:,1],table[:,2:] = labels,dist,bands+offset
    columns = ["path_label","path_distance_Ainv"]+[f"band_{i+1:04d}_meV" for i in range(bands.shape[1])]
    return _write_table(folder,"bands_"+name+".txt",table,columns,context,
                        flavor_or_valley=name,path_tick_rows_zero_based=ticks.tolist(),
                        path_tick_x_Ainv=dist[ticks].tolist())


def _surface_tables(folder,name,contour,geom,context,offset,mu_bare,square_points,
                    *, lower_block_weights=None, block_lines_out=None):
    energies = np.asarray(contour["evals_"+("Kp" if name.startswith("Kp") else "K")+"_meV"],float)
    kpts = np.asarray(contour["kpts"],float)
    midpoint = energies.shape[1]//2
    vb,cb = midpoint-1,midpoint
    if midpoint < 1 or energies.shape[1] < 2:
        raise ValueError("spectrum lacks the neutrality valence/conduction pair")
    context = {**context,"flavor_or_valley": name,"valence_band_number": vb+1,"conduction_band_number": cb+1}
    columns = ["VB_kx_Ainv","VB_ky_Ainv","VB_E_rel_mu_meV",
               "CB_kx_Ainv","CB_ky_Ainv","CB_E_rel_mu_meV"]
    raw = np.column_stack([kpts,energies[:,vb]+offset,kpts,energies[:,cb]+offset])
    inventory = [_write_table(folder,"fermi_bands_raw_"+name+".txt",raw,columns,context,
                              sampling="original_saved_reciprocal_parallelogram")]
    crossing = np.flatnonzero((energies.min(axis=0) <= mu_bare) & (energies.max(axis=0) >= mu_bare))
    indices = np.unique(np.r_[vb,cb,crossing]).astype(int)
    try:
        surface = PeriodicBandSurface(geom,kpts,energies[:,indices])
    except UndersampledReciprocalGrid:
        unavailable = dict(contour_status="unavailable_insufficient_mesh",
                           reason="two_dimensional_interpolation_requires_2_samples_per_reciprocal_axis")
        inventory.append(_write_table(folder,"fermi_bands_square_C3_"+name+".txt",
            np.empty((0,6)),columns,context,**unavailable))
        inventory.append(_write_table(folder,"fermi_lines_C3_"+name+".txt",
            np.empty((0,2)),["kx_Ainv","ky_Ainv"],context,**unavailable))
        if lower_block_weights is not None:
            from .origin_block_export import bounded_probability, block_context
            projection = bounded_probability(lower_block_weights)
            if projection.shape != energies.shape:
                raise ValueError("raw contour block weights must match every saved band")
            block_columns = columns[:3]+["VB_W_lower_block"]+columns[3:]+["CB_W_lower_block"]
            weighted = np.column_stack([raw[:,:3],projection[:,vb],raw[:,3:],projection[:,cb]])
            inventory.append(_write_table(folder,"fermi_bands_block_weight_raw_"+name+".txt",
                weighted,block_columns,block_context(context),sampling="original_saved_grid_eigenvector_probability"))
            inventory.append(_write_table(folder,"fermi_bands_block_weight_square_C3_"+name+".txt",
                np.empty((0,8)),block_columns,block_context(context),**unavailable))
            inventory.append(_write_table(folder,"fermi_lines_block_weight_C3_"+name+".txt",
                np.empty((0,3)),["kx_Ainv","ky_Ainv","W_lower_block"],block_context(context),**unavailable))
            if block_lines_out is not None:
                block_lines_out[name] = []
        return inventory
    x,y,queries = square_mesh(geom,square_points)
    sampled = surface(queries,c3_complete=True)+offset
    vb_column,cb_column = np.searchsorted(indices,[vb,cb])
    square = np.column_stack([queries,sampled[:,vb_column],queries,sampled[:,cb_column]])
    completion = "C3_group_average_of_three_rotated_periodic_bilinear_interpolants"
    mismatch = surface.c3_mismatch(kpts)
    inventory.append(_write_table(folder,"fermi_bands_square_C3_"+name+".txt",square,columns,context,
        sampling=completion,square_grid_shape=[square_points,square_points],
        square_bounds_Ainv=[float(x.min()),float(x.max()),float(y.min()),float(y.max())],
        finite_cutoff_C3_note="plotting_completion_not_new_diagonalization_or_spectral_convergence",
        raw_C3_interpolation_mismatch_meV=mismatch))
    segments, line_columns, band_numbers = [],[],[]
    for column,band in enumerate(indices):
        field = lambda points: surface(points,c3_complete=True)[:,column]+offset
        lines = zero_level_lines(x,y,sampled[:,column].reshape(x.shape),energy_function=field)
        for component,segment in enumerate(lines,1):
            segments.append(segment)
            line_columns.extend([f"band_{band+1:04d}_line_{component:03d}_kx_Ainv",
                                 f"band_{band+1:04d}_line_{component:03d}_ky_Ainv"])
            band_numbers.append(int(band+1))
    inventory.append(_write_table(folder,"fermi_lines_C3_"+name+".txt",pack_lines(segments),
        line_columns or ["kx_Ainv","ky_Ainv"],context,numeric_format="%.17g",sampling=completion,
        contour_level_E_rel_mu_meV=0.,contour_components=len(segments),
        contour_vertex_energy_tolerance_meV=1e-7,
        component_band_numbers=band_numbers,
        contour_status="crossing" if segments else "no_one_dimensional_crossing",
        missing_value="nan_padding_between_independent_XY_column_pairs"))
    if lower_block_weights is not None:
        weighted, colored_lines = surface_block_tables(folder, name, context, geom, kpts,
            lower_block_weights, raw, square, queries, indices, vb, cb, segments,
            line_columns, band_numbers, _write_table)
        inventory.extend(weighted)
        if block_lines_out is not None:
            block_lines_out[name] = colored_lines
    return inventory


def _base_context(hamiltonian,stage,mu):
    if not np.isfinite(mu):
        raise ValueError("Origin stage requires a finite physical chemical potential")
    return {"stage": stage,"stacking_top_to_bottom": hamiltonian["stacking"],
            "theta_deg": hamiltonian["theta_deg"],"mu_meV": float(mu),
            "energy_reference": "E_minus_mu" if stage == "before_stoner" else "E_plus_Sigma_f_minus_mu_common",
            "momentum_units": "Angstrom^-1","DOS_units": "Angstrom^-2 meV^-1",
            "layer_order": "top_to_bottom"}


def _stage_common(folder,spectrum,geom,context,mus,layer_dos,sigma,dos_points, *, projections=None):
    if isinstance(dos_points,bool) or int(dos_points) != dos_points or dos_points < 3:
        raise ValueError("dos_points must be an integer >= 3")
    centers = np.linspace(-100,100,int(dos_points))
    energies = [spectrum["evals_K_meV"],spectrum["evals_Kp_meV"]]*2
    dos = gaussian_dos_window(energies,spectrum["weights"],mus,centers,sigma)
    layers = np.asarray(layer_dos,float)
    if layers.shape != (int(spectrum["n_layers"]),) or not np.isfinite(layers).all():
        raise ValueError("saved layer DOS must be finite with one value per layer")
    inventory = [_write_table(folder,"dos.txt",np.column_stack([centers,dos]),
                               ["E_rel_mu_meV","DOS_Ainv2_meVinv"],context,
                               dos_sigma_meV=sigma,gaussian_state_tail_cutoff_sigma=12)]
    inventory.append(_write_table(folder,"layer_dos.txt",np.column_stack([np.arange(1,len(layers)+1),layers]),
        ["layer_number_top_to_bottom","layer_DOS_at_mu_Ainv2_meVinv"],context,dos_sigma_meV=sigma))
    inventory.append(_write_table(folder,"first_mbz_boundary.txt",first_mbz_boundary(geom),
        ["kx_Ainv","ky_Ainv"],context,sampling="exact_closed_Wigner_Seitz_hexagon"))
    if projections is not None:
        inventory.append(block_dos_table(folder, energies, spectrum["weights"], mus,
            centers, sigma, dos, projections, layers, context, _write_table, gaussian_dos_window))
    return inventory


def _remove_generated_directory(path,root):
    path,root = Path(path),Path(root)
    if path.resolve().parent != root.resolve() or not path.name.startswith((".origin_stage_",".origin_previous_")):
        raise ValueError("refusing cleanup outside an exporter-owned staging directory")
    shutil.rmtree(str(path))


@contextmanager
def _stage_directory(detail_dir,stage):
    root = Path(detail_dir) / "origin_txt"
    root.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=".origin_stage_"+stage+"_",dir=str(root)))
    try:
        yield folder
    finally:
        if folder.exists():
            _remove_generated_directory(folder,root)


def _publish_stage(folder,context,inventory,**extra):
    manifest = {"format_version": FORMAT_VERSION,"status": "ok",**context,
                "files": inventory,"total_txt_bytes": sum(item["bytes"] for item in inventory),**extra}
    atomic_json(folder / "manifest.json",manifest)
    parent = folder.parent
    destination = parent / context["stage"]
    previous = parent / (".origin_previous_"+context["stage"]+"_"+uuid.uuid4().hex)
    if destination.exists():
        old_manifest = destination / "manifest.json"
        if not old_manifest.is_file() or json.loads(old_manifest.read_text(encoding="utf-8")).get("format_version") != FORMAT_VERSION:
            raise ValueError("refusing to replace an unrecognized Origin output directory")
        os.replace(str(destination),str(previous))
    try:
        os.replace(str(folder),str(destination))
    except Exception:
        if previous.exists():
            os.replace(str(previous),str(destination))
        raise
    if previous.exists():
        _remove_generated_directory(previous,parent)
    # A newly published Hartree generation invalidates previous after-stage
    # output even when optimization subsequently fails before saving its cache.
    if context["stage"] == "before_stoner":
        _invalidate_after(parent.parent,"pending_stoner","before_stage_regenerated")
    stages = {}
    for stage in ("before_stoner","after_stoner"):
        path = parent / stage / "manifest.json"
        if path.is_file():
            stages[stage] = json.loads(path.read_text(encoding="utf-8"))
    atomic_json(parent / "manifest.json",{"format_version": FORMAT_VERSION,"stages": stages})
    instructions = (
        "Origin ASCII import: tab delimiter; # lines are main-header metadata.\n"
        "The next 3 tabular rows are Long Name, Units and Comments (mu), followed by data.\n"
        "Use System Default import settings; if a saved theme overrides header detection:\n"
        "Main Header = leading character #; Subheader Lines = 3; count from top;\n"
        "Long Names = 1; Units = 2; Comments From = 3, To = 3. Do not combine delimiters.\n"
        "bands: col 1 = K/Gamma/M/Kprime labels at tick rows; col 2 = X; all later columns = Y.\n"
        "fermi_bands: each (kx,ky,E) group is an independent XYZ dataset; energy is relative to stage mu.\n"
        "fermi_lines: each two columns are one independent XY line; nan pads shorter lines.\n"
        "Check contour_status in line-file metadata: no_one_dimensional_crossing means no crossing; unavailable_insufficient_mesh means interpolation was not possible.\n"
        "first_mbz_boundary: plot as an XY line on top of the contour.\n"
        "dos: X = E relative to mu (meV), Y = total DOS (Angstrom^-2 meV^-1).\n"
        "layer_dos: X = top-to-bottom layer 1..4, Y = total layer DOS at mu; plot a column chart.\n"
        "Before: two spin-degenerate valleys. After: four separate flavors sharing one physical mu.\n"
        "Every data TXT states absolute mu; after-stage headers also state Sigma and mu-Sigma for each flavor.\n"
        "raw files preserve saved eigenvalues. square_C3 and lines_C3 are interpolated plotting completion\n"
        "using the average of three rotated periodic interpolants; they do not establish finite-cutoff convergence.\n"
        "Line vertices are refined on the same continuous field to an energy residual <= 1e-7 meV.\n"
        "The original finite square plane-wave cutoff is not closed under C3. No original data are changed.\n"
        "Block bands: col 1 labels; col 2 X; then each (E, W_lower_block) pair is one colored band.\n"
        "Block surfaces: each (kx, ky, E, W_lower_block) is an independent XYZ/color group.\n"
        "Block lines: each (kx, ky, W_lower_block) triplet is one independent colored XY line.\n"
        "W=0 is upper bilayer layers 1-2; W=1 is lower bilayer layers 3-4; sum all plane waves.\n"
        "C3 line/square colors are interpolated plotting probabilities, using the same C3 projection.\n"
        "block_dos: E, DOS_upper, DOS_lower, DOS_all, DOS_lower/DOS_all (nan if DOS_all=0).\n"
        "block_resolved_bands_and_contours.pdf/.png show every valley/flavor with this same 0..1 scale.\n"
    )
    for stage,data in stages.items():
        instructions += "# "+stage+" mu_meV = "+str(data.get("mu_meV"))+"; status = "+data["status"]+"\n"
    with atomic_output(parent / "README.txt") as temporary:
        temporary.write_text(instructions,encoding="utf-8")
    return manifest


def _invalidate_after(detail_dir,status,reason):
    with _stage_directory(detail_dir,"after_stoner") as folder:
        return _publish_stage(folder,{"stage": "after_stoner","mu_meV": None},[],
                              status=status,reason=reason)


def invalidate_detail_outputs(detail_dir):
    """A new point calculation retires both previously published physical stages."""
    detail_dir = Path(detail_dir)
    if not (detail_dir / "origin_txt").exists():
        return
    for stage in ("before_stoner", "after_stoner"):
        with _stage_directory(detail_dir, stage) as folder:
            _publish_stage(folder, {"stage": stage, "mu_meV": None}, [],
                           status="pending_calculation", reason="new_point_calculation_started")


def _cache_source_mismatch(cache,scf):
    """Do not combine DOS from an old cache with refreshed path/contour data."""
    for key in ("kpts","weights","evals_K_meV","evals_Kp_meV"):
        if key not in cache or key not in scf or not np.array_equal(cache[key],scf[key]):
            return key
    if float(cache["mu_full_meV"]) != float(scf["full_mu_meV"]):
        return "full_mu_meV"
    if json.loads(str(cache["hamiltonian_json"].item())) != json.loads(str(scf["hamiltonian_json"].item())):
        return "hamiltonian_identity"
    if "U_scf_meV" in cache and "U_meV" in scf and not np.array_equal(cache["U_scf_meV"],scf["U_meV"]):
        return "U_scf_meV"
    return None


def export_scf_detail(scf_dir, *, square_points=145, dos_points=1001):
    scf_dir = Path(scf_dir)
    spectrum = _load(scf_dir / "spectrum.npz")
    geom,hamiltonian = _geometry(spectrum)
    mu = float(spectrum["full_mu_meV"])
    context = _base_context(hamiltonian,"before_stoner",mu)
    sigma = float(_load(scf_dir / "dos.npz")["dos_sigma_meV"])
    contour = _load(scf_dir / "full_band_contours_before_stoner.npz")
    paths = {name: _load(scf_dir / ("full_bands_"+name+".npz")) for name in ("K","Kp")}
    blocks = get_saved_block_data(scf_dir, spectrum, contour, paths)
    band_data, line_data = {}, {}
    with _stage_directory(scf_dir.parent,"before_stoner") as folder:
        inventory = _stage_common(folder,spectrum,geom,context,[mu]*4,spectrum["layer_dos"],sigma,dos_points,
            projections=[blocks["grid_K"], blocks["grid_Kp"]]*2)
        for name in ("K","Kp"):
            inventory.append(_band_table(folder,name,paths[name],context,-mu))
            entry, band_data[name] = band_block_table(folder,name,paths[name],context,-mu,
                                                       blocks["path_"+name],_write_table)
            inventory.append(entry)
            inventory.extend(_surface_tables(folder,name,contour,geom,context,-mu,mu,square_points,
                lower_block_weights=blocks["contour_"+name],block_lines_out=line_data))
        if any(item.get("contour_status") == "unavailable_insufficient_mesh" for item in inventory):
            context = {**context, "contour_status": "unavailable_insufficient_mesh"}
        figures = render_block_stage(folder,context,band_data,line_data,first_mbz_boundary(geom),
                                     point_info=_point_info(scf_dir))
        return _publish_stage(folder,context,inventory,source_detail=str(scf_dir.parent),
                              block_weight_source=blocks["source"],figures=figures)


def export_stoner_detail(detail_dir, *, square_points=145, dos_points=1001):
    detail_dir = Path(detail_dir)
    cache = _load(detail_dir / "full_band_stoner_cache.npz")
    if not bool(cache["stoner_dos_valid"]):
        return _invalidate_after(detail_dir,"invalid_stoner","saved_common_fermi_level_not_valid")
    scf = _load(detail_dir / "scf" / "spectrum.npz")
    mismatch = _cache_source_mismatch(cache,scf)
    if mismatch:
        return _invalidate_after(detail_dir,"stale_stoner_cache","bare_spectrum_source_mismatch: "+mismatch)
    mu = float(cache["stoner_mu_common_meV"])
    shifts,mu_f = np.asarray(cache["stoner_self_energy_f_meV"],float),np.asarray(cache["mu_f_meV"],float)
    if (shifts.shape != (4,) or mu_f.shape != (4,) or not np.isfinite(shifts).all()
            or not np.isfinite(mu_f).all() or not np.allclose(mu_f+shifts,mu,rtol=0,atol=1e-8)):
        raise ValueError("saved Stoner shifts and flavor mu do not share the physical common mu")
    geom,hamiltonian = _geometry(cache)
    context = _base_context(hamiltonian,"after_stoner",mu)
    scf_dir = detail_dir / "scf"
    sigma = float(_load(scf_dir / "dos.npz")["dos_sigma_meV"])
    contour = _load(scf_dir / "full_band_contours_before_stoner.npz")
    paths = {name: _load(scf_dir / ("full_bands_"+name+".npz")) for name in ("K","Kp")}
    blocks = get_saved_block_data(scf_dir,scf,contour,paths)
    band_data, line_data = {}, {}
    with _stage_directory(detail_dir,"after_stoner") as folder:
        inventory = _stage_common(folder,cache,geom,context,mu_f,cache["stoner_layer_dos"],sigma,dos_points,
            projections=[blocks["grid_K"], blocks["grid_Kp"]]*2)
        for f,name in enumerate(FLAVORS):
            flavor_context = {**context,"self_energy_Sigma_meV": float(shifts[f]),"bare_spectrum_fermi_level_mu_minus_Sigma_meV": float(mu_f[f])}
            valley = "Kp" if name.startswith("Kp") else "K"
            inventory.append(_band_table(folder,name,paths[valley],flavor_context,shifts[f]-mu))
            entry,band_data[name] = band_block_table(folder,name,paths[valley],flavor_context,
                shifts[f]-mu,blocks["path_"+valley],_write_table)
            inventory.append(entry)
            inventory.extend(_surface_tables(folder,name,contour,geom,flavor_context,shifts[f]-mu,mu_f[f],square_points,
                lower_block_weights=blocks["contour_"+valley],block_lines_out=line_data))
        if any(item.get("contour_status") == "unavailable_insufficient_mesh" for item in inventory):
            context = {**context, "contour_status": "unavailable_insufficient_mesh"}
        figures = render_block_stage(folder,context,band_data,line_data,first_mbz_boundary(geom),
                                     point_info=_point_info(scf_dir))
        return _publish_stage(folder,context,inventory,source_detail=str(detail_dir),
                              self_energy_f_meV=shifts.tolist(),mu_f_meV=mu_f.tolist(),
                              block_weight_source=blocks["source"],figures=figures)


def _point_info(scf_dir):
    path = Path(scf_dir) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")).get("point", {}) if path.is_file() else {}


def export_point(detail_dir, *, square_points=145, dos_points=1001):
    detail_dir = Path(detail_dir)
    before = export_scf_detail(detail_dir / "scf",square_points=square_points,dos_points=dos_points)
    after = export_stoner_detail(detail_dir,square_points=square_points,dos_points=dos_points) if (detail_dir / "full_band_stoner_cache.npz").is_file() else _invalidate_after(detail_dir,"missing_stoner","no_saved_stoner_cache")
    return {"point": str(detail_dir),"output": str(detail_dir / "origin_txt"),
            "status": "ok" if after["status"] == "ok" else "scf_only",
            "before": before,"after": after}


def find_detail_points(run):
    run = Path(run)
    if not run.is_dir():
        raise ValueError("saved run directory does not exist: "+str(run))
    found = []
    for spectrum in run.rglob("spectrum.npz"):
        relative = spectrum.relative_to(run)
        if (spectrum.parent.name == "scf" and not any(part.startswith(("_",".")) for part in relative.parts)
                and (spectrum.parent / "full_bands_K.npz").is_file()
                and (spectrum.parent / "full_bands_Kp.npz").is_file()):
            found.append(spectrum.parent.parent)
    return sorted(set(found))


def _export_job(payload):
    path,square_points,dos_points = payload
    return export_point(path,square_points=square_points,dos_points=dos_points)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Export full Hartree/Stoner Origin TXT and block weights; old data reconstructs eigenvectors at saved U without repeating SCF/Stoner.")
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--square-points",type=int,default=145)
    parser.add_argument("--dos-points",type=int,default=1001)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 61:
        parser.error("workers must be between 1 and 61")
    points = find_detail_points(args.run)
    if not points:
        parser.error("no saved detailed SCF point with path bands was found")
    jobs = [(str(path),args.square_points,args.dos_points) for path in points]
    print("Exporting "+str(len(points))+" saved points with "+str(min(args.workers,len(jobs)))+" workers; no Hartree/Stoner iteration. Old weights may require fixed-U diagonalization.",flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        pending = {pool.submit(_export_job,job): job[0] for job in jobs}
        for future in as_completed(pending):
            try:
                result = future.result()
                record = {"point": result["point"],"output": result["output"],"status": result["status"],
                          "before_mu_meV": result["before"]["mu_meV"],
                          "after_mu_meV": result["after"].get("mu_meV"),
                          "after_status": result["after"]["status"],
                          "after_reason": result["after"].get("reason"),
                          "txt_files": len(result["before"]["files"])+len(result["after"].get("files",[])),
                          "txt_bytes": result["before"]["total_txt_bytes"]+result["after"].get("total_txt_bytes",0)}
            except Exception as error:
                record = {"point": pending[future],"status": "failed","error": repr(error)}
            records.append(record)
            print("["+str(len(records))+"/"+str(len(jobs))+"] "+record["status"]+": "+record["point"],flush=True)
    summary = {"format_version": FORMAT_VERSION,"points": sorted(records,key=lambda r:r["point"]),
               "no_hartree_or_stoner_recalculation": True,"square_points": args.square_points,"dos_points": args.dos_points}
    atomic_json(args.run / "origin_export_summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
