"""Origin files preserve the saved physical stages and reciprocal geometry."""
import json
from pathlib import Path

import numpy as np
import pytest

from tdbg_scf.lattice import MoireGeometry, make_uniform_mbz_grid
from tdbg_scf.origin_geometry import PeriodicBandSurface, first_mbz_boundary, square_mesh, zero_level_lines, pack_lines
from tdbg_scf.origin_export import export_point, find_detail_points, _write_table


@pytest.mark.parametrize("columns,names,units", [
    (["path_label", "path_distance_Ainv", "band_0001_meV"],
     ["path_label", "path_distance", "band_0001"], ["", "Angstrom^-1", "meV"]),
    (["E_rel_mu_meV", "DOS_Ainv2_meVinv"],
     ["E_rel_mu", "DOS"], ["meV", "Angstrom^-2 meV^-1"]),
    (["VB_kx_Ainv", "VB_ky_Ainv", "VB_E_rel_mu_meV"],
     ["VB_kx", "VB_ky", "VB_E_rel_mu"], ["Angstrom^-1", "Angstrom^-1", "meV"]),
    (["layer_number", "layer_DOS_Ainv2_meVinv"],
     ["layer_number", "layer_DOS"], ["", "Angstrom^-2 meV^-1"]),
])
def test_origin_column_labels_are_three_tabular_subheader_rows(tmp_path, columns, names, units):
    values = np.arange(2*len(columns), dtype=float).reshape(2, len(columns))
    info = _write_table(tmp_path, "headers.txt", values, columns,
                        {"stage": "before_stoner", "mu_meV": 4.49394790783},
                        flavor_or_valley="K")
    text = (tmp_path / "headers.txt").read_text(encoding="utf-8")
    non_metadata = [line for line in text.splitlines() if not line.startswith("#")]
    assert non_metadata[0].split("\t") == names
    assert non_metadata[1].split("\t") == units
    comments = non_metadata[2].split("\t")
    assert len(comments) == len(columns)
    assert all("mu=4.49394790783 meV" in item and "before_stoner" in item and "K" in item
               for item in comments)
    assert "# columns = " not in text
    assert info["long_names"] == names and info["units"] == units
    np.testing.assert_array_equal(np.loadtxt(non_metadata[3:], delimiter="\t", ndmin=2), values)


def numeric_table(path, cols=None):
    return np.loadtxt(data_lines(path), delimiter="\t", usecols=cols, ndmin=2)


def data_lines(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")][3:]


def analytic_bands(geom, kpts):
    uv = np.linalg.solve(np.column_stack([geom.b1m, geom.b2m]), kpts.T).T
    u, v = uv.T
    scalar = np.cos(2*np.pi*u) + np.cos(2*np.pi*v) + np.cos(2*np.pi*(u-v))
    return np.column_stack([-20+.3*scalar, -1+.3*scalar, 1+.3*scalar, 20+.3*scalar])


def gaussian_total(energies, weights, mu_values, centers, sigma):
    result = np.zeros_like(centers)
    for e, mu in zip(energies, mu_values):
        z = (e[..., None] - mu - centers) / sigma
        kernel = np.exp(-.5*z*z) / (np.sqrt(2*np.pi)*sigma)
        result += np.einsum("k,kbc->c", weights, kernel)
    return result


def gaussian_projected(energies, weights, projections, mu_values, centers, sigma):
    result = np.zeros_like(centers)
    for e, projection, mu in zip(energies, projections, mu_values):
        z = (e[..., None] - mu - centers) / sigma
        kernel = np.exp(-.5*z*z) / (np.sqrt(2*np.pi)*sigma)
        result += np.einsum("k,kb,kbc->c", weights, projection, kernel)
    return result


@pytest.fixture
def saved_point(tmp_path):
    root = tmp_path / "details" / "point"
    scf = root / "scf"
    scf.mkdir(parents=True)
    geom = MoireGeometry(theta_deg=1.2)
    kpts, weights = make_uniform_mbz_grid(geom, 8, 8)
    energies = analytic_bands(geom, kpts)
    mu_before, mu_after, sigma = .8, .5, .4
    shifts = np.array([-.2, 0., .3, -.4])
    mu_f = mu_after-shifts
    dos_before = gaussian_total([energies]*4, weights, [mu_before]*4, np.array([0.]), sigma)[0]
    dos_after = gaussian_total([energies]*4, weights, mu_f, np.array([0.]), sigma)[0]
    tri = .5 + .1*np.tanh(energies)
    tri_valleys = [tri, 1-tri]*2
    tri_before = gaussian_projected([energies]*4, weights, tri_valleys,
                                    [mu_before]*4, np.array([0.]), sigma)[0]
    tri_after = gaussian_projected([energies]*4, weights, tri_valleys,
                                   mu_f, np.array([0.]), sigma)[0]
    def layer_dos(total, lower):
        return np.r_[np.array([.4, .6])*(total-lower), np.array([.4, .6])*lower]
    ham = dict(theta_deg=1.2, a_cc_A=1.42, stacking="BA-AB", n_layers=4)
    meta = dict(hamiltonian_json=json.dumps(ham), stacking="BA-AB", n_layers=4)
    np.savez(scf / "spectrum.npz", **meta, kpts=kpts, weights=weights,
             evals_K_meV=energies, evals_Kp_meV=energies,
             full_mu_meV=mu_before, layer_dos=layer_dos(dos_before, tri_before),
             lower_block_weight_K=tri, lower_block_weight_Kp=1-tri)
    np.savez(scf / "dos.npz", energy_meV=np.linspace(-80,80,12), dos=np.zeros(12),
             full_mu_meV=mu_before, dos_sigma_meV=sigma)
    np.savez(scf / "full_band_contours_before_stoner.npz", **meta, kpts=kpts,
             evals_K_meV=energies, evals_Kp_meV=energies, band_indices=np.arange(4),
             lower_block_weight_K=tri, lower_block_weight_Kp=1-tri)
    path, dist, ticks, labels = geom.high_symmetry_path(2)
    path_bands = analytic_bands(geom, path)
    for valley in ("K", "Kp"):
        np.savez(scf / ("full_bands_"+valley+".npz"), **meta,
                 dist=dist, bands=path_bands, ticks=ticks, labels=labels, mu_meV=mu_before,
                 kpts=path, lower_block_weight=(.5+.1*np.tanh(path_bands)) if valley == "K"
                 else (.5-.1*np.tanh(path_bands)))
    np.savez(root / "full_band_stoner_cache.npz", **meta,
             kpts=kpts, weights=weights, evals_K_meV=energies, evals_Kp_meV=energies,
             mu_full_meV=mu_before, stoner_mu_common_meV=mu_after,
             mu_f_meV=mu_f, stoner_self_energy_f_meV=shifts, stoner_dos_valid=True,
             scf_layer_dos=layer_dos(dos_before, tri_before),
             stoner_layer_dos=layer_dos(dos_after, tri_after))
    (root / "config_used.json").write_text(json.dumps({"args": {"dos_sigma_meV": sigma},
        "hamiltonian": ham, "task": {"n_cm2": 0., "D_Vnm": 0.}}), encoding="utf-8")
    return root, energies, weights, path_bands, dist, mu_before, mu_after, shifts, sigma


def test_first_mbz_is_closed_hexagon_with_reciprocal_cell_area():
    geom = MoireGeometry(theta_deg=1.2)
    polygon = first_mbz_boundary(geom)
    assert polygon.shape == (7, 2)
    np.testing.assert_array_equal(polygon[0], polygon[-1])
    x, y = polygon.T
    area = abs(np.dot(x[:-1], y[1:])-np.dot(y[:-1], x[1:]))/2
    assert area == pytest.approx(geom.mBZ_area_A2_inv, rel=1e-13)
    assert np.ptp(np.linalg.norm(polygon[:-1], axis=1)) < 1e-14


def test_periodic_interpolation_preserves_source_and_lattice_translations():
    geom = MoireGeometry(theta_deg=1.2)
    kpts, _ = make_uniform_mbz_grid(geom, 8, 10)
    bands = analytic_bands(geom, kpts)
    surface = PeriodicBandSurface(geom, kpts, bands)
    np.testing.assert_allclose(surface(kpts), bands, atol=1e-11)
    queries = np.array([[.013, -.007], [-.025, .021]])
    np.testing.assert_allclose(surface(queries+3*geom.b1m-2*geom.b2m), surface(queries), atol=1e-11)


def test_c3_completion_is_continuous_group_projection_and_preserves_raw_samples():
    geom = MoireGeometry(theta_deg=1.2)
    kpts, _ = make_uniform_mbz_grid(geom, 8, 8)
    # A square cutoff can break raw C3. Project only the plotting interpolant.
    values = analytic_bands(geom, kpts) + 100*kpts[:, :1]
    surface = PeriodicBandSurface(geom, kpts, values)
    theta = 2*np.pi/3
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = np.array([[.011, .003], [.006, -.002], [.004, .0008]])
    projected = (surface(points)+surface(points @ rotation.T)
                 +surface(points @ (rotation @ rotation).T))/3
    np.testing.assert_allclose(surface(points, c3_complete=True), projected, atol=1e-11)
    np.testing.assert_allclose(surface(kpts), values, atol=1e-11)
    np.testing.assert_allclose(surface(points @ rotation.T, c3_complete=True),
                               surface(points, c3_complete=True), atol=1e-11)
    np.testing.assert_allclose(surface(points+geom.b1m, c3_complete=True),
                               surface(points, c3_complete=True), atol=1e-11)


def test_square_covers_the_entire_first_mbz():
    geom = MoireGeometry(theta_deg=1.2)
    x, y, points = square_mesh(geom, 31)
    boundary = first_mbz_boundary(geom)
    assert x.shape == y.shape == (31, 31) and points.shape == (961, 2)
    assert x.min() < boundary[:,0].min() and x.max() > boundary[:,0].max()
    assert y.min() < boundary[:,1].min() and y.max() > boundary[:,1].max()


def test_export_preserves_all_bands_labels_and_shared_stoner_mu(saved_point):
    root, _, _, path_bands, dist, mu0, mu1, shifts, _ = saved_point
    result = export_point(root, square_points=31, dos_points=201)
    assert result["status"] == "ok"
    before = root / "origin_txt" / "before_stoner"
    after = root / "origin_txt" / "after_stoner"
    raw = np.genfromtxt(data_lines(before / "bands_K.txt"), delimiter="\t", dtype=str)
    assert raw.shape == (7, 6)
    assert raw[[0,2,4,6],0].tolist() == ["K", "Gamma", "M", "Kprime"]
    np.testing.assert_allclose(raw[:,1:].astype(float), np.column_stack([dist,path_bands-mu0]), atol=1e-10)
    for f, name in enumerate(("K_up", "Kp_up", "K_down", "Kp_down")):
        data = numeric_table(after / ("bands_"+name+".txt"), [1,2,3,4,5])
        np.testing.assert_allclose(data[:,1:], path_bands+shifts[f]-mu1, atol=1e-10)
        assert ("mu_meV = "+format(mu1,".12g")) in (after / ("bands_"+name+".txt")).read_text()
    for stage in (before, after):
        for path in stage.glob("*.txt"):
            assert "# mu_meV = " in path.read_text(), path.name


def test_dos_window_spin_counting_and_layer_histogram(saved_point):
    root, energies, weights, _, _, mu0, mu1, shifts, sigma = saved_point
    export_point(root, square_points=31, dos_points=201)
    for stage, mus in (("before_stoner",[mu0]*4), ("after_stoner",mu1-shifts)):
        folder = root / "origin_txt" / stage
        dos = numeric_table(folder / "dos.txt")
        assert dos.shape == (201,2)
        np.testing.assert_array_equal(dos[[0,-1],0], [-100,100])
        expected = gaussian_total([energies]*4, weights, mus, dos[:,0], sigma)
        np.testing.assert_allclose(dos[:,1], expected, rtol=2e-10, atol=1e-15)
        layer = numeric_table(folder / "layer_dos.txt")
        np.testing.assert_array_equal(layer[:,0], np.arange(1,5))
        assert layer[:,1].sum() == pytest.approx(dos[100,1], rel=2e-10)


def test_raw_contour_columns_are_vb_cb_in_correct_energy_reference(saved_point):
    root, energies, _, _, _, mu0, mu1, shifts, _ = saved_point
    export_point(root, square_points=31, dos_points=201)
    before = numeric_table(root / "origin_txt" / "before_stoner" / "fermi_bands_raw_K.txt")
    assert before.shape == (len(energies),6)
    np.testing.assert_allclose(before[:,2], energies[:,1]-mu0, atol=1e-10)
    np.testing.assert_allclose(before[:,5], energies[:,2]-mu0, atol=1e-10)
    after = numeric_table(root / "origin_txt" / "after_stoner" / "fermi_bands_raw_K_down.txt")
    np.testing.assert_allclose(after[:,2], energies[:,1]+shifts[2]-mu1, atol=1e-10)
    np.testing.assert_allclose(after[:,5], energies[:,2]+shifts[2]-mu1, atol=1e-10)


def test_block_resolved_bands_keep_full_band_order_and_scalar_stoner_weights(saved_point):
    root, _, _, path_bands, dist, mu0, mu1, shifts, _ = saved_point
    result = export_point(root, square_points=31, dos_points=201)
    before = numeric_table(root / "origin_txt" / "before_stoner" / "bands_block_weight_K.txt",
                           list(range(1, 10)))
    assert before.shape == (7, 9)
    np.testing.assert_allclose(before[:,0], dist)
    np.testing.assert_allclose(before[:,1::2], path_bands-mu0, atol=1e-10)
    np.testing.assert_allclose(before[:,2::2], .5+.1*np.tanh(path_bands), atol=1e-10)
    for f, name in enumerate(("K_up", "Kp_up", "K_down", "Kp_down")):
        after = numeric_table(root / "origin_txt" / "after_stoner" / ("bands_block_weight_"+name+".txt"),
                              list(range(1,10)))
        np.testing.assert_allclose(after[:,1::2], path_bands+shifts[f]-mu1, atol=1e-10)
        sign = -1 if name.startswith("Kp") else 1
        np.testing.assert_allclose(after[:,2::2], .5+sign*.1*np.tanh(path_bands), atol=1e-10)
    info = next(item for item in result["before"]["files"] if item["filename"] == "bands_block_weight_K.txt")
    assert info["units"] == ["", "Angstrom^-1"] + ["meV", ""]*4


def test_block_dos_is_a_spectral_projection_and_matches_saved_layer_dos(saved_point):
    root, energies, weights, _, _, mu0, mu1, shifts, sigma = saved_point
    export_point(root, square_points=31, dos_points=201)
    projections = [.5+.1*np.tanh(energies), .5-.1*np.tanh(energies)]*2
    for stage, mus in (("before_stoner", [mu0]*4), ("after_stoner", mu1-shifts)):
        folder = root / "origin_txt" / stage
        data = numeric_table(folder / "block_dos.txt")
        assert data.shape == (201,5)
        expected = gaussian_projected([energies]*4, weights, projections, mus, data[:,0], sigma)
        np.testing.assert_allclose(data[:,2], expected, rtol=2e-10, atol=1e-15)
        total = numeric_table(folder / "dos.txt")[:,1]
        np.testing.assert_allclose(data[:,1]+data[:,2], total, rtol=2e-10, atol=1e-15)
        np.testing.assert_allclose(data[:,3], total, rtol=2e-10)
        positive = total > 1e-28
        np.testing.assert_allclose(data[positive,4], expected[positive]/total[positive], rtol=2e-10)
        assert np.isnan(data[total == 0,4]).all()
        layers = numeric_table(folder / "layer_dos.txt")
        assert data[100,2] == pytest.approx(layers[2:,1].sum(), rel=2e-10)


def test_block_surface_and_line_triplets_preserve_scalar_contour_coordinates(saved_point):
    root, energies, _, *_ = saved_point
    export_point(root, square_points=31, dos_points=201)
    folder = root / "origin_txt" / "before_stoner"
    scalar = numeric_table(folder / "fermi_bands_raw_K.txt")
    weighted = numeric_table(folder / "fermi_bands_block_weight_raw_K.txt")
    assert weighted.shape == (len(energies),8)
    np.testing.assert_array_equal(weighted[:,:3], scalar[:,:3])
    np.testing.assert_array_equal(weighted[:,4:7], scalar[:,3:6])
    np.testing.assert_allclose(weighted[:,3], .5+.1*np.tanh(energies[:,1]), atol=1e-10)
    np.testing.assert_allclose(weighted[:,7], .5+.1*np.tanh(energies[:,2]), atol=1e-10)
    lines = numeric_table(folder / "fermi_lines_C3_K.txt")
    colored = numeric_table(folder / "fermi_lines_block_weight_C3_K.txt")
    assert colored.shape[0] == lines.shape[0] and colored.shape[1]*2 == lines.shape[1]*3
    for i in range(lines.shape[1]//2):
        np.testing.assert_array_equal(colored[:,3*i:3*i+2], lines[:,2*i:2*i+2])
        color = colored[:,3*i+2]
        assert np.all((color[np.isfinite(color)] >= 0) & (color[np.isfinite(color)] <= 1))
    no_crossing = root / "origin_txt" / "after_stoner" / "fermi_lines_block_weight_C3_K_down.txt"
    assert data_lines(no_crossing) == []


def test_discovery_excludes_backups_and_accepts_a_detail_directory(saved_point):
    root, *_ = saved_point
    backup = root.parents[1] / "_stoner_backups" / "old" / "scf"
    backup.mkdir(parents=True)
    (backup / "spectrum.npz").write_bytes(b"old")
    assert find_detail_points(root) == [root]
    assert find_detail_points(root.parents[1]) == [root]


def test_invalid_stoner_still_exports_the_successful_scf(saved_point):
    root, *_ = saved_point
    export_point(root, square_points=31, dos_points=201)
    path = root / "full_band_stoner_cache.npz"
    with np.load(path, allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    data["stoner_dos_valid"] = False
    data["stoner_mu_common_meV"] = np.nan
    data["mu_f_meV"] = np.full(4, np.nan)
    np.savez(path, **data)
    result = export_point(root, square_points=31, dos_points=201)
    assert result["status"] == "scf_only"
    assert (root / "origin_txt" / "before_stoner" / "dos.txt").is_file()
    assert not (root / "origin_txt" / "after_stoner" / "dos.txt").exists()
    index = json.loads((root / "origin_txt" / "manifest.json").read_text())
    assert index["stages"]["after_stoner"]["status"] == "invalid_stoner"


def test_disconnected_contours_keep_independent_xy_pairs_and_nan_padding():
    axis = np.linspace(-1,1,101)
    x,y = np.meshgrid(axis,axis)
    z = ((x-.45)**2+y**2-.22**2)*((x+.45)**2+y**2-.16**2)
    lines = zero_level_lines(x,y,z)
    assert len(lines) == 2
    assert all(np.linalg.norm(line[0]-line[-1]) < 1e-14 for line in lines)
    packed = pack_lines(lines)
    assert packed.shape[1] == 4
    for i,line in enumerate(lines):
        np.testing.assert_array_equal(packed[:len(line),2*i:2*i+2],line)
        assert np.isnan(packed[len(line):,2*i:2*i+2]).all()
    assert zero_level_lines(x,y,np.ones_like(x)) == []


def test_gapped_flavor_has_an_explicitly_empty_line_file(saved_point):
    root, *_ = saved_point
    export_point(root,square_points=31,dos_points=201)
    path = root / "origin_txt" / "after_stoner" / "fermi_lines_C3_K_down.txt"
    assert "contour_components = 0" in path.read_text()
    assert data_lines(path) == []
    label_rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    assert len(label_rows) == 3
    assert label_rows[0].split("\t") == ["kx", "ky"]


def test_malformed_nonperiodic_mesh_is_rejected():
    geom = MoireGeometry(theta_deg=1.2)
    kpts,_ = make_uniform_mbz_grid(geom,4,4)
    with pytest.raises(ValueError,match="complete rectangular"):
        PeriodicBandSurface(geom,kpts[:-1],np.ones((15,2)))


def test_failed_reexport_keeps_the_complete_previous_stage(saved_point):
    root, *_ = saved_point
    export_point(root,square_points=31,dos_points=201)
    folder = root / "origin_txt" / "before_stoner"
    original = {path.name:path.read_bytes() for path in folder.iterdir()}
    spectrum_path = root / "scf" / "spectrum.npz"
    with np.load(spectrum_path,allow_pickle=False) as z:
        changed = {key:z[key] for key in z.files}
    changed["full_mu_meV"] = 5.
    np.savez(spectrum_path,**changed)
    (root / "scf" / "full_bands_K.npz").write_bytes(b"corrupted artifact")
    with pytest.raises(ValueError):
        export_point(root,square_points=31,dos_points=201)
    assert {path.name:path.read_bytes() for path in folder.iterdir()} == original


def test_new_scf_spectrum_cannot_republish_an_older_valid_stoner_cache(saved_point):
    root, *_ = saved_point
    for filename in ("spectrum.npz","full_band_contours_before_stoner.npz","full_bands_K.npz","full_bands_Kp.npz"):
        path = root / "scf" / filename
        with np.load(path,allow_pickle=False) as z:
            data = {key:z[key] for key in z.files}
        for key in ("evals_K_meV","evals_Kp_meV","bands","full_mu_meV","mu_meV"):
            if key in data:
                data[key] = data[key]+10
        np.savez(path,**data)
    result = export_point(root,square_points=31,dos_points=201)
    assert result["status"] == "scf_only"
    assert result["after"]["status"] == "stale_stoner_cache"
    assert not (root / "origin_txt" / "after_stoner" / "dos.txt").exists()


def test_c3_contours_do_not_interpolate_across_a_discontinuous_sector_seam():
    geom = MoireGeometry(theta_deg=1.2)
    kpts,_ = make_uniform_mbz_grid(geom,64,64)
    uv = np.linalg.solve(np.column_stack([geom.b1m,geom.b2m]),kpts.T).T
    surface = PeriodicBandSurface(geom,kpts,np.sin(2*np.pi*(uv[:,1]-uv[:,0])))
    x,y,queries = square_mesh(geom,145)
    completed = surface(queries,c3_complete=True).reshape(x.shape)
    lines = zero_level_lines(x,y,completed)
    assert lines
    error = max(np.max(np.abs(surface(line,c3_complete=True))) for line in lines)
    assert error < .01, "Fermi lines include an artificial symmetry-sector seam"


@pytest.mark.parametrize("sign",[-1,1])
def test_level_touching_a_line_is_retained_but_isolated_and_flat_zeros_are_empty(sign):
    axis = np.linspace(-1,1,101)
    x,y = np.meshgrid(axis,axis)
    lines = zero_level_lines(x,y,sign*x*x)
    assert len(lines) == 1
    assert np.max(np.abs(lines[0][:,0])) < 1e-14
    assert np.ptp(lines[0][:,1]) == pytest.approx(2.)
    assert zero_level_lines(x,y,sign*(x*x+y*y)) == []
    assert zero_level_lines(x,y,np.zeros_like(x)) == []


def test_fermi_line_vertices_are_refined_to_mu_on_the_continuous_interpolant():
    geom = MoireGeometry(theta_deg=1.2)
    kpts,_ = make_uniform_mbz_grid(geom,16,16)
    uv = np.linalg.solve(np.column_stack([geom.b1m,geom.b2m]),kpts.T).T
    surface = PeriodicBandSurface(geom,kpts,np.sin(2*np.pi*(uv[:,1]-uv[:,0])))
    x,y,queries = square_mesh(geom,31)
    field = lambda points: surface(points,c3_complete=True)-.05
    lines = zero_level_lines(x,y,field(queries).reshape(x.shape),energy_function=field)
    assert lines
    assert max(np.max(np.abs(field(line))) for line in lines) < 1e-7


def test_fermi_line_refinement_uses_the_correct_edge_near_a_grid_corner():
    geom = MoireGeometry(theta_deg=1.2)
    kpts,_ = make_uniform_mbz_grid(geom,16,16)
    uv = np.linalg.solve(np.column_stack([geom.b1m,geom.b2m]),kpts.T).T
    u,v = uv.T
    values = 1000*(np.cos(2*np.pi*u)+np.cos(2*np.pi*v)+np.cos(2*np.pi*(u-v)))
    surface = PeriodicBandSurface(geom,kpts,values)
    x,y,queries = square_mesh(geom,31)
    completed = surface(queries,c3_complete=True).reshape(x.shape)
    mu = completed[0,1]+(completed[0,2]-completed[0,1])*5e-9
    field = lambda points: surface(points,c3_complete=True)-mu
    lines = zero_level_lines(x,y,completed-mu,energy_function=field)
    assert lines
    assert max(np.max(np.abs(field(line))) for line in lines) < 1e-7


def test_origin_txt_fermi_vertices_keep_the_energy_tolerance_after_import(tmp_path):
    from tdbg_scf.origin_export import _surface_tables
    geom = MoireGeometry(theta_deg=1.2)
    kpts,_ = make_uniform_mbz_grid(geom,16,16)
    uv = np.linalg.solve(np.column_stack([geom.b1m,geom.b2m]),kpts.T).T
    u,v = uv.T
    values = 1000*(np.cos(2*np.pi*u)+np.cos(2*np.pi*v)+np.cos(2*np.pi*(u-v)))
    energies = np.column_stack([np.full(len(kpts),-4000), values,
                               np.full(len(kpts),4000),np.full(len(kpts),5000)])
    contour = dict(kpts=kpts,evals_K_meV=energies)
    inventory = _surface_tables(tmp_path,"K",contour,geom,
                                dict(stage="before_stoner",mu_meV=1.13),-1.13,1.13,145)
    entry = inventory[-1]
    assert entry["contour_components"]
    lines = numeric_table(tmp_path / entry["filename"])
    surface = PeriodicBandSurface(geom,kpts,values)
    for component in range(entry["contour_components"]):
        vertices = lines[:,2*component:2*component+2]
        vertices = vertices[np.isfinite(vertices).all(axis=1)]
        assert np.max(np.abs(surface(vertices,c3_complete=True)-1.13)) <= 1e-7


def test_detailed_pipeline_exports_both_stages_without_extra_scf(tmp_path,monkeypatch):
    import argparse
    from examples import run_full_band_stoner as workflow
    from tdbg_scf import FullSCFSolver
    parser = argparse.ArgumentParser()
    workflow.add_hamiltonian_args(parser)
    workflow.add_full_scf_args(parser)
    workflow.add_stoner_args(parser)
    workflow.add_analysis_args(parser)
    args = parser.parse_args(["--cutoff","0","--theta-deg","1.2",
                             "--grid-n1","2","--grid-n2","2","--full-max-iter","100",
                             "--kBT-meV","2","--points-per-segment","2","--dos-bins","12"])
    calls = []
    solve = FullSCFSolver.solve
    def track(solver,config):
        calls.append(config)
        return solve(solver,config)
    monkeypatch.setattr(FullSCFSolver,"solve",track)
    task = dict(n_index=0,D_index=0,n_cm2=1e11,D_Vnm=.02,label="small")
    row = workflow.run_point(task,args,detail_dir=tmp_path)
    assert row["status"] == "ok",row.get("error")
    assert row["stoner_dos_valid"]
    assert len(calls) == 1
    for stage in ("before_stoner","after_stoner"):
        path = tmp_path / "origin_txt" / stage / "dos.txt"
        assert path.is_file(), "detailed pipeline did not export Origin data"
        np.testing.assert_array_equal(numeric_table(path)[[0,-1],0],[-100,100])


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (3, 1)])
def test_undersampled_contour_exports_raw_data_without_claiming_no_crossing(tmp_path, shape):
    from tdbg_scf.origin_export import _surface_tables
    geom = MoireGeometry(theta_deg=1.2)
    kpts, _ = make_uniform_mbz_grid(geom, *shape)
    energies = np.tile([-1., 1.], (len(kpts), 1))
    context = dict(stage="before_stoner", mu_meV=0., stacking_top_to_bottom="BA-AB")
    colored = {}
    files = _surface_tables(tmp_path, "K", {"kpts": kpts, "evals_K_meV": energies},
                            geom, context, 0., 0., 31,
                            lower_block_weights=np.full_like(energies, .5), block_lines_out=colored)
    raw = numeric_table(tmp_path / "fermi_bands_raw_K.txt")
    assert len(raw) == len(kpts) and np.isfinite(raw).all()
    unavailable = [item for item in files if "lines" in item["filename"] or "square" in item["filename"]]
    assert unavailable and all(item["rows"] == 0 for item in unavailable)
    assert all(item["contour_status"] == "unavailable_insufficient_mesh" for item in unavailable)
    assert colored["K"] == []
