"""Origin companions for full-basis bilayer/lower_block spectral projections."""
from __future__ import annotations

import numpy as np

from .origin_geometry import PeriodicBandSurface


def block_context(context):
    upper, lower = context["stacking_top_to_bottom"].split("-")
    return {**context, "color_weight": "lower_block_probability",
            "lower_block_layers_top_to_bottom": [3, 4],
            "color_weight_definition": "sum_l3_to4_s_G_abs_u_squared_divided_by_full_norm",
            "color_0": upper+"_layers_1_2", "color_1": lower+"_layers_3_4"}


def bounded_probability(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or np.any(values < -1e-12) or np.any(values > 1+1e-12):
        raise ValueError("block probability must be finite and between zero and one")
    return np.clip(values, 0, 1)


def band_block_table(folder, name, path, context, offset, projection, writer):
    energies = np.asarray(path["bands"], float)+offset
    projection = bounded_probability(projection)
    if projection.shape != energies.shape:
        raise ValueError("path block weights must match every saved band")
    dist, ticks = np.asarray(path["dist"], float), np.asarray(path["ticks"], int)
    labels = np.full(len(dist), "", dtype=object)
    labels[ticks] = ["K", "Gamma", "M", "Kprime"]
    values = np.empty((len(dist), 2+2*energies.shape[1]), dtype=object)
    values[:, 0], values[:, 1] = labels, dist
    values[:, 2::2], values[:, 3::2] = energies, projection
    columns = ["path_label", "path_distance_Ainv"]
    for band in range(energies.shape[1]):
        columns.extend([f"band_{band+1:04d}_E_rel_mu_meV", f"band_{band+1:04d}_W_lower_block"])
    info = writer(folder, "bands_block_weight_"+name+".txt", values, columns,
                  block_context(context), flavor_or_valley=name,
                  sampling="saved_path_full_basis_eigenvector_probability",
                  path_tick_rows_zero_based=ticks.tolist(), path_tick_x_Ainv=dist[ticks].tolist())
    return info, dict(dist=dist, energies_meV=energies, weights_lower_block=projection,
                      ticks=ticks, labels=["K", "Gamma", "M", "Kprime"])


def surface_block_tables(folder, name, context, geom, kpts, projection, raw, square,
                         queries, indices, vb, cb, segments, line_columns, band_numbers, writer):
    projection = bounded_probability(projection)
    field = PeriodicBandSurface(geom, kpts, projection[:, indices])
    sampled = bounded_probability(field(queries, c3_complete=True))
    vb_col, cb_col = np.searchsorted(indices, [vb, cb])
    columns = ["VB_kx_Ainv", "VB_ky_Ainv", "VB_E_rel_mu_meV", "VB_W_lower_block",
               "CB_kx_Ainv", "CB_ky_Ainv", "CB_E_rel_mu_meV", "CB_W_lower_block"]
    context = block_context(context)
    raw_weighted = np.column_stack([raw[:, :3], projection[:, vb], raw[:, 3:], projection[:, cb]])
    square_weighted = np.column_stack([square[:, :3], sampled[:, vb_col], square[:, 3:], sampled[:, cb_col]])
    completion = "C3_group_average_of_three_rotated_periodic_bilinear_interpolants"
    inventory = [writer(folder, "fermi_bands_block_weight_raw_"+name+".txt", raw_weighted,
                        columns, context, sampling="original_saved_grid_eigenvector_probability"),
                 writer(folder, "fermi_bands_block_weight_square_C3_"+name+".txt", square_weighted,
                        columns, context, sampling=completion,
                        color_weight_source="C3_interpolated_full_basis_probability")]
    colored_lines, columns = [], []
    for i, (line, number) in enumerate(zip(segments, band_numbers)):
        column = int(np.searchsorted(indices, number-1))
        color = bounded_probability(field(line, c3_complete=True)[:, column])
        colored_lines.append((line, color))
        prefix = line_columns[2*i][:-len("_kx_Ainv")]
        columns.extend([line_columns[2*i], line_columns[2*i+1], prefix+"_W_lower_block"])
    table = np.full((max((len(line) for line in segments), default=0),
                     3*len(segments) if segments else 3), np.nan)
    for i, (line, color) in enumerate(colored_lines):
        table[:len(line), 3*i:3*i+2] = line
        table[:len(line), 3*i+2] = color
    inventory.append(writer(folder, "fermi_lines_block_weight_C3_"+name+".txt", table,
        columns or ["kx_Ainv", "ky_Ainv", "W_lower_block"], context,
        numeric_format="%.17g", sampling=completion, contour_level_E_rel_mu_meV=0.,
        contour_components=len(segments), component_band_numbers=band_numbers,
        contour_vertex_energy_tolerance_meV=1e-7,
        color_weight_source="C3_interpolated_full_basis_probability_not_new_eigenvector_at_vertex",
        contour_status="crossing" if segments else "no_one_dimensional_crossing",
        missing_value="nan_padding_between_independent_XY_weight_triplets"))
    return inventory, colored_lines


def block_dos_table(folder, energies, k_weights, mus, centers, sigma, total, projections,
                    expected_layer_dos, context, writer, dos_evaluator):
    lower = dos_evaluator(energies, k_weights, mus, centers, sigma, state_weights=projections)
    if np.any(lower > total+1e-12*np.maximum(total, 1e-12)) or np.any(lower < 0):
        raise ValueError("projected DOS must lie between zero and total DOS")
    lower = np.minimum(lower, total)
    upper = total-lower
    fraction = np.divide(lower, total, out=np.full_like(total, np.nan), where=total > 0)
    at_mu = float(dos_evaluator(energies, k_weights, mus, np.array([0.]), sigma,
                               state_weights=projections)[0])
    expected = float(np.sum(np.asarray(expected_layer_dos)[2:]))
    if not np.isclose(at_mu, expected, rtol=2e-8, atol=1e-14):
        raise ValueError("projected lower_block DOS at mu disagrees with saved layers 3 through 4")
    upper_name, lower_name = context["stacking_top_to_bottom"].split("-")
    columns = ["E_rel_mu_meV", "DOS_"+upper_name+"_Ainv2_meVinv",
               "DOS_"+lower_name+"_Ainv2_meVinv", "DOS_ALL_Ainv2_meVinv",
               "fraction_DOS_"+lower_name]
    return writer(folder, "block_dos.txt", np.column_stack([centers, upper, lower, total, fraction]),
        columns, block_context(context), dos_sigma_meV=sigma,
        ratio_definition="DOS_lower_block_over_DOS_all",
        ratio_when_total_DOS_zero="undefined_nan", lower_block_DOS_at_mu=at_mu,
        saved_layer_DOS_3_to_4_at_mu=expected)
