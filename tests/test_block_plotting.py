"""Block colors preserve physical energy references and disconnected contours."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pytest


def boundary():
    angles = np.arange(7) * np.pi / 3.0
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    vertices[-1] = vertices[0]
    return vertices


def stage_data(stage="before_stoner"):
    names = ("K", "Kp") if stage == "before_stoner" else ("K_up", "Kp_up", "K_down", "Kp_down")
    context = {"stage": stage, "stacking_top_to_bottom": "BA-AB",
               "theta_deg": 1.2, "mu_meV": 14.0}
    data = {"dist": np.array([0.0, 0.4, 0.8, 1.2]),
            "energies_meV": np.array([[-120., 180., -60.], [-80., 180., -40.],
                                       [0., 180., -20.], [120., 180., 40.]]),
            "weights_lower_block": np.array([[0., 0., 1.], [.2, .1, .8],
                                            [.8, .2, .2], [1., .3, 0.]]),
            "ticks": np.arange(4), "labels": ["K", "Gamma", "M", "Kprime"]}
    return context, {name: data for name in names}, {name: [] for name in names}


@pytest.mark.parametrize("stage,rows", [("before_stoner", 2), ("after_stoner", 4)])
def test_figure_has_a_band_and_contour_panel_for_each_flavor(stage, rows):
    from tdbg_scf.block_plotting import make_block_stage_figure

    context, bands, lines = stage_data(stage)
    # The layout has a stable physical ordering even if the caller's dict does not.
    bands = dict(reversed(list(bands.items())))
    fig = make_block_stage_figure(context, bands, lines, boundary(),
                                 point_info={"nu_total": -1.0, "D_Vnm": .1, "cutoff": 3})
    try:
        assert len(fig.axes) == 2 * rows + 1
        names = ("K", "Kp") if rows == 2 else ("K_up", "Kp_up", "K_down", "Kp_down")
        assert [ax.get_gid() for ax in fig.axes[:-1]] == [kind + "_" + name
                   for name in names for kind in ("bands", "fermi")]
        assert fig.axes[-1].get_gid() == "block_weight_colorbar"
        width, height = fig.get_size_inches()
        assert height > width
        if rows == 4:
            assert height >= 14
        figure_text = " ".join(text.get_text() for text in fig.texts)
        assert "BA" in figure_text and "AB" in figure_text and "C3" in figure_text
        assert "full-basis" in figure_text and "layers 3" in figure_text
    finally:
        plt.close(fig)


def test_every_visible_band_uses_its_segment_average_weight_without_a_mu_shift():
    from tdbg_scf.block_plotting import make_block_stage_figure

    context, bands, lines = stage_data()
    original = bands["K"]["energies_meV"].copy()
    fig = make_block_stage_figure(context, bands, lines, boundary())
    try:
        ax = fig.axes[0]
        collections = [item for item in ax.collections if isinstance(item, LineCollection)]
        assert len(collections) == 1
        collection = collections[0]
        assert len(collection.get_segments()) == 6
        np.testing.assert_allclose(collection.get_array(), [.1, .5, .9, .9, .5, .1])
        assert (collection.norm.vmin, collection.norm.vmax) == (0.0, 1.0)
        assert collection.cmap(0.0)[:3] != collection.cmap(1.0)[:3]
        assert min(collection.cmap(.5)[:3]) < .9  # Mixed states remain visible on white.
        np.testing.assert_array_equal(collection.get_segments()[1][:, 1], [-80., 0.])
        np.testing.assert_array_equal(bands["K"]["energies_meV"], original)
        assert ax.get_ylim() == (-100.0, 100.0)
    finally:
        plt.close(fig)


def test_contour_components_are_never_joined_and_empty_flavors_have_no_colored_lines():
    from tdbg_scf.block_plotting import make_block_stage_figure

    context, bands, lines = stage_data()
    first = np.array([[-.6, -.3], [-.2, .1], [.3, .2]])
    second = np.array([[-.5, .5], [.1, .6]])
    lines["K"] = [(first, np.array([0., .5, 1.])),
                  (second, np.array([1., 0.]))]
    fig = make_block_stage_figure(context, bands, lines, boundary())
    try:
        ax, empty_ax = fig.axes[1], fig.axes[3]
        collection, = [item for item in ax.collections if isinstance(item, LineCollection)]
        expected = np.array([[first[0], first[1]], [first[1], first[2]],
                             [second[0], second[1]]])
        np.testing.assert_array_equal(collection.get_segments(), expected)
        np.testing.assert_allclose(collection.get_array(), [.25, .75, .5])
        assert collection.norm is fig.axes[0].collections[0].norm
        assert (collection.norm.vmin, collection.norm.vmax) == (0.0, 1.0)
        assert not empty_ax.collections
        assert any("No" in text.get_text() and "contours" in text.get_text()
                   for text in empty_ax.texts)
        for panel in (ax, empty_ax):
            assert panel.get_aspect() == 1.0
            np.testing.assert_allclose(panel.get_xlim(), [-1.2, 1.2])
            np.testing.assert_allclose(panel.get_ylim(), [-1.2, 1.2])
            # Only the explicitly supplied, closed MBZ boundary is a plain line.
            assert len(panel.lines) == 1
            np.testing.assert_array_equal(panel.lines[0].get_xydata(), boundary())
    finally:
        plt.close(fig)


def test_renderer_writes_only_pdf_png_and_reports_actual_file_sizes(tmp_path):
    from tdbg_scf.block_plotting import render_block_stage

    context, bands, lines = stage_data()
    initial_figures = plt.get_fignums()
    records = render_block_stage(tmp_path, context, bands, lines, boundary())
    names = ["block_resolved_bands_and_contours.pdf", "block_resolved_bands_and_contours.png"]
    assert [record["filename"] for record in records] == names
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names)
    for record in records:
        path = Path(tmp_path) / record["filename"]
        assert record["bytes"] == path.stat().st_size > 1000
    pdf_bytes = (tmp_path / names[0]).read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"/FontFile2" in pdf_bytes
    assert b"/Subtype /Type3" not in pdf_bytes
    assert (tmp_path / names[1]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == initial_figures
