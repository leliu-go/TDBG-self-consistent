import numpy as np

from tdbg_scf.solver_projected import select_active_indices


def test_overlap_selection_tracks_subspace_when_energy_window_changes():
    evecs = np.tile(np.eye(4, dtype=np.complex128), (3, 1, 1))
    evals = np.array(
        [
            [-20.0, -1.0, 1.0, 20.0],
            [-1.2, -1.1, 30.0, 31.0],
            [-1.3, -1.2, 32.0, 33.0],
        ]
    )

    selected, diagnostics = select_active_indices(
        evals,
        evecs,
        mu=0.0,
        n_active=2,
        mode="overlap",
        seed_index=0,
    )

    assert selected.tolist() == [[1, 2], [1, 2], [1, 2]]
    assert diagnostics["ambiguous_overlap_points"] == 0
    assert diagnostics["min_overlap_score_gap"] > 0.9


def test_overlap_selection_tracks_two_dimensional_grid_neighbors():
    evecs = np.tile(np.eye(4, dtype=np.complex128), (4, 1, 1))
    evals = np.array(
        [
            [-20.0, -1.0, 1.0, 20.0],
            [-1.2, -1.1, 30.0, 31.0],
            [-1.3, -1.2, 32.0, 33.0],
            [-1.4, -1.3, 34.0, 35.0],
        ]
    )

    selected, diagnostics = select_active_indices(
        evals,
        evecs,
        mu=0.0,
        n_active=2,
        mode="overlap",
        grid_shape=(2, 2),
        seed_index=0,
    )

    assert selected.tolist() == [[1, 2], [1, 2], [1, 2], [1, 2]]
    assert diagnostics["tracked_points"] == 3
