import numpy as np


def test_flavor_order_and_names():
    from tdbg_scf.flavors import FLAVORS, flavor_names

    assert flavor_names() == ["K_up", "Kp_up", "K_down", "Kp_down"]
    assert [(f.valley, f.spin) for f in FLAVORS] == [(1, 1), (-1, 1), (1, -1), (-1, -1)]


def test_filling_density_round_trip():
    from tdbg_scf.filling import (
        density_cm2_to_filling,
        filling_to_density_cm2,
        moire_cell_area_A2_from_weights,
    )

    weights = np.array([0.25, 0.25])
    area = moire_cell_area_A2_from_weights(weights)
    assert area == 2.0
    density = filling_to_density_cm2(3.0, area)
    assert density == 1.5e16
    assert density_cm2_to_filling(density, area) == 3.0
