"""Small independent counterexamples for the four-layer Hartree solver."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import root

from tdbg_scf.density import find_mu_for_density, full_layer_density
from tdbg_scf.electrostatics import LayerElectrostatics
from tdbg_scf.mixing import AndersonMixer
from tdbg_scf.solver_full import FullSCFConfig, FullSCFSolver


class FourLevelHamiltonian:
    """One localized orbital per layer, using the real four-layer electrostatics."""

    def __init__(self):
        self.potentials = []

    def layer_projectors_diagonal(self):
        return np.eye(4)

    def diagonalize(self, kpts, U):
        self.potentials.append(np.array(U, copy=True))
        e, v = np.linalg.eigh(np.diag(np.array([-.1, .1, -.2, .2]) + U))
        return e[None], v[None]


def test_damped_anderson_solves_a_stiff_linear_fixed_point():
    mixer = AndersonMixer(beta=.5, memory=6, fallback_alpha=.05)
    x = np.zeros(2)
    b = np.array([1., -1.])
    for _ in range(30):
        fx = -5.75*x + b
        if np.max(np.abs(fx-x)) < 1e-10:
            break
        x = mixer.update(x, fx)
    np.testing.assert_allclose(x, b/6.75, atol=1e-10, rtol=0)


def test_scf_returns_and_reuses_the_state_that_passed_tolerance(monkeypatch):
    import tdbg_scf.solver_full as full

    weights = np.array([1e-5])
    electro = LayerElectrostatics(n_layers=4)
    auxiliary = FourLevelHamiltonian()

    def response(U):
        e, v = auxiliary.diagonalize(np.zeros((1, 2)), U)
        mu = find_mu_for_density(e, weights, 0., kBT_meV=.2)
        n = full_layer_density(mu, e, v, weights, np.eye(4), kBT_meV=.2)
        return electro.update_U_from_density(n, 0.)

    fixed = root(lambda U: response(U)-U, np.zeros(4), tol=1e-11)
    assert fixed.success
    direction = np.array([1., -1., 0., 0.])
    h = 1e-7
    sensitivity = np.max(np.abs((response(fixed.x+h*direction)-response(fixed.x-h*direction))/(2*h)-direction))
    tolerance = 1e-4
    start = fixed.x + .8*tolerance/sensitivity*direction
    assert np.max(np.abs(response(start)-start)) < tolerance
    monkeypatch.setattr(full, "linear_potential_from_D", lambda *a, **kw: start.copy())

    ham = FourLevelHamiltonian()
    result = FullSCFSolver(ham, np.zeros((1, 2)), weights).solve(
        FullSCFConfig(max_iter=1, tol_meV=tolerance, kBT_meV=.2))

    np.testing.assert_array_equal(result.U_meV, ham.potentials[0])
    np.testing.assert_allclose(result.U_meV, start, atol=2e-17, rtol=0)
    assert result.converged and result.residual_meV < tolerance
    assert len(ham.potentials) == 1
    expected_e, expected_v = np.linalg.eigh(np.diag(np.array([-.1, .1, -.2, .2])+result.U_meV))
    np.testing.assert_allclose(result.evals[0], expected_e)
    np.testing.assert_allclose(np.abs(result.evecs[0]), np.abs(expected_v))


@pytest.mark.parametrize("target_cm2", [-1e11, 1e11])
def test_zero_temperature_scf_conserves_requested_fractional_charge(target_cm2):
    result = FullSCFSolver(FourLevelHamiltonian(), np.zeros((1, 2)), np.array([1e-5])).solve(
        FullSCFConfig(target_density_cm2=target_cm2, initial_U="zero", kBT_meV=0., max_iter=1))
    assert np.sum(result.n_layer_cm2) == pytest.approx(target_cm2, abs=1e4)
    assert np.isfinite(result.U_meV).all() and np.isfinite(result.residual_meV)


def test_zero_temperature_mu_uses_the_midpoint_of_a_filled_gap():
    mu = find_mu_for_density(np.array([[-10., 10.]]), np.ones(1), 0., degeneracy=1, kBT_meV=0.)
    assert mu == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("max_iter", [0, -1])
def test_empty_iteration_budget_is_rejected(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        FullSCFSolver(FourLevelHamiltonian(), np.zeros((1, 2)), np.array([1e-5])).solve(
            FullSCFConfig(max_iter=max_iter))


@pytest.mark.parametrize("overrides", [dict(converged=False), dict(residual_meV=np.nan),
    dict(n_layer_cm2=np.zeros(4)), dict(mu_meV=np.inf)])
def test_shared_guard_rejects_untrusted_scf_states(overrides):
    from tdbg_scf.scf_checks import require_converged_scf, scf_convergence_diagnostics

    values = dict(converged=True, residual_meV=0., U_meV=np.zeros(4),
                  n_layer_cm2=np.full(4, 2.5e10), mu_meV=0.)
    values.update(overrides)
    result = SimpleNamespace(**values)
    assert not scf_convergence_diagnostics(result, 1e11, 1e-5)["scf_valid"]
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        require_converged_scf(result, 1e11, 1e-5)
