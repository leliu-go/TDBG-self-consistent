"""Normalized electronic order from central-band electrons and valence holes."""
from __future__ import annotations

import math
from itertools import chain, repeat
import numpy as np

from ..band_gaps import GAP_SPECTRUM_SOURCE
from .dos import _stable_cumsum

FLAVORS = ('K_up', 'Kp_up', 'K_down', 'Kp_down')
POLARIZATION_FIELDS = ('spin_polarization_active_norm',
                       'valley_polarization_active_norm',
                       'spin_valley_polarization_active_norm')
LABEL_SIGNS = np.array([[1., 1., -1., -1.], [1., -1., 1., -1.],
                        [1., -1., -1., 1.]])
REPLACED_CARRIER_FIELDS = tuple(name.replace('_active_', '_carrier_')
                                for name in POLARIZATION_FIELDS)


def undefined_polarizations():
    return {field: float('nan') for field in POLARIZATION_FIELDS}


def polarizations_from_active_carriers(electron_nu_f, hole_nu_f):
    """P_a = |sum q_a (e-h)| / sum (e+h); holes keep electron labels."""
    electron = np.asarray(electron_nu_f, dtype=float)
    hole = np.asarray(hole_nu_f, dtype=float)
    if any(c.shape != (4,) or not np.isfinite(c).all() or
           np.any(c < 0.) or np.any(c > 1. + 1e-8) for c in (electron, hole)):
        raise ValueError('Need four central electron and hole populations in [0,1]')
    total = float(math.fsum(np.r_[electron, hole]))
    if total <= 1e-12:
        return undefined_polarizations()
    values = []
    for signs in LABEL_SIGNS:
        # A vacancy contributes the negative of the missing electron's
        # generator. Separate positive and negative sums preserve tiny minorities.
        positive = math.fsum(np.r_[electron[signs > 0.], hole[signs < 0.]])
        negative = math.fsum(np.r_[electron[signs < 0.], hole[signs > 0.]])
        value = abs(positive - negative) / total
        if value > 1. + 1e-12:
            raise ValueError('Active-carrier polarization exceeds one')
        if min(positive, negative) > 0. and value >= 1.:
            value = float(np.nextafter(1., 0.))
        values.append(min(value, 1.))
    return dict(zip(POLARIZATION_FIELDS, values))


def count_flavor_active_carriers(energies_meV, weights, A_M_A2, nu):
    """Fill the complete spectrum at T=0, then count conduction electrons and valence holes.

    Use the solver's half-filled full-band neutrality reference and equal
    fractional occupation of degenerate levels. Selecting the observable
    does not project or change the Hamiltonian or the Stoner energy.
    """
    if np.iscomplexobj(energies_meV):
        raise ValueError('Need real full-band eigenvalues')
    e, w = np.asarray(energies_meV, dtype=float), np.asarray(weights, dtype=float)
    if (e.ndim != 2 or not e.size or e.shape[1] % 2 or
            w.shape != (e.shape[0],) or not np.isfinite(e).all() or
            not np.isfinite(w).all() or np.any(w < 0.) or not np.any(w > 0.) or
            not np.isfinite(A_M_A2) or A_M_A2 <= 0. or not np.isfinite(nu)):
        raise ValueError('Need a complete even-band spectrum and finite quadrature')
    if np.any(np.diff(e, axis=1) < -1e-10):
        raise ValueError('Full bands must be ordered at each k')
    if abs(math.fsum(w) * float(A_M_A2) - 1.) > 1e-8:
        raise ValueError('Quadrature must give one state per band per moire cell')
    nk, nb = e.shape
    center = nb // 2
    valence = np.tile(np.arange(nb) == center - 1, nk)
    conduction = np.tile(np.arange(nb) == center, nk)
    dnu = np.repeat(w, nb) * float(A_M_A2)
    positive = dnu > 0.
    energies, dnu = e.ravel()[positive], dnu[positive]
    valence, conduction = valence[positive], conduction[positive]
    order = np.argsort(energies, kind='mergesort')
    energies, dnu = energies[order], dnu[order]
    valence, conduction = valence[order], conduction[order]
    capacity = math.fsum(dnu)
    if float(nu) < -.5 * capacity - 1e-10 or float(nu) > .5 * capacity + 1e-10:
        raise ValueError('Flavor filling is outside complete spectrum capacity')
    filling = float(np.clip(nu, -.5 * capacity, .5 * capacity))
    # Endpoint clamping has the same all-empty/all-full meaning as the full
    # solver, even when binary quadrature sums differ by an ulp from one.
    if filling <= -.5 * capacity:
        return 0., min(float(math.fsum(dnu[valence])), 1.)
    if filling >= .5 * capacity:
        return min(float(math.fsum(dnu[conduction])), 1.), 0.
    tol = max(1e-10, 16 * np.finfo(float).eps * max(1., float(np.max(np.abs(energies)))))
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(energies) > tol)]
    ends = np.r_[starts[1:], len(dnu)]
    capacities = np.add.reduceat(dnu, starts)
    uniform = bool(np.all(dnu == dnu[0]))
    if uniform:
        approximate = dnu[0] * (ends - len(dnu) // 2)
    else:
        approximate = _stable_cumsum(capacities) - .5 * capacity
        reference = -np.repeat(w, center) * float(A_M_A2)
    index = min(int(np.searchsorted(approximate, filling, side='left')), len(starts) - 1)

    def residual(position):
        # Subtract the requested filling in the same compensated sum. Rounding
        # a large prefix first can invent or erase a tiny minority population.
        if uniform:
            count = int(position) - len(dnu) // 2
            signed_capacity = dnu[0] if count >= 0 else -dnu[0]
            return math.fsum(chain(repeat(signed_capacity, abs(count)), (-filling,)))
        return math.fsum(chain(dnu[:position], reference, (-filling,)))

    lo, hi = 0, len(starts)
    while True:
        lower, upper = residual(starts[index]), residual(ends[index])
        if lower <= 0. <= upper:
            break
        if lower > 0.:
            hi = index
        else:
            lo = index + 1
        if lo >= hi:
            raise ValueError('Cannot locate the signed filling in the full spectrum')
        index = (lo + hi) // 2
    fraction = np.zeros(len(starts))
    vacancy = np.ones(len(starts))
    fraction[:index] = 1.
    vacancy[:index] = 0.
    if upper <= 0.:
        fraction[index], vacancy[index] = 1., 0.
    elif lower < 0.:
        exact_capacity = math.fsum(dnu[starts[index]:ends[index]])
        fraction[index] = min(-lower / exact_capacity, 1.)
        # Do not form 1-fraction: that would erase a real tiny vacancy.
        vacancy[index] = min(upper / exact_capacity, 1.)
    conduction_capacity = np.add.reduceat(dnu * conduction, starts)
    valence_capacity = np.add.reduceat(dnu * valence, starts)
    # Count vacancies directly from the signed filling boundaries.
    electron = float(math.fsum(conduction_capacity * fraction))
    hole = float(math.fsum(valence_capacity * vacancy))
    if min(electron, hole) < 0. or max(electron, hole) > 1. + 1e-8:
        raise ValueError('Central carrier populations violate one-band capacity')
    return min(electron, 1.), min(hole, 1.)


def count_flavor_active_occupation(energies_meV, weights, A_M_A2, nu):
    electron, hole = count_flavor_active_carriers(energies_meV, weights, A_M_A2, nu)
    return 1. + electron - hole


def _occupation_record(electron, hole, nu_f, method):
    electron, hole, nu = (np.asarray(c, dtype=float) for c in (electron, hole, nu_f))
    if nu.shape != (4,) or not np.isfinite(nu).all():
        raise ValueError('Need four finite flavor fillings')
    metrics = polarizations_from_active_carriers(electron, hole)
    active_net = electron - hole
    remote_net = nu - active_net
    return {'active_electron_nu_f': electron.tolist(),
            'active_hole_nu_f': hole.tolist(),
            'active_carrier_nu_f': (electron + hole).tolist(),
            'active_occupation_nu_f': (1. + active_net).tolist(),
            'active_net_filling_nu_f': active_net.tolist(),
            'remote_net_filling_nu_f': remote_net.tolist(),
            'remote_reference_unchanged': bool(np.max(np.abs(remote_net)) <= 1e-8),
            'method': method, 'polarizations': metrics}


def full_spectrum_active_occupations(evals_K, evals_Kp, weights, A_M_A2, nu_f):
    fillings = np.asarray(nu_f, dtype=float)
    if fillings.shape != (4,) or not np.isfinite(fillings).all():
        raise ValueError('Need four finite flavor fillings')
    counts = [count_flavor_active_carriers(spectrum, weights, A_M_A2, filling)
              for spectrum, filling in zip((evals_K, evals_Kp, evals_K, evals_Kp), fillings)]
    return _occupation_record(*np.asarray(counts).T, fillings,
                              'full_spectrum_T0_active_electrons_and_holes')


def carriers_from_certified_occupations(row, occupied, method, edge_tolerance_meV=1e-7,
                                        *, active_net_filling=None):
    """Split a known central population only when edges exclude e/h coexistence."""
    occupied = np.asarray(occupied, dtype=float)
    if (occupied.shape != (4,) or not np.isfinite(occupied).all() or
            np.any(occupied < 0.) or np.any(occupied > 2. + 1e-8)):
        raise ValueError('Need four certified central occupations in [0,2]')
    try:
        lower, upper, gap = [float(row[key]) for key in
            ('gap_nu0_lower_edge_meV', 'gap_nu0_upper_edge_meV', 'indirect_gap_nu0_meV')]
    except (KeyError, TypeError):
        return None
    if not np.isfinite([lower, upper, gap]).all():
        return None
    if abs(gap - (upper - lower)) > 1e-6:
        raise ValueError('Saved central band edges and gap are inconsistent')
    try:
        mu = np.array([row['mu_' + f + '_meV'] for f in FLAVORS], dtype=float)
    except (KeyError, TypeError):
        mu = None
    if mu is not None and not np.isfinite(mu).all():
        return None
    if gap <= edge_tolerance_meV and mu is None:
        return None
    if active_net_filling is None:
        net = occupied - 1.
    else:
        net = np.asarray(active_net_filling, dtype=float)
        if (net.shape != (4,) or not np.isfinite(net).all() or
                np.any(np.abs(net) > 1. + 1e-8) or
                not np.allclose(1. + net, occupied, rtol=0., atol=1e-8)):
            raise ValueError('Certified active net filling contradicts central occupation')
    electrons = np.maximum(net, 0.)
    holes = np.maximum(-net, 0.)
    if mu is not None:
        for index, (value, population) in enumerate(zip(mu, occupied)):
            empty_conduction = value < upper - edge_tolerance_meV
            full_valence = value > lower + edge_tolerance_meV
            if empty_conduction and population > 1. + 1e-8:
                raise ValueError('Empty conduction certificate contradicts central occupation')
            if full_valence and population < 1. - 1e-8:
                raise ValueError('Filled valence certificate contradicts central occupation')
            if empty_conduction and full_valence and active_net_filling is None:
                # A strict central gap proves exactly zero carriers, including
                # when a previously summed population differs from one by roundoff.
                electrons[index] = holes[index] = 0.
            elif gap <= edge_tolerance_meV and not (empty_conduction or full_valence):
                return None
    return _occupation_record(electrons, holes,
        [row['nu_' + f] for f in FLAVORS], method)


def certified_active_occupations_from_row(row, edge_tolerance_meV=1e-7):
    """Recover central occupations only when remote/active band edges prove them."""
    if (row.get('status') != 'ok' or row.get('stoner_dos_valid') is not True or
            row.get('gap_spectrum_source') != GAP_SPECTRUM_SOURCE):
        return None
    try:
        nu = np.array([row['nu_' + f] for f in FLAVORS], dtype=float)
        edges = []
        gaps = []
        for label in ('nu_minus4', 'nu_plus4'):
            lower, upper, gap = [float(row[key]) for key in
                ('gap_' + label + '_lower_edge_meV', 'gap_' + label + '_upper_edge_meV',
                 'indirect_gap_' + label + '_meV')]
            if abs(gap - (upper - lower)) > 1e-6:
                raise ValueError('Saved active/remote edges and gaps are inconsistent')
            edges.extend((lower, upper))
            gaps.append(gap)
    except (KeyError, TypeError):
        return None
    if not np.isfinite(np.r_[nu, edges, gaps]).all():
        return None
    if min(gaps) > edge_tolerance_meV:
        return carriers_from_certified_occupations(row, np.clip(1. + nu, 0., 2.),
            'certified_isolated_active_manifold', edge_tolerance_meV,
            active_net_filling=np.clip(nu, -1., 1.))
    try:
        mu = np.array([row['mu_' + f + '_meV'] for f in FLAVORS], dtype=float)
    except (KeyError, TypeError):
        return None
    if not np.isfinite(mu).all():
        return None
    remote_lower_max, active_min, active_max, remote_upper_min = edges
    counts = []
    net_counts = []
    for value, filling in zip(mu, nu):
        if value < active_min - edge_tolerance_meV:
            occupied = 0.
            if filling > -1. + 1e-8:
                raise ValueError('Empty active manifold contradicts saved flavor filling')
        elif value > active_max + edge_tolerance_meV:
            occupied = 2.
            if filling < 1. - 1e-8:
                raise ValueError('Filled active manifold contradicts saved flavor filling')
        elif (value > remote_lower_max + edge_tolerance_meV and
              value < remote_upper_min - edge_tolerance_meV):
            occupied = 1. + filling
            if occupied < -1e-8 or occupied > 2. + 1e-8:
                raise ValueError('Remote reference certificate contradicts saved filling')
            occupied = float(np.clip(occupied, 0., 2.))
        else:
            return None
        counts.append(occupied)
        net_counts.append(-1. if occupied == 0. else 1. if occupied == 2. else filling)
    return carriers_from_certified_occupations(row, counts,
        'certified_flavor_active_remote_edges', edge_tolerance_meV,
        active_net_filling=net_counts)


def active_polarization_metadata():
    return {'fields': list(POLARIZATION_FIELDS), 'flavor_order': list(FLAVORS),
        'formula_version': 'active_electron_minus_hole_v1',
        'formula': 'abs(sum_f(label_sign_f * (active_electron_nu_f - active_hole_nu_f))) / sum_f(active_electron_nu_f + active_hole_nu_f)',
        'label_signs': dict(zip(POLARIZATION_FIELDS, LABEL_SIGNS.tolist())),
        'active_bands_per_flavor': 2, 'active_band_indices_zero_based': 'Nb/2-1,Nb/2',
        'active_manifold': 'central_valence_and_conduction_bands_per_spin_and_valley',
        'reference': 'central_valence_full_conduction_empty_for_every_electronic_flavor',
        'hole_label_convention': 'missing_electron_flavor_negative_electronic_generator_contribution',
        'spin_valley_convention': 'electronic_s_times_tau_occupation_order_parameter',
        'occupation': 'T0_full_spectrum_fixed_flavor_filling_equal_fraction_in_degenerate_levels',
        'normalization': 'total_active_conduction_electrons_plus_valence_holes',
        'zero_denominator': 'NaN_when_total_active_carriers_le_1e-12',
        'validity': 'requires_status_ok_and_stoner_dos_valid_true',
        'meaning_of_one': 'all_active_electronic_generator_contributions_have_same_sign',
        'hamiltonian_and_stoner_energy': 'full_band_models_unchanged',
        'remote_occupation': 'recorded_separately_never_assumed_balanced_in_overlap_regimes',
        'replaced_fields': list(REPLACED_CARRIER_FIELDS),
        'existing_net_charge_columns': 'preserved_distinct_from_active_carrier_fractions'}


def active_result_fields(result):
    """Serialize central-band observables only when full spectra were supplied."""
    record = getattr(result, "active_occupation_record", None)
    return {**(undefined_polarizations() if record is None else record["polarizations"]),
            "active_occupation": record,
            "active_polarization_definition": active_polarization_metadata()}
