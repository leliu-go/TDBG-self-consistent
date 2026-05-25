"""Placeholders for active-space form-factor construction."""
from __future__ import annotations


def require_form_factors(active_vecs):
    if active_vecs is None:
        raise NotImplementedError("active wavefunction form factors are required")
    return active_vecs
