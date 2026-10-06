"""Milestone 1: B-spline basis mathematics."""
import math

import numpy as np
import pytest

from cad_d4d.geometry import bspline_basis as bb


def cox_de_boor(knots, i, p, u):
    """Naive recursive reference (right-closed at the final knot)."""
    if p == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        if u == knots[-1] and knots[i] < knots[i + 1] == knots[-1]:
            return 1.0
        return 0.0
    left = 0.0
    if knots[i + p] != knots[i]:
        left = (u - knots[i]) / (knots[i + p] - knots[i]) * cox_de_boor(knots, i, p - 1, u)
    right = 0.0
    if knots[i + p + 1] != knots[i + 1]:
        right = (knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1]) * cox_de_boor(knots, i + 1, p - 1, u)
    return left + right


def random_knots(rng, n_interior, degree=3, with_repeats=True):
    inner = np.sort(rng.uniform(0.05, 0.95, n_interior))
    if with_repeats and n_interior >= 2:
        inner[1] = inner[0]  # a double knot
    return np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_matches_recursive_cox_de_boor(rng):
    knots = random_knots(rng, 5)
    us = np.linspace(0, 1, 101)
    B = bb.basis_matrix(knots, 3, us)
    ref = np.array([[cox_de_boor(knots, a, 3, u) for a in range(B.shape[1])] for u in us])
    assert np.max(np.abs(B - ref)) < 1e-13


def test_bernstein_case():
    knots = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    us = np.linspace(0, 1, 33)
    B = bb.basis_matrix(knots, 3, us)
    ref = np.stack([math.comb(3, i) * us**i * (1 - us) ** (3 - i) for i in range(4)], axis=1)
    assert np.max(np.abs(B - ref)) < 1e-14
    dB = bb.basis_matrix(knots, 3, us, deriv=1)
    dref = np.stack([
        -3 * (1 - us) ** 2,
        3 * (1 - us) ** 2 - 6 * us * (1 - us),
        6 * us * (1 - us) - 3 * us**2,
        3 * us**2,
    ], axis=1)
    assert np.max(np.abs(dB - dref)) < 1e-13


def test_partition_of_unity_and_nonnegativity(rng):
    for trial in range(5):
        knots = random_knots(rng, 2 + trial)
        us = np.linspace(0, 1, 257)
        B = bb.basis_matrix(knots, 3, us)
        assert np.max(np.abs(B.sum(axis=1) - 1.0)) < 1e-14
        assert B.min() > -1e-15
        dB = bb.basis_matrix(knots, 3, us, deriv=1)
        assert np.max(np.abs(dB.sum(axis=1))) < 1e-11


def test_derivatives_match_finite_differences(rng):
    knots = random_knots(rng, 4, with_repeats=False)
    cps = rng.normal(size=(bb.num_basis(knots, 3), 3))
    us = np.linspace(0.01, 0.99, 97)
    h = 1e-6
    for d in (1, 2):
        analytic = bb.evaluate_curve(knots, 3, cps, us, deriv=d)
        lower = bb.evaluate_curve(knots, 3, cps, us - h, deriv=d - 1)
        upper = bb.evaluate_curve(knots, 3, cps, us + h, deriv=d - 1)
        fd = (upper - lower) / (2 * h)
        # Skip samples within h of a knot (derivative kinks for d=2).
        mask = np.min(np.abs(us[:, None] - knots[None, :]), axis=1) > 10 * h
        assert np.max(np.abs(analytic[mask] - fd[mask])) < 1e-6 * max(1, np.abs(analytic).max())


def test_knot_insertion_is_exact(rng):
    knots = random_knots(rng, 3)
    cps = rng.normal(size=(bb.num_basis(knots, 3), 3))
    us = np.linspace(0, 1, 1001)
    before = bb.evaluate_curve(knots, 3, cps, us)
    k, P = knots, cps
    for t in [0.3, 0.3, 0.71, knots[4]]:  # includes repeated and existing knots
        A, k = bb.insertion_matrix(k, 3, t)
        P = A @ P
    after = bb.evaluate_curve(k, 3, P, us)
    assert np.max(np.abs(before - after)) < 1e-13


def test_refinement_matrix_matches_sequential(rng):
    knots = random_knots(rng, 2, with_repeats=False)
    new = np.sort(np.concatenate([knots, [0.2, 0.55, 0.55]]))
    A = bb.refinement_matrix(knots, new, 3)
    cps = rng.normal(size=(bb.num_basis(knots, 3), 3))
    us = np.linspace(0, 1, 301)
    err = bb.evaluate_curve(knots, 3, cps, us) - bb.evaluate_curve(new, 3, A @ cps, us)
    assert np.max(np.abs(err)) < 1e-13
    with pytest.raises(ValueError):
        bb.refinement_matrix(new, knots, 3)


def test_segment_extraction_is_exact(rng):
    knots = random_knots(rng, 4)
    cps = rng.normal(size=(bb.num_basis(knots, 3), 3))
    for a, b in [(0.0, 0.37), (0.21, 0.83), (0.5, 1.0), (0.0, 1.0), (knots[4], knots[6])]:
        A, seg = bb.segment_matrix(knots, 3, a, b)
        s = np.linspace(0, 1, 201)
        orig = bb.evaluate_curve(knots, 3, cps, a + s * (b - a))
        sub = bb.evaluate_curve(seg, 3, A @ cps, s)
        assert np.max(np.abs(orig - sub)) < 1e-13
        assert seg[0] == 0.0 and seg[-1] == 1.0


def test_reverse_knots(rng):
    knots = random_knots(rng, 3)
    cps = rng.normal(size=(bb.num_basis(knots, 3), 3))
    s = np.linspace(0, 1, 101)
    fwd = bb.evaluate_curve(knots, 3, cps, s)
    rev = bb.evaluate_curve(bb.reverse_knots(knots), 3, cps[::-1], 1 - s)
    assert np.max(np.abs(fwd - rev)) < 1e-13


def test_multiset_utilities():
    a = np.array([0, 0, 0.5, 0.5, 1.0])
    b = np.array([0, 0.5, 1.0])
    assert np.allclose(bb.knot_multiset_difference(a, b), [0, 0.5])
    assert bb.is_knot_subset(b, a)
    assert not bb.is_knot_subset(a, b)
    assert bb.multiplicity(a, 0.5 + 1e-12) == 2


def test_vectorized_basis_matches_reference(rng):
    for trial in range(6):
        knots = random_knots(rng, 1 + trial)
        if trial >= 3:  # triple (C0) knot
            knots = np.sort(np.concatenate([knots, [knots[4]] * 2]))
        us = np.concatenate([np.linspace(0, 1, 97), knots[3:-3]])
        for d in range(4):
            A = bb.basis_matrix(knots, 3, us, d)
            R = bb.basis_matrix_reference(knots, 3, us, d)
            assert np.max(np.abs(A - R)) < 1e-9 * max(1.0, np.abs(R).max())
