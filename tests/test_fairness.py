"""Fairness energy: refinement invariance of the bending quadrature."""
import numpy as np
import pytest

from cad_d4d.losses.fairness import fairness_operator
from cad_d4d.rewrites.knot_insert import KnotInsert
from cad_d4d.rewrites.local_refine import LocalRefine
from cad_d4d.rewrites.split_face import SplitFace

from helpers import perturbed_state


def energy(state, kind="bending"):
    F = fairness_operator(state, kind)
    return float(np.sum((F @ state.values()) ** 2))


@pytest.mark.parametrize("make", [
    lambda fid: KnotInsert(fid, "u", 0.3),
    lambda fid: SplitFace(fid, "v", 0.55),
    lambda fid: LocalRefine(fid, 0.4, 0.6, width=0.3, refine_knots=2),
])
def test_bending_energy_invariant_under_exact_rewrites(make):
    s = perturbed_state(21)
    fid = sorted(s.cx.faces)[1]
    s2 = make(fid).apply(s).state
    e1, e2 = energy(s), energy(s2)
    assert abs(e1 - e2) < 1e-10 * e1


def test_bending_energy_is_quadratic_and_positive():
    s = perturbed_state(0, sigma=0.0)
    e = energy(s)
    assert e > 1.0
    s2 = s.copy()
    s2.set_values(2.0 * s.values())
    assert abs(energy(s2) - 4.0 * e) < 1e-10 * e


def test_control_net_variant_available():
    s = perturbed_state(22)
    assert energy(s, "control_net") > 0
