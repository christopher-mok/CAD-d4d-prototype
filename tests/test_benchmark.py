"""Benchmark infrastructure: target generators and comparison math."""
import math

import numpy as np
import pytest

from cad_d4d.benchmark.analysis import compare, interp_loglog
from cad_d4d.device import to_numpy
from cad_d4d.geometry.state import watertightness_error
from cad_d4d.targets import analytic as A
from cad_d4d.targets.random_grammar import GrammarSpec, random_grammar_state
from cad_d4d.targets.synthetic import TargetConfig


def test_mesh_radial_sphere_volume_and_orientation():
    V, T, N = A.mesh_radial(A.sphere(1.0), res=41)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert abs(vol - 4 / 3 * math.pi) < 0.01
    assert np.all(np.einsum("ij,ij->i", N, V) > 0.99)  # outward unit normals


def test_analytic_target_sdf_matches_radial_function():
    r = A.add(A.sphere(), A.gaussian_bumps([[1, 0, 0]], [0.2], [0.3]))
    tg = A.AnalyticSpec("t", r, cfg=TargetConfig(sdf_res=48, n_coverage=500)).build()
    d = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    radii = r(d)
    assert abs(radii[0] - 1.2) < 1e-6
    phi = to_numpy(tg.sdf(d * radii[:, None], exact=True))
    assert np.max(np.abs(phi)) < 2e-3
    assert to_numpy(tg.sdf(d * (radii[:, None] + 0.1), exact=True)).min() > 0.05


def test_superellipsoid_is_boxier_than_sphere():
    d = np.array([[1, 1, 1.0]]) / np.sqrt(3)
    assert A.superellipsoid(4.0)(d)[0] > 1.2


@pytest.mark.parametrize("spec", [GrammarSpec("a", seed=1, interior=2, near_edge=1),
                                  GrammarSpec("b", seed=2, interior=1, edge=1, corner=1)])
def test_random_grammar_targets_are_valid(spec):
    s = random_grammar_state(spec)
    s.cx.check_invariants()
    assert s.n_faces > 6
    assert watertightness_error(s) < 1e-12


def test_loglog_comparison():
    cps = np.array([10.0, 100.0, 1000.0])
    fits = np.array([1e-2, 1e-4, 1e-6])
    assert interp_loglog(31.6227766, cps, fits) == pytest.approx(1e-3, rel=1e-6)
    assert interp_loglog(5.0, cps, fits) is None
    eff, save = compare({"n_cp": 100.0, "fit": 1e-5}, cps, fits)
    assert eff == pytest.approx(0.1)          # 10x better than uniform at 100 cps
    assert save == pytest.approx(10 ** 0.5)   # uniform needs ~316 cps for 1e-5
