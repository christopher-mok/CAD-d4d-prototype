"""Carrier (edge) knot refinement and the split-line crease penalty."""
import numpy as np
import pytest

from cad_d4d.geometry import bspline_basis as bb
from cad_d4d.geometry.state import root_deviation, watertightness_error
from cad_d4d.losses.fairness import fairness_operator
from cad_d4d.rewrites.carrier_knots import CarrierKnotInsert, CarrierKnotRemove, carrier_faces
from cad_d4d.rewrites.knot_remove import KnotRemove
from cad_d4d.rewrites.merge_face import MergeFace
from cad_d4d.rewrites.split_face import SplitFace

from cad_d4d.optimization.discretization import DiscretizationConfig

from helpers import perturbed_state

CREASE_W = 1e4  # opt-in value (the default is 0)


def random_dofs_watertight(state, seed=0):
    s = state.copy()
    s.set_values(s.values() + 0.3 * np.random.default_rng(seed).normal(size=s.values().shape))
    return watertightness_error(s)


def test_carrier_knot_insert_is_exact_and_adds_boundary_dofs():
    s = perturbed_state(31)
    cid = sorted(s.cx.carriers)[4]
    faces = carrier_faces(s, cid)
    assert len(faces) == 2
    out = CarrierKnotInsert(cid, 0.4).apply(s)
    assert out.ok, out.reason
    s2 = out.state
    assert root_deviation(s, s2) < 1e-13
    assert watertightness_error(s2) < 1e-13 and random_dofs_watertight(s2) < 1e-12
    assert len(bb.interior_knots(s2.cx.carriers[cid].knots, 3)) == 1
    # +1 carrier DOF, +2 interior DOFs in each incident 4x4 face (one new strip)
    assert s2.n_control_points == s.n_control_points + 1 + 2 * 2
    # the new carrier DOF moves the shared boundary of both faces consistently
    P = s2.values()
    P[s2.dof_map.carrier_dof[cid]] += 0.05
    s2.set_values(P)
    assert watertightness_error(s2) < 1e-13
    assert root_deviation(s, s2) > 1e-3


def test_carrier_knot_insert_with_hanging_vertex():
    s = perturbed_state(32)
    fid = sorted(s.cx.faces)[2]
    s1 = SplitFace(fid, "u", 0.4).apply(s).state  # hanging vertices on two carriers
    hosted = [v for v in s1.cx.vertices.values() if not v.free]
    cid, s_host = hosted[0].host
    out = CarrierKnotInsert(cid, 0.5 * s_host + 0.25).apply(s1)
    assert out.ok, out.reason
    assert root_deviation(s1, out.state) < 1e-13
    assert random_dofs_watertight(out.state) < 1e-12


def test_carrier_knot_remove_roundtrip_and_gating():
    s = perturbed_state(33)
    cid = sorted(s.cx.carriers)[2]
    s2 = CarrierKnotInsert(cid, 0.6).apply(s).state
    back = CarrierKnotRemove(cid, 0.6, eps=1e-9).apply(s2)
    assert back.ok, back.reason
    assert back.deviation < 1e-12
    # the faces still carry the (now unnecessary) strip until KnotRemove simplifies them
    fid = sorted(carrier_faces(s2, cid))[0]
    f = back.state.cx.faces[fid]
    axis = "u" if bb.multiplicity(f.knots_u, 0.6) or bb.multiplicity(f.knots_u, 0.4) else "v"
    t = [k for k in bb.interior_knots(f.knots_u if axis == "u" else f.knots_v, 3)][0]
    assert KnotRemove(fid, axis, float(t), eps=1e-9).apply(back.state).ok
    # face knots required by a refined carrier cannot be removed first
    assert not KnotRemove(fid, axis, float(t), eps=10.0).apply(s2).ok
    # epsilon gating
    P = s2.values()
    P[s2.dof_map.carrier_dof[cid][1]] += np.array([0.0, 0.0, 0.05])
    s2.set_values(P)
    gated = CarrierKnotRemove(cid, 0.6, eps=1e-3).apply(s2)
    assert not gated.ok and gated.deviation > 1e-3


def crease_energy(state, w=1.0):
    F = fairness_operator(state, "bending", crease_weight=w)
    Fb = fairness_operator(state, "bending", crease_weight=0.0)
    P = state.values()
    return float(np.sum((F @ P) ** 2) - np.sum((Fb @ P) ** 2))


def test_crease_penalty_zero_after_exact_split_and_positive_for_creases():
    s = perturbed_state(34)
    fid = sorted(s.cx.faces)[0]
    out = SplitFace(fid, "u", 0.5).apply(s)
    s2 = out.state
    assert crease_energy(s) == 0.0  # no split lines in the cube layout
    assert abs(crease_energy(s2)) < 1e-20  # C2 across the fresh split line
    a, b = out.info["children"]
    P = s2.values()
    P[s2.dof_map.face_dof[a][-1, :]] += np.array([0.0, 0.0, 0.05])  # crease next to the split line
    s2.set_values(P)
    assert crease_energy(s2) > 1e-6


def c1_merge_deviation(state, a, b):
    """Smallest deviation over merges keeping 0, 1 or 2 copies of the split knot.

    eps=0 makes every attempt fail, so the outcome reports the best deviation;
    keeping the double (C1) knot is the most permissive attempt."""
    return MergeFace(a, b, eps=0.0, max_keep=2).apply(state).deviation


def optimized_split(reachable_target, crease_weight, steps=60):
    from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
    from cad_d4d.optimization.continuous import ContinuousOptimizer
    from cad_d4d.optimization.discretization import DiscretizationConfig
    s = reachable_target.ground_truth.copy()
    out = SplitFace(sorted(s.cx.faces)[0], "u", 0.5).apply(s)
    s2 = out.state
    obj = ShapeObjective(reachable_target, ObjectiveConfig(
        lambda_fair=2e-8, discretization=DiscretizationConfig(crease_weight=crease_weight)))
    ContinuousOptimizer(obj).run(s2, steps)
    a, b = out.info["children"]
    return c1_merge_deviation(s2, a, b)


def test_crease_penalty_keeps_unused_split_mergeable(reachable_target):
    """Without the penalty, optimized split children develop a crease at the split line;
    with it they stay C1 there, so a (partial, C1) MergeFace remains nearly exact."""
    free = optimized_split(reachable_target, 0.0)
    penalized = optimized_split(reachable_target, CREASE_W)
    assert penalized < 0.25 * free
