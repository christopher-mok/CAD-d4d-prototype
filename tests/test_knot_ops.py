"""Milestone 5: exact KnotInsert and epsilon-gated KnotRemove."""
import numpy as np

from cad_d4d.geometry.state import watertightness_error
from cad_d4d.rewrites.knot_insert import KnotInsert
from cad_d4d.rewrites.knot_remove import KnotRemove, knot_remove_candidates

from helpers import perturbed_state, same_face_deviation


def test_knot_insert_is_exact_and_adds_dofs():
    s = perturbed_state()
    fid = sorted(s.cx.faces)[2]
    out = KnotInsert(fid, "u", 0.37).apply(s)
    assert out.ok
    s2 = out.state
    assert same_face_deviation(s, s2, s.cx.faces) < 1e-14
    assert watertightness_error(s2) < 1e-14
    nu, nv = s.cx.faces[fid].shape
    assert s2.n_control_points == s.n_control_points + (nv - 2)
    # input state untouched
    assert s.cx.faces[fid].shape == (nu, nv)


def test_repeated_and_multiple_insertions_exact():
    s = perturbed_state(1)
    fid = sorted(s.cx.faces)[0]
    cur = s
    for axis, t in [("u", 0.3), ("v", 0.6), ("u", 0.3), ("v", 0.15), ("u", 0.3)]:
        out = KnotInsert(fid, axis, t).apply(cur)
        assert out.ok, out.reason
        cur = out.state
    assert same_face_deviation(s, cur, s.cx.faces) < 1e-13
    assert watertightness_error(cur) < 1e-13
    # multiplicity cap (degree 3)
    assert not KnotInsert(fid, "u", 0.3).apply(cur).ok


def test_knot_insert_then_remove_roundtrip_is_exact():
    s = perturbed_state(2)
    fid = sorted(s.cx.faces)[3]
    s2 = KnotInsert(fid, "v", 0.42).apply(s).state
    out = KnotRemove(fid, "v", 0.42, eps=1e-9).apply(s2)
    assert out.ok, out.reason
    assert out.deviation < 1e-12
    assert out.state.n_control_points == s.n_control_points
    assert same_face_deviation(s, out.state, s.cx.faces) < 1e-12


def test_knot_remove_rejects_when_refined_dofs_were_used():
    s = perturbed_state(3)
    fid = sorted(s.cx.faces)[1]
    s2 = KnotInsert(fid, "u", 0.5).apply(s).state
    P = s2.values()
    k = s2.dof_map.face_dof[fid][1, 0]  # a new interior DOF next to the inserted knot
    P[k] += np.array([0.0, 0.0, 0.08])
    s2.set_values(P)
    out = KnotRemove(fid, "u", 0.5, eps=1e-3).apply(s2)
    assert not out.ok and "deviation" in out.reason
    assert out.deviation > 1e-3
    # a lenient epsilon accepts (and refits) -- gating is purely geometric
    assert KnotRemove(fid, "u", 0.5, eps=1.0).apply(s2).ok


def test_knot_remove_refuses_knots_required_by_carriers():
    s = perturbed_state(4, n_knots=1)  # carriers and faces share the knot 0.5
    fid = sorted(s.cx.faces)[0]
    out = KnotRemove(fid, "u", 0.5, eps=10.0).apply(s)
    assert not out.ok and "boundary" in out.reason
    assert len(knot_remove_candidates(s, 1e-3)) == 12
