"""Milestone 6: SplitFace, MergeFace, SplitEdge/MergeEdge, LocalRefine."""
import numpy as np
import pytest

from cad_d4d.geometry.state import root_deviation, watertightness_error
from cad_d4d.rewrites.knot_insert import KnotInsert
from cad_d4d.rewrites.local_refine import LocalRefine
from cad_d4d.rewrites.merge_face import MergeFace, merge_face_candidates
from cad_d4d.rewrites.split_edge import MergeEdge, SplitEdge
from cad_d4d.rewrites.split_face import SplitFace

from helpers import perturbed_state


def assert_watertight_for_random_dofs(state, seed=0):
    s = state.copy()
    rng = np.random.default_rng(seed)
    s.set_values(s.values() + 0.3 * rng.normal(size=s.values().shape))
    assert watertightness_error(s) < 1e-12  # fp noise grows with chained refinement maps


@pytest.mark.parametrize("axis", ["u", "v"])
def test_split_face_is_exact(axis):
    s = perturbed_state(5, n_knots=1)
    fid = sorted(s.cx.faces)[2]
    out = SplitFace(fid, axis, 0.37).apply(s)
    assert out.ok, out.reason
    s2 = out.state
    assert s2.n_faces == 7
    assert root_deviation(s, s2) < 1e-13
    assert watertightness_error(s2) < 1e-13
    assert_watertight_for_random_dofs(s2)
    a, b = out.info["children"]
    pa = s2.cx.faces[a].provenance
    assert pa.op == "SplitFace" and pa.parents == (fid,) and pa.axis == axis and abs(pa.param - 0.37) < 1e-12


def test_split_creates_hanging_vertices_not_neighbor_refinement():
    s = perturbed_state(6)
    fid = sorted(s.cx.faces)[0]
    neigh = s.cx.neighbors(fid)
    s2 = SplitFace(fid, "u", 0.5).apply(s).state
    for g in neigh:  # neighbours keep their knots/DOFs; only wires changed
        assert s2.cx.faces[g].shape == s.cx.faces[g].shape
    hanging = [v for v in s2.cx.vertices.values() if not v.free]
    assert len(hanging) == 2
    # new DOFs: 2 interior carrier points + children interiors (2x 4) - parent interior (4)
    assert s2.n_control_points == s.n_control_points + 2 + 4


def test_split_edge_and_merge_edge_preserve_geometry():
    s = perturbed_state(7)
    eid = sorted(s.cx.edges)[3]
    faces = s.cx.faces_of_edge(eid)
    out = SplitEdge(eid, 0.3).apply(s)
    assert out.ok
    s2 = out.state
    assert root_deviation(s, s2) < 1e-14
    assert s2.n_control_points == s.n_control_points  # topological only
    vid = out.info["vertex"]
    for fid in faces:  # both incident wires reference the two subedges
        assert vid in s2.cx.face_vertices(fid)
    assert watertightness_error(s2) < 1e-14
    back = MergeEdge(vid).apply(s2)
    assert back.ok
    assert len(back.state.cx.edges) == len(s.cx.edges)
    assert root_deviation(s, back.state) < 1e-14


def test_merge_edge_refused_at_face_corner():
    s = perturbed_state(8)
    fid = sorted(s.cx.faces)[0]
    s2 = SplitFace(fid, "u", 0.5).apply(s).state
    hanging = [v.id for v in s2.cx.vertices.values() if not v.free]
    for vid in hanging:  # corners of the two children
        assert not MergeEdge(vid).apply(s2).ok


def test_merge_face_exact_inverse_of_split():
    s = perturbed_state(9)
    fid = sorted(s.cx.faces)[4]
    split = SplitFace(fid, "v", 0.6).apply(s)
    a, b = split.info["children"]
    cands = merge_face_candidates(split.state, 1e-6)
    assert any({m.face_a, m.face_b} == {a, b} for m in cands)
    out = MergeFace(a, b, eps=1e-9).apply(split.state)
    assert out.ok, out.reason
    m = out.state
    assert m.n_faces == s.n_faces and m.n_control_points == s.n_control_points
    assert len(m.cx.vertices) == len(s.cx.vertices) and len(m.cx.edges) == len(s.cx.edges)
    assert root_deviation(s, m) < 1e-11
    assert watertightness_error(m) < 1e-13


def test_merge_face_epsilon_gating_rejects_bad_approximation():
    s = perturbed_state(10)
    fid = sorted(s.cx.faces)[1]
    split = SplitFace(fid, "u", 0.5).apply(s)
    a, b = split.info["children"]
    s2 = split.state
    # (1) a perturbation away from the split line keeps the children C1 across it:
    #     a full merge fails, a partial merge keeping a double knot is still exact
    far = s2.copy()
    P = far.values()
    P[far.dof_map.face_dof[a][0, 0]] += np.array([0.0, 0.1, 0.1])
    far.set_values(P)
    assert not MergeFace(a, b, eps=1e-3, max_keep=0).apply(far).ok
    partial = MergeFace(a, b, eps=1e-9).apply(far)
    assert partial.ok and partial.info["kept_multiplicity"] == 2 and partial.deviation < 1e-10
    assert partial.state.n_faces == s.n_faces
    # (2) a crease at the split line cannot be represented by any merge
    P = s2.values()
    P[s2.dof_map.face_dof[a][-1, 0]] += np.array([0.0, 0.1, 0.1])  # next to the split line
    s2.set_values(P)
    out = MergeFace(a, b, eps=1e-3).apply(s2)
    assert not out.ok and out.deviation > 1e-3
    assert MergeFace(a, b, eps=10.0).apply(s2).ok


def test_local_refine_exact_and_local():
    s = perturbed_state(11)
    fid = sorted(s.cx.faces)[2]
    out = LocalRefine(fid, 0.5, 0.45, width=0.3, refine_knots=2).apply(s)
    assert out.ok, out.reason
    s2 = out.state
    assert root_deviation(s, s2) < 1e-13
    assert watertightness_error(s2) < 1e-13
    assert_watertight_for_random_dofs(s2)
    # neighbours untouched structurally
    for g in s.cx.faces:
        if g != fid:
            assert s2.cx.faces[g].shape == s.cx.faces[g].shape
    # the refined child covers only the window of the original face ...
    target = s2.cx.faces[out.info["refined_face"]]
    area_frac = (target.domain[1] - target.domain[0]) * (target.domain[3] - target.domain[2])
    assert area_frac == pytest.approx(0.09, rel=1e-9)
    # ... whereas strip refinement (KnotInsert u and v) gives new DOFs whose support spans the face.
    strip = KnotInsert(fid, "u", 0.5).apply(s).state
    strip = KnotInsert(fid, "v", 0.45).apply(strip).state
    assert local_support_fraction(s2, out.info["refined_face"]) < 0.25 * local_support_fraction(strip, fid)


def local_support_fraction(state, fid, n=60):
    """Fraction of the root-face parameter area influenced by the interior DOFs of ``fid``."""
    f = state.cx.faces[fid]
    from cad_d4d.geometry.patch import basis
    t = np.linspace(0, 1, n)
    Bu = basis(f.knots_u, 3, t)[:, 1:-1]
    Bv = basis(f.knots_v, 3, t)[:, 1:-1]
    influenced = (np.abs(Bu).sum(1)[:, None] * np.abs(Bv).sum(1)[None, :]) > 1e-12
    dom_area = (f.domain[1] - f.domain[0]) * (f.domain[3] - f.domain[2])
    return influenced.mean() * dom_area


def test_local_refine_near_boundary_skips_tiny_splits():
    s = perturbed_state(12)
    fid = sorted(s.cx.faces)[0]
    out = LocalRefine(fid, 0.05, 0.5, width=0.3).apply(s)
    assert out.ok
    assert out.state.n_faces == s.n_faces + 3  # only the u_hi split + two v splits
    assert root_deviation(s, out.state) < 1e-13


def test_local_refine_inverse_via_knot_remove_and_merges():
    """Unused LocalRefine structure can be simplified back with gated inverse rewrites."""
    from cad_d4d.rewrites.knot_remove import knot_remove_candidates
    s = perturbed_state(13)
    fid = sorted(s.cx.faces)[3]
    cur = LocalRefine(fid, 0.5, 0.5, width=0.3, refine_knots=1).apply(s).state
    for _ in range(20):
        progressed = False
        for rw in knot_remove_candidates(cur, 1e-9) + merge_face_candidates(cur, 1e-9):
            out = rw.apply(cur)
            if out.ok:
                cur, progressed = out.state, True
                break
        if not progressed:
            break
    assert cur.n_faces == s.n_faces
    assert cur.n_control_points == s.n_control_points
    assert root_deviation(s, cur) < 1e-9
    assert watertightness_error(cur) < 1e-13


def test_local_refine_with_boundary_refinement_is_exact():
    s = perturbed_state(14)
    fid = sorted(s.cx.faces)[2]
    plain = LocalRefine(fid, 0.5, 0.45, width=0.3, refine_knots=2).apply(s)
    out = LocalRefine(fid, 0.5, 0.45, width=0.3, refine_knots=2, refine_boundary=True).apply(s)
    assert out.ok, out.reason
    s2 = out.state
    assert root_deviation(s, s2) < 1e-13
    assert_watertight_for_random_dofs(s2)
    # the window's carriers now carry the child's knots -> more (boundary) DOFs than plain
    assert s2.n_control_points > plain.state.n_control_points
    target = s2.cx.faces[out.info["refined_face"]]
    from cad_d4d.geometry.topology import SIDES
    for side in SIDES:
        for use in target.sides[side]:
            c = s2.cx.carriers[s2.cx.edges[use.edge].carrier]
            assert len(c.knots) > 8  # every bounding carrier was refined


def test_face_refine_bisects_spans_exactly():
    from cad_d4d.rewrites.local_refine import FaceRefine
    s = perturbed_state(15, n_knots=1)
    fid = sorted(s.cx.faces)[0]
    out = FaceRefine(fid).apply(s)
    assert out.ok, out.reason
    s2 = out.state
    f = s2.cx.faces[fid]
    assert len(f.knots_u) == len(s.cx.faces[fid].knots_u) + 2  # 2 spans -> 2 midpoints
    assert root_deviation(s, s2) < 1e-13
    assert_watertight_for_random_dofs(s2)


def test_equidistributed_knots_follow_residual_mass():
    from cad_d4d.rewrites.local_refine import equidistributed_knots
    t = np.linspace(0, 1, 401)
    mass = np.exp(-((t - 0.8) / 0.05) ** 2)          # residual concentrated near t = 0.8
    ks = equidistributed_knots(t, mass, existing=np.array([0.5]), m=3, min_gap=0.03, floor=0.2)
    assert len(ks) >= 2 and all(0.6 < k < 0.95 for k in ks)
    assert min(np.diff([0.0, *ks, 1.0])) >= 0.03 and all(abs(k - 0.5) >= 0.03 for k in ks)
    flat = equidistributed_knots(t, np.zeros_like(t), existing=np.array([]), m=3, min_gap=0.03)
    assert np.allclose(flat, [0.25, 0.5, 0.75])      # no residual: uniform placement


def test_residual_refine_is_exact_and_refines_carriers():
    from cad_d4d.geometry.topology import SIDES
    from cad_d4d.rewrites.local_refine import ResidualRefine
    s = perturbed_state(16)
    fid = sorted(s.cx.faces)[0]
    out = ResidualRefine(fid, knots_u=[0.7, 0.85], knots_v=[0.3]).apply(s)
    assert out.ok, out.reason
    s2 = out.state
    f = s2.cx.faces[fid]
    assert len(f.knots_u) == 10 and len(f.knots_v) == 9
    assert root_deviation(s, s2) < 1e-13
    assert_watertight_for_random_dofs(s2)
    for side in SIDES:
        for use in f.sides[side]:
            assert len(s2.cx.carriers[s2.cx.edges[use.edge].carrier].knots) > 8
    assert not ResidualRefine(fid).apply(s).ok


def test_residual_refine_proposals_come_from_sampler(reachable_target, coarse_sphere):
    from cad_d4d.device import to_tensor
    from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
    from cad_d4d.optimization.proposal_sampling import ProposalConfig, ProposalSampler
    obj = ShapeObjective(reachable_target, ObjectiveConfig())
    s = coarse_sphere
    terms = obj.terms(s, to_tensor(s.values()))
    smp = ProposalSampler(ProposalConfig(kind_weights={"ResidualRefine": 1.0}, adaptive_scale=False), seed=0)
    props = smp.refinements(obj, s, terms)
    assert props and all(p.kind == "ResidualRefine" for p in props)
    assert all(p.apply(s).ok for p in props[:3])
