"""CSG representation: SDF signs, union/difference, and topology measurement on known solids."""
import numpy as np
import pytest
import torch

from cad_d4d.csg import MATERIAL, VOID, Box, Capsule, Grid, Sphere, measure, measure_multi, summary
from cad_d4d.csg.measure import cubical_euler, voxel_topology
from cad_d4d.csg.mesh import marching_tetrahedra, mesh_topology, node_lattice
from cad_d4d.csg.primitives import axis_angle, rotation_matrix
from cad_d4d.csg.targets import solid_of


def T(*p):
    return torch.tensor(p, dtype=torch.float64).reshape(-1, 3)


def test_primitive_sdf_signs_and_distances():
    s = Sphere((0, 0, 0), 0.5)
    assert float(s.sdf(T(0, 0, 0))) == pytest.approx(-0.5)
    assert float(s.sdf(T(1, 0, 0))) == pytest.approx(0.5)
    b = Box((0, 0, 0), (0.5, 0.3, 0.2), (0, 0, 0), 0.0)
    assert float(b.sdf(T(0.0, 0.0, 0.0))) == pytest.approx(-0.2)
    assert float(b.sdf(T(0.8, 0.0, 0.0))) == pytest.approx(0.3)
    assert float(b.sdf(T(0.8, 0.7, 0.0))) == pytest.approx(0.5)      # corner distance (0.3, 0.4)
    c = Capsule((0, 0, -1), (0, 0, 1), 0.2)
    assert float(c.sdf(T(0.5, 0, 0.3))) == pytest.approx(0.3)
    assert float(c.sdf(T(0, 0, 1.5))) == pytest.approx(0.3)
    # rotated box: rotating the query by R^T gives the same distance
    w = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    rb = Box((0.1, 0, 0), (0.5, 0.3, 0.2), w, 0.05)
    R = rotation_matrix(w)
    q = torch.tensor([0.4, 0.1, 0.05], dtype=torch.float64)
    ref = Box((0, 0, 0), (0.5, 0.3, 0.2), (0, 0, 0), 0.05)
    assert float(rb.sdf((torch.tensor([0.1, 0, 0], dtype=torch.float64) + R @ q)[None])) == \
        pytest.approx(float(ref.sdf(q[None])), abs=1e-12)
    assert torch.allclose(rotation_matrix(axis_angle(R)), R, atol=1e-10)


def test_union_and_difference_conventions():
    a = (MATERIAL, "body", Sphere((-0.3, 0, 0), 0.4))
    b = (MATERIAL, "body", Sphere((0.3, 0, 0), 0.4))
    v = (VOID, "cavity", Sphere((0.3, 0, 0), 0.2))
    X = T(-0.3, 0, 0, 0.3, 0, 0, 0.55, 0, 0, 2, 0, 0)
    u = solid_of(a, b).sdf(X)
    assert torch.allclose(u, torch.minimum(a[2].sdf(X), b[2].sdf(X)))       # union = min
    d = solid_of(a, b, v).sdf(X)
    assert torch.allclose(d, torch.maximum(u, -v[2].sdf(X)))                # difference = max(d, -v)
    hard = solid_of(a, b, v).hard(X)
    assert hard.tolist() == [True, False, True, False]                     # void removes (0.3,0,0) only
    occ = solid_of(a, b, v).occupancy(X, eps=0.03)
    assert float(occ[0]) > 0.99 and float(occ[1]) < 0.01 and float(occ[3]) < 1e-6


def test_copy_is_independent_and_canonicalize_drops_dead_features():
    s = solid_of((MATERIAL, "body", Sphere((0, 0, 0), 0.5)), (VOID, "cavity", Sphere((2.0, 0, 0), 0.1)),
                 (MATERIAL, "body", Sphere((0.05, 0, 0), 0.1)))
    c = s.copy()
    c.features[0].prim.params["r"] += 0.1
    assert float(s.features[0].prim.params["r"]) == 0.5
    removed = s.canonicalize(Grid(32))
    assert sorted(removed) == [1, 2] and len(s.features) == 1   # void outside material, swallowed body


SOLIDS = {
    "ball": (lambda X: X.norm(dim=1) - 0.7, dict(components=1, cavities=0, tunnels=0, genus=0)),
    "hollow": (lambda X: torch.maximum(X.norm(dim=1) - 0.8, 0.4 - X.norm(dim=1)),
               dict(components=1, cavities=1, tunnels=0, genus=0)),
    "torus": (lambda X: torch.sqrt((torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) - 0.6) ** 2 + X[:, 2] ** 2) - 0.25,
              dict(components=1, cavities=0, tunnels=1, genus=1)),
    "two_balls": (lambda X: torch.minimum((X - torch.tensor([0.5, 0, 0], dtype=torch.float64)).norm(dim=1),
                                          (X + torch.tensor([0.5, 0, 0], dtype=torch.float64)).norm(dim=1)) - 0.3,
                  dict(components=2, cavities=0, tunnels=0, genus=0)),
    "double_torus": (lambda X: torch.minimum(
        torch.sqrt((torch.sqrt((X[:, 0] - 0.45) ** 2 + X[:, 1] ** 2) - 0.4) ** 2 + X[:, 2] ** 2),
        torch.sqrt((torch.sqrt((X[:, 0] + 0.45) ** 2 + X[:, 1] ** 2) - 0.4) ** 2 + X[:, 2] ** 2)) - 0.15,
        dict(components=1, cavities=0, tunnels=2, genus=2)),
}


@pytest.mark.parametrize("name", list(SOLIDS))
def test_topology_of_known_solids(name):
    fn, want = SOLIDS[name]
    m = measure_multi(fn, ns=(48, 72))
    assert m["stable"], m
    for k, v in want.items():
        assert m[k] == v, (k, m)


def test_cavity_vs_exterior_connected_void():
    occ = np.zeros((12, 12, 12), bool)
    occ[2:10, 2:10, 2:10] = True
    occ[5:7, 5:7, 5:7] = False                       # enclosed: a cavity
    t = voxel_topology(occ, min_voxels=1)
    assert (t["components"], t["cavities"], t["tunnels"]) == (1, 1, 0)
    occ[5:7, 5:7, 5:10] = False                      # the same void, opened to the outside: a dent
    t = voxel_topology(occ, min_voxels=1)
    assert (t["components"], t["cavities"], t["tunnels"]) == (1, 0, 0)
    occ[5:7, 5:7, 2:10] = False                      # through both faces: a tunnel
    t = voxel_topology(occ, min_voxels=1)
    assert (t["components"], t["cavities"], t["tunnels"]) == (1, 0, 1)


def test_voxel_conventions_and_tiny_components():
    assert cubical_euler(np.ones((1, 1, 1), bool)) == 1
    occ = np.zeros((6, 6, 6), bool)
    occ[1, 1, 1] = occ[2, 2, 2] = True              # corner contact: one 26-connected component
    assert voxel_topology(occ, min_voxels=1)["components"] == 1
    big = np.zeros((16, 16, 16), bool)
    big[2:8, 2:8, 2:8] = True
    big[12, 12, 12] = True                           # single-voxel debris
    t = voxel_topology(big, min_voxels=8)
    assert t["components"] == 1 and t["tiny_components"] == 1


def test_mesh_is_welded_and_invalid_meshes_report_no_genus():
    P = node_lattice(-1.2, 1.2, 40)
    f = np.linalg.norm(P, axis=1) - 0.7
    V, F, touches = marching_tetrahedra(f, P, 41)
    mt = mesh_topology(F, len(V))
    assert mt["closed"] and mt["manifold"] and mt["genus"] == [0] and not touches
    # welded: no two vertices share a position, every edge has exactly two faces
    assert len(np.unique(np.round(V, 9), axis=0)) == len(V)
    # triangle soup (unwelded copy) is not closed -> no genus
    soup_F = np.arange(3 * len(F)).reshape(-1, 3)
    soup = mesh_topology(soup_F, 3 * len(F))
    assert not soup["closed"] and soup["genus"] is None
    # two tetrahedra sharing one vertex: closed but not manifold -> no genus
    tet = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    bow = np.vstack([tet, np.where(tet == 0, 0, tet + 3)])
    bt = mesh_topology(bow, 7)
    assert bt["closed"] and not bt["manifold"] and bt["genus"] is None


def test_measure_flags_surfaces_open_at_the_grid_boundary():
    m = measure(lambda X: X.norm(dim=1) - 1.5, n=24)
    assert m["mesh"]["touches_boundary"] and m["genus"] is None and not m["consistent"]
    s = summary(measure(solid_of((MATERIAL, "body", Sphere((0, 0, 0), 0.6))).sdf, n=40))
    assert s["consistent"] and s["genus"] == 0
