"""Milestone 3: multiple patches with guaranteed shared boundaries."""
import numpy as np
import pytest
import torch

from cad_d4d.device import to_tensor
from cad_d4d.geometry.builders import build_cube_complex, ellipsoid_map
from cad_d4d.geometry.state import watertightness_error
from cad_d4d.geometry.tessellation import SurfaceSampler
from cad_d4d.geometry.topology import TopologyError


def signed_volume(X, tri):
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


@pytest.mark.parametrize("n_knots", [0, 2])
def test_cube_sphere_invariants_and_watertightness(n_knots):
    s = build_cube_complex(n_interior_knots=n_knots)
    s.cx.check_invariants()
    assert s.n_faces == 6
    for fid in s.cx.faces:
        s.dof_map.face_rows(fid, check=True)  # corner/shared rows consistent
    assert watertightness_error(s) < 1e-14


def test_watertight_for_arbitrary_parameters():
    """Watertightness holds by construction for *any* DOF vector."""
    s = build_cube_complex(ellipsoid_map(), n_interior_knots=1)
    rng = np.random.default_rng(0)
    for scale in (0.01, 1.0, 10.0):
        s.set_values(s.values() + scale * rng.normal(size=s.values().shape))
        assert watertightness_error(s) < 1e-13 * max(1.0, scale)


def test_dof_counts():
    s = build_cube_complex()
    # 8 vertices + 12 carriers * 2 interior + 6 faces * 4 interior
    assert s.n_control_points == 8 + 24 + 24
    assert s.n_raw_control_points == 96


def test_sampler_matches_direct_evaluation_and_orientation():
    s = build_cube_complex(n_interior_knots=1)
    sm = SurfaceSampler(s)
    X, Xu, Xv = sm.evaluate_np(s.values())
    for k, fid in enumerate(sm.face_order):
        mask = sm.sample_face == k
        direct = s.evaluate(fid, sm.sample_uv[mask])
        assert np.allclose(direct, X[mask], atol=1e-14)
    n = np.cross(Xu, Xv)
    assert np.all(np.einsum("ij,ij->i", n, X) > 0), "normals must point outward"
    assert abs(signed_volume(X, sm.tri) - 4 / 3 * np.pi) < 0.1


def test_shared_dofs_accumulate_gradients_from_both_faces():
    s = build_cube_complex()
    sm = SurfaceSampler(s)
    P = to_tensor(s.values()).requires_grad_(True)
    X, _, _ = sm.evaluate(P)
    dm = s.dof_map
    cid = next(iter(s.cx.carriers))
    k = dm.carrier_dof[cid][0]
    grads = []
    for face_k in range(len(sm.face_order)):
        P.grad = None
        X, _, _ = sm.evaluate(P)
        X[torch.as_tensor(sm.sample_face == face_k, device=X.device), 0].sum().backward()
        grads.append(P.grad[k, 0].item())
    nonzero = [g for g in grads if abs(g) > 1e-12]
    assert len(nonzero) == 2  # exactly the two incident faces see the carrier DOF
    P.grad = None
    X, _, _ = sm.evaluate(P)
    X[:, 0].sum().backward()
    assert abs(P.grad[k, 0].item() - sum(grads)) < 1e-12


def test_invariant_checker_detects_orientation_error():
    s = build_cube_complex()
    f = next(iter(s.cx.faces.values()))
    f.sides["v0"][0].reversed = not f.sides["v0"][0].reversed
    with pytest.raises(TopologyError):
        s.cx.check_invariants()
