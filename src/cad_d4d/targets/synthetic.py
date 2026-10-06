"""Synthetic targets generated from the same grammar.

    valid coarse complex -> known rewrites -> validity-preserving perturbation
    -> freeze geometry (SDF grid, coverage samples) -> discard the program

The frozen ``Target`` only exposes geometry. The generating state is kept as
``ground_truth`` for diagnostics/visualization; optimizers must not read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from ..device import to_tensor
from ..geometry.state import CADState
from ..geometry.tessellation import SamplingConfig, SurfaceSampler
from ..geometry.validity import ValidityConfig, jacobian_check, self_intersections
from ..losses.target_sdf import SDFGrid


@dataclass
class TargetConfig:
    sdf_res: int = 64
    sdf_pad: float = 0.25
    mesh_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig(min_res=65, per_span=16, max_res=129))
    n_coverage: int = 3000
    seed: int = 0


class Target:
    def __init__(self, sdf: SDFGrid, points: np.ndarray, normals: np.ndarray, V: np.ndarray, T: np.ndarray,
                 ground_truth: CADState | None = None):
        self.sdf = sdf
        self.points = to_tensor(points)
        self.normals = to_tensor(normals)
        self.V, self.T = V, T
        self.ground_truth = ground_truth
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        self.volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    @staticmethod
    def from_state(state: CADState, cfg: TargetConfig | None = None) -> "Target":
        cfg = cfg or TargetConfig()
        sm = SurfaceSampler(state, cfg.mesh_sampling)
        X, Xu, Xv = sm.evaluate_np(state.values())
        n = np.cross(Xu, Xv)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        return Target.from_mesh(X, sm.tri, n, cfg, ground_truth=state.copy())

    @staticmethod
    def from_mesh(V: np.ndarray, T: np.ndarray, vertex_normals: np.ndarray, cfg: TargetConfig | None = None,
                  ground_truth: CADState | None = None) -> "Target":
        """Freeze any closed, outward-oriented triangle mesh as a target.

        Used for targets outside the grammar (analytic shapes); ``ground_truth`` is
        then None.
        """
        cfg = cfg or TargetConfig()
        V, T = np.asarray(V, float), np.asarray(T, np.int64)
        sdf = SDFGrid.from_mesh(V, T, vertex_normals, res=cfg.sdf_res, pad=cfg.sdf_pad)
        rng = np.random.default_rng(cfg.seed)
        A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        t = rng.choice(len(area), size=cfg.n_coverage, p=area / area.sum())
        r1, r2 = rng.random(cfg.n_coverage), rng.random(cfg.n_coverage)
        s1 = np.sqrt(r1)
        bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
        pts = bary[:, :1] * A[t] + bary[:, 1:2] * B[t] + bary[:, 2:] * C[t]
        tri_n = np.cross(B - A, C - A)[t]
        tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-300)
        return Target(sdf, pts, tri_n, V, T, ground_truth=ground_truth)


def is_valid_state(state: CADState, cfg: ValidityConfig | None = None) -> bool:
    """Standalone validity check (no reference normals: degenerate + self-intersection)."""
    from ..optimization.discretization import get_discretization
    cfg = cfg or ValidityConfig()
    disc = get_discretization(state)
    with torch.no_grad():
        X, Xu, Xv = disc.check.evaluate(to_tensor(state.values()))
    ok, _ = jacobian_check(Xu, Xv, disc.check.sample_face_t, None, cfg.eps_jacobian_rel, False)
    if not ok:
        return False
    if cfg.check_self_intersection:
        return disc.count_self_intersections(X) == 0
    return True


def make_synthetic_target(base: CADState, program: Sequence = (),
                          perturb: Callable[[CADState, np.random.Generator], None] | None = None,
                          cfg: TargetConfig | None = None, max_tries: int = 8) -> Target:
    """Apply ``program`` (rewrites), then ``perturb`` (shrinking it until valid), and freeze."""
    cfg = cfg or TargetConfig()
    state = base.copy()
    for rw in program:
        out = rw.apply(state)
        if out.state is None:
            raise RuntimeError(f"target program step failed: {rw} ({out.reason})")
        state = out.state
    if perturb is not None:
        P0 = state.values()
        for k in range(max_tries):
            rng = np.random.default_rng(cfg.seed)
            trial = state.copy()
            trial.set_values(P0)
            perturb(trial, rng)
            if k:
                trial.set_values(P0 + (trial.values() - P0) * 0.5**k)
            if is_valid_state(trial):
                state = trial
                break
        else:
            raise RuntimeError("could not find a valid perturbation")
    return Target.from_state(state, cfg)


def perturb_dofs(sigma: float, kinds=("v", "c", "f")) -> Callable:
    """Gaussian perturbation of canonical DOFs of the given kinds."""
    def f(state: CADState, rng: np.random.Generator):
        P = state.values()
        for k, owner in enumerate(state.dof_map.owners):
            if owner[0] in kinds:
                P[k] += sigma * rng.normal(size=3)
        state.set_values(P)
    return f


def scale_dofs(scales=(1.0, 1.0, 1.0)) -> Callable:
    """Anisotropic scaling of all DOFs (stays inside the representable space)."""
    def f(state: CADState, rng: np.random.Generator):
        state.set_values(state.values() * np.asarray(scales))
    return f


def compose(*fns) -> Callable:
    def f(state, rng):
        for g in fns:
            g(state, rng)
    return f


def bump_refined_faces(height: float, min_interior: int = 4, face_filter: Callable | None = None) -> Callable:
    """Displace the inner control points of refined faces along the face normal.

    For each face with at least ``min_interior`` interior control points per
    direction, the interior DOFs that are not adjacent to the boundary rows are
    moved by ``height`` along the face's center normal. Boundary rows and their
    neighbours stay fixed, so the bump joins the rest of the face smoothly (C1).
    """
    def f(state: CADState, rng: np.random.Generator):
        P = state.values()
        dm = state.dof_map
        for fid, face in state.cx.faces.items():
            ni, nj = face.interior.shape[:2]
            if min(ni, nj) < min_interior or (face_filter and not face_filter(face)):
                continue
            Su = state.evaluate(fid, np.array([[0.5, 0.5]]), (1, 0))[0]
            Sv = state.evaluate(fid, np.array([[0.5, 0.5]]), (0, 1))[0]
            n = np.cross(Su, Sv)
            n /= np.linalg.norm(n)
            idx = dm.face_dof[fid][1:-1, 1:-1].ravel()
            P[idx] += height * n
        state.set_values(P)
    return f


def localized_bump_target(base: CADState, face_index: int = 0, uv=(0.5, 0.5), width: float = 0.4,
                          height: float = 0.25, refine_knots: int = 2, extra: Callable | None = None,
                          cfg: TargetConfig | None = None) -> Target:
    """Reachable target with a localized feature: LocalRefine + bump of the refined child."""
    from ..rewrites.local_refine import LocalRefine
    fid = sorted(base.cx.faces)[face_index]
    program = [LocalRefine(fid, uv[0], uv[1], width=width, refine_knots=refine_knots)]
    perturb = bump_refined_faces(height)
    if extra is not None:
        perturb = compose(extra, perturb)
    return make_synthetic_target(base, program, perturb, cfg)
