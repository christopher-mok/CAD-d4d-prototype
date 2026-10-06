"""Differentiable surface proxy: fixed parameter grids per face.

For a fixed structure every sample position (and tangent) is linear in the
canonical DOFs, so the whole proxy reduces to three cached sparse matrices::

    X  = G  @ P,   X_u = G_u @ P,   X_v = G_v @ P

The triangle connectivity of the grid is fixed between rewrites. The proxy is
an evaluator only -- it is never the canonical geometry.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from ..device import DTYPE, get_device
from .patch import basis
from .state import CADState


@dataclass
class SamplingConfig:
    min_res: int = 9
    per_span: int = 4
    max_res: int = 33


def _torch_sparse(M: sp.spmatrix) -> torch.Tensor:
    M = M.tocoo()
    dev = get_device()
    idx = torch.as_tensor(np.vstack([M.row, M.col]), dtype=torch.long, device=dev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.sparse_coo_tensor(idx, torch.as_tensor(M.data, dtype=DTYPE, device=dev), M.shape,
                                       check_invariants=False).coalesce()


def face_resolution(knots: np.ndarray, cfg: SamplingConfig) -> int:
    spans = len(np.unique(np.round(knots, 12))) - 1
    return int(np.clip(cfg.per_span * spans + 1, cfg.min_res, cfg.max_res))


class SurfaceSampler:
    """Grid samples of every face of a state, as sparse linear maps of the DOFs."""

    def __init__(self, state: CADState, cfg: SamplingConfig | None = None):
        cfg = cfg or SamplingConfig()
        dm = state.dof_map
        cx = state.cx
        self.face_order = list(dm.face_order)
        G_blocks, Gu_blocks, Gv_blocks = [], [], []
        tris, tri_face, tri_cell, tri_bnd = [], [], [], []
        sample_face, sample_uv, quad_w, sample_bnd = [], [], [], []
        self.face_slices: dict[int, tuple[int, int, int]] = {}
        off = 0
        for k, fid in enumerate(self.face_order):
            f = cx.faces[fid]
            nu_s, nv_s = face_resolution(f.knots_u, cfg), face_resolution(f.knots_v, cfg)
            us, vs = np.linspace(0, 1, nu_s), np.linspace(0, 1, nv_s)
            Bu, dBu = basis(f.knots_u, f.degree_u, us), basis(f.knots_u, f.degree_u, us, 1)
            Bv, dBv = basis(f.knots_v, f.degree_v, vs), basis(f.knots_v, f.degree_v, vs, 1)
            R = dm.face_rows(fid)
            G_blocks.append(sp.csr_matrix(np.kron(Bu, Bv)) @ R)
            Gu_blocks.append(sp.csr_matrix(np.kron(dBu, Bv)) @ R)
            Gv_blocks.append(sp.csr_matrix(np.kron(Bu, dBv)) @ R)
            self.face_slices[fid] = (off, nu_s, nv_s)
            U, V = np.meshgrid(us, vs, indexing="ij")
            sample_uv.append(np.stack([U.ravel(), V.ravel()], 1))
            sample_face.append(np.full(nu_s * nv_s, k))
            sample_bnd.append(((U == 0) | (U == 1) | (V == 0) | (V == 1)).ravel())
            wu = np.full(nu_s, 1.0 / (nu_s - 1)); wu[[0, -1]] *= 0.5
            wv = np.full(nv_s, 1.0 / (nv_s - 1)); wv[[0, -1]] *= 0.5
            quad_w.append(np.outer(wu, wv).ravel())
            I, J = np.meshgrid(np.arange(nu_s - 1), np.arange(nv_s - 1), indexing="ij")
            I, J = I.ravel(), J.ravel()
            a = off + I * nv_s + J
            b = off + (I + 1) * nv_s + J
            c = off + (I + 1) * nv_s + J + 1
            d = off + I * nv_s + J + 1
            tris.append(np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]))
            cell = np.stack([I, J], 1)
            tri_cell.append(np.concatenate([cell, cell]))
            bnd = (I == 0) | (J == 0) | (I == nu_s - 2) | (J == nv_s - 2)
            tri_bnd.append(np.concatenate([bnd, bnd]))
            tri_face.append(np.full(2 * len(I), k))
            off += nu_s * nv_s
        self.n_samples = off
        self.G_np = sp.vstack(G_blocks).tocsr()
        self.Gu_np = sp.vstack(Gu_blocks).tocsr()
        self.Gv_np = sp.vstack(Gv_blocks).tocsr()
        self.G = _torch_sparse(self.G_np)
        self.Gu = _torch_sparse(self.Gu_np)
        self.Gv = _torch_sparse(self.Gv_np)
        self.tri = np.concatenate(tris)
        self.tri_face = np.concatenate(tri_face)
        self.tri_cell = np.concatenate(tri_cell)
        self.tri_boundary = np.concatenate(tri_bnd)
        self.sample_face = np.concatenate(sample_face)
        self.sample_uv = np.concatenate(sample_uv)
        self.sample_boundary = np.concatenate(sample_bnd)
        dev = get_device()
        self.quad_w = torch.as_tensor(np.concatenate(quad_w), dtype=DTYPE, device=dev)
        # device copies of the fixed connectivity
        self.tri_t = torch.as_tensor(self.tri, dtype=torch.long, device=dev)
        self.tri_face_t = torch.as_tensor(self.tri_face, dtype=torch.long, device=dev)
        self.tri_cell_t = torch.as_tensor(self.tri_cell, dtype=torch.long, device=dev)
        self.sample_face_t = torch.as_tensor(self.sample_face, dtype=torch.long, device=dev)
        self.sample_boundary_t = torch.as_tensor(self.sample_boundary, device=dev)

    def evaluate(self, P: torch.Tensor):
        """Returns (X, X_u, X_v), each (n_samples, 3), differentiable in P."""
        return torch.sparse.mm(self.G, P), torch.sparse.mm(self.Gu, P), torch.sparse.mm(self.Gv, P)

    def evaluate_np(self, P: np.ndarray):
        return self.G_np @ P, self.Gu_np @ P, self.Gv_np @ P

    def face_grid(self, fid: int, X):
        off, nu, nv = self.face_slices[fid]
        return X[off: off + nu * nv].reshape(nu, nv, -1)


def export_obj(path, X: np.ndarray, tri: np.ndarray) -> None:
    with open(path, "w") as fh:
        for p in X:
            fh.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        for t in tri:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
