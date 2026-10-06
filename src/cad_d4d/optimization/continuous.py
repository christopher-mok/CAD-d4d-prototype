"""Continuous optimization within a fixed structure.

    p <- p - eta * M^{-1} grad_p L_smooth

with the preconditioner M of ``preconditioner.Preconditioner`` (default:
semi-implicit lumped mass + fairness Hessian), Armijo sufficient decrease and
geometric validity backtracking (degenerate Jacobian, orientation flip,
nonlocal self-intersection). No Adam: plain preconditioned gradient descent.

Backtracking order per trial step: Jacobian/orientation check (cheap), Armijo
test, then the self-intersection test (expensive) only for steps that would
otherwise be accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..device import to_numpy, to_tensor
from ..geometry.state import CADState
from ..geometry.validity import ValidityConfig, jacobian_check
from ..losses.objective import ShapeObjective
from .discretization import Discretization
from .preconditioner import Preconditioner


@dataclass
class ContinuousConfig:
    eta: float = 0.5           # initial / nominal step size
    eta_max: float = 4.0
    grow: float = 1.5
    armijo_c: float = 1e-4
    max_backtracks: int = 20
    max_failed_steps: int = 3  # consecutive failed steps before a phase is abandoned
    preconditioner: str = "semi_implicit"
    mass_blend: float = 0.0  # consistent_mass only: share of the lumped mass blended in
    freeze_contact: bool = True  # on self-contact, retry the step with the intersecting faces' DOFs frozen
    validity: ValidityConfig = field(default_factory=ValidityConfig)


@dataclass
class GradientInfo:
    terms: dict
    g: torch.Tensor           # (n_dof, 3)
    precond: Preconditioner   # applies M^{-1}

    @property
    def m(self) -> torch.Tensor:
        """Lumped mass (the diagonal part of M)."""
        return self.precond.m

    @property
    def direction(self) -> torch.Tensor:
        if getattr(self, "_dir", None) is None:
            self._dir = self.precond.solve(self.g)
        return self._dir

    @property
    def descent_capacity(self) -> float:
        """D = g^T M^{-1} g (first-order loss decrease per unit step size)."""
        return float((self.g * self.direction).sum())


class ContinuousOptimizer:
    def __init__(self, objective: ShapeObjective, cfg: ContinuousConfig | None = None):
        self.obj = objective
        self.cfg = cfg or ContinuousConfig()

    # -- gradients -------------------------------------------------------
    def gradient(self, state: CADState, P: torch.Tensor | None = None, disc: Discretization | None = None) -> GradientInfo:
        disc = disc or self.obj.disc(state)
        P = to_tensor(state.values() if P is None else P).detach().clone().requires_grad_(True)
        terms = self.obj.terms(state, P, disc)
        (g,) = torch.autograd.grad(terms["smooth"], P)
        pc = Preconditioner(disc, terms["w"], self.cfg.preconditioner,
                            lambda_fair=self.obj.cfg.lambda_fair, tau=self.cfg.eta, mass_blend=self.cfg.mass_blend)
        terms = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in terms.items()}
        return GradientInfo(terms, g, pc)

    # -- validity ----------------------------------------------------------
    def is_valid(self, disc: Discretization, P: torch.Tensor, ref_normals: torch.Tensor | None,
                 local: bool = True, nonlocal_: bool = True):
        """Validity of the configuration P: ``local`` = Jacobian/orientation, ``nonlocal_`` = self-intersection."""
        vc = self.cfg.validity
        with torch.no_grad():
            X, Xu, Xv = disc.check.evaluate(to_tensor(P))
            ok, info = True, {}
            if local:
                ok, info = jacobian_check(Xu, Xv, disc.check.sample_face_t, ref_normals, vc.eps_jacobian_rel,
                                          vc.check_orientation)
            if ok and nonlocal_ and vc.check_self_intersection:
                n_hit, hits = disc.count_self_intersections(X, return_hits=True)
                info["n_self_intersections"] = n_hit
                if hits:
                    tf = disc.check.tri_face_t[torch.as_tensor(hits, device=X.device).ravel()]
                    info["hit_faces"] = {disc.check.face_order[k] for k in tf.unique().tolist()}
                ok = n_hit == 0
        return ok, info

    def self_intersections(self, disc: Discretization, P: torch.Tensor) -> int:
        """Number of nonlocal self-intersections of the configuration P (0 if the check is disabled)."""
        if not self.cfg.validity.check_self_intersection:
            return 0
        with torch.no_grad():
            return disc.count_self_intersections(disc.check.evaluate(to_tensor(P))[0])

    def reference_normals(self, disc: Discretization, P: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, Xu, Xv = disc.check.evaluate(to_tensor(P))
            return torch.linalg.cross(Xu, Xv, dim=1)

    # -- main loop ---------------------------------------------------------
    def step(self, state: CADState, P: torch.Tensor, eta: float, disc: Discretization):
        """One preconditioned step with backtracking. Returns (P_new or None, log).

        Active set for self-contact: if backtracking was forced by self-intersections, a
        second line search runs with the DOFs of the intersecting faces frozen (the step
        re-solved on the free DOFs), and the better of the two accepted steps is taken. A
        near-fold in one face then no longer shrinks the step of the whole model.
        """
        cfg = self.cfg
        gi = self.gradient(state, P, disc)
        d = gi.direction
        L0 = float(gi.terms["smooth"])
        ref = self.reference_normals(disc, P.detach())
        log = {"loss": L0, "sdf": float(gi.terms["sdf"]), "coverage": float(gi.terms["coverage"]),
               "fair": float(gi.terms["fair"]), "D": float((gi.g * d).sum()), "backtracks": 0, "invalid": 0}
        ctx = {"n_cur": None}  # self-intersections of the current configuration (computed only if needed)
        best = self._line_search(state, P, d, log["D"], L0, eta, disc, ref, log, ctx)
        if ctx.get("hit_faces") and cfg.freeze_contact:
            frozen = torch.zeros(len(d), dtype=torch.bool, device=d.device)
            for fid in ctx["hit_faces"]:
                frozen[torch.as_tensor(state.dof_map.face_rows(fid).indices, device=d.device, dtype=torch.long)] = True
            d_free = gi.precond.solve_restricted(gi.g, ~frozen)
            gd_free = float((gi.g * d_free).sum())
            if gd_free > 0:
                sub = {"backtracks": 0, "invalid": 0}
                alt = self._line_search(state, P, d_free, gd_free, L0, eta, disc, ref, sub, {"n_cur": ctx["n_cur"]})
                log["invalid"] += sub["invalid"]
                if alt is not None and (best is None or alt[2] < best[2]):
                    best = alt
                    log["frozen_faces"] = sorted(ctx["hit_faces"])
        if best is None:
            log.update(eta=0.0, new_loss=L0, backtracks=cfg.max_backtracks)
            return None, log
        P_new, t, L1, k = best
        log.update(eta=t, new_loss=L1, backtracks=k)
        return P_new, log

    def _line_search(self, state, P, d, gd, L0, t, disc, ref, log, ctx):
        """Armijo backtracking along -d with validity checks. Returns (P, t, L, backtracks) or None.

        Records in ``ctx`` the faces of the first self-intersection that forced a backtrack."""
        cfg = self.cfg
        for k in range(cfg.max_backtracks):
            P_try = P.detach() - t * d
            ok, _ = self.is_valid(disc, P_try, ref, local=True, nonlocal_=False)
            if ok:
                with torch.no_grad():
                    L1 = float(self.obj.terms(state, P_try, disc)["smooth"])
                if L1 <= L0 - cfg.armijo_c * t * gd:
                    ok, info = self.is_valid(disc, P_try, ref, local=False, nonlocal_=True)
                    if not ok:
                        # Monotone rule: a step may not *add* self-intersections. A state that already
                        # contains one (e.g. a fold revealed when a rewrite re-sampled the check
                        # tessellation) must not freeze the optimizer: it may move, and unfold.
                        if ctx["n_cur"] is None:
                            ctx["n_cur"] = self.self_intersections(disc, P.detach())
                        ok = info.get("n_self_intersections", 0) <= ctx["n_cur"]
                        if not ok and "hit_faces" not in ctx:
                            ctx["hit_faces"] = info.get("hit_faces", set())
                    if ok:
                        return P_try, t, L1, k
                    log["invalid"] += 1
            else:
                log["invalid"] += 1
            t *= 0.5
        return None

    def run(self, state: CADState, n_steps: int, eta: float | None = None, callback=None) -> list[dict]:
        """Optimize ``state``'s values in place for ``n_steps``; ages birth records."""
        cfg = self.cfg
        disc = self.obj.disc(state)
        P = to_tensor(state.values())
        eta = state.meta.get("eta", cfg.eta) if eta is None else eta
        logs = []
        failures = 0
        for it in range(n_steps):
            P_new, log = self.step(state, P, eta, disc)
            for r in state.birth_records:
                r.age += 1
            logs.append(log)
            if callback:
                callback(it, log, P if P_new is None else P_new)
            if P_new is None:
                failures += 1
                if failures >= cfg.max_failed_steps:
                    eta = cfg.eta  # stalled: restart the next phase from the nominal step
                    break
                eta *= 0.5  # the next step backtracks from here again (a tiny restart would cost ~30 steps to regrow)
                continue
            failures = 0
            P = P_new
            eta = min(cfg.eta_max, log["eta"] * cfg.grow)
        state.set_values(to_numpy(P))
        state.meta["eta"] = eta
        return logs
