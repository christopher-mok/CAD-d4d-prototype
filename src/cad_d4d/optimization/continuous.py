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
                            lambda_fair=self.obj.cfg.lambda_fair, tau=self.cfg.eta)
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
                n_hit = disc.count_self_intersections(X)
                info["n_self_intersections"] = n_hit
                ok = n_hit == 0
        return ok, info

    def reference_normals(self, disc: Discretization, P: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, Xu, Xv = disc.check.evaluate(to_tensor(P))
            return torch.linalg.cross(Xu, Xv, dim=1)

    # -- main loop ---------------------------------------------------------
    def step(self, state: CADState, P: torch.Tensor, eta: float, disc: Discretization):
        """One preconditioned step with backtracking. Returns (P_new or None, log)."""
        cfg = self.cfg
        gi = self.gradient(state, P, disc)
        d = gi.direction
        gd = float((gi.g * d).sum())
        L0 = float(gi.terms["smooth"])
        ref = self.reference_normals(disc, P.detach())
        t = eta
        log = {"loss": L0, "sdf": float(gi.terms["sdf"]), "coverage": float(gi.terms["coverage"]),
               "fair": float(gi.terms["fair"]), "D": gd, "backtracks": 0, "invalid": 0}
        for k in range(cfg.max_backtracks):
            P_try = P.detach() - t * d
            ok, _ = self.is_valid(disc, P_try, ref, local=True, nonlocal_=False)
            if ok:
                with torch.no_grad():
                    L1 = float(self.obj.terms(state, P_try, disc)["smooth"])
                if L1 <= L0 - cfg.armijo_c * t * gd:
                    ok, _ = self.is_valid(disc, P_try, ref, local=False, nonlocal_=True)
                    if ok:
                        log.update(eta=t, new_loss=L1, backtracks=k)
                        return P_try, log
                    log["invalid"] += 1
            else:
                log["invalid"] += 1
            t *= 0.5
        log.update(eta=0.0, new_loss=L0, backtracks=cfg.max_backtracks)
        return None, log

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
