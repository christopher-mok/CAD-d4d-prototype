"""Shape-matching (and physics) objective.

    L = lambda_sdf * L_sdf + lambda_cov * L_cov + lambda_normal * L_normal
        + lambda_volume * ((V - V_target) / V_target)^2
        + sum_k weight_k * physics_k(p)                         (e.g. normalized FEM compliance)
        + lambda_fair * L_fair                                  (differentiable, "smooth")
      + lambda_complex * C(s)                                   (structural, piecewise constant)

The shape-target terms are skipped when ``target`` is None (physics-only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..device import to_tensor
from ..geometry.state import CADState
from ..optimization.discretization import Discretization, DiscretizationConfig, get_discretization
from ..targets.synthetic import Target
from .complexity import ComplexityConfig, effective_complexity, structural_complexity
from .coverage import coverage_loss
from .normals import normal_loss


@dataclass
class ObjectiveConfig:
    lambda_sdf: float = 1.0
    lambda_coverage: float = 1.0
    lambda_normal: float = 0.0
    lambda_fair: float = 2e-8
    lambda_volume: float = 0.0  # ((V - V_target) / V_target)^2, V by the divergence theorem
    volume_target: float | None = None  # defaults to the target shape's volume
    physics: list = field(default_factory=list)  # differentiable terms: callable(state, P, disc, terms) with .weight
    sdf_eval: str = "trilinear"  # "trilinear" (smooth, used for optimization) | "exact" (narrow band)
    coverage_k: int = 6
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
    discretization: DiscretizationConfig = field(default_factory=DiscretizationConfig)


class ShapeObjective:
    def __init__(self, target: Target | None, cfg: ObjectiveConfig | None = None):
        """``target`` may be None for purely physics-driven objectives (fit terms are then zero)."""
        self.target = target
        self.cfg = cfg or ObjectiveConfig()

    def disc(self, state: CADState) -> Discretization:
        return get_discretization(state, self.cfg.discretization)

    def terms(self, state: CADState, P: torch.Tensor, disc: Discretization | None = None) -> dict:
        """Differentiable loss terms at parameters ``P`` for ``state``'s structure."""
        cfg = self.cfg
        disc = disc or self.disc(state)
        sm = disc.sampler
        X, Xu, Xv = sm.evaluate(P)
        n = torch.linalg.cross(Xu, Xv, dim=1)
        nn = torch.linalg.norm(n, dim=1)
        area = (nn * sm.quad_w).detach()
        w = area / area.sum()
        zero = torch.zeros((), dtype=P.dtype, device=P.device)
        out = {"X": X, "Xu": Xu, "Xv": Xv, "w": w}
        if self.target is not None:
            out["phi"] = phi = self.target.sdf(X, exact=cfg.sdf_eval == "exact")
            out["sdf"] = (w * phi**2).sum()
        else:
            out["sdf"] = zero
        if cfg.lambda_coverage > 0 and self.target is not None:
            cs = disc.coverage
            Xc = X if cs is sm else torch.sparse.mm(cs.G, P)
            out["coverage"], out["cov_d"], _ = coverage_loss(
                self.target.points, Xc, cs.tri_t, cfg.coverage_k)
        else:
            out["coverage"] = zero
        if cfg.lambda_normal > 0 and self.target is not None:
            out["normal"] = normal_loss(self.target.sdf, X, n / torch.clamp(nn[:, None], min=1e-300), w)
        else:
            out["normal"] = zero
        if cfg.lambda_volume > 0:
            from ..occupancy.field import enclosed_volume
            out["volume"] = enclosed_volume(X, sm.tri_t)
            if cfg.volume_target is None and self.target is None:
                raise ValueError("lambda_volume > 0 needs ObjectiveConfig.volume_target when there is no target")
            vt = cfg.volume_target if cfg.volume_target is not None else self.target.volume
            out["volume_err"] = ((out["volume"] - vt) / vt) ** 2
        else:
            out["volume_err"] = zero
        FP = torch.sparse.mm(disc.F, P)
        out["fair"] = (FP**2).sum()
        out["fit"] = (cfg.lambda_sdf * out["sdf"] + cfg.lambda_coverage * out["coverage"]
                      + cfg.lambda_normal * out["normal"] + cfg.lambda_volume * out["volume_err"])
        for term in cfg.physics:
            val = term(state, P, disc, out)
            out[f"physics_{term.name}"] = val
            out["fit"] = out["fit"] + term.weight * val
        out["smooth"] = out["fit"] + cfg.lambda_fair * out["fair"]
        return out

    def smooth_value(self, state: CADState, P=None) -> float:
        P = to_tensor(state.values() if P is None else P)
        with torch.no_grad():
            return float(self.terms(state, P)["smooth"])

    def report(self, state: CADState) -> dict:
        """Scalar summary for logging."""
        with torch.no_grad():
            t = self.terms(state, to_tensor(state.values()))
        cc = self.cfg.complexity
        C_full = structural_complexity(state, cc)
        C_eff = effective_complexity(state, cc)
        smooth = float(t["smooth"])
        return {
            "sdf": float(t["sdf"]), "coverage": float(t["coverage"]), "normal": float(t["normal"]),
            "fair": float(t["fair"]), "fit": float(t["fit"]), "smooth": smooth,
            "complexity": C_full, "complexity_eff": C_eff,
            "total": smooth + cc.lambda_complex * C_full,
            "total_eff": smooth + cc.lambda_complex * C_eff,
            "n_faces": state.n_faces, "n_cp": state.n_control_points,
        }
