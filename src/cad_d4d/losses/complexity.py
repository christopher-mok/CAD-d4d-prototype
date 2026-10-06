"""Structural complexity C(s) and its birth / grace-period discount.

    C(s) = c_face * #faces + c_cp * #canonical control points

Newly born structure (from accepted refinements) is temporarily charged at a
reduced rate. A birth record of age ``a`` (continuous steps since birth) with
remaining structural increment dC is discounted by

    discount(a) = (1 - lambda_birth / lambda_complex) * ramp(a)
    ramp(a)     = 1                                  for a <= T_hold
                = 1 - (a - T_hold) / T_ramp          for T_hold < a < T_hold + T_ramp
                = 0                                  afterwards

so its effective cost is ``lambda_birth * dC`` during the hold (grace) period
and then ramps linearly to the full ``lambda_complex * dC``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..geometry.state import CADState


@dataclass
class ComplexityConfig:
    c_face: float = 1.0
    c_cp: float = 1.0
    lambda_complex: float = 1e-6
    lambda_birth: float = 1e-7
    hold_steps: int = 40
    ramp_steps: int = 40
    use_grace: bool = True

    @property
    def grace_steps(self) -> int:
        return self.hold_steps + self.ramp_steps


def structural_complexity(state: CADState, cfg: ComplexityConfig) -> float:
    return cfg.c_face * state.n_faces + cfg.c_cp * state.n_control_points


def complexity_delta(before: CADState, after: CADState, cfg: ComplexityConfig) -> float:
    return structural_complexity(after, cfg) - structural_complexity(before, cfg)


def birth_discount(age: int, cfg: ComplexityConfig) -> float:
    if not cfg.use_grace or cfg.grace_steps <= 0:
        return 0.0
    ratio = cfg.lambda_birth / cfg.lambda_complex if cfg.lambda_complex > 0 else 0.0
    if age <= cfg.hold_steps:
        ramp = 1.0
    elif cfg.ramp_steps > 0:
        ramp = max(0.0, 1.0 - (age - cfg.hold_steps) / cfg.ramp_steps)
    else:
        ramp = 0.0
    return (1.0 - ratio) * ramp


def effective_complexity(state: CADState, cfg: ComplexityConfig) -> float:
    """Complexity with young refinements discounted (still in their grace period)."""
    C = structural_complexity(state, cfg)
    for r in state.birth_records:
        if r.remaining > 0:
            C -= birth_discount(r.age, cfg) * r.remaining
    return C


def complexity_cost(state: CADState, cfg: ComplexityConfig, effective: bool = True) -> float:
    C = effective_complexity(state, cfg) if effective else structural_complexity(state, cfg)
    return cfg.lambda_complex * C
