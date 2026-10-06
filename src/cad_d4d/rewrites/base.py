"""Rewrite interface.

A rewrite maps a valid state to a new valid state (never mutating its input).

* ``exact``      -- geometry-preserving by construction (KnotInsert, SplitFace,
                    SplitEdge, LocalRefine, topological MergeEdge).
* ``refinement`` -- increases structure / continuous dimension.

Inexact simplifications (KnotRemove, MergeFace) refit by least squares and
report the measured geometric ``deviation``; they fail if it exceeds their
epsilon. Rewrites contain no SRD policy (scoring/acceptance lives in
``cad_d4d.optimization``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..geometry.state import CADState
from ..geometry.topology import TopologyError


@dataclass
class RewriteOutcome:
    state: CADState | None
    reason: str = ""
    deviation: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state is not None


class Rewrite(ABC):
    kind: str = "rewrite"
    exact: bool = False
    refinement: bool = False

    @abstractmethod
    def touched_faces(self, state: CADState) -> set[int]:
        """Faces whose geometry, structure or boundary wire this rewrite changes."""

    @abstractmethod
    def _apply(self, state: CADState) -> RewriteOutcome:
        """Implementation; receives a private copy it may mutate."""

    def apply(self, state: CADState, check: bool = True) -> RewriteOutcome:
        try:
            out = self._apply(state.copy())
        except (TopologyError, ValueError) as exc:
            return RewriteOutcome(None, f"{type(exc).__name__}: {exc}")
        if out.state is not None:
            out.state.structure_changed()
            if check:
                try:
                    out.state.cx.check_invariants()
                except TopologyError as exc:
                    return RewriteOutcome(None, f"invariant violated: {exc}")
        return out

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def describe(self) -> dict:
        return {"kind": self.kind, "exact": self.exact, "refinement": self.refinement, **self.params()}

    def location(self, state: CADState) -> np.ndarray | None:
        """A representative 3D point (for visualization)."""
        return None

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"


AXES = ("u", "v")


def apply_along_axis(net: np.ndarray, A: np.ndarray, axis: str) -> np.ndarray:
    """Apply a 1D linear map A (n_new, n_old) along the u or v index of a net (nu, nv, 3)."""
    if axis == "u":
        return np.einsum("ia,ajk->ijk", A, net)
    return np.einsum("ja,iak->ijk", A, net)


def face_center_point(state: CADState, fid: int, uv=(0.5, 0.5)) -> np.ndarray:
    return state.evaluate(fid, np.array([uv]))[0]
