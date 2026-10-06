"""Optional normal alignment with the target SDF gradient.

    L_normal = sum_i w_i (1 - n_i . grad phi(x_i) / |grad phi(x_i)|)
"""
from __future__ import annotations

import torch

from .target_sdf import SDFGrid


def normal_loss(sdf: SDFGrid, X: torch.Tensor, n_unit: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    g = sdf.gradient(X)
    g = g / torch.clamp(torch.linalg.norm(g, dim=1, keepdim=True), min=1e-12)
    return (w * (1.0 - (n_unit * g).sum(1))).sum()
