"""Compute device and dtype for all differentiable evaluation.

The device defaults to CUDA when available; override with the environment
variable ``CAD_D4D_DEVICE`` (e.g. ``cpu``) or :func:`set_device`. Structural
computation (knot vectors, sparse DOF maps, rewrites) stays in NumPy/SciPy on
the CPU; per-structure operators are uploaded once and all per-step work
(proxy evaluation, losses, gradients, validity checks) runs on the device.

Geometry is float64 everywhere. Brute-force candidate searches (nearest
triangles, bounding-box overlap) may use float32 because their results are
refined exactly in float64 afterwards.
"""
from __future__ import annotations

import os

import numpy as np
import torch

DTYPE = torch.float64
_device: torch.device | None = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        name = os.environ.get("CAD_D4D_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        _device = torch.device(name)
    return _device


def set_device(name: str | torch.device) -> None:
    global _device
    _device = torch.device(name)


def is_cuda() -> bool:
    return get_device().type == "cuda"


def to_tensor(x, dtype=DTYPE) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=dtype, device=get_device())


def to_numpy(t) -> np.ndarray:
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)
