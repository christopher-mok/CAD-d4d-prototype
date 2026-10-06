"""Comparison of adaptive runs against the uniform-refinement curve of a target."""
from __future__ import annotations

import numpy as np


def interp_loglog(x, xs, ys):
    """Interpolate y(x) along a curve given at xs (increasing), in log-log; None outside range."""
    lx, lxs, lys = np.log(x), np.log(xs), np.log(ys)
    if lx < lxs[0] or lx > lxs[-1]:
        return None
    return float(np.exp(np.interp(lx, lxs, lys)))


def uniform_curve(rows):
    """(control points, fit) of the fixed uniform runs, sorted by control points."""
    pts = sorted((r["n_cp"], r["fit"]) for r in rows if r["kind"] == "fixed")
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def compare(row, cps, fits):
    """(efficiency, cp_saving) of one run against a uniform curve.

    efficiency = fit / uniform fit at the same #cp      (< 1: run better)
    cp_saving  = uniform #cp for the same fit / run #cp (> 1: run smaller)
    Either is None when the run lies outside the curve's range.
    """
    eff = interp_loglog(row["n_cp"], cps, fits)
    order = np.argsort(fits)
    cp_needed = interp_loglog(row["fit"], fits[order], cps[order])
    return (row["fit"] / eff if eff else None), (cp_needed / row["n_cp"] if cp_needed else None)


def annotate(rows: list[dict]) -> list[dict]:
    """Copy of ``rows`` with efficiency / cp_saving filled in per (target, tag)."""
    out = []
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["target"], r["tag"]), []).append(r)
    for grp in groups.values():
        cps, fits = uniform_curve(grp)
        for r in grp:
            r = dict(r)
            if r["kind"] != "fixed" and len(cps) >= 2:
                r["efficiency"], r["cp_saving"] = compare(r, cps, fits)
            out.append(r)
    return out
