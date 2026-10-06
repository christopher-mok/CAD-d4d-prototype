"""Benchmark target catalog.

``TUNE`` targets may be looked at while developing / tuning; ``TEST`` targets
are for reporting only. Grammar targets are reachable by construction; analytic
targets generally are not.
"""
from __future__ import annotations

import numpy as np

from ..targets import analytic as A
from ..targets.random_grammar import GrammarSpec


def _rng(seed):
    return np.random.default_rng(seed)


def catalog() -> dict[str, dict]:
    """name -> {"spec": builder with .build(), "split": "tune"|"test", "family": ...}"""
    T = {}

    def add(name, spec, split, family):
        T[name] = {"spec": spec, "split": split, "family": family}

    # --- tuning targets ------------------------------------------------------
    add("g_tune", GrammarSpec("g_tune", seed=101, interior=2, near_edge=1, edge=1), "tune", "grammar")
    add("a_tune", A.AnalyticSpec("a_tune", A.add(A.sphere(), A.random_bumps(_rng(201), 5))), "tune", "analytic")

    # --- test targets: grammar (reachable) ------------------------------------
    add("g_multi", GrammarSpec("g_multi", seed=102, interior=3, near_edge=1), "test", "grammar")
    add("g_edges", GrammarSpec("g_edges", seed=103, interior=1, edge=2, corner=1), "test", "grammar")
    add("g_mixed", GrammarSpec("g_mixed", seed=104, interior=2, near_edge=1, edge=1, corner=1,
                               global_scale=(1.15, 0.95, 0.85)), "test", "grammar")

    # --- test targets: analytic (out of grammar) -----------------------------
    add("a_bumps6", A.AnalyticSpec("a_bumps6", A.add(A.sphere(), A.random_bumps(_rng(301), 6))), "test", "analytic")
    add("a_superellipsoid", A.AnalyticSpec("a_superellipsoid", A.superellipsoid(4.0)), "test", "analytic")
    add("a_super_bumps", A.AnalyticSpec("a_super_bumps", A.add(A.superellipsoid(3.0),
                                                                A.random_bumps(_rng(302), 4))), "test", "analytic")
    add("a_ridge", A.AnalyticSpec("a_ridge", A.add(A.sphere(), A.ridge((1.0, 1.0, 0.3), 0.15, 0.2))),
        "test", "analytic")
    add("a_blob", A.AnalyticSpec("a_blob", A.add(A.sphere(), A.blob(_rng(303)))), "test", "analytic")
    add("a_lobed", A.AnalyticSpec("a_lobed", A.add(A.sphere(), A.random_bumps(_rng(304), 3, width=(0.25, 0.4))),
                                  scale=(1.3, 0.9, 0.8)), "test", "analytic")
    # --- harder test targets ---------------------------------------------------
    add("g_dense", GrammarSpec("g_dense", seed=105, interior=4, near_edge=2, edge=2, corner=2,
                               width=(0.2, 0.35)), "test", "grammar")
    add("a_many_bumps", A.AnalyticSpec("a_many_bumps", A.add(A.sphere(), A.random_bumps(
        _rng(305), 12, height=(0.06, 0.18), width=(0.12, 0.3)))), "test", "analytic")
    add("a_super6_dents", A.AnalyticSpec("a_super6_dents", A.add(A.superellipsoid(6.0), A.random_bumps(
        _rng(306), 5, height=(0.08, 0.2), width=(0.15, 0.3)))), "test", "analytic")
    add("a_ridge_bumps", A.AnalyticSpec("a_ridge_bumps", A.add(A.sphere(), A.ridge((0.3, 1.0, 0.8), 0.12, 0.15),
                                                                A.random_bumps(_rng(307), 4))), "test", "analytic")
    return T
