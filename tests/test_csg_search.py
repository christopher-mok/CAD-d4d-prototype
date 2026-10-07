"""Residual direction, objective normalization, and the trial-based topology search
(isolation, equal budgets, rejection/acceptance, provenance, inverse-edit protection)."""
import numpy as np
import pytest
import torch

from cad_d4d.csg import (MATERIAL, VOID, AddCavity, Grid, RemoveCavity, SearchConfig, Sphere, Target, TopoObjective,
                         TopologySearch)
from cad_d4d.csg.objective import fit
from cad_d4d.csg.proposals import propose
from cad_d4d.csg.residuals import analyze
from cad_d4d.csg.search import Protection
from cad_d4d.csg.targets import CASES, solid_of


def sphere_target(r=0.5, c=(0, 0, 0)):
    c = torch.tensor(c, dtype=torch.float64)
    return Target("sphere", lambda X: (X - c).norm(dim=1) - r)


def test_residual_direction_missing_vs_excess():
    g = Grid(32)
    obj = TopoObjective(sphere_target(0.6), g)
    small = solid_of((MATERIAL, "body", Sphere((0, 0, 0), 0.35)))
    ev = analyze(small, obj.rho_t, g, obj.cfg.eps)
    assert ev.add and not ev.remove                              # current too small: material missing
    assert ev.add[0].material_contacts and ev.add[0].volume > 0.5
    big = solid_of((MATERIAL, "body", Sphere((0, 0, 0), 0.85)))
    ev = analyze(big, obj.rho_t, g, obj.cfg.eps)
    assert ev.remove and not ev.add                              # current too big: excess material
    assert ev.remove[0].exterior_contacts                        # ... at the outer surface


def test_enclosed_excess_is_cavity_evidence_and_surface_excess_is_not():
    tgt, init, _ = CASES["cavity"]()
    g = Grid(32)
    obj = TopoObjective(tgt, g)
    ev = analyze(init, obj.rho_t, g, obj.cfg.eps)
    enclosed = [r for r in ev.remove if not r.exterior_contacts and not r.cavity_contacts]
    assert len(enclosed) == 1 and enclosed[0].r_in > 0.2
    ops = [c.op.name for c in propose(init, ev, g)]
    assert "AddCavity" in ops and "BridgeVoid" not in ops
    tgt, init, _ = CASES["tunnel"]()
    obj = TopoObjective(tgt, g)
    ev = analyze(init, obj.rho_t, g, obj.cfg.eps)
    through = [r for r in ev.remove if len(r.exterior_contacts) >= 2]
    assert through and not [r for r in ev.remove if not r.exterior_contacts and r.voxels > 50]
    assert "BridgeVoid" in [c.op.name for c in propose(init, ev, g)]


def test_objective_terms_are_resolution_independent():
    tgt = sphere_target(0.6)
    s = solid_of((MATERIAL, "body", Sphere((0.05, 0, 0), 0.5)))
    v = [TopoObjective(tgt, Grid(n)).value(s) for n in (24, 48, 72)]
    for k in ("volume", "surface"):
        vals = np.array([x[k] for x in v])
        assert vals.max() / vals.min() < 1.15, (k, vals)
    # a perfect reconstruction has zero volume/surface loss; complexity counts features
    perfect = solid_of((MATERIAL, "body", Sphere((0, 0, 0), 0.6)))
    p = TopoObjective(tgt, Grid(40)).value(perfect)
    assert p["volume"] < 1e-12 and p["surface"] < 1e-6 and p["complexity"] == 1


def small_search(**kw):
    cfg = SearchConfig(rounds=2, trial_steps=15, final_steps=10, **kw)
    return cfg


def test_trials_are_isolated_and_budgets_equal(monkeypatch):
    tgt, init, _ = CASES["cavity"]()
    obj = TopoObjective(tgt, Grid(28))
    snapshot = [f.prim.describe() for f in init.features]
    seen = []
    import cad_d4d.csg.search as S
    real_fit = S.fit

    def spy(solid, objective, steps, cfg=None):
        seen.append((id(solid), steps))
        return real_fit(solid, objective, steps, cfg)

    monkeypatch.setattr(S, "fit", spy)
    res = TopologySearch(obj, small_search()).run(init)
    assert [f.prim.describe() for f in init.features] == snapshot and len(init.features) == 1   # input untouched
    trial_steps = [s for _, s in seen[1:-1]]                    # between the initial and the final fit
    assert set(trial_steps) == {15}                             # baseline and every candidate: same K
    assert len({i for i, _ in seen[1:-1]}) == len(seen[1:-1])   # every trial optimized its own copy
    assert res.costs["candidate_steps"] == 15 * res.costs["n_candidates"]
    assert res.costs["baseline_steps"] == 15 * len(res.rounds)


def test_rejection_keeps_the_optimized_baseline():
    tgt, init, _ = CASES["cavity"]()
    obj = TopoObjective(tgt, Grid(28))
    res = TopologySearch(obj, small_search(accept_abs=10.0)).run(init)   # unreachable threshold
    assert not res.events and len(res.solid.features) == 1
    assert all(t["status"] in ("rejected", "failed", "protected") for t in res.trials)
    assert any(t["status"] == "rejected" and t["score"] > 0 for t in res.trials)   # good edits were seen, refused
    # the committed state is the fitted baseline: better than the initial state
    assert obj.value(res.solid)["total"] < obj.value(init)["total"]


def test_acceptance_commits_the_optimized_candidate_with_provenance():
    tgt, init, _ = CASES["cavity"]()
    obj = TopoObjective(tgt, Grid(28))
    res = TopologySearch(obj, small_search()).run(init)
    assert res.events and res.events[0]["op"] == "AddCavity"
    e = res.events[0]
    assert e["score"] > e["threshold"] and e["measured_after"].endswith("cavities 1, tunnels 0")
    fid = e["created"][0]
    f = res.solid.get(fid)
    assert f.role == VOID and f.provenance["op"] == "AddCavity" and f.provenance["round"] == 0
    assert f.provenance["rule"].startswith("enclosed excess material")
    acc = [t for t in res.trials if t["status"] == "accepted"]
    assert len(acc) == 1 and acc[0]["trial_steps"] == 15


def test_inverse_edits_are_temporarily_protected():
    p = Protection("AddCavity", created=[5], location=np.zeros(3), radius=0.3, until_round=2)
    assert p.blocks(RemoveCavity(fid=5), 1)                      # removing the new feature: blocked
    assert not p.blocks(RemoveCavity(fid=5), 3)                  # ... only temporarily
    assert not p.blocks(RemoveCavity(fid=9, point=(0.9, 0, 0)), 1)   # another feature far away: allowed
    assert not p.blocks(AddCavity((0, 0, 0), radius=0.1), 1)     # not the inverse
    q = Protection("RemoveCavity", created=[], location=np.zeros(3), radius=0.2, until_round=1)
    assert q.blocks(AddCavity((0.1, 0, 0), radius=0.1), 1)       # re-adding nearby: blocked


def test_search_does_not_edit_when_the_structure_is_already_right():
    tgt = Target("ball", lambda X: X.norm(dim=1) - 0.55)
    init = solid_of((MATERIAL, "body", Sphere((0.03, 0, 0), 0.5)))
    res = TopologySearch(TopoObjective(tgt, Grid(28)), small_search()).run(init)
    assert not res.events and len(res.solid.features) == 1


def test_fit_projects_to_minimum_size():
    tgt = Target("ball", lambda X: X.norm(dim=1) - 0.55)
    s = solid_of((MATERIAL, "body", Sphere((0, 0, 0), 0.5)), (VOID, "cavity", Sphere((0.0, 0, 0), 0.07)))
    fit(s, TopoObjective(tgt, Grid(24)), 40)
    assert float(s.features[1].prim.params["r"]) >= 0.06 - 1e-12
    with pytest.raises(KeyError):
        s.get(99)
