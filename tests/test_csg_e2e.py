"""Deterministic end-to-end topology reconstructions (CPU, ~3 s each).

1 cavity            solid target with an enclosed cavity, initialized without it
2 tunnel            target with a through-tunnel, initialized without it
3 two_bodies        two disconnected bodies, initialized as one body
4 bridge            connected target needing a material bridge, initialized as two bodies
5 spurious_cavity   solid target initialized with an unnecessary cavity
5 spurious_tunnel   solid target initialized with an unnecessary tunnel
6 extra_body        target initialized with an unsupported extra body

Success requires, on evaluation grids independent of the optimization grid:
* the intended components / cavities / tunnels / genus at two resolutions
  (56^3 and 80^3, mesh and voxel measurements consistent);
* no sub-threshold ("tiny") components or cavities;
* the same topology when the surface is offset by +-0.03 (so a feature thinner
  than ~2 cells -- a near-zero bridge, tunnel or cavity -- fails);
* occupancy IoU >= 0.9 at 64^3 and at least a 5x lower volume error than the
  initial state.
"""
import pytest
import torch

from cad_d4d.csg import Grid, SearchConfig, TopoObjective, TopologySearch, measure_multi
from cad_d4d.csg.targets import CASES

EXPECTED_OPS = {"cavity": {"AddCavity"}, "tunnel": {"BridgeVoid"}, "two_bodies": {"PinchBody"},
                "bridge": {"BridgeBodies", "AddBody"}, "spurious_cavity": {"RemoveCavity"},
                "spurious_tunnel": {"CloseTunnel"}, "extra_body": {"RemoveBody"}}


def iou(a_sdf, b_sdf, n=64):
    g = Grid(n, offset=0.21)
    with torch.no_grad():
        a, b = a_sdf(g.points) < 0, b_sdf(g.points) < 0
    return float((a & b).sum()) / float((a | b).sum())


@pytest.mark.parametrize("name", list(CASES))
def test_end_to_end_case(name):
    torch.manual_seed(0)
    tgt, init, want = CASES[name]()
    obj = TopoObjective(tgt, Grid(32))
    res = TopologySearch(obj, SearchConfig()).run(init)
    m = measure_multi(res.solid.sdf, ns=(56, 80))
    assert m["stable"], m
    for k, v in want.items():
        assert m[k] == v, (k, m)
    assert m["genus"] == want["tunnels"]
    for r in m["per_resolution"]:
        assert r["tiny_components"] == 0 and r["tiny_cavities"] == 0
    for delta in (-0.03, 0.03):
        off = measure_multi(lambda X, d=delta: res.solid.sdf(X) + d, ns=(64,))
        assert all(off[k] == v for k, v in want.items()), (delta, off)
    assert iou(res.solid.sdf, tgt.sdf) >= 0.9
    assert obj.value(res.solid)["volume"] < 0.2 * obj.value(init)["volume"]
    # the change came from a measured grammar edit, not from silent drift
    ops = {e["op"] for e in res.events}
    assert ops & EXPECTED_OPS[name], (ops, res.events)
    assert all(e["measured_before"] != e["measured_after"] for e in res.events)
    # the target itself has the intended topology under the same measurement
    t = measure_multi(tgt.sdf, ns=(56, 80))
    assert all(t[k] == v for k, v in want.items())
