"""Topology grammar: geometric effect and preconditions of every rewrite, inverse edits on
existing (not just most recent) features, and measured (not assumed) effects."""
import numpy as np

from cad_d4d.csg import (MATERIAL, VOID, AddBody, AddCavity, BridgeBodies, BridgeVoid, Capsule, CloseTunnel, Grid,
                         OpContext, OpenCavity, PinchBody, RemoveBody, RemoveCavity, Sphere)
from cad_d4d.csg.grammar import topo
from cad_d4d.csg.primitives import Box
from cad_d4d.csg.targets import rbox, solid_of

CTX = OpContext(Grid(40, offset=0.37))


def counts(s):
    t = topo(s, CTX)
    return t["components"], t["cavities"], t["tunnels"]


def block():
    return solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.45))))


def two_balls():
    return solid_of((MATERIAL, "body", Sphere((-0.5, 0, 0), 0.32)), (MATERIAL, "body", Sphere((0.5, 0, 0), 0.32)))


def test_add_body_and_preconditions():
    s = block()
    out = AddBody((0.0, 0.0, 0.6), radius=0.2).apply(s, CTX)   # outside, but touching: grows the body
    assert out.ok and counts(out.solid) == (1, 0, 0) and out.info["volume_change"] > 0
    out = AddBody((0.0, 0.95, 0.0), radius=0.15).apply(s, CTX)  # outside: a new body
    assert out.ok and counts(out.solid) == (2, 0, 0) and counts(s) == (1, 0, 0)
    assert out.solid.get(out.info["created"][0]).provenance["op"] == "AddBody"
    bad = AddBody((0.0, 0.0, 0.0), radius=0.2).apply(s, CTX)
    assert not bad.ok and "inside existing material" in bad.reason


def test_remove_body_works_on_original_features_and_refuses_the_last_one():
    s = two_balls()
    out = RemoveBody(0).apply(s, CTX)                    # an original body, not a recent edit
    assert out.ok and counts(out.solid) == (1, 0, 0) and out.info["removed"] == [0]
    last = RemoveBody(1).apply(out.solid, CTX)
    assert not last.ok and "last material" in last.reason
    assert not RemoveBody(7).apply(s, CTX).ok


def test_bridge_bodies_connects_and_detects_no_gap():
    s = two_balls()
    out = BridgeBodies((-0.4, 0, 0), (0.4, 0, 0), 0.12).apply(s, CTX)
    assert out.ok and counts(out.solid) == (1, 0, 0)
    joined = out.solid
    again = BridgeBodies((-0.4, 0, 0), (0.4, 0, 0), 0.12).apply(joined, CTX)
    assert not again.ok and "already connected" in again.reason
    far = BridgeBodies((-0.4, 0, 0), (0.4, 0.9, 0), 0.1).apply(s, CTX)
    assert not far.ok and "farther than the bridge radius" in far.reason
    # a second bridge between the same two bodies makes a handle (measured: tunnels +1)
    loop = BridgeBodies((-0.4, 0.2, 0.0), (0.4, 0.2, 0.0), 0.08).apply(
        solid_of((MATERIAL, "body", Sphere((-0.55, 0, 0), 0.3)), (MATERIAL, "body", Sphere((0.55, 0, 0), 0.3)),
                 (MATERIAL, "bridge", Capsule((-0.4, -0.2, 0), (0.4, -0.2, 0), 0.08))), CTX)
    assert loop.ok and counts(loop.solid) == (1, 0, 1)


def test_pinch_body_separates_any_neck_including_original_bridges():
    s = solid_of((MATERIAL, "body", Sphere((-0.5, 0, 0), 0.3)), (MATERIAL, "body", Sphere((0.5, 0, 0), 0.3)),
                 (MATERIAL, "bridge", Capsule((-0.4, 0, 0), (0.4, 0, 0), 0.1)))
    assert counts(s) == (1, 0, 0)
    out = PinchBody((0, 0, 0), (1, 0, 0), 0.08, 0.3).apply(s, CTX)
    assert out.ok and counts(out.solid) == (2, 0, 0)
    groove = PinchBody((0.0, 0, 0.0), (1, 0, 0), 0.08, 0.15).apply(block(), CTX)   # does not cut through
    assert not groove.ok and "separation" in groove.reason
    assert not PinchBody((0, 0.9, 0), (1, 0, 0), 0.08, 0.3).apply(s, CTX).ok        # not in material


def test_add_cavity_requires_enclosure():
    s = block()
    out = AddCavity((0.0, 0.0, 0.0), radius=0.2).apply(s, CTX)
    assert out.ok and counts(out.solid) == (1, 1, 0)
    near_wall = AddCavity((0.0, 0.0, 0.3), radius=0.2).apply(s, CTX)
    assert not near_wall.ok and "not enclosed" in near_wall.reason


def test_remove_cavity_feature_and_plug_variants():
    s = solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.45))), (VOID, "cavity", Sphere((0.2, 0, 0), 0.15)),
                 (VOID, "cavity", Sphere((-0.3, 0, 0), 0.15)))
    assert counts(s) == (1, 2, 0)
    out = RemoveCavity(fid=1).apply(s, CTX)              # the older of two existing cavities
    assert out.ok and counts(out.solid) == (1, 1, 0)
    assert not RemoveCavity(fid=0).apply(s, CTX).ok      # material, not a void
    # a cavity formed by material arrangement (a closed shell of six slabs) has no void feature:
    # RemoveCavity fills it with a plug
    t = 0.12
    walls = []
    for ax in range(3):
        for sx in (-1, 1):
            c = [0.0, 0.0, 0.0]
            c[ax] = sx * 0.5
            h = [0.62, 0.62, 0.62]
            h[ax] = t
            walls.append((MATERIAL, "body", Box(c, h, (0, 0, 0), 0.0)))
    shell = solid_of(*walls)
    assert counts(shell) == (1, 1, 0)
    ball = RemoveCavity(point=(0, 0, 0), radius=0.4).apply(shell, CTX)   # a ball leaves the corners empty
    assert not ball.ok and "cavities did not decrease" in ball.reason
    plug = RemoveCavity(point=(0, 0, 0), axes=np.eye(3), half_extents=(0.45, 0.45, 0.45)).apply(shell, CTX)
    assert plug.ok and counts(plug.solid) == (1, 0, 0)
    assert not RemoveCavity(point=(0, 0, 0.9), radius=0.2).apply(shell, CTX).ok   # not in a cavity
    assert not RemoveCavity(fid=1).apply(block(), CTX).ok                          # no cavity at all


def test_bridge_void_open_cavity_and_close_tunnel():
    s = block()
    tun = BridgeVoid((0, 0, -0.8), (0, 0, 0.8), 0.2).apply(s, CTX)
    assert tun.ok and counts(tun.solid) == (1, 0, 1)
    inside = BridgeVoid((0, 0, 0), (0, 0, 0.8), 0.2).apply(s, CTX)
    assert not inside.ok and "inside material" in inside.reason
    hollow = AddCavity((0.0, 0.0, 0.0), radius=0.2).apply(s, CTX).solid
    opened = OpenCavity((0, 0, 0), (0, 0, 0.8), 0.1).apply(hollow, CTX)
    assert opened.ok and counts(opened.solid) == (1, 0, 0)
    not_cav = OpenCavity((0.0, 0.9, 0), (0, 0, 0.8), 0.1).apply(hollow, CTX)
    assert not not_cav.ok
    # CloseTunnel on an existing channel feature (not the most recent edit)
    two = BridgeVoid((0.45, 0, -0.8), (0.45, 0, 0.8), 0.12).apply(tun.solid, CTX).solid
    assert counts(two) == (1, 0, 2)
    first_channel = [f.id for f in two.voids][0]
    closed = CloseTunnel(fid=first_channel).apply(two, CTX)
    assert closed.ok and counts(closed.solid) == (1, 0, 1)
    # a handle made by material (two bridges): CloseTunnel fills the hole with a plug
    ring = solid_of((MATERIAL, "body", Sphere((-0.55, 0, 0), 0.3)), (MATERIAL, "body", Sphere((0.55, 0, 0), 0.3)),
                    (MATERIAL, "bridge", Capsule((-0.4, -0.25, 0), (0.4, -0.25, 0), 0.1)),
                    (MATERIAL, "bridge", Capsule((-0.4, 0.25, 0), (0.4, 0.25, 0), 0.1)))
    assert counts(ring) == (1, 0, 1)
    plug = CloseTunnel(center=(0, 0, 0), axes=np.eye(3), half_extents=(0.35, 0.3, 0.1)).apply(ring, CTX)
    assert plug.ok and counts(plug.solid) == (1, 0, 0)
    assert not CloseTunnel(center=(0, 0, 0), axes=np.eye(3), half_extents=(0.2, 0.2, 0.1)).apply(block(), CTX).ok


def test_ops_never_mutate_their_input():
    s = two_balls()
    before = [f.prim.describe() for f in s.features]
    for op in (AddBody((0, 0.8, 0), radius=0.15), RemoveBody(0), BridgeBodies((-0.4, 0, 0), (0.4, 0, 0), 0.1),
               PinchBody((-0.5, 0, 0), (1, 0, 0), 0.05, 0.4)):
        op.apply(s, CTX)
    assert [f.prim.describe() for f in s.features] == before and len(s.features) == 2


def test_failed_effect_is_reported_not_substituted():
    s = block()
    # a "cavity" big enough to pass the depth check on a coarse wall estimate but measured as enclosed: ok;
    # an AddBody whose primitive is below the check-grid resolution is refused explicitly
    tiny = AddBody((0.0, 0.0, 1.0), radius=0.01).apply(s, CTX)
    assert not tiny.ok and "resolution" in tiny.reason
    out = PinchBody((0.0, 0.0, 0.0), (0, 0, 1), 0.05, 0.2).apply(s, CTX)
    assert not out.ok and out.solid is None and "measured" in out.reason


def test_pinch_split_variant_replaces_the_body_by_two():
    s = solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.8, 0.3, 0.3))))
    out = PinchBody((0, 0, 0), (1, 0, 0), 0.08, 0.5, split=0).apply(s, CTX)
    assert out.ok and counts(out.solid) == (2, 0, 0)
    assert out.info["removed"] == [0] and len(out.info["created"]) == 2 and not out.solid.voids
    for fid in out.info["created"]:
        f = out.solid.get(fid)
        assert f.role == MATERIAL and f.provenance["split_from"] == 0 and f.provenance["op"] == "PinchBody"
    miss = PinchBody((0, 0, 0), (1, 0, 0), 0.08, 0.5, split=3).apply(s, CTX)
    assert not miss.ok and "not an existing material feature" in miss.reason
