"""Residual-guided topology search over CSG solids (a separate representation from the
explicit B-spline patch complex; results are CSG reconstructions, not editable B-spline CAD)."""
from .grammar import (INVERSE, AddBody, AddCavity, BridgeBodies, BridgeVoid, CloseTunnel, OpContext, OpenCavity,
                      PinchBody, RemoveBody, RemoveCavity)
from .measure import measure, measure_multi, summary, voxel_topology
from .objective import FitConfig, ObjectiveConfig, Target, TopoObjective, fit
from .primitives import Box, Capsule, Sphere
from .search import SearchConfig, TopologySearch
from .solid import MATERIAL, VOID, CSGSolid, Grid

__all__ = ["INVERSE", "AddBody", "AddCavity", "BridgeBodies", "BridgeVoid", "CloseTunnel", "OpContext", "OpenCavity",
           "PinchBody", "RemoveBody", "RemoveCavity", "measure", "measure_multi", "summary", "voxel_topology",
           "FitConfig", "ObjectiveConfig", "Target", "TopoObjective", "fit", "Box", "Capsule", "Sphere",
           "SearchConfig", "TopologySearch", "MATERIAL", "VOID", "CSGSolid", "Grid"]
