"""Physics on fixed grids driven by the explicit B-spline solid (FEM compliance)."""
from .fem import FEMConfig, FixedGridFEM, hex8_stiffness
from .terms import ComplianceTerm, fem_on_grid

__all__ = ["ComplianceTerm", "FEMConfig", "FixedGridFEM", "fem_on_grid", "hex8_stiffness"]
