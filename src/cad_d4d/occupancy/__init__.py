"""Occupancy fields of the explicit solid (bridge to fixed-grid FEM / topology optimization)."""
from .field import (OccupancyGrid, enclosed_volume, signed_distance, soft_occupancy, unsigned_distance,
                    winding_number)

__all__ = ["OccupancyGrid", "enclosed_volume", "signed_distance", "soft_occupancy", "unsigned_distance",
           "winding_number"]
