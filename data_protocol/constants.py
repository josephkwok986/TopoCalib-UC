"""Shared constants for TopoCalib-UC data preprocessing."""

from __future__ import annotations

from pathlib import Path


DEFAULT_ERROR_LOG = Path("ERROR.txt")

EDGE_TYPE_OTHER = 0
EDGE_TYPE_SMOOTH = 1
EDGE_TYPE_CONVEX = 2
EDGE_TYPE_CONCAVE = 3

EDGE_TYPE_NAMES = {
    EDGE_TYPE_OTHER: "other",
    EDGE_TYPE_SMOOTH: "smooth",
    EDGE_TYPE_CONVEX: "convex",
    EDGE_TYPE_CONCAVE: "concave",
}

SURFACE_TYPE_OTHER = -1
SURFACE_TYPE_PLANE = 0
SURFACE_TYPE_CYLINDER = 1
SURFACE_TYPE_CONE = 2
SURFACE_TYPE_SPHERE = 3
SURFACE_TYPE_TORUS = 4

SURFACE_TYPE_NAMES = {
    SURFACE_TYPE_OTHER: "other",
    SURFACE_TYPE_PLANE: "plane",
    SURFACE_TYPE_CYLINDER: "cylinder",
    SURFACE_TYPE_CONE: "cone",
    SURFACE_TYPE_SPHERE: "sphere",
    SURFACE_TYPE_TORUS: "torus",
}
