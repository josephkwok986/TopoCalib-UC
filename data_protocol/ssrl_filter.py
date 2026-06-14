"""SSRL-compatible geometry filtering for CAD parts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods

from .constants import (
    SURFACE_TYPE_CONE,
    SURFACE_TYPE_CYLINDER,
    SURFACE_TYPE_PLANE,
    SURFACE_TYPE_SPHERE,
    SURFACE_TYPE_TORUS,
)


STEP_EXTENSIONS = (".step", ".stp", ".STEP", ".STP")

ALLOWED_SURFACE_TYPES = {
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
}

ALLOWED_CURVE_TYPES = {
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
}

SURFACE_TYPE_TO_ID = {
    GeomAbs_Plane: SURFACE_TYPE_PLANE,
    GeomAbs_Cylinder: SURFACE_TYPE_CYLINDER,
    GeomAbs_Cone: SURFACE_TYPE_CONE,
    GeomAbs_Sphere: SURFACE_TYPE_SPHERE,
    GeomAbs_Torus: SURFACE_TYPE_TORUS,
}


@dataclass(frozen=True)
class GeometryCheck:
    passed: bool
    num_faces: int
    num_solids: int
    reason: str = ""


def read_step_shape(step_path: Path) -> Any:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed with status {status}")
    reader.TransferRoots()
    return reader.OneShape()


def count_shape_items(shape: Any, kind: Any) -> int:
    count = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def iter_faces(shape: Any):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        yield topods.Face(explorer.Current())
        explorer.Next()


def iter_edges(shape: Any):
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        yield topods.Edge(explorer.Current())
        explorer.Next()


def check_shape_geometry(shape: Any, *, expected_faces: int | None = None) -> GeometryCheck:
    num_solids = count_shape_items(shape, TopAbs_SOLID)
    num_faces = count_shape_items(shape, TopAbs_FACE)

    if num_solids != 1:
        return GeometryCheck(False, num_faces, num_solids, f"num_solids={num_solids}")
    if expected_faces is not None and num_faces != expected_faces:
        return GeometryCheck(False, num_faces, num_solids, f"face_count_mismatch step={num_faces} labels={expected_faces}")

    for face in iter_faces(shape):
        surface_type = BRepAdaptor_Surface(face).GetType()
        if surface_type not in ALLOWED_SURFACE_TYPES:
            return GeometryCheck(False, num_faces, num_solids, f"surface_type={int(surface_type)}")

    for edge in iter_edges(shape):
        curve_type = BRepAdaptor_Curve(edge).GetType()
        if curve_type not in ALLOWED_CURVE_TYPES:
            return GeometryCheck(False, num_faces, num_solids, f"curve_type={int(curve_type)}")

    return GeometryCheck(True, num_faces, num_solids)


def check_step_geometry(step_path: Path, *, expected_faces: int | None = None) -> GeometryCheck:
    try:
        return check_shape_geometry(read_step_shape(step_path), expected_faces=expected_faces)
    except Exception as exc:
        return GeometryCheck(False, 0, 0, f"geometry_error={type(exc).__name__}: {exc}")


def find_step_files(step_dir: Path) -> dict[str, Path]:
    step_files: dict[str, Path] = {}
    for path in step_dir.rglob("*"):
        if path.is_file() and path.suffix in STEP_EXTENSIONS:
            step_files.setdefault(path.stem, path)
    return step_files


def surface_type_id(face: Any) -> int:
    return SURFACE_TYPE_TO_ID[BRepAdaptor_Surface(face).GetType()]

