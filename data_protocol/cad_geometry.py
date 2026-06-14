"""CAD geometry extraction helpers based on pythonocc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from OCC.Core import ChFi3d as ChFi3d_module
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.ChFi3d import chfi3d
from OCC.Core.ChFiDS import ChFiDS_Concave, ChFiDS_Convex
from OCC.Core.GeomAbs import GeomAbs_G1
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
    TopTools_ListIteratorOfListOfShape,
)
from OCC.Core.TopoDS import topods

from .constants import (
    EDGE_TYPE_CONCAVE,
    EDGE_TYPE_CONVEX,
    EDGE_TYPE_NAMES,
    EDGE_TYPE_OTHER,
    EDGE_TYPE_SMOOTH,
)
from .ssrl_filter import iter_faces, surface_type_id


_IS_TANGENT_FACES = (
    getattr(chfi3d, "IsTangentFaces", None)
    or getattr(chfi3d, "chfi3d_IsTangentFaces", None)
    or getattr(ChFi3d_module, "chfi3d_IsTangentFaces", None)
)
_DEFINE_CONNECT_TYPE = (
    getattr(chfi3d, "DefineConnectType", None)
    or getattr(chfi3d, "chfi3d_DefineConnectType", None)
    or getattr(ChFi3d_module, "chfi3d_DefineConnectType", None)
)


@dataclass(frozen=True)
class CadGeometry:
    surface_type: np.ndarray
    edges: np.ndarray
    edge_type: np.ndarray
    face_features_raw: np.ndarray
    meta: dict[str, Any]


def _shape_key(shape: Any) -> int:
    return int(shape.HashCode(2_147_483_647))


def _face_features(face: Any, surface_id: int) -> list[float]:
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    centroid = props.CentreOfMass()
    return [
        float(props.Mass()),
        float(centroid.X()),
        float(centroid.Y()),
        float(centroid.Z()),
        float(surface_id),
    ]


def _edge_relation_type(edge: Any, face_a: Any, face_b: Any, *, sin_tol: float = 1.0e-4) -> int:
    try:
        if _IS_TANGENT_FACES is not None and _IS_TANGENT_FACES(edge, face_a, face_b, GeomAbs_G1):
            return EDGE_TYPE_SMOOTH
    except Exception:
        return EDGE_TYPE_OTHER

    try:
        if _DEFINE_CONNECT_TYPE is None:
            return EDGE_TYPE_OTHER
        connect_type = _DEFINE_CONNECT_TYPE(edge, face_a, face_b, sin_tol, False)
    except Exception:
        return EDGE_TYPE_OTHER
    if connect_type == ChFiDS_Convex:
        return EDGE_TYPE_CONVEX
    if connect_type == ChFiDS_Concave:
        return EDGE_TYPE_CONCAVE
    return EDGE_TYPE_OTHER


def extract_cad_geometry(shape: Any) -> CadGeometry:
    faces = list(iter_faces(shape))
    face_index_by_hash = {_shape_key(face): idx for idx, face in enumerate(faces)}

    surface_types: list[int] = []
    features: list[list[float]] = []
    for face in faces:
        sid = surface_type_id(face)
        surface_types.append(sid)
        features.append(_face_features(face, sid))

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    edge_records: dict[tuple[int, int], int] = {}
    edge_type_counts = {name: 0 for name in EDGE_TYPE_NAMES.values()}

    for idx in range(1, edge_to_faces.Size() + 1):
        edge = topods.Edge(edge_to_faces.FindKey(idx))
        face_list = edge_to_faces.FindFromIndex(idx)
        iterator = TopTools_ListIteratorOfListOfShape(face_list)
        incident: list[int] = []
        mapped_faces: dict[int, Any] = {}
        while iterator.More():
            face = topods.Face(iterator.Value())
            face_idx = face_index_by_hash.get(_shape_key(face))
            if face_idx is not None:
                incident.append(face_idx)
                mapped_faces.setdefault(face_idx, face)
            iterator.Next()

        unique = sorted(set(incident))
        if len(unique) != 2:
            continue

        f1 = mapped_faces.get(unique[0], faces[unique[0]])
        f2 = mapped_faces.get(unique[1], faces[unique[1]])
        edge_type = _edge_relation_type(edge, f1, f2)

        key = (unique[0], unique[1])
        edge_records[key] = min(edge_records.get(key, edge_type), edge_type)
        edge_type_counts[EDGE_TYPE_NAMES[edge_type]] += 1

    if edge_records:
        edges = np.asarray(sorted(edge_records.keys()), dtype=np.int64)
        edge_type = np.asarray([edge_records[tuple(edge)] for edge in edges], dtype=np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    return CadGeometry(
        surface_type=np.asarray(surface_types, dtype=np.int64),
        edges=edges,
        edge_type=edge_type,
        face_features_raw=np.asarray(features, dtype=np.float32),
        meta={
            "num_occ_faces": len(faces),
            "num_occ_face_map": int(face_map.Size()),
            "num_raw_edge_records": int(edge_to_faces.Size()),
            "num_smooth_edges": int(edge_type_counts["smooth"]),
            "num_convex_edges": int(edge_type_counts["convex"]),
            "num_concave_edges": int(edge_type_counts["concave"]),
            "num_other_edges": int(edge_type_counts["other"]),
            "edge_type_note": "Fusion360 boundary relation type uses OCC G1 tangent for smooth, then OCC DefineConnectType for convex/concave, with other as fallback.",
        },
    )
