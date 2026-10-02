"""Mesh repair and export helpers used by every public inference entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tools.mesh_hole_fill import (
    _surviving_orig_count,
    count_holes,
    export_obj_keep_all_vertices,
    global_orient_fix,
    meshflow_append,
    run_fill_pipeline,
)
from tools.normal_correct import correct_normals

_NORMAL_CORRECT_MAX_FACES = 100_000
_NORMAL_CORRECT_GUARD_TOL = 0.02


def _correct_face_normals(
    mesh: trimesh.Trimesh,
    *,
    max_faces: int = _NORMAL_CORRECT_MAX_FACES,
    guard_tol: float = _NORMAL_CORRECT_GUARD_TOL,
) -> tuple[trimesh.Trimesh, dict[str, Any], bool]:
    """Apply winding-only per-face correction with a visible-area guard."""
    face_count = len(mesh.faces)
    if face_count > max_faces:
        return mesh, {"method": "per-face", "status": "skipped-too-large"}, False
    try:
        faces_before = np.asarray(mesh.faces)
        faces_after, _, info, _ = correct_normals(
            np.asarray(mesh.vertices, dtype=np.float64),
            faces_before,
            verbose=False,
        )
        if faces_after.shape != faces_before.shape:
            raise RuntimeError("normal correction changed the face array shape")
        if info["frontvis_after"] < info["frontvis_before"] - guard_tol:
            return mesh, {"method": "per-face", "status": "reverted"}, False
        mesh.faces = faces_after
        return (
            mesh,
            {
                "method": "per-face",
                "status": "applied",
                "flipped_faces": int(info["n_flip"]),
                "backend": str(info["backend"]),
            },
            True,
        )
    except Exception as error:  # optional ray backends may be unavailable
        return (
            mesh,
            {
                "method": "per-face",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
            False,
        )


def fill_mesh_holes(
    mesh: trimesh.Trimesh,
    *,
    normal_correct: bool = True,
    global_orient: bool = True,
    boundary_vertices: int = 2,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Run the release meshflow, Liepa, and orientation-repair sequence."""
    if mesh is None or len(mesh.faces) == 0:
        return mesh, {"status": "skipped-empty"}

    original = mesh.copy()
    try:
        faces_before = len(original.faces)
        holes_before = count_holes(original)
        anchor_count = _surviving_orig_count(original)
        meshflow_mesh, meshflow_added = meshflow_append(
            original,
            fill_quad_rings=True,
            ring_max_n=4,
            bnd_verts=int(boundary_vertices),
        )
        result, fill_stats = run_fill_pipeline(
            meshflow_mesh,
            dedup=True,
            zipper=True,
            zipper_tol=None,
            quadfill=True,
            poly_max_n=6,
            remove_intersect=False,
            orient_new=True,
            verbose=False,
            orient_anchor=anchor_count,
        )

        orientation: dict[str, Any] = {"status": "disabled"}
        oriented = False
        if normal_correct:
            result, orientation, oriented = _correct_face_normals(result)
        if not oriented and global_orient:
            vertices_before = np.asarray(result.vertices).copy()
            face_count_before = len(result.faces)
            result, orient_stats = global_orient_fix(result, verbose=False)
            if not np.array_equal(vertices_before, np.asarray(result.vertices)):
                raise RuntimeError("global orientation repair changed vertices")
            if face_count_before != len(result.faces):
                raise RuntimeError("global orientation repair changed face count")
            orientation = {
                "method": "global",
                "status": "reverted" if orient_stats.get("reverted") else "applied",
                "flipped_faces": int(orient_stats.get("n_flip", 0)),
            }

        _ = result.vertex_normals
        return result, {
            "status": "applied",
            "faces_added": int(len(result.faces) - faces_before),
            "meshflow_faces_added": int(meshflow_added),
            "holes_before": int(holes_before),
            "holes_after": int(fill_stats["holes_after"]),
            "orientation": orientation,
        }
    except Exception as error:
        return original, {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def export_obj(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Export an OBJ while retaining unreferenced vertices and vertex normals."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_obj_keep_all_vertices(mesh, str(destination), write_normals=True)
    return destination


def export_normalized_glb(
    mesh: trimesh.Trimesh,
    path: str | Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Center a mesh, scale it into ``[-0.5, 0.5]``, and export GLB."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        return None, None

    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (upper + lower) / 2.0
    centered = vertices - center
    extent = float(np.abs(centered).max())
    scale = 1.0 if extent < 1e-12 else 0.5 / extent
    normalized = trimesh.Trimesh(
        vertices=(centered * scale).astype(np.float32),
        faces=faces,
        process=False,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.export(str(destination))
    transform = {
        "center": [float(value) for value in center],
        "scale": float(scale),
        "range": [-0.5, 0.5],
    }
    return destination, transform
