"""``pcd<q>_alpha<n>`` meshes: od3d's ``alpha<n>`` recipe on a trimmed point cloud.

    o3b dataset build-mesh -d every9d_v7 --mesh-type pcd99_alpha500 -s every9d_v7_test_verify100

``alpha500`` is an alpha shape of the whole masked point cloud
(``pcl/meta_mask/meta/<category>/<sequence>/pcl.ply``), so every point the masks
let through (table, hand, floaters) ends up inside the mesh, and the loader's box
is the mesh's bounds. ``pcd99_alpha500`` first keeps the points inside the central
99 % of the cloud along each canonical axis (0.5 % cut from either end of X, Y
and Z, so at most 3 % of the points go) and then builds the same mesh.

The trim needs axes, and the box is axis-aligned in the canonical frame, so the
axes are those of the config's ``tform_obj_type``. The mesh itself is written in
the raw frame like every other mesh type, at
``mesh/<mesh_type>/meta_mask/meta/<category>/<sequence>/mesh.ply``; that path
does not name the ``tform_obj_type`` whose axes cut it.

``alpha_mesh`` follows ``od3d.od3d_datasets.sequence.preprocess_mesh`` (branch
``alpha``) step by step, so that ``pcd100_alpha500`` is ``alpha500`` up to its
random subsample: the alpha comes from the nearest-neighbour spacing of at most
10,000 points, grows until the shape is watertight and vertex-manifold, and the
result is decimated to at most ``n`` vertices.
"""
from __future__ import annotations

import logging
import re
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^pcd([0-9]+(?:\.[0-9]+)?)_alpha([0-9]+)$")
_PCL_SFM = ("meta_mask", "meta")
_SPACING_POINTS = 10000      # od3d's cap for the pairwise distances ("11 GB")
_MAX_ROUNDS = 60             # od3d loops without a bound; alpha grows 1.3x a round


def parse_mesh_type(mesh_type: str) -> Optional[tuple]:
    """``pcd99_alpha500`` -> ``(99.0, 500)``; None for any other mesh type."""
    m = _NAME.match(mesh_type)
    if not m:
        return None
    q = float(m.group(1))
    if not 0.0 < q <= 100.0:
        raise ValueError(f"{mesh_type}: quantile {q} outside (0, 100]")
    return q, int(m.group(2))


def trim_mask(points: np.ndarray, tform_obj, q: float) -> np.ndarray:
    """Points inside the central ``q`` % of the cloud along each canonical axis."""
    T = np.asarray(tform_obj, dtype=np.float64)
    canonical = np.asarray(points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    lo, hi = np.percentile(canonical, [(100.0 - q) / 2, 100.0 - (100.0 - q) / 2], axis=0)
    return ((canonical >= lo) & (canonical <= hi)).all(axis=1)


def alpha_mesh(points: np.ndarray, vertices_count: int, seed: int = 0) -> Optional[tuple]:
    """od3d's ``alpha<vertices_count>`` mesh of ``points``: ``(verts, faces)`` or None."""
    import open3d
    import torch

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return None
    pts = torch.from_numpy(points).float()
    keep = torch.randperm(len(pts), generator=torch.Generator().manual_seed(seed))[:_SPACING_POINTS]
    sub = pts[keep]
    count = len(sub)
    dist = torch.cdist(sub[None], sub[None])
    if dist.max() < 1e-3:
        return None
    positive = dist > 0.0
    dist[~positive] = dist[positive].min() if positive.any() else 1.0
    alpha = float(dist.quantile(dim=-1, q=max(0.01, 3.0 / len(sub))).mean())
    del dist, positive

    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(points)
    mesh, out = None, None
    for _ in range(_MAX_ROUNDS):
        try:
            while (mesh is None or len(mesh.triangles) < 4 or len(mesh.vertices) < 4
                   or not mesh.is_watertight()):
                if mesh is not None:
                    count = max(int(count * 0.95), 4)
                    alpha *= 1.1
                    pcl = pcl.farthest_point_down_sample(count)
                mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha)
        except Exception as e:
            logger.warning(f"alpha {alpha} failed with {e}")
        if mesh is not None:
            mesh = mesh.remove_unreferenced_vertices()
            faces_count = vertices_count * 2
            out = mesh
            while len(out.vertices) > vertices_count:
                faces_count = int(faces_count * 0.9)
                out = mesh.simplify_quadric_decimation(target_number_of_triangles=faces_count)
        alpha *= 1.3
        if mesh is not None and mesh.is_watertight() and mesh.is_vertex_manifold():
            break
    else:
        return None
    if out is None or len(out.triangles) < 4:
        return None
    return np.asarray(out.vertices, dtype=np.float32), np.asarray(out.triangles, dtype=np.int64)


def path_pcl(path_preprocess, sequence_name: str) -> Path:
    return Path(path_preprocess) / "pcl" / _PCL_SFM[0] / _PCL_SFM[1] / sequence_name / "pcl.ply"


def path_mesh(path_preprocess, mesh_type: str, sequence_name: str) -> Path:
    return Path(path_preprocess) / "mesh" / mesh_type / _PCL_SFM[0] / _PCL_SFM[1] / sequence_name / "mesh.ply"


def build_mesh(path_preprocess, mesh_type: str, sequence_name: str, tform_obj,
               override: bool = False) -> Optional[dict]:
    """Build one sequence's mesh. Returns its stats, or None if it could not be built."""
    import trimesh

    q, vertices_count = parse_mesh_type(mesh_type)
    out = path_mesh(path_preprocess, mesh_type, sequence_name)
    if out.exists() and not override:
        return {"sequence_name": sequence_name, "cached": True}
    src = path_pcl(path_preprocess, sequence_name)
    if not src.exists():
        logger.warning(f"{sequence_name}: no point cloud at {src}")
        return None
    points = np.asarray(trimesh.load(src, process=False).vertices, dtype=np.float64)
    mask = trim_mask(points, tform_obj, q)
    built = alpha_mesh(points[mask], vertices_count, seed=zlib.crc32(sequence_name.encode()))
    if built is None:
        logger.warning(f"{sequence_name}: no watertight alpha shape from {int(mask.sum())} points")
        return None
    verts, faces = built
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("mesh.tmp.ply")
    trimesh.Trimesh(verts, faces, process=False).export(tmp)
    tmp.rename(out)
    return {"sequence_name": sequence_name, "cached": False, "points": len(points),
            "points_kept": int(mask.sum()), "verts": len(verts), "faces": len(faces)}


def run_build(cfg, mesh_type: str, ids, override: bool = False, workers: int = 4) -> list:
    """Build ``mesh_type`` for every id. Returns the ids that could not be built."""
    from concurrent.futures import ThreadPoolExecutor

    from o3b.dataset import tform_obj_store

    if parse_mesh_type(mesh_type) is None:
        raise ValueError(f"{mesh_type}: expected pcd<quantile>_alpha<vertices>, e.g. pcd99_alpha500")
    if cfg.tform_obj_type == "raw":
        raise ValueError("the trim runs along canonical axes: use a config with a canonical "
                         "tform_obj_type (e.g. -d every9d_v7), not raw")
    table = tform_obj_store.load_all(cfg.path_preprocess, cfg.tform_obj_type)
    if table is None:
        raise FileNotFoundError(f"no tform_obj/{cfg.tform_obj_type}.db under {cfg.path_preprocess}")
    ids = sorted(ids)

    def one(sequence_name):
        if sequence_name not in table:
            logger.warning(f"{sequence_name}: no tform_obj of type {cfg.tform_obj_type}")
            return sequence_name, None
        try:
            return sequence_name, build_mesh(cfg.path_preprocess, mesh_type, sequence_name,
                                             table[sequence_name], override)
        except Exception as e:      # one bad cloud must not lose a long build
            logger.warning(f"{sequence_name}: {e}")
            return sequence_name, None

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (name, stats) in enumerate(ex.map(one, ids)):
            if stats is None:
                failed.append(name)
                note = "FAILED"
            elif stats["cached"]:
                note = "exists"
            else:
                note = (f"{stats['points_kept']}/{stats['points']} points, "
                        f"{stats['verts']} verts, {stats['faces']} faces")
            print(f"[{i + 1}/{len(ids)}] {name}: {note}", flush=True)
    return failed
