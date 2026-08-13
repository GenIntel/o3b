"""Camera / mesh math helpers for HouseCorr3DFrame indexing.

The camera part mirrors the logic from
od3d.od3d_datasets.omni6dpose.dataset.extract_meta but using only torch /
standard Python — no od3d dependency required.

The second half (:class:`ObjMeshAligner`) reconciles the mesh PAM ships for an
object id with the pose annotation in ``*_meta.json``.  The two disagree in two
ways, and both break every mesh-derived modality (cam_bbox3d, fo_mask_amodal,
obj_kpts2d_mask, obj_size) while leaving the mask-derived ones (cam_bbox2d)
correct:

1. **Units.** ``meta.scale`` converts the *original* scan into metres.  The PAM
   copies of the ROPE objects are already stored in metres, so applying it a
   second time shrinks them by 1000×; the SOPE meshes really are in millimetres
   and do need it.  The id prefix is no guide — SOPE also references ``real-*``
   objects whose meshes are in millimetres, and those id sets are disjoint from
   ROPE's.  ``meta.bbox_side_len`` is the metric extent in every case, so
   ``max(bbox_side_len) / max(mesh_extent)`` is the scale that holds for both;
   ``HouseCorr3D._obj_scale_from_row`` derives exactly that at load time.

2. **Orientation.** A few instances (e.g. ``real-bread_002``) ship a mesh whose
   axes are a permutation of the frame the pose annotation refers to, so the
   mesh projects nowhere near its own mask whatever the scale.  The fix is an
   axis-aligned rotation of the mesh, a property of the *object* rather than of
   the frame — hence resolved once at index time and written to the
   ``obj_tform4x4_obj_mesh`` column.

The rotation is chosen by mask agreement, not by extents alone: extents are
invariant under sign flips, and an instance whose annotated box merely
disagrees a little with its scan (``real-teddy_bear_001``) must keep identity.
So all 24 axis-aligned rotations are scored by projecting the mesh onto the
object's mask, and one is stored only when it clearly beats identity.

Only the *pose* is corrected — object space (NCDS verts, keypoints,
obj_bbox3d) stays in the mesh's own frame, since cross-instance correspondence
is defined by the per-object keypoint annotations, which live in that frame too.
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_cam_intr4x4(
    intrinsics: dict,
    rgb_path: Optional[Path] = None,
    img_size: Optional[tuple] = None,
) -> list:
    """Build 4×4 intrinsic matrix, adjusting for any image downsampling.

    Omni6DPose stores intrinsics at the sensor resolution, but the
    saved color PNG may have been downsampled.  We correct fx/fy/cx/cy
    by the actual downsample ratio.

    Pass ``img_size=(width, height)`` to skip file I/O (e.g. when the
    same dimensions are shared across all frames in a scene).
    """
    fx = float(intrinsics.get("fx", 0))
    fy = float(intrinsics.get("fy", 0))
    cx = float(intrinsics.get("cx", 0))
    cy = float(intrinsics.get("cy", 0))
    meta_h = float(intrinsics.get("height", 0))
    meta_w = float(intrinsics.get("width", 0))

    # correct for downsampling if the PNG was saved at a lower resolution
    if meta_h > 0 and meta_w > 0:
        try:
            if img_size is not None:
                actual_w, actual_h = img_size
            elif rgb_path is not None and rgb_path.exists():
                actual_w, actual_h = _png_size(rgb_path)
            else:
                actual_w, actual_h = None, None
            if actual_w and actual_h:
                ds_h = meta_h / actual_h
                ds_w = meta_w / actual_w
                fx /= ds_w;  cx /= ds_w
                fy /= ds_h;  cy /= ds_h
        except Exception:
            pass

    K = [
        [fx,  0.0, cx,  0.0],
        [0.0, fy,  cy,  0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return K


def build_cam_tform4x4_obj(cam_meta: dict) -> list:
    """world→camera transform from camera quaternion/translation."""
    q = cam_meta.get("quaternion", [1, 0, 0, 0])   # wxyz
    t = cam_meta.get("translation", [0, 0, 0])
    world_tform4x4_cam = _tform4x4_from_quat_wxyz_and_transl(q, t)
    cam_tform4x4_world = _inv_tform4x4(world_tform4x4_cam)
    return cam_tform4x4_world.tolist()


def build_obj_cam_tform(obj: dict) -> list:
    """camera→object transform from per-object quaternion_wxyz/translation fields."""
    q = obj.get("quaternion_wxyz", [1, 0, 0, 0])
    t = obj.get("translation", [0, 0, 0])
    # Omni6DPose stores cam_tform4x4_obj directly as (q, t) in camera space
    cam_tform4x4_obj = _tform4x4_from_quat_wxyz_and_transl(q, t)

    # apply isotropic scale (mesh units → metres) to the rotation block
    scale = obj.get("meta", {}).get("scale", None)
    if scale is not None:
        s = float(scale[0]) if isinstance(scale, (list, tuple)) else float(scale)
        cam_tform4x4_obj[:3, :3] *= s

    return cam_tform4x4_obj.tolist()


# ── private math helpers ─────────────────────────────────────────────────────

def _tform4x4_from_quat_wxyz_and_transl(q, t) -> torch.Tensor:
    """Build a 4×4 SE(3) matrix from a wxyz quaternion and a 3-vector translation."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

    n = w*w + x*x + y*y + z*z
    if n < 1e-10:
        R = torch.eye(3)
    else:
        s = 2.0 / n
        R = torch.tensor([
            [1 - s*(y*y + z*z),     s*(x*y - w*z),     s*(x*z + w*y)],
            [    s*(x*y + w*z), 1 - s*(x*x + z*z),     s*(y*z - w*x)],
            [    s*(x*z - w*y),     s*(y*z + w*x), 1 - s*(x*x + y*y)],
        ], dtype=torch.float64)

    M = torch.eye(4, dtype=torch.float64)
    M[:3, :3] = R
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M.float()


def _inv_tform4x4(T: torch.Tensor) -> torch.Tensor:
    """Invert an SE(3) 4×4 matrix."""
    R  = T[:3, :3]
    tr = T[:3,  3]
    Rt = R.T
    M  = torch.eye(4, dtype=T.dtype)
    M[:3, :3] = Rt
    M[:3,  3] = -(Rt @ tr)
    return M


def _png_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of a PNG by reading only the 24-byte file header.

    Much faster than loading the full image, especially over NFS.
    PNG spec: bytes 16-19 = width, bytes 20-23 = height (big-endian uint32).
    """
    import struct
    with open(path, "rb") as f:
        f.seek(16)
        data = f.read(8)
    w, h = struct.unpack(">II", data)
    return w, h


# ── mesh ↔ annotation alignment (see module docstring) ───────────────────────

_MESH_EXTS = (".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx")

#: relative tolerance under which the mesh extent and ``meta.bbox_side_len``
#: count as the same box — a scan and its annotated box routinely differ by a
#: couple of percent, and ``real-handbag_002`` is 2.6 % off on one axis.
EXTENT_RTOL = 0.05

#: a rotation must reach this mask IoU, and beat identity by ``MIN_IOU_GAIN``,
#: before it is stored.  Measured gains are unambiguous: real-bread_002 goes
#: 0.21 → 0.89, while real-teddy_bear_001 (annotation noise, not a rotation)
#: gains only 0.01 and so keeps identity.
MIN_IOU_ABS = 0.30
MIN_IOU_GAIN = 0.05

#: mask pixels a frame needs before it is worth scoring, and how many such
#: frames are averaged before an object's rotation is fixed.
MIN_MASK_PX = 200
SCORE_FRAMES = 3

#: vertices subsampled per mesh for scoring (24 rotations × this many points).
SCORE_VERTS = 4000


def axis_rotations() -> "list[np.ndarray]":
    """The 24 proper rotations that map the coordinate axes onto each other."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if abs(np.linalg.det(R) - 1.0) < 1e-6:
                out.append(R)
    return out


def find_mesh_file(mesh_dir: Path) -> Optional[Path]:
    """First mesh file directly inside *mesh_dir* (same order as o3b.io._load_mesh)."""
    if mesh_dir.is_file():
        return mesh_dir if mesh_dir.suffix.lower() in _MESH_EXTS else None
    if not mesh_dir.is_dir():
        return None
    for ext in _MESH_EXTS:
        hits = sorted(mesh_dir.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def read_mesh_verts(path: Path) -> "Optional[np.ndarray]":
    """(N, 3) vertices of *path*.

    ``.obj`` is parsed directly — indexing touches every object once and only
    needs positions, so skipping trimesh's faces/materials/texture handling is
    worth the few lines.
    """
    try:
        if path.suffix.lower() == ".obj":
            xyz = []
            with open(path, "r", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("v "):
                        p = line.split()
                        xyz.append((float(p[1]), float(p[2]), float(p[3])))
            return np.asarray(xyz, dtype=np.float64) if xyz else None
        import trimesh
        return np.asarray(trimesh.load(path, force="mesh").vertices, dtype=np.float64)
    except Exception as exc:
        logger.warning("could not read vertices of %s (%s)", path, exc)
        return None


def _mask_iou(pts_uv: "np.ndarray", mask: "np.ndarray") -> float:
    """IoU between a dilated splat of projected points and a filled mask."""
    import cv2
    pred = np.zeros(mask.shape, dtype=np.uint8)
    pred[pts_uv[:, 1], pts_uv[:, 0]] = 1
    # the projection is a point cloud, the mask a filled region: dilate so the
    # two are comparable without paying for a rasteriser
    pred = cv2.dilate(pred, np.ones((5, 5), np.uint8)).astype(bool)
    union = int((pred | mask).sum())
    return float((pred & mask).sum()) / union if union else 0.0


def score_rotations(
    verts: "np.ndarray",             # (N, 3) mesh vertices, mesh units
    scale: float,                    # mesh units → metres
    cam_intr4x4: "np.ndarray",       # (4, 4)
    cam_tform4x4_obj: "np.ndarray",  # (4, 4) raw annotation pose (CV, +Z forward)
    mask: "np.ndarray",              # (H, W) bool object mask
    rotations: "list[np.ndarray]",
) -> "np.ndarray":
    """Mask IoU of the mesh projected under each candidate rotation.

    Works in the raw annotation conventions (CV camera, unrotated object frame),
    so it is independent of ``cam_tform4x4_cam_raw`` / ``obj_gl_tform4x4_obj_raw``
    — those are applied identically to every candidate and cancel out.
    """
    H, W = mask.shape
    fx, fy = float(cam_intr4x4[0, 0]), float(cam_intr4x4[1, 1])
    cx, cy = float(cam_intr4x4[0, 2]), float(cam_intr4x4[1, 2])
    # the raw pose carries meta.scale in its rotation block; strip it so the
    # derived scale is the only one applied
    U, _, Vh = np.linalg.svd(cam_tform4x4_obj[:3, :3])
    R_cam, t_cam = U @ Vh, cam_tform4x4_obj[:3, 3]

    scores = np.zeros(len(rotations))
    for i, R in enumerate(rotations):
        v_cam = (R_cam @ (scale * (R @ verts.T))).T + t_cam
        z = v_cam[:, 2]
        keep = z > 1e-6
        if not keep.any():
            continue
        u = np.rint(cx + fx * v_cam[keep, 0] / z[keep]).astype(np.int64)
        v = np.rint(cy + fy * v_cam[keep, 1] / z[keep]).astype(np.int64)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inb.any():
            continue
        scores[i] = _mask_iou(np.stack([u[inb], v[inb]], axis=1), mask)
    return scores


class ObjMeshAligner:
    """Resolves each object's mesh→annotation rotation once, and caches it.

    :meth:`correction` returns the 4×4 to store in
    ``frames.obj_tform4x4_obj_mesh``, or None when the mesh already agrees with
    the annotation — the overwhelmingly common case, decided from extents alone
    without touching a mask.  *frame_ctx* is only called for the handful of
    objects that do disagree, so indexing keeps reading nothing but meta.json
    for everything else.

    Reading each indexed object's vertices costs ~0.14 s (≈11 min if a single
    index run covers all 5002 PAM objects; far less for the usual per-category
    run), so the extents are cached to *cache_path* and later index runs reuse
    them.  Delete that file to re-measure after the meshes change.
    """

    def __init__(self, path_object_meshes: Path, cache_path: Optional[Path] = None):
        self.path_object_meshes = Path(path_object_meshes)
        self.cache_path = Path(cache_path) if cache_path else None
        self._rotations = axis_rotations()
        self._entries: dict = {}        # oid → {extent, corr, iou, mismatch}
        self._pending: dict = {}        # oid → summed IoU per rotation
        self._pending_n: dict = {}
        self._dirty = False
        self._load_cache()

    # ── cache ────────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                self._entries = json.loads(self.cache_path.read_text())
            except Exception as exc:
                logger.warning("ignoring unreadable alignment cache %s (%s)",
                               self.cache_path, exc)
                self._entries = {}

    def save_cache(self) -> None:
        if not (self.cache_path and self._dirty):
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._entries, indent=1, sort_keys=True))
            self._dirty = False
        except Exception as exc:
            logger.warning("could not write alignment cache %s (%s)", self.cache_path, exc)

    @property
    def resolved_corrections(self) -> dict:
        """oid → 4×4, for the objects that ended up needing a rotation.

        A rotation is only decided once several frames of the object have been
        scored, so rows written before that (and rows an earlier index run
        wrote) still carry NULL — the caller backfills them from this.
        """
        return {oid: e["corr"] for oid, e in self._entries.items() if e.get("corr")}

    @property
    def n_corrected(self) -> int:
        return sum(1 for e in self._entries.values() if e.get("corr"))

    @property
    def n_unresolved(self) -> int:
        return sum(1 for e in self._entries.values()
                   if e.get("mismatch") and not e.get("corr"))

    # ── mesh geometry ────────────────────────────────────────────────────────

    def _mesh_extent(self, oid: str) -> "Optional[np.ndarray]":
        entry = self._entries.get(oid)
        if entry is not None and "extent" in entry:
            e = entry["extent"]
            return np.asarray(e, dtype=np.float64) if e else None
        path = find_mesh_file(self.path_object_meshes / oid)
        verts = read_mesh_verts(path) if path else None
        extent = (verts.max(0) - verts.min(0)) if verts is not None and len(verts) else None
        self._entries.setdefault(oid, {})["extent"] = (
            extent.tolist() if extent is not None else None
        )
        self._dirty = True
        return extent

    def _mesh_verts_sample(self, oid: str) -> "Optional[np.ndarray]":
        path = find_mesh_file(self.path_object_meshes / oid)
        verts = read_mesh_verts(path) if path else None
        if verts is None or not len(verts):
            return None
        if len(verts) > SCORE_VERTS:
            idx = np.random.RandomState(0).choice(len(verts), SCORE_VERTS, replace=False)
            verts = verts[idx]
        return verts

    # ── resolution ───────────────────────────────────────────────────────────

    def correction(
        self,
        oid: str,
        bbox_side_len,
        frame_ctx: "Callable[[], Optional[tuple]]",
    ) -> Optional[list]:
        """4×4 mesh→annotation rotation for *oid*, or None if none is needed.

        *frame_ctx* must return ``(cam_intr4x4, cam_tform4x4_obj, mask)`` as
        numpy arrays for the current frame, or None when the frame cannot be
        scored (no mask, unreadable EXR, …).
        """
        entry = self._entries.get(oid)
        if entry is not None and "corr" in entry:
            return entry["corr"]

        extent = self._mesh_extent(oid)
        if extent is None or bbox_side_len is None:
            return None
        bbox = np.asarray(bbox_side_len, dtype=np.float64)
        if bbox.shape != (3,) or not np.all(extent > 0):
            return None

        if np.allclose(extent, bbox, rtol=EXTENT_RTOL):
            self._entries.setdefault(oid, {})["corr"] = None
            self._dirty = True
            return None

        # Extents disagree — a rotation may explain it.  Decide from the mask.
        self._entries.setdefault(oid, {})["mismatch"] = True
        return self._resolve_by_mask(oid, extent, bbox, frame_ctx)

    def _resolve_by_mask(self, oid, extent, bbox, frame_ctx) -> Optional[list]:
        ctx = frame_ctx()
        if ctx is None:
            return None  # unscoreable frame — retry on a later one, cache nothing
        cam_intr4x4, cam_tform4x4_obj, mask = ctx
        if mask is None or int(mask.sum()) < MIN_MASK_PX:
            return None
        verts = self._mesh_verts_sample(oid)
        if verts is None:
            self._entries.setdefault(oid, {})["corr"] = None
            self._dirty = True
            return None

        scale = float(bbox.max() / extent.max())
        scores = score_rotations(verts, scale, cam_intr4x4, cam_tform4x4_obj,
                                 mask, self._rotations)
        total = self._pending.get(oid)
        self._pending[oid] = scores if total is None else total + scores
        self._pending_n[oid] = self._pending_n.get(oid, 0) + 1
        if self._pending_n[oid] < SCORE_FRAMES:
            return None  # keep averaging over further frames of this object

        mean = self._pending.pop(oid) / self._pending_n.pop(oid)
        best = int(np.argmax(mean))
        # axis_rotations() enumerates the identity permutation with all-positive
        # signs first, so rotations[0] is the identity
        iou_best, iou_id = float(mean[best]), float(mean[0])
        take = iou_best >= MIN_IOU_ABS and iou_best >= iou_id + MIN_IOU_GAIN

        corr = None
        if take:
            corr4x4 = np.eye(4)
            corr4x4[:3, :3] = self._rotations[best]
            corr = corr4x4.tolist()
            logger.warning(
                "%s: mesh extent %s disagrees with annotated %s — storing an "
                "axis-aligned mesh→annotation rotation (mask IoU %.2f vs %.2f "
                "for identity)", oid, np.round(extent, 4).tolist(),
                np.round(bbox, 4).tolist(), iou_best, iou_id)
        else:
            logger.warning(
                "%s: mesh extent %s disagrees with annotated %s but no rotation "
                "improves the mask fit (best IoU %.2f vs %.2f for identity) — "
                "leaving the pose uncorrected", oid, np.round(extent, 4).tolist(),
                np.round(bbox, 4).tolist(), iou_best, iou_id)

        self._entries.setdefault(oid, {}).update(
            corr=corr, iou=[round(iou_best, 4), round(iou_id, 4)])
        self._dirty = True
        return corr
