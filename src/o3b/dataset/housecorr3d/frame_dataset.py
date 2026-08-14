"""Helper functions for HouseCorr3D frame-object data.

Provides:
  - _index_scene()           : insert one scene's frame-object rows into frames.db
  - modality loaders         : _load_image_tensor, _load_depth_tensor, _load_mask_tensor
  - viser visualization      : _visualize_frame_objects_viser and helpers

The scene builders those browsers call are dataset-agnostic (they take a
FrameObject, not a HouseCorr3D row) and generic o3b modules depend on them, so
they live in :mod:`o3b.data.viz_viser`; they are re-imported below under their
original names.
"""
from __future__ import annotations

import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import torch

from o3b.data.viz_viser import (  # noqa: F401  (re-exported for callers)
    _add_axes_to_scene,
    _add_depth_pc_to_scene,
    _add_frame_object_to_scene,
    _add_frustum_to_scene,
    _add_rgb_image_to_scene,
    _build_frame_sidebar_imgs,
    _corners8_to_size_tform,
    _depth_to_pts3d_cam,
    _fo_to_obj_centric,
    _kpts_index_colors,
    _mask_dt_to_sidebar_img,
    _mask_to_sidebar_img,
    _rot3x3_to_wxyz,
)

logger = logging.getLogger(__name__)

_exr_warned = False


def _warn_exr_once(path, err) -> None:
    """Report the first EXR read failure in this process, then stay quiet.

    A silent None here is how whole categories of frames disappear from a shard
    build (mask None → cam_bbox2d None → item dropped), and the cause is
    normally environmental rather than per-file: a cv2 built without the
    OpenEXR codec, or the codec gate latched off before o3b was imported (see
    the OPENCV_IO_ENABLE_OPENEXR note in o3b/__init__.py).  So the first failure
    carries the environment detail and the rest go to debug.
    """
    global _exr_warned
    if _exr_warned:
        logger.debug("EXR read failed: %s (%s)", path, err)
        return
    _exr_warned = True
    detail = ""
    try:
        import cv2
        in_build = any("OpenEXR" in l for l in cv2.getBuildInformation().splitlines())
        detail = (f" [cv2 {cv2.__version__}, OpenEXR in build: {in_build}, "
                  f"OPENCV_IO_ENABLE_OPENEXR={os.environ.get('OPENCV_IO_ENABLE_OPENEXR')!r}]")
    except Exception:
        pass
    logger.warning("EXR read failed: %s (%s)%s — further failures logged at debug level",
                   path, err, detail)


def _visualize_frame_object_pairs_viser(
    dataset, n: Optional[int] = None, debug: bool = False, obj_centric: bool = False,
) -> None:
    """Interactive viser browser for HouseCorr3D frame-object *pairs*.

    Each pair (query=src, target=trgt) is shown side-by-side in the 3-D scene;
    keypoint spheres are colored by index so the same color marks corresponding
    keypoints across the two frames.  A sidebar shows each frame's rgb/kpts.
    """
    import time

    try:
        import viser  # noqa: F401
    except ImportError:
        print("Install viser: pip install viser")
        return

    from o3b.data.viz import make_viser_server

    server = make_viser_server()
    server.scene.add_light_ambient("/ambient", intensity=3.0)

    if n is None:
        n = len(dataset)
    if n == 0:
        print("No frame-object pairs found matching the current config filters.")
        return

    idx = [0]
    handles: list = []
    _img_handle = [None]
    _mod_dd = [None]
    _mod_imgs = [{}]

    def _clear() -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _update_sidebar_img() -> None:
        if _mod_dd[0] is None or _img_handle[0] is None:
            return
        mod = _mod_dd[0].value
        if mod in _mod_imgs[0]:
            _img_handle[0].image = _mod_imgs[0][mod]

    def _load(i: int) -> None:
        _clear()
        # dataset[i] reads the sharded cache when one is loaded and the raw
        # frames.db rows otherwise, applying the transform either way
        pair = dataset[i]
        src, trgt = pair.src_object, pair.trgt_object

        # shared per-index keypoint colors so src/trgt correspond
        K = src.obj_kpts3d.shape[0] if src.obj_kpts3d is not None else 0
        kpt_colors = _kpts_index_colors(K) if K else None

        # offset target along +x by ~1.5 * its size so the two don't overlap
        offset_x = 1.5 * max(float(src.obj_size or 0.3), float(trgt.obj_size or 0.3)) + 0.3

        handles.extend(src.viz(
            server=server, node_prefix="/src", offset_x=0.0, obj_centric=obj_centric, kpt_colors=kpt_colors))
        handles.extend(trgt.viz(
            server=server, node_prefix="/trgt", offset_x=offset_x, obj_centric=obj_centric, kpt_colors=kpt_colors))

        # sidebar: src + trgt modality images
        src_imgs  = {f"src/{k}": v  for k, v in _build_frame_sidebar_imgs(src).items()}
        trgt_imgs = {f"trgt/{k}": v for k, v in _build_frame_sidebar_imgs(trgt).items()}
        imgs = {**src_imgs, **trgt_imgs}
        _mod_imgs[0] = imgs
        if imgs:
            keys = list(imgs.keys())
            if _mod_dd[0] is None:
                _mod_dd[0] = server.gui.add_dropdown("Modality", options=keys, initial_value=keys[0])

                @_mod_dd[0].on_update
                def _(_e):
                    _update_sidebar_img()
            else:
                _mod_dd[0].options = keys
                if _mod_dd[0].value not in keys:
                    _mod_dd[0].value = keys[0]
            mod = _mod_dd[0].value if _mod_dd[0].value in imgs else keys[0]
            if _img_handle[0] is None:
                _img_handle[0] = server.gui.add_image(imgs[mod], label="Frame")
            else:
                _img_handle[0].image = imgs[mod]

        label.value = (f"[{i + 1}/{n}]  {src.object_id}  <-->  {trgt.object_id}")
        print(f"  [{i + 1}/{n}] {src.object_id}  <-->  {trgt.object_id}")

    with server.gui.add_folder("Navigation"):
        label    = server.gui.add_text("Item", initial_value="loading…")
        btn_prev = server.gui.add_button("← Prev")
        btn_next = server.gui.add_button("Next →")

    @btn_prev.on_click
    def _(_):
        idx[0] = (idx[0] - 1) % n
        _load(idx[0])

    @btn_next.on_click
    def _(_):
        idx[0] = (idx[0] + 1) % n
        _load(idx[0])

    _load(0)
    print(f"\nViser running at http://localhost:{server.get_port()}")
    print("Use Prev / Next in the panel to browse. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")


def _visualize_frame_objects_viser(
    dataset, n: Optional[int] = None, debug: bool = False, obj_centric: bool = False,
) -> None:
    """Interactive viser browser for HouseCorr3D frame-object items.

    obj_centric=False (default): camera-centric — camera at origin, object transformed.
    obj_centric=True:            object-centric — object mesh at origin, camera placed
                                 at inv(cam_tform4x4_obj_ncds) in NCDS/world space.
    """
    import time

    try:
        import viser
    except ImportError:
        print("Install viser: pip install viser")
        return

    import numpy as np

    from o3b.data.viz import make_viser_server

    server = make_viser_server()
    server.scene.add_light_ambient("/ambient", intensity=3.0)

    if n is None:
        n = len(dataset)
    idx     = [0]
    handles: list = []
    _img_handle = [None]   # GuiImageHandle for the sidebar modality view
    _mod_dd     = [None]   # GuiDropdown handle
    _mod_imgs   = [{}]     # current dict[str, uint8 HxWx3]

    def _clear() -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _update_sidebar_img() -> None:
        if _mod_dd[0] is None or _img_handle[0] is None:
            return
        mod = _mod_dd[0].value
        imgs = _mod_imgs[0]
        if mod in imgs:
            _img_handle[0].image = imgs[mod]

    def _load(i: int) -> None:
        _clear()
        # dataset[i] reads the sharded cache when one is loaded and the raw
        # frames.db rows otherwise, applying the transform either way
        fo = dataset[i]

        # 3-D scene (mesh, camera frustum, rgb panel, depth pc, axes, keypoints)
        handles.extend(fo.viz(server=server, obj_centric=obj_centric))

        # ── sidebar modality images ───────────────────────────────────────────
        imgs = _build_frame_sidebar_imgs(fo)
        _mod_imgs[0] = imgs
        if imgs:
            keys = list(imgs.keys())
            if _mod_dd[0] is None:
                _mod_dd[0] = server.gui.add_dropdown(
                    "Modality", options=keys, initial_value=keys[0]
                )

                @_mod_dd[0].on_update
                def _(_e):
                    _update_sidebar_img()
            else:
                _mod_dd[0].options = keys
                if _mod_dd[0].value not in keys:
                    _mod_dd[0].value = keys[0]

            mod = _mod_dd[0].value if _mod_dd[0].value in imgs else keys[0]
            if _img_handle[0] is None:
                _img_handle[0] = server.gui.add_image(imgs[mod], label="Frame")
            else:
                _img_handle[0].image = imgs[mod]

        # from the item itself, so the label works for shard-backed items too
        # (frame_object_id is the frames.db frame_id)
        cat = fo.category if fo.category is not None else ""
        fid = fo.frame_object_id or str(i)
        label.value = f"[{i + 1}/{n}]  {fid}  cat={cat}"
        print(f"  [{i + 1}/{n}] {fid}  cat={cat}")

    with server.gui.add_folder("Navigation"):
        label    = server.gui.add_text("Item",   initial_value="loading…")
        btn_prev = server.gui.add_button("← Prev")
        btn_next = server.gui.add_button("Next →")

    @btn_prev.on_click
    def _(_):
        idx[0] = (idx[0] - 1) % n
        _load(idx[0])

    @btn_next.on_click
    def _(_):
        idx[0] = (idx[0] + 1) % n
        _load(idx[0])

    _load(0)
    print(f"\nViser running at http://localhost:{server.get_port()}")
    print("Use Prev / Next in the panel to browse. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")


# ── indexing helpers ──────────────────────────────────────────────────────────

def _align_frame_ctx(
    cam_intr4x4_list,
    cam_tform4x4_obj_list,
    mask_path: Optional[Path],
    mask_id: Optional[int],
) -> Optional[tuple]:
    """(cam_intr4x4, cam_tform4x4_obj, mask) for ObjMeshAligner, or None.

    Called only for objects whose mesh extent contradicts their annotated box,
    so the mask EXR read here stays off the normal indexing path (which reads
    meta.json and nothing else).
    """
    if (cam_intr4x4_list is None or cam_tform4x4_obj_list is None
            or mask_path is None or mask_id is None):
        return None
    mask = _load_mask_tensor(mask_path, int(mask_id))
    if mask is None:
        return None
    import numpy as np
    return (
        np.asarray(cam_intr4x4_list, dtype=np.float64),
        np.asarray(cam_tform4x4_obj_list, dtype=np.float64),
        mask.numpy(),
    )


def _index_scene(
    cur,
    scene_dir: Path,
    scene_name: str,
    split: str,
    data_type: str,
    path_raw: Path,
    kpts_preprocess: Path,
    limit: Optional[int] = None,
    filter_kpts: bool = False,
    categories: Optional[set] = None,
    cat_counts: Optional[dict] = None,
    aligner: "Optional[object]" = None,
) -> tuple[int, int]:
    """Insert frame-object rows for one scene. Returns (n_total, n_matching).

    n_matching counts rows that satisfy the load-time filter, so that *limit*
    reflects the number of actually-loadable items:
    - is_valid=1 is always required (matches the load-time query)
    - filter_kpts=True additionally requires has_kpts=1

    When *categories* is given, only those categories are indexed and *limit* is
    applied **per category** (tracked across scenes via the shared *cat_counts*
    dict); the scene stops once every requested category has reached its quota.
    Otherwise *limit* is a global cap and the scene stops once it is hit.

    *aligner* is a shared :class:`ObjMeshAligner` resolving the per-object
    mesh→annotation rotation stored in ``obj_tform4x4_obj_mesh``; it only reads
    a mask for objects whose mesh extent contradicts ``meta.bbox_side_len``.
    """
    from o3b.dataset.housecorr3d._frame_utils import (
        build_cam_intr4x4,
        build_cam_tform4x4_obj,
        build_obj_cam_tform,
        _png_size,
    )

    if cat_counts is None:
        cat_counts = {}

    def _all_categories_full() -> bool:
        return (categories is not None and limit is not None
                and all(cat_counts.get(c, 0) >= limit for c in categories))

    frame_ids_color = sorted(
        p.stem[: -len("_color")]
        for p in scene_dir.iterdir()
        if p.name.endswith("_color.png")
    )

    # Read image dimensions once for the scene (all frames share the same resolution).
    scene_img_size: tuple[int, int] | None = None
    for fid in frame_ids_color:
        p = scene_dir / f"{fid}_color.png"
        if p.exists():
            try:
                scene_img_size = _png_size(p)
            except Exception:
                pass
            break

    n       = 0  # rows_written  : rows inserted into frames.db
    n_match = 0  # rows_loadable : subset the loader will keep
    for frame_id_raw in frame_ids_color:
        meta_path = scene_dir / f"{frame_id_raw}_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        cam_meta   = meta.get("camera", {})
        intrinsics = cam_meta.get("intrinsics", {})

        try:
            cam_intr4x4_list   = build_cam_intr4x4(intrinsics, img_size=scene_img_size)
            cam_tform4x4_world = build_cam_tform4x4_obj(cam_meta)
        except Exception:
            cam_intr4x4_list   = None
            cam_tform4x4_world = None

        rgb_path   = scene_dir / f"{frame_id_raw}_color.png"
        depth_path = scene_dir / f"{frame_id_raw}_depth.exr"
        mask_path  = scene_dir / f"{frame_id_raw}_mask.exr"

        rgb_rpath   = str(rgb_path.relative_to(path_raw))   if rgb_path.exists()   else None
        depth_rpath = str(depth_path.relative_to(path_raw)) if depth_path.exists() else None
        mask_rpath  = str(mask_path.relative_to(path_raw))  if mask_path.exists()  else None

        for obj_idx, (obj_name, obj) in enumerate(meta.get("objects", {}).items()):
            is_valid  = int(bool(obj.get("is_valid", True)))
            category  = obj.get("meta", {}).get("class_name")
            object_id = obj.get("meta", {}).get("oid") or obj_name
            # Mask EXR pixel value = integer prefix of the object key (e.g. "5_mango_..." → 5),
            # NOT the sequential 'id' field (1, 2, 3...).
            try:
                mask_id = int(obj_name.split("_")[0])
            except (ValueError, IndexError):
                mask_id = None
            bbox_side_len = obj.get("meta", {}).get("bbox_side_len")  # [w, h, d] metres
            scale_raw = obj.get("meta", {}).get("scale", None)
            if isinstance(scale_raw, (list, tuple)):
                obj_scale = float(scale_raw[0])   # isotropic
            elif scale_raw is not None:
                obj_scale = float(scale_raw)
            else:
                obj_scale = 1.0
            has_kpts = 1 if (kpts_preprocess / object_id / "kpts3d.pt").exists() else 0

            matches = bool(is_valid and (not filter_kpts or has_kpts))
            if categories is not None:
                # only index requested categories, capped per category
                if category not in categories:
                    continue
                if limit is not None and matches and cat_counts.get(category, 0) >= limit:
                    continue

            # per-object cam_tform4x4_obj: uses the object's own quaternion/translation
            try:
                cam_tform4x4_obj_list = build_obj_cam_tform(obj)
            except Exception:
                cam_tform4x4_obj_list = cam_tform4x4_world

            # Rotation that carries the PAM mesh into the frame this pose
            # annotation refers to, for the few objects whose mesh axes are a
            # permutation of it (see ObjMeshAligner).  None for everything else.
            obj_tform4x4_obj_mesh = None
            if aligner is not None:
                obj_tform4x4_obj_mesh = aligner.correction(
                    object_id, bbox_side_len,
                    frame_ctx=partial(
                        _align_frame_ctx,
                        cam_intr4x4_list=cam_intr4x4_list,
                        cam_tform4x4_obj_list=cam_tform4x4_obj_list,
                        mask_path=mask_path if mask_rpath else None,
                        mask_id=mask_id,
                    ),
                )

            frame_id = f"{data_type}/{scene_name}/{frame_id_raw}/{obj_idx}"

            cur.execute(
                """
                INSERT OR IGNORE INTO frames
                    (frame_id, scene_name, object_idx, mask_id, split, data_type,
                     category, object_id,
                     rgb_path, depth_path, mask_path,
                     cam_intr4x4, cam_tform4x4_obj, obj_size3d,
                     obj_scale, obj_tform4x4_obj_mesh, has_kpts, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_id, scene_name, obj_idx, mask_id, split, data_type,
                    category, object_id,
                    rgb_rpath, depth_rpath, mask_rpath,
                    json.dumps(cam_intr4x4_list)      if cam_intr4x4_list      is not None else None,
                    json.dumps(cam_tform4x4_obj_list) if cam_tform4x4_obj_list is not None else None,
                    json.dumps(bbox_side_len)          if bbox_side_len         is not None else None,
                    obj_scale,
                    json.dumps(obj_tform4x4_obj_mesh) if obj_tform4x4_obj_mesh is not None else None,
                    has_kpts, is_valid,
                ),
            )
            n += 1
            # Match the load-time filter (always requires is_valid=1) so that
            # *limit* caps the number of actually-loadable rows.
            if matches:
                n_match += 1
                if categories is not None and category is not None:
                    cat_counts[category] = cat_counts.get(category, 0) + 1
            if categories is not None:
                if _all_categories_full():
                    return n, n_match
            elif limit is not None and n_match >= limit:
                return n, n_match
    return n, n_match


# ── keypoint 2-D visibility (occlusion-aware) ─────────────────────────────────

def render_scene_depth(
    meshes_cam: "list[tuple[torch.Tensor, torch.Tensor]]",
    cam_intr4x4: torch.Tensor,
    H: int,
    W: int,
) -> "Optional[torch.Tensor]":
    """Render a depth (z-)buffer of all object meshes in camera space.

    Args:
        meshes_cam: list of (verts_cam (N, 3), faces (F, 3)) already transformed
            into OpenGL camera space (-Z forward).
        cam_intr4x4: (4, 4) full-image intrinsics.
        H, W: render resolution (must match cam_intr4x4's image).

    Returns:
        (H, W) float32 depth in metres (0 = background / no hit), or None.
    """
    import numpy as np
    import trimesh
    from o3b.cv.visual.show import render_trimesh_to_tensor

    tms = []
    for verts, faces in meshes_cam:
        if verts is None or faces is None or verts.numel() == 0 or faces.numel() == 0:
            continue
        tms.append(trimesh.Trimesh(
            vertices=verts.detach().cpu().numpy().astype(np.float64),
            faces=faces.detach().cpu().numpy(),
            process=False,
        ))
    if not tms:
        return None
    scene = trimesh.util.concatenate(tms)
    # Verts are already in camera space → render with an identity extrinsic.
    _, depth = render_trimesh_to_tensor(
        scene, cam_intr4x4.float().cpu(), torch.eye(4), H=H, W=W,
    )
    return depth[0]  # (H, W)


def visible_vertices_from_render(
    target_verts_cam: torch.Tensor,  # (N, 3) target mesh verts in camera space
    depth_render: torch.Tensor,      # (H, W) rendered all-objects scene depth (metres)
    cam_intr4x4: torch.Tensor,       # (4, 4)
    H: int,
    W: int,
    obj_size: "Optional[float]" = None,
    abs_eps: float = 0.002,
    rel_eps: float = 0.02,
) -> torch.Tensor:
    """Determine which target mesh vertices are visible (front-most surface).

    A vertex is visible if it projects in front of the camera, inside the image,
    and is the front-most surface at its pixel in the all-objects render — i.e.
    nothing (another object or the object's own geometry) lies in front of it.
    Occlusion is resolved by the rasteriser's z-buffer in ``depth_render``; the
    small epsilon only absorbs render/projection discretisation, it is not a
    free depth-matching tolerance.

    Returns (N,) bool over the mesh vertices.
    """
    N = target_verts_cam.shape[0]
    vis = torch.zeros(N, dtype=torch.bool)
    if N == 0 or depth_render is None:
        return vis

    eps = max(abs_eps, rel_eps * float(obj_size)) if obj_size else abs_eps
    fx, fy = float(cam_intr4x4[0, 0]), float(cam_intr4x4[1, 1])
    cx, cy = float(cam_intr4x4[0, 2]), float(cam_intr4x4[1, 2])

    verts = target_verts_cam.float()
    X, Y, Z = verts[:, 0], verts[:, 1], verts[:, 2]
    zpos = (-Z).clamp(min=1e-6)                  # OpenGL: in front ⇒ Z < 0
    u = (cx + fx * X / zpos).round().long()
    v = (cy - fy * Y / zpos).round().long()
    inb = (Z < 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = inb.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return vis
    dr = depth_render[v[idx], u[idx]]
    # not occluded: rendered front surface is at the vertex (within eps), not closer
    front = (dr > 0) & ((zpos[idx] - dr) <= eps)
    vis[idx[front]] = True
    return vis


def kpts2d_mask_from_visible_verts(
    kpts_ncds: torch.Tensor,        # (K, 3) keypoints in NCDS space
    verts_ncds: torch.Tensor,       # (N, 3) mesh verts in NCDS space
    visible_verts: torch.Tensor,    # (N,) bool
    obj_kpts3d_mask: "Optional[torch.Tensor]" = None,  # (K,) bool annotation validity
    norm_size: float = 2.0,         # NCDS object size (obj_size_ncds)
    rel_radius: float = 0.05,       # visible vertex must be within rel_radius * norm_size
) -> torch.Tensor:
    """A keypoint is visible iff a visible vertex lies within ``rel_radius *
    norm_size`` of it (distance in NCDS space). Gated by ``obj_kpts3d_mask`` so
    only annotated keypoints can be marked visible.

    Returns (K,) bool.
    """
    K = kpts_ncds.shape[0]
    out = torch.zeros(K, dtype=torch.bool)
    if visible_verts is not None and visible_verts.any():
        radius = rel_radius * float(norm_size)
        vis_pts = verts_ncds.float()[visible_verts.bool()]   # (M, 3)
        nearest = torch.cdist(kpts_ncds.float(), vis_pts).min(dim=1).values  # (K,)
        out = nearest < radius
    if obj_kpts3d_mask is not None:
        out = out & obj_kpts3d_mask.bool().cpu()
    return out


# ── modality loaders ─────────────────────────────────────────────────────────

def _load_image_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load PNG/JPEG → (3, H, W) float32 in [0, 1]."""
    if not path.exists():
        return None
    try:
        import torchvision.io as tio
        img = tio.read_image(str(path))          # (C, H, W) uint8
        return img.float() / 255.0
    except Exception:
        return None


def _load_depth_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load depth EXR → (H, W) float32 in metres."""
    if not path.exists():
        return None
    try:
        import cv2, numpy as np
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            _warn_exr_once(path, "cv2.imread returned None")
            return None
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return torch.from_numpy(arr.astype(np.float32))
    except Exception as e:
        _warn_exr_once(path, e)
        return None


def _load_mask_tensor(path: Path, mask_id: int) -> Optional[torch.Tensor]:
    """Load scene mask EXR and return a bool (H, W) for the given object mask_id.

    Omni6DPose stores all objects in one EXR.  Channel 2 (BGR) scaled by 255
    gives an integer object-id per pixel matching the 'id' field in meta.json.
    """
    if not path.exists():
        return None
    try:
        import cv2, numpy as np
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            _warn_exr_once(path, "cv2.imread returned None")
            return None
        ids = np.array(arr[:, :, 2] * 255, dtype=np.uint8)
        ids[ids == 255] = 0  # bug fix for test_real subset (spurious 255 values)
        return torch.from_numpy(ids == mask_id)
    except Exception as e:
        _warn_exr_once(path, e)
        return None
