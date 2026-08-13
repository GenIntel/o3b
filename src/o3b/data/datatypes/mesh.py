from __future__ import annotations
import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import igl
import numpy as np
from torch import Tensor
import torch

logger = logging.getLogger(__name__)

# NCDS object size: the longest object side spans [-1, 1] (see o3b.io._load_mesh).
OBJ_SIZE_NCDS = 2.0


def verts_ncds(verts: Tensor, rescale: bool = True, recenter: bool = False) -> Tensor:
    """Normalise mesh verts towards NCDS: ``rescale`` makes the longest side span
    ``OBJ_SIZE_NCDS``, ``recenter`` moves the AABB centre to the origin.

    Meshes loaded via ``o3b.io._load_mesh`` are already NCDS, so both are a no-op
    for them.  A *predicted* mesh is only nominally NCDS — an instance-deformation
    head moves its extent and centre — while ``pred_cam_tform4x4_obj`` is an NCDS
    pose (the metric scale lives in the pose, not the verts), so any drift turns
    into a scale/offset error once the mesh is posed into camera space.

    The scale factor is taken from the pre-shift AABB, so the two flags are
    order-independent.
    """
    v = verts.float()
    if v.numel() == 0 or not (rescale or recenter):
        return v
    v_min, v_max = v.min(dim=0).values, v.max(dim=0).values
    if recenter:
        v = v - (v_min + v_max) * 0.5
    if rescale:
        v = v * (OBJ_SIZE_NCDS / (v_max - v_min).max().clamp(min=1e-8))
    return v


@contextlib.contextmanager
def _file_lock(path: Path):
    """Exclusive advisory lock on ``<path>``, held for the duration of the block.

    Used to serialise mesh conversion across the shard-build DataLoader workers:
    consecutive frame rows of one object are dealt to different workers, so
    without a lock every worker converts (and writes) the same mesh at once.

    Degrades to a no-op when the filesystem doesn't support flock (some network
    mounts); the tmp-file + ``os.replace`` in ``Mesh.save`` keeps concurrent
    writers safe on its own, the lock only avoids the duplicated work.
    """
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:
        logger.debug("could not lock %s (%s) — proceeding unlocked", path, e)
        if fd is not None:
            os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def weld_mesh(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge exactly-coincident vertices (e.g. xatlas UV-seam duplicates).

    Textured mc* meshes from convert_mesh_to_mc duplicate vertices along UV
    seams, cutting the closed marching-cubes surface into charts; connectivity-
    based matching (LBO eigenbasis, geodesics) then sees the seams as boundary
    cuts. Welding restores watertight connectivity.

    Returns:
        verts_w: (Nw, 3) welded vertices (first occurrence per position)
        faces_w: (F, 3) faces remapped to welded indices
        inv:     (N,) welded index of each input vertex
        rep:     (Nw,) input index of each welded vertex's representative
    """
    _, first, inv = np.unique(
        verts.round(decimals=6), axis=0, return_index=True, return_inverse=True
    )
    inv = inv.reshape(-1)
    return verts[first], inv[faces], inv, first


def mean_by_index(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    """Group-mean of (N, F) values over an (N,) index map into n groups."""
    out = np.zeros((n, values.shape[1]), dtype=np.float64)
    np.add.at(out, index, values.astype(np.float64))
    counts = np.bincount(index, minlength=n).astype(np.float64)
    return out / counts[:, None]


@dataclass
class Mesh:
    verts:       Tensor           # (V, 3)
    faces:       Tensor           # (F, 3)  int64
    vert_colors: Optional[Tensor] = None  # (V, 3) float RGB
    verts_uvs:   Optional[Tensor] = None  # (V, 2) float UV — y-flipped to image convention
    faces_uvs:   Optional[Tensor] = None  # (F, 3) int64 UV face indices
    texture:     Optional[Tensor] = None  # (3, H, W) float
    vert_feats:  Optional[Tensor] = None  # (V, C) float — per-vertex feature vectors

    def save(self, path: Path) -> None:
        """Export this mesh to a GLB file, and vert_feats to a sibling .pt file.

        Both files are written to a process-unique temporary name and then
        ``os.replace``d into place, so a concurrent reader never observes a
        half-written GLB (which trimesh reports as an unparseable file).  The
        feature sidecar is committed *before* the GLB, since ``load`` keys off
        the GLB and then looks for the sidecar.
        """
        from o3b.io import _mesh_to_trimesh
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # keep the real extension last — trimesh picks the export format from it
        tmp_name = lambda p: p.with_name(f".{p.stem}.{os.getpid()}.tmp{p.suffix}")

        if self.vert_feats is not None:
            feats_path = path.parent / (path.stem + "_vert_feats.pt")
            feats_tmp  = tmp_name(feats_path)
            torch.save(self.vert_feats, feats_tmp)
            os.replace(str(feats_tmp), str(feats_path))

        mesh_tmp = tmp_name(path)
        try:
            _mesh_to_trimesh(self).export(str(mesh_tmp))
            os.replace(str(mesh_tmp), str(path))
        finally:
            if mesh_tmp.exists():
                mesh_tmp.unlink()

    @classmethod
    def load(cls, path: Path) -> "Mesh":
        """Load a mesh from any supported file format, plus vert_feats if present."""
        from o3b.io import _load_mesh
        mesh, _ = _load_mesh(Path(path))
        if mesh is None:
            raise FileNotFoundError(f"Could not load mesh from {path}")
        feats_path = Path(path).parent / (Path(path).stem + "_vert_feats.pt")
        if feats_path.exists():
            mesh.vert_feats = torch.load(feats_path, weights_only=True)
        return mesh

    def to_pytorch3d(self, device=None):
        """Convert to a PyTorch3D ``Meshes`` object (single mesh, batch size 1).

        Texture priority:
          1. ``vert_colors`` (V, 3) → ``Textures(verts_rgb=[...])``.
          2. ``texture`` (3, H, W) + ``verts_uvs`` → ``TexturesUV``.
          3. Fallback: uniform grey ``verts_rgb``.
        """
        from pytorch3d.structures.meshes import Meshes
        from pytorch3d.renderer.mesh.textures import Textures, TexturesUV

        verts = self.verts.float()
        faces = self.faces.long()
        if device is not None:
            verts = verts.to(device)
            faces = faces.to(device)

        if self.vert_colors is not None:
            colors = self.vert_colors.float().clamp(0, 1).to(verts.device)
            textures = Textures(verts_rgb=[colors])
        elif self.texture is not None and self.verts_uvs is not None:
            tex_map = self.texture.float().permute(1, 2, 0).to(verts.device)  # (H, W, 3)
            uvs = self.verts_uvs.float().to(verts.device)
            # Mesh stores verts_uvs in image convention (v=0 at top).
            # PyTorch3D TexturesUV expects OpenGL convention (v=0 at bottom):
            # it maps v via lerp([1,-1], uv) so v=0→bottom row, v=1→top row.
            uvs = torch.stack([uvs[:, 0], 1.0 - uvs[:, 1]], dim=1)
            # If faces_uvs is absent, UV index == vertex index → reuse faces.
            f_uvs = (self.faces_uvs.long() if self.faces_uvs is not None else faces).to(verts.device)
            textures = TexturesUV(maps=[tex_map], faces_uvs=[f_uvs], verts_uvs=[uvs])
        else:
            V = verts.shape[0]
            grey = torch.full((V, 3), 0.5, dtype=torch.float32, device=verts.device)
            textures = Textures(verts_rgb=[grey])

        return Meshes(verts=[verts], faces=[faces], textures=textures)

    @classmethod
    def load_or_convert(
        cls,
        converted_path: Path,
        default_path: Path,
        mesh_type: str,
    ) -> "Mesh":
        """Return the converted mesh, generating and caching it on first call.

        If *converted_path* exists and loads, it is returned directly.
        Otherwise the mesh at *default_path* is loaded, converted to *mesh_type*
        (e.g. ``"mc64"``), saved to *converted_path*, and returned.

        An existing file that fails to load (truncated leftover from a killed
        run, or a corrupt cache entry) is not fatal: it is reconverted and
        overwritten.  Conversion runs under a per-mesh lock so parallel workers
        asking for the same object convert it once instead of N times.
        """
        converted_path = Path(converted_path)
        mesh = cls._try_load(converted_path)
        if mesh is not None:
            return mesh

        with _file_lock(converted_path.with_name(converted_path.name + ".lock")):
            # another process may have converted it while we waited for the lock
            mesh = cls._try_load(converted_path)
            if mesh is not None:
                return mesh
            converted = convert_mesh(mesh_type, cls.load(default_path))
            converted.save(converted_path)
        return converted

    @classmethod
    def _try_load(cls, path: Path) -> Optional["Mesh"]:
        """``load`` the mesh at *path*, or return None if it is absent/unreadable."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            return cls.load(path)
        except Exception as e:
            logger.warning("ignoring unreadable cached mesh %s (%s) — reconverting", path, e)
            return None


def _parse_mc_type(type_str: str) -> dict:
    """Parse mesh type strings like 'mc16', 'mc16_vuni4_r256_fdinov2s'.

    Returns dict with keys: res, view_sampling, n_views, resolution, feature_model.
    """
    import re
    m = re.fullmatch(
        r"mc(\d+)(?:_v(uni|rand)(\d+)(?:_r(\d+))?(?:_f(\w+))?)?",
        type_str,
    )
    if not m:
        raise ValueError(f"Cannot parse mesh type: {type_str!r}")
    return {
        "res":           int(m.group(1)),
        "view_sampling": m.group(2),                          # 'uni', 'rand', or None
        "n_views":       int(m.group(3)) if m.group(3) else None,
        "resolution":    int(m.group(4)) if m.group(4) else None,
        "feature_model": m.group(5),                          # e.g. 'dinov2s', or None
    }


def convert_mesh(type_str: str, mesh: Mesh) -> Mesh:
    if not type_str.startswith("mc"):
        raise ValueError(f"Unknown mesh type: {type_str}")
    params = _parse_mc_type(type_str)
    mc_mesh = convert_mesh_to_mc(mesh, params["res"])
    if params["feature_model"] is not None:
        vert_feats = _extract_vert_feats(
            mc_mesh,
            n_views=params["n_views"] or 4,
            resolution=params["resolution"] or 256,
            feature_model_name=params["feature_model"],
        )
        mc_mesh.vert_feats = vert_feats
    return mc_mesh


def _extract_vert_feats(
    mesh: "Mesh",
    n_views: int,
    resolution: int,
    feature_model_name: str,
) -> Tensor:
    """Render mesh from n_views uniform viewpoints, extract features with the named model,
    project mesh vertices onto each feature map, and return per-vertex mean features (V, C).
    """
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(feature_model_name)
    
    # ── Models that consume the mesh directly (own rendering or 3D encoder) ─────
    _SELF_RENDERING = {"diff3f", "dm", "dmmv", "dmweld", "dmweldflip", "partfield"}
    if feature_model_name.lower() in _SELF_RENDERING:
        from o3b.model.model import OD3D_Model
        from o3b.data.datatypes.object import ObjectBatch
        model = OD3D_Model.create_by_name(feature_model_name)
        model.eval()
        obj_batch = ObjectBatch(mesh=mesh)
        obj_batch = model(obj_batch)
        return obj_batch.verts3d_feats[0].cpu()  # (V, F)

    H = W = resolution

    # ── 1. Render mesh from uniformly sampled viewpoints ─────────────────────
    from o3b.data.viz import sample_uniform_viewpoints, render_mesh_from_viewpoints
    from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size

    batch = sample_uniform_viewpoints(n_views, mesh=mesh)
    modalities = render_mesh_from_viewpoints(batch, H=H, W=W, renderer="pyrender")
    rgbs    = modalities["rgb"].to(device)                               # (N, 3, H, W) float [0,1]
    depths  = modalities["depth"][:, 0].to(device) if "depth"   in modalities else None  # (N, H, W)
    normals = modalities["normals"].to(device)      if "normals" in modalities else None  # (N, 3, H, W)

    cam_tform4x4_obj = batch.cam_tform4x4_obj                   # (N, 4, 4)
    cam_intr4x4_single = get_default_camera_intrinsics_from_img_size(W, H)  # (4, 4)
    cam_intr4x4 = cam_intr4x4_single.unsqueeze(0).expand(n_views, -1, -1)  # (N, 4, 4)

    # ── 2. Load feature model and extract featmaps ────────────────────────────
    from o3b.model.model import OD3D_Model
    from o3b.data.datatypes.frame import FrameBatch
    model = OD3D_Model.create_by_name(feature_model_name)
    model.eval().to(device)

    with torch.no_grad():
        frames_in  = FrameBatch(rgb=rgbs, depth=depths, normal=normals)
        frames_out = FrameBatch(rgb=rgbs, depth=depths, normal=normals)
        _, frames_out = model(frames_in, frames_out)
    featmaps = frames_out.featmap  # (N, C, H_f, W_f)

    # ── 3. Project vertices onto each feature map ─────────────────────────────
    from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast

    N = n_views
    verts = mesh.verts.float().to(device)                        # (V, 3)
    V = verts.shape[0]

    cam_tform4x4_obj = cam_tform4x4_obj.float().to(device)       # (N, 4, 4)
    cam_intr4x4      = cam_intr4x4.float().to(device)            # (N, 4, 4)

    # pts3d (1,V,3), tform (N,1,4,4), intr (N,1,4,4) → (N,V,2)
    verts_2d = proj3d2d_tform4x4_intr4x4_broadcast(
        pts3d=verts.unsqueeze(0),
        tform4x4=cam_tform4x4_obj.unsqueeze(1),
        intr4x4=cam_intr4x4.unsqueeze(1),
    )                                                             # (N, V, 2)

    # visibility: positive depth and within image bounds
    valid = (
        (verts_2d[..., 0] >= 0) & (verts_2d[..., 0] <= W - 1)
        & (verts_2d[..., 1] >= 0) & (verts_2d[..., 1] <= H - 1)
    )                                                             # (N, V)

    # normalise pixel coords to [-1, 1] in feature-map space
    grid_x = (verts_2d[..., 0] / (W - 1)) * 2 - 1
    grid_y = (verts_2d[..., 1] / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)    # (N, V, 1, 2)

    sampled = F.grid_sample(
        featmaps.float(), grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )                                                             # (N, C, V, 1)
    sampled = sampled.squeeze(-1)                                 # (N, C, V)

    # masked mean across views
    valid_f = valid.float().unsqueeze(1)                          # (N, 1, V)
    vert_feats = (sampled * valid_f).sum(0) / valid_f.sum(0).clamp(min=1)  # (C, V)
    return vert_feats.T.cpu()                                     # (V, C)


def convert_mesh_to_mc(mesh: Mesh, res: int = 64) -> Mesh:
    import igl
    import numpy as np
    V, F = mesh.verts.cpu().numpy(), mesh.faces.cpu().numpy()

    min_v = V.min(axis=0)
    max_v = V.max(axis=0)
    padding = 0.05 * (max_v - min_v)
    min_v -= padding
    max_v += padding

    x = np.linspace(min_v[0], max_v[0], res)
    y = np.linspace(min_v[1], max_v[1], res)
    z = np.linspace(min_v[2], max_v[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    S, _, C, _ = igl.signed_distance(grid_points, V, F)

    # Surface snapping: grid points that are positive but within 10% of a voxel
    # diagonal from the surface are projected onto the surface and their SDF is
    # clamped to a small negative value.  This forces a zero-crossing on the
    # adjacent grid edge even when the feature is thinner than one voxel, and
    # places the projected point exactly on the surface so MC interpolates it
    # correctly.
    cell_half_size = (max_v - min_v) / res * 0.5

    snap_mask = np.abs(C[:, 0] - grid_points[:, 0]) < cell_half_size[0]
    snap_mask &= np.abs(C[:, 1] - grid_points[:, 1]) < cell_half_size[1]
    snap_mask &= np.abs(C[:, 2] - grid_points[:, 2]) < cell_half_size[2]
    
    grid_points[snap_mask] = C[snap_mask]  # snap to surface point exactly (after clipping to cell)

    S = np.where(snap_mask, -1e-5, S)

    SDF = S.reshape((res, res, res))


    verts, faces, _ = igl.marching_cubes(SDF.reshape(-1), grid_points, res, res, res, 0.0)
    print(f"MC{res} mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    has_color = mesh.vert_colors is not None or (
        mesh.texture is not None and mesh.verts_uvs is not None
    )
    if has_color:
        verts_t, faces_t, verts_uvs, texture = _build_texture_atlas(verts, faces, V, F, mesh)
        return Mesh(verts=verts_t, faces=faces_t, verts_uvs=verts_uvs, texture=texture)

    return Mesh(
        verts=torch.from_numpy(verts).float(),
        faces=torch.from_numpy(faces[:, [0, 2, 1]]).long(),  # flip winding: igl→viser convention
    )


def _build_texture_atlas(
    new_verts, new_faces, V_orig, F_orig, orig_mesh: Mesh, atlas_size: int = 512
):
    """
    UV-parameterize the MC mesh with xatlas, rasterize the atlas to 3D positions,
    then look up colors on the original mesh via closest-point sampling.

    Returns (verts_flat, faces_flat, verts_uvs, texture) where vertex and UV
    indexing are unified (no separate faces_uvs needed).
    """
    import xatlas
    import igl
    import numpy as np

    # 1. UV parameterization
    vmapping, uv_faces, uvs = xatlas.parametrize(
        new_verts.astype(np.float32), new_faces.astype(np.uint32)
    )
    # vmapping  (N_uv,)  → new_verts index for each UV vertex
    # uv_faces  (F, 3)   face indices into uvs / vmapping
    # uvs       (N_uv, 2) in [0,1]; xatlas convention: v=0 at bottom

    H = W = atlas_size
    verts_3d_uv = new_verts[vmapping]  # 3D position per UV vertex

    # 2. Rasterize: for each atlas texel find the 3D point on the MC surface.
    # We flip v here (xatlas v=0 bottom → image row 0 top) so the final texture
    # is stored in image convention and verts_uvs can be stored as-is.
    atlas_3d = np.full((H, W, 3), np.nan, dtype=np.float32)

    for face in uv_faces:
        tri_uv = uvs[face]       # (3, 2)
        tri_3d = verts_3d_uv[face]  # (3, 3)

        px = tri_uv[:, 0] * W
        py = (1.0 - tri_uv[:, 1]) * H  # v-flip: xatlas bottom→image top

        x0 = max(int(np.floor(px.min())), 0)
        y0 = max(int(np.floor(py.min())), 0)
        x1 = min(int(np.ceil(px.max())),  W - 1)
        y1 = min(int(np.ceil(py.max())),  H - 1)
        if x0 > x1 or y0 > y1:
            continue

        v0 = np.array([px[0], py[0]])
        v1 = np.array([px[1], py[1]])
        v2 = np.array([px[2], py[2]])
        e0, e1 = v1 - v0, v2 - v0
        denom = e0[0] * e1[1] - e0[1] * e1[0]
        if abs(denom) < 1e-8:
            continue

        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        dp = pts - v0
        b1 = (dp[:, 0] * e1[1] - dp[:, 1] * e1[0]) / denom
        b2 = (e0[0] * dp[:, 1] - e0[1] * dp[:, 0]) / denom
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        if not inside.any():
            continue

        bary = np.stack([b0[inside], b1[inside], b2[inside]], axis=1)
        pts_in = pts[inside].astype(int)
        atlas_3d[pts_in[:, 1], pts_in[:, 0]] = bary @ tri_3d

    # 3. For each valid texel look up color from the original mesh
    valid_mask = ~np.isnan(atlas_3d[:, :, 0])
    valid_pts  = atlas_3d[valid_mask]
    atlas_rgb  = np.zeros((H, W, 3), dtype=np.float32)

    if len(valid_pts) > 0:
        _, face_ids, closest = igl.point_mesh_squared_distance(valid_pts, V_orig, F_orig)
        bary_orig = igl.barycentric_coordinates(
            closest,
            V_orig[F_orig[face_ids, 0]],
            V_orig[F_orig[face_ids, 1]],
            V_orig[F_orig[face_ids, 2]],
        )

        if orig_mesh.vert_colors is not None:
            vc = orig_mesh.vert_colors.cpu().numpy()
            colors = (
                bary_orig[:, 0:1] * vc[F_orig[face_ids, 0]]
                + bary_orig[:, 1:2] * vc[F_orig[face_ids, 1]]
                + bary_orig[:, 2:3] * vc[F_orig[face_ids, 2]]
            )
        elif orig_mesh.texture is not None and orig_mesh.verts_uvs is not None:
            uv_orig = orig_mesh.verts_uvs.cpu().numpy()
            F_uv = orig_mesh.faces_uvs.cpu().numpy() if orig_mesh.faces_uvs is not None else F_orig
            uv0 = uv_orig[F_uv[face_ids, 0]]
            uv1 = uv_orig[F_uv[face_ids, 1]]
            uv2 = uv_orig[F_uv[face_ids, 2]]
            uvs_interp = (
                bary_orig[:, 0:1] * uv0
                + bary_orig[:, 1:2] * uv1
                + bary_orig[:, 2:3] * uv2
            )
            # verts_uvs are image-convention (v=0 at top); grid_sample y=-1 is top row
            tex = orig_mesh.texture  # (3, H_t, W_t)
            u_g = torch.from_numpy(uvs_interp[:, 0].astype(np.float32)) * 2 - 1
            v_g = torch.from_numpy(uvs_interp[:, 1].astype(np.float32)) * 2 - 1
            grid = torch.stack([u_g, v_g], dim=-1).view(1, 1, -1, 2)
            sampled = torch.nn.functional.grid_sample(
                tex.unsqueeze(0), grid,
                mode="bilinear", align_corners=True, padding_mode="border",
            )
            colors = sampled[0, :, 0, :].T.cpu().numpy()
        else:
            colors = np.full((len(valid_pts), 3), 0.7, dtype=np.float32)

        atlas_rgb[valid_mask] = colors.clip(0, 1)

    # Dilate valid colors into empty border texels to eliminate seam bleeding.
    # distance_transform_edt returns, for every invalid pixel, the index of the
    # nearest valid pixel — so we can simply copy colors from there.
    invalid_mask = ~valid_mask
    if invalid_mask.any() and valid_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest = distance_transform_edt(invalid_mask, return_indices=True)
        atlas_rgb[invalid_mask] = atlas_rgb[nearest[0][invalid_mask], nearest[1][invalid_mask]]

    # 4. Assemble flattened mesh (UV index == vertex index, no separate faces_uvs)
    texture   = torch.from_numpy(atlas_rgb).permute(2, 0, 1)  # (3, H, W)
    # Store UVs in image convention: v-flip matches the v-flip applied during rasterization
    uv_stored = uvs.astype(np.float32).copy()
    uv_stored[:, 1] = 1.0 - uv_stored[:, 1]

    return (
        torch.from_numpy(verts_3d_uv.astype(np.float32)),       # verts   (N_uv, 3)
        torch.from_numpy(uv_faces[:, [0, 2, 1]].astype(np.int64)),  # faces (F, 3) — winding flipped
        torch.from_numpy(uv_stored),                             # verts_uvs (N_uv, 2)
        texture,                                                 # (3, H, W)
    )
