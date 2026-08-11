from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor
from o3b.data.datatypes.mesh import Mesh
from o3b.data.datatypes.frame import Frame, _stack_field
from o3b.data.datatypes.object import Object


def _mask_dt_overlay(dt: Tensor, cmap, alpha: float = 0.7):
    """RGBA overlay for a normalised mask distance transform (0 outside the mask).

    The DT is normalised by the larger image side, so its values stay in the low
    percent range; it is rescaled by its own maximum to keep the interior
    gradient visible, and only the mask interior (dt > 0) is painted.
    """
    import numpy as np
    d = dt.float().cpu().numpy()
    if d.ndim == 3:
        d = d[0]
    ov = cmap(d / (d.max() + 1e-8)).astype(np.float32)
    ov[..., 3] = (d > 0) * alpha
    return ov


@dataclass(kw_only=True)
class FrameObject(Frame, Object):
    frame_object_id:  str
    cam_bbox2d:       Optional[Tensor] = None  # (4,)     xyxy pixels
    cam_bbox3d:       Optional[Tensor] = None  # (8, 3)   bounding-box corners in camera space
    fo_mask:          Optional[Tensor] = None  # (H, W)   bool  object-instance mask
    # (H, W) float32 distance transform of fo_mask, normalised to [0, 1] by the
    # larger image side (0 outside the mask) — see get_mask_distance_transform_norm.
    # Distances are in the pixel grid of the *current* image, so CropCamBBox2D
    # recomputes it from the cropped mask instead of resampling it.
    fo_mask_dt:       Optional[Tensor] = None
    # (H, W) bool amodal object mask: the GT mesh rendered under the GT pose,
    # so it also covers the parts of the object that other objects occlude in
    # this frame (it is still cut off at the image border).  fo_mask, in
    # contrast, is the annotated *visible* instance mask.
    fo_mask_amodal:    Optional[Tensor] = None
    # (H, W) float32 distance transform of fo_mask_amodal, same normalisation
    # and same crop-time recomputation as fo_mask_dt.
    fo_mask_amodal_dt: Optional[Tensor] = None
    obj_kpts2d_mask:  Optional[Tensor] = None  # (K,)     bool  keypoint visible in this frame (occlusion-aware)
    cam_tform4x4_obj:      Optional[Tensor] = None  # (4, 4)  cam←obj SE(3)
    cam_tform4x4_obj_ncds: Optional[Tensor] = None  # (4, 4)  ncds→cam (= cam_tform4x4_obj @ obj_ncds0c_tform4x4_obj)

    def viz(self, show: bool = True, server=None, node_prefix: str = "",
            offset_x: float = 0.0, obj_centric: bool = False, kpt_colors=None):
        """Visualise this frame-object.

        server=None (default): interactive 2-D overlay viewer with CheckButtons
          to toggle modalities (rgb, fo_mask, fo_mask_dt, fo_mask_amodal,
          fo_mask_amodal_dt, depth, depth_mask, frame_mask, cam_bbox2d/3d,
          keypoints); returns the composed (3, H, W) tensor.

        server given (a viser server): add the frame-object to the 3-D scene
          (mesh, camera frustum, rgb panel, depth point cloud, camera + object
          axes, keypoint spheres) under *node_prefix*, translated by *offset_x*
          along world X; returns the list of viser handles.
        """
        if server is not None:
            from o3b.data.viz_viser import _add_frame_object_to_scene
            return _add_frame_object_to_scene(
                server, self, prefix=node_prefix, offset_x=offset_x,
                obj_centric=obj_centric, kpt_colors=kpt_colors,
            )

        import numpy as np

        layers: dict[str, tuple] = {}

        if self.rgb is not None:
            rgb = self.rgb.float()
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            layers["rgb"] = ("rgb", rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy())

        if self.fo_mask is not None:
            m = self.fo_mask.float().cpu().numpy()
            ov = np.zeros((*m.shape, 4), dtype=np.float32)
            ov[..., 1] = 0.9
            ov[..., 3] = m * 0.5
            layers["fo_mask"] = ("overlay", ov)

        # The distance transforms cover the same pixels as the masks they come
        # from, so they start toggled off (see `off_by_default` below).
        if self.fo_mask_dt is not None:
            import matplotlib.cm as _cm
            layers["fo_mask_dt"] = ("overlay", _mask_dt_overlay(self.fo_mask_dt, _cm.viridis))

        if self.fo_mask_amodal is not None:
            am = self.fo_mask_amodal.float().cpu().numpy()
            ov = np.zeros((*am.shape, 4), dtype=np.float32)
            ov[..., 1] = 0.8; ov[..., 2] = 0.9   # cyan, vs the green visible mask
            ov[..., 3] = am * 0.4
            layers["fo_mask_amodal"] = ("overlay", ov)

        if self.fo_mask_amodal_dt is not None:
            import matplotlib.cm as _cm
            layers["fo_mask_amodal_dt"] = ("overlay",
                                           _mask_dt_overlay(self.fo_mask_amodal_dt, _cm.magma))

        if self.depth is not None:
            import matplotlib.cm as _cm
            d = self.depth.float().cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            ov = _cm.plasma(d).astype(np.float32)
            ov[..., 3] = 0.6
            layers["depth"] = ("overlay", ov)

        if self.depth_mask is not None:
            dm = self.depth_mask.float().cpu().numpy()
            ov = np.zeros((*dm.shape, 4), dtype=np.float32)
            ov[..., 2] = 0.9
            ov[..., 3] = dm * 0.4
            layers["depth_mask"] = ("overlay", ov)

        if self.mask is not None:
            fm = self.mask.float().cpu().numpy()
            ov = np.zeros((*fm.shape, 4), dtype=np.float32)
            ov[..., 0] = 1.0; ov[..., 1] = 0.55
            ov[..., 3] = fm * 0.4
            layers["frame_mask"] = ("overlay", ov)

        if self.cam_bbox2d is not None:
            layers["cam_bbox2d"] = ("draw_bbox", self.cam_bbox2d.cpu())

        if (self.cam_bbox3d is not None
                and self.cam_intr4x4 is not None
                and self.cam_bbox3d.shape == torch.Size([8, 3])):
            layers["cam_bbox3d"] = ("draw_bbox3d_corners", (
                self.cam_bbox3d.cpu(),
                self.cam_intr4x4.cpu(),
            ))
        elif (self.cam_bbox3d is not None
                and self.cam_intr4x4 is not None
                and self.cam_tform4x4_obj is not None
                and self.cam_bbox3d.shape == torch.Size([3])):
            layers["cam_bbox3d"] = ("draw_bbox3d", (
                self.cam_bbox3d.cpu(),
                self.cam_intr4x4.cpu(),
                self.cam_tform4x4_obj.cpu(),
            ))

        _tform_kpts = self.cam_tform4x4_obj_ncds if self.cam_tform4x4_obj_ncds is not None \
            else self.cam_tform4x4_obj
        if (self.obj_kpts3d is not None
                and self.cam_intr4x4 is not None
                and _tform_kpts is not None):
            layers["obj_kpts3d"] = ("draw_kpts2d", (
                self.obj_kpts3d.cpu(),
                self.obj_kpts3d_mask.cpu() if self.obj_kpts3d_mask is not None else None,
                self.cam_intr4x4.cpu(),
                _tform_kpts.cpu(),
            ))
            if self.obj_kpts2d_mask is not None:
                layers["obj_kpts2d_mask"] = ("draw_kpts2d", (
                    self.obj_kpts3d.cpu(),
                    self.obj_kpts2d_mask.cpu(),
                    self.cam_intr4x4.cpu(),
                    _tform_kpts.cpu(),
                ))

        if not layers:
            return None

        label_order    = list(layers.keys())
        off_by_default = {"fo_mask_dt", "fo_mask_amodal_dt"}
        active         = {k: k not in off_by_default for k in label_order}

        def _hw() -> tuple[int, int]:
            for kind, data in layers.values():
                if kind in ("rgb", "overlay"):
                    return data.shape[:2]
            return 480, 640

        def _compose_numpy() -> np.ndarray:
            H, W = _hw()
            canvas = np.zeros((H, W, 3), dtype=np.float32)
            if "rgb" in layers and active.get("rgb", True):
                canvas = layers["rgb"][1].copy()
            elif "rgb" in layers:
                canvas[:] = 0.15

            for k in label_order:
                if k == "rgb" or not active.get(k, True):
                    continue
                kind, data = layers[k]
                if kind == "overlay":
                    alpha = data[..., 3:4]
                    canvas = canvas * (1 - alpha) + data[..., :3] * alpha

            # apply tensor-based bbox draws
            canvas_t = torch.from_numpy(canvas).permute(2, 0, 1)  # (3,H,W) float [0,1]

            if "cam_bbox2d" in layers and active.get("cam_bbox2d", True):
                from o3b.cv.visual.draw import draw_bbox
                try:
                    canvas_t = draw_bbox(
                        canvas_t, layers["cam_bbox2d"][1],
                        color=(255, 255, 0), line_width=2,
                    ).float().div(255.0)
                except Exception:
                    pass

            if "cam_bbox3d" in layers and active.get("cam_bbox3d", True):
                kind_bbox, data_bbox = layers["cam_bbox3d"]
                try:
                    if kind_bbox == "draw_bbox3d_corners":
                        from o3b.cv.visual.draw import draw_bbox3d_corners
                        corners_cam, cam_intr4x4 = data_bbox
                        canvas_t = draw_bbox3d_corners(
                            canvas_t, corners_cam, cam_intr4x4,
                        ).float().div(255.0)
                    else:
                        from o3b.cv.visual.draw import draw_bbox3d
                        obj_size3d, cam_intr4x4, cam_tform4x4_obj = data_bbox
                        canvas_t = draw_bbox3d(
                            canvas_t, obj_size3d, cam_intr4x4, cam_tform4x4_obj,
                        ).float().div(255.0)
                except Exception:
                    pass

            for _kpts_key in ("obj_kpts3d", "obj_kpts2d_mask"):
                if _kpts_key in layers and active.get(_kpts_key, True):
                    from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast
                    from o3b.data.datatypes.object import _draw_kpts2d_on_imgs
                    try:
                        kpts3d, kpts_mask, cam_intr4x4, cam_tform = layers[_kpts_key][1]
                        H_c, W_c = canvas_t.shape[-2:]
                        kpts2d = proj3d2d_tform4x4_intr4x4_broadcast(
                            pts3d=kpts3d.unsqueeze(0),
                            tform4x4=cam_tform.unsqueeze(0).unsqueeze(0),
                            intr4x4=cam_intr4x4.unsqueeze(0).unsqueeze(0),
                        )  # (1, K, 2)
                        canvas_t = _draw_kpts2d_on_imgs(
                            canvas_t.unsqueeze(0),
                            kpts2d,
                            mask=kpts_mask,
                            radius=max(H_c, W_c) // 50,
                        )[0]
                    except Exception:
                        pass

            return np.clip(canvas_t.permute(1, 2, 0).numpy(), 0, 1)

        if show:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import CheckButtons

            fig, ax = plt.subplots(figsize=(9, 6))
            fig.subplots_adjust(bottom=0.14)
            ax.axis("off")
            cat_str = str(self.category) if self.category is not None else ""
            title   = getattr(self, "frame_id", "")
            if cat_str:
                title = f"{title}  [{cat_str}]"
            ax.set_title(title, fontsize=9)

            im = ax.imshow(_compose_numpy())
            if cat_str:
                ax.text(
                    0.01, 0.97, cat_str,
                    transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    color="white", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55),
                )

            def _redraw(_label: str = "") -> None:
                im.set_data(_compose_numpy())
                fig.canvas.draw_idle()

            _redraw()

            ax_cb = fig.add_axes([0.05, 0.01, 0.9, 0.10])
            check = CheckButtons(ax_cb, label_order,
                                 actives=[active[k] for k in label_order])

            def on_toggle(label: str) -> None:
                active[label] = not active[label]
                _redraw(label)

            check.on_clicked(on_toggle)
            plt.show()

        canvas = _compose_numpy()
        return torch.from_numpy(canvas).permute(2, 0, 1)

    def viz3d(self, max_pts: int = 50_000) -> Optional[object]:
        """Build a wandb.Object3D with colored point cloud, mesh, and bbox.

        Contents (all in OpenGL camera space — X right, Y up, -Z forward, to match
        cam_tform4x4_obj which is stored in the OpenGL convention):
          - colored 3-D point cloud unprojected from depth (filtered by fo_mask)
          - mesh transformed to camera space (if mesh is present)
          - 3-D bounding box from cam_bbox3d + cam_tform4x4_obj

        When the mesh has faces (and trimesh is available) everything is exported
        as a GLB scene so the mesh renders as an actual surface, the depth points
        as a colored point cloud, and the bbox as orange cylinder edges.
        Otherwise falls back to a wandb lidar point cloud with the mesh verts as
        points and the bbox as wandb box corners.

        The point cloud is downsampled by a factor of 4, and the whole scene is
        median-centered and quantile-scaled so that 80% of the points lie inside
        [-1, 1]³ (the same center/scale is applied to mesh and bbox).

        Returns None when no geometry is available or wandb is not installed.
        """
        try:
            import wandb
        except ImportError:
            return None

        import numpy as np
        from o3b.cv.geometry.transform import depth2pts3d_grid, transf3d_broadcast

        pts_xyz: list = []
        pts_rgb: list = []
        boxes:   list = []

        tform = self.cam_tform4x4_obj

        # ── 1. Colored point cloud from depth ─────────────────────────────────
        if self.depth is not None and self.cam_intr4x4 is not None:
            depth = self.depth.float().cpu()           # (H, W)
            intr  = self.cam_intr4x4.float().cpu()     # (4, 4)

            # depth2pts3d_grid expects (..., 1, H, W).  opengl=True so the point cloud
            # matches cam_tform4x4_obj (OpenGL, -Z forward) used to place the mesh/bbox.
            pts3d      = depth2pts3d_grid(depth.unsqueeze(-3), intr, opengl=True)  # (3, H, W)
            pts3d_flat = pts3d.permute(1, 2, 0).reshape(-1, 3)  # (HW, 3)

            if self.rgb is not None:
                rgb = self.rgb.float().cpu()
                if rgb.max() > 1.5:
                    rgb = rgb / 255.0
                rgb_flat = (rgb.permute(1, 2, 0).reshape(-1, 3) * 255).byte()  # (HW, 3) uint8
            else:
                rgb_flat = torch.full((pts3d_flat.shape[0], 3), 128, dtype=torch.uint8)

            valid = depth.reshape(-1) > 0
            if self.depth_mask is not None:
                valid = valid & self.depth_mask.bool().cpu().reshape(-1)

            if self.fo_mask is not None:
                fo_flat = self.fo_mask.bool().cpu().reshape(-1)
                obj_mask = valid & fo_flat
                bg_mask  = valid & ~fo_flat

                if obj_mask.any():
                    pts_xyz.append(pts3d_flat[obj_mask].numpy().astype(np.float32))
                    pts_rgb.append(rgb_flat[obj_mask].numpy())

                n_bg = min(int(bg_mask.sum()), max_pts // 4)
                if n_bg > 0:
                    idx = torch.where(bg_mask)[0][torch.randperm(int(bg_mask.sum()))[:n_bg]]
                    pts_xyz.append(pts3d_flat[idx].numpy().astype(np.float32))
                    pts_rgb.append(rgb_flat[idx].numpy())
            else:
                total = int(valid.sum())
                n = min(total, max_pts)
                if n > 0:
                    idx = torch.where(valid)[0][torch.randperm(total)[:n]]
                    pts_xyz.append(pts3d_flat[idx].numpy().astype(np.float32))
                    pts_rgb.append(rgb_flat[idx].numpy())

        # ── 1b. Downsample the point cloud by a factor of 4 ──────────────────
        if pts_xyz:
            pts_all = np.concatenate(pts_xyz, axis=0).astype(np.float32)
            rgb_all = np.concatenate(pts_rgb, axis=0)
            keep = np.random.permutation(len(pts_all))[: max(1, len(pts_all) // 4)]
            pts_xyz = [pts_all[keep]]
            pts_rgb = [rgb_all[keep]]

        # ── 1c. Median centering + quantile scaling ──────────────────────────
        # Normalise the whole scene so 80% of the points lie inside [-1, 1]³:
        # center = per-axis median, scale = 0.8-quantile of the per-point
        # max-abs coordinate after centering.  The same (center, scale) is
        # applied to mesh verts and bbox corners to keep everything aligned.
        norm_center = np.zeros(3, dtype=np.float32)
        norm_scale  = 1.0
        if pts_xyz and len(pts_xyz[0]) > 0:
            _p = pts_xyz[0]
            norm_center = np.median(_p, axis=0).astype(np.float32)
            _d = np.abs(_p - norm_center).max(axis=1)
            _s = float(np.quantile(_d, 0.8))
            norm_scale = _s if _s > 1e-6 else 1.0
            pts_xyz = [((_p - norm_center) / norm_scale).astype(np.float32)]

        def _normalize(x: "torch.Tensor") -> "torch.Tensor":
            c = torch.from_numpy(norm_center).to(dtype=x.dtype)
            return (x - c) / norm_scale

        # ── 2. Mesh in camera space ───────────────────────────────────────────
        # Preferred: an actual triangle mesh (rendered via a GLB scene below).
        # Fallback (no faces / no trimesh): mesh verts as extra points.
        mesh_tm  = None
        verts_cam = None
        mesh_vc   = None
        if self.mesh is not None and tform is not None:
            verts     = self.mesh.verts.float().cpu()          # (V, 3)
            verts_cam = transf3d_broadcast(verts, tform.float().cpu())  # (V, 3)
            verts_cam = _normalize(verts_cam)

            if self.mesh.vert_colors is not None:
                mesh_vc = (self.mesh.vert_colors.float().cpu().clamp(0, 1) * 255).byte()
            else:
                mesh_vc = torch.full((len(verts), 3), 200, dtype=torch.uint8)  # light grey

            faces = getattr(self.mesh, "faces", None)
            if faces is not None and len(faces) > 0:
                try:
                    import trimesh
                    mesh_tm = trimesh.Trimesh(
                        vertices=verts_cam.numpy().astype(np.float64),
                        faces=faces.long().cpu().numpy(),
                        vertex_colors=mesh_vc.numpy(),
                        process=False,
                    )
                except ImportError:
                    mesh_tm = None

        # ── 3. 3-D bounding box ───────────────────────────────────────────────
        if self.cam_bbox3d is not None:
            bbox = self.cam_bbox3d.float().cpu()
            label = str(self.category) if self.category is not None else "object"

            if bbox.shape == torch.Size([8, 3]):
                # cam-space corners (8, 3) — standard post-dataset-loading format
                corners_norm = _normalize(bbox)
                boxes.append({"corners": corners_norm.tolist(), "label": label, "color": [255, 165, 0]})
            elif bbox.shape == torch.Size([3]) and tform is not None:
                # legacy: side-length (3,) in NCDS space + NCDS→cam tform
                hx, hy, hz = (bbox / 2).tolist()
                corners_obj = torch.tensor([
                    [-hx, -hy, -hz], [ hx, -hy, -hz],
                    [ hx,  hy, -hz], [-hx,  hy, -hz],
                    [-hx, -hy,  hz], [ hx, -hy,  hz],
                    [ hx,  hy,  hz], [-hx,  hy,  hz],
                ], dtype=torch.float32)
                corners_cam = transf3d_broadcast(corners_obj, tform.float().cpu())  # (8, 3)
                corners_cam = _normalize(corners_cam)
                boxes.append({"corners": corners_cam.tolist(), "label": label, "color": [255, 165, 0]})

        # ── 4. Preferred: GLB scene — mesh as a real surface + point cloud + bbox ─
        if mesh_tm is not None:
            try:
                import tempfile
                import trimesh
                scene = trimesh.Scene()
                scene.add_geometry(mesh_tm, node_name="mesh")
                if pts_xyz:
                    # GLTF POINTS primitives don't render vertex colors reliably in
                    # the wandb (Babylon.js) viewer → bake each point as a tiny
                    # triangle splat instead; triangles always render with COLOR_0.
                    pts  = np.concatenate(pts_xyz, axis=0).astype(np.float64)  # (N, 3)
                    cols = np.concatenate(pts_rgb, axis=0).astype(np.uint8)    # (N, 3)
                    N = pts.shape[0]
                    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) if N > 1 else 1.0
                    r = max(diag * 0.0015, 1e-5)
                    # fixed-orientation equilateral triangle around each point
                    offsets = np.array([
                        [ 0.0,   r,  0.0],
                        [-r * 0.866, -r * 0.5, 0.0],
                        [ r * 0.866, -r * 0.5, 0.0],
                    ], dtype=np.float64)  # (3, 3)
                    splat_verts = (pts[:, None, :] + offsets[None]).reshape(-1, 3)   # (3N, 3)
                    splat_faces = np.arange(3 * N, dtype=np.int64).reshape(-1, 3)    # (N, 3)
                    splat_cols  = np.repeat(cols, 3, axis=0)                          # (3N, 3)
                    scene.add_geometry(
                        trimesh.Trimesh(
                            vertices=splat_verts,
                            faces=splat_faces,
                            vertex_colors=splat_cols,
                            process=False,
                        ),
                        node_name="pointcloud",
                    )
                # bbox wireframe as thin cylinders (guaranteed to render as geometry)
                EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)]
                for box in boxes:
                    c = np.asarray(box["corners"], dtype=np.float64)
                    diag = float(np.linalg.norm(c.max(0) - c.min(0)))
                    r = max(diag * 0.004, 1e-4)
                    for i, j in EDGES:
                        if np.linalg.norm(c[j] - c[i]) < 1e-8:
                            continue
                        cyl = trimesh.creation.cylinder(
                            radius=r, segment=np.stack([c[i], c[j]]), sections=6,
                        )
                        cyl.visual.face_colors = [255, 165, 0, 255]
                        scene.add_geometry(cyl)
                f = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
                f.write(scene.export(file_type="glb"))
                f.close()
                return wandb.Object3D.from_file(f.name)
            except Exception:
                pass  # fall through to the lidar point-cloud representation

        # ── fallback: lidar point cloud (mesh verts as points) ───────────────
        if verts_cam is not None:
            pts_xyz.append(verts_cam.numpy().astype(np.float32))
            pts_rgb.append(mesh_vc.numpy())

        if not pts_xyz and not boxes:
            return None

        if pts_xyz:
            cloud = np.concatenate(
                [np.concatenate(pts_xyz, axis=0),
                 np.concatenate(pts_rgb, axis=0).astype(np.float32)],
                axis=1,
            )  # (N, 6): xyz rgb 0-255
        else:
            cloud = np.zeros((1, 6), dtype=np.float32)

        return wandb.Object3D({
            "type": "lidar/beta",
            "points": cloud,
            "boxes": np.array(boxes, dtype=object),  # wandb calls .tolist() on this
        })


@dataclass
class FrameObjectBatch:
    """Stacked across B samples. Each sample = 1 frame + 1 object."""
    # frame
    cam_intr4x4:      Optional[Tensor]       = None  # (B, 4, 4)
    rgb:              Optional[Tensor]       = None  # (B, 3, H, W)
    depth:            Optional[Tensor]       = None  # (B, H, W)
    depth_mask:       Optional[Tensor]       = None  # (B, H, W)
    frame_mask:       Optional[Tensor]       = None  # (B, H, W)
    feat:             Optional[Tensor]       = None  # (B, F)
    featmap:          Optional[Tensor]       = None  # (B, F, H, W)
    featmap_lvls:     Optional[List[Tensor]] = None  # L x (B, F, H_l, W_l)
    # frame-object
    cam_bbox2d:       Optional[Tensor]       = None  # (B, 4)
    cam_bbox3d:       Optional[Tensor]       = None  # (B, 8, 3)
    fo_mask:          Optional[Tensor]       = None  # (B, H, W)
    fo_mask_dt:       Optional[Tensor]       = None  # (B, H, W) float32, normalised DT of fo_mask
    fo_mask_amodal:    Optional[Tensor]      = None  # (B, H, W) bool, GT mesh rendered under the GT pose
    fo_mask_amodal_dt: Optional[Tensor]      = None  # (B, H, W) float32, normalised DT of fo_mask_amodal
    obj_kpts2d_mask:  Optional[Tensor]       = None  # (B, K)    bool
    cam_tform4x4_obj: Optional[Tensor]       = None  # (B, 4, 4)
    cam_tform4x4_obj_ncds: Optional[Tensor]  = None  # (B, 4, 4) ncds→cam
    # object
    pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    verts3d_feats:           Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask:      Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    category:                Optional[object] = None  # (B,) int64, or list[str] of names
    mesh:                    Optional[Mesh]   = None  # shared mesh for all B viewpoints
    # per-sample GT meshes — a training batch mixes instances, so vertex counts
    # differ and these stay a list (like FrameObjectPairBatch.src_meshes)
    meshes:                  Optional[list]   = None  # list of B Mesh


@dataclass(kw_only=True)
class FrameObjectPair:
    """A query (src) + target (trgt) pair of FrameObjects of the same category."""
    src_object:     FrameObject
    trgt_object:    FrameObject
    src_object_id:  Optional[str] = None
    trgt_object_id: Optional[str] = None


@dataclass
class FrameObjectPairBatch:
    """Stacked across B FrameObjectPair samples (src = query, trgt = target)."""
    # ── keypoints (NCDS object space) ─────────────────────────────────────────
    src_obj_kpts3d:       Optional[Tensor] = None  # (B, K, 3)
    trgt_obj_kpts3d:      Optional[Tensor] = None  # (B, K, 3)
    src_obj_kpts3d_mask:  Optional[Tensor] = None  # (B, K) bool
    trgt_obj_kpts3d_mask: Optional[Tensor] = None  # (B, K) bool
    src_obj_kpts2d_mask:  Optional[Tensor] = None  # (B, K) bool — visible in frame
    trgt_obj_kpts2d_mask: Optional[Tensor] = None  # (B, K) bool
    # ── ground-truth poses ────────────────────────────────────────────────────
    src_cam_tform4x4_obj:       Optional[Tensor] = None  # (B, 4, 4) cam←obj (metric)
    trgt_cam_tform4x4_obj:      Optional[Tensor] = None  # (B, 4, 4)
    src_cam_tform4x4_obj_ncds:  Optional[Tensor] = None  # (B, 4, 4) ncds→cam
    trgt_cam_tform4x4_obj_ncds: Optional[Tensor] = None  # (B, 4, 4)
    src_obj_size:               Optional[Tensor] = None  # (B,) metric max bbox extent
    trgt_obj_size:              Optional[Tensor] = None  # (B,)
    src_obj_syms:               Optional[Tensor] = None  # (B, 3) see FrameObject.obj_syms
    trgt_obj_syms:              Optional[Tensor] = None  # (B, 3)
    # ── predicted poses (filled by a pose-estimation method; None → use GT) ────
    src_pred_cam_tform4x4_obj:  Optional[Tensor] = None  # (B, 4, 4)
    trgt_pred_cam_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    src_pred_obj_size3d:        Optional[Tensor] = None  # (B, 3) predicted side lengths
    trgt_pred_obj_size3d:       Optional[Tensor] = None  # (B, 3)
    # ── frame inputs (stacked tensors, or per-sample lists if sizes differ) ────
    src_cam_intr4x4:  Optional[Tensor] = None  # (B, 4, 4)
    trgt_cam_intr4x4: Optional[Tensor] = None
    src_rgb:          Optional[object] = None  # (B, 3, H, W)
    trgt_rgb:         Optional[object] = None
    src_depth:        Optional[object] = None  # (B, H, W)
    trgt_depth:       Optional[object] = None
    src_depth_mask:   Optional[object] = None  # (B, H, W) bool
    trgt_depth_mask:  Optional[object] = None
    src_fo_mask:      Optional[object] = None  # (B, H, W) bool object instance mask
    trgt_fo_mask:     Optional[object] = None
    src_fo_mask_dt:   Optional[object] = None  # (B, H, W) float32 normalised DT of fo_mask
    trgt_fo_mask_dt:  Optional[object] = None
    # ── per-sample meshes (for qualitative rendering) + category ───────────────
    src_meshes:   Optional[list]   = None  # list of B Mesh
    trgt_meshes:  Optional[list]   = None  # list of B Mesh
    src_category: Optional[Tensor] = None  # (B,) int64
    trgt_category: Optional[Tensor] = None  # (B,) int64
    # 3-D bounding-box corners in camera space
    src_cam_bbox3d:  Optional[Tensor] = None  # (B, 8, 3)
    trgt_cam_bbox3d: Optional[Tensor] = None  # (B, 8, 3)
    # ── predicted features / meshes (filled by a method) ───────────────────────
    src_featmap:    Optional[Tensor] = None  # (B, F, Hf, Wf) L2-normalised feature maps
    trgt_featmap:   Optional[Tensor] = None
    src_pred_mesh:  Optional[list]   = None  # list of B Mesh, canonical (GL) NCDS frame
    trgt_pred_mesh: Optional[list]   = None


def collate_frame_object_pairs(
    samples: list[FrameObjectPair],
    include: Optional[set[str]] = None,
) -> FrameObjectPairBatch:
    def _get(attr, side):
        return _stack_field([getattr(getattr(s, f"{side}_object"), attr, None) for s in samples])

    def _scalar(attr, side):
        return _stack_field([
            torch.as_tensor(getattr(getattr(s, f"{side}_object"), attr), dtype=torch.float32)
            if getattr(getattr(s, f"{side}_object"), attr, None) is not None else None
            for s in samples
        ])

    def _stack_or_list(attr, side):
        vals = [getattr(getattr(s, f"{side}_object"), attr, None) for s in samples]
        if any(v is None for v in vals):
            return None
        return (torch.stack(vals, dim=0)
                if len({tuple(v.shape) for v in vals}) == 1 else vals)

    def _meshes(side):
        ms = [getattr(getattr(s, f"{side}_object"), "mesh", None) for s in samples]
        return ms if any(m is not None for m in ms) else None

    def _cat(side):
        return _stack_field([
            torch.tensor(getattr(getattr(s, f"{side}_object"), "category_id"))
            if getattr(getattr(s, f"{side}_object"), "category_id", None) is not None else None
            for s in samples
        ])

    out = FrameObjectPairBatch()
    for side in ("src", "trgt"):
        setattr(out, f"{side}_obj_kpts3d",            _get("obj_kpts3d", side))
        setattr(out, f"{side}_obj_kpts3d_mask",       _get("obj_kpts3d_mask", side))
        setattr(out, f"{side}_obj_kpts2d_mask",       _get("obj_kpts2d_mask", side))
        setattr(out, f"{side}_cam_tform4x4_obj",      _get("cam_tform4x4_obj", side))
        setattr(out, f"{side}_cam_tform4x4_obj_ncds", _get("cam_tform4x4_obj_ncds", side))
        setattr(out, f"{side}_obj_size",              _scalar("obj_size", side))
        setattr(out, f"{side}_obj_syms",              _get("obj_syms", side))
        setattr(out, f"{side}_cam_intr4x4",           _get("cam_intr4x4", side))
        setattr(out, f"{side}_rgb",                   _stack_or_list("rgb", side))
        setattr(out, f"{side}_depth",                 _stack_or_list("depth", side))
        setattr(out, f"{side}_depth_mask",            _stack_or_list("depth_mask", side))
        setattr(out, f"{side}_fo_mask",               _stack_or_list("fo_mask", side))
        setattr(out, f"{side}_fo_mask_dt",            _stack_or_list("fo_mask_dt", side))
        setattr(out, f"{side}_meshes",                _meshes(side))
        setattr(out, f"{side}_category",              _cat(side))
        setattr(out, f"{side}_cam_bbox3d",            _get("cam_bbox3d", side))
    return out


def collate_frame_objects(
    samples: list[FrameObject],
    include: Optional[set[str]] = None,
) -> FrameObjectBatch:
    def _get(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_lvls(attr):
        if include and attr not in include:
            return None
        per_sample = [getattr(s, attr) for s in samples]
        if any(v is None for v in per_sample):
            return None
        return [
            torch.stack([per_sample[b][l] for b in range(len(per_sample))])
            for l in range(len(per_sample[0]))
        ]

    return FrameObjectBatch(
        cam_intr4x4      = _get("cam_intr4x4"),
        rgb              = _get("rgb"),
        depth            = _get("depth"),
        depth_mask       = _get("depth_mask"),
        frame_mask       = _get("mask"),
        feat             = _get("feat"),
        featmap          = _get("featmap"),
        featmap_lvls     = _get_lvls("featmap_lvls"),
        cam_bbox2d       = _get("cam_bbox2d"),
        cam_bbox3d       = _get("cam_bbox3d"),
        fo_mask          = _get("fo_mask"),
        fo_mask_dt       = _get("fo_mask_dt"),
        fo_mask_amodal    = _get("fo_mask_amodal"),
        fo_mask_amodal_dt = _get("fo_mask_amodal_dt"),
        obj_kpts2d_mask  = _get("obj_kpts2d_mask"),
        cam_tform4x4_obj = _get("cam_tform4x4_obj"),
        cam_tform4x4_obj_ncds = _get("cam_tform4x4_obj_ncds"),
        pts3d                   = _get("pts3d"),
        pts3d_feats             = _get("pts3d_feats"),
        pts3d_feats_mask        = _get("pts3d_feats_mask"),
        verts3d_feats           = _get("verts3d_feats"),
        verts3d_feats_mask      = _get("verts3d_feats_mask"),
        obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj"),
        obj_kpts3d              = _get("obj_kpts3d"),
        obj_kpts3d_mask         = _get("obj_kpts3d_mask"),
        # GT meshes stay per-sample (variable vertex counts); ObjGeo3D's pcl_cd
        # uses their vertices as the target point cloud
        meshes = (
            [s.mesh for s in samples]
            if (include is None or "mesh" in include)
            and any(s.mesh is not None for s in samples)
            else None
        ),
        # category may be an int id or a name string (some loaders store names)
        category = (
            [s.category for s in samples]
            if any(isinstance(s.category, str) for s in samples)
            else _stack_field([
                torch.tensor(s.category) if s.category is not None else None
                for s in samples
            ])
        ) if (include is None or "category" in include) else None,
    )


def get_frame_object_from_batch(batch, item_id: int, prefix: str = "") -> FrameObject:
    """Extract a single FrameObject from a batch at the given index.

    Works on FrameObjectBatch (prefix="") and FrameObjectPairBatch
    (prefix="src_" or "trgt_").  When ``pred_cam_tform4x4_obj`` is present it
    overrides the GT pose and ``pred_obj_size3d`` is used to compute cam-space
    bbox corners.  Otherwise ``cam_bbox3d`` (already (8,3) cam-space corners)
    is used directly.
    """
    def _get(key):
        v = getattr(batch, f"{prefix}{key}", None)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v[item_id].cpu()
        if isinstance(v, list):
            return v[item_id] if item_id < len(v) else None
        return None

    pred_pose = _get("pred_cam_tform4x4_obj")
    cam_tform4x4_obj = pred_pose if pred_pose is not None else _get("cam_tform4x4_obj_ncds")

    cam_bbox3d = None
    if pred_pose is not None:
        raw_size = _get("pred_obj_size3d")
        if raw_size is not None:
            from o3b.cv.geometry.transform import transf3d_broadcast as _t3d
            size3d = raw_size.float()
            L = size3d.max().clamp(min=1e-6)
            hx, hy, hz = (size3d / L).tolist()  # NCDS half-extents (max axis = 1.0)
            corners_ncds = torch.tensor([
                [-hx, -hy, -hz], [ hx, -hy, -hz],
                [ hx,  hy, -hz], [-hx,  hy, -hz],
                [-hx, -hy,  hz], [ hx, -hy,  hz],
                [ hx,  hy,  hz], [-hx,  hy,  hz],
            ], dtype=torch.float32)
            cam_bbox3d = _t3d(corners_ncds, pred_pose.float())  # (8, 3)
    else:
        raw_bbox = _get("cam_bbox3d")
        if raw_bbox is not None and raw_bbox.shape == torch.Size([8, 3]):
            cam_bbox3d = raw_bbox

    return FrameObject(
        frame_id="",
        object_id="",
        frame_object_id=f"{prefix}{item_id}",
        cam_intr4x4=_get("cam_intr4x4"),
        rgb=_get("rgb"),
        depth=_get("depth"),
        depth_mask=_get("depth_mask"),
        fo_mask=_get("fo_mask"),
        fo_mask_amodal=_get("fo_mask_amodal"),
        cam_bbox3d=cam_bbox3d,
        cam_tform4x4_obj=cam_tform4x4_obj,
        mesh=_get("pred_mesh"),
        obj_kpts3d=_get("obj_kpts3d"),
        obj_kpts3d_mask=_get("obj_kpts3d_mask"),
    )


def get_frame_object_list_from_batch(batch, prefix: str = "") -> list[FrameObject]:
    """Extract all FrameObjects from a batch as a list.

    See get_frame_object_from_batch for field/prefix semantics.
    """
    B = None
    for key in ("rgb", "cam_intr4x4", "depth", "fo_mask"):
        v = getattr(batch, f"{prefix}{key}", None)
        if isinstance(v, torch.Tensor):
            B = v.shape[0]
            break
        if isinstance(v, list):
            B = len(v)
            break
    if B is None:
        return []
    return [get_frame_object_from_batch(batch, i, prefix=prefix) for i in range(B)]
