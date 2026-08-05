from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

from o3b.task.task import OD3D_Task, register_task


@register_task("ObjGeo3D")
class ObjGeo3D(OD3D_Task):
    """Object-space geometry regularizers — no camera, no image observations.

    Split out of :class:`~o3b.task.cam_geo2d3d.task.CamGeo2D3D`, which keeps the
    2D-observation terms (the mask sub-losses). These act purely on the 3D
    object: the DMTet SDF field, the mesh's own smoothness, and the per-instance
    deformation. They need neither ``batch.cam_tform4x4_obj`` nor
    ``batch.fo_mask``, so they can be applied without a rendered view.

    Ported from ``od3d.methods.nemo.method.py``'s ``train.loss.geo`` branch (the
    sdf/smooth/deform regularizers). Constructor field names and nesting match
    ``od3d/config/methods/common3d.yaml``'s ``train.loss.geo`` block — each
    sub-loss is a ``{weight: ...}`` dict, independently gated by its own weight,
    mirroring NeMo's per-sub-loss ``if weight > 0.0`` guards.

    - ``sdf_reg``: eikonal regularizer on the DMTet SDF gradient
      (``meshes.get_geo_sdf_reg_loss``). Routes gradient straight into the SDF
      MLP parameters rather than through the marching-tets verts.
    - ``smooth``: NeMo itself skips this for DMTet/Flexicubes meshes (their
      geometry has no smoothness regularizer of this form), so it is a no-op for
      those mesh types — matched here.
    - ``deform_reg`` / ``deform_smooth_reg`` / ``deform_latent_reg``: need an
      ``instance_deform`` (NeMo's per-instance mesh deformation net output);
      they return 0 when it is None, matching NeMo's own behaviour.
    - ``pcl_cd``: symmetric chamfer distance between the predicted mesh's
      vertices and the GT mesh's vertices, ported from
      ``od3d.methods.nemo.method.get_geo_loss``'s ``cd_mesh`` branch. This is
      the one sub-loss that reads ``batch`` — it needs the GT meshes
      (``batch.meshes``). Returns 0 when the batch carries none.

    forward() returns ``(loss, metrics)``: ``loss`` sums the already-weighted
    active sub-losses; ``metrics`` holds detached per-sub-loss floats for
    logging, keyed ``loss_obj_<name>``. With ``return_qualit=True`` a third
    element is returned: a dict of ``{name: HxWx3 float numpy image}`` showing
    the predicted and the GT object from several viewpoints (see
    :meth:`_build_qualit`).
    """

    def __init__(
        self,
        sdf_reg: Optional[dict] = None,
        smooth: Optional[dict] = None,
        deform_reg: Optional[dict] = None,
        deform_smooth_reg: Optional[dict] = None,
        deform_latent_reg: Optional[dict] = None,
        pcl_cd: Optional[dict] = None,
        qualit_max_samples: int = 4,
        qualit_res: int = 256,
        qualit_views: int = 4,
    ):
        self.w_sdf_reg = float((sdf_reg or {}).get("weight", 0.0))
        self.w_smooth = float((smooth or {}).get("weight", 0.0))
        self.w_deform_reg = float((deform_reg or {}).get("weight", 0.0))
        self.w_deform_smooth_reg = float((deform_smooth_reg or {}).get("weight", 0.0))
        self.deform_latent_reg_type = (deform_latent_reg or {}).get("type", "unit")
        self.w_deform_latent_reg = float((deform_latent_reg or {}).get("weight", 0.0))
        self.w_pcl_cd = float((pcl_cd or {}).get("weight", 0.0))
        # qualitative object comparison: how many samples of a batch are drawn
        # (one grid row each), the panel resolution, and how many viewpoints
        # (grid columns) the objects are splatted from
        self.qualit_max_samples = int(qualit_max_samples)
        self.qualit_res = int(qualit_res)
        self.qualit_views = int(qualit_views)

    def is_active(self) -> bool:
        return any([
            self.w_sdf_reg > 0.0, self.w_smooth > 0.0,
            self.w_deform_reg > 0.0, self.w_deform_smooth_reg > 0.0,
            self.w_deform_latent_reg > 0.0, self.w_pcl_cd > 0.0,
        ])

    def __call__(self, batch=None, **kwargs):
        return self.forward(batch, **kwargs)

    def forward(
        self,
        batch=None,
        meshes=None,
        objects_ids: Optional[torch.Tensor] = None,
        instance_deform=None,
        return_qualit: bool = False,
    ):
        """Args:
            batch: only read by ``pcl_cd`` (for ``batch.meshes``); the other
                regularizers need nothing but ``meshes``, so it may be None.
            meshes: the OD3D_Objects3D whose geometry is regularized.
            objects_ids: (B,) mesh id per batch element.
            instance_deform: OD3D_Objects3D_Deform or None.
        """
        device = getattr(meshes, "device", None) or torch.device("cpu")
        if objects_ids is None:
            objects_ids = torch.zeros(1, dtype=torch.long, device=device)
        device = objects_ids.device

        loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}

        if self.w_sdf_reg > 0 and hasattr(meshes, "get_geo_sdf_reg_loss"):
            loss_sdf_reg = meshes.get_geo_sdf_reg_loss(objects_ids=objects_ids).mean() * self.w_sdf_reg
            loss = loss + loss_sdf_reg
            metrics["loss_obj_sdf_reg"] = float(loss_sdf_reg.detach())

        if self.w_smooth > 0:
            from o3b.cv.geometry.objects3d.dmtet_x_gaussians.dmtet_x_gaussians import DMTet_x_Gaussians
            from o3b.cv.geometry.objects3d.flexicubes.flexicubes import Flexicubes
            if not isinstance(meshes, (DMTet_x_Gaussians, Flexicubes)):
                loss_smooth = meshes.get_geo_smooth_loss(
                    objects_ids=objects_ids, instance_deform=None,
                ).mean() * self.w_smooth
                loss = loss + loss_smooth
                metrics["loss_obj_smooth"] = float(loss_smooth.detach())
            # DMTet/Flexicubes: NeMo skips this sub-loss entirely (matched)

        if self.w_deform_reg > 0:
            if instance_deform is not None:
                loss_deform_reg = instance_deform.verts_deform.norm(p=2, dim=-1).mean() * self.w_deform_reg
            else:
                loss_deform_reg = torch.zeros((), device=device)
            loss = loss + loss_deform_reg
            metrics["loss_obj_deform_reg"] = float(loss_deform_reg.detach())

        if self.w_deform_smooth_reg > 0:
            if instance_deform is not None:
                loss_deform_smooth_reg = meshes.get_geo_deform_smooth_loss(
                    objects_ids=objects_ids, instance_deform=instance_deform,
                ).mean() * self.w_deform_smooth_reg
            else:
                loss_deform_smooth_reg = torch.zeros((), device=device)
            loss = loss + loss_deform_smooth_reg
            metrics["loss_obj_deform_smooth_reg"] = float(loss_deform_smooth_reg.detach())

        if self.w_deform_latent_reg > 0 and instance_deform is not None and instance_deform.latent is not None:
            if self.deform_latent_reg_type == "unit":
                loss_deform_latent_reg = (
                    (instance_deform.latent.norm(dim=-1) - 1.0).abs() ** 2
                ).mean() * self.w_deform_latent_reg
                loss = loss + loss_deform_latent_reg
                metrics["loss_obj_deform_latent_reg"] = float(loss_deform_latent_reg.detach())
            # "kl" type not ported: requires instance_deform.latent_mu/logvar,
            # which our (currently unused) instance-deform path doesn't set.

        cd = None
        if self.w_pcl_cd > 0:
            cd = self._pcl_chamfer_distance(batch, meshes, objects_ids, instance_deform)
            if cd is None:
                logger.warning("ObjGeo3D: pcl_cd.weight > 0 but the batch carries no "
                               "GT meshes (batch.meshes); skipping.")
            else:
                loss_pcl_cd = cd.mean() * self.w_pcl_cd
                loss = loss + loss_pcl_cd
                metrics["loss_obj_pcl_cd"] = float(loss_pcl_cd.detach())

        metrics["loss_obj_geo"] = float(loss.detach())

        if return_qualit:
            qualit: dict = {}
            try:
                # cd is None unless pcl_cd computed it — the images do not pay
                # for a chamfer distance just to label their rows
                qualit = self._build_qualit(
                    batch, meshes, objects_ids, instance_deform, cd,
                )
            except Exception as e:
                logger.warning(f"ObjGeo3D: qualit rendering failed: {e}", exc_info=True)
            return loss, metrics, qualit

        return loss, metrics

    # ── qualitative: predicted object vs. GT object ───────────────────────────

    def _build_qualit(self, batch, meshes, objects_ids, instance_deform, cd=None) -> dict:
        """Grid showing the predicted and the GT object from several viewpoints.

        One row per sample (up to ``qualit_max_samples``), one panel per
        viewpoint (``qualit_views`` azimuths around the object at a fixed
        elevation). Both objects are splatted as point clouds into the same
        panel — the GT into the red channel, the prediction into the green one,
        so overlap reads yellow, unexplained GT geometry red and hallucinated
        predicted geometry green. Brightness falls off with depth so the shape
        stays readable. The first panel of each row carries the colour legend,
        plus the sample's chamfer distance when ``pcl_cd`` computed one.

        Both clouds are in the same (NCDS) frame — see
        :meth:`_pcl_chamfer_distance` — so they are drawn with one shared
        camera per row, fitted to their joint extent.

        Args:
            cd: (B,) chamfer distances used for the row labels, or None when
                ``pcl_cd`` is inactive (no chamfer is computed for the images).

        Returns:
            ``{"obj_geo_pcl": HxWx3 float32 numpy image in [0, 1]}`` — empty if
            the prediction has no vertices.
        """
        import math

        from o3b.cv.geometry.transform import transf4x4_from_spherical
        from o3b.cv.visual.draw import draw_text_in_rgb
        from o3b.cv.visual.show import imgs_to_img

        pred_verts, pred_mask = self._get_pred_verts(meshes, objects_ids, instance_deform)
        pred_verts = pred_verts.detach().float().cpu()
        pred_mask = pred_mask.detach().cpu()
        gt_verts, gt_mask = self._get_gt_verts(batch, objects_ids, dtype=torch.float32)
        if gt_verts is not None:
            gt_verts = gt_verts.detach().float().cpu()
            gt_mask = gt_mask.detach().cpu()

        n = min(self.qualit_max_samples, pred_verts.shape[0])
        V = max(1, self.qualit_views)
        res = self.qualit_res

        # viewpoints: V azimuths around the object at a slight elevation
        azim = torch.linspace(0.0, 2 * math.pi, V + 1)[:V]
        cams_tform4x4_obj = transf4x4_from_spherical(
            azim=azim,
            elev=torch.full((V,), math.pi / 9),
            theta=torch.zeros(V),
            dist=torch.full((V,), 3.0),
        )  # (V, 4, 4), OpenGL convention (-Z forward, +Y up)
        # ``_proj`` is a plain pinhole (+Z forward, +Y down) — flip GL → CV so
        # the object is in front of the camera and drawn the right way up
        cams_tform4x4_obj = torch.diag(
            torch.tensor([1.0, -1.0, -1.0, 1.0]),
        ) @ cams_tform4x4_obj

        rows = []
        for b in range(n):
            pred_b = pred_verts[b][pred_mask[b]] if pred_mask is not None else pred_verts[b]
            gt_b = (gt_verts[b][gt_mask[b]]
                    if gt_verts is not None else torch.zeros((0, 3)))
            if len(pred_b) == 0 and len(gt_b) == 0:
                continue

            # one focal length per row, fitted so the joint cloud fills the
            # panel — both objects stay directly comparable across viewpoints
            pts_all = torch.cat([pred_b, gt_b], dim=0)
            focal = self._fit_focal(pts_all, cams_tform4x4_obj, res)

            panels = []
            for v in range(V):
                gt_splat = self._splat(gt_b, cams_tform4x4_obj[v], focal, res)
                pred_splat = self._splat(pred_b, cams_tform4x4_obj[v], focal, res)
                panels.append(torch.stack(
                    [gt_splat, pred_splat, torch.zeros_like(gt_splat)], dim=0,
                ))  # (3, res, res)

            label = "gt red | pred green"
            if cd is not None and b < len(cd):
                label += f"\ncd {float(cd[b]):.4f}"
            elif gt_verts is None:
                label = "pred green (no gt)"
            panels[0] = draw_text_in_rgb(panels[0], text=label,
                                         fontColor=(255, 255, 255))
            rows.append(panels)

        if not rows:
            return {}

        img = imgs_to_img(rows, pad=2, pad_value=1.0)  # (3, n*res, V*res)
        return {"obj_geo_pcl": img.permute(1, 2, 0).numpy()}

    @staticmethod
    def _proj(pts3d, cam_tform4x4_obj, focal, res):
        """Project (N, 3) object-space points into a centred pinhole camera.

        Returns ``(uv (M, 2), z (M,))`` for the points in front of the camera.
        """
        pts_cam = pts3d @ cam_tform4x4_obj[:3, :3].T + cam_tform4x4_obj[:3, 3]
        z = pts_cam[:, 2]
        front = z > 1e-6
        pts_cam, z = pts_cam[front], z[front]
        c = (res - 1) / 2.0
        uv = torch.stack([
            focal * pts_cam[:, 0] / z + c,
            # image rows grow downwards, camera +Y is down in OpenCV — the
            # spherical helper returns OpenCV poses, so no extra flip
            focal * pts_cam[:, 1] / z + c,
        ], dim=-1)
        return uv, z

    def _fit_focal(self, pts3d, cams_tform4x4_obj, res, fill: float = 0.85):
        """Focal length that makes the widest view of ``pts3d`` fill the panel."""
        if len(pts3d) == 0:
            return float(res)
        extent = 0.0
        for v in range(cams_tform4x4_obj.shape[0]):
            uv, _ = self._proj(pts3d, cams_tform4x4_obj[v], focal=1.0, res=res)
            if len(uv) == 0:
                continue
            c = (res - 1) / 2.0
            extent = max(extent, float((uv - c).abs().max()))
        if extent <= 1e-6:
            return float(res)
        return fill * ((res - 1) / 2.0) / extent

    @staticmethod
    def _splat(pts3d, cam_tform4x4_obj, focal, res, radius: int = 1):
        """Depth-shaded point splat of (N, 3) object-space points → (res, res).

        Nearer points are brighter (1.0) than far ones (0.4); each point is
        dilated to a ``2*radius+1`` square so sparse clouds stay visible.
        """
        import torch.nn.functional as F

        canvas = torch.zeros(res * res)
        if len(pts3d) == 0:
            return canvas.reshape(res, res)

        uv, z = ObjGeo3D._proj(pts3d, cam_tform4x4_obj, focal, res)
        u = uv[:, 0].round().long()
        v = uv[:, 1].round().long()
        keep = (u >= 0) & (u < res) & (v >= 0) & (v < res)
        u, v, z = u[keep], v[keep], z[keep]
        if len(u) == 0:
            return canvas.reshape(res, res)

        z_min, z_max = float(z.min()), float(z.max())
        shade = (torch.ones_like(z) if z_max - z_min < 1e-6
                 else 1.0 - 0.6 * (z - z_min) / (z_max - z_min))
        # nearest point wins where several project onto the same pixel
        canvas.scatter_reduce_(0, v * res + u, shade, reduce="amax")
        canvas = canvas.reshape(1, 1, res, res)
        if radius > 0:
            canvas = F.max_pool2d(canvas, kernel_size=2 * radius + 1,
                                  stride=1, padding=radius)
        return canvas[0, 0]

    def _pcl_chamfer_distance(self, batch, meshes, objects_ids, instance_deform):
        """Symmetric chamfer distance between predicted and GT mesh vertices.

        Ported from ``od3d.methods.nemo.method.get_geo_loss``'s ``cd_mesh``
        branch: both point clouds are the meshes' vertices, padded per batch
        element with a validity mask, compared by ``batch_chamfer_distance``.

        Both clouds are already in the same frame: ``o3b.io._load_mesh``
        normalises GT mesh verts so the longest axis spans [-1, 1] (that *is*
        the NCDS frame — ``obj_ncds0c_tform4x4_obj`` maps them back to metric),
        and the predicted DMTet mesh lives in the same NCDS frame the training
        poses (``cam_tform4x4_obj_ncds``) consume. So no transform is applied.

        Returns (B,) chamfer distances, or None when no GT mesh is available.
        """
        from o3b.cv.geometry.metrics.dist import batch_chamfer_distance

        pred_verts, pred_mask = self._get_pred_verts(meshes, objects_ids, instance_deform)
        gt_verts, gt_mask = self._get_gt_verts(batch, objects_ids, dtype=pred_verts.dtype)
        if gt_verts is None:
            return None

        return batch_chamfer_distance(pred_verts, gt_verts, pred_mask, gt_mask)

    def _get_pred_verts(self, meshes, objects_ids, instance_deform):
        """Predicted point cloud: the current mesh verts (+ instance deform).

        Returns ``(verts (B, P, 3), mask (B, P))`` on ``objects_ids``' device.
        """
        device = objects_ids.device
        pred_verts = meshes.get_verts_stacked_with_mesh_ids(objects_ids).to(device)
        pred_mask = meshes.get_verts_stacked_mask_with_mesh_ids(objects_ids).to(device)
        if instance_deform is not None:
            pred_verts = pred_verts + instance_deform.verts_deform[:, :pred_verts.shape[1]]
        return pred_verts, pred_mask

    def _get_gt_verts(self, batch, objects_ids, dtype=torch.float32):
        """GT point cloud: the batch meshes' verts, padded to (B, G_max, 3).

        ``batch.meshes`` is one GT mesh per sample; ``batch.mesh`` (a single
        shared mesh) is accepted as a fallback. Returns ``(verts, mask)``, or
        ``(None, None)`` when the batch carries no GT geometry.
        """
        gt_meshes = getattr(batch, "meshes", None) if batch is not None else None
        if gt_meshes is None:
            gt_mesh = getattr(batch, "mesh", None) if batch is not None else None
            if gt_mesh is None:
                return None, None
            gt_meshes = [gt_mesh] * len(objects_ids)  # one shared mesh
        if all(m is None for m in gt_meshes):
            return None, None

        device = objects_ids.device
        B = len(objects_ids)

        gt_list = []
        for b in range(B):
            m = gt_meshes[b]
            v = None if m is None else getattr(m, "verts", None)
            gt_list.append(
                torch.zeros((0, 3), device=device, dtype=dtype)
                if v is None else v.to(device=device, dtype=dtype),
            )
        G_max = max(len(v) for v in gt_list)
        if G_max == 0:
            return None, None
        gt_verts = torch.zeros((B, G_max, 3), device=device, dtype=dtype)
        gt_mask = torch.zeros((B, G_max), device=device, dtype=torch.bool)
        for b, v in enumerate(gt_list):
            gt_verts[b, :len(v)] = v
            gt_mask[b, :len(v)] = True
        return gt_verts, gt_mask
