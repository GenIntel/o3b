from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

from o3b.task.task import OD3D_Task, register_task


@register_task("CamGeo2D3D")
class CamGeo2D3D(OD3D_Task):
    """Geometry losses between the rendered meshes and 2D/GT observations.

    Ported from ``od3d.methods.nemo.method.py``'s
    ``self.config.train.loss.geo`` branch (``get_geo_mask_metrics`` plus the
    sdf/deform regularizers). Constructor field names and nesting match
    ``od3d/config/methods/common3d.yaml``'s ``train.loss.geo`` block — each
    sub-loss is a ``{weight: ...}`` dict, independently gated by its own
    weight, mirroring NeMo's per-sub-loss ``if weight > 0.0`` guards.

    Sub-losses requiring data this pipeline doesn't produce are accepted for
    config-shape compatibility but contribute nothing unless their inputs are
    actually available:
    - ``mask_dt`` / ``mask_inv_dt``: need GT mask distance-transform
      modalities (``batch.mask_dt`` / ``batch.mask_inv_dt``), not produced by
      HouseCorr3DFrame.
    - ``deform_reg`` / ``deform_smooth_reg`` / ``deform_latent_reg``: need an
      ``instance_deform`` (NeMo's per-instance mesh deformation net output);
      MorpheusMethod.train_method doesn't compute one, so these are always 0
      (matching NeMo's own behaviour when ``instance_deform is None``).
    - ``smooth``: NeMo itself skips this for DMTet/Flexicubes meshes (their
      geometry has no smoothness regularizer of this form) — MorpheusMethod
      always uses DMTet, so this is always a no-op here too.
    - ``rec`` (chamfer/point-face distance against a per-sample GT mesh) is
      not ported: ``FrameObjectBatch`` carries one shared mesh, not a
      per-sample GT mesh, so NeMo's reconstruction loss has no batch shape to
      operate on in this pipeline.

    forward() returns ``(loss, metrics)``: ``loss`` sums the already-weighted
    active sub-losses and carries gradients into the meshes' geometry (e.g.
    DMTet SDF); ``metrics`` holds detached per-sub-loss floats for logging.
    """

    def __init__(
        self,
        mask: Optional[dict] = None,
        mask_mse: Optional[dict] = None,
        mask_dt: Optional[dict] = None,
        mask_inv_dt: Optional[dict] = None,
        sdf_reg: Optional[dict] = None,
        smooth: Optional[dict] = None,
        deform_reg: Optional[dict] = None,
        deform_smooth_reg: Optional[dict] = None,
        deform_latent_reg: Optional[dict] = None,
        down_sample_rate: float = 8.0,
        pose_field: str = "cam_tform4x4_obj_ncds",
        pose_convention: str = "gl",
    ):
        self.w_mask = float((mask or {}).get("weight", 0.0))
        self.w_mask_mse = float((mask_mse or {}).get("weight", 0.0))
        self.w_mask_dt = float((mask_dt or {}).get("weight", 0.0))
        self.w_mask_inv_dt = float((mask_inv_dt or {}).get("weight", 0.0))
        self.w_sdf_reg = float((sdf_reg or {}).get("weight", 0.0))
        self.w_smooth = float((smooth or {}).get("weight", 0.0))
        self.w_deform_reg = float((deform_reg or {}).get("weight", 0.0))
        self.w_deform_smooth_reg = float((deform_smooth_reg or {}).get("weight", 0.0))
        self.deform_latent_reg_type = (deform_latent_reg or {}).get("type", "unit")
        self.w_deform_latent_reg = float((deform_latent_reg or {}).get("weight", 0.0))
        # render resolution for the mask sub-losses: (H, W) // down_sample_rate.
        # Independent of any feature backbone (unlike CamApp2D3D's, which is
        # derived from the actual featmap) since geo doesn't need one.
        self.down_sample_rate = float(down_sample_rate)
        # dataset poses are OpenGL (-Z forward); the rasterizer expects
        # OpenCV (+Z forward) — "gl" applies the CV←GL flip before rendering
        self.pose_field = pose_field
        self.pose_convention = pose_convention

    def is_active(self) -> bool:
        return any([
            self.w_mask > 0.0, self.w_mask_mse > 0.0,
            self.w_mask_dt > 0.0, self.w_mask_inv_dt > 0.0,
            self.w_sdf_reg > 0.0, self.w_smooth > 0.0,
            self.w_deform_reg > 0.0, self.w_deform_smooth_reg > 0.0,
            self.w_deform_latent_reg > 0.0,
        ])

    def __call__(self, batch, **kwargs):
        return self.forward(batch, **kwargs)

    def forward(
        self,
        batch,
        meshes=None,
        objects_ids: Optional[torch.Tensor] = None,
        instance_deform=None,
        return_qualit: bool = False,
    ):
        from o3b.cv.geometry.objects3d.objects3d import PROJECT_MODALITIES
        from o3b.cv.visual.resize import resize

        H, W = batch.rgb.shape[-2:]
        cams_tform4x4_obj = self._cams_tform4x4_obj(batch)
        device = cams_tform4x4_obj.device
        B = cams_tform4x4_obj.shape[0]
        if objects_ids is None:
            objects_ids = torch.zeros(B, dtype=torch.long, device=device)

        loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}

        need_mask = (self.w_mask > 0 or self.w_mask_mse > 0
                     or self.w_mask_dt > 0 or self.w_mask_inv_dt > 0)
        if need_mask:
            # single (non-list) modality → render() returns the tensor
            # directly, matching NeMo's get_geo_mask_metrics call style
            pred_masks = meshes.render(
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                imgs_sizes=torch.LongTensor([H, W]).to(device),
                objects_ids=objects_ids,
                modalities=PROJECT_MODALITIES.MASK,
                instance_deform=instance_deform,
                down_sample_rate=self.down_sample_rate,
            )  # (B, 1, h, w)
            gt_masks = resize(
                (batch.fo_mask.float().unsqueeze(1) > 0.5) * 1.0,
                H_out=pred_masks.shape[-2], W_out=pred_masks.shape[-1],
            )

            if self.w_mask > 0:
                # NeMo: numerator differentiable, union (denominator) detached
                inter = gt_masks * pred_masks
                union = inter + (1.0 - gt_masks) * pred_masks + (1.0 - pred_masks) * gt_masks
                iou = inter.flatten(1).sum(dim=-1) / union.detach().flatten(1).sum(dim=-1) + 1e-10
                loss_mask = -iou.mean() * self.w_mask
                loss = loss + loss_mask
                metrics["loss_geo_mask"] = float(loss_mask.detach())

            if self.w_mask_mse > 0:
                loss_mask_mse = ((gt_masks - pred_masks) ** 2).flatten(1).mean(dim=-1).mean() * self.w_mask_mse
                loss = loss + loss_mask_mse
                metrics["loss_geo_mask_mse"] = float(loss_mask_mse.detach())

            if self.w_mask_dt > 0:
                gt_mask_dt = getattr(batch, "mask_dt", None)
                if gt_mask_dt is not None:
                    gt_dt = resize(gt_mask_dt.float().unsqueeze(1),
                                   H_out=pred_masks.shape[-2], W_out=pred_masks.shape[-1])
                    loss_mask_dt = -(gt_dt * pred_masks).flatten(1).mean(dim=-1).mean() * self.w_mask_dt
                    loss = loss + loss_mask_dt
                    metrics["loss_geo_mask_dt"] = float(loss_mask_dt.detach())
                else:
                    logger.debug("CamGeo2D3D: mask_dt.weight > 0 but batch has no mask_dt; skipping.")

            if self.w_mask_inv_dt > 0:
                gt_mask_inv_dt = getattr(batch, "mask_inv_dt", None)
                if gt_mask_inv_dt is not None:
                    gt_inv_dt = resize(gt_mask_inv_dt.float().unsqueeze(1),
                                       H_out=pred_masks.shape[-2], W_out=pred_masks.shape[-1])
                    pred_masks_inv = 1.0 - pred_masks
                    loss_mask_inv_dt = -(gt_inv_dt * pred_masks_inv).flatten(1).mean(dim=-1).mean() * self.w_mask_inv_dt
                    loss = loss + loss_mask_inv_dt
                    metrics["loss_geo_mask_inv_dt"] = float(loss_mask_inv_dt.detach())
                else:
                    logger.debug("CamGeo2D3D: mask_inv_dt.weight > 0 but batch has no mask_inv_dt; skipping.")

        if self.w_sdf_reg > 0 and hasattr(meshes, "get_geo_sdf_reg_loss"):
            loss_sdf_reg = meshes.get_geo_sdf_reg_loss(objects_ids=objects_ids).mean() * self.w_sdf_reg
            loss = loss + loss_sdf_reg
            metrics["loss_geo_sdf_reg"] = float(loss_sdf_reg.detach())

        if self.w_smooth > 0:
            from o3b.cv.geometry.objects3d.dmtet_x_gaussians.dmtet_x_gaussians import DMTet_x_Gaussians
            from o3b.cv.geometry.objects3d.flexicubes.flexicubes import Flexicubes
            if not isinstance(meshes, (DMTet_x_Gaussians, Flexicubes)):
                loss_smooth = meshes.get_geo_smooth_loss(
                    objects_ids=objects_ids, instance_deform=None,
                ).mean() * self.w_smooth
                loss = loss + loss_smooth
                metrics["loss_geo_smooth"] = float(loss_smooth.detach())
            # DMTet/Flexicubes: NeMo skips this sub-loss entirely (matched)

        if self.w_deform_reg > 0:
            if instance_deform is not None:
                loss_deform_reg = instance_deform.verts_deform.norm(p=2, dim=-1).mean() * self.w_deform_reg
            else:
                loss_deform_reg = torch.zeros((), device=device)
            loss = loss + loss_deform_reg
            metrics["loss_geo_deform_reg"] = float(loss_deform_reg.detach())

        if self.w_deform_smooth_reg > 0:
            if instance_deform is not None:
                loss_deform_smooth_reg = meshes.get_geo_deform_smooth_loss(
                    objects_ids=objects_ids, instance_deform=instance_deform,
                ).mean() * self.w_deform_smooth_reg
            else:
                loss_deform_smooth_reg = torch.zeros((), device=device)
            loss = loss + loss_deform_smooth_reg
            metrics["loss_geo_deform_smooth_reg"] = float(loss_deform_smooth_reg.detach())

        if self.w_deform_latent_reg > 0 and instance_deform is not None and instance_deform.latent is not None:
            if self.deform_latent_reg_type == "unit":
                loss_deform_latent_reg = (
                    (instance_deform.latent.norm(dim=-1) - 1.0).abs() ** 2
                ).mean() * self.w_deform_latent_reg
                loss = loss + loss_deform_latent_reg
                metrics["loss_geo_deform_latent_reg"] = float(loss_deform_latent_reg.detach())
            # "kl" type not ported: requires instance_deform.latent_mu/logvar,
            # which our (currently unused) instance-deform path doesn't set.

        metrics["loss_geo"] = float(loss.detach())
        return loss, metrics

    def _cams_tform4x4_obj(self, batch):
        cams_tform4x4_obj = getattr(batch, self.pose_field, None)
        if cams_tform4x4_obj is None:
            cams_tform4x4_obj = batch.cam_tform4x4_obj
        if self.pose_convention == "gl":
            flip = torch.diag(torch.tensor(
                [1.0, -1.0, -1.0, 1.0],
                device=cams_tform4x4_obj.device, dtype=cams_tform4x4_obj.dtype,
            ))
            cams_tform4x4_obj = flip @ cams_tform4x4_obj
        return cams_tform4x4_obj
