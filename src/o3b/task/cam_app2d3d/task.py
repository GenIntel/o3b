from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

from o3b.task.task import OD3D_Task, register_task


@register_task("CamApp2D3D")
class CamApp2D3D(OD3D_Task):
    """Appearance loss: cross-entropy between projected 2D image features and
    the 3D vertex feature bank, under the GT pose.

    Ported from ``od3d.methods.nemo.method.py``'s
    ``self.config.train.loss.appear`` branch, built on o3b's existing port of
    ``OD3D_Objects3D.get_label_and_sim_feats2d_img_to_all``. Constructor field
    names match ``od3d/config/methods/common3d.yaml``'s ``train.loss.appear``
    block (``type``, ``dense_loss``, ``dense_detach_geo``, ``dropout``,
    ``use_mask_rgb``, ``use_mask_object``, ``weight``); only the
    ``cross_entropy*`` family is ported (``nll_*`` / ``l2*`` / ``sim_max*``
    from NeMo are not).

    forward() returns ``(loss, metrics)``: ``loss`` is already scaled by
    ``weight`` and carries gradients into ``meshes`` (feature bank) and
    ``featmap`` (model head); ``metrics`` holds detached floats for logging.
    """

    def __init__(
        self,
        weight: float = 0.1,
        type: str = "cross_entropy_coarse",
        dense_loss: bool = True,
        dense_detach_geo: bool = False,
        dropout: float = 0.0,
        use_mask_rgb: bool = True,
        use_mask_object: bool = False,
        add_clutter: bool = True,
        add_other_objects: bool = True,
        sample_clutter_count: int = 5,
        sim_temp: float = 0.07,
        pose_field: str = "cam_tform4x4_obj_ncds",
        pose_convention: str = "gl",
    ):
        if "cross_entropy" not in type:
            raise NotImplementedError(
                f"CamApp2D3D only supports cross_entropy* appear types "
                f"(got {type!r}); nll_*/l2*/sim_max* from NeMo are not ported."
            )
        self.weight = float(weight)
        self.dense_loss = bool(dense_loss)
        self.dense_detach_geo = bool(dense_detach_geo)
        self.smooth_labels = "smooth" in type
        self.coarse_labels = "coarse" in type
        # use_mask_rgb mirrors NeMo's DTD-background-composite mask
        # (``batch.rgb_mask``); HouseCorr3D has no such modality, so this is a
        # no-op unless the batch happens to carry one.
        self.use_mask_rgb = bool(use_mask_rgb)
        self.use_mask_object = bool(use_mask_object)
        self.add_clutter = bool(add_clutter)
        self.add_other_objects = bool(add_other_objects)
        self.sample_clutter_count = int(sample_clutter_count)
        self.sim_temp = float(sim_temp)
        # dataset poses are OpenGL (-Z forward); the rasterizer expects
        # OpenCV (+Z forward) — "gl" applies the CV←GL flip before projecting
        self.pose_field = pose_field
        self.pose_convention = pose_convention

        self.criterion = torch.nn.CrossEntropyLoss()
        self.dropout1d = torch.nn.Dropout1d(p=float(dropout))

    def is_active(self) -> bool:
        return self.weight > 0.0

    def __call__(self, batch, **kwargs):
        return self.forward(batch, **kwargs)

    def forward(
        self,
        batch,
        meshes=None,
        featmap: Optional[torch.Tensor] = None,
        objects_ids: Optional[torch.Tensor] = None,
        instance_deform=None,
        return_qualit: bool = False,
    ):
        from o3b.cv.visual.resize import resize

        H, W = batch.rgb.shape[-2:]
        Hf, Wf = featmap.shape[-2:]
        device = featmap.device

        cams_tform4x4_obj = self._cams_tform4x4_obj(batch)
        # featmap resolution isn't always a clean integer divisor of (H, W)
        # (e.g. DINOv2 + a ResNetHead with pre_upsampling) — downscale the
        # intrinsics directly to the featmap's actual size instead of using a
        # single down_sample_rate, which would round (Hf, Wf) inconsistently
        # between this mask and the render call inside get_label_feats2d_img.
        cams_intr4x4 = batch.cam_intr4x4.clone().float()
        cams_intr4x4[:, 0] *= Wf / W
        cams_intr4x4[:, 1] *= Hf / H

        feats2d_img_mask = torch.ones((featmap.shape[0], 1, Hf, Wf), device=device)
        if self.use_mask_rgb and getattr(batch, "rgb_mask", None) is not None:
            feats2d_img_mask = feats2d_img_mask * resize(batch.rgb_mask, H_out=Hf, W_out=Wf)
        if self.use_mask_object and getattr(batch, "fo_mask", None) is not None:
            feats2d_img_mask = feats2d_img_mask * resize(
                batch.fo_mask.float().unsqueeze(1), H_out=Hf, W_out=Wf,
            )

        labels, labels_mask, _noise_pxl2d, sim = meshes.get_label_and_sim_feats2d_img_to_all(
            feats2d_img=featmap,
            imgs_sizes=torch.LongTensor([Hf, Wf]).to(device),
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            down_sample_rate=1.0,
            feats2d_img_mask=feats2d_img_mask,
            add_clutter=self.add_clutter,
            add_other_objects=self.add_other_objects,
            sample_clutter_count=self.sample_clutter_count,
            dense=self.dense_loss,
            smooth_labels=self.smooth_labels,
            coarse_labels=self.coarse_labels,
            sim_temp=self.sim_temp,
            instance_deform=instance_deform,
            detach_objects=self.dense_detach_geo,
            detach_deform=self.dense_detach_geo,
        )

        # dense: sim/labels are (B, V(+1), H, W) soft distributions per pixel;
        # dropout1d zeroes whole pixel positions (matches NeMo's
        # training_step: sim.flatten(2).permute(0,2,1) → drop over the H*W
        # "channel" dim → permute back)
        if sim.dim() == 4:
            sim = self.dropout1d(sim.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(*sim.shape)
        elif sim.dim() == 2:
            sim = self.dropout1d(sim)

        loss_app = self.criterion(sim, labels) * self.weight
        metrics = {"loss_app": float(loss_app.detach())}
        return loss_app, metrics

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
