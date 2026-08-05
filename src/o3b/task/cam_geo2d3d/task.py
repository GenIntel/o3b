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
    ``self.config.train.loss.geo`` branch (``get_geo_mask_metrics``). Constructor
    field names and nesting match ``od3d/config/methods/common3d.yaml``'s
    ``train.loss.geo`` block — each sub-loss is a ``{weight: ...}`` dict,
    independently gated by its own weight, mirroring NeMo's per-sub-loss
    ``if weight > 0.0`` guards.

    This task covers only the terms that compare a *rendered view* against the
    batch's 2D observations. NeMo's object-space regularizers (``sdf_reg``,
    ``smooth``, ``deform_reg``, ``deform_smooth_reg``, ``deform_latent_reg``)
    live in :class:`~o3b.task.obj_geo3d.task.ObjGeo3D`, which needs no camera
    and no image.

    ``mask_dt`` / ``mask_inv_dt`` use ``batch.mask_dt`` / ``batch.mask_inv_dt``
    when the dataset provides them and otherwise derive them from
    ``batch.fo_mask`` via ``get_mask_distance_transform_norm`` — the same
    fallback ``od3d.od3d_datasets.frame.Frame.get_mask_dt`` uses. See
    ``_get_mask_dt``.

    ``use_mask_amodal`` swaps the GT supervision mask (``batch.fo_mask``, the
    annotated *visible* mask) for the object's amodal silhouette under the GT
    pose — NeMo's ``train.replace_mask_with_rendered_mask``. That silhouette
    comes from the dataset's ``fo_mask_amodal`` modality when the batch carries
    one (baked in at index/shard-build time, and the matching
    ``fo_mask_amodal_dt`` with it); otherwise the batch's GT meshes are
    rasterised on the fly, see :meth:`_render_gt_mesh_masks`. Unlike NeMo,
    which only replaces ``batch.mask``, the ``mask_dt`` / ``mask_inv_dt``
    targets follow the same silhouette, so every sub-loss is supervised by one
    mask.

    ``rec`` (chamfer/point-face distance against a per-sample GT mesh) is not
    ported: ``FrameObjectBatch`` carries one shared mesh, not a per-sample GT
    mesh, so NeMo's reconstruction loss has no batch shape to operate on here.

    forward() returns ``(loss, metrics)``: ``loss`` sums the already-weighted
    active sub-losses and carries gradients into the meshes' geometry (e.g.
    DMTet SDF); ``metrics`` holds detached per-sub-loss floats for logging.
    With ``return_qualit=True`` a third element is returned: a dict of
    ``{name: HxWx3 float numpy image}`` comparing the rendered mask against the
    GT mask (see :meth:`_build_qualit`).
    """

    def __init__(
        self,
        mask: Optional[dict] = None,
        mask_mse: Optional[dict] = None,
        mask_dt: Optional[dict] = None,
        mask_inv_dt: Optional[dict] = None,
        down_sample_rate: float = 8.0,
        pose_field: str = "cam_tform4x4_obj_ncds",
        pose_convention: str = "gl",
        use_mask_amodal: bool = False,
        qualit_max_samples: int = 4,
        qualit_res: int = 256,
    ):
        self.w_mask = float((mask or {}).get("weight", 0.0))
        self.w_mask_mse = float((mask_mse or {}).get("weight", 0.0))
        self.w_mask_dt = float((mask_dt or {}).get("weight", 0.0))
        self.w_mask_inv_dt = float((mask_inv_dt or {}).get("weight", 0.0))
        # render resolution for the mask sub-losses: (H, W) // down_sample_rate.
        # Independent of any feature backbone (unlike CamApp2D3D's, which is
        # derived from the actual featmap) since geo doesn't need one.
        self.down_sample_rate = float(down_sample_rate)
        # dataset poses are OpenGL (-Z forward); the rasterizer expects
        # OpenCV (+Z forward) — "gl" applies the CV←GL flip before rendering
        self.pose_field = pose_field
        self.pose_convention = pose_convention
        # supervise against the object's amodal silhouette instead of the
        # dataset's fo_mask (NeMo's train.replace_mask_with_rendered_mask)
        self.use_mask_amodal = bool(use_mask_amodal)
        # warn once, not once per batch, when the replacement can't be honoured
        self._warned_no_gt_meshes = False
        # qualitative mask comparison: how many samples of a batch are drawn
        # (one grid row each) and the longer side each panel is scaled to
        self.qualit_max_samples = int(qualit_max_samples)
        self.qualit_res = int(qualit_res)

    def is_active(self) -> bool:
        return any([
            self.w_mask > 0.0, self.w_mask_mse > 0.0,
            self.w_mask_dt > 0.0, self.w_mask_inv_dt > 0.0,
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
        qualit: dict = {}

        need_mask = (self.w_mask > 0 or self.w_mask_mse > 0
                     or self.w_mask_dt > 0 or self.w_mask_inv_dt > 0
                     or return_qualit)
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

            # GT silhouette. With use_mask_amodal: the dataset's baked
            # fo_mask_amodal if it carries one, else the GT meshes rasterised
            # here and now. Otherwise the annotated fo_mask.
            # ``gt_masks_render`` / ``amodal`` also select which distance
            # transform the mask_dt / mask_inv_dt targets come from.
            H_out, W_out = pred_masks.shape[-2], pred_masks.shape[-1]
            gt_masks_render = None
            amodal = None
            if self.use_mask_amodal:
                amodal = getattr(batch, "fo_mask_amodal", None)
                if amodal is None:
                    gt_masks_render = self._render_gt_mesh_masks(
                        batch, cams_tform4x4_obj, H, W, meshes,
                        H_out=H_out, W_out=W_out,
                    )

            # dataset modalities may still sit on the CPU (the dataloader's
            # device); .to() is a no-op once a caller has moved them
            if amodal is not None:
                gt_masks = self._binarise(amodal.to(device), H_out, W_out)
            elif gt_masks_render is not None:
                gt_masks = gt_masks_render
            else:
                gt_masks = self._binarise(batch.fo_mask.to(device), H_out, W_out)

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
                gt_mask_dt = self._get_mask_dt(batch, inv=False, device=device,
                                               mask=gt_masks_render,
                                               amodal=amodal is not None)
                if gt_mask_dt is not None:
                    gt_dt = resize(gt_mask_dt,
                                   H_out=pred_masks.shape[-2], W_out=pred_masks.shape[-1])
                    loss_mask_dt = -(gt_dt * pred_masks).flatten(1).mean(dim=-1).mean() * self.w_mask_dt
                    loss = loss + loss_mask_dt
                    metrics["loss_geo_mask_dt"] = float(loss_mask_dt.detach())
                else:
                    logger.warning("CamGeo2D3D: mask_dt.weight > 0 but the batch has "
                                   "neither mask_dt nor fo_mask; skipping.")

            if self.w_mask_inv_dt > 0:
                gt_mask_inv_dt = self._get_mask_dt(batch, inv=True, device=device,
                                                   mask=gt_masks_render,
                                                   amodal=amodal is not None)
                if gt_mask_inv_dt is not None:
                    gt_inv_dt = resize(gt_mask_inv_dt,
                                       H_out=pred_masks.shape[-2], W_out=pred_masks.shape[-1])
                    pred_masks_inv = 1.0 - pred_masks
                    loss_mask_inv_dt = -(gt_inv_dt * pred_masks_inv).flatten(1).mean(dim=-1).mean() * self.w_mask_inv_dt
                    loss = loss + loss_mask_inv_dt
                    metrics["loss_geo_mask_inv_dt"] = float(loss_mask_inv_dt.detach())
                else:
                    logger.warning("CamGeo2D3D: mask_inv_dt.weight > 0 but the batch has "
                                   "neither mask_inv_dt nor fo_mask; skipping.")

            if return_qualit:
                try:
                    qualit = self._build_qualit(batch, pred_masks, gt_masks)
                except Exception as e:
                    logger.warning(f"CamGeo2D3D: qualit rendering failed: {e}",
                                   exc_info=True)

        metrics["loss_geo"] = float(loss.detach())
        if return_qualit:
            return loss, metrics, qualit
        return loss, metrics

    # ── qualitative: rendered mask vs. GT mask ────────────────────────────────

    def _build_qualit(self, batch, pred_masks, gt_masks) -> dict:
        """Grid comparing the rendered mask against the GT mask.

        One row per sample (up to ``qualit_max_samples``), four panels each:
        RGB (labelled with the sample's hard mask IoU) | RGB tinted red where
        the GT mask is | RGB tinted green where the rendered mask is | an
        agreement map putting the GT mask in the red channel and the rendered
        mask in the green one, so overlap reads yellow, missed GT area red and
        over-covering render area green.

        Args:
            pred_masks: (B, 1, h, w) soft rendered mask, render resolution.
            gt_masks:   (B, 1, h, w) binarised GT mask at the same resolution —
                the amodal silhouette under ``use_mask_amodal``, so the panels
                show the supervision actually used.

        Returns:
            ``{"geo_masks": HxWx3 float32 numpy image in [0, 1]}`` — empty if
            the batch carries no RGB to draw on.
        """
        from o3b.cv.visual.draw import draw_text_in_rgb
        from o3b.cv.visual.resize import resize
        from o3b.cv.visual.show import imgs_to_img

        rgb = getattr(batch, "rgb", None)
        if rgb is None:
            return {}

        pred_masks = pred_masks.detach()
        gt_masks = gt_masks.detach()

        n = min(self.qualit_max_samples, pred_masks.shape[0])
        H, W = rgb.shape[-2:]
        # only ever downscale — panels of a small image stay at native size
        scale = min(1.0, self.qualit_res / max(H, W))
        H_q, W_q = max(1, int(round(H * scale))), max(1, int(round(W * scale)))

        alpha = 0.5
        red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
        green = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1)

        rows = []
        for b in range(n):
            rgb_b = resize(rgb[b, :3].detach().float().cpu().clamp(0, 1),
                           H_out=H_q, W_out=W_q)
            gt_b = resize(gt_masks[b].float().cpu(), H_out=H_q, W_out=W_q).clamp(0, 1)
            pred_b = resize(pred_masks[b].float().cpu(), H_out=H_q, W_out=W_q).clamp(0, 1)

            # IoU of the hard masks at render (not panel) resolution
            p_bin = pred_masks[b] > 0.5
            g_bin = gt_masks[b] > 0.5
            union = float((p_bin | g_bin).sum())
            iou = float((p_bin & g_bin).sum()) / union if union > 0 else float("nan")

            rgb_labelled = draw_text_in_rgb(rgb_b.clone(), text=f"iou {iou:.2f}",
                                            fontColor=(255, 255, 255))
            rows.append([
                rgb_labelled,
                (rgb_b * (1 - alpha * gt_b) + red * (alpha * gt_b)).clamp(0, 1),
                (rgb_b * (1 - alpha * pred_b) + green * (alpha * pred_b)).clamp(0, 1),
                torch.cat([gt_b, pred_b, torch.zeros_like(gt_b)], dim=0),
            ])

        img = imgs_to_img(rows, pad=2, pad_value=1.0)  # (3, n*H_q, 4*W_q)
        return {"geo_masks": img.permute(1, 2, 0).numpy()}

    @staticmethod
    def _binarise(mask, H_out, W_out):
        """(B, H, W) or (B, 1, H, W) mask → hard (B, 1, H_out, W_out) in {0, 1}."""
        from o3b.cv.visual.resize import resize

        mask = mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        return resize((mask > 0.5) * 1.0, H_out=H_out, W_out=W_out)

    def _render_gt_mesh_masks(self, batch, cams_tform4x4_obj, H, W,
                              meshes, H_out, W_out):
        """Silhouette of the batch's GT meshes under the GT pose, or None.

        The live fallback for ``use_mask_amodal`` when the dataset carries no
        ``fo_mask_amodal`` modality (which is the same silhouette, baked in
        once per item instead of rasterised per batch).

        NeMo's ``replace_mask_with_rendered_mask`` renders ``batch.mesh`` with
        the batch's camera and uses that in place of the annotated mask — the
        supervision then matches the GT geometry exactly, without annotation
        noise or occlusions.

        ``batch.meshes`` is one GT mesh per sample (``batch.mesh``, a single
        shared mesh, is accepted as a fallback); their verts live in the same
        NCDS frame as the predicted meshes — see
        ``ObjGeo3D._pcl_chamfer_distance`` — so the very same
        ``cams_tform4x4_obj`` renders both. They are rasterised straight to the
        predicted masks' resolution (``down_sample_rate``) with the same
        rasterizer backend as ``meshes``, under ``no_grad``: this is a target,
        not a prediction.

        Returns:
            (B, 1, H_out, W_out) hard mask in {0, 1}, or None when the batch
            carries no usable GT mesh (a warning is logged once and the caller
            falls back to ``batch.fo_mask``).
        """
        from o3b.cv.geometry.objects3d.objects3d import PROJECT_MODALITIES

        B = cams_tform4x4_obj.shape[0]
        device = cams_tform4x4_obj.device

        gt_meshes = getattr(batch, "meshes", None)
        # a single shared mesh is rasterised once and referenced by every sample
        shared = gt_meshes is None
        if shared:
            gt_mesh = getattr(batch, "mesh", None)
            gt_meshes = None if gt_mesh is None else [gt_mesh]
        usable = gt_meshes is not None and (shared or len(gt_meshes) == B) and all(
            m is not None and getattr(m, "verts", None) is not None
            and getattr(m, "faces", None) is not None and len(m.faces) > 0
            for m in gt_meshes
        )
        if not usable:
            if not self._warned_no_gt_meshes:
                self._warned_no_gt_meshes = True
                logger.warning(
                    "CamGeo2D3D: use_mask_amodal is set but the batch carries "
                    "neither a fo_mask_amodal modality nor usable GT meshes "
                    "(batch.meshes); falling back to batch.fo_mask.",
                )
            return None

        try:
            from o3b.cv.geometry.objects3d.meshes.meshes import Meshes

            kwargs = {}
            rasterizer = getattr(meshes, "rasterizer", None)
            if rasterizer is not None:
                kwargs["rasterizer"] = rasterizer
            with torch.no_grad():
                gt_objects3d = Meshes(
                    verts=[m.verts.to(device=device, dtype=torch.float32)
                           for m in gt_meshes],
                    faces=[m.faces.to(device=device) for m in gt_meshes],
                    # pure rasterisation target: no vertex features, no clutter
                    # feature, no verts parameter to optimise (feats stay
                    # "param" so Meshes.to() skips the absent feature tensor)
                    feats_objects=False,
                    feat_clutter=False,
                    verts_requires_param=False,
                    device=device,
                    **kwargs,
                )
                gt_masks = gt_objects3d.render(
                    cams_tform4x4_obj=cams_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=torch.LongTensor([H, W]).to(device),
                    # one mesh per sample, in batch order (or the shared one)
                    objects_ids=(torch.zeros(B, dtype=torch.long, device=device)
                                 if shared else torch.arange(B, device=device)),
                    modalities=PROJECT_MODALITIES.MASK,
                    down_sample_rate=self.down_sample_rate,
                )  # (B, 1, H_out, W_out)
        except Exception as e:
            if not self._warned_no_gt_meshes:
                self._warned_no_gt_meshes = True
                logger.warning(
                    f"CamGeo2D3D: rendering the GT meshes for use_mask_amodal "
                    f"failed ({e}); falling back to batch.fo_mask.",
                    exc_info=True,
                )
            return None

        gt_masks = (gt_masks.detach().float() > 0.5) * 1.0
        if gt_masks.shape[-2:] != (H_out, W_out):
            from o3b.cv.visual.resize import resize
            gt_masks = resize(gt_masks, H_out=H_out, W_out=W_out)
        return gt_masks

    def _get_mask_dt(self, batch, inv: bool, device=None, mask=None, amodal=False):
        """GT mask distance transform as (B, 1, H, W) on *device*, or None.

        Mirrors ``od3d.od3d_datasets.frame.Frame.get_mask_dt`` /
        ``get_mask_inv_dt``: use the modality if the batch carries one,
        otherwise derive it from the binarised GT mask with
        ``get_mask_distance_transform_norm`` (normalised to [0, 1] by the
        larger image side; 0 outside the mask).

        Which silhouette the DT describes follows the one the other sub-losses
        are supervised by, so the whole task stays consistent:

        - default: ``fo_mask`` — DT from the dataset's ``fo_mask_dt`` (baked
          into the shards at build time), or the legacy ``mask_dt`` /
          ``mask_inv_dt`` batch fields, or derived from ``fo_mask`` here.
        - ``amodal``: ``fo_mask_amodal`` — DT from the dataset's
          ``fo_mask_amodal_dt``, or derived from ``fo_mask_amodal`` here.
        - ``mask`` given: that tensor (the GT meshes rasterised live, when the
          dataset carries no amodal modality). Derived at the render
          resolution — the normalisation by the larger side makes that
          scale-invariant.

        Each source caches into its own batch field, so the three never mix.
        The result is cached back so that both sub-losses (and repeated calls
        within a step) pay the distance transform only once. The mask is handed
        to ``get_mask_distance_transform_norm`` on the device it already lives
        on, so a live (GPU) mask takes that function's exact ``torch`` backend
        rather than a round trip through cv2 on the CPU.

        A dataset-provided DT arrives on whatever device the dataloader left it
        on (CPU), so it is moved to *device* — the render device — before it
        meets the predicted masks.
        """
        if mask is not None:
            field = "rendered_mask_inv_dt" if inv else "rendered_mask_dt"
            provided, src = None, mask
        elif amodal:
            field = "mask_amodal_inv_dt" if inv else "mask_amodal_dt"
            provided = None if inv else getattr(batch, "fo_mask_amodal_dt", None)
            src = getattr(batch, "fo_mask_amodal", None)
        else:
            field = "mask_inv_dt" if inv else "mask_dt"
            provided = None if inv else getattr(batch, "fo_mask_dt", None)
            src = getattr(batch, "fo_mask", None)

        dt = getattr(batch, field, None)
        if dt is None:
            dt = provided
        if dt is None:
            fo_mask = src
            if fo_mask is None:
                return None
            from o3b.cv.visual.mask import get_mask_distance_transform_norm

            mask_bin = fo_mask > 0.5
            if mask_bin.dim() == 3:
                mask_bin = mask_bin.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
            if inv:
                mask_bin = ~mask_bin
            # left on the render device: the "auto" backend then keeps a CUDA
            # mask on the GPU (exact, ~10x faster than the cv2 round trip)
            dt = get_mask_distance_transform_norm(mask_bin).to(
                device=fo_mask.device, dtype=torch.float32,
            )
            try:
                setattr(batch, field, dt)
            except AttributeError:
                pass  # frozen batch type — recompute next time
        else:
            dt = dt.float()
            if dt.dim() == 3:
                dt = dt.unsqueeze(1)
        if device is not None:
            dt = dt.to(device)
        return dt

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
