from dataclasses import replace as _dc_replace

import torch

from o3b.data.transforms.transform import O3B_Transform


class CropCamBBox2D(O3B_Transform):
    """Crop a FrameObject to its cam_bbox2d region and resize to (H, W).

    All pixel-aligned tensors (rgb, depth, fo_mask, depth_mask, mask) are
    cropped and resized.  cam_intr4x4 is updated with the 2-D crop transform.
    cam_bbox2d is mapped to coordinates in the new image.

    Args:
        H: output image height in pixels.
        W: output image width in pixels.
        scale_bbox: multiplicative padding around the bbox (default 1.0 = tight).
        bbox_overlap: extra crop margin on each side as a fraction of the bbox
            size (default 0.15). 0.15 means the crop extends by 0.15 * bbox_size
            beyond each edge, so the crop spans (1 + 2 * bbox_overlap) of the bbox.
            Combined multiplicatively with scale_bbox.
        crop_type: how to square the cam_bbox2d region before resizing it to
            (H, W) (default "crop_resize_ratio"):
              - "crop_large_side": pad the shorter bbox side up to the longer
                one, so the crop contains the full bbox (aspect preserved).
              - "crop_small_side": shrink the longer bbox side down to the
                shorter one, i.e. a center-crop that discards the excess of
                the longer side (aspect preserved).
              - "crop_resize_ratio": keep the bbox's own (possibly
                non-square) aspect ratio; it gets resized/distorted to fit
                (H, W) instead.
    """

    def __init__(self, H: int, W: int, scale_bbox: float = 1.0,
                 bbox_overlap: float = 0.15,
                 crop_type: str = "crop_resize_ratio"):
        super().__init__()
        self.H = H
        self.W = W
        self.scale_bbox = scale_bbox
        self.bbox_overlap = bbox_overlap
        self.crop_type = crop_type

    @property
    def _crop_scale(self) -> float:
        """Effective bbox scale: multiplicative padding plus per-side overlap."""
        return self.scale_bbox * (1.0 + 2.0 * self.bbox_overlap)

    def __call__(self, fo):
        """
        Args:
            fo: FrameObject with cam_bbox2d set.

        Returns:
            A new FrameObject with cropped modalities and updated intrinsics.
            Returns None if cam_bbox2d is missing or the crop region is too small.
        """
        from o3b.cv.visual.crop import crop_with_bbox
        from o3b.cv.geometry.transform import transf2d_bbox

        if fo.cam_bbox2d is None:
            return None

        bbox = fo.cam_bbox2d.float()
        # Degenerate box (x0==x1 or y0==y1): resizing it to (H, W) would stretch
        # a single row/column of pixels across the whole crop.
        if float(bbox[2] - bbox[0]) < 1.0 or float(bbox[3] - bbox[1]) < 1.0:
            return None

        updates = {}

        # ---- bilinear fields ----
        if fo.rgb is not None:
            rgb_crop, cam_crop_tform_cam = crop_with_bbox(
                img=fo.rgb.float(),
                bbox=bbox,
                H_out=self.H,
                W_out=self.W,
                mode="bilinear",
                scale_bbox=self._crop_scale,
                crop_type=self.crop_type,
            )
            updates["rgb"] = rgb_crop.to(dtype=fo.rgb.dtype)
        else:
            # still need cam_crop_tform_cam for intrinsics — use a dummy 1-pixel img
            _dummy = torch.zeros(1, 2, 2)
            _, cam_crop_tform_cam = crop_with_bbox(
                img=_dummy,
                bbox=bbox,
                H_out=self.H,
                W_out=self.W,
                mode="bilinear",
                scale_bbox=self._crop_scale,
                crop_type=self.crop_type,
            )

        # ---- nearest fields ----
        def _crop_nearest(img_2d):
            """Crop a (H, W) or (1, H, W) tensor with nearest interpolation."""
            squeeze = img_2d.dim() == 2
            t = img_2d.float()
            if squeeze:
                t = t.unsqueeze(0)
            out, _ = crop_with_bbox(
                img=t,
                bbox=bbox,
                H_out=self.H,
                W_out=self.W,
                mode="nearest_v2",
                scale_bbox=self._crop_scale,
                crop_type=self.crop_type,
            )
            if squeeze:
                out = out.squeeze(0)
            return out.to(dtype=img_2d.dtype)

        if fo.depth is not None:
            updates["depth"] = _crop_nearest(fo.depth)
        if fo.fo_mask is not None:
            updates["fo_mask"] = _crop_nearest(fo.fo_mask)
        if fo.fo_mask_amodal is not None:
            updates["fo_mask_amodal"] = _crop_nearest(fo.fo_mask_amodal)
        if fo.depth_mask is not None:
            updates["depth_mask"] = _crop_nearest(fo.depth_mask)
        if fo.mask is not None:
            updates["mask"] = _crop_nearest(fo.mask)

        # ---- mask distance transform ----
        # Recomputed rather than resampled: fo_mask_dt holds distances in the pixel
        # grid of its own image, normalised by that image's larger side.  Cropping
        # and resizing changes both, so a cropped copy of the full-frame DT would
        # carry the wrong scale.
        if fo.fo_mask_dt is not None and "fo_mask" in updates:
            from o3b.cv.visual.mask import get_mask_distance_transform_norm
            updates["fo_mask_dt"] = get_mask_distance_transform_norm(
                updates["fo_mask"].bool().cpu(),
            ).float()
        if fo.fo_mask_amodal_dt is not None and "fo_mask_amodal" in updates:
            from o3b.cv.visual.mask import get_mask_distance_transform_norm
            updates["fo_mask_amodal_dt"] = get_mask_distance_transform_norm(
                updates["fo_mask_amodal"].bool().cpu(),
            ).float()

        # ---- camera intrinsics ----
        if fo.cam_intr4x4 is not None:
            updates["cam_intr4x4"] = (cam_crop_tform_cam.float() @ fo.cam_intr4x4.float())

        # ---- bbox in new image ----
        updates["cam_bbox2d"] = transf2d_bbox(bbox=bbox.clone(), transf3x3=cam_crop_tform_cam)

        return _dc_replace(fo, **updates)
