from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class FrameObjectPairQualitBatch:
    """Qualitative outputs for a batch of B frame-object pairs.

    ``imgs`` is a single composite per sample stacking two rows of
    query|target panels with the predicted correspondences drawn on top:
      * top row    – query|target frame RGB; query keypoints linked to their
        predicted target location, GT target keypoints shown as hollow rings.
      * bottom row – source|target objects rendered from a top-down camera, with
        the same source→target correspondence lines and GT/predicted keypoints.
    """

    imgs: Optional[Tensor] = None  # (B, 3, H, W) float32 [0,1]

    extra: dict = field(default_factory=dict)

    def to_wandb_log(self, prefix: str = "qualit", wb=None, log_imgs: bool = True) -> dict:
        out: dict = {}
        if wb is None or not log_imgs:
            return out
        from o3b.cv.io import to_wandb_uint8
        if self.imgs is not None:
            out[f"{prefix}/correspondences"] = [
                wb.Image(to_wandb_uint8(img), caption=f"i{i}")
                for i, img in enumerate(self.imgs)
            ]
        # extra image groups (e.g. featmap-PCA / mesh-NOCS correspondence panels):
        # each value is a (B, 3, H, W) tensor or a list of (3, H, W) tensors.
        for name, imgs in self.extra.items():
            if imgs is None:
                continue
            try:
                out[f"{prefix}/{name}"] = [
                    wb.Image(to_wandb_uint8(img), caption=f"i{i}")
                    for i, img in enumerate(imgs)
                ]
            except Exception:
                pass
        return out
