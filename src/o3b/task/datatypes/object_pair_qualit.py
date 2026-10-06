from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class ObjectPairQualitBatch:
    """Qualitative outputs for a batch of B object pairs."""

    # predicted correspondences: src vertex index for each trgt vertex
    trgt_src_vert_corr:       Optional[Tensor] = None  # (B, V_trgt)   int64
    trgt_src_vert_corr_mask:  Optional[Tensor] = None  # (B, V_trgt)   bool

    # predicted keypoints on the target in the src canonical frame
    trgt_obj_kpts3d_in_src:   Optional[Tensor] = None  # (B, K, 3)

    # keypoint correspondence images
    imgs:                     Optional[Tensor] = None  # (B, 3, H, W)  float32 [0,1]

    # part correspondence images:
    #   left  = source mesh colored by part label
    #   right = target mesh colored by nearest-source-vertex part label (feature NN)
    part_imgs:                Optional[Tensor] = None  # (B, 3, H, W)  float32 [0,1]

    # keypoint correspondence images, but meshes are colored by per-vertex
    # features (normalized directly if <=3 dims, else first 3 PCA components)
    # instead of RGB — same layout/coloring rules as `imgs`
    feat_imgs:                Optional[Tensor] = None  # (B, 3, H, W)  float32 [0,1]

    # extra per-task qualitative outputs
    extra: dict = field(default_factory=dict)

    def to_wandb_log(
        self,
        prefix: str = "qualit",
        wb=None,
        log_imgs: bool = True,
    ) -> dict:
        out: dict = {}
        if wb is None or not log_imgs:
            return out

        from o3b.cv.io import to_wandb_uint8

        def _log_tensor_imgs(tensor, key):
            if tensor is None:
                return
            out[key] = [
                wb.Image(
                    to_wandb_uint8(img),
                    caption=f"i{i}",
                )
                for i, img in enumerate(tensor)
            ]

        _log_tensor_imgs(self.imgs,      f"{prefix}/correspondences")
        _log_tensor_imgs(self.part_imgs, f"{prefix}/part_correspondences")
        _log_tensor_imgs(self.feat_imgs, f"{prefix}/feat_correspondences")
        return out
