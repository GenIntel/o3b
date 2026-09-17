from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class FrameObjectQuantBatch:
    """Quantitative pose metrics for a batch of B frame-objects.

    Every accuracy comes in two variants, and both are reported rather than one
    being chosen: ``*_sym`` quotients out the object's annotated symmetry
    (``obj_syms``) before measuring the rotation error, the plain one does not.
    A mug's handle makes its pose observable; a bottle's does not, and scoring a
    bottle as if it did charges the model for an error no image could resolve.
    The published table gives both numbers per cell for exactly this reason.
    """

    # ── rotation ─────────────────────────────────────────────────────────────
    pose_rot_err_rad:      Optional[Tensor] = None  # (B,) geodesic error
    pose_rot_err_rad_sym:  Optional[Tensor] = None  # (B,) … minimised over symmetries
    pose_rot_acc_10deg:     Optional[Tensor] = None  # (B,) float 0/1
    pose_rot_acc_30deg:     Optional[Tensor] = None
    pose_rot_acc_10deg_sym: Optional[Tensor] = None
    pose_rot_acc_30deg_sym: Optional[Tensor] = None

    # ── translation (metric; meaningless where the dataset is not metric) ────
    pose_transl_err_m:      Optional[Tensor] = None  # (B,) metres
    pose_transl_acc_5cm:    Optional[Tensor] = None
    pose_transl_acc_10cm:   Optional[Tensor] = None

    # ── combined, as od3d reports them ───────────────────────────────────────
    pose_rot_transl_acc_10deg_5cm:  Optional[Tensor] = None
    pose_rot_transl_acc_30deg_10cm: Optional[Tensor] = None

    # ── size / 3-D IoU ───────────────────────────────────────────────────────
    pose_size3d_err_m:  Optional[Tensor] = None  # (B,) mean abs side-length error
    pose_bbox3d_iou:    Optional[Tensor] = None  # (B,) axis-aligned 3-D IoU
    pose_bbox3d_acc_25: Optional[Tensor] = None
    pose_bbox3d_acc_50: Optional[Tensor] = None
    pose_bbox3d_acc_75: Optional[Tensor] = None

    extra: dict = field(default_factory=dict)

    _FIELDS = (
        "pose_rot_err_rad", "pose_rot_err_rad_sym",
        "pose_rot_acc_10deg", "pose_rot_acc_30deg",
        "pose_rot_acc_10deg_sym", "pose_rot_acc_30deg_sym",
        "pose_transl_err_m", "pose_transl_acc_5cm", "pose_transl_acc_10cm",
        "pose_rot_transl_acc_10deg_5cm", "pose_rot_transl_acc_30deg_10cm",
        "pose_size3d_err_m", "pose_bbox3d_iou",
        "pose_bbox3d_acc_25", "pose_bbox3d_acc_50", "pose_bbox3d_acc_75",
    )

    def mean(self) -> dict:
        """Flat dict of scalar means over the batch (nan-safe)."""
        import torch as _torch

        out = {}
        for fname in self._FIELDS:
            val = getattr(self, fname)
            if val is None:
                continue
            finite = val.float()[_torch.isfinite(val.float())]
            if len(finite):
                out[fname] = finite.mean().item()
        for k, v in self.extra.items():
            if v is not None and hasattr(v, "mean"):
                finite = v.float()[_torch.isfinite(v.float())]
                if len(finite):
                    out[k] = finite.mean().item()
        return out

    def to_wandb_log(self, prefix: str = "batch", wb=None) -> dict:
        out: dict = {f"{prefix}/{k}": v for k, v in self.mean().items()}
        if wb is None or self.pose_rot_err_rad is None:
            return out
        import math

        data = self.pose_rot_err_rad.detach().cpu().float().numpy() * 180.0 / math.pi
        if data.size:
            out[f"{prefix}/pose_rot_err_deg"] = wb.Histogram(data)
        return out
