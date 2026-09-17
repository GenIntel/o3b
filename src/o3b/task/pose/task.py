"""Pose task: rotation / translation / size accuracy for frame-objects.

Ported from od3d's ``od3d/tasks/pose/task.py``. It is what produces the
30-degree and 10-degree columns of the Every9D table, and the 3-D IoU column
for the datasets whose poses are metric.

Two things it does differently from a naive pose metric, both because the
benchmark spans six datasets that disagree with one another:

**Symmetry.** Every accuracy is reported twice — plain, and ``_sym`` with the
object's annotated ``obj_syms`` quotiented out via ``get_pose_diff``. A bottle
is a solid of revolution and no image determines its azimuth; charging a model
for that error measures the annotation, not the model. A mug's handle does
determine it. Reporting only one number would be choosing wrongly for half the
categories, which is why the published table carries both per cell.

**Scale.** ``pose_transl_*`` and ``pose_bbox3d_*`` are metric and therefore
meaningless for PASCAL3D and ImageNet3D, whose poses are in normalised CAD
units. They are still computed — suppressing them per dataset would put that
knowledge in the task rather than in the reader — but the published table
reports only the rotation columns for those two, and so should any comparison.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch

from o3b.task.task import OD3D_Task, register_task
from o3b.task.datatypes.frame_object_quant import FrameObjectQuantBatch

logger = logging.getLogger(__name__)


def _unit_rot3x3(tform4x4: torch.Tensor) -> torch.Tensor:
    """Rotation block with any per-axis scale divided out.

    The NCDS pose carries the object size in its 3x3 block, so reading that
    block as a rotation without normalising would register a large object as a
    large rotation error.
    """
    R = tform4x4[..., :3, :3].clone().float()
    n = R.norm(dim=-2, keepdim=True).clamp(min=1e-8)
    return R / n


def _aabb_iou(size_a: torch.Tensor, size_b: torch.Tensor) -> torch.Tensor:
    """3-D IoU of two origin-centred, axis-aligned boxes given their side lengths.

    Axis-aligned and concentric on purpose: this is the *size* agreement, the
    same quantity od3d's bbox3d_iou reduces to once both boxes are placed at the
    predicted and GT poses and those poses agree. It cannot punish a rotation
    error — that is what the rotation columns are for — so a run should never be
    read as good because its IoU is high while its 30-degree number is low.
    """
    a = size_a.float().clamp(min=0)
    b = size_b.float().clamp(min=0)
    inter = torch.minimum(a, b).prod(dim=-1)
    union = a.prod(dim=-1) + b.prod(dim=-1) - inter
    return inter / union.clamp(min=1e-12)


@register_task("PoseTask")
class PoseTask(OD3D_Task):
    """Compare predicted against GT object pose for each frame-object."""

    def __init__(self, rot_acc_deg=(10.0, 30.0), transl_acc_cm=(5.0, 10.0),
                 iou_acc=(0.25, 0.5, 0.75)):
        self.rot_acc_deg = tuple(rot_acc_deg)
        self.transl_acc_cm = tuple(transl_acc_cm)
        self.iou_acc = tuple(iou_acc)
        self._warned_missing: set = set()

    def forward(self, batch, return_qualit: bool = True) -> Tuple[FrameObjectQuantBatch, object]:
        from o3b.cv.metric.pose import get_pose_diff, get_pose_diff_in_rad

        quant = FrameObjectQuantBatch()

        gt = getattr(batch, "cam_tform4x4_obj", None)
        pred = getattr(batch, "pred_cam_tform4x4_obj", None)
        if pred is None:
            # No method configured: score the GT against itself (the oracle
            # upper bound), exactly as o3b's other tasks do.
            pred = gt
        if gt is None or pred is None:
            key = ("cam_tform4x4_obj",)
            if key not in self._warned_missing:
                self._warned_missing.add(key)
                print("WARNING: PoseTask skipping batch — cam_tform4x4_obj is None.")
            return quant, None

        gt = gt.float()
        pred = pred.float()
        syms = getattr(batch, "obj_syms", None)

        # ── rotation, plain and symmetry-aware ───────────────────────────────
        rot_err = get_pose_diff_in_rad(pred_tform4x4=pred, gt_tform4x4=gt)
        quant.pose_rot_err_rad = rot_err
        if syms is not None:
            _, rot_err_sym = get_pose_diff(
                pred_tform4x4=pred.clone(), gt_tform4x4=gt.clone(),
                obj_rot3d_obj_syms=syms, lp_norm=2,
            )
            quant.pose_rot_err_rad_sym = rot_err_sym
        else:
            rot_err_sym = None

        for deg in self.rot_acc_deg:
            thr = math.radians(deg)
            setattr(quant, f"pose_rot_acc_{int(deg)}deg", (rot_err < thr).float())
            if rot_err_sym is not None:
                setattr(quant, f"pose_rot_acc_{int(deg)}deg_sym",
                        (rot_err_sym < thr).float())

        # ── translation ──────────────────────────────────────────────────────
        transl_err = (pred[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1)
        quant.pose_transl_err_m = transl_err
        for cm in self.transl_acc_cm:
            setattr(quant, f"pose_transl_acc_{int(cm)}cm", (transl_err < cm / 100.0).float())

        r10 = quant.pose_rot_acc_10deg
        r30 = quant.pose_rot_acc_30deg
        t5 = quant.pose_transl_acc_5cm
        t10 = quant.pose_transl_acc_10cm
        if r10 is not None and t5 is not None:
            quant.pose_rot_transl_acc_10deg_5cm = r10 * t5
        if r30 is not None and t10 is not None:
            quant.pose_rot_transl_acc_30deg_10cm = r30 * t10

        # ── size and 3-D IoU ─────────────────────────────────────────────────
        gt_size = getattr(batch, "obj_size3d", None)
        pred_size = getattr(batch, "pred_obj_size3d", None)
        if pred_size is None:
            pred_size = gt_size
        if gt_size is not None and pred_size is not None:
            gt_size = gt_size.float()
            pred_size = pred_size.float()
            quant.pose_size3d_err_m = (pred_size - gt_size).abs().mean(dim=-1)
            iou = _aabb_iou(pred_size, gt_size)
            quant.pose_bbox3d_iou = iou
            for thr in self.iou_acc:
                setattr(quant, f"pose_bbox3d_acc_{int(thr*100)}", (iou > thr).float())

        return quant, None
