"""PoseTask's Omni6DPose pose columns (*_symcal) against cutoop's own metrics."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from o3b.task.pose.task import PoseTask
from test_pose_cat_align import _random_rot, _small_noise


def _batch(n=400, seed=0, rot_deg=12.0, transl_m=0.04, code=(1, 1, 1)):
    gen = torch.Generator().manual_seed(seed)
    gt = torch.eye(4).repeat(n, 1, 1)
    gt[:, :3, :3] = _random_rot(n, gen)
    gt[:, :3, 3] = torch.randn(n, 3, generator=gen) * 0.3
    pred = gt.clone()
    pred[:, :3, :3] = gt[:, :3, :3] @ _small_noise(n, rot_deg, gen)
    pred[:, :3, 3] += torch.randn(n, 3, generator=gen) * transl_m / math.sqrt(3)
    size = torch.rand(n, 3, generator=gen) * 0.2 + 0.05
    return SimpleNamespace(
        cam_tform4x4_obj_ncds=gt, pred_cam_tform4x4_obj=pred, obj_size3d=size,
        pred_obj_size3d=size * (1 + 0.1 * torch.randn(n, 3, generator=gen)),
        obj_syms=torch.tensor(code).float().repeat(n, 1))


def test_matches_cutoop_without_symmetry():
    cutoop = pytest.importorskip("cutoop")
    from cutoop.eval_utils import DetectMatch
    from cutoop.rotation import SymLabel

    b = _batch()
    ours = PoseTask()(b, return_qualit=False)[0].mean()
    n = len(b.obj_syms)
    dm = DetectMatch(
        gt_affine=b.cam_tform4x4_obj_ncds.double().numpy(), gt_size=b.obj_size3d.double().numpy(),
        gt_sym_labels=np.array([SymLabel(False, "none", "none", "none")] * n),
        gt_class_labels=np.zeros(n, dtype=int),
        pred_affine=b.pred_cam_tform4x4_obj.double().numpy(), pred_size=b.pred_obj_size3d.double().numpy(),
        image_path=[""] * n, camera_intrinsics=[None] * n)
    m = dm.calibrate_rotation(silent=True).metrics(
        criterion=dm.criterion(computeIOU=False)).class_means

    assert ours["pose_rot_err_deg_symcal"] == pytest.approx(m.deg_mean, abs=1e-2)
    assert ours["pose_transl_err_cm"] == pytest.approx(m.sht_mean, abs=1e-3)
    for i, (deg, cm) in enumerate(((5, 2), (5, 5), (10, 2), (10, 5))):
        assert ours[f"pose_acc_{deg}deg_{cm}cm_symcal"] == pytest.approx(m.pose_acc[i], abs=1e-6)
        assert ours[f"pose_vus_{deg}deg_{cm}cm_symcal"] == pytest.approx(m.pose_auc[i].auc, abs=2e-3)
    assert 0.05 < ours["pose_vus_10deg_5cm_symcal"] < 0.95   # the test is not vacuous


def test_continuous_axis_is_free():
    """A turn about a continuous symmetry axis costs nothing (up to cutoop's
    2 pi / 100 sampling); the same turn without the symmetry costs its angle."""
    b = _batch(n=50, rot_deg=0.0, transl_m=0.0, code=(1, -1, 1))
    a = math.radians(72.0)   # 20 steps of 2 pi / 100: on the sampled group
    turn = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    b.pred_cam_tform4x4_obj[:, :3, :3] = b.cam_tform4x4_obj_ncds[:, :3, :3] @ turn
    q = PoseTask()(b, return_qualit=False)[0].mean()
    assert q["pose_rot_err_deg_symcal"] < 0.1
    assert q["pose_acc_5deg_2cm_symcal"] == 1.0
    assert q["pose_vus_5deg_2cm_symcal"] > 0.97
    b.obj_syms = torch.ones(50, 3)
    q = PoseTask()(b, return_qualit=False)[0].mean()
    assert q["pose_rot_err_deg_symcal"] == pytest.approx(72.0, abs=0.05)
    assert q["pose_acc_10deg_5cm_symcal"] == 0.0


def test_skipped_during_training():
    task = PoseTask()
    task.during_training = True
    q = task(_batch(n=8), return_qualit=False)[0].mean()
    assert not any(k.endswith("_symcal") or k == "pose_transl_err_cm" for k in q)
    assert "pose_bbox3d_auc_25_sym" in q
    task.during_training = False
    assert "pose_vus_5deg_2cm_symcal" in task(_batch(n=8), return_qualit=False)[0].mean()
