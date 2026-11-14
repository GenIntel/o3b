"""H19: the exact symmetry-aware rotation error (``exact=True``) against a brute-force
minimum over the symmetry group with continuous axes sampled densely."""
import math

import pytest
import torch

from o3b.cv.metric.pose import (get_obj_rot3x3_obj_sym, get_pose_diff,
                                get_pose_diff3x3_in_rad, get_pose_diff3x3_with_syms)
from o3b.cv.metric.pt3d_so3 import so3_exp_map

# A continuous axis only ever comes with 180 deg flips (cutoop's `cylinder`), which keep
# it up to sign; (4, -1, 1) would not be a group (90 deg about x turns y into z).
CODES = [(1, 1, 1), (1, 1, 2), (2, 2, 2), (1, 1, 4), (1, -1, 1), (2, -1, 1), (1, 1, -1),
         (1, -1, 2), (1, 2, -1), (-1, -1, 1), (-1, -1, -1)]
STEPS = 3600  # 0.1 deg: brute-force error <= 0.05 deg above the exact one


def _rand_rot(n, gen, max_angle=math.pi):
    axis = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    ang = torch.rand(n, 1, generator=gen, dtype=torch.float64) * max_angle
    return so3_exp_map(axis * ang)


def _group_dense(code):
    """(S, 3, 3): the discrete members times a continuous axis sampled at STEPS
    (one continuous axis at most; o3b's builder deduplicates pairwise, S^2)."""
    assert sum(c < 0 for c in code) <= 1
    disc = get_obj_rot3x3_obj_sym(torch.tensor([max(c, 1) for c in code], dtype=torch.float64)).double()
    if all(c > 0 for c in code):
        return disc
    a = torch.zeros(3, dtype=torch.float64)
    a[[i for i, c in enumerate(code) if c < 0][0]] = 1.0
    phi = torch.arange(STEPS, dtype=torch.float64) * (2 * math.pi / STEPS)
    turns = so3_exp_map(phi[:, None] * a)                                  # (STEPS, 3, 3)
    return (disc[:, None] @ turns[None]).reshape(-1, 3, 3)


def _angle(R):
    cos = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.arccos(cos.clamp(-1, 1))


@pytest.mark.parametrize("code", CODES)
def test_exact_equals_bruteforce(code):
    gen = torch.Generator().manual_seed(sum((i + 3) * c for i, c in enumerate(code)) + 100)
    n = 64
    D = _rand_rot(n, gen)
    syms = torch.tensor(code, dtype=torch.float64).expand(n, 3)
    err = get_pose_diff3x3_in_rad(D, obj_rot3d_obj_syms=syms, exact=True)
    if sum(c < 0 for c in code) >= 2:
        assert torch.all(err.abs() < 1e-6)
        return
    G = _group_dense(code)
    brute = _angle(G.transpose(-1, -2)[None] @ D[:, None]).min(dim=-1).values
    assert torch.all(err <= brute + 1e-6)
    tol = 0.0 if all(c > 0 for c in code) else math.radians(0.05) + 1e-6
    assert torch.all(brute - err <= tol + 1e-6), (brute - err).max()


@pytest.mark.parametrize("code", CODES)
def test_exact_never_above_first_order(code):
    gen = torch.Generator().manual_seed(7)
    D = _rand_rot(64, gen)
    syms = torch.tensor(code, dtype=torch.float64).expand(64, 3)
    exact = get_pose_diff3x3_in_rad(D, obj_rot3d_obj_syms=syms, exact=True)
    # the first-order code runs in float32 only (its symmetry table is float)
    first = get_pose_diff3x3_in_rad(D.float(), obj_rot3d_obj_syms=syms.float()).double()
    assert torch.all(exact <= first + 1e-4)
    if all(c > 0 for c in code):
        assert torch.allclose(exact, first, atol=1e-4)


def test_hypothesis_example():
    """theta = 180 deg about an axis 45 deg off the symmetry axis z: 127 deg first order, 90 exact."""
    n = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64) / math.sqrt(2)
    D = so3_exp_map(n * math.pi * (1 - 1e-9))
    syms = torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64)
    assert math.degrees(get_pose_diff3x3_in_rad(D, syms, exact=True).item()) == pytest.approx(90.0, abs=1e-3)
    first = get_pose_diff3x3_in_rad(D.float(), syms.float()).item()
    assert math.degrees(first) == pytest.approx(127.28, abs=0.05)


@pytest.mark.parametrize("code", CODES)
def test_turned_gt_is_symmetry_equivalent(code):
    """get_pose_diff(exact=True) hands back R_gt G with G in the symmetry group and
    angle(R_gt G, R_pred) = the exact error."""
    gen = torch.Generator().manual_seed(11)
    n = 32
    R_gt, R_pred = _rand_rot(n, gen), _rand_rot(n, gen)
    gt = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    pred = gt.clone()
    gt[:, :3, :3], pred[:, :3, :3] = R_gt, R_pred
    syms = torch.tensor(code, dtype=torch.float64).expand(n, 3)
    gt_sym, err = get_pose_diff(pred, gt, obj_rot3d_obj_syms=syms, exact=True)
    G = R_gt.transpose(-1, -2) @ gt_sym[:, :3, :3]
    assert torch.allclose(_angle(gt_sym[:, :3, :3].transpose(-1, -2) @ R_pred), err, atol=1e-4)  # arccos near 0
    if sum(c < 0 for c in code) >= 2:
        return  # any rotation is a symmetry
    # G is a group member: distance to the densely sampled group <= half a step
    Gd = _group_dense(code)
    dist = _angle(Gd.transpose(-1, -2)[None] @ G[:, None]).min(dim=-1).values
    assert torch.all(dist <= math.radians(360 / STEPS / 2) + 1e-5)


def test_gradient_finite_and_matches_fixed_turn():
    gen = torch.Generator().manual_seed(3)
    log = (torch.randn(16, 3, generator=gen, dtype=torch.float64) * 0.8).requires_grad_()
    syms = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64).expand(16, 3)
    err = get_pose_diff3x3_in_rad(so3_exp_map(log), syms, exact=True)
    err.sum().backward()
    assert torch.isfinite(log.grad).all()
    # envelope theorem: the gradient equals the finite difference of the exact minimum
    eps = 1e-6
    for i in range(3):
        d = torch.zeros(3, dtype=torch.float64)
        d[i] = eps
        with torch.no_grad():
            fd = (get_pose_diff3x3_in_rad(so3_exp_map(log + d), syms, exact=True)
                  - get_pose_diff3x3_in_rad(so3_exp_map(log - d), syms, exact=True)) / (2 * eps)
        assert torch.allclose(log.grad[:, i], fd, atol=1e-4)


def test_with_syms_shape_batched():
    gen = torch.Generator().manual_seed(5)
    D = _rand_rot(6, gen).reshape(2, 3, 3, 3)
    syms = torch.tensor([[2.0, -1.0, 1.0]] * 3 + [[1.0, 1.0, 2.0]] * 3,
                        dtype=torch.float64).reshape(2, 3, 3)
    E, err = get_pose_diff3x3_with_syms(D, syms, exact=True)
    assert E.shape == (2, 3, 3, 3) and err.shape == (2, 3)
    assert torch.allclose(_angle(E), err, atol=1e-6)


@pytest.mark.parametrize("code,label", [((1, -1, 1), ("none", "any", "none")),
                                        ((2, -1, 1), ("half", "any", "none")),
                                        ((1, 1, -1), ("none", "none", "any"))])
def test_posetask_symexact_equals_cutoop_rot_diff_sym(code, label):
    """PoseTask's _symexact error = cutoop's analytic rot_diff_sym (Omni6DPose)."""
    pytest.importorskip("cutoop")
    from cutoop.rotation import SymLabel, rot_diff_sym
    from types import SimpleNamespace
    from o3b.task.pose.task import PoseTask

    gen = torch.Generator().manual_seed(13)
    n = 64
    gt = torch.eye(4).repeat(n, 1, 1)
    pred = gt.clone()
    gt[:, :3, :3] = _rand_rot(n, gen).float()
    pred[:, :3, :3] = _rand_rot(n, gen).float()
    size = torch.full((n, 3), 0.1)
    b = SimpleNamespace(cam_tform4x4_obj_ncds=gt, pred_cam_tform4x4_obj=pred, obj_size3d=size,
                        pred_obj_size3d=size, obj_syms=torch.tensor(code).float().repeat(n, 1))
    q = PoseTask()(b, return_qualit=False)[0]
    ours = torch.rad2deg(q.pose_rot_err_rad_symexact).double()
    sym = SymLabel(False, *label)
    ref = torch.tensor([math.degrees(rot_diff_sym(gt[i, :3, :3].double().numpy(),
                                                  pred[i, :3, :3].double().numpy(), sym))
                        for i in range(n)], dtype=torch.float64)
    assert torch.allclose(ours, ref, atol=0.02), (ours - ref).abs().max()
    first = torch.rad2deg(q.pose_rot_err_rad_sym).double()
    assert torch.all(ours <= first + 0.02) and (first - ours).max() > 1.0   # not vacuous
