"""PoseTask(cat_axes_align=True): the per-category axis relabelling."""
import math
from types import SimpleNamespace

import torch

from o3b.cv.metric.pose import get_pose_diff, get_pose_diff_in_rad
from o3b.task.pose.cat_align import category_axes_align, proper_signed_permutations
from o3b.task.pose.task import PoseTask


def _random_rot(n, gen):
    q = torch.randn(n, 4, generator=gen)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(n, 3, 3)


def _small_noise(n, deg, gen):
    axis = torch.randn(n, 3, generator=gen)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    ang = torch.rand(n, generator=gen) * math.radians(deg)
    K = torch.zeros(n, 3, 3)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    s, c = ang.sin()[:, None, None], ang.cos()[:, None, None]
    return torch.eye(3) + s * K + (1 - c) * K @ K


def _tform(R, scale=1.0):
    T = torch.eye(4).repeat(R.shape[0], 1, 1)
    T[:, :3, :3] = R * scale
    return T


def test_24_proper_permutations():
    Ps = proper_signed_permutations()
    assert Ps.shape == (24, 3, 3)
    assert torch.equal(Ps[0], torch.eye(3))
    assert torch.allclose(torch.det(Ps), torch.ones(24))
    assert len({tuple(P.flatten().tolist()) for P in Ps}) == 24


def test_identity_choice_matches_plain_and_sym_errors():
    gen = torch.Generator().manual_seed(0)
    n = 200
    gt = _random_rot(n, gen)
    pred = gt @ _small_noise(n, 60, gen)
    syms = torch.tensor([[1, -1, 1], [1, 2, 1], [1, 1, 1], [2, 2, 2]]).float()[
        torch.randint(0, 4, (n,), generator=gen)]
    # one category: alignment can only help, and the identity column must
    # equal the task's own errors
    from o3b.task.pose.cat_align import _errors
    D = gt.transpose(-1, -2) @ pred
    e_plain = _errors(D, None)
    e_sym = _errors(D, syms)
    ref_plain = get_pose_diff_in_rad(_tform(pred, 0.3), _tform(gt, 0.7))
    _, ref_sym = get_pose_diff(_tform(pred), _tform(gt), obj_rot3d_obj_syms=syms, lp_norm=2)
    assert torch.allclose(e_plain, ref_plain, atol=1e-4)
    assert torch.allclose(e_sym, ref_sym, atol=1e-4)


def test_consistent_relabelling_is_undone_per_category():
    gen = torch.Generator().manual_seed(1)
    Ps = proper_signed_permutations()
    n = 300
    gt = _random_rot(n, gen)
    cats = [["mug", "bottle", "chair"][i % 3] for i in range(n)]
    true_P = {"mug": Ps[0], "bottle": Ps[5], "chair": Ps[17]}
    pred = torch.stack([gt[i] @ _small_noise(1, 5, gen)[0] @ true_P[c].T
                        for i, c in enumerate(cats)])
    metrics, choice = category_axes_align(cats, pred, gt, {"": None}, device="cpu")
    assert metrics["pose_rot_acc_10deg_catalign"] == 1.0
    for c, P in true_P.items():
        assert choice["plain"][c][1] == P.int().tolist()
    assert metrics["pose_catalign_n_changed"] == 2.0


def test_posetask_flag_off_by_default_and_on():
    gen = torch.Generator().manual_seed(2)
    n = 32
    gt = _random_rot(n, gen)
    P = proper_signed_permutations()[9]
    pred = gt @ P.T
    batch = SimpleNamespace(cam_tform4x4_obj_ncds=_tform(gt), pred_cam_tform4x4_obj=_tform(pred),
                            obj_syms=torch.ones(n, 3), category=["mug"] * n)
    off = PoseTask()
    off(batch)
    assert off.finalize() == ({}, {})
    on = PoseTask(cat_axes_align=True)
    quant, _ = on(batch)
    assert quant.pose_rot_acc_30deg_sym.mean() < 1.0
    metrics, _ = on.finalize()
    assert metrics["pose_rot_acc_30deg_sym_catalign"] == 1.0
    assert metrics["pose_rot_acc_10deg_catalign"] == 1.0
    assert on._align_buf == []


def test_merge_of_shard_dumps_matches_one_pass(tmp_path):
    from o3b.task.pose import merge as M
    gen = torch.Generator().manual_seed(3)
    Ps = proper_signed_permutations()
    n = 96
    gt = _random_rot(n, gen)
    cats = [["mug", "bottle"][i % 2] for i in range(n)]
    pred = torch.stack([gt[i] @ (Ps[4].T if c == "mug" else torch.eye(3)) for i, c in enumerate(cats)])
    syms = torch.ones(n, 3)
    # one pass over everything
    task = PoseTask(cat_axes_align=True)
    full = SimpleNamespace(cam_tform4x4_obj_ncds=_tform(gt), pred_cam_tform4x4_obj=_tform(pred),
                           obj_syms=syms, category=cats)
    q, _ = task(full)
    ref, _ = task.finalize()
    # three shard dumps, the mugs and bottles split across them; shard 1 twice
    for s, (a, b) in enumerate([(0, 40), (40, 70), (70, 96)]):
        qs, _ = PoseTask()(SimpleNamespace(cam_tform4x4_obj_ncds=_tform(gt[a:b]),
                                           pred_cam_tform4x4_obj=_tform(pred[a:b]),
                                           obj_syms=syms[a:b], category=cats[a:b]))
        rec = {"category": cats[a:b], "frame_object_id": list(range(a, b)),
               "pred_cam_tform4x4_obj": _tform(pred[a:b]), "cam_tform4x4_obj_ncds": _tform(gt[a:b]),
               "obj_syms": syms[a:b]}
        rec.update({k: getattr(qs, k) for k in qs._FIELDS if getattr(qs, k) is not None})
        for rep in range(2 if s == 1 else 1):
            torch.save([rec], tmp_path / f"0101_00000{rep}__pose_x__m__b40_s{s:03d}.rank0.pt")
    recs = M.collect(sorted(map(str, tmp_path.glob("*.pt"))))
    assert sum(len(r["category"]) for r in recs) == n
    metrics, _ = M.merge(recs, cat_axes_align=True)
    assert metrics["pose_rot_acc_30deg_sym"] == q.pose_rot_acc_30deg_sym.mean().item() == 0.5
    for k in ("pose_rot_acc_30deg_sym_catalign", "pose_rot_acc_10deg_catalign"):
        assert metrics[k] == ref[k] == 1.0


def test_supercategory_align_one_P_per_supercategory():
    from o3b.task.pose.cat_align import supercategory_of, supercategory_axes_align
    assert supercategory_of("apple") == supercategory_of("pear") == "fruit"
    assert supercategory_of("not_a_category") == "unmapped:not_a_category"
    # ImageNet3D names that differ from their UCO3D name (enum-keyed map)
    assert supercategory_of("aeroplane") == supercategory_of("airplane")
    from o3b.dataset.imagenet3d.enum import MAP_CATEGORIES_IMAGENET3D_TO_UCO3D as M
    assert sum(supercategory_of(k.value).startswith("unmapped:") for k in M) <= 9
    gen = torch.Generator().manual_seed(4)
    Ps = proper_signed_permutations()
    n = 300
    gt = _random_rot(n, gen)
    cats = [["apple", "pear", "chair"][i % 3] for i in range(n)]
    # apple and pear share a super-category but are turned differently: one P
    # can fix only one of them; chair (furniture) is fixed on its own
    true_P = {"apple": Ps[5], "pear": Ps[11], "chair": Ps[17]}
    pred = torch.stack([gt[i] @ true_P[c].T for i, c in enumerate(cats)])
    metrics, choice = supercategory_axes_align(cats, pred, gt, {"": None}, device="cpu")
    assert abs(metrics["pose_rot_acc_10deg_supercatalign"] - 2 / 3) < 1e-6
    assert set(choice["super_plain"]) == {"fruit", "furniture"}
    assert metrics["pose_supercatalign_n_changed"] == 2.0
    task = PoseTask(supercat_axes_align=True)
    task(SimpleNamespace(cam_tform4x4_obj_ncds=_tform(gt), pred_cam_tform4x4_obj=_tform(pred),
                         obj_syms=torch.ones(n, 3), category=cats))
    m, _ = task.finalize()
    assert "pose_rot_acc_30deg_sym_supercatalign" in m and not any(k.endswith("_catalign") for k in m)


def test_orientation_consistency():
    from o3b.task.pose.cat_align import orientation_consistency
    gen = torch.Generator().manual_seed(5)
    Ps = proper_signed_permutations()
    n = 400
    gt = _random_rot(n, gen)
    cats = [["apple", "pear", "chair", "mug"][i % 4] for i in range(n)]
    # apple: one convention; pear: 70 % as is, 30 % turned by P2 (inconsistent);
    # chair: 90 % as is, 10 % turned (below the 20 % share: consistent);
    # mug: all turned by the same P (consistent, just relabelled)
    def P_of(i, c):
        k = i // 4
        if c == "pear" and k % 10 < 3: return Ps[2]
        if c == "chair" and k % 10 == 0: return Ps[2]
        if c == "mug": return Ps[7]
        return Ps[0]
    pred = torch.stack([gt[i] @ P_of(i, c).T for i, c in enumerate(cats)])
    m, d = orientation_consistency(cats, pred, gt, {"": None}, device="cpu")
    assert {g for g, r in d["cat"].items() if not r[0]} == {"pear"}
    assert m["pose_cat_consistent"] == 0.75 and m["pose_cat_consistent_n"] == 4
    # fruit pools apple + pear: P2 fixes 30 of its 200 frames = 15 % -> consistent;
    # chair (furniture) and mug (kitchen) are alone in their super-category: skipped
    assert set(d["supercat"]) == {"fruit"} and d["supercat"]["fruit"][0]
    # a symmetric object: every yaw is equivalent, so P2 = 180 deg about top
    # never counts against P1
    syms = torch.tensor([[1.0, 2.0, 1.0]]).repeat(n, 1)  # continuous about Y (top)
    m, _ = orientation_consistency(cats, pred, gt, {"_sym": syms}, device="cpu")
    assert m["pose_cat_consistent_sym"] == 1.0


def test_view_consistency():
    from o3b.task.pose.cat_align import view_consistency
    gen = torch.Generator().manual_seed(6)
    Ps = proper_signed_permutations()
    n = 160
    gt = _random_rot(n, gen)
    objs = [i // 16 for i in range(n)]   # 10 objects x 16 views
    # objects 0-4: one wrong frame per object for all its views (consistent);
    # objects 5-9: half the views also turned 180 deg about the object's top
    # (P2, in the GT object frame, where a symmetry would act)
    pred = torch.stack([gt[i] @ (Ps[2].T if o >= 5 and i % 2 else torch.eye(3)) @
                        _small_noise(1, 5, gen)[0] @ Ps[7].T for i, o in enumerate(objs)])
    m, d = view_consistency(objs, pred, gt, {"": None}, device="cpu")
    assert all(d["obj"][o][1] == 1.0 for o in range(5))
    # 16 views split 8 / 8: 2 * C(8, 2) = 56 of the 120 pairs agree
    assert all(abs(d["obj"][o][1] - 56 / 120) < 1e-6 for o in range(5, 10))
    assert m["pose_view_consistent_n"] == 10
    # random rotations: almost no pair agrees within 30 degrees
    m, _ = view_consistency(objs, _random_rot(n, gen), gt, {"": None}, device="cpu")
    assert m["pose_view_consistent"] < 0.05
    # a 2-fold symmetry about top makes the 180-degree turn invisible
    syms = torch.tensor([[1.0, 2.0, 1.0]]).repeat(n, 1)
    m, _ = view_consistency(objs, pred, gt, {"_sym": syms}, device="cpu")
    assert m["pose_view_consistent_sym"] == 1.0
