"""Category-level axis alignment of predicted poses (``PoseTask(cat_axes_align=True)``).

A model can learn a category's canonical frame with its axes relabelled — X and
Z swapped for every mug, say — and then score as wrong on every frame of that
category although its predictions are consistent with one another. This module
answers how much of a method's error is that, and nothing else: per category it
re-labels the predicted object axes **once for all of that category's
predictions**, by one of the 24 proper signed permutations ``P`` (a
proper rotation of the object frame, ``det P = +1``; a mirror would make the
predicted pose improper), and keeps the ``P`` that scores best:

    R_pred  ->  R_pred @ P_c         for every frame of category c

``P_c`` is chosen separately for each symmetry variant (plain, ``_sym``,
``_symcat``) by that variant's own 30-degree accuracy, ties broken by the
smaller mean error; the 10-degree accuracy and the mean error are then read off
with the same ``P_c``. So it is an oracle over 24 hypotheses per category,
fitted on the test set itself — an upper bound on what a fixed frame convention
fix could buy, not a fair number to put next to a method that was not given it.
With very few frames in a category the choice overfits; the per-category counts
are logged next to the choice.

Cost: 24 times the rotation-error part of the eval, done once at the end on
the gathered predictions (vectorised per symmetry code), plus the predictions
held in host memory — about 150 bytes per sample. Hence off by default.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import torch


def proper_signed_permutations() -> torch.Tensor:
    """(24, 3, 3): every signed permutation matrix with determinant +1,
    identity first."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = torch.zeros(3, 3)
            for col, row in enumerate(perm):
                P[row, col] = signs[col]
            if torch.det(P) > 0:
                mats.append(P)
    mats.sort(key=lambda P: (not torch.equal(P, torch.eye(3)),))
    return torch.stack(mats)


def _unit_rows(R: torch.Tensor) -> torch.Tensor:
    # get_pose_diff normalises rows; the NCDS pose has a uniform scale, so
    # column and row normalisation agree, but match it exactly anyway
    return R / torch.linalg.norm(R, dim=-1, keepdim=True).clamp(min=1e-12)


def _errors_plain(D: torch.Tensor) -> torch.Tensor:
    """(N,) geodesic angle of the relative rotations D = R_gt^T R_pred @ P."""
    from o3b.cv.metric.pose import get_pose_diff3x3_in_rad
    return get_pose_diff3x3_in_rad(D)


def _errors_sym(D: torch.Tensor, code: tuple) -> torch.Tensor:
    """(N,) the symmetry-aware error of get_pose_diff3x3_with_syms for one
    symmetry code shared by all N, without its per-sample python loop: the
    group is built once and broadcast."""
    from o3b.cv.metric.pose import get_obj_rot3x3_obj_sym
    from o3b.cv.metric.pt3d_so3 import so3_log_map
    syms = torch.tensor(code, dtype=D.dtype, device=D.device)
    G = get_obj_rot3x3_obj_sym(syms).to(D)                          # (S, 3, 3)
    DG = D[:, None] @ G[None]                                        # (N, S, 3, 3)
    log = so3_log_map(DG.reshape(-1, 3, 3)).reshape(*DG.shape[:-2], 3)
    log = log * (syms > 0.0)
    return log.norm(dim=-1).min(dim=-1).values


def _errors(D: torch.Tensor, syms: Optional[torch.Tensor], chunk: int = 8192) -> torch.Tensor:
    """(N,) rotation errors of D (N, 3, 3), plain if syms is None, else
    grouped by symmetry code."""
    out = torch.empty(D.shape[0], dtype=D.dtype, device=D.device)
    if syms is None:
        for s in range(0, D.shape[0], chunk):
            out[s:s + chunk] = _errors_plain(D[s:s + chunk])
        return out
    codes = torch.round(syms.float()).to(torch.int64)
    uniq, inv = torch.unique(codes, dim=0, return_inverse=True)
    for u in range(uniq.shape[0]):
        idx = (inv == u).nonzero(as_tuple=True)[0]
        code = tuple(int(c) for c in uniq[u].tolist())
        for s in range(0, idx.numel(), chunk):
            j = idx[s:s + chunk]
            out[j] = _errors_sym(D[j], code)
    return out


def category_axes_align(categories: list, pred_rot: torch.Tensor, gt_rot: torch.Tensor,
                        syms: dict, rot_acc_deg=(5.0, 10.0, 30.0), select_deg: float = 30.0,
                        device: Optional[str] = None) -> tuple[dict, dict]:
    """Metrics with one axis relabelling per category.

    Args:
        categories: N category labels (any hashable).
        pred_rot, gt_rot: (N, 3, 3) rotation blocks (scale is divided out here).
        syms: {suffix: (N, 3) obj_syms or None}; suffix "" means plain.
        select_deg: the accuracy threshold P_c is chosen by.

    Returns:
        metrics: {f"pose_rot_acc_{d}deg{suffix}_catalign", f"pose_rot_err_rad{suffix}_catalign"}
            means over all N samples.
        choice: {suffix: {category: (P index, P as nested list, n samples)}}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Ps = proper_signed_permutations().to(device=device, dtype=torch.float32)
    pred = _unit_rows(pred_rot.float().to(device))
    gt = _unit_rows(gt_rot.float().to(device))
    D0 = gt.transpose(-1, -2) @ pred                                  # (N, 3, 3)
    N = D0.shape[0]

    cat_ids: dict = {}
    cat_idx = torch.tensor([cat_ids.setdefault(c, len(cat_ids)) for c in categories],
                           device=device, dtype=torch.long)
    C = len(cat_ids)
    names = list(cat_ids)
    counts = torch.bincount(cat_idx, minlength=C)

    thr_sel = math.radians(select_deg)
    metrics: dict = {}
    choice: dict = {}
    for suffix, s in syms.items():
        if s is None and suffix != "":
            continue
        s_dev = None if s is None else s.to(device)
        err = torch.stack([_errors(D0 @ P, s_dev) for P in Ps])     # (24, N)

        # per category: hits at select_deg and summed error for each P
        hits = torch.zeros(len(Ps), C, device=device)
        esum = torch.zeros(len(Ps), C, device=device)
        hits.index_add_(1, cat_idx, (err < thr_sel).float())
        esum.index_add_(1, cat_idx, err)
        # lexicographic: most hits, then smallest mean error; identity wins
        # exact ties because it is P index 0 and argmax returns the first
        score = hits - (esum / counts.clamp(min=1)) / (2 * math.pi + 1)
        best = score.argmax(dim=0)                                   # (C,)
        err_best = err[best[cat_idx], torch.arange(N, device=device)]

        metrics[f"pose_rot_err_rad{suffix}_catalign"] = err_best.mean().item()
        for deg in rot_acc_deg:
            metrics[f"pose_rot_acc_{int(deg)}deg{suffix}_catalign"] = \
                (err_best < math.radians(deg)).float().mean().item()
        metrics[f"pose_catalign_n_changed{suffix}"] = float((best != 0).sum().item())
        choice[suffix or "plain"] = {
            names[c]: (int(best[c]), Ps[best[c]].int().tolist(), int(counts[c]))
            for c in range(C)
        }
    return metrics, choice


def supercategory_of(category: str) -> str:
    """UCO3D super-category of a category of any pose test set.

    A UCO3D (Every9D) name maps directly; an Omni6DPose, ImageNet3D or HANDAL
    name goes through that dataset's map to UCO3D first. A category no map
    covers is its own group (``"unmapped:<category>"``), so it is aligned on its
    own, as under cat_axes_align.
    """
    return _super_lookup().get(str(category), f"unmapped:{category}")


_SUPER_LOOKUP: Optional[dict] = None


def _super_lookup() -> dict:
    global _SUPER_LOOKUP
    if _SUPER_LOOKUP is None:
        from o3b.dataset.uco3d.enum import MAP_UCO3D_CATEGORIES_TO_SUPER as SUP
        from o3b.dataset.housecorr3d.omni6dpose_uco3d import MAP_CATEGORIES_OMNI6DPOSE_TO_UCO3D
        from o3b.dataset.imagenet3d.enum import MAP_CATEGORIES_IMAGENET3D_TO_UCO3D
        from o3b.dataset.handal.enum import MAP_CATEGORIES_HANDAL_TO_UCO3D
        # the maps are keyed by enum members (str() of an IMAGENET3D_CATEGORIES
        # member is "IMAGENET3D_CATEGORIES.AEROPLANE", not "aeroplane")
        name = lambda x: str(getattr(x, "value", x))
        sup = {name(k): name(v) for k, v in SUP.items()}
        lut = dict(sup)
        for m in (MAP_CATEGORIES_OMNI6DPOSE_TO_UCO3D, MAP_CATEGORIES_IMAGENET3D_TO_UCO3D,
                  MAP_CATEGORIES_HANDAL_TO_UCO3D):
            for k, v in m.items():
                if v is not None and name(k) not in lut and name(v) in sup:
                    lut[name(k)] = sup[name(v)]
        _SUPER_LOOKUP = lut
    return _SUPER_LOOKUP


def supercategory_axes_align(categories: list, pred_rot, gt_rot, syms: dict, **kw) -> tuple[dict, dict]:
    """category_axes_align with one relabelling per UCO3D super-category:
    metrics renamed ``*_catalign`` -> ``*_supercatalign``, choices keyed by
    super-category."""
    metrics, choice = category_axes_align([supercategory_of(c) for c in categories],
                                          pred_rot, gt_rot, syms, **kw)
    metrics = {k.replace("_catalign", "_supercatalign").replace("pose_catalign_", "pose_supercatalign_"): v
               for k, v in metrics.items()}
    return metrics, {f"super{k}" if k.startswith("_") else f"super_{k}": v for k, v in choice.items()}


def orientation_consistency(categories: list, pred_rot: torch.Tensor, gt_rot: torch.Tensor,
                            syms: dict, thr_deg: float = 30.0, share: float = 0.2, min_n: int = 16,
                            device: Optional[str] = None) -> tuple[dict, dict]:
    """Is each category, and each super-category, predicted in one axis convention?

    Per frame and per proper signed permutation P (24): correct if the error of
    R_pred @ P is below thr_deg. For a group of frames, P1 = the P with the most
    correct frames. The group is **inconsistent** if another P is correct on
    >= ``share`` of the group's frames *that P1 gets wrong* (the frames P1 already
    fixes are left out, so a P that symmetry makes equivalent to P1 never
    counts): both conventions appear, and whichever single P is chosen the other
    frames stay wrong.

        pose_cat_consistent{suffix}       share of categories (>= min_n frames)
                                          that are consistent
        pose_supercat_consistent{suffix}  the same over UCO3D super-categories,
                                          frames pooled, those with >= 2 categories

    Returns (metrics, {f"{cat|supercat}{suffix}": {group: (consistent, n, P1, P2,
    share of P1, share of P2)}}).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Ps = proper_signed_permutations().to(device=device, dtype=torch.float32)
    D0 = _unit_rows(gt_rot.float().to(device)).transpose(-1, -2) @ _unit_rows(pred_rot.float().to(device))
    thr = math.radians(thr_deg)
    cats = [str(c) for c in categories]
    groups = {"cat": {}, "supercat": {}}
    for i, c in enumerate(cats):
        groups["cat"].setdefault(c, []).append(i)
        groups["supercat"].setdefault(supercategory_of(c), []).append(i)
    n_cats_in = {s: len({cats[i] for i in ii}) for s, ii in groups["supercat"].items()}
    metrics, detail = {}, {}
    for suffix, s in syms.items():
        if s is None and suffix != "":
            continue
        s_dev = None if s is None else s.to(device)
        ok = torch.stack([_errors(D0 @ P, s_dev) for P in Ps]) < thr        # (24, N)
        for level, gs in groups.items():
            res = {}
            for g, ii in gs.items():
                if len(ii) < min_n or (level == "supercat" and n_cats_in[g] < 2):
                    continue
                o = ok[:, torch.tensor(ii, device=device)]
                h = o.sum(1)
                p1 = int(h.argmax())
                h2 = (o & ~o[p1][None]).sum(1)
                p2 = int(h2.argmax())
                f1, f2 = h[p1].item() / len(ii), h2[p2].item() / len(ii)
                res[g] = (f2 < share, len(ii), p1, p2, f1, f2)
            key = f"pose_{level}_consistent{suffix}"
            metrics[key] = sum(r[0] for r in res.values()) / len(res) if res else float("nan")
            metrics[f"pose_{level}_consistent_n{suffix}"] = float(len(res))
            detail[f"{level}{suffix}"] = res
    return metrics, detail
