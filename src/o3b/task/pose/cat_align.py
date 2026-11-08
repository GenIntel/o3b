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


def orientcategory_of(category: str) -> str:
    """UCO3D orientation category of a category of any pose test set: the
    categories that share one rule triple (left, back, top) of UCO3D's
    orientation tree (``MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS``), which Every9D's
    canonical frames are labelled by, keyed ``"<left>|<back>|<top>"`` (``-`` for
    an axis without a rule). Names map to UCO3D as in supercategory_of; a
    category no map covers is its own group (``"unmapped:<category>"``)."""
    return _orient_lookup().get(str(category), f"unmapped:{category}")


_SUPER_LOOKUP: Optional[dict] = None
_ORIENT_LOOKUP: Optional[dict] = None
# the maps are keyed by enum members (str() of an IMAGENET3D_CATEGORIES member
# is "IMAGENET3D_CATEGORIES.AEROPLANE", not "aeroplane")
_name = lambda x: str(getattr(x, "value", x))


def _via_uco3d(group: dict) -> dict:
    """{category of any test set: group}, from {UCO3D category: group}."""
    from o3b.dataset.housecorr3d.omni6dpose_uco3d import MAP_CATEGORIES_OMNI6DPOSE_TO_UCO3D
    from o3b.dataset.imagenet3d.enum import MAP_CATEGORIES_IMAGENET3D_TO_UCO3D
    from o3b.dataset.handal.enum import MAP_CATEGORIES_HANDAL_TO_UCO3D
    lut = dict(group)
    for m in (MAP_CATEGORIES_OMNI6DPOSE_TO_UCO3D, MAP_CATEGORIES_IMAGENET3D_TO_UCO3D,
              MAP_CATEGORIES_HANDAL_TO_UCO3D):
        for k, v in m.items():
            if v is not None and _name(k) not in lut and _name(v) in group:
                lut[_name(k)] = group[_name(v)]
    return lut


def _super_lookup() -> dict:
    global _SUPER_LOOKUP
    if _SUPER_LOOKUP is None:
        from o3b.dataset.uco3d.enum import MAP_UCO3D_CATEGORIES_TO_SUPER as SUP
        _SUPER_LOOKUP = _via_uco3d({_name(k): _name(v) for k, v in SUP.items()})
    return _SUPER_LOOKUP


def _orient_lookup() -> dict:
    global _ORIENT_LOOKUP
    if _ORIENT_LOOKUP is None:
        from o3b.dataset.uco3d.map_orient_tree import MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS as OK
        key = lambda v: "|".join("-" if x is None else _name(x) for x in v)
        _ORIENT_LOOKUP = _via_uco3d({_name(k): key(v) for k, v in OK.items()})
    return _ORIENT_LOOKUP


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
        pose_orientcat_consistent{suffix} the same over UCO3D orientation
                                          categories (one rule triple of the
                                          orientation tree, orientcategory_of)

    Returns (metrics, {f"{cat|supercat|orientcat}{suffix}": {group: (consistent, n, P1, P2,
    share of P1, share of P2)}}).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Ps = proper_signed_permutations().to(device=device, dtype=torch.float32)
    D0 = _unit_rows(gt_rot.float().to(device)).transpose(-1, -2) @ _unit_rows(pred_rot.float().to(device))
    thr = math.radians(thr_deg)
    cats = [str(c) for c in categories]
    groups = {"cat": {}, "supercat": {}, "orientcat": {}}
    for i, c in enumerate(cats):
        groups["cat"].setdefault(c, []).append(i)
        groups["supercat"].setdefault(supercategory_of(c), []).append(i)
        groups["orientcat"].setdefault(orientcategory_of(c), []).append(i)
    n_cats_in = {lv: {g: len({cats[i] for i in ii}) for g, ii in groups[lv].items()}
                 for lv in ("supercat", "orientcat")}
    metrics, detail = {}, {}
    for suffix, s in syms.items():
        if s is None and suffix != "":
            continue
        s_dev = None if s is None else s.to(device)
        ok = torch.stack([_errors(D0 @ P, s_dev) for P in Ps]) < thr        # (24, N)
        for level, gs in groups.items():
            res = {}
            for g, ii in gs.items():
                if len(ii) < min_n or (level != "cat" and n_cats_in[level][g] < 2):
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


def view_consistency(objects: list, pred_rot: torch.Tensor, gt_rot: torch.Tensor, syms: dict,
                     thr_deg: float = 30.0, min_n: int = 2,
                     device: Optional[str] = None) -> tuple[dict, dict]:
    """Does a method see one object frame from every viewpoint of an object?

    Per frame, D = R_gt^T R_pred takes the predicted object frame to the GT
    one; the camera motion between views cancels in it. If a method assigns an
    object the same frame from every view (rightly labelled or not), D is the
    same on all of that object's frames, up to the object's symmetry, which acts
    on the GT side: D_i = g_i D. So per pair of frames (i, j) of one object the
    view disagreement is the (symmetry-aware) angle of D_i D_j^T, which is in
    the symmetry group when the two views agree. Accuracy plays no part: a
    method that is consistently wrong scores 1, random rotations near 0.

        pose_view_consistent{suffix}     per object, the share of its frame pairs
                                         that agree within thr_deg; mean over
                                         objects (>= min_n frames)
        pose_view_disagree_deg{suffix}   per object the median pair angle;
                                         median over objects
        pose_view_consistent_n{suffix}   number of objects scored

    ``objects`` is any per-frame key of the object instance (its frame's GT
    object frame must be one and the same across the frames sharing a key).
    Returns (metrics, {f"obj{suffix}": {object: (n, share agreeing, median deg)}}).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    D0 = _unit_rows(gt_rot.float().to(device)).transpose(-1, -2) @ _unit_rows(pred_rot.float().to(device))
    groups = {}
    for i, o in enumerate(objects):
        groups.setdefault(o, []).append(i)
    groups = {o: ii for o, ii in groups.items() if len(ii) >= min_n}
    # every pair i < j of every object at once
    a, b, owner = [], [], []
    for k, ii in enumerate(groups.values()):
        t = torch.tensor(ii)
        iu = torch.triu_indices(len(ii), len(ii), 1)
        a.append(t[iu[0]]); b.append(t[iu[1]]); owner.append(torch.full((iu.shape[1],), k))
    metrics, detail = {}, {}
    if not groups:
        return metrics, detail
    a, b, owner = (torch.cat(x).to(device) for x in (a, b, owner))
    M = D0[a] @ D0[b].transpose(-1, -2)
    thr = math.radians(thr_deg)
    for suffix, s in syms.items():
        if s is None and suffix != "":
            continue
        # a pair shares one object, hence one symmetry code
        ang = _errors(M, None if s is None else s.to(device)[a])
        ok = (ang < thr).float()
        n_obj = len(groups)
        share = torch.zeros(n_obj, device=device).index_add_(0, owner, ok) / \
            torch.bincount(owner, minlength=n_obj).float()
        med = torch.stack([ang[owner == k].median() for k in range(n_obj)])
        metrics[f"pose_view_consistent{suffix}"] = share.mean().item()
        metrics[f"pose_view_disagree_deg{suffix}"] = math.degrees(med.median().item())
        metrics[f"pose_view_consistent_n{suffix}"] = float(n_obj)
        detail[f"obj{suffix}"] = {o: (len(ii), share[k].item(), math.degrees(med[k].item()))
                                  for k, (o, ii) in enumerate(groups.items())}
    return metrics, detail


def pair_consistency(groups: list, pred_rot: torch.Tensor, gt_rot: torch.Tensor, syms: dict,
                     distinct: Optional[list] = None, thr_deg: float = 30.0, max_pairs: int = 2000,
                     seed: int = 0, device: Optional[str] = None,
                     return_pairs: bool = False) -> tuple[dict, dict]:
    """Are two predictions of one group off from the GT by the same rotation?

    Per frame D = R_gt^T R_pred (predicted object frame -> GT frame). Two frames
    i, j are *consistent relative to each other* if D_i D_j^T is within thr_deg of
    the symmetry group, i.e. both predictions differ from their GT by one and the
    same relabelling (none, if both are right). Pairs are drawn within a group
    (``groups[i]``) and, if ``distinct`` is given, only between frames whose
    ``distinct`` key differs, so

        a video:               groups = object,                 distinct = None
        a category:            groups = category,               distinct = object
        an orientation cat.:   groups = orientcategory_of(cat), distinct = category

    Per group, all valid pairs if there are <= max_pairs, else max_pairs drawn
    uniformly (seeded). Score: per group the share of consistent pairs, mean
    over groups. No relabelling is chosen on the test set, and accuracy only
    enters as far as two right predictions are consistent: random rotations
    score near 0, a method that is wrong the same way everywhere scores 1.

    Symmetry: the pair's error is taken with the code of frame i and of frame
    j and the smaller one kept (codes only differ where symmetry is labelled
    per object, as in Omni6DPose).

        pose_pair_consistent{suffix}     mean over groups
        pose_pair_consistent_n{suffix}   groups with >= 1 valid pair

    Returns (metrics, {f"group{suffix}": {group: (pairs, share consistent)}}),
    plus ``detail[f"pairs{suffix}"] = (i, j, angle in rad)`` if return_pairs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    D0 = _unit_rows(gt_rot.float().to(device)).transpose(-1, -2) @ _unit_rows(pred_rot.float().to(device))
    gen = torch.Generator().manual_seed(seed)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    dk = None
    if distinct is not None:
        ids = {}
        dk = torch.tensor([ids.setdefault(d, len(ids)) for d in distinct])
    a, b, owner, names = [], [], [], []
    for g, ii in by.items():
        t = torch.tensor(ii)
        n = len(ii)
        if n < 2:
            continue
        n_all = n * (n - 1) // 2
        if n_all <= max_pairs:
            iu = torch.triu_indices(n, n, 1)
            pa, pb = t[iu[0]], t[iu[1]]
            if dk is not None:
                keep = dk[pa] != dk[pb]
                pa, pb = pa[keep], pb[keep]
        else:
            # rejection sampling of distinct-key pairs
            pa, pb = [], []
            tries = 0
            while sum(map(len, pa)) < max_pairs and tries < 20:
                x = t[torch.randint(n, (2 * max_pairs,), generator=gen)]
                y = t[torch.randint(n, (2 * max_pairs,), generator=gen)]
                keep = x != y if dk is None else dk[x] != dk[y]
                pa.append(x[keep]); pb.append(y[keep]); tries += 1
            pa, pb = torch.cat(pa)[:max_pairs], torch.cat(pb)[:max_pairs]
        if len(pa) == 0:
            continue
        a.append(pa); b.append(pb); owner.append(torch.full((len(pa),), len(names))); names.append(g)
    metrics, detail = {}, {}
    if not names:
        return metrics, detail
    a, b, owner = (torch.cat(x).to(device) for x in (a, b, owner))
    M = D0[a] @ D0[b].transpose(-1, -2)
    thr = math.radians(thr_deg)
    for suffix, s in syms.items():
        if s is None and suffix != "":
            continue
        if s is None:
            ang = _errors(M, None)
        else:
            s = s.to(device)
            ang = _errors(M, s[a])
            diff = (s[a] != s[b]).any(-1)
            if diff.any():
                ang[diff] = torch.minimum(ang[diff], _errors(M[diff], s[b][diff]))
        ok = (ang < thr).float()
        ng = len(names)
        share = torch.zeros(ng, device=device).index_add_(0, owner, ok) / \
            torch.bincount(owner, minlength=ng).float()
        metrics[f"pose_pair_consistent{suffix}"] = share.mean().item()
        metrics[f"pose_pair_consistent_n{suffix}"] = float(ng)
        cnt = torch.bincount(owner, minlength=ng)
        detail[f"group{suffix}"] = {g: (int(cnt[k]), share[k].item()) for k, g in enumerate(names)}
        if return_pairs:
            detail[f"pairs{suffix}"] = (a.cpu(), b.cpu(), ang.cpu())
    return metrics, detail
