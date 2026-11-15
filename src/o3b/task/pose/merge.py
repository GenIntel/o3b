"""Merge per-sample pose dumps of a sharded eval into full-set metrics.

A method too slow for one job's walltime (SAM3D, seconds per frame) is scored as
many short jobs, each one block of the eval set (``eval.shard_size`` /
``eval.shard_index``) with ``eval.dump_samples`` set. This module joins their
dumps and reports what one job over the whole set would have:

* every per-sample metric PoseTask writes (``pose_rot_acc_*``, ``*_sym``,
  ``*_symcat``, IoU, ...) as its mean over all samples. A single run reports
  the mean of per-batch means instead, which differs only in the last decimals
  where a batch is short;
* with ``--class-mean``, every one of them also as cutoop's class mean
  (``<metric>_classmean``: per category, then over categories), the average the
  Omni6DPose / GenPose++ / Beyond 'Templates' tables report; ``--per-category``
  writes the per-category means as CSV. Works on the dump of a single run too;
* with ``--cat-axes-align``, the ``*_catalign`` metrics over the joined set, so
  each category's relabelling is chosen on all of its frames, not on the slice
  one job happened to see.

A shard that ran twice (a requeue after preemption) leaves two dumps; the newest
per shard index is kept. ``--expect N`` fails unless exactly N distinct samples
were joined.

    python -m o3b.task.pose.merge '/work/.../dumps/full_shards/*pose_every9d__sam3d__*' \\
        --cat-axes-align --expect 60716 --out sam3d_every9d_full.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Optional

import torch

_SHARD_RE = re.compile(r"__b(\d+)_s(\d+)")


def _shard_key(path: str) -> Optional[tuple]:
    m = _SHARD_RE.search(os.path.basename(path))
    rank = re.search(r"\.rank(\d+)\.pt$", path)
    return None if m is None else (int(m.group(1)), int(m.group(2)), int(rank.group(1)) if rank else 0)


def collect(paths: list[str]) -> list[dict]:
    """Batch records of the newest dump per (shard size, shard index, rank)."""
    newest: dict = {}
    for p in paths:
        key = _shard_key(p) or (p,)
        if key not in newest or os.path.getmtime(p) > os.path.getmtime(newest[key]):
            newest[key] = p
    recs: list = []
    for p in sorted(newest.values()):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    return recs


def per_category_means(recs: list[dict]) -> dict:
    """{metric: {category: (mean, n)}} of every per-sample metric PoseTask writes,
    non-finite values dropped within a category (as merge() drops them overall)."""
    from o3b.task.datatypes.frame_object_quant import FrameObjectQuantBatch

    cats = [str(c) for r in recs for c in (r.get("category") or [])]
    ids: dict = {}
    idx = torch.tensor([ids.setdefault(c, len(ids)) for c in cats], dtype=torch.long)
    out: dict = {}
    for k in FrameObjectQuantBatch._FIELDS:
        if not all(isinstance(r.get(k), torch.Tensor) for r in recs if r.get("category")):
            continue
        v = torch.cat([r[k].float().reshape(-1) for r in recs if r.get("category")])
        if v.numel() != idx.numel():
            continue  # not one value per sample (should not happen for PoseTask fields)
        ok = torch.isfinite(v)
        s = torch.zeros(len(ids)).index_add_(0, idx[ok], v[ok])
        n = torch.zeros(len(ids)).index_add_(0, idx[ok], torch.ones(int(ok.sum())))
        out[k] = {c: ((s[i] / n[i]).item(), int(n[i])) for c, i in ids.items() if n[i] > 0}
    return out


def class_means(recs: list[dict]) -> dict:
    """cutoop's class mean (Omni6DPose / GenPose++ / Beyond 'Templates' tables):
    each metric averaged within a category, then over categories, every category
    weighted equally. Returned as ``<metric>_classmean`` plus ``n_classes``."""
    per = per_category_means(recs)
    out = {f"{k}_classmean": sum(m for m, _ in d.values()) / len(d) for k, d in per.items() if d}
    out["n_classes"] = float(len({str(c) for r in recs for c in (r.get("category") or [])}))
    return out


def merge(recs: list[dict], cat_axes_align: bool = False,
          rot_acc_deg=(5.0, 10.0, 30.0), supercat_axes_align: bool = False,
          class_mean: bool = False) -> tuple[dict, dict]:
    from o3b.task.datatypes.frame_object_quant import FrameObjectQuantBatch

    metrics: dict = {}
    for k in FrameObjectQuantBatch._FIELDS:
        vals = [r[k] for r in recs if isinstance(r.get(k), torch.Tensor)]
        if vals:
            v = torch.cat([x.float().reshape(-1) for x in vals])
            v = v[torch.isfinite(v)]
            if v.numel():
                metrics[k] = v.mean().item()
    metrics["n_samples"] = float(sum(len(r.get("category") or []) for r in recs))
    if class_mean:
        metrics.update(class_means(recs))
    choice: dict = {}
    if cat_axes_align or supercat_axes_align:
        from o3b.task.pose.cat_align import category_axes_align, supercategory_axes_align

        def _cat(key):
            vals = [r.get(key) for r in recs]
            return None if any(v is None for v in vals) else torch.cat(vals)

        gt = _cat("cam_tform4x4_obj_ncds")
        if gt is None:
            gt = _cat("cam_tform4x4_obj")
        args = ([str(c) for r in recs for c in r["category"]],
                _cat("pred_cam_tform4x4_obj")[..., :3, :3], gt[..., :3, :3],
                {"": None, "_sym": _cat("obj_syms"), "_symcat": _cat("obj_syms_cat")})
        if cat_axes_align:
            extra, choice = category_axes_align(*args, rot_acc_deg=rot_acc_deg)
            metrics.update(extra)
        if supercat_axes_align:
            extra, c2 = supercategory_axes_align(*args, rot_acc_deg=rot_acc_deg)
            metrics.update(extra)
            choice.update(c2)
        from o3b.task.pose.cat_align import orientation_consistency
        extra, _ = orientation_consistency(*args)
        keep = (("pose_cat_",) if cat_axes_align else ()) + (("pose_supercat_", "pose_orientcat_") if supercat_axes_align else ())
        metrics.update({k: v for k, v in extra.items() if k.startswith(keep)})
    return metrics, choice


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("patterns", nargs="+", help="glob(s) of <dump>.rank<R>.pt files")
    ap.add_argument("--cat-axes-align", action="store_true")
    ap.add_argument("--supercat-axes-align", action="store_true",
                    help="one relabelling per UCO3D super-category (*_supercatalign)")
    ap.add_argument("--class-mean", action="store_true",
                    help="also <metric>_classmean: mean over categories of per-category means (cutoop)")
    ap.add_argument("--per-category", default=None,
                    help="write per-category means and counts of every metric as CSV")
    ap.add_argument("--expect", type=int, default=None,
                    help="fail unless this many distinct samples were joined")
    ap.add_argument("--out", default=None, help="write metrics (+ choices) as JSON")
    args = ap.parse_args(argv)

    paths = sorted({p for pat in args.patterns for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no dump matched")
    recs = collect(paths)
    ids = [i for r in recs for i in (r.get("frame_object_id") or [])]
    n = sum(len(r.get("category") or []) for r in recs)
    print(f"{len(paths)} files, {n} samples"
          + (f", {len(set(map(str, ids)))} distinct frame ids" if ids else
             " (no frame_object_id in the dumps: duplicates only caught by shard index)"))
    if ids and len(set(map(str, ids))) != n:
        raise SystemExit("duplicate samples across dumps (overlapping shards?)")
    if args.expect is not None and n != args.expect:
        raise SystemExit(f"expected {args.expect} samples, joined {n} (shards missing?)")
    metrics, choice = merge(recs, cat_axes_align=args.cat_axes_align,
                            supercat_axes_align=args.supercat_axes_align,
                            class_mean=args.class_mean)
    if args.per_category:
        import csv
        per = per_category_means(recs)
        keys = list(per)
        cats = sorted({c for d in per.values() for c in d})
        with open(args.per_category, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "n"] + keys)
            for c in cats:
                n = max(d[c][1] for d in per.values() if c in d)
                w.writerow([c, n] + [f"{per[k][c][0]:.6f}" if c in per[k] else "" for k in keys])
        print(f"-> {args.per_category} ({len(cats)} categories)")
    for k, v in metrics.items():
        print(f"  {k:<40} {v:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metrics": metrics, "catalign_choice": choice, "files": paths}, f, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
