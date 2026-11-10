"""Score hand annotations against the official labels.

    o3b dataset annotate-obb-compare -d uco3d

The official box is what the UCO3D loader serves: the bounds of ``mesh.ply`` in
the canonical frame ``tform_obj`` maps it to, axis-aligned there.  Rotations are
compared in od3d's axis order, the one ``obj_syms_for_category`` is written in,
with the symmetry applied on the object side.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from o3b.dataset.verify_labels import store
from o3b.dataset.verify_labels.pack import default_dir, load_object


def official_box(tform_obj, verts: Optional[np.ndarray] = None) -> dict:
    """The official box in the raw frame; without ``verts`` only its rotation."""
    T = np.asarray(tform_obj, dtype=np.float64)
    A = T[:3, :3]
    s = float(np.linalg.norm(A, axis=0).mean())
    R = A / s
    box = {"R_obj_raw": (store.G3 @ R).tolist(), "center_raw": None, "extents_raw": None}
    if verts is not None:
        canonical = np.asarray(verts, dtype=np.float64) @ A.T + T[:3, 3]
        lo, hi = canonical.min(0), canonical.max(0)
        box["center_raw"] = (R.T @ (((lo + hi) / 2 - T[:3, 3]) / s)).tolist()
        box["extents_raw"] = (np.abs(store.G3) @ ((hi - lo) / s)).tolist()
    return box


def _has_box(a: dict) -> bool:
    return a.get("center_raw") is not None and a.get("extents_raw") is not None


def score(annotation: dict, official: dict, category: str) -> dict:
    """Rotation, box and size agreement of one annotation with one reference."""
    from o3b.cv.metric.bbox3d import bbox3d_iou_pair
    from o3b.cv.metric.pose import get_pose_diff3x3_in_rad
    from o3b.dataset.uco3d.obj_syms import obj_syms_for_category

    Ra = store.G3.T @ np.asarray(annotation["R_obj_raw"], dtype=np.float64)     # raw -> od3d axes
    Rb = store.G3.T @ np.asarray(official["R_obj_raw"], dtype=np.float64)
    syms = obj_syms_for_category(category)
    diff = torch.tensor(Ra @ Rb.T, dtype=torch.float32)[None]
    out = {"rot_deg": float(np.degrees(get_pose_diff3x3_in_rad(diff).item())),
           "rot_sym_deg": float("nan"), "iou": float("nan"), "iou_sym": float("nan"),
           "center_pct_diag": float("nan"), "volume_ratio": float("nan")}
    out["rot_sym_deg"] = out["rot_deg"] if syms is None else float(np.degrees(
        get_pose_diff3x3_in_rad(diff, obj_rot3d_obj_syms=syms[None]).item()))

    if _has_box(annotation) and _has_box(official):
        ca, cb = np.asarray(annotation["center_raw"]), np.asarray(official["center_raw"])
        ea, eb = np.asarray(annotation["extents_raw"]), np.asarray(official["extents_raw"])
        corners_b = store.corners(official["R_obj_raw"], cb, eb)
        out["iou"] = bbox3d_iou_pair(store.corners(annotation["R_obj_raw"], ca, ea), corners_b)
        out["center_pct_diag"] = float(100 * np.linalg.norm(ca - cb) / np.linalg.norm(eb))
        out["volume_ratio"] = float(np.prod(ea) / np.prod(eb))
        out["iou_sym"] = out["iou"]
        if syms is not None:
            from o3b.task.pose.task import symmetry_calibrated_rot3x3
            calibrated = symmetry_calibrated_rot3x3(
                torch.tensor(Ra.T, dtype=torch.float32)[None],
                torch.tensor(Rb.T, dtype=torch.float32)[None], syms[None])[0].double().numpy()
            # Sizes stay with their axes, as in cutoop's protocol (PoseTask). Any turn the
            # symmetry allows is as valid as that one -- the box as drawn included -- so
            # score the best overlap over all of them (_best_sym_iou), not only the turn
            # closest in rotation, which for a non-round object free about an axis can
            # land ~90 deg off with its extents swapped.
            out["iou_sym"] = max(out["iou"], bbox3d_iou_pair(
                store.corners(store.G3 @ calibrated.T, ca, ea), corners_b),
                _best_sym_iou(Ra, Rb, ca, ea, corners_b, syms))
    return out


def _axis_rot(axis: int, deg: float) -> np.ndarray:
    a, b = (axis + 1) % 3, (axis + 2) % 3
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    Q = np.eye(3)
    Q[a, a], Q[a, b], Q[b, a], Q[b, b] = c, -s, s, c
    return Q


def _best_sym_iou(Ra, Rb, ca, ea, corners_b, syms) -> float:
    """Best IoU over every turn of the annotated box that its symmetry allows.

    ``Ra`` / ``Rb`` map raw points to od3d's object axes, the ones ``syms`` is
    written in. A half turn about one of the box's own axes maps the box onto
    itself, so only continuous axes matter: one such axis is searched over
    [0, 180) deg (a box repeats every half turn), 2 deg then 0.25 deg around the
    best; full symmetry (all three continuous) puts the box on the reference's
    axes with its extents in any of the six orders.
    """
    from itertools import permutations

    from o3b.cv.metric.bbox3d import bbox3d_iou_pair

    code = [int(round(float(c))) for c in syms]
    free = [i for i, c in enumerate(code) if c < 0]
    if not free:
        return 0.0
    if len(free) == 3:
        return max(bbox3d_iou_pair(store.corners(store.G3 @ Rb, ca, np.asarray(ea)[list(p)]), corners_b)
                   for p in permutations(range(3)))
    axis = free[0]          # two continuous axes do not occur: two defined axes fix the third

    def iou_at(deg):
        return bbox3d_iou_pair(store.corners(store.G3 @ _axis_rot(axis, deg) @ Ra, ca, ea), corners_b)

    coarse = {d: iou_at(d) for d in np.arange(0.0, 180.0, 2.0)}
    best = max(coarse, key=coarse.get)
    fine = {d: iou_at(d) for d in np.arange(best - 2.0, best + 2.0001, 0.25)}
    return max(max(coarse.values()), max(fine.values()))


def summarize(rows: list, seed: int = 0, n_boot: int = 2000) -> dict:
    """Headline numbers of one cohort, with bootstrap 95 % intervals."""
    def stats(values, fn):
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        rng = np.random.default_rng(seed)
        boot = [fn(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)]
        return fn(v), *np.percentile(boot, [2.5, 97.5])

    sym = [r["rot_sym_deg"] for r in rows]
    iou = [r["iou_sym"] for r in rows]
    return {
        "n": len(rows),
        "rot_sym_median": stats(sym, np.median),
        "rot_sym_le10": stats(sym, lambda v: 100 * (v <= 10).mean()),
        "rot_sym_le30": stats(sym, lambda v: 100 * (v <= 30).mean()),
        "rot_sym_gt90": int(np.sum(np.asarray(sym) > 90)),
        "rot_median": stats([r["rot_deg"] for r in rows], np.median),
        "iou_sym_median": stats(iou, np.median),
        "iou_sym_ge50": stats(iou, lambda v: 100 * (v >= 0.5).mean()),
        "iou_sym_ge75": stats(iou, lambda v: 100 * (v >= 0.75).mean()),
    }


def _fmt(stat, unit: str = "") -> str:
    return "n/a" if stat is None else f"{stat[0]:.3g}{unit} [{stat[1]:.3g}, {stat[2]:.3g}]"


def summary_markdown(cohorts: dict, header: str = "", min_n: int = 0) -> str:
    lines = [header, "",
             "| cohort | n | cannot | sym° median | ≤10° % | ≤30° % | >90° | IoU sym median | IoU≥0.5 % | IoU≥0.75 % |",
             "|---|--:|--:|---|---|---|--:|---|---|---|"]
    for name, c in cohorts.items():
        s = c["summary"]
        if s["n"] < min_n and not name.startswith("floor_") and name != "old_anchored":
            lines.append(f"| {name} | {s['n']} | {c['cannot']} | hidden until {min_n} are saved | | | | | | |")
            continue
        lines.append(f"| {name} | {s['n']} | {c['cannot']} | {_fmt(s['rot_sym_median'])} | "
                     f"{_fmt(s['rot_sym_le10'])} | {_fmt(s['rot_sym_le30'])} | {s['rot_sym_gt90']} | "
                     f"{_fmt(s['iou_sym_median'])} | {_fmt(s['iou_sym_ge50'])} | {_fmt(s['iou_sym_ge75'])} |")
    return "\n".join(lines)


def _sample_ids(root) -> dict:
    """Sample list file stem -> ids, from ``<root>/samples``."""
    from o3b.dataset.subset_store import _parse
    return {p.stem: set(_parse(p.read_text())) for p in sorted((Path(root) / "samples").glob("*.yaml"))}


def _sample_headers(root) -> dict:
    """Sample list file stem -> its ``# key: value`` header lines, as strings."""
    out = {}
    for p in sorted((Path(root) / "samples").glob("*.yaml")):
        head = {}
        for line in p.read_text().splitlines():
            if not line.startswith("#"):
                break
            key, sep, value = line[1:].partition(":")
            if sep:
                head[key.strip()] = value.strip()
        out[p.stem] = head
    return out


def run_compare(root, tforms: dict, pack_root=None, out_csv=None, db_path=None, min_n: int = 0,
                mesh_root=None) -> tuple:
    """Score every annotation on disk. Returns ``(cohorts, markdown)``.

    Cohorts: one per sample list (blind annotations plus the old labels it
    reuses), ``<sample>_blind`` (its blind annotations alone), ``old_anchored``
    (every old label; edited from the pipeline's pose, so anchored), and the floors ``repeat`` (annotator vs
    themselves) and ``emil`` (old labels vs a second annotator).  An annotation
    edited after its official box was shown is scored at its last blind version;
    one with no blind version left is dropped and counted.

    ``mesh_root`` (``<path_preprocess>/mesh/<mesh_type>/meta_mask/meta``) takes the
    official box from another mesh type than the pack's; an object without that
    mesh is scored on rotation only.
    """
    pack_root = Path(pack_root) if pack_root else default_dir()
    root = Path(root)
    samples = _sample_ids(root)
    heads = _sample_headers(root)
    flag = lambda name, key: heads.get(name, {}).get(key, "").lower() == "true"
    # ``redo`` lists only re-open objects of other lists; ``exclude`` lists take
    # their objects out of every cohort (cohort "excluded", kept visible).
    split_of = {i: name for name, ids in samples.items()
                if not flag(name, "redo") and not flag(name, "exclude") for i in ids}
    split_of.update({i: "excluded" for name, ids in samples.items() if flag(name, "exclude") for i in ids})
    after_review = 0
    rows, cannot, unblinded = [], {}, 0

    verts_cache = {}

    def verts_of(seq):
        if seq not in verts_cache:
            try:
                if mesh_root is not None:
                    import trimesh
                    path = Path(mesh_root) / seq / "mesh.ply"
                    verts_cache[seq] = (np.asarray(trimesh.load(path, process=False).vertices)
                                        if path.exists() else None)
                else:
                    verts_cache[seq] = load_object(pack_root, seq)["verts"]
            except FileNotFoundError:
                verts_cache[seq] = None
        return verts_cache[seq]

    def add(cohort, version, reference, seq):
        rows.append({"cohort": cohort, "sequence_name": seq,
                     **score(version, reference, seq.split("/")[0])})

    for record in store.iter_all(root, store.ANNOTATIONS):
        seq = record["sequence_name"]
        cohort = split_of.get(seq, "unsampled")
        # A relabel asked for after the official box was reviewed is scored as
        # saved: it is not blind, and the header counts how many there are.
        if record.get("after_review"):
            version = record
            after_review += 1
        else:
            version = store.blind_version(record)
        if version is None:
            unblinded += 1
            continue
        if version.get("status") == "cannot_annotate":
            cannot[cohort] = cannot.get(cohort, 0) + 1
            continue
        if seq not in tforms:
            print(f"  {seq}: not in the official table")
            continue
        off = official_box(tforms[seq], verts_of(seq))
        add(cohort, version, off, seq)
        add(f"{cohort}_blind", version, off, seq)
        if version.get("start", record.get("start")) == store.PRELABELS:
            add(f"{cohort}_prelabel_reviewed", version, off, seq)

    for record in store.iter_all(root, store.PRELABELS):
        # The model's boxes as they were, before review: how much the review changed.
        seq = record["sequence_name"]
        if seq in tforms:
            add(f"claude_{split_of.get(seq, 'unsampled')}", record, official_box(tforms[seq], verts_of(seq)), seq)

    for record in store.iter_all(root, store.OLD):
        seq = record["sequence_name"]
        if seq in tforms:
            off = official_box(tforms[seq], verts_of(seq))
            add("old_anchored", record, off, seq)
            # A new annotation of the same object replaces the old label in its sample.
            if seq in split_of and store.load(root, seq) is None:
                add(split_of[seq], record, off, seq)

    # The floors: person against person, no official label involved.
    for record in store.iter_all(root, store.REPEATS):
        seq = record["sequence_name"]
        first = store.load(root, seq)
        if first and record.get("status") == "done" and first.get("status") == "done":
            add("floor_repeat", record, first, seq)
    for record in store.iter_all(root, store.EMIL):
        seq = record["sequence_name"]
        old = store.load(root, seq, store.OLD)
        if old:
            add("floor_emil", record, old, seq)

    names = sorted({r["cohort"] for r in rows} | set(cannot))
    cohorts = {name: {"summary": summarize([r for r in rows if r["cohort"] == name]),
                      "cannot": cannot.get(name, 0)} for name in names}

    db_note = ""
    if db_path is not None and Path(db_path).exists():
        db_note = (f" official: {Path(db_path).name}, sha256 "
                   f"{hashlib.sha256(Path(db_path).read_bytes()).hexdigest()[:12]}, "
                   f"mtime {int(Path(db_path).stat().st_mtime)}.")
    header = (f"**Compared {len(rows)} annotations.**{db_note}"
              + (f" {unblinded} dropped: no blind version." if unblinded else "")
              + (f" {after_review} relabelled after reviewing the official box." if after_review else ""))
    out_csv = Path(out_csv) if out_csv else root / "quantitative" / "h09_compare.csv"
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return cohorts, summary_markdown(cohorts, header, min_n)
