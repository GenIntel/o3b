"""Per-category axis editor for canonical object poses.

    o3b dataset axes-tform-obj-type -d every9d_v5
    o3b dataset axes-tform-obj-type -d every9d_v5 -r every9d_v3

UCO3D's canonical poses are labelled per sequence, but the *axis convention* is a
property of the category — "the bottle's up is +Y", not "this bottle's up".  When
a category comes out with its axes permuted or mirrored relative to o3b's
right/top/back = X/Y/Z, the fix is one rotation applied to every sequence of that
category.  This is the tool for finding that rotation and writing it.

The viewer is 2-D: a grid of the category's objects — 8 across, 3 of each
object's own frames stacked below it, two such blocks, so one screen holds 16
objects — each with the current 3-D bounding box and an axis triad drawn into
the image: red = right (X), green = top (Y), blue = back (Z).
That is the view that actually answers "are these axes right?", because it shows
the convention against the object as photographed.  The category's own axis rules
from UCO3D's orientation tree sit next to it, stated in the config's frame and
*fixed* there: they are what the arrows are supposed to become, so they do not
move when the arrows do.  Swap/flip keys rotate the whole category at once and
the arrows follow live; save writes the composed transform back for every
sequence in it.

**Only proper rotations are written.**  A single swap or a single flip is a
reflection (det = -1) — applying one would mirror every object in the category,
turning left shoes into right shoes.  They have to come in pairs, and save
refuses while the pending transform has det != +1.

Where the pending rotation P goes.  The dataset shows verts in
``G @ T @ v_raw`` (T = the stored tform_obj, G = the config's
``obj_gl_tform4x4_obj_raw``).  P is applied in *that* displayed frame, so the
transform to store is ``inv(G) @ P @ G @ T`` — which reduces to ``P @ T`` when
the config sets no G.
"""
from __future__ import annotations

import math
import sys
import time
from functools import partial
from typing import Optional

import logging

from pathlib import Path

import numpy as np
import torch

from o3b.dataset.grid import (  # noqa: F401  (re-exported for the subset editor)
    CACHE_MAX_BYTES, GRID_COLS, GRID_OBJECTS, GRID_VIEWS, PREFETCH,
)
from o3b.dataset.uco3d.dataset import _load_meta_yaml

logger = logging.getLogger(__name__)

# Axis semantics, matching o3b's convention and o3b/dataset/tform.py's colours.
_AXES = [
    ("X", "right", (0.9, 0.2, 0.2)),
    ("Y", "top",   (0.2, 0.85, 0.2)),
    ("Z", "back",  (0.25, 0.45, 1.0)),
]


# ── the category's own axis rules ─────────────────────────────────────────────

def _reverse_rule(rule: str) -> str:
    """`vec_bottom_top` seen along the opposite direction is `vec_top_bottom`.

    Every rule in the tree is ``<kind>_<a>_<b>``, so reversing is swapping the
    ends.  ``axis_*`` rules name a *line* rather than a direction (which is why
    obj_syms reads them as 180-degree symmetric), so reversing one changes
    nothing and it is returned untouched.
    """
    parts = rule.split("_")
    if len(parts) == 3 and parts[0] == "vec":
        return f"{parts[0]}_{parts[2]}_{parts[1]}"
    return rule


def axis_description(category: str, M: "Optional[np.ndarray]" = None) -> str:
    """UCO3D's per-axis orientation rules, expressed in the *displayed* frame.

    The tree indexes its rules by UCO3D's own canonical axes.  What the viewer
    draws is those axes mapped through ``M`` — the config's
    ``obj_gl_tform4x4_obj_raw`` — so quoting the rules unpermuted labels the
    wrong arrow: for `chair` under o3b's current G it would claim the green
    (top) arrow is ``user_backward``, when the axis drawn there is UCO3D's Z,
    ``stand_upward``.  Passing ``M`` re-indexes them, and reverses a ``vec_``
    rule whose axis ends up flipped.

    ``M`` is the config's G alone — never G with the axes editor's pending
    swaps composed on top.  The rules state what X, Y and Z are *supposed* to
    mean, which is the fixed thing the pending rotation is being judged
    against; folding the pending rotation in would move the statement along
    with the arrows and leave nothing to compare them to.

    An axis with no rule is one the labelling never pinned down, which is
    exactly the axis a symmetry lives on (see uco3d/obj_syms.py).
    """
    try:
        from o3b.dataset.uco3d.map_orient_tree import MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS
        from o3b.dataset.uco3d.orient_tree import MAP_CONDITION_TO_RULE
    except Exception:
        return "_(no orientation tree available)_"

    keys = MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS.get(category)
    if keys is None:
        return f"**{category}** — _not in the UCO3D orientation tree_"

    frame = "as displayed" if M is not None else "UCO3D frame"
    lines = [f"**{category}** — UCO3D axis definition ({frame}):", ""]
    for i, (axis, name, _) in enumerate(_AXES):
        if M is None:
            src, sign = i, 1.0
        else:
            src = int(np.argmax(np.abs(M[i])))
            sign = float(np.sign(M[i, src])) or 1.0
        key = keys[src]
        via = "" if (M is None or (src == i and sign > 0)) else \
              f"  [= {'-' if sign < 0 else '+'}{_AXES[src][0]}_uco3d]"
        if key is None:
            lines.append(f"- `{axis}` ({name}): _undefined — free / symmetric_{via}")
            continue
        rule = MAP_CONDITION_TO_RULE.get(key, "")
        shown = _reverse_rule(rule) if (sign < 0 and rule) else rule
        note = " _(reversed)_" if (sign < 0 and shown != rule) else ""
        lines.append(f"- `{axis}` ({name}): **{key}**{via}")
        if shown:
            lines.append(f"    - {shown}{note}")
    return "\n".join(lines)


# ── the pending per-category rotation ─────────────────────────────────────────

class Pending:
    """The 3x3 the user is building out of swaps and flips."""

    def __init__(self) -> None:
        self.P = np.eye(3, dtype=np.float64)
        self._history: list[np.ndarray] = []

    def _apply(self, M: np.ndarray) -> None:
        self._history.append(self.P.copy())
        self.P = M @ self.P

    def swap(self, i: int, j: int) -> None:
        M = np.eye(3)
        M[i, i] = M[j, j] = 0.0
        M[i, j] = M[j, i] = 1.0
        self._apply(M)

    def flip(self, i: int) -> None:
        M = np.eye(3)
        M[i, i] = -1.0
        self._apply(M)

    def undo(self) -> bool:
        if not self._history:
            return False
        self.P = self._history.pop()
        return True

    def reset(self) -> None:
        self._history.append(self.P.copy())
        self.P = np.eye(3, dtype=np.float64)

    @property
    def det(self) -> float:
        return float(np.linalg.det(self.P))

    @property
    def is_rotation(self) -> bool:
        """det == +1 and orthonormal — a rotation, not a reflection."""
        return (abs(self.det - 1.0) < 1e-6
                and np.abs(self.P @ self.P.T - np.eye(3)).max() < 1e-6)

    @property
    def is_identity(self) -> bool:
        return np.abs(self.P - np.eye(3)).max() < 1e-9

    def as_text(self) -> str:
        rows = ["[" + ", ".join(f"{v:5.1f}" for v in r) + "]" for r in self.P]
        return "  ".join(rows)


# ── loading a category ────────────────────────────────────────────────────────

def _load_category(cls, cfg, category: str, n_objects: int, n_views: int,
                   cache: bool = True):
    """Per-frame overlay inputs for up to n_objects of *category*.

    Two measurements shape this. Geometry costs 0.01 s/item; decoding the rgb
    costs 0.70 s/item, so the rgb path is the only thing worth optimising. And
    re-opening the mp4 for every frame costs 0.33 s where reusing one
    VideoCapture across a sequence's frames costs 0.10 s — the container parse,
    not the decode, dominates.  So: pull geometry from the dataset with rgb
    switched *off*, then fetch the frames per sequence through a single reused
    capture, sequences in parallel (cv2 releases the GIL while decoding), and
    cache the finished crops so revisiting a category is free.
    """
    from dataclasses import replace as _r

    tag = f"{n_objects}|{n_views}"
    if cache:
        hit = _cache_load(cfg, category, tag)
        if hit is not None:
            return hit

    extra = dict(cfg.extra or {})
    extra["frames_count_max_per_sequence"] = n_views
    extra["sequences_count_max_per_category"] = n_objects
    view_cfg = _r(
        cfg,
        categories=[category],
        filter_count_max=n_objects * n_views,
        transform=None,
        modalities=view_modalities(),
        extra=extra,
    )
    try:
        dataset = cls(view_cfg)
    except Exception as exc:
        print(f"  could not build dataset for {category}: {exc}", file=sys.stderr)
        return []

    objects = collect_frames(dataset, cfg)[:n_objects]
    if cache and objects:
        _cache_store(cfg, category, tag, objects)
    return objects


def view_modalities() -> set:
    """What the overlay needs — deliberately no "rgb".

    rgb is the 0.70 s/item, and ``collect_frames`` fetches it far more cheaply
    through one reused VideoCapture per sequence.
    """
    return {"cam_intr4x4", "cam_bbox3d", "cam_tform4x4_obj_ncds", "obj_bbox3d", "mesh"}


def collect_frames(dataset, cfg, indices=None) -> list:
    """``[(object_id, [frame, ...]), …]`` for *indices* of an already-built dataset.

    Geometry comes from the dataset (rgb switched off), the pixels from one
    reused VideoCapture per sequence, sequences in parallel.  Objects come back
    in the order their first frame was seen, so a caller that passed a page of
    indices gets that page back in the same order.
    """
    jobs: dict = {}
    for i in (range(len(dataset)) if indices is None else indices):
        try:
            fo = dataset[i]
        except Exception as exc:
            print(f"  item {i}: {exc}", file=sys.stderr)
            continue
        if fo is None or fo.cam_bbox3d is None or fo.cam_tform4x4_obj_ncds is None:
            continue
        row = dataset._frame_rows[dataset._frame_rows_id[i]]
        meta = _load_meta_yaml(
            Path(cfg.path_preprocess) / "meta" / "frames"
            / row["category"] / row["sequence"] / f'{row["frame"]}.yaml'
        )
        if meta is None or not meta.get("rfpath_rgb"):
            continue
        jobs.setdefault(fo.object_id, {"video": Path(cfg.path_raw) / meta["rfpath_rgb"],
                                       "frames": []})
        jobs[fo.object_id]["frames"].append((float(meta.get("timestamp", 0.0)), fo))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as pool:
        results = list(pool.map(_read_sequence, jobs.items()))

    return [(seq, frames) for seq, frames in results if frames]


def _read_sequence(item):
    """(object_id, [frame, ...]) — one VideoCapture for all of a sequence's frames."""
    import cv2

    object_id, job = item
    cap = cv2.VideoCapture(str(job["video"]))
    if not cap.isOpened():
        print(f"  could not open {job['video']}", file=sys.stderr)
        return object_id, []
    out = []
    try:
        for timestamp, fo in sorted(job["frames"], key=lambda x: x[0]):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
            fo.rgb = rgb[:3]
            try:
                frame = _crop_frame(fo)
            except Exception as exc:      # one bad frame must not lose the page
                print(f"  {object_id}: {exc}", file=sys.stderr)
                continue
            if frame is not None:
                out.append(frame)
    finally:
        cap.release()
    return object_id, out


# ── crop cache ────────────────────────────────────────────────────────────────

def cache_dir() -> Path:
    return Path.home() / ".o3b" / "cache" / "axes"


def _cache_key(cfg, category: str, tag: str) -> Path:
    """Where a set of crops for *category* is cached.

    *tag* distinguishes selections of that category — the axes editor's
    objects/views counts, the subset editor's page — and is hashed, so it can be
    as long as a page's object ids.  The file name keeps the category as a
    prefix so ``_cache_drop`` can forget all of them at once.
    """
    import hashlib

    raw = f"{cfg.path_preprocess}|{cfg.tform_obj_type}|{category}|{tag}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return cache_dir() / f"{category}_{h}.npz"


def cache_has(cfg, category: str, tag: str) -> bool:
    """Is this page already on disk? — what the prefetcher checks before working."""
    return _cache_key(cfg, category, tag).exists()


def _cache_load(cfg, category: str, tag: str):
    path = _cache_key(cfg, category, tag)
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        n = int(z["n_frames"].sum())
        objects, k = [], 0
        for si, cnt in enumerate(z["n_frames"]):
            frames = []
            for _ in range(int(cnt)):
                frames.append({
                    "rgb": torch.from_numpy(z[f"rgb_{k}"]).float() / 255.0,
                    "intr": torch.from_numpy(z[f"intr_{k}"]),
                    "cam_bbox3d": torch.from_numpy(z[f"cbox_{k}"]),
                    "obj_bbox3d": torch.from_numpy(z[f"obox_{k}"]),
                    "ncds": torch.from_numpy(z[f"ncds_{k}"]),
                })
                k += 1
            objects.append((str(z["seqs"][si]), frames))
        logger.debug(f"axes cache hit: {path.name} ({n} frames)")
        return objects
    except Exception as exc:
        logger.debug(f"ignoring bad cache {path}: {exc}")
        return None


def _cache_store(cfg, category: str, tag: str, objects) -> None:
    path = _cache_key(cfg, category, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict = {"n_frames": np.array([len(f) for _s, f in objects], dtype=np.int32),
                  "seqs": np.array([s for s, _f in objects])}
    k = 0
    for _seq, frames in objects:
        for f in frames:
            blob[f"rgb_{k}"] = (f["rgb"].clamp(0, 1) * 255).to(torch.uint8).numpy()
            blob[f"intr_{k}"] = f["intr"].numpy()
            blob[f"cbox_{k}"] = f["cam_bbox3d"].numpy()
            blob[f"obox_{k}"] = (f["obj_bbox3d"] if f["obj_bbox3d"] is not None
                                 else torch.zeros(8, 3)).numpy()
            blob[f"ncds_{k}"] = f["ncds"].numpy()
            k += 1
    try:
        np.savez(path, **blob)
    except Exception as exc:
        logger.debug(f"could not write {path}: {exc}")
        return
    _cache_prune()


def _cache_prune(max_bytes: int = CACHE_MAX_BYTES, d: "Optional[Path]" = None) -> None:
    """Drop the least recently used crops once the cache outgrows its budget.

    A page of 16 objects x 3 views is ~15 MB of uncompressed uint8, and the
    prefetcher now fills pages the user may never open, so an unbounded cache
    would reach tens of GB over a labelling session.  Least-recently-*used*, by
    atime where the filesystem keeps one and mtime otherwise: a page revisited
    all afternoon should outlive one seen once.
    """
    d = d or cache_dir()
    if not d.is_dir():
        return
    try:
        files = [(f, f.stat()) for f in d.glob("*.npz")]
    except OSError:
        return
    total = sum(s.st_size for _f, s in files)
    if total <= max_bytes:
        return
    for f, s in sorted(files, key=lambda it: max(it[1].st_atime, it[1].st_mtime)):
        if total <= max_bytes:
            break
        try:
            f.unlink()
            total -= s.st_size
        except OSError:
            pass


def _cache_drop(cfg, category: str) -> None:
    """Forget a category's crops — its geometry moved, so they are stale."""
    d = cache_dir()
    if d.is_dir():
        for f in d.glob(f"{category}_*.npz"):
            try:
                f.unlink()
            except OSError:
                pass


# How far a crop may outgrow the image before the view counts as degenerate
# rather than merely wide-angle.  4x is already an object filling a sixteenth of
# the frame's area; the failures this guards are 300x and up.
_CROP_HALF_MAX = 4.0


def _crop_frame(fo, margin: float = 0.45, cell: int = 320):
    """Crop the rgb around the projected 3-D box and adjust the intrinsics to match.

    The pad-crop-resize is ``o3b.cv.visual.crop.crop_with_bbox``: a rect wider
    than *cell* is resized *first*, so a box running off the image pays for its
    padding at cell resolution instead of the source's — 1.8 ms against 116 ms
    for a 1700 px rect out of 1920x1080, and the source-resolution pad is what
    used to run out of memory outright.  It also hands back the pixel-space
    shift+scale it used, which is exactly the intrinsics update.
    """
    from o3b.cv.visual.crop import crop_with_bbox

    intr = fo.cam_intr4x4.float().clone()
    uv = _project(fo.cam_bbox3d.float(), intr)                      # (8, 2)
    H, W = fo.rgb.shape[-2], fo.rgb.shape[-1]

    cx, cy = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    half = float(max((uv[:, 0].max() - uv[:, 0].min()),
                     (uv[:, 1].max() - uv[:, 1].min()))) * (0.5 + margin)
    half = max(half, 16.0)

    # A box corner at (or behind) the pinhole divides by ~0 in _project, so uv
    # comes back finite but astronomical, and the crop rect is then hundreds of
    # thousands of pixels wide.  Such a view shows nothing once downscaled to
    # `cell` anyway — drop it, the object keeps its other frames.
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(half)):
        return None
    if half > _CROP_HALF_MAX * max(H, W):
        return None
    if cx + half < 0 or cy + half < 0 or cx - half > W or cy - half > H:
        return None            # rect misses the image entirely

    # The rect, as inclusive integer pixel bounds — `crop_with_bbox` reads a bbox
    # that way (its width is x1 - x0 + 1), and the square it then crops is
    # exactly this one: already square, so `crop_large_side` leaves it alone.
    x0i, y0i = int(round(cx - half)), int(round(cy - half))
    src = int(round(2.0 * half)) + 1
    img = fo.rgb.float()
    if img.max() > 1.5:
        img = img / 255.0
    crop, cam_crop_tform_cam = crop_with_bbox(
        img, [x0i, y0i, x0i + src - 1, y0i + src - 1],
        H_out=cell, W_out=cell, scale_bbox=1.0)
    crop = crop.clamp(0, 1)

    # intrinsics for the cropped-and-resized image, straight from the crop, so
    # they stay right whichever of its two orders it worked in
    intr_c = cam_crop_tform_cam @ intr
    return {
        "rgb": crop,
        "intr": intr_c,
        "cam_bbox3d": fo.cam_bbox3d.float(),
        "obj_bbox3d": (fo.obj_bbox3d.float() if fo.obj_bbox3d is not None else None),
        "ncds": fo.cam_tform4x4_obj_ncds.float(),
    }


# ── 2-D overlay ───────────────────────────────────────────────────────────────

def _project(pts_cam: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
    """(N, 3) camera-space points -> (N, 2) pixels, OpenGL convention."""
    from o3b.cv.geometry.transform import proj3d2d_broadcast

    K = intr.clone()
    K[:, 1] = -K[:, 1]          # OpenGL: -Y up, -Z forward
    K[:, 2] = -K[:, 2]
    return proj3d2d_broadcast(pts_cam, K)


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _edge_axis_colors(obj_bbox3d, P: np.ndarray):
    """Colour each box edge by the axis it runs along, relabelled through P.

    The box itself then carries the same red/green/blue meaning as the arrows, so
    a swap visibly recolours it even though the geometry does not move.
    """
    if obj_bbox3d is None:
        return [(0.9, 0.55, 0.1)] * len(_BOX_EDGES)
    b = obj_bbox3d.numpy()
    cols = []
    for i, j in _BOX_EDGES:
        a = int(np.argmax(np.abs(b[j] - b[i])))           # old axis of this edge
        new = int(np.argmax(np.abs(P[:, a])))             # its label after P
        cols.append(_AXES[new][2])
    return cols


def _draw_overlay(frame, P: np.ndarray, arrow_frac: float = 0.55) -> np.ndarray:
    """rgb crop + 3-D box + axis arrows, as a uint8 HxWx3 image."""
    from o3b.cv.visual.draw import draw_lines

    # draw_lines round-trips through cv2 and hands back uint8 0..255 CHW, so feed
    # it uint8 from the start: the float branch of tensor_to_cv_img renormalises
    # by min/max whenever a value strays outside [0, 1].
    img = (frame["rgb"].clamp(0, 1) * 255).to(torch.uint8)
    intr = frame["intr"]

    # ── the current estimated 3-D bounding box, coloured per axis ────────────
    uv = _project(frame["cam_bbox3d"], intr)                       # (8, 2)
    lines = torch.stack([torch.stack([uv[i], uv[j]]) for i, j in _BOX_EDGES])
    cols = _edge_axis_colors(frame["obj_bbox3d"], P)
    colors = torch.tensor([[c, c] for c in cols], dtype=torch.float32)
    img = draw_lines(img=img, lines=lines, colors=colors, thickness=2)

    # ── axis arrows from the object centre ───────────────────────────────────
    M = frame["ncds"]
    R, origin = M[:3, :3], M[:3, 3]                       # NCDS origin = object centre
    extent = float((frame["cam_bbox3d"].max(0).values - frame["cam_bbox3d"].min(0).values).max())
    length = arrow_frac * max(extent, 1e-6)

    # new axis i sits along P^-1 e_i in the object frame the pose still uses
    Pinv = np.linalg.inv(P)
    tips_cam, arrow_cols = [], []
    for i, (_axis, _name, colour) in enumerate(_AXES):
        d = torch.tensor(Pinv[:, i], dtype=torch.float32)
        d = d / (d.norm() + 1e-8)
        d_cam = R @ d
        # An axis pointing towards the camera projects to a point that runs off
        # to infinity, which draws as one arrow swamping the whole cell. Shorten
        # it so the tip stays comfortably in front of the image plane (OpenGL:
        # in front means z < 0) — the direction is still readable, the length
        # just stops being meaningful, which it already was not under a strong
        # perspective foreshortening.
        L = length
        if float(d_cam[2]) > 1e-6:
            L = min(L, 0.9 * max(float(-origin[2]) - 1e-3, 0.0) / float(d_cam[2]))
        tips_cam.append(origin + d_cam * L)
        arrow_cols.append(colour)
    tips_uv = _project(torch.stack(tips_cam), intr)
    o_uv = _project(origin[None], intr)[0]

    # Bound the arrows in *screen* space. An axis pointing towards the camera is
    # magnified by perspective — measured 364 px in a 320 px cell — and swamps
    # the frame while telling you nothing extra; one pointing away shrinks to a
    # few pixels and disappears. Clamping keeps the direction, which is the part
    # being judged, readable in both cases.
    cell = float(img.shape[-1])
    lo, hi = 0.12 * cell, 0.40 * cell
    capped = []
    for tip in tips_uv:
        v = tip - o_uv
        n = float(v.norm())
        if n < 1e-6:
            capped.append(tip)
            continue
        capped.append(o_uv + v / n * min(max(n, lo), hi))
    tips_uv = capped

    segs, seg_cols = [], []
    for tip, colour in zip(tips_uv, arrow_cols):
        segs.append(torch.stack([o_uv, tip]))
        seg_cols.append(colour)
        # arrowhead: two short barbs back along the shaft
        v = tip - o_uv
        n = float(v.norm())
        if n > 6.0:
            u = v / n
            perp = torch.tensor([-u[1], u[0]])
            for s in (+1.0, -1.0):
                barb = tip - u * (0.22 * n) + perp * (0.11 * n) * s
                segs.append(torch.stack([tip, barb]))
                seg_cols.append(colour)
    img = draw_lines(img=img, lines=torch.stack(segs),
                     colors=torch.tensor([[c, c] for c in seg_cols], dtype=torch.float32),
                     thickness=3)
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def block_of(index: int, cols: int = GRID_COLS, views: int = GRID_VIEWS) -> tuple:
    """Where object *index* sits: (row of its first view, column).

    Objects run across, their own views down: ``cols`` objects side by side form
    a block ``views`` rows tall, and the next ``cols`` objects start the block
    below.  The default 8 x (2 blocks of 3) is one screen holding 16 objects
    with three views each — enough of a category to judge it at a glance, which
    a one-object-per-row grid could not fit.
    """
    block, col = divmod(index, cols)
    return block * views, col


def _render_grid(objects, P: np.ndarray, cell: int = 320,
                 cols: int = GRID_COLS, views: int = GRID_VIEWS) -> "Optional[np.ndarray]":
    """One image: ``cols`` objects across, ``views`` of each stacked downwards."""
    if not objects:
        return None
    ncol = min(cols, len(objects))
    nrow = (-(-len(objects) // cols)) * views
    out = np.zeros((nrow * cell, ncol * cell, 3), dtype=np.uint8)
    for i, (_seq, frames) in enumerate(objects):
        row0, col = block_of(i, cols, views)
        for j, frame in enumerate(frames[:views]):
            y, x = (row0 + j) * cell, col * cell
            out[y:y + cell, x:x + cell] = _draw_overlay(frame, P)
    return out


# ── prefetching, shared with the subset editor ────────────────────────────────

class Prefetcher:
    """Warms the crop cache for what the user is about to look at.

    One background thread runs a *plan* — the jobs for the next few pages and
    categories — which every navigation replaces wholesale, so a jump elsewhere
    abandons work that is no longer interesting (the job already running still
    finishes; there is nothing to interrupt inside cv2).  Jobs only write the
    on-disk cache, so a mis-predicted one costs nothing but the decode, and the
    UI never waits on this thread.

    Jobs must not touch tkinter, and must not share a dataset instance with the
    UI: the loaders keep unsynchronised mesh/tform caches, so two threads inside
    the same instance can race a ``clear()`` against a lookup.  Each editor
    hands this its own dataset objects (see ``_pf_category`` there).
    """

    def __init__(self, name: str = "o3b-prefetch") -> None:
        import threading

        self._plan: list = []
        self._done: set = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def plan(self, jobs) -> int:
        """Replace the queue with ``[(key, callable), …]``. Returns what is queued.

        A key already run in this session is dropped, so standing still does not
        re-do the same work on every redraw.
        """
        with self._lock:
            self._plan = [(k, fn) for k, fn in jobs if k not in self._done]
            n = len(self._plan)
        self._wake.set()
        return n

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                with self._lock:
                    if self._stopped or not self._plan:
                        break
                    key, fn = self._plan.pop(0)
                    self._done.add(key)
                try:
                    fn()
                except Exception as exc:                     # never kill the thread
                    logger.debug(f"prefetch {key} failed: {exc}")
            if self._stopped:
                return

    def stop(self, join: float = 0.0) -> None:
        """Stop taking jobs, and optionally wait *join* seconds for the last one.

        Worth waiting for at process exit: a daemon thread killed while it is
        inside cv2's decoder takes the C++ runtime down with it ("terminate
        called without an active exception", core dumped) — after the work is
        safely on disk, but it is an ugly way to end a session.  Waiting costs
        nothing when the queue is idle, which it usually is.
        """
        with self._lock:
            self._stopped = True
            self._plan = []
        self._wake.set()
        if join > 0:
            self._thread.join(join)


# ── window chrome, shared with the subset editor ──────────────────────────────

def maximize(root) -> None:
    """Open filling the screen: the grid alone is cols x 320 px wide."""
    for attempt in (lambda: root.attributes("-zoomed", True),   # X11
                    lambda: root.state("zoomed")):              # Windows / some WMs
        try:
            attempt()
            return
        except Exception:
            continue
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")


def combo_focused(root, combo) -> bool:
    """True while the category combobox (or its dropdown) holds the focus.

    Not ``root.focus_get() is combo``: when the dropdown is open the focus sits
    on ttk's internal ``.popdown`` toplevel, which tkinter never registered as a
    child, so ``focus_get`` raises ``KeyError: 'popdown'``.  Comparing the widget
    *paths* answers the same question without asking tkinter to resolve one.
    """
    try:
        focus = root.tk.call("focus")
    except Exception:
        return False
    if not focus:
        return False
    path, combo_path = str(focus), str(combo)
    return path == combo_path or path.startswith(combo_path + ".")


def fit(img: np.ndarray, max_w: int, max_h: int) -> tuple:
    """Downscale *img* to fit the box (never up). Returns (image, scale).

    The scale comes back because clicks land in the scaled image and have to be
    mapped to a cell in the unscaled one.
    """
    import cv2

    scale = min(1.0, max_w / img.shape[1], max_h / img.shape[0])
    if scale >= 1.0:
        return img, 1.0
    size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


# ── saving ────────────────────────────────────────────────────────────────────

def stored_tform(P: np.ndarray, G: "Optional[np.ndarray]", T: torch.Tensor) -> torch.Tensor:
    """The transform to write: inv(G) @ P @ G @ T, with P acting in display space."""
    P4 = torch.eye(4, dtype=torch.float32)
    P4[:3, :3] = torch.tensor(P, dtype=torch.float32)
    if G is None:
        return P4 @ T.float()
    from o3b.cv.geometry.transform import inv_tform4x4
    G4 = torch.tensor(G, dtype=torch.float32)
    return inv_tform4x4(G4) @ P4 @ G4 @ T.float()


def save_category(cfg, category: str, P: np.ndarray, G,
                  objects: "Optional[set]" = None) -> tuple[int, str]:
    """Apply P to every sequence of *category*. Returns (n_written, destination).

    *objects* narrows that to a set of object ids (``<category>/<sequence>``) —
    what ``select-and-axes-tform-obj-type`` writes, where the rotation is the
    verdict on a hand-picked few rather than on the whole category. ``None``
    keeps the category-wide behaviour the axes editor wants.
    """
    from o3b.dataset import tform_obj_store as store

    pp, tot = cfg.path_preprocess, cfg.tform_obj_type
    table = store.load_all(pp, tot)

    def _wanted(object_id: str) -> bool:
        return (object_id.split("/", 1)[0] == category
                and (objects is None or object_id in objects))

    if table is not None:                      # SQLite: one transaction
        rows = [(category, k.split("/", 1)[1], stored_tform(P, G, T))
                for k, T in table.items() if _wanted(k)]
        if not rows:
            return 0, str(store.db_path(pp, tot))
        store.write_many(pp, tot, rows)
        return len(rows), str(store.db_path(pp, tot))

    # legacy per-sequence files
    root = store.dir_path(pp, tot) / "meta_mask" / "meta" / category
    if not root.is_dir():
        return 0, str(root)
    n = 0
    for seq_dir in sorted(root.iterdir()):
        f = seq_dir / "tform_obj.pt"
        if not f.exists() or not _wanted(f"{category}/{seq_dir.name}"):
            continue
        try:
            T = torch.load(f, map_location="cpu", weights_only=True).float()
        except Exception as exc:
            print(f"  skipping {f}: {exc}", file=sys.stderr)
            continue
        torch.save(stored_tform(P, G, T), f)
        n += 1
    return n, str(root)


def _categories(cfg) -> list[str]:
    """Categories that actually have transforms, cheapest source first."""
    from o3b.dataset import tform_obj_store as store

    table = store.load_all(cfg.path_preprocess, cfg.tform_obj_type)
    if table is not None:
        return sorted({k.split("/", 1)[0] for k in table})
    root = store.dir_path(cfg.path_preprocess, cfg.tform_obj_type) / "meta_mask" / "meta"
    if root.is_dir():
        return sorted(p.name for p in root.iterdir() if p.is_dir())
    return list(cfg.categories or [])


# ── HUD ───────────────────────────────────────────────────────────────────────

def _hud(lines, width: int, height: int) -> np.ndarray:
    """Left-hand text panel, drawn straight into a BGR array."""
    import cv2

    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    y = 26
    for text, colour, scale in lines:
        if text == "---":
            cv2.line(panel, (10, y - 8), (width - 10, y - 8), (70, 70, 70), 1)
            y += 12
            continue
        cv2.putText(panel, _ascii(text)[: int(width / (7.2 * scale))], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, colour, 1, cv2.LINE_AA)
        y += int(20 * scale)
        if y > height - 8:
            break
    return panel


def _ascii(s: str) -> str:
    """cv2's Hershey fonts are ASCII-only — anything else draws as '???'."""
    return (s.replace("\u2014", "-").replace("\u2013", "-").replace("\u00b7", "|")
             .replace("\u2192", "->").replace("\u2194", "<->")
             .encode("ascii", "replace").decode("ascii"))


def _plain_axis_lines(category: str, M: "Optional[np.ndarray]" = None):
    """axis_description() without the markdown, for the HUD panel.

    Only *wrapping* underscores are markdown emphasis here — the rule names
    themselves contain them (``user_backward``), so a blanket strip would turn
    those into ``userbackward``.
    """
    import re

    out = []
    for line in axis_description(category, M).splitlines():
        s = line.replace("**", "").replace("`", "").strip()
        # Drop markdown italics only where an underscore borders whitespace, so
        # `vec_front_back` and `+Z_uco3d` survive intact.
        s = re.sub(r"(?<!\w)_(?=\S)", "", s)
        s = re.sub(r"(?<=\S)_(?!\w)", "", s)
        if s.strip():
            out.append(_ascii(s.strip()))
    return out


def axis_rules_lines(cfg, category: str) -> list:
    """The category's UCO3D axis rules as ``_hud`` lines, for either editor.

    Both editors draw the object in the frame the config's
    ``obj_gl_tform4x4_obj_raw`` defines, so both quote the rules re-indexed
    through it and through nothing else — see ``axis_description``.  Sharing
    this is what keeps the two panels saying the same thing about a category,
    and what keeps the axes editor's panel constant while its arrows turn.
    """
    import textwrap

    G = getattr(cfg, "obj_gl_tform4x4_obj_raw", None)
    M = np.array(G, dtype=np.float64)[:3, :3] if G is not None else None
    out = []
    for line in _plain_axis_lines(category, M):
        # 58 characters is what 380 px of HUD holds at scale 0.9, and the tree
        # has rules that overrun it by twenty
        # (`along_stand_grab_one_hand_grab_away_from_function`). _hud truncates,
        # so wrap here instead — a rule cut off mid-word names a different rule.
        for i, part in enumerate(textwrap.wrap(line, 56) or [""]):
            out.append((("    " + part) if i else part, (190, 190, 190), 0.9))
    return out


# ── viewer ────────────────────────────────────────────────────────────────────

def run_axes_editor(cls, cfg, *, reference=None, ref_cls=None,
                    n_objects: int = GRID_OBJECTS, n_views: int = GRID_VIEWS,
                    cols: int = GRID_COLS, prefetch: int = PREFETCH,
                    category: Optional[str] = None) -> None:
    """One window: a category dropdown over a grid of overlaid frames.

    tkinter for the shell (it has a real combobox, and 966 categories want a
    scrollable, type-to-filter list rather than a slider), cv2 only for drawing
    into the numpy frame, PIL to blit it — 5.7 ms against the 70 ms render, so
    the display is never the bottleneck.

    The window opens maximised and the composed image is scaled down to fit it:
    16 objects x 3 views at 320 px is 2560 x 1920, which no screen shows 1:1.
    """
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    cats = _categories(cfg)
    if not cats:
        print("No categories with tform_obj found — is the dataset mounted?", file=sys.stderr)
        sys.exit(1)
    idx = cats.index(category) if category in cats else 0

    G = np.array(cfg.obj_gl_tform4x4_obj_raw, dtype=np.float64) \
        if cfg.obj_gl_tform4x4_obj_raw is not None else None

    pending = Pending()
    st = {"objects": [], "ref": [], "status": "", "idx": idx}
    prefetcher = Prefetcher() if prefetch > 0 else None

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"cannot open a window ({exc}).\nThis tool needs a display — run it on "
              f"your workstation, not over a headless ssh session.", file=sys.stderr)
        sys.exit(1)
    root.title("o3b  axes-tform-obj-type")
    maximize(root)
    # The HUD takes 380 px on the left; the rest of the screen is the grid's,
    # minus the toolbar and the window manager's own chrome.
    max_w = max(320, root.winfo_screenwidth() - 400)
    max_h = max(320, root.winfo_screenheight() - 140)

    bar = ttk.Frame(root)
    bar.pack(fill=tk.X, padx=6, pady=4)
    ttk.Label(bar, text="Category").pack(side=tk.LEFT, padx=(2, 6))
    cat_var = tk.StringVar(value=cats[idx])
    combo = ttk.Combobox(bar, textvariable=cat_var, values=cats, width=34)
    combo.pack(side=tk.LEFT)
    ttk.Button(bar, text="< prev", width=8,
               command=lambda: _step(-1)).pack(side=tk.LEFT, padx=(8, 2))
    ttk.Button(bar, text="next >", width=8,
               command=lambda: _step(+1)).pack(side=tk.LEFT, padx=2)
    save_btn = ttk.Button(bar, text="SAVE category", command=lambda: _save())
    save_btn.pack(side=tk.LEFT, padx=(16, 2))
    ttk.Button(bar, text="reload", width=8,
               command=lambda: _reload()).pack(side=tk.LEFT, padx=2)

    canvas = ttk.Label(root)
    canvas.pack()
    canvas.focus_set()
    _photo = {"ref": None}          # keep a reference or tk garbage-collects it

    def _hud_lines():
        cat = cats[st["idx"]]
        ok = pending.is_rotation
        lines = [(cat, (255, 255, 255), 1.25),
                 (f"[{st['idx'] + 1}/{len(cats)}]", (170, 170, 170), 0.9),
                 ("---", None, 1)]
        # The rules re-indexed through G alone. They are the *target* — "the red
        # arrow has to be the axis this rule describes" — so they stay put while
        # the swaps and flips turn the arrows towards them; re-indexing them
        # through the pending rotation as well would relabel the goal on every
        # keystroke and there would be nothing left to aim at.
        lines += axis_rules_lines(cfg, cat)
        lines += [("---", None, 1),
                  ("X right (red)  Y top (green)  Z back (blue)", (200, 200, 200), 0.9),
                  ("---", None, 1),
                  ("P (rows = new X, Y, Z):", (170, 170, 200), 0.85)]
        lines += [("   [" + ", ".join(f"{v:5.1f}" for v in row) + "]",
                   (200, 200, 255), 0.85) for row in pending.P]
        lines += [(f"det(P) = {pending.det:+.3f}"
                   + ("  rotation OK" if ok else "  REFLECTION - cannot save"),
                   (120, 230, 120) if ok else (110, 110, 245), 1.0),
                  ("---", None, 1),
                  ("1/2/3 swap XY YZ ZX    4/5/6 flip X Y Z", (180, 180, 180), 0.85),
                  ("u undo   r reset   s SAVE category", (180, 180, 180), 0.85),
                  ("n/p next/prev category   q quit", (180, 180, 180), 0.85),
                  ("dropdown above: type to filter, enter to pick", (180, 180, 180), 0.85),
                  ("---", None, 1),
                  (st["status"], (200, 200, 120), 0.85)]
        if st["ref"]:
            lines += [(f"reference below: {len(st['ref'])} objects", (160, 200, 200), 0.85)]
        return lines

    def _show():
        grid = _render_grid(st["objects"], pending.P, cols=cols, views=n_views)
        ref = (_render_grid(st["ref"], np.eye(3), cols=cols, views=n_views)
               if st["ref"] else None)
        if grid is None:
            grid = np.zeros((320, 320, 3), np.uint8)
        if ref is not None:
            w = max(grid.shape[1], ref.shape[1])
            grid = np.pad(grid, ((0, 0), (0, w - grid.shape[1]), (0, 0)))
            ref = np.pad(ref, ((0, 0), (0, w - ref.shape[1]), (0, 0)))
            grid = np.vstack([grid, np.full((6, w, 3), 60, np.uint8), ref])
        # Scale the grid, not the HUD: shrinking the text along with a
        # 2560 px-wide page of thumbnails would make it unreadable.
        grid, _scale = fit(grid, max_w, max_h)
        h = max(grid.shape[0], 560)
        if grid.shape[0] < h:
            grid = np.pad(grid, ((0, h - grid.shape[0]), (0, 0), (0, 0)))
        frame = np.hstack([_hud(_hud_lines(), 380, h), grid])
        _photo["ref"] = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.configure(image=_photo["ref"])
        save_btn.state(["!disabled"] if (pending.is_rotation and not pending.is_identity)
                       else ["disabled"])

    def _prefetch():
        """Warm the categories the n/p keys are about to reach.

        Each job re-enters ``_load_category``, which builds its own dataset — so
        nothing is shared with the instance the UI just used — and writes the
        crops the next `_load` will find already there.
        """
        if prefetcher is None:
            return
        jobs = []
        for d in range(1, prefetch + 1):
            cat = cats[(st["idx"] + d) % len(cats)]
            jobs.append((f"{cfg.tform_obj_type}|{cat}",
                         partial(_load_category, cls, cfg, cat, n_objects, n_views)))
            if reference is not None:
                jobs.append((f"ref|{reference.tform_obj_type}|{cat}",
                             partial(_load_category, ref_cls, reference, cat,
                                     n_objects, n_views)))
        prefetcher.plan(jobs)

    def _load(i: int):
        st["idx"] = i
        cat = cats[i]
        cat_var.set(cat)
        st["status"] = f"loading {cat} …"
        _show()
        root.update()
        t0 = time.time()
        st["objects"] = _load_category(cls, cfg, cat, n_objects, n_views)
        st["ref"] = (_load_category(ref_cls, reference, cat, n_objects, n_views)
                     if reference is not None else [])
        st["status"] = (f"{cat}: {len(st['objects'])} objects in {time.time() - t0:.2f}s"
                        if st["objects"] else f"{cat}: nothing loadable")
        _show()
        _prefetch()

    def _step(d: int):
        pending.reset(); pending._history.clear()
        _load((st["idx"] + d) % len(cats))
        canvas.focus_set()

    def _quit():
        if prefetcher is not None:
            prefetcher.stop()      # a job mid-decode still finishes; it is a daemon
        root.destroy()

    def _reload():
        _cache_drop(cfg, cats[st["idx"]])
        _load(st["idx"])
        canvas.focus_set()

    def _save():
        if pending.is_identity:
            st["status"] = "nothing to save"
        elif not pending.is_rotation:
            st["status"] = f"REFUSED: det(P) = {pending.det:+.3f}, not a rotation"
            print(st["status"], file=sys.stderr)
        else:
            n, where = save_category(cfg, cats[st["idx"]], pending.P, G)
            print(f"  {cats[st['idx']]}: wrote {n} transforms -> {where}")
            pending.reset(); pending._history.clear()
            _cache_drop(cfg, cats[st["idx"]])   # geometry moved; crops are stale
            _load(st["idx"])
            st["status"] = f"wrote {n} transforms"
        _show()

    # ── dropdown ──────────────────────────────────────────────────────────────
    def _pick(_e=None):
        want = cat_var.get().strip()
        if want in cats:
            pending.reset(); pending._history.clear()
            _load(cats.index(want))
        canvas.focus_set()

    def _rank(text: str) -> list:
        """Categories matching *text*, best first.

        Ranked rather than raw substring order: "chair" has to offer `chair`
        before `armchair`, or enter picks the wrong one.
        """
        text = text.strip().lower()
        if not text:
            return cats
        exact  = [c for c in cats if c.lower() == text]
        prefix = [c for c in cats if c.lower().startswith(text) and c not in exact]
        inside = [c for c in cats if text in c.lower()
                  and c not in exact and c not in prefix]
        return exact + prefix + inside

    def _filter(e):
        """Type in the box to narrow the dropdown; enter picks the best hit."""
        if e.keysym in ("Return", "KP_Enter"):
            hits = _rank(cat_var.get())
            if hits:
                cat_var.set(hits[0])
                _pick()
            return
        if e.keysym in ("Up", "Down", "Escape"):
            return
        combo["values"] = _rank(cat_var.get()) or cats

    combo.bind("<<ComboboxSelected>>", _pick)
    combo.bind("<KeyRelease>", _filter)

    # ── keys (ignored while the dropdown has focus, so typing filters) ────────
    _KEYS = {
        "1": lambda: pending.swap(0, 1), "2": lambda: pending.swap(1, 2),
        "3": lambda: pending.swap(2, 0), "4": lambda: pending.flip(0),
        "5": lambda: pending.flip(1),    "6": lambda: pending.flip(2),
        "u": pending.undo,               "r": pending.reset,
    }

    def _on_key(e):
        if combo_focused(root, combo):
            return
        k = e.keysym.lower()
        if k == "q" or k == "escape":
            _quit()
        elif k in ("n", "period", "right"):
            _step(+1)
        elif k in ("p", "comma", "left"):
            _step(-1)
        elif k == "s":
            _save()
        elif k == "c":
            _reload()
        elif e.char in _KEYS:
            _KEYS[e.char]()
            _show()

    root.bind_all("<Key>", _on_key)
    root.protocol("WM_DELETE_WINDOW", _quit)

    _load(idx)
    print("\nKeys: 1/2/3 swap · 4/5/6 flip · u undo · r reset · s save · "
          "n/p category · q quit   (dropdown: type to filter)\n")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    if prefetcher is not None:
        # the window is gone by now, so waiting out a decode costs the user nothing
        prefetcher.stop(join=30)
    print("Stopping.")
