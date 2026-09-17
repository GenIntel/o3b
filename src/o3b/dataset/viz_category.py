"""One category as a contact sheet: N objects down, F frames across.

    o3b dataset viz-cat -d every9d_v7_test -c chair -n 8 -f 8

Every row is one object and runs left to right in time — the F frames the
sequence contributes, evenly spaced over it, in timestamp order.  That is the
one arrangement the grid editors cannot give: they stack an object's views
*downwards* so that sixteen objects fit on a screen (``o3b/dataset/grid.py``),
which is the right shape for judging a category at a glance and the wrong one
for reading a single object's pose as the camera moves around it.  Here a row is
a track, and a canonical frame that drifts, flips at a symmetry, or was labelled
off a bad reconstruction shows up as the triad turning along the row while the
object does not.

The cells are the rgb and nothing else: cropped around the projected 3-D box and
cropped *tight* — ``SHEET_MARGIN`` against the editors' 0.45, so the object
fills its cell instead of sitting in half a cell of scene — on a white ground,
with no box, no axes and no names written over them.  This is a figure, not a
labelling tool, and anything drawn into it has to be argued for: ``--overlay``
adds the editors' 3-D box and triad (red = right/X, green = top/Y, blue =
back/Z) and ``--labels`` writes each row's sequence name into it, both off by
default.  ``--margin`` loosens the crop back towards the editors'.

White rather than black in both of the places padding shows — a row that decoded
fewer frames than it was asked for, and a crop that ran off the edge of its
image, which at a tight margin is most of them.  In a figure those should read
as nothing there, not as a black tile or a black border.  The crop's fill goes
down into ``crop_with_bbox`` rather than being painted over the cell afterwards:
the resize blends the padding with the pixels beside it, so a cell patched after
the fact keeps a black seam along the edge.

Unlike the editors this *keeps* the config's ``subset_name``, because which
objects to show is exactly what a config like ``every9d_v7_test`` is for: the
sheet shows the test split's own objects, not the category's first N.  For the
same reason it does not share their crop cache entries — a frame-granularity
subset (``every9d_v6_test``) thins the frame pool before the even spacing picks
from it, so the same object and the same F can mean different frames here than
in a page the editors cached.  Its own tag, in the same directory, so an axes
save still drops these sheets along with everything else for the category.

Objects come in walk order by default, which is stable and stops the walk at N.
``--seed`` samples instead: the category is walked to ``--pool`` objects and N of
them are drawn at random, so two sheets of the same category can show different
objects.  Over an sshfs mount without ``frames.db`` that walk is a round trip per
sequence, so keep ``--pool`` finite there.
"""
from __future__ import annotations

import sys
from dataclasses import replace as _r
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from o3b.dataset.axes_tform_obj import (
    _ascii, _cache_load, _cache_store, _categories, _draw_overlay,
    collect_frames, view_modalities,
)
from o3b.dataset.grid import SHEET_GAP, SHEET_MARGIN

_CELL = 320                       # what _crop_frame renders; --cell scales it down

_GROUND = 255                     # what shows through where a frame is missing
_PAD    = 1.0                     # ... and what fills a crop that left its image
_LABEL  = (235, 235, 235)


# ── the category's objects ────────────────────────────────────────────────────

def _sheet_cfg(cfg, category: str, n_objects: Optional[int], n_frames: int):
    """The config a sheet is walked with.

    ``subset_name`` survives, unlike in the editors: the sheet is a view of the
    split the config names.  ``sharded_name`` and ``transform`` do not — the
    crops are cut from the raw rgb at full resolution, and a materialised cache
    would answer with whatever it was built for.
    """
    extra = dict(cfg.extra or {})
    extra["frames_count_max_per_sequence"] = n_frames
    extra["sequences_count_max_per_category"] = n_objects
    extra.pop("subset_ids", None)
    return _r(
        cfg,
        categories=[category],
        filter_count_max=None,
        sharded_name=None,
        transform=None,
        modalities=view_modalities(),
        extra=extra,
    )


def _walk_category(cls, cfg, category: str, n_objects: Optional[int], n_frames: int):
    """(dataset, [object_id, …], {object_id: [item index, …]}) for *category*.

    The walk only — no pixels.  ``n_objects`` is where it stops, so the usual
    call costs a listing of the category's first N sequences rather than of all
    of them.
    """
    try:
        dataset = cls(_sheet_cfg(cfg, category, n_objects, n_frames))
    except Exception as exc:
        print(f"  could not build dataset for {category}: {exc}", file=sys.stderr)
        return None, [], {}

    order: list[str] = []
    by_obj: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        row = dataset._frame_rows[dataset._frame_rows_id[i]]
        oid = row["object_id"]
        if oid not in by_obj:
            by_obj[oid] = []
            order.append(oid)
        by_obj[oid].append(i)
    return dataset, order, by_obj


def _sheet_tag(cfg, oids: list[str], n_frames: int, margin: float) -> str:
    """The crop cache's key for a sheet.

    The subset is part of it: it thins an object's frame pool before the even
    spacing picks from it, so two configs over the same objects and the same F
    do not in general mean the same frames.  So are the margin and the pad,
    which are baked into the cached pixels — naming them here is what retires an
    entry cached under an earlier framing instead of drawing it again.
    ``_cache_key`` hashes all of it, so its length does not matter.
    """
    return (f"sheet|{cfg.subset_name or '-'}|{n_frames}|{margin:g}|{_PAD:g}|"
            + "+".join(oids))


def load_sheet(dataset, cfg, category: str, oids: list[str], by_obj: dict,
               n_frames: int, margin: float = SHEET_MARGIN, cache: bool = True):
    """``[(object_id, [frame, …]), …]`` for the chosen objects, cached on disk.

    Frames come back in timestamp order (``collect_frames`` sorts a sequence's
    frames as it decodes them), which is what makes a row a track.
    """
    tag = _sheet_tag(cfg, oids, n_frames, margin)
    if cache:
        hit = _cache_load(cfg, category, tag)
        if hit is not None:
            return hit

    indices = [i for oid in oids for i in by_obj.get(oid, [])]
    objects = collect_frames(dataset, cfg, indices, margin=margin, pad=_PAD)
    if cache and objects:
        _cache_store(cfg, category, tag, objects)
    return objects


# ── the sheet ─────────────────────────────────────────────────────────────────

def render_sheet(objects, *, overlay: bool = False, labels: bool = False,
                 cell: int = _CELL, gap: int = SHEET_GAP) -> Optional[np.ndarray]:
    """One image: an object per row, its frames left to right in time.

    Nothing is drawn over the cells unless asked for, and the only line between
    them is *gap* pixels of the white ground — down and across alike, so every
    frame reads as its own image.  ``--gap 0`` closes it up.

    Rows are padded to the widest one rather than to the requested F: a frame
    whose 3-D box projects behind the pinhole is dropped by ``_crop_frame``, so
    a sequence can contribute fewer than it was asked for, and the row then ends
    early instead of shifting a neighbour's column out of line.
    """
    import cv2

    if not objects:
        return None
    ncol = max(len(frames) for _oid, frames in objects)
    if ncol == 0:
        return None
    gap = max(0, gap)
    nrow, step = len(objects), cell + gap
    out = np.full((nrow * step - gap, ncol * step - gap, 3), _GROUND,
                  dtype=np.uint8)
    P = np.eye(3)                 # nothing pending here — the frame as it is stored
    for i, (oid, frames) in enumerate(objects):
        y0 = i * step
        for j, frame in enumerate(frames):
            img = _draw_overlay(frame, P) if overlay else _rgb(frame)
            if img.shape[0] != cell:
                img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
            out[y0:y0 + cell, j * step:j * step + cell] = img
        if labels:
            # the sequence alone: every row of the sheet is the same category
            scale = 0.9 * cell / _CELL
            for colour, thick in (((0, 0, 0), 5), (_LABEL, 2)):
                cv2.putText(out, _ascii(oid.split("/", 1)[-1]),
                            (int(10 * scale), y0 + int(32 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, colour,
                            max(1, int(thick * scale)), cv2.LINE_AA)
    return out


def _rgb(frame) -> np.ndarray:
    """The crop with nothing drawn on it, as a uint8 HxWx3 image."""
    img = (frame["rgb"].clamp(0, 1) * 255).to(torch.uint8)
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


# ── output ────────────────────────────────────────────────────────────────────

def _out_path(out: Path, dataset_name: str, category: str, many: bool) -> Path:
    """Where one category's sheet is written.

    A directory (or an extension-less path) collects them under generated names;
    a file name is used as given, with the category folded in when the run
    covers more than one so the sheets cannot overwrite each other.
    """
    if out.is_dir() or not out.suffix:
        return out / f"{dataset_name or 'sheet'}_{category}.png"
    if many:
        return out.with_name(f"{out.stem}_{category}{out.suffix}")
    return out


def _save(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"  wrote {path}  ({img.shape[1]}x{img.shape[0]})")


def _show(img: np.ndarray, title: str) -> bool:
    """Open the sheet in a matplotlib window. False when there is no display."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  cannot show ({exc}) — pass -o to write a file instead",
              file=sys.stderr)
        return False
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(min(16.0, w / 160), min(16.0, w / 160) * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    return True


# ── entry point ───────────────────────────────────────────────────────────────

def run_category_sheet(cls, cfg, *, dataset_name: str = "",
                       categories: Optional[list] = None,
                       n_objects: int = 8, n_frames: int = 8,
                       overlay: bool = False, labels: bool = False,
                       margin: float = SHEET_MARGIN,
                       cell: int = _CELL, gap: int = SHEET_GAP,
                       seed: Optional[int] = None,
                       pool: Optional[int] = None,
                       out: Optional[Path] = None, show: bool = True,
                       cache: bool = True) -> None:
    import random

    cats = [c for c in (categories or cfg.categories or []) if c]
    if not cats:
        known = _categories(cfg)
        print("viz-cat needs a category: pass -c <name>.\n"
              f"  {len(known)} available"
              + (f", e.g. {', '.join(known[:6])}" if known else ""), file=sys.stderr)
        sys.exit(1)

    shown = 0
    for category in cats:
        print(f"{category}: {n_objects} objects x {n_frames} frames")
        # Without a seed the walk stops at the objects the sheet shows; with one
        # it has to see the pool the sample is drawn from.
        walk_max = n_objects if seed is None else pool
        dataset, order, by_obj = _walk_category(cls, cfg, category, walk_max, n_frames)
        if not order:
            print(f"  no objects for {category}", file=sys.stderr)
            continue
        if seed is None:
            oids = order[:n_objects]
        else:
            rng = random.Random(seed)
            oids = sorted(rng.sample(order, min(n_objects, len(order))),
                          key=order.index)
        print(f"  {len(oids)}/{len(order)} objects: {', '.join(o.split('/', 1)[-1] for o in oids)}")

        objects = load_sheet(dataset, cfg, category, oids, by_obj, n_frames,
                             margin=margin, cache=cache)
        img = render_sheet(objects, overlay=overlay, labels=labels, cell=cell,
                           gap=gap)
        if img is None:
            print(f"  nothing decoded for {category}", file=sys.stderr)
            continue
        if out is not None:
            _save(img, _out_path(Path(out), dataset_name, category, len(cats) > 1))
        if show:
            shown += _show(img, f"o3b  viz-cat  {dataset_name} {category}")

    if shown:
        import matplotlib.pyplot as plt
        plt.show()
