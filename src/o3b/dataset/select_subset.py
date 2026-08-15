"""Pick the objects that make up a named subset.

    o3b dataset select-subset -d every9d_v3 -s every9d_v3_test

Writes ``<path_preprocess>/subset/<name>.yaml`` — the list of object ids a config
with ``subset_name: <name>`` then restricts itself to (see
``o3b/dataset/subset_store.py``).  For UCO3D an object id is
``<category>/<sequence>``, so the file is "these sequences and no others"; the
usual reason to build one is carving a hand-checked test split out of a
labelling round.

The page is the axes editor's: 8 objects across, 3 views of each stacked below
it, two such blocks — a 6 x 8 grid holding 16 objects, scaled to fill the screen.
So is the overlay: every cell carries the 3-D box, its edges coloured by the
axis they run along, and the triad from the object centre — red = right (X),
green = top (Y), blue = back (Z).  What changes is the verdict.  There, a
category got one rotation; here, **an object is in the subset or it is not** — a
whole column goes in, never one view of it, because a sequence contributes all
of its frames or none.  Click a column (or Space on the cursor) to toggle it.

So is the panel beside it: the category's rules from UCO3D's orientation tree,
re-indexed into the frame the cells are drawn in — what X, Y and Z are supposed
to mean for a `chair` — because that is what the triad in each cell has to be
checked against before a sequence goes into a hand-picked split.  It is the
axes editor's panel unchanged (``axes_tform_obj.axis_rules_lines``), which is
also why that one no longer folds its pending rotation in: in both tools these
rules are the fixed statement the arrows are judged against.

A category opens with whatever is already in the subset on page 1, so resuming
shows the standing selection rather than making you page for it.

A category is walked in pages of ``--objects``.  Only the current page's pixels
are decoded, so paging deep into a 1000-sequence category costs the same as
opening its first page, and pages already visited come back from the same crop
cache the axes editor fills, ``~/.o3b/cache/axes`` — same entries, same
``tform_obj_type`` in the key, so an axis edit drops these pages too, as it must
when it has just recoloured their boxes.  A background thread warms the first
page of the next ``--prefetch`` categories while you work on this one, so
stepping to the next category costs 0.06 s where decoding it costs 8 s.

The selection is saved on every navigation and on quit, not only on ``s``: the
work is minutes of clicking that nothing else reproduces.

``--target-count N`` turns the run into a quota — N objects per category.  The
categories already at N are counted, reported and then *left out of the list*,
so what you page through is only what is still short of it; the HUD carries both
the overall tally and this category's own ``n/N``.  The filtering happens once,
at start: a category that reaches N while you work stays where it is (a list
that reshuffled under the click would move the page you are on), and ``n``/``p``
skip it from then on.  Naming one with ``--category`` still opens it, finished
or not.

``--init-count N`` is the other way round: it offers exactly the categories that
*start* with N objects selected, for reviewing a slice of the standing selection
rather than filling one up — ``--init-count 5`` walks the categories a
``--target-count 5`` run finished, ``--init-count 0`` the ones it never reached.
Read once, at start, from the same startup counts, so toggling an object does
not drop the category you are working on out from under you.  The two compose
(``--init-count 3 --target-count 5`` opens the categories that have 3 and want
5) and ``--category`` still overrides both.

``--max-count N`` caps how many objects a category *offers*, and is what makes
opening a big one bearable.  The walk is a directory listing plus a stat per
sequence — ~0.25 s each over sshfs — so a 2000-sequence category costs minutes
while its first 50 cost seconds, and a run that wants 5 of them out of the
category was never going to look at the other 1950.  Everything already selected
is offered whatever the cap says (a selection larger than N keeps all of it):
what is in the subset must stay editable, so with a selection in the category the
cap is filled *around* it, walking the head of the category and then adding the
selected ids wherever they sit.  Without one the cap is simply where the walk
breaks, and costs one dataset build rather than two.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import replace as _r
from functools import partial
from typing import Optional

import numpy as np

from o3b.dataset import subset_store
from o3b.dataset.axes_tform_obj import (
    Prefetcher, axis_rules_lines,
    _ascii, _cache_load, _cache_store, _categories, _draw_overlay, _hud,
    block_of, collect_frames, combo_focused, fit, maximize, view_modalities,
)
from o3b.dataset.grid import GRID_COLS, GRID_OBJECTS, GRID_VIEWS, PREFETCH

_CELL = 320                       # what _crop_frame renders; display is scaled

_SELECTED   = (90, 220, 90)       # drawn with cv2 into an array PIL shows as RGB
_UNSELECTED = (70, 70, 70)
_CURSOR     = (60, 210, 240)


def _object_id(entry: str) -> str:
    """A subset entry as the object it belongs to.

    This tool selects whole objects, so an existing file written at frame
    granularity — ``<category>/<sequence>/<frame>``, which is how od3d writes its
    own subsets — is read as the sequences it touched.  Saving then writes those
    sequences whole, which is the point: a row is in or out, never partly in.
    """
    parts = entry.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else entry


# ── one category's objects ────────────────────────────────────────────────────

def _view_cfg(cfg, category: str, n_views: int, *,
              seqs_max: Optional[int] = None, subset_ids=None):
    """The config the editor walks a category with.

    Without the config's own ``subset_name``: the point of this tool is to decide
    what belongs in the subset, so it has to see the objects that are not in it
    yet.  And without its ``sharded_name``: the editor wants the object as it is
    on disk right now, not a materialised (and possibly stale, and possibly not
    yet built) cache of it.  ``subset_ids`` is the *editor's* selection of what
    to walk, which is a different question and goes through ``extra``.
    """
    extra = dict(cfg.extra or {})
    extra["frames_count_max_per_sequence"] = n_views
    extra["sequences_count_max_per_category"] = seqs_max
    if subset_ids is None:
        extra.pop("subset_ids", None)
    else:
        extra["subset_ids"] = list(subset_ids)
    return _r(
        cfg,
        categories=[category],
        filter_count_max=None,
        subset_name=None,
        sharded_name=None,
        transform=None,
        modalities=view_modalities(),
        extra=extra,
    )


def _walk_object_ids(dataset) -> list[str]:
    """The dataset's object ids, in walk order, each once."""
    order: list[str] = []
    seen: set[str] = set()
    for i in range(len(dataset)):
        oid = dataset._frame_rows[dataset._frame_rows_id[i]]["object_id"]
        if oid not in seen:
            seen.add(oid)
            order.append(oid)
    return order


def _capped_ids(cls, cfg, category: str, max_objects: int,
                keep_here: list[str]) -> Optional[list[str]]:
    """Which objects a capped category shows: *keep_here* plus the walk's first.

    Only needed when something in this category is already selected — the
    selection has to survive the cap wherever in the category it sits, and the
    cap alone would cut it off.  So walk the head of the category (``seqs_max``
    breaks the loop, which is where the saving is), then fill up to
    *max_objects* around the ids that must be there.

    A selection larger than the cap keeps all of it: hiding something already in
    the subset would make it uneditable, which is worse than a long page.  The
    walk this returns ids for is cheap by then — the head is warm in the page
    cache, and only the selected ids beyond it cost a round trip.
    """
    try:
        probe = cls(_view_cfg(cfg, category, 1, seqs_max=max_objects))
    except Exception as exc:
        print(f"  could not walk {category}: {exc}", file=sys.stderr)
        return None
    chosen = list(keep_here)
    seen = set(chosen)
    for oid in _walk_object_ids(probe):
        if len(chosen) >= max_objects:
            break
        if oid not in seen:
            seen.add(oid)
            chosen.append(oid)
    return chosen


def open_category(cls, cfg, category: str, n_views: int, *,
                  max_objects: Optional[int] = None, keep=()):
    """(dataset, [object_id, …], {object_id: [item index, …]}) for a category.

    The walk only — no pixels.  One dataset serves every page of the category,
    and ``load_page`` loads just the rows a page shows.

    *max_objects* caps how many objects the category offers, which is what makes
    opening a big one bearable: the walk is a directory listing plus a stat per
    sequence, ~0.25 s each over sshfs, so a 2000-sequence category costs minutes
    while its first 50 cost seconds.  *keep* is the ids that must be offered
    whatever the cap says — the editor passes its current selection, so nothing
    already chosen can fall off the end of a category and become uneditable.
    """
    keep_here = sorted({o for o in keep if o.split("/", 1)[0] == category})
    seqs_max = ids = None
    if max_objects is not None:
        if keep_here:
            ids = _capped_ids(cls, cfg, category, max_objects, keep_here)
            if ids is None:
                return None, [], {}
        else:
            # nothing to protect, so the cap is just the walk's own break
            seqs_max = max_objects
    try:
        dataset = cls(_view_cfg(cfg, category, n_views,
                                seqs_max=seqs_max, subset_ids=ids))
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


def page_tag(oids: list[str], n_views: int) -> str:
    """The crop cache's key for a page — its objects, not its number."""
    return "page|" + str(n_views) + "|" + "+".join(oids)


def load_page(dataset, cfg, category: str, oids: list[str], by_obj: dict,
              n_views: int, cache: bool = True):
    """``[(object_id, [frame, …]), …]`` for one page, cached on disk.

    Keyed by the page's object ids rather than its number, so inserting or
    removing a sequence upstream re-uses the crops of the objects that stayed
    rather than invalidating every page after it.

    The axes editor's cache, entry for entry — the overlay drawn over these
    crops is its overlay, so the pages have to carry the same geometry and be
    keyed by the same ``tform_obj_type``.  Sharing the directory is what makes
    an axis edit drop these pages too: ``_cache_drop`` globs the category, and a
    rewritten transform recolours their boxes and turns their arrows.
    """
    tag = page_tag(oids, n_views)
    if cache:
        hit = _cache_load(cfg, category, tag)
        if hit is not None:
            return hit

    indices = [i for oid in oids for i in by_obj.get(oid, [])]
    objects = collect_frames(dataset, cfg, indices)
    if cache and objects:
        _cache_store(cfg, category, tag, objects)
    return objects


# ── page rendering ────────────────────────────────────────────────────────────

def render_page(objects, selected: set, cursor: int,
                cols: int = GRID_COLS, views: int = GRID_VIEWS,
                P: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """One page: ``cols`` objects across, ``views`` of each down, in blocks.

    The border wraps an object's whole column of views, because that column *is*
    the verdict: a sequence goes into the subset with all of its frames or not
    at all.

    Each cell carries the axes editor's overlay — the 3-D box with its edges
    coloured by the axis they run along, and the triad from the object centre:
    red = right (X), green = top (Y), blue = back (Z).  *P* is that editor's
    pending rotation, and this tool has none, so it defaults to identity: what
    is drawn is the object frame as stored, which is what the verdict is about.
    ``select-and-axes-tform-obj-type`` passes a real one, being the same page
    with a rotation pending over it.
    """
    import cv2

    if not objects:
        return None
    P = np.eye(3) if P is None else P
    ncol = min(cols, len(objects))
    nrow = (-(-len(objects) // cols)) * views
    out = np.zeros((nrow * _CELL, ncol * _CELL, 3), dtype=np.uint8)
    for i, (oid, frames) in enumerate(objects):
        row0, col = block_of(i, cols, views)
        x0, y0 = col * _CELL, row0 * _CELL
        for j, frame in enumerate(frames[:views]):
            y = y0 + j * _CELL
            out[y:y + _CELL, x0:x0 + _CELL] = _draw_overlay(frame, P)
        on = oid in selected
        x1, y1 = x0 + _CELL - 2, y0 + views * _CELL - 2
        cv2.rectangle(out, (x0 + 1, y0 + 1), (x1, y1),
                      _SELECTED if on else _UNSELECTED, 10 if on else 3)
        if i == cursor:
            cv2.rectangle(out, (x0 + 12, y0 + 12), (x1 - 10, y1 - 10), _CURSOR, 3)
        # Big enough to still read once the page is scaled to the screen — at
        # 8 columns that is roughly half size.
        label = f"{i + 1}{' IN' if on else ''}"
        for colour, thick in (((0, 0, 0), 9), (_SELECTED if on else (235, 235, 235), 3)):
            cv2.putText(out, _ascii(label), (x0 + 16, y0 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, thick, cv2.LINE_AA)
    return out


# ── viewer ────────────────────────────────────────────────────────────────────

def run_subset_editor(cls, cfg, subset_name: str, *, dataset_name: str = "",
                      n_objects: int = GRID_OBJECTS, n_views: int = GRID_VIEWS,
                      cols: int = GRID_COLS, prefetch: int = PREFETCH,
                      category: Optional[str] = None,
                      max_height: Optional[int] = None,
                      target_count: Optional[int] = None,
                      init_count: Optional[int] = None,
                      max_count: Optional[int] = None) -> None:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    if not cfg.path_preprocess:
        print("this dataset config resolves no path_preprocess, so there is nowhere "
              "to store the subset.", file=sys.stderr)
        sys.exit(1)

    all_cats = _categories(cfg)
    if not all_cats:
        print("No categories found — is the dataset mounted?", file=sys.stderr)
        sys.exit(1)

    out_path = subset_store.path(cfg.path_preprocess, subset_name)
    existing = subset_store.load_if_exists(cfg.path_preprocess, subset_name)
    selected: set[str] = {_object_id(i) for i in existing.ids} if existing else set()
    print(f"subset {subset_name!r} -> {out_path}\n  starting from "
          f"{len(selected)} selected object ids")
    if existing is not None and len(selected) != len(existing.ids):
        print(f"  ({len(existing.ids)} entries in the file collapsed to whole "
              f"objects — saving will rewrite it at that granularity)")

    def _counts() -> Counter:
        """Selected objects per category — the whole file, in one pass.

        Object ids are ``<category>/<sequence>``, so the selection already says
        how far every category has got; nothing has to be walked to find out.
        """
        return Counter(o.split("/", 1)[0] for o in selected)

    # --init-count / --target-count: which categories this run is for. Filtered
    # once, at start, off the selection as it was loaded — a category whose count
    # changes while you work stays on the list until you leave it (n/p then skips
    # a finished one), because a list that reshuffled under the click would move
    # the page you are on.
    have = _counts()
    cats = list(all_cats)
    if init_count is not None:
        cats = [c for c in cats if have[c] == init_count]
        print(f"  init count {init_count}: {len(cats)}/{len(all_cats)} categories "
              f"start with exactly {init_count} objects selected")
        if not cats and category not in all_cats:
            print(f"No category has exactly {init_count} selected — nothing to review.")
            return
    if target_count is not None:
        done = [c for c in all_cats if have[c] >= target_count]
        cats = [c for c in cats if have[c] < target_count]
        print(f"  target {target_count} objects per category: "
              f"{len(done)}/{len(all_cats)} categories already there, "
              f"{len(cats)} to go")
        if not cats and category not in all_cats:
            print("Every category has reached the target — nothing to select.")
            return
    if category and category in all_cats and category not in cats:
        # explicitly asked for: offered even though a filter dropped it
        cats = sorted(set(cats) | {category}, key=all_cats.index)

    idx = cats.index(category) if category in cats else 0

    st = {"idx": idx, "page": 0, "cursor": 0, "status": "", "dirty": False,
          "dataset": None, "oids": [], "by_obj": {}, "objects": [], "scale": 1.0,
          "closed": False}
    prefetcher = Prefetcher() if prefetch > 0 else None
    # Walked categories held for the prefetch thread only — never the UI's own
    # dataset instance, whose mesh/tform caches are not synchronised.
    pf_cats: dict = {}

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"cannot open a window ({exc}).\nThis tool needs a display — run it on "
              f"your workstation, not over a headless ssh session.", file=sys.stderr)
        sys.exit(1)
    root.title(f"o3b  select-subset  {subset_name}")
    maximize(root)
    # The HUD takes 380 px on the left; the rest of the screen is the page's,
    # minus the toolbar and the window manager's own chrome.
    max_w = max(320, root.winfo_screenwidth() - 400)
    max_h = max_height or max(320, root.winfo_screenheight() - 140)

    bar = ttk.Frame(root)
    bar.pack(fill=tk.X, padx=6, pady=4)
    ttk.Label(bar, text="Category").pack(side=tk.LEFT, padx=(2, 6))
    cat_var = tk.StringVar(value=cats[idx])
    combo = ttk.Combobox(bar, textvariable=cat_var, values=cats, width=30)
    combo.pack(side=tk.LEFT)
    ttk.Button(bar, text="< cat", width=7,
               command=lambda: _step_category(-1)).pack(side=tk.LEFT, padx=(8, 2))
    ttk.Button(bar, text="cat >", width=7,
               command=lambda: _step_category(+1)).pack(side=tk.LEFT, padx=2)
    ttk.Button(bar, text="< page", width=7,
               command=lambda: _step_page(-1)).pack(side=tk.LEFT, padx=(12, 2))
    ttk.Button(bar, text="page >", width=7,
               command=lambda: _step_page(+1)).pack(side=tk.LEFT, padx=2)
    ttk.Button(bar, text="all", width=5,
               command=lambda: _set_page(True)).pack(side=tk.LEFT, padx=(12, 2))
    ttk.Button(bar, text="none", width=6,
               command=lambda: _set_page(False)).pack(side=tk.LEFT, padx=2)
    ttk.Button(bar, text="SAVE", command=lambda: _save()).pack(side=tk.LEFT, padx=(16, 2))

    canvas = ttk.Label(root)
    canvas.pack()
    canvas.focus_set()
    _photo = {"ref": None}          # keep a reference or tk garbage-collects it

    # ── state helpers ─────────────────────────────────────────────────────────

    def _n_pages() -> int:
        return max(1, -(-len(st["oids"]) // n_objects))

    def _page_oids() -> list[str]:
        start = st["page"] * n_objects
        return st["oids"][start:start + n_objects]

    def _cat_counts() -> tuple[int, int]:
        oids = st["oids"]
        return sum(1 for o in oids if o in selected), len(oids)

    def _ordered(oids: list, sel=None) -> list:
        """What is already in the subset first, the rest after it.

        So opening a category shows the standing selection on page 1 — that is
        what you review when you come back to one, and in a 1500-object category
        the alternative is paging to find it.  Applied when the category loads,
        not on every toggle: a page that reshuffled under the click would make
        the next click land somewhere else.
        """
        sel = selected if sel is None else sel
        return ([o for o in oids if o in sel]
                + [o for o in oids if o not in sel])

    # ── drawing ───────────────────────────────────────────────────────────────

    def _hud_lines():
        cat = cats[st["idx"]]
        n_sel, n_tot = _cat_counts()
        lines = [(f"subset  {subset_name}", (255, 255, 255), 1.2),
                 (_ascii(str(out_path)), (150, 150, 150), 0.75),
                 (f"{len(selected)} objects selected in total"
                  + ("  *unsaved*" if st["dirty"] else ""),
                  (120, 230, 120) if not st["dirty"] else (110, 200, 245), 0.95)]
        if init_count is not None:
            lines += [(f"init count {init_count}: {len(cats)}/{len(all_cats)} "
                       f"categories on the list", (200, 200, 120), 0.9)]
        if target_count is not None:
            have = _counts()
            done = sum(1 for c in all_cats if have[c] >= target_count)
            lines += [(f"target {target_count} per category: {done}/{len(all_cats)} "
                       f"there, {len(all_cats) - done} to go",
                       (120, 230, 120) if done == len(all_cats) else (200, 200, 120),
                       0.9)]
        lines += [("---", None, 1),
                  (cat, (255, 255, 255), 1.1)]
        if max_count is not None:
            lines += [(f"max count {max_count} objects offered per category",
                       (200, 200, 120), 0.85)]
        lines += [
                  (f"category [{st['idx'] + 1}/{len(cats)}]   "
                   f"page [{st['page'] + 1}/{_n_pages()}]", (170, 170, 170), 0.9),
                  (f"{n_sel}/{n_tot} objects of this category selected"
                   + (" (shown first)" if n_sel else ""), (190, 190, 190), 0.9)]
        if target_count is not None:
            reached = n_sel >= target_count
            lines += [(f"target: {n_sel}/{target_count}"
                       + ("  REACHED - n for the next category" if reached
                          else f"  {target_count - n_sel} more to go"),
                       (120, 230, 120) if reached else (200, 200, 120), 0.9)]
        lines += [("---", None, 1)]
        # UCO3D's rules for this category, in the frame the cells are drawn in —
        # the axes editor's panel, unchanged, because the same question is being
        # asked of the same triad: is this object's frame the one the category is
        # supposed to have? A column whose arrows contradict these rules is one
        # to leave out of the subset (or to send back to axes-tform-obj-type).
        lines += axis_rules_lines(cfg, cat)
        lines += [("---", None, 1),
                  ("X right (red)  Y top (green)  Z back (blue)", (200, 200, 200), 0.9),
                  ("---", None, 1),
                  (f"a column = one object, {n_views} views of it", (200, 200, 200), 0.85),
                  ("click a column, or Space, to toggle it", (200, 200, 200), 0.85),
                  ("a all on page   x none on page", (180, 180, 180), 0.85),
                  ("arrows move the cursor   1-9 toggle object n", (180, 180, 180), 0.85),
                  ("f/b or PgDn/PgUp page   n/p category", (180, 180, 180), 0.85),
                  ("s save   q quit (saves)", (180, 180, 180), 0.85),
                  ("dropdown above: type to filter, enter to pick", (180, 180, 180), 0.85),
                  ("---", None, 1),
                  (_ascii(st["status"]), (200, 200, 120), 0.85)]
        return lines

    def _show():
        if st["closed"]:
            # The last autosave runs after the window is gone (Ctrl-C in the
            # terminal leaves the selection dirty), and _save redraws — building
            # a PhotoImage without a root raises.
            return
        grid = render_page(st["objects"], selected, st["cursor"], cols, n_views)
        if grid is None:
            grid = np.zeros((_CELL, _CELL, 3), np.uint8)
        # Scale the page, not the HUD: 8 x (2 x 3) cells is 2560 x 1920 px, and
        # shrinking the text along with it would make it unreadable.
        grid, st["scale"] = fit(grid, max_w, max_h)
        h = max(grid.shape[0], 560)
        if grid.shape[0] < h:
            grid = np.pad(grid, ((0, h - grid.shape[0]), (0, 0), (0, 0)))
        frame = np.hstack([_hud(_hud_lines(), 380, h), grid])
        _photo["ref"] = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.configure(image=_photo["ref"])

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_category(i: int):
        st["idx"], st["page"], st["cursor"] = i, 0, 0
        cat = cats[i]
        cat_var.set(cat)
        st["objects"] = []
        st["status"] = f"walking {cat} …"
        _show()
        root.update()
        t0 = time.time()
        st["dataset"], order, st["by_obj"] = open_category(
            cls, cfg, cat, n_views, max_objects=max_count, keep=selected)
        st["oids"] = _ordered(order)
        st["status"] = f"{cat}: {len(st['oids'])} objects in {time.time() - t0:.2f}s"
        _load_page()

    def _load_page():
        st["cursor"] = 0
        cat = cats[st["idx"]]
        oids = _page_oids()
        if not oids:
            st["objects"] = []
            st["status"] = f"{cat}: nothing to show"
            _show()
            _prefetch()          # an empty category still points at the next ones
            return
        st["status"] = f"loading page {st['page'] + 1}/{_n_pages()} …"
        _show()
        root.update()
        t0 = time.time()
        st["objects"] = load_page(st["dataset"], cfg, cat, oids, st["by_obj"], n_views)
        st["status"] = (f"page {st['page'] + 1}/{_n_pages()}: {len(st['objects'])} "
                        f"objects in {time.time() - t0:.2f}s")
        _show()
        _prefetch()

    # ── prefetching ───────────────────────────────────────────────────────────
    # All three run on the prefetch thread. They must not touch tk or st.

    def _pf_category(cat: str, sel=frozenset()):
        """This category's walk, on the prefetch thread's own dataset instance.

        *sel* is the UI thread's snapshot, never the live set: with --max-count
        it decides which objects the cap has to keep, and reading a set another
        thread is mutating would be a race for the sake of a page that is only
        being guessed at anyway.
        """
        if cat not in pf_cats:
            if len(pf_cats) > prefetch + 1:
                pf_cats.clear()
            pf_cats[cat] = open_category(cls, cfg, cat, n_views,
                                         max_objects=max_count, keep=sel)
        return pf_cats[cat]

    def _pf_first_page(cat: str, sel: frozenset):
        """Warm what ``_load_category`` would show first — selection order and all.

        *sel* is a snapshot taken on the UI thread when the plan was made, both
        so this thread never reads a set another one is mutating, and so the page
        it warms is the page that will be asked for.  Selecting more objects in
        *this* category afterwards would make that a miss, which costs a normal
        load and nothing else.
        """
        dataset, order, by_obj = _pf_category(cat, sel)
        if dataset is not None and order:
            load_page(dataset, cfg, cat, _ordered(order, sel)[:n_objects],
                      by_obj, n_views)

    def _prefetch():
        """Queue the *first page* of the next `prefetch` categories, nothing else.

        Deeper pages of the current category are deliberately not warmed: most
        categories are judged from their first page and moved on from, so
        decoding page 2 competes with the next category for the same thread and
        usually loses.
        """
        if prefetcher is None:
            return
        sel = frozenset(selected)      # snapshot: the jobs run on another thread
        jobs = []
        for d in range(1, prefetch + 1):
            nxt = cats[(st["idx"] + d) % len(cats)]
            jobs.append((f"{nxt}|p0", partial(_pf_first_page, nxt, sel)))
        queued = prefetcher.plan(jobs)
        if queued:
            st["status"] += f"   (+{queued} prefetching)"
            _show()

    # ── selection ─────────────────────────────────────────────────────────────

    def _toggle(row: int):
        if not (0 <= row < len(st["objects"])):
            return
        oid = st["objects"][row][0]
        if oid in selected:
            selected.discard(oid)
        else:
            selected.add(oid)
        st["cursor"] = row
        st["dirty"] = True
        _show()

    def _set_page(on: bool):
        for oid, _frames in st["objects"]:
            selected.add(oid) if on else selected.discard(oid)
        st["dirty"] = True
        st["status"] = ("selected" if on else "cleared") + " this page"
        _show()

    def _save(quiet: bool = False):
        path = subset_store.save(
            cfg.path_preprocess, subset_name, selected,
            header={"dataset": dataset_name or cfg.class_name,
                    "tform_obj_type": cfg.tform_obj_type},
        )
        st["dirty"] = False
        st["status"] = f"saved {len(selected)} object ids"
        if not quiet:
            print(f"  wrote {len(selected)} object ids -> {path}")
        _show()

    def _autosave():
        """Save before anything that could take a while or lose the window."""
        if st["dirty"]:
            _save(quiet=True)

    # ── navigation ────────────────────────────────────────────────────────────

    def _step_category(d: int):
        """Next category — skipping any that has since reached the target.

        The list was filtered once at start, so what n/p has to skip is only the
        categories finished during this session.  Landing back where it started
        means every one of them is done, which the status line then says.
        """
        _autosave()
        i = st["idx"]
        for _ in range(len(cats)):
            i = (i + d) % len(cats)
            if target_count is None or _counts()[cats[i]] < target_count:
                break
        if i == st["idx"] and target_count is not None:
            st["status"] = "every remaining category has reached the target"
            _show()
            canvas.focus_set()
            return
        _load_category(i)
        canvas.focus_set()

    def _step_page(d: int):
        _autosave()
        st["page"] = (st["page"] + d) % _n_pages()
        _load_page()
        canvas.focus_set()

    def _pick(_e=None):
        want = cat_var.get().strip()
        if want in cats:
            _autosave()
            _load_category(cats.index(want))
        canvas.focus_set()

    def _rank(text: str) -> list:
        text = text.strip().lower()
        if not text:
            return cats
        exact  = [c for c in cats if c.lower() == text]
        prefix = [c for c in cats if c.lower().startswith(text) and c not in exact]
        inside = [c for c in cats if text in c.lower()
                  and c not in exact and c not in prefix]
        return exact + prefix + inside

    def _filter(e):
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

    # ── input ─────────────────────────────────────────────────────────────────

    def _on_click(e):
        # x is measured from the left of the composed image, whose first 380 px
        # are the HUD; both are in *display* pixels, so undo the page's scale.
        if e.x < 380:
            return
        s = max(st["scale"], 1e-6)
        col = int((e.x - 380) / s / _CELL)
        row = int(e.y / s / _CELL)
        if col >= cols:
            return
        _toggle((row // n_views) * cols + col)

    canvas.bind("<Button-1>", _on_click)

    def _quit():
        _autosave()
        if prefetcher is not None:
            prefetcher.stop()   # a job mid-decode still finishes; it is a daemon
        st["closed"] = True
        root.destroy()

    def _on_key(e):
        if combo_focused(root, combo):
            return
        k = e.keysym.lower()
        if k in ("q", "escape"):
            _quit()
        elif k in ("n", "period"):
            _step_category(+1)
        elif k in ("p", "comma"):
            _step_category(-1)
        elif k in ("f", "next"):            # "next" is the Page Down keysym
            _step_page(+1)
        elif k in ("b", "prior"):
            _step_page(-1)
        elif k in ("right", "left", "down", "up"):
            # the cursor walks the page as it is laid out: one object sideways,
            # a whole block of `cols` vertically
            step = {"right": +1, "left": -1, "down": +cols, "up": -cols}[k]
            st["cursor"] = min(max(st["cursor"] + step, 0), max(0, len(st["objects"]) - 1))
            _show()
        elif k == "space":
            _toggle(st["cursor"])
        elif k == "a":
            _set_page(True)
        elif k == "x":
            _set_page(False)
        elif k == "s":
            _save()
        elif e.char.isdigit() and e.char != "0":
            _toggle(int(e.char) - 1)

    root.bind_all("<Key>", _on_key)
    root.protocol("WM_DELETE_WINDOW", _quit)

    _load_category(idx)
    print("\nKeys: click/Space toggle - arrows move cursor - a all - x none - "
          "f/b page - n/p category - s save - q quit\n")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    st["closed"] = True
    _autosave()
    if prefetcher is not None:
        # the window is gone by now, so waiting out a decode costs the user nothing
        prefetcher.stop(join=30)
    print(f"Stopping. {len(selected)} object ids in {out_path}")
    if target_count is not None:
        have = _counts()
        done = sum(1 for c in all_cats if have[c] >= target_count)
        print(f"  target {target_count} per category: {done}/{len(all_cats)} "
              f"categories there, {len(all_cats) - done} to go")
