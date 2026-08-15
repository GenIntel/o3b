"""Select a category's objects, then rotate the axes of *those* objects.

    o3b dataset select-and-axes-tform-obj-type -d every9d_v6
    o3b dataset select-and-axes-tform-obj-type -d every9d_v6 -s every9d_v6_test

The two editors this joins answer the same question at different granularities.
``axes-tform-obj-type`` fixes a *category*: one rotation, every sequence in it.
``select-subset`` picks *objects*, and writes nothing but a list of them.  What
neither does is the common middle case — a category whose canonical poses are
right for most of its sequences and turned for a handful, where the whole-
category rotation would break the ones that were already correct.

So this tool runs in two modes over one page of objects, ``Tab`` (or ``m``)
between them:

**SELECT** — ``select-subset``'s page.  Click a column (or Space on the cursor)
to toggle that object; ``a``/``x`` take the whole page, ``A``/``X`` the whole
category.  The overlay is the object frame as stored, which is what you are
judging.

**AXES** — ``axes-tform-obj-type``'s keys, over the objects the selection left.
``1``/``2``/``3`` swap XY/YZ/ZX, ``4``/``5``/``6`` flip X/Y/Z, ``u`` undo, ``r``
reset; the arrows and box colours follow live.  ``s`` writes ``inv(G) @ P @ G @
T`` back for the selected sequences **and no others**.

Only proper rotations are written, exactly as in the axes editor: a lone swap or
flip is a reflection (det = -1) and would mirror the objects it touched, so save
refuses until they pair up.  And save refuses on an empty selection — what gets
written is always precisely what is selected, which is what ``A`` is for when
the answer really is "the whole category".

Entering AXES snapshots the selection into the page it shows, so deselecting an
object there (its border greys out) drops it from the save without the page
moving under the cursor; leaving and re-entering re-pages what is left.

With ``-s NAME`` the selection is also a subset file — loaded at start, autosaved
on every navigation, the same ``<path_preprocess>/subset/NAME.yaml`` that
``select-subset`` writes and a config's ``subset_name`` reads.  Without it the
selection is a scratch pad for the rotation and is gone at exit.

Pages, crop cache and prefetching are ``select_subset``'s, entry for entry, and
so is ``--max-count N``: at most N objects offered per category, so a big one
opens in seconds, with everything already selected offered on top of the cap.
"""
from __future__ import annotations

import sys
import time
from functools import partial
from typing import Optional

import numpy as np

from o3b.dataset import subset_store
from o3b.dataset.axes_tform_obj import (
    Pending, Prefetcher, axis_rules_lines, save_category,
    _ascii, _cache_drop, _categories, _hud, combo_focused, fit, maximize,
)
from o3b.dataset.grid import GRID_COLS, GRID_OBJECTS, GRID_VIEWS, PREFETCH
from o3b.dataset.select_subset import (
    _CELL, _object_id, load_page, open_category, render_page,
)

_MODE_SELECT = "select"
_MODE_AXES = "axes"


def run_select_axes_editor(cls, cfg, subset_name: Optional[str] = None, *,
                           dataset_name: str = "",
                           n_objects: int = GRID_OBJECTS, n_views: int = GRID_VIEWS,
                           cols: int = GRID_COLS, prefetch: int = PREFETCH,
                           category: Optional[str] = None,
                           max_height: Optional[int] = None,
                           max_count: Optional[int] = None) -> None:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    if not cfg.path_preprocess:
        print("this dataset config resolves no path_preprocess, so there is nothing "
              "to read or write.", file=sys.stderr)
        sys.exit(1)

    cats = _categories(cfg)
    if not cats:
        print("No categories with tform_obj found — is the dataset mounted?",
              file=sys.stderr)
        sys.exit(1)
    idx = cats.index(category) if category in cats else 0

    G = np.array(cfg.obj_gl_tform4x4_obj_raw, dtype=np.float64) \
        if cfg.obj_gl_tform4x4_obj_raw is not None else None

    # The selection: object ids, across every category visited. Persisted only
    # when the caller named a subset — otherwise it exists to aim the rotation.
    selected: set[str] = set()
    out_path = None
    if subset_name:
        out_path = subset_store.path(cfg.path_preprocess, subset_name)
        existing = subset_store.load_if_exists(cfg.path_preprocess, subset_name)
        selected = {_object_id(i) for i in existing.ids} if existing else set()
        print(f"subset {subset_name!r} -> {out_path}\n  starting from "
              f"{len(selected)} selected object ids")
    else:
        print("no -s given: the selection aims the rotation and is not written "
              "anywhere. The rotation itself is written to "
              f"tform_obj/{cfg.tform_obj_type}.")

    pending = Pending()
    st = {"idx": idx, "page": 0, "cursor": 0, "status": "", "dirty": False,
          "mode": _MODE_SELECT, "dataset": None, "oids": [], "by_obj": {},
          "axes_oids": [], "objects": [], "scale": 1.0, "closed": False}
    prefetcher = Prefetcher() if prefetch > 0 else None
    pf_cats: dict = {}          # walks owned by the prefetch thread alone

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"cannot open a window ({exc}).\nThis tool needs a display — run it on "
              f"your workstation, not over a headless ssh session.", file=sys.stderr)
        sys.exit(1)
    root.title("o3b  select-and-axes-tform-obj-type"
               + (f"  {subset_name}" if subset_name else ""))
    maximize(root)
    max_w = max(320, root.winfo_screenwidth() - 400)
    max_h = max_height or max(320, root.winfo_screenheight() - 140)

    bar = ttk.Frame(root)
    bar.pack(fill=tk.X, padx=6, pady=4)
    ttk.Label(bar, text="Category").pack(side=tk.LEFT, padx=(2, 6))
    cat_var = tk.StringVar(value=cats[idx])
    combo = ttk.Combobox(bar, textvariable=cat_var, values=cats, width=28)
    combo.pack(side=tk.LEFT)
    ttk.Button(bar, text="< cat", width=7,
               command=lambda: _step_category(-1)).pack(side=tk.LEFT, padx=(8, 2))
    ttk.Button(bar, text="cat >", width=7,
               command=lambda: _step_category(+1)).pack(side=tk.LEFT, padx=2)
    ttk.Button(bar, text="< page", width=7,
               command=lambda: _step_page(-1)).pack(side=tk.LEFT, padx=(12, 2))
    ttk.Button(bar, text="page >", width=7,
               command=lambda: _step_page(+1)).pack(side=tk.LEFT, padx=2)
    mode_btn = ttk.Button(bar, text="mode: SELECT", width=14,
                          command=lambda: _toggle_mode())
    mode_btn.pack(side=tk.LEFT, padx=(16, 2))
    save_btn = ttk.Button(bar, text="SAVE rotation", command=lambda: _save_rotation())
    save_btn.pack(side=tk.LEFT, padx=(12, 2))

    canvas = ttk.Label(root)
    canvas.pack()
    canvas.focus_set()
    _photo = {"ref": None}          # keep a reference or tk garbage-collects it

    # ── state helpers ─────────────────────────────────────────────────────────

    def _axes_mode() -> bool:
        return st["mode"] == _MODE_AXES

    def _pool() -> list:
        """The objects the current mode pages over.

        SELECT walks the category; AXES walks the snapshot taken when it was
        entered, so a toggle there changes what is written without reshuffling
        the page it was made on.
        """
        return st["axes_oids"] if _axes_mode() else st["oids"]

    def _n_pages() -> int:
        return max(1, -(-len(_pool()) // n_objects))

    def _page_oids() -> list[str]:
        start = st["page"] * n_objects
        return _pool()[start:start + n_objects]

    def _cat_selected() -> list[str]:
        return [o for o in st["oids"] if o in selected]

    def _ordered(oids: list, sel=None) -> list:
        """Selected first — the standing selection is what you come back to."""
        sel = selected if sel is None else sel
        return [o for o in oids if o in sel] + [o for o in oids if o not in sel]

    # ── drawing ───────────────────────────────────────────────────────────────

    def _hud_lines():
        cat = cats[st["idx"]]
        n_sel, n_tot = len(_cat_selected()), len(st["oids"])
        head = ("MODE: SELECT objects" if not _axes_mode() else "MODE: AXES rotation")
        lines = [(head, (120, 230, 120) if not _axes_mode() else (245, 200, 110), 1.15),
                 ("Tab or m switches mode", (150, 150, 150), 0.8),
                 ("---", None, 1),
                 (cat, (255, 255, 255), 1.1),
                 (f"category [{st['idx'] + 1}/{len(cats)}]   "
                  f"page [{st['page'] + 1}/{_n_pages()}]", (170, 170, 170), 0.9),
                 (f"{n_sel}/{n_tot} objects selected here"
                  + (f"   ({len(selected)} in all)" if len(selected) != n_sel else ""),
                  (190, 190, 190), 0.9)]
        if subset_name:
            lines += [(f"subset {subset_name}" + ("  *unsaved*" if st["dirty"] else ""),
                       (110, 200, 245) if st["dirty"] else (150, 200, 150), 0.85)]
        if max_count is not None:
            lines += [(f"max count {max_count} objects offered per category",
                       (200, 200, 120), 0.85)]
        lines += [("---", None, 1)]
        # The category's rules, in the config's frame and fixed there: they are
        # what the arrows are supposed to become, in both modes.
        lines += axis_rules_lines(cfg, cat)
        lines += [("---", None, 1),
                  ("X right (red)  Y top (green)  Z back (blue)", (200, 200, 200), 0.9),
                  ("---", None, 1)]
        if _axes_mode():
            ok = pending.is_rotation
            lines += [("P (rows = new X, Y, Z):", (170, 170, 200), 0.85)]
            lines += [("   [" + ", ".join(f"{v:5.1f}" for v in row) + "]",
                       (200, 200, 255), 0.85) for row in pending.P]
            lines += [(f"det(P) = {pending.det:+.3f}"
                       + ("  rotation OK" if ok else "  REFLECTION - cannot save"),
                       (120, 230, 120) if ok else (110, 110, 245), 1.0),
                      (f"s writes it to the {n_sel} selected object(s)",
                       (200, 200, 200), 0.85),
                      ("1/2/3 swap XY YZ ZX   4/5/6 flip X Y Z", (180, 180, 180), 0.85),
                      ("u undo   r reset   s SAVE rotation", (180, 180, 180), 0.85),
                      ("click/Space still drops an object", (180, 180, 180), 0.85)]
        else:
            lines += [("a column = one object, "
                       f"{n_views} views of it", (200, 200, 200), 0.85),
                      ("click a column, or Space, to toggle it", (200, 200, 200), 0.85),
                      ("a/x all/none on page   A/X whole category",
                       (180, 180, 180), 0.85),
                      ("arrows move the cursor   1-9 toggle object n",
                       (180, 180, 180), 0.85)]
            if subset_name:
                lines += [("s saves the subset file", (180, 180, 180), 0.85)]
        lines += [("f/b or PgDn/PgUp page   n/p category", (180, 180, 180), 0.85),
                  ("q quit", (180, 180, 180), 0.85),
                  ("---", None, 1),
                  (_ascii(st["status"]), (200, 200, 120), 0.85)]
        return lines

    def _show():
        if st["closed"]:
            # The last autosave runs after the window is gone (Ctrl-C in the
            # terminal leaves the selection dirty), and _save_subset redraws —
            # building a PhotoImage without a root raises.
            return
        grid = render_page(st["objects"], selected, st["cursor"], cols, n_views,
                           P=pending.P if _axes_mode() else None)
        if grid is None:
            grid = np.zeros((_CELL, _CELL, 3), np.uint8)
        # Scale the page, not the HUD — the text would stop being readable.
        grid, st["scale"] = fit(grid, max_w, max_h)
        h = max(grid.shape[0], 620)
        if grid.shape[0] < h:
            grid = np.pad(grid, ((0, h - grid.shape[0]), (0, 0), (0, 0)))
        frame = np.hstack([_hud(_hud_lines(), 380, h), grid])
        _photo["ref"] = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.configure(image=_photo["ref"])
        mode_btn.configure(text="mode: AXES" if _axes_mode() else "mode: SELECT")
        save_btn.state(["!disabled"] if (_axes_mode() and pending.is_rotation
                                         and not pending.is_identity
                                         and _cat_selected())
                       else ["disabled"])

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_category(i: int):
        st["idx"], st["page"], st["cursor"] = i, 0, 0
        st["mode"] = _MODE_SELECT       # a new category starts from its selection
        pending.reset()
        pending._history.clear()
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
        st["axes_oids"] = []
        st["status"] = f"{cat}: {len(st['oids'])} objects in {time.time() - t0:.2f}s"
        _load_page()

    def _rewalk():
        """Re-open the current category, keeping mode, page and selection.

        What a save needs: the loader reads the tform_obj table once, into a
        dict it then keeps (it has to survive the fork into DataLoader workers),
        so a dataset built before the write still hands out the old transforms.
        Dropping the crop cache alone would re-decode the same wrong geometry.
        """
        cat = cats[st["idx"]]
        st["dataset"], order, st["by_obj"] = open_category(
            cls, cfg, cat, n_views, max_objects=max_count, keep=selected)
        st["oids"] = _ordered(order)
        still = set(order)
        st["axes_oids"] = [o for o in st["axes_oids"] if o in still]
        # a sequence can drop out of the walk (its transform is gone), so the
        # page that was open may no longer exist
        st["page"] = min(st["page"], _n_pages() - 1)
        _load_page()

    def _load_page():
        st["cursor"] = 0
        cat = cats[st["idx"]]
        oids = _page_oids()
        if not oids:
            st["objects"] = []
            st["status"] = (f"{cat}: nothing selected to rotate" if _axes_mode()
                            else f"{cat}: nothing to show")
            _show()
            _prefetch()
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

    # ── prefetching (on the prefetch thread: no tk, no st, no shared dataset) ──

    def _pf_category(cat: str, sel=frozenset()):
        if cat not in pf_cats:
            if len(pf_cats) > prefetch + 1:
                pf_cats.clear()
            # sel is the UI thread's snapshot — with --max-count it decides which
            # objects the cap must keep, and the live set belongs to that thread
            pf_cats[cat] = open_category(cls, cfg, cat, n_views,
                                         max_objects=max_count, keep=sel)
        return pf_cats[cat]

    def _pf_first_page(cat: str, sel: frozenset):
        dataset, order, by_obj = _pf_category(cat, sel)
        if dataset is not None and order:
            load_page(dataset, cfg, cat, _ordered(order, sel)[:n_objects],
                      by_obj, n_views)

    def _prefetch():
        """The first page of the next `prefetch` categories, as in select-subset."""
        if prefetcher is None:
            return
        sel = frozenset(selected)      # snapshot: the jobs run on another thread
        jobs = [(f"{cats[(st['idx'] + d) % len(cats)]}|p0",
                 partial(_pf_first_page, cats[(st["idx"] + d) % len(cats)], sel))
                for d in range(1, prefetch + 1)]
        queued = prefetcher.plan(jobs)
        if queued:
            st["status"] += f"   (+{queued} prefetching)"
            _show()

    # ── selection ─────────────────────────────────────────────────────────────

    def _toggle(row: int):
        if not (0 <= row < len(st["objects"])):
            return
        oid = st["objects"][row][0]
        selected.discard(oid) if oid in selected else selected.add(oid)
        st["cursor"] = row
        st["dirty"] = True
        _show()

    def _set_page(on: bool):
        for oid, _frames in st["objects"]:
            selected.add(oid) if on else selected.discard(oid)
        st["dirty"] = True
        st["status"] = ("selected" if on else "cleared") + " this page"
        _show()

    def _set_category(on: bool):
        """The whole category — what "rotate all of it" needs, said explicitly."""
        for oid in st["oids"]:
            selected.add(oid) if on else selected.discard(oid)
        st["dirty"] = True
        st["status"] = (("selected" if on else "cleared")
                        + f" all {len(st['oids'])} objects of {cats[st['idx']]}")
        _show()

    # ── modes ─────────────────────────────────────────────────────────────────

    def _toggle_mode():
        if _axes_mode():
            st["mode"] = _MODE_SELECT
            pending.reset()
            pending._history.clear()
            st["oids"] = _ordered(st["oids"])
            st["status"] = "select mode: pending rotation dropped"
        else:
            sel = _cat_selected()
            if not sel:
                st["status"] = "select some objects first — a rotation needs a target"
                _show()
                return
            st["mode"] = _MODE_AXES
            st["axes_oids"] = sel
            st["status"] = f"axes mode: {len(sel)} selected object(s)"
        st["page"], st["cursor"] = 0, 0
        _load_page()
        canvas.focus_set()

    def _save_rotation():
        cat = cats[st["idx"]]
        targets = set(_cat_selected())
        if not _axes_mode():
            st["status"] = "Tab into AXES mode to rotate"
        elif not targets:
            st["status"] = "REFUSED: nothing selected in this category"
        elif pending.is_identity:
            st["status"] = "nothing to save"
        elif not pending.is_rotation:
            st["status"] = f"REFUSED: det(P) = {pending.det:+.3f}, not a rotation"
            print(st["status"], file=sys.stderr)
        else:
            n, where = save_category(cfg, cat, pending.P, G, objects=targets)
            print(f"  {cat}: wrote {n} of {len(targets)} selected transforms -> {where}")
            pending.reset()
            pending._history.clear()
            _cache_drop(cfg, cat)      # geometry moved; those crops are stale
            st["status"] = f"wrote {n} transforms"
            _rewalk()                  # …and so is the dataset that fed them
            return
        _show()

    # ── the subset file (only with -s) ────────────────────────────────────────

    def _save_subset(quiet: bool = False):
        if not subset_name:
            st["status"] = "no -s given: nothing to write the selection to"
            _show()
            return
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
        if subset_name and st["dirty"]:
            _save_subset(quiet=True)

    # ── navigation ────────────────────────────────────────────────────────────

    def _step_category(d: int):
        _autosave()
        _load_category((st["idx"] + d) % len(cats))
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

    # The swap/flip keys, live only in AXES mode.
    _AXIS_KEYS = {
        "1": lambda: pending.swap(0, 1), "2": lambda: pending.swap(1, 2),
        "3": lambda: pending.swap(2, 0), "4": lambda: pending.flip(0),
        "5": lambda: pending.flip(1),    "6": lambda: pending.flip(2),
        "u": pending.undo,               "r": pending.reset,
    }

    def _on_key(e):
        if combo_focused(root, combo):
            return
        k = e.keysym.lower()
        if k in ("q", "escape"):
            _quit()
        elif k == "m":
            # Tab has its own binding below, but it is also tk's focus-traversal
            # key and a window manager can eat it before tk sees it; m is the
            # alias that cannot be intercepted.
            _toggle_mode()
        elif k in ("n", "period"):
            _step_category(+1)
        elif k in ("p", "comma"):
            _step_category(-1)
        elif k in ("f", "next"):            # "next" is the Page Down keysym
            _step_page(+1)
        elif k in ("b", "prior"):
            _step_page(-1)
        elif k in ("right", "left", "down", "up"):
            step = {"right": +1, "left": -1, "down": +cols, "up": -cols}[k]
            st["cursor"] = min(max(st["cursor"] + step, 0),
                               max(0, len(st["objects"]) - 1))
            _show()
        elif k == "space":
            _toggle(st["cursor"])
        elif k == "s":
            _save_rotation() if _axes_mode() else _save_subset()
        elif _axes_mode() and e.char in _AXIS_KEYS:
            _AXIS_KEYS[e.char]()
            _show()
        elif not _axes_mode():
            # select-mode-only keys, kept out of AXES so 1-9 stay the axis keys
            if e.char == "a":
                _set_page(True)
            elif e.char == "x":
                _set_page(False)
            elif e.char == "A":
                _set_category(True)
            elif e.char == "X":
                _set_category(False)
            elif e.char.isdigit() and e.char != "0":
                _toggle(int(e.char) - 1)

    def _on_tab(e):
        """Tab is tk's focus-traversal key, so it needs its own binding.

        Bound on the same tag as ``<Key>``, which tk resolves by specificity —
        so this is where Tab arrives, and "break" is what stops focus moving to
        the next widget behind it.  In the dropdown it stays traversal.
        """
        if combo_focused(root, combo):
            return None
        _toggle_mode()
        return "break"

    root.bind_all("<Key>", _on_key)
    root.bind_all("<Tab>", _on_tab)
    root.protocol("WM_DELETE_WINDOW", _quit)

    _load_category(idx)
    print("\nSELECT: click/Space toggle - a/x page - A/X category - arrows - 1-9\n"
          "AXES:   1/2/3 swap - 4/5/6 flip - u undo - r reset - s write rotation\n"
          "Both:   Tab/m switch mode - f/b page - n/p category - q quit\n")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    st["closed"] = True
    _autosave()
    if prefetcher is not None:
        # the window is gone by now, so waiting out a decode costs the user nothing
        prefetcher.stop(join=30)
    if subset_name:
        print(f"Stopping. {len(selected)} object ids in {out_path}")
    else:
        print("Stopping.")
