"""Per-category axis editor for canonical object poses.

    o3b dataset axes-tform-obj-type -d every9d_v5
    o3b dataset axes-tform-obj-type -d every9d_v5 -r every9d_v3

UCO3D's canonical poses are labelled per sequence, but the *axis convention* is a
property of the category — "the bottle's up is +Y", not "this bottle's up".  When
a category comes out with its axes permuted or mirrored relative to o3b's
right/top/back = X/Y/Z, the fix is one rotation applied to every sequence of that
category.  This is the tool for finding that rotation and writing it.

The viewer shows several objects of one category side by side, each already in
its canonical frame with an axis triad, plus the frames they came from so you can
tell what you are looking at, and the category's own axis rules from UCO3D's
orientation tree.  Swap/flip keys rotate the whole category at once; save writes
the composed transform back for every sequence in it.

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

import sys
import time
from typing import Optional

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Axis semantics, matching o3b's convention and o3b/dataset/tform.py's colours.
_AXES = [
    ("X", "right", (0.9, 0.2, 0.2)),
    ("Y", "top",   (0.2, 0.85, 0.2)),
    ("Z", "back",  (0.25, 0.45, 1.0)),
]


# ── the category's own axis rules ─────────────────────────────────────────────

def axis_description(category: str) -> str:
    """UCO3D's per-axis orientation rules for a category, as markdown.

    The orientation tree says how each canonical axis was *defined* for the
    category; an axis with no rule is one the labelling never pinned down, which
    is exactly the axis a symmetry lives on (see uco3d/obj_syms.py).
    """
    try:
        from o3b.dataset.uco3d.map_orient_tree import MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS
        from o3b.dataset.uco3d.orient_tree import MAP_CONDITION_TO_RULE
    except Exception:
        return "_(no orientation tree available)_"

    keys = MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS.get(category)
    if keys is None:
        return f"**{category}** — _not in the UCO3D orientation tree_"

    lines = [f"**{category}** — UCO3D axis definition:", ""]
    for (axis, name, _), key in zip(_AXES, keys):
        if key is None:
            lines.append(f"- `{axis}` ({name}): _undefined — free / symmetric_")
        else:
            rule = MAP_CONDITION_TO_RULE.get(key, "")
            lines.append(f"- `{axis}` ({name}): **{key}**")
            if rule:
                lines.append(f"    - {rule}")
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

def _load_category(cls, cfg, category: str, n_objects: int, n_views: int):
    """[(sequence, verts, faces, [rgb, ...])] for up to n_objects of *category*."""
    from dataclasses import replace as _r

    extra = dict(cfg.extra or {})
    extra["frames_count_max_per_sequence"] = n_views
    extra["sequences_count_max_per_category"] = n_objects
    view_cfg = _r(
        cfg,
        categories=[category],
        filter_count_max=n_objects * n_views,
        transform=None,            # full frames, no crop
        modalities={"rgb", "mesh", "cam_tform4x4_obj", "cam_tform4x4_obj_ncds"},
        extra=extra,
    )
    try:
        dataset = cls(view_cfg)
    except Exception as exc:
        print(f"  could not build dataset for {category}: {exc}", file=sys.stderr)
        return []

    by_seq: dict = {}
    for i in range(len(dataset)):
        try:
            fo = dataset[i]
        except Exception as exc:
            print(f"  item {i}: {exc}", file=sys.stderr)
            continue
        if fo is None or fo.mesh is None:
            continue
        entry = by_seq.setdefault(
            fo.object_id, {"verts": fo.mesh.verts.numpy(), "faces": fo.mesh.faces.numpy(), "rgb": []}
        )
        if fo.rgb is not None and len(entry["rgb"]) < n_views:
            entry["rgb"].append(fo.rgb)
    return [(k, v["verts"], v["faces"], v["rgb"]) for k, v in by_seq.items()][:n_objects]


def _tile(rgbs, cell: int = 128) -> "Optional[np.ndarray]":
    """Lay out [[rgb, ...], ...] as one uint8 image grid (rows = objects)."""
    import torch.nn.functional as F

    rows = [r for r in rgbs if r]
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    out = np.zeros((len(rows) * cell, ncol * cell, 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, rgb in enumerate(row):
            img = rgb.float()
            if img.max() > 1.5:
                img = img / 255.0
            img = F.interpolate(img[None], size=(cell, cell), mode="bilinear",
                                align_corners=False)[0]
            out[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = (
                img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
    return out


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


def save_category(cfg, category: str, P: np.ndarray, G) -> tuple[int, str]:
    """Apply P to every sequence of *category*. Returns (n_written, destination)."""
    from o3b.dataset import tform_obj_store as store

    pp, tot = cfg.path_preprocess, cfg.tform_obj_type
    table = store.load_all(pp, tot)

    if table is not None:                      # SQLite: one transaction
        rows = [(category, k.split("/", 1)[1], stored_tform(P, G, T))
                for k, T in table.items() if k.split("/", 1)[0] == category]
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
        if not f.exists():
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


# ── viewer ────────────────────────────────────────────────────────────────────

def run_axes_editor(cls, cfg, *, reference=None, ref_cls=None, n_objects: int = 5,
                    n_views: int = 3, category: Optional[str] = None) -> None:
    try:
        import viser
    except ImportError:
        print("Install viser: pip install viser", file=sys.stderr)
        sys.exit(1)

    from o3b.data.viz import make_viser_server
    from o3b.dataset.tform import _make_arrow

    cats = _categories(cfg)
    if not cats:
        print("No categories with tform_obj found — is the dataset mounted?", file=sys.stderr)
        sys.exit(1)
    if category and category in cats:
        cats = [category] + [c for c in cats if c != category]

    G = np.array(cfg.obj_gl_tform4x4_obj_raw, dtype=np.float64) \
        if cfg.obj_gl_tform4x4_obj_raw is not None else None

    server = make_viser_server()
    server.scene.add_light_ambient("/ambient", intensity=3.0)

    pending = Pending()
    state = {"category": cats[0], "objects": [], "ref_objects": [], "dirty": False}
    handles: list = []

    def _clear():
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _add_row(objects, y_offset: float, prefix: str, opacity: float):
        """Place objects along +X at a given Y, each with its axis triad."""
        for k, (seq, verts, faces, _) in enumerate(objects):
            v = verts.astype(np.float64) @ pending.P.T if prefix == "cur" else verts.astype(np.float64)
            span = float(np.abs(v).max()) + 1e-6
            off = np.array([k * 3.0, y_offset, 0.0])
            handles.append(server.scene.add_mesh_simple(
                f"/{prefix}/{k}/mesh", vertices=(v + off).astype(np.float32),
                faces=faces.astype(np.uint32), color=(200, 200, 210), opacity=opacity,
            ))
            for a, (axis, name, colour) in enumerate(_AXES):
                d = np.eye(3)[a]
                arrow = _make_arrow(d, colour, 1.6 * span)
                arrow.apply_translation(off)
                handles.append(server.scene.add_mesh_trimesh(
                    f"/{prefix}/{k}/axis_{axis}", arrow))
            handles.append(server.scene.add_label(
                f"/{prefix}/{k}/label", text=seq.split("/")[-1],
                position=(off + np.array([0.0, 0.0, span * 1.9])).astype(np.float32)))

    def _redraw():
        _clear()
        _add_row(state["objects"], 0.0, "cur", 1.0)
        if state["ref_objects"]:
            _add_row(state["ref_objects"], -3.5, "ref", 0.55)
        _refresh_status()

    def _refresh_status():
        det = pending.det
        ok = pending.is_rotation
        txt_p.value = pending.as_text()
        txt_det.value = f"{det:+.3f}" + ("  OK (rotation)" if ok else "  REFLECTION — cannot save")
        btn_save.disabled = not (ok and state["dirty"])
        md_status.content = (
            f"### {state['category']}\n\n"
            f"{len(state['objects'])} objects"
            + (f" · reference: {len(state['ref_objects'])}" if state["ref_objects"] else "")
            + (f"\n\n**unsaved changes**" if state["dirty"] else "")
        )

    def _load(cat: str):
        state["category"] = cat
        md_axes.content = "_loading…_"
        state["objects"] = _load_category(cls, cfg, cat, n_objects, n_views)
        state["ref_objects"] = (_load_category(ref_cls, reference, cat, n_objects, n_views)
                                if reference is not None else [])
        md_axes.content = axis_description(cat)
        tile = _tile([o[3] for o in state["objects"]])
        if tile is not None:
            img_frames.image = tile
        ref_tile = _tile([o[3] for o in state["ref_objects"]]) if state["ref_objects"] else None
        if ref_tile is not None:
            img_ref.image = ref_tile
        img_ref.visible = ref_tile is not None
        _redraw()

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Category"):
        dd_cat = server.gui.add_dropdown("Category", options=cats, initial_value=cats[0])
        md_status = server.gui.add_markdown("")
        b_prev = server.gui.add_button("← Prev  [")
        b_next = server.gui.add_button("Next →  ]")

    with server.gui.add_folder("UCO3D axis definition"):
        md_axes = server.gui.add_markdown("_loading…_")

    with server.gui.add_folder("Rotate the whole category"):
        server.gui.add_markdown(
            "X=**right** (red) · Y=**top** (green) · Z=**back** (blue)\n\n"
            "A swap or a flip alone is a *reflection*; pair them to get a rotation."
        )
        b_sw_xy = server.gui.add_button("Swap X↔Y   (1)")
        b_sw_yz = server.gui.add_button("Swap Y↔Z   (2)")
        b_sw_zx = server.gui.add_button("Swap Z↔X   (3)")
        b_fl_x  = server.gui.add_button("Flip X     (4)")
        b_fl_y  = server.gui.add_button("Flip Y     (5)")
        b_fl_z  = server.gui.add_button("Flip Z     (6)")
        b_undo  = server.gui.add_button("Undo       (U)")
        b_reset = server.gui.add_button("Reset      (R)")
        txt_p   = server.gui.add_text("P (rows)", initial_value="", disabled=True)
        txt_det = server.gui.add_text("det(P)", initial_value="", disabled=True)
        btn_save = server.gui.add_button("SAVE to disk  (S)", color="red")

    with server.gui.add_folder("Frames"):
        img_frames = server.gui.add_image(np.zeros((8, 8, 3), np.uint8), label="dataset")
        img_ref = server.gui.add_image(np.zeros((8, 8, 3), np.uint8), label="reference")
        img_ref.visible = False

    def _act(fn):
        def inner(*_a):
            fn()
            state["dirty"] = not pending.is_identity
            _redraw()
        return inner

    _ops = [
        (b_sw_xy, "Swap X-Y",  "1", lambda: pending.swap(0, 1)),
        (b_sw_yz, "Swap Y-Z",  "2", lambda: pending.swap(1, 2)),
        (b_sw_zx, "Swap Z-X",  "3", lambda: pending.swap(2, 0)),
        (b_fl_x,  "Flip X",    "4", lambda: pending.flip(0)),
        (b_fl_y,  "Flip Y",    "5", lambda: pending.flip(1)),
        (b_fl_z,  "Flip Z",    "6", lambda: pending.flip(2)),
        (b_undo,  "Undo",      "U", pending.undo),
        (b_reset, "Reset",     "R", pending.reset),
    ]
    for btn, label, hotkey, fn in _ops:
        btn.on_click(_act(fn))
        # Real key bindings, via viser's command palette (CommandHandle fires
        # on_trigger, not on_click). Wrapped because the palette is flagged
        # experimental upstream: without it the buttons still work.
        try:
            server.gui.add_command(label, hotkey=hotkey).on_trigger(_act(fn))
        except Exception as exc:
            logger.debug(f"no hotkey for {label}: {exc}")

    def _do_save(*_a):
        if not pending.is_rotation:
            print(f"REFUSED: det(P) = {pending.det:+.4f}, not a rotation.", file=sys.stderr)
            _refresh_status()
            return
        if pending.is_identity:
            return
        n, where = save_category(cfg, state["category"], pending.P, G)
        print(f"  wrote {n} transforms for {state['category']} → {where}")
        pending.reset()
        pending._history.clear()
        state["dirty"] = False
        _load(state["category"])          # re-read from disk: what you see is stored

    btn_save.on_click(_do_save)
    try:
        server.gui.add_command("Save axes", hotkey="S").on_trigger(_do_save)
    except Exception as exc:
        logger.debug(f"no hotkey for save: {exc}")

    def _goto(delta: int):
        i = (cats.index(state["category"]) + delta) % len(cats)
        pending.reset(); pending._history.clear(); state["dirty"] = False
        dd_cat.value = cats[i]

    b_prev.on_click(lambda _: _goto(-1))
    b_next.on_click(lambda _: _goto(+1))
    for label, hotkey, d in (("Prev category", "arrowleft", -1),
                             ("Next category", "arrowright", +1)):
        try:
            server.gui.add_command(label, hotkey=hotkey).on_trigger(
                lambda _e, d=d: _goto(d))
        except Exception as exc:
            logger.debug(f"no hotkey for {label}: {exc}")

    @dd_cat.on_update
    def _(_e):
        if state["dirty"]:
            print(f"  discarding unsaved rotation for {state['category']}")
        pending.reset(); pending._history.clear(); state["dirty"] = False
        _load(dd_cat.value)

    _load(cats[0])
    print(f"\nViser running at http://localhost:{server.get_port()}")
    print("Keys: 1/2/3 swap · 4/5/6 flip · U undo · R reset · S save · ←/→ category")
    print("Ctrl-C to exit.\n")
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")
