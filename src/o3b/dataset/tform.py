"""
Interactive axis-convention viewer  (o3b dataset tform -d <config>).

Shows objects from the dataset one at a time in Viser with three coloured
axis arrows:

    X (red)   = Right
    Y (green) = Top
    Z (blue)  = Back

Six buttons let you modify the current axis assignment interactively:

    Switch Right-Top   — swap the Right and Top direction vectors
    Switch Top-Back    — swap the Top  and Back  direction vectors
    Switch Back-Right  — swap the Back and Right direction vectors
    Flip Right-Top     — negate both Right and Top  vectors
    Flip Top-Back      — negate both Top   and Back  vectors
    Flip Back-Right    — negate both Back  and Right vectors

The resulting obj_gl_tform4x4_obj_raw is displayed in YAML format, ready to paste
into the dataset config.
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from typing import Optional

import numpy as np


# ── tform helpers ─────────────────────────────────────────────────────────────

def _tform_from_axes(right: np.ndarray, top: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Build a 4×4 obj_gl_tform4x4_obj_raw from the three semantic direction vectors.

    Rows of T[:3,:3] are the semantic directions in object space:
        row 0 = right  (maps to canonical X)
        row 1 = top    (maps to canonical Y)
        row 2 = back   (maps to canonical Z)
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.stack([right, top, back], axis=0)
    return T


def _tform_to_yaml(T: np.ndarray) -> str:
    lines = ["# rows: right, top, back → X, Y, Z (canonical)", "obj_gl_tform4x4_obj_raw:"]
    for row in T:
        lines.append("  - [" + ", ".join(f"{v:8.5f}" for v in row) + "]")
    return "\n".join(lines)


# ── arrow mesh ────────────────────────────────────────────────────────────────

def _make_arrow(vec: np.ndarray, color_rgb: tuple, length: float) -> "trimesh.Trimesh":
    """Return a coloured cylinder+cone arrow pointing along *vec* of total *length*."""
    import trimesh
    from scipy.spatial.transform import Rotation

    shaft_r = length * 0.025
    cone_h  = length * 0.15
    shaft_h = length - cone_h

    shaft = trimesh.creation.cylinder(radius=shaft_r, height=shaft_h, sections=10)
    shaft.apply_translation([0.0, 0.0, shaft_h * 0.5])
    cone  = trimesh.creation.cone(radius=shaft_r * 3, height=cone_h, sections=10)
    cone.apply_translation([0.0, 0.0, shaft_h + cone_h * 0.5])

    arrow = trimesh.util.concatenate([shaft, cone])

    # rotate from +Z to vec
    tgt = np.array(vec, dtype=np.float64)
    tgt /= np.linalg.norm(tgt) + 1e-8
    src = np.array([0.0, 0.0, 1.0])
    cross = np.cross(src, tgt)
    cn    = float(np.linalg.norm(cross))
    if cn < 1e-6:
        R = np.eye(3) if float(np.dot(src, tgt)) > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        R = Rotation.from_rotvec(cross / cn * np.arctan2(cn, float(np.dot(src, tgt)))).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    arrow.apply_transform(T)

    rgba = np.array([int(c * 255) for c in color_rgb] + [220], dtype=np.uint8)
    arrow.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba, (len(arrow.vertices), 1))
    )
    return arrow


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_meshes(cls, cfg, limit: int) -> list[tuple[str, object]]:
    from o3b.dataset.dataset import ItemType

    view_cfg = deepcopy(cfg)
    view_cfg.item_type         = ItemType.OBJECT
    view_cfg.object_modalities = {"mesh"}
    view_cfg.obj_gl_tform4x4_obj_raw      = None   # show untransformed mesh
    view_cfg.filter_count_max  = limit

    try:
        dataset = cls(view_cfg)
    except Exception as exc:
        print(f"ERROR: could not build dataset: {exc}", file=sys.stderr)
        return []

    meshes = []
    for i in range(len(dataset)):
        try:
            obj = dataset[i]
            if obj.mesh is not None:
                meshes.append((obj.object_id, obj.mesh))
        except Exception as exc:
            print(f"  WARN: item {i}: {exc}")
    return meshes


# ── viser viewer ──────────────────────────────────────────────────────────────

def run_tform_viewer(cls, cfg, limit: int = 20) -> None:
    """Launch the interactive axis-convention viser viewer."""
    print("Loading meshes …")
    meshes = _load_meshes(cls, cfg, limit)
    if not meshes:
        print("No meshes found — nothing to display.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(meshes)} object(s) ready\n")

    try:
        import viser
    except ImportError:
        print("ERROR: pip install viser", file=sys.stderr)
        sys.exit(1)

    from o3b.io import _mesh_to_trimesh
    from o3b.data.viz import make_viser_server

    server = make_viser_server()
    server.scene.add_light_ambient("/ambient", intensity=3.0)

    # ── mutable state ──────────────────────────────────────────────────────────
    obj_idx  = [0]
    mesh_h   = [None]
    axes_len = [1.0]

    # canonical axis directions in object space (rows of T[:3,:3])
    right = [np.array([1.0, 0.0, 0.0], np.float32)]
    top   = [np.array([0.0, 1.0, 0.0], np.float32)]
    back  = [np.array([0.0, 0.0, 1.0], np.float32)]

    arrow_hs: list = []
    label_hs: list = []

    # ── helpers ────────────────────────────────────────────────────────────────

    def _remove(h) -> None:
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass

    def _refresh_scene() -> None:
        """Redraw arrows, labels, and update the YAML display."""
        for h in arrow_hs + label_hs:
            _remove(h)
        arrow_hs.clear()
        label_hs.clear()

        al = axes_len[0]
        for name, vec, color in (
            ("right", right[0], (1.0, 0.0, 0.0)),
            ("top",   top[0],   (0.0, 0.80, 0.0)),
            ("back",  back[0],  (0.2, 0.45, 1.0)),
        ):
            # arrow
            try:
                arrow = _make_arrow(vec, color, al)
                arrow_hs.append(
                    server.scene.add_mesh_trimesh(f"/axes/{name}", arrow)
                )
            except Exception as exc:
                print(f"  WARN: arrow {name}: {exc}")

            # label at arrow tip
            tip = (float(vec[0] * al), float(vec[1] * al), float(vec[2] * al))
            try:
                label_hs.append(
                    server.scene.add_label(f"/axes/label_{name}", text=name, position=tip)
                )
            except Exception:
                pass

        T   = _tform_from_axes(right[0], top[0], back[0])
        det = float(np.linalg.det(T[:3, :3]))
        txt_det.value   = (f"det = {det:+.0f}  ⚠ negative — reflection!" if det < 0
                           else f"det = {det:+.0f}  ✓ proper rotation")
        txt_tform.value = _tform_to_yaml(T)

    def _load_obj(i: int) -> None:
        _remove(mesh_h[0])
        oid, mesh = meshes[i]
        v = mesh.verts.float().cpu().numpy()
        axes_len[0] = float((v.max(axis=0) - v.min(axis=0)).max()) * 0.8
        try:
            mesh_h[0] = server.scene.add_mesh_trimesh("/object", _mesh_to_trimesh(mesh))
        except Exception as exc:
            print(f"  WARN: {oid}: {exc}")
            mesh_h[0] = None
        lbl_obj.value = oid

    # ── button actions ─────────────────────────────────────────────────────────

    def _switch(a: list, b: list) -> None:
        a[0], b[0] = b[0].copy(), a[0].copy()
        _refresh_scene()

    def _flip(a: list) -> None:
        a[0] = -a[0]
        _refresh_scene()

    # ── GUI ────────────────────────────────────────────────────────────────────

    with server.gui.add_folder("Navigation"):
        lbl_obj = server.gui.add_text("object", initial_value=meshes[0][0])
        b_prev  = server.gui.add_button("← Prev")
        b_next  = server.gui.add_button("Next →")

    with server.gui.add_folder(
        "Axis assignment  —  X = Right (red)  ·  Y = Top (green)  ·  Z = Back (blue)"
    ):
        b_sw_rt  = server.gui.add_button("Switch Right ↔ Top")
        b_sw_tb  = server.gui.add_button("Switch Top   ↔ Back")
        b_sw_br  = server.gui.add_button("Switch Back  ↔ Right")
        b_fl_r   = server.gui.add_button("Flip Right")
        b_fl_t   = server.gui.add_button("Flip Top")
        b_fl_b   = server.gui.add_button("Flip Back")
        txt_det   = server.gui.add_text("determinant", initial_value="")
        txt_tform = server.gui.add_text("obj_gl_tform4x4_obj_raw (YAML)", initial_value="")

    @b_prev.on_click
    def _(_):
        obj_idx[0] = (obj_idx[0] - 1) % len(meshes)
        _load_obj(obj_idx[0])
        _refresh_scene()

    @b_next.on_click
    def _(_):
        obj_idx[0] = (obj_idx[0] + 1) % len(meshes)
        _load_obj(obj_idx[0])
        _refresh_scene()

    @b_sw_rt.on_click
    def _(_): _switch(right, top)

    @b_sw_tb.on_click
    def _(_): _switch(top, back)

    @b_sw_br.on_click
    def _(_): _switch(back, right)

    @b_fl_r.on_click
    def _(_): _flip(right)

    @b_fl_t.on_click
    def _(_): _flip(top)

    @b_fl_b.on_click
    def _(_): _flip(back)

    # initial draw
    _load_obj(0)
    _refresh_scene()

    print(f"Viser:  http://localhost:{server.get_port()}")
    print("Use the side panel to adjust axes, then copy obj_gl_tform4x4_obj_raw into your config.")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")
