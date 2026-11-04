"""Annotate oriented 3-D boxes on a sample, blind to the official labels.

    o3b dataset annotate-obb -d uco3d

A viser page: the object's coloured point cloud (the mesh when a pack has none)
with a box you drag in 3-D (the centre gizmo moves and turns it, one handle per
face sizes it, keys swap / flip / tilt it), each axis arrow named right / top /
back, and real frames beside it with the box projected live, because the images
are the evidence and the 3-D view only repeats the reconstruction.  Every save lands on disk at once, so a session resumes where
it stopped.

The official box is not loaded until asked for, and only for an object already
saved.  The object's split is not shown and the two samples are shuffled together,
so expectations about a split cannot reach the labels.  The start box is the
mesh's principal axes, which knows nothing of the official pose.
"""
from __future__ import annotations

import getpass
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from o3b.dataset.verify_labels import store
from o3b.dataset.verify_labels.pack import default_dir, load_object
from o3b.dataset.verify_labels.sample import SEED

REPEAT_COUNT = 10          # per sample, redone at the end for the intra-annotator floor
FRAMES_SHOWN = 4
PANEL_W = 340
COMPARE_MIN_N = 10        # below this a cohort's totals would give single objects away
_AXIS_RGB = [(230, 60, 60), (60, 200, 60), (70, 130, 255)]   # X right, Y top, Z back
AXIS_NAMES = ("right", "top", "back")


def pca_box(verts: np.ndarray) -> tuple:
    """(R, center, extents): the box along the mesh's principal axes, as a start."""
    v = np.asarray(verts, dtype=np.float64)
    _, vecs = np.linalg.eigh(np.cov(v.T))
    R = vecs[:, ::-1].T
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return (R, *fit_to(v, R))


def fit_to(verts: np.ndarray, R: np.ndarray) -> tuple:
    """(center, extents) of the tightest box around ``verts`` under rotation R."""
    local = np.asarray(verts, dtype=np.float64) @ R.T
    lo, hi = local.min(0), local.max(0)
    return ((lo + hi) / 2) @ R, hi - lo


class BoxState:
    """The box being edited, with undo. No GUI in here, so tests can drive it."""

    def __init__(self, R, center, extents):
        self.R = np.asarray(R, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.extents = np.asarray(extents, dtype=np.float64)
        self.undo_stack: list = []

    def snapshot(self) -> None:
        self.undo_stack.append((self.R.copy(), self.center.copy(), self.extents.copy()))
        del self.undo_stack[:-200]

    def undo(self) -> None:
        if self.undo_stack:
            self.R, self.center, self.extents = self.undo_stack.pop()

    def relabel(self, Q: np.ndarray) -> None:
        """Change the axis *labels* by a signed permutation; the box stays put.

        A lone swap or flip mirrors the frame. That is allowed in passing, so two
        of them can pair up, and ``is_proper`` is what a save checks.
        """
        Q = np.asarray(Q, dtype=np.float64)
        self.R = Q @ self.R
        self.extents = np.abs(Q) @ self.extents

    def rotate(self, axis: int, degrees: float) -> None:
        """Turn the box about its own axis through its centre, right-handed."""
        a, b = (axis + 1) % 3, (axis + 2) % 3
        c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
        Q = np.eye(3)
        Q[a, a], Q[a, b], Q[b, a], Q[b, b] = c, s, -s, c
        self.R = Q @ self.R

    @property
    def is_proper(self) -> bool:
        return float(np.linalg.det(self.R)) > 0

    def display_R(self) -> np.ndarray:
        """R as a rotation for the gizmos, which cannot show a mirrored frame."""
        R = self.R.copy()
        if not self.is_proper:
            R[2] *= -1
        return R

    def move_face(self, axis: int, sign: int, face_world: np.ndarray) -> None:
        """Put one face where ``face_world`` projects onto its axis; the opposite stays."""
        now = self.center + sign * self.extents[axis] / 2 * self.R[axis]
        d = float((np.asarray(face_world) - now) @ self.R[axis]) * sign
        d = max(d, 1e-3 * float(self.extents.max()) - self.extents[axis])
        self.extents[axis] += d
        self.center = self.center + sign * d / 2 * self.R[axis]

    def set_pose(self, R, center) -> None:
        """Take a pose from the gizmo, keeping a pending mirror."""
        mirrored = not self.is_proper
        self.R, self.center = store.proper_rotation(R), np.asarray(center, dtype=np.float64)
        if mirrored:
            self.R[2] *= -1

    def faces(self) -> list:
        """[(axis, sign, world position)] for the six face centres."""
        return [(a, s, self.center + s * self.extents[a] / 2 * self.R[a])
                for a in range(3) for s in (1, -1)]

    def record(self) -> dict:
        return {"R_obj_raw": self.R.tolist(), "center_raw": self.center.tolist(),
                "extents_raw": self.extents.tolist()}


def queue_of(samples: dict, seed: int = SEED, repeats: int = REPEAT_COUNT) -> tuple:
    """(main, repeat): the samples shuffled together, and what is redone at the end."""
    rng = np.random.default_rng(seed)
    everything = sorted(i for ids in samples.values() for i in ids)
    main = [everything[int(i)] for i in rng.permutation(len(everything))]
    repeat = []
    for name in sorted(samples):
        ids = sorted(samples[name])
        repeat += [ids[int(i)] for i in rng.choice(len(ids), size=min(repeats, len(ids)), replace=False)]
    return main, [repeat[int(i)] for i in rng.permutation(len(repeat))]


def draw_box(bgr: np.ndarray, K, M, R, center, extents, colors=None, thickness: int = 2) -> np.ndarray:
    """Project a box onto an image: one colour per axis, plus the triad at its centre."""
    import cv2

    pts = np.concatenate([store.corners(R, center, extents),
                          np.asarray(center)[None],
                          np.asarray(center) + (np.asarray(extents)[:, None] / 2 * np.asarray(R))])
    cam = pts @ M[:3, :3].T + M[:3, 3]
    z = cam[:, 2]
    uv = (cam[:, :2] / np.maximum(z, 1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
    uv = np.clip(uv, -1e4, 1e4).astype(int)
    out = bgr.copy()

    def line(i, j, col, width):
        if z[i] > 1e-6 and z[j] > 1e-6:
            cv2.line(out, tuple(map(int, uv[i])), tuple(map(int, uv[j])), col[::-1], width, cv2.LINE_AA)

    for (i, j), axis in zip(store.BOX_EDGES, store.BOX_EDGE_AXIS):
        line(i, j, colors or _AXIS_RGB[axis], thickness)
    if colors is None:
        for axis in range(3):
            line(8, 9 + axis, _AXIS_RGB[axis], thickness + 1)
            if z[9 + axis] > 1e-6 and z[8] > 1e-6:
                # Beyond the tip, on the far side from the centre, so the arrow never crosses it.
                text = f"{'XYZ'[axis]} {AXIS_NAMES[axis]}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                d = (uv[9 + axis] - uv[8]).astype(float)
                d /= max(np.linalg.norm(d), 1e-6)
                x, y = uv[9 + axis] + 8 * d
                x = x - (tw if d[0] < 0 else 0) - tw / 2 * (1 - abs(d[0]))
                y = y + (th if d[1] > 0 else 0) + th / 2 * (1 - abs(d[1]))
                org = (int(np.clip(x, 2, out.shape[1] - tw - 2)), int(np.clip(y, th + 2, out.shape[0] - 4)))
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, _AXIS_RGB[axis][::-1], 2,
                            cv2.LINE_AA)
    return out


def crop_around(verts, K, M, shape, margin: float = 0.25) -> tuple:
    """Square (x0, y0, side) around the projected mesh, inside the image."""
    cam = verts @ M[:3, :3].T + M[:3, 3]
    cam = cam[cam[:, 2] > 1e-6]
    h, w = shape[:2]
    if len(cam) == 0:
        return 0, 0, min(h, w)
    uv = (cam[:, :2] / cam[:, 2:]) @ K[:2, :2].T + K[:2, 2]
    lo, hi = uv.min(0), uv.max(0)
    side = int(min(max(hi - lo) * (1 + 2 * margin), min(h, w)))
    side = max(side, 32)
    c = (lo + hi) / 2
    return (int(np.clip(c[0] - side / 2, 0, w - side)),
            int(np.clip(c[1] - side / 2, 0, h - side)), side)


class Annotator:
    """Queue, progress and persistence, separate from the page that drives them."""

    def __init__(self, root, samples: dict, pack_root=None, annotator: Optional[str] = None,
                 seed: int = SEED, repeats: int = REPEAT_COUNT, include_old: bool = False):
        self.root = Path(root)
        # Objects with an old label and box are done already; only the rest is queued.
        old = {r["sequence_name"] for r in store.iter_all(root, store.OLD) if r.get("center_raw")}
        # With include_old they are queued too, after the rest, starting from the
        # blind box like any other; the new annotation then replaces the old label.
        self.all_samples = samples
        self.samples = {name: set(ids) - old for name, ids in samples.items()}
        self.pack_root = Path(pack_root) if pack_root else default_dir()
        self.annotator = annotator or getpass.getuser()
        self.main, self.repeat = queue_of(self.samples, seed, repeats)
        if include_old:
            rest = sorted(set().union(*samples.values()) - set(self.main))
            rng = np.random.default_rng(seed + 1)
            self.main += [rest[int(i)] for i in rng.permutation(len(rest))]
            self.samples = samples
        self.index = 0
        self.seq: Optional[str] = None
        self.data: Optional[dict] = None
        self.box: Optional[BoxState] = None
        self.started = time.monotonic()
        self.seconds_before = 0.0
        self.go_to_first_open()

    @property
    def items(self) -> list:
        return [(s, store.ANNOTATIONS) for s in self.main] + [(s, store.REPEATS) for s in self.repeat]

    def is_done(self, seq: str, kind: str) -> bool:
        return store.path_of(self.root, seq, kind).exists()

    def progress(self) -> dict:
        left = {}
        for name, ids in self.samples.items():
            left[name] = sum(not self.is_done(s, store.ANNOTATIONS) for s in ids)
        rep = sum(not self.is_done(s, store.REPEATS) for s in self.repeat)
        return {"left": sum(left.values()), "total": len(self.main), "by_sample": left,
                "repeat_left": rep, "repeat_total": len(self.repeat)}

    def go_to_first_open(self) -> None:
        for i, (seq, kind) in enumerate(self.items):
            if not self.is_done(seq, kind):
                self.load(i)
                return
        self.load(len(self.items) - 1)

    def load(self, index: int) -> None:
        """Open one object: its saved box when it has one, else the blind start."""
        self.index = int(np.clip(index, 0, len(self.items) - 1))
        self.seq, self.kind = self.items[self.index]
        self.data = load_object(self.pack_root, self.seq)
        saved = store.load(self.root, self.seq, self.kind)
        if saved and saved.get("status") == "done":
            self.box = BoxState(saved["R_obj_raw"], saved["center_raw"], saved["extents_raw"])
        else:
            self.box = BoxState(*pca_box(self.data["verts"]))
        self.seconds_before = float(saved.get("seconds_spent", 0.0)) if saved else 0.0
        self.started = time.monotonic()

    def _common(self) -> dict:
        split = next((n for n, ids in self.all_samples.items() if self.seq in ids), None)
        return {"sequence_name": self.seq, "split": split, "annotator": self.annotator,
                "frame_convention": "o3b obj axes (X right, Y top, Z back), raw SfM frame",
                "seconds_spent": self.seconds_before + time.monotonic() - self.started}

    def save(self, status: str = "done", reason: str = "") -> dict:
        if status == "done" and not self.box.is_proper:
            raise ValueError("the axes are mirrored: pair this up with another swap or flip")
        record = {**self._common(), "status": status, "reason": reason,
                  **(self.box.record() if status == "done" else
                     {"R_obj_raw": None, "center_raw": None, "extents_raw": None})}
        return store.save(self.root, record, self.kind)

    def saved(self) -> Optional[dict]:
        return store.load(self.root, self.seq, self.kind)


def label_ops() -> dict:
    """Button label -> axis relabelling. Each one alone mirrors the frame."""
    ops = {}
    for i, j, name in ((0, 1, "XY"), (1, 2, "YZ"), (2, 0, "ZX")):
        Q = np.eye(3)
        Q[[i, j]] = Q[[j, i]]
        ops[f"Swap {name}"] = Q
    for i, name in enumerate("XYZ"):
        Q = np.eye(3)
        Q[i, i] = -1
        ops[f"Flip {name}"] = Q
    return ops


# One keyboard column per view, top to bottom: look along an axis, tilt about it,
# swap the two axes a quarter turn about it exchanges. Front looks along Z, top
# along Y, right along X. Q/A W/S E/D are viser's camera keys too: key_script
# takes them first, so the camera flies on the arrows alone.
KEYS = {"1": "Front", "q": "Z +1", "a": "Z -1", "z": "Swap XY",
        "2": "Top", "w": "Y +1", "s": "Y -1", "x": "Swap ZX",
        "3": "Right", "e": "X +1", "d": "X -1", "c": "Swap YZ",
        "4": "Flip X", "5": "Flip Y", "6": "Flip Z"}


# View name -> ((axis, sign) the camera sits on, (axis, sign) that points up on screen).
# Z is back, so the front is seen from -Z; the top view keeps the back up, so it
# continues the front view tipped over the top (X reads the same way in both).
VIEWS = {"Front": ((2, -1), (1, 1)), "Top": ((1, 1), (2, 1)), "Right": ((0, 1), (1, 1))}


def frames_up(data: dict) -> np.ndarray:
    """Camera up in the raw frame, averaged over the frames."""
    up = -(data["M"][:, 1, :3].mean(0))
    return up / max(np.linalg.norm(up), 1e-9)


def key_script() -> str:
    """HTML that makes each key press its button; viser has no key events of its own."""
    table = ",".join(f"'{k}':'{v}'" for k, v in KEYS.items())
    js = ("if(!window.__obbKeys){window.__obbKeys=1;var m={" + table + "};"
          "window.addEventListener('keydown',function(e){"
          "var t=e.target.tagName;if(t==='INPUT'||t==='TEXTAREA'||e.ctrlKey||e.metaKey||e.altKey)return;"
          "var l=m[e.key.toLowerCase()];if(!l)return;"
          "var b=[].slice.call(document.querySelectorAll('button')).filter(function(x){return x.innerText.trim()===l})[0];"
          "if(b){b.click();e.preventDefault();e.stopImmediatePropagation()}},true)}")
    return f'<img src="x" style="display:none" onerror="{js}">'


def run_viewer(cfg, root, pack_root=None, port: Optional[int] = None, annotator: Optional[str] = None,
               official: str = "every9d_v7", include_old: bool = False) -> None:
    """Serve the page until interrupted."""
    import cv2
    import viser.transforms as vtf

    from o3b.data.viz import make_viser_server
    from o3b.dataset import tform_obj_store
    from o3b.dataset.axes_tform_obj import axis_rules_lines
    from o3b.dataset.verify_labels.compare import _sample_ids, official_box, run_compare, score

    samples = _sample_ids(root)
    if not samples:
        raise SystemExit(f"no samples under {root}/samples: run annotate-obb-sample first")
    app = Annotator(root, samples, pack_root, annotator, include_old=include_old)
    server = make_viser_server(**({"port": port} if port else {}))

    lock = threading.RLock()
    dirty = threading.Event()
    state = {"page": 0, "official": None, "images": [], "gizmo": None, "faces": []}
    official_table = {}

    def tforms():
        if not official_table:
            table = tform_obj_store.load_all(cfg.path_preprocess, official)
            if table is None:
                raise FileNotFoundError(
                    f"no tform_obj/{official}.db under {cfg.path_preprocess}: is it mounted?")
            official_table.update(table)
        return official_table

    server.gui.configure_theme(control_width="large")
    with server.gui.add_folder("Progress"):
        md_progress = server.gui.add_markdown("")
    with server.gui.add_folder("Save"):
        md_warn = server.gui.add_markdown("")
        btn_save = server.gui.add_button("Save & next", color="green")
        btn_prev = server.gui.add_button("Previous")
        btn_skip = server.gui.add_button("Skip for now")
        num_item = server.gui.add_number("Item", 1, min=1, max=len(app.items), step=1,
                                         hint="Saved objects can be opened and saved again; "
                                              "earlier versions are kept.")
        btn_go = server.gui.add_button("Go to item")
        txt_reason = server.gui.add_text("Reason", "")
        btn_cannot = server.gui.add_button("Cannot annotate", color="red")
    with server.gui.add_folder("Object"):
        md_object = server.gui.add_markdown("")
    with server.gui.add_folder("Frames"):
        images = [server.gui.add_image(np.zeros((8, 8, 3), np.uint8), format="jpeg")
                  for _ in range(FRAMES_SHOWN)]
        btn_page = server.gui.add_button("Other frames")
    with server.gui.add_folder("Box"):
        btn_fit = server.gui.add_button("Fit box to object", hint="Keeps the rotation; fits the alpha500 mesh, as the official box does")
        btn_undo = server.gui.add_button("Undo")
        btn_reset = server.gui.add_button("Reset to start box")
        server.gui.add_markdown(
            "Keys: **1** front, **Q/A** tilt ±1° about it, **Z** swap XY; "
            "**2** top, **W/S**, **X** swap ZX; **3** right, **E/D**, **C** swap YZ; "
            "**4 5 6** flip X Y Z. The camera moves on the arrows.")
        ops = label_ops()
        def relabel(ev):
            with lock:
                app.box.snapshot()
                app.box.relabel(ops[ev.target.value])
                sync()
        def rotate(ev):
            axis, degrees = ev.target.value.split()
            with lock:
                app.box.snapshot()
                app.box.rotate("XYZ".index(axis), float(degrees))
                sync()
        def view(ev):
            clients = [ev.client] if ev.client is not None else server.get_clients().values()
            for client in clients:
                view_along_box(client, ev.target.value)
        server.gui.add_button_group("View", list(VIEWS)).on_click(view)
        for axis, view_name in (("Z", "front"), ("Y", "top"), ("X", "right")):
            server.gui.add_button_group(f"Turn {axis} ({view_name})",
                                        [f"{axis} +1", f"{axis} -1"]).on_click(rotate)
        server.gui.add_button_group("Swap", ["Swap XY", "Swap ZX", "Swap YZ"]).on_click(relabel)
        server.gui.add_button_group("Flip", ["Flip X", "Flip Y", "Flip Z"]).on_click(relabel)
        server.gui.add_html(key_script())
    with server.gui.add_folder("Compare with Every9D"):
        btn_compare = server.gui.add_button("Compare", color="blue",
                                            hint="Scores every saved annotation; shows totals only")
        chk_official = server.gui.add_checkbox("Show official box (this object)", False, disabled=True,
                                               hint="Only on a saved object. Marks it as seen for good.")
        md_compare = server.gui.add_markdown("")

    def faces_and_wire():
        box = app.box
        wxyz = vtf.SO3.from_matrix(box.display_R().T).wxyz
        c = store.corners(box.R, box.center, box.extents)
        pts = np.array([[c[i], c[j]] for i, j in store.BOX_EDGES], dtype=np.float32)
        cols = np.array([[_AXIS_RGB[a]] * 2 for a in store.BOX_EDGE_AXIS], dtype=np.uint8)
        server.scene.add_line_segments("/box", pts, cols, line_width=3)
        tips = box.center + (box.extents[:, None] / 2 + 0.25 * box.extents.max()) * box.R
        server.scene.add_line_segments(
            "/axes", np.stack([np.broadcast_to(box.center, (3, 3)), tips], 1).astype(np.float32),
            np.array([[c] * 2 for c in _AXIS_RGB], dtype=np.uint8), line_width=6)
        for axis in range(3):
            server.scene.add_label(f"/axis_name_{axis}", f"{'XYZ'[axis]} {AXIS_NAMES[axis]}",
                                   position=tuple(tips[axis]), anchor="center-left")
        return wxyz

    def sync(skip_handle=None):
        """Scene and panels follow the box. The handle being dragged is left alone."""
        with lock:
            box = app.box
            wxyz = faces_and_wire()
            if state["gizmo"] is not None and state["gizmo"] is not skip_handle:
                state["gizmo"].position = box.center
                state["gizmo"].wxyz = wxyz
            for handle, (_, _, pos) in zip(state["faces"], box.faces()):
                if handle is not skip_handle:
                    handle.position = pos
                    handle.wxyz = wxyz
            md_warn.content = "" if box.is_proper else ("**Axes are mirrored.** Another swap or "
                                                        "flip is needed before saving.")
            dirty.set()

    def build_handles():
        box = app.box
        size = float(box.extents.max())
        wxyz = vtf.SO3.from_matrix(box.display_R().T).wxyz
        for h in state["faces"] + ([state["gizmo"]] if state["gizmo"] else []):
            h.remove()
        gizmo = server.scene.add_transform_controls(
            "/gizmo", scale=size * 0.5, disable_sliders=True, position=box.center, wxyz=wxyz)
        state["gizmo"] = gizmo

        @gizmo.on_drag_start
        def _(_ev):
            app.box.snapshot()

        @gizmo.on_update
        def _(_ev):
            with lock:
                app.box.set_pose(vtf.SO3(np.asarray(gizmo.wxyz)).as_matrix().T, gizmo.position)
                sync(skip_handle=gizmo)

        faces = []
        for axis, sign, pos in box.faces():
            active = tuple(i == axis for i in range(3))
            h = server.scene.add_transform_controls(
                f"/face_{axis}_{'p' if sign > 0 else 'n'}", scale=size * 0.2, active_axes=active,
                disable_rotations=True, disable_sliders=True, position=pos, wxyz=wxyz)

            def bind(h, axis, sign):
                @h.on_drag_start
                def _(_ev):
                    app.box.snapshot()

                @h.on_update
                def _(_ev):
                    with lock:
                        app.box.move_face(axis, sign, np.asarray(h.position))
                        sync(skip_handle=h)

                @h.on_drag_end
                def _(_ev):
                    sync()
            bind(h, axis, sign)
            faces.append(h)
        state["faces"] = faces

    def refresh_official():
        """Draw the official box of a saved object, when it is asked for."""
        saved = app.saved()
        allowed = bool(saved)
        chk_official.disabled = not allowed
        if not allowed and chk_official.value:
            chk_official.value = False
        state["official"] = None
        server.scene.add_line_segments("/official", np.zeros((1, 2, 3), np.float32),
                                       (255, 200, 0), visible=False)
        if allowed and chk_official.value:
            seq = app.seq
            try:
                off = official_box(tforms()[seq], app.data["verts"])
            except (KeyError, FileNotFoundError) as e:
                md_compare.content = f"Cannot read the official box: {e}"
                return
            store.mark_revealed(app.root, seq, app.kind)
            state["official"] = off
            c = store.corners(off["R_obj_raw"], off["center_raw"], off["extents_raw"])
            pts = np.array([[c[i], c[j]] for i, j in store.BOX_EDGES], dtype=np.float32)
            server.scene.add_line_segments("/official", pts, (255, 200, 0), line_width=4)
            if saved.get("status") == "done":
                s = score(saved, off, seq.split("/")[0])
                md_compare.content = (f"**This object:** {s['rot_sym_deg']:.1f}° sym ({s['rot_deg']:.1f}° naive), "
                                      f"IoU {s['iou_sym']:.3f}, centre {s['center_pct_diag']:.1f} % of diagonal.")
        dirty.set()

    def show_object():
        with lock:
            data, seq = app.data, app.seq
            state["page"] = 0
            state["crops"] = {}
            if "pcl_xyz" in data:
                xyz = data["pcl_xyz"]
                diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
                server.scene.add_point_cloud("/object", xyz, data["pcl_rgb"], point_size=diag * 0.004,
                                             point_shape="circle", precision="float32",
                                             point_shading="flat")
            else:
                server.scene.add_mesh_simple("/object", data["verts"], data["faces"], color=(190, 190, 190),
                                             opacity=0.75, side="double", flat_shading=True)
            # Camera up in the raw frame, read off the frames, so the object stands upright.
            server.scene.set_up_direction(tuple(frames_up(data)))
            build_handles()
            sync()
            p = app.progress()
            by = ", ".join(f"{n.split('_verify')[0].replace('every9d_v7_', '')} {v}"
                           for n, v in sorted(p["by_sample"].items()))
            extra = "" if p["left"] else (
                f"\n\nRepeat round (floor): **{p['repeat_left']} of {p['repeat_total']} left**.")
            md_progress.content = f"**{p['left']} of {p['total']} left** ({by}).{extra}"
            rules = "  \n".join(t for t, _, _ in axis_rules_lines(cfg, seq.split("/")[0]))
            saved = app.saved()
            status = "no annotation yet" if not saved else f"saved: {saved.get('status')}"
            md_object.content = (f"**{seq}**  \nItem {app.index + 1} of {len(app.items)} — {status}.  \n"
                                 f"Red = X right, green = Y top, blue = Z back: wrong ones are "
                                 f"fixed with the keys below (view, flip, swap, ±1° tilt).\n\n{rules}")
            txt_reason.value = (saved or {}).get("reason", "") or ""
            num_item.value = app.index + 1
            refresh_official()
            for client in server.get_clients().values():
                frame_to_camera(client)

    def frame_to_camera(client):
        data, box = app.data, app.box
        M = data["M"][0]
        with client.atomic():
            client.camera.up_direction = tuple(frames_up(data))
            client.camera.position = -M[:3, :3].T @ M[:3, 3]
            client.camera.look_at = box.center

    def view_along_box(client, name):
        """Look at the box from one of its own sides, with its top (or back) up."""
        with lock:
            R, center, extents = app.box.R.copy(), app.box.center.copy(), app.box.extents.copy()
        side, up = VIEWS[name]
        fov = float(getattr(client.camera, "fov", 0.0) or np.radians(50))
        dist = 0.7 * float(np.linalg.norm(extents)) / np.tan(fov / 2)
        with client.atomic():
            client.camera.up_direction = tuple(R[up[0]] * up[1])
            client.camera.position = center + dist * side[1] * R[side[0]]
            client.camera.look_at = center

    def redraw():
        """Frame panels, off the callback threads: a drag fires far faster than this runs."""
        while True:
            dirty.wait()
            dirty.clear()
            with lock:
                data, box = app.data, (app.box.R.copy(), app.box.center.copy(), app.box.extents.copy())
                shown, page = state["official"], state["page"]
                crops = state.setdefault("crops", {})
            try:
                n = len(data["images"])
                picks = [(page + i * max(n // FRAMES_SHOWN, 1)) % n for i in range(FRAMES_SHOWN)]
                for panel, f in zip(images, picks):
                    if f not in crops:
                        crops[f] = cv2.imread(str(data["images"][f]))
                    bgr, K, M = crops[f], data["K"][f].copy(), data["M"][f]
                    # Crop first, so the axis names are kept inside what is shown.
                    x, y, side = crop_around(data["verts"], K, M, bgr.shape)
                    K[0, 2] -= x
                    K[1, 2] -= y
                    drawn = draw_box(bgr[y:y + side, x:x + side], K, M, *box)
                    if shown is not None:
                        drawn = draw_box(drawn, K, M, shown["R_obj_raw"], shown["center_raw"],
                                         shown["extents_raw"], colors=(255, 200, 0), thickness=1)
                    cell = cv2.resize(drawn, (PANEL_W, PANEL_W), interpolation=cv2.INTER_AREA)
                    panel.image = cell[..., ::-1]
            except Exception as e:      # a frame failing must not kill the thread for the session
                print(f"redraw: {e}", flush=True)
            time.sleep(0.05)

    threading.Thread(target=redraw, daemon=True).start()

    def on(button):
        return button.on_click

    @on(btn_page)
    def _(_ev):
        state["page"] += 1
        dirty.set()

    @on(btn_fit)
    def _(_ev):
        with lock:
            app.box.snapshot()
            app.box.center, app.box.extents = fit_to(app.data["verts"], app.box.R)
            sync()

    @on(btn_undo)
    def _(_ev):
        with lock:
            app.box.undo()
            sync()

    @on(btn_reset)
    def _(_ev):
        with lock:
            app.box.snapshot()
            R, c, e = pca_box(app.data["verts"])
            app.box.R, app.box.center, app.box.extents = R, c, e
            sync()

    def go(index):
        with lock:
            app.load(index)
            show_object()

    @on(btn_save)
    def _(_ev):
        with lock:
            try:
                app.save("done")
            except ValueError as e:
                md_warn.content = f"**Not saved:** {e}."
                return
            go(app.index + 1)

    @on(btn_prev)
    def _(_ev):
        go(app.index - 1)

    @on(btn_go)
    def _(_ev):
        go(int(num_item.value) - 1)

    @on(btn_skip)
    def _(_ev):
        go(app.index + 1)

    @on(btn_cannot)
    def _(_ev):
        with lock:
            if not txt_reason.value.strip():
                md_object.content += "\n\n**Enter a reason first.**"
                return
            app.save("cannot_annotate", txt_reason.value.strip())
            go(app.index + 1)

    @chk_official.on_update
    def _(_ev):
        with lock:
            refresh_official()

    @on(btn_compare)
    def _(_ev):
        md_compare.content = "Comparing…"
        try:
            db = tform_obj_store.db_path(cfg.path_preprocess, official)
            _, markdown = run_compare(app.root, tforms(), app.pack_root, db_path=db, min_n=COMPARE_MIN_N)
        except Exception as e:
            md_compare.content = f"Compare failed: {e}"
            return
        md_compare.content = markdown + f"\n\nWritten to `{app.root}/quantitative/h09_compare.csv`."

    @server.on_client_connect
    def _(client):
        frame_to_camera(client)

    show_object()
    print(f"annotating as {app.annotator}, {app.progress()['left']} objects left. Ctrl-C to stop; "
          "every save is already on disk.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
