from __future__ import annotations
import colorsys
import time
from typing import TYPE_CHECKING, Optional

from o3b.data.datatypes.object import _draw_kpts2d_on_imgs

if TYPE_CHECKING:
    from torch import Tensor
    from o3b.data.datatypes import Mesh, FrameObjectBatch


def _make_kpts_spheres(kpts_np, mask_np, radius: float = 0.02):
    try:
        import trimesh
        import numpy as np
    except ImportError:
        return None

    template = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    n_total = len(kpts_np)
    meshes = []
    for i in range(n_total):
        if not mask_np[i]:
            continue
        r, g, b = colorsys.hsv_to_rgb(i / max(n_total, 1), 0.9, 0.88)
        color = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        s = template.copy()
        s.apply_translation(kpts_np[i])
        s.visual.vertex_colors = np.tile(color, (len(s.vertices), 1))
        meshes.append(s)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _make_kpts_spheres_indexed(pts_np, idx_np, mask_np, n_total: int, radius: float = 0.012):
    """Like _make_kpts_spheres, but each sphere's color is keyed off an explicit
    per-point keypoint index (idx_np) rather than its position in pts_np, and
    n_total sets the hue denominator — so colors match the primary keypoint of
    the same index. Spheres are translucent to distinguish them from the solid
    primary keypoints (used for symmetric keypoint candidates)."""
    try:
        import trimesh
        import numpy as np
    except ImportError:
        return None

    template = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    meshes = []
    for pt, idx, keep in zip(pts_np, idx_np, mask_np):
        if not keep:
            continue
        r, g, b = colorsys.hsv_to_rgb(idx / max(n_total, 1), 0.9, 0.88)
        color = np.array([int(r * 255), int(g * 255), int(b * 255), 140], dtype=np.uint8)
        s = template.copy()
        s.apply_translation(pt)
        s.visual.vertex_colors = np.tile(color, (len(s.vertices), 1))
        meshes.append(s)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _rot_mat_to_wxyz(R):
    """Convert 3×3 rotation matrix (numpy) to (w, x, y, z) quaternion."""
    import numpy as np
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def _add_debug_cameras(server, dist: float = 5.0) -> list:
    """Add front/top/right camera frustums with labels to a viser scene."""
    import numpy as np

    handles = []
    batch = get_front_top_right_viewpoints(dist=dist)
    cam_tform4x4_obj = batch.cam_tform4x4_obj.numpy()  # (3, 4, 4)

    names  = ["front",          "top",            "right"]
    colors = [(255, 200,   0),  (0,  200, 255),   (255,   0, 200)]

    # Viser renders the frustum opening along the frustum's local +Z.
    # Our camera +Z is "back" (away from object); flip it with a 180° Y-rotation
    # so the frustum opening points toward the object (camera forward = -Z_cam).
    flip_y = np.diag(np.array([-1., 1., -1.]))   # diag(-1,1,-1): 180° around Y

    for i, (name, color) in enumerate(zip(names, colors)):
        tform = cam_tform4x4_obj[i]              # (4,4) cam←obj
        obj_tform_cam = np.linalg.inv(tform)
        pos = obj_tform_cam[:3, 3]
        # rotate so frustum +Z aligns with camera forward (toward object)
        R_viser = obj_tform_cam[:3, :3] @ flip_y
        wxyz = _rot_mat_to_wxyz(R_viser)

        try:
            h = server.scene.add_camera_frustum(
                f"/debug_cameras/{name}",
                fov=0.5,
                aspect=1.0,
                scale=0.4,
                wxyz=wxyz,
                position=tuple(pos.tolist()),
                color=color,
            )
            handles.append(h)
        except Exception:
            pass

        try:
            h2 = server.scene.add_label(
                f"/debug_cameras/{name}/label",
                text=name,
                position=(0.0, 0.0, 0.0),
            )
            handles.append(h2)
        except Exception:
            pass

    return handles


def _add_canonical_axes(server, axes_length: float = 1.6) -> None:
    """Add a permanent canonical-axis frame and labels to a viser server.

    Uses viser's built-in add_frame (one scene node, no trimesh computation):
        X (red)   = Right
        Y (green) = Top
        Z (blue)  = Back

    axes_length=1.6 (0.8 × max extent 2 for objects normalised to [-1,1]³).
    """
    al = axes_length
    try:
        server.scene.add_frame(
            "/canonical_axes",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=al,
            axes_radius=al * 0.025,
            origin_radius=al * 0.04,
        )
    except Exception:
        return

    for tip, name in (
        ((al,  0.0, 0.0), "right"),
        ((0.0, al,  0.0), "top"),
        ((0.0, 0.0, al),  "back"),
    ):
        try:
            server.scene.add_label(
                f"/canonical_axes/label_{name}", text=name, position=tip
            )
        except Exception:
            pass


def make_viser_server(**kwargs):
    """Create a viser server, honouring the O3B_VISER_PORT environment variable.

    `o3b dataset viz --port N` (and `--remote`, which tunnels a port from the
    compute node) set O3B_VISER_PORT so the server binds a *known* port. viser
    silently increments the port when it is already taken, which would break a
    tunnel pointing at the requested one — so bind-test first and fail loudly.
    """
    import os
    import socket
    import viser

    port = os.environ.get("O3B_VISER_PORT")
    if port and "port" not in kwargs:
        port = int(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(
                    f"Requested viser port {port} is already in use on this host "
                    f"({exc}). Re-run with a different --port."
                ) from exc
        kwargs["port"] = port
    return viser.ViserServer(**kwargs)


def visualize_dataset(
    dataset,
    render: bool = False,
    render_frames: int = 4,
    renderer: str = "pyrender",
    debug: bool = False,
) -> None:
    """Browse a dataset interactively with Prev / Next navigation via viser.

    render=False: passes the viser server to item.viz(server=server) so items
    add their 3D content (mesh, point cloud, keypoints) directly.
    render=True: additionally renders render_frames viewpoints with the chosen
    renderer and displays the image strips in the viser GUI sidebar with a
    modality dropdown (rgb / depth / normals / rgb_kpts).
    """
    try:
        import viser
    except ImportError:
        print("Install viser: pip install viser")
        return

    import numpy as np
    import torch

    server = make_viser_server()
    server.scene.add_light_ambient("/ambient", intensity=3.0)
    _add_canonical_axes(server)
    if debug:
        _add_debug_cameras(server)

    n = len(dataset)
    idx       = [0]
    handles:  list = []
    # render mode: cache per item index
    # entry: {"strips": {name: uint8 HxNWx3}, "modalities": {name: (N,C,H,W)},
    #         "cam_tform4x4_obj": (N,4,4), "cam_intr4x4": (4,4), "H": int, "W": int}
    _render_cache: dict = {}
    _img_handle = [None]      # GuiImageHandle or None
    _mod_dd     = [None]      # GuiDropdown handle or None

    def _clear() -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _compute_render(i: int) -> "dict | None":
        if i in _render_cache:
            return _render_cache[i]
        item = dataset[i]
        mesh = getattr(item, "mesh", None)
        if mesh is None:
            _render_cache[i] = None
            return None
        batch_vp = _get_render_viewpoints(render_frames, mesh=mesh)
        from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size
        H, W = 512, 512
        _intr = get_default_camera_intrinsics_from_img_size(H=H, W=W)
        batch_vp.cam_intr4x4 = _intr.unsqueeze(0).expand(
            batch_vp.cam_tform4x4_obj.shape[0], -1, -1
        )
        modalities = render_mesh_from_viewpoints(batch_vp, renderer=renderer, H=H, W=W)
        if modalities is None:
            _render_cache[i] = None
            return None

        # keypoint overlay on rgb
        kpts3d    = getattr(item, "obj_kpts3d",      None)
        kpts_mask = getattr(item, "obj_kpts3d_mask", None)
        if kpts3d is not None and "rgb" in modalities:
            from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast
            _H, _W = modalities["rgb"].shape[-2], modalities["rgb"].shape[-1]
            _cam_t = batch_vp.cam_tform4x4_obj.float()
            _N     = _cam_t.shape[0]
            _intr_n = _intr.unsqueeze(0).expand(_N, -1, -1).float()
            _kpts = (kpts3d.float() if isinstance(kpts3d, torch.Tensor)
                     else torch.tensor(kpts3d, dtype=torch.float32))
            _K = _kpts.shape[0]
            _kpts2d = proj3d2d_tform4x4_intr4x4_broadcast(
                pts3d=_kpts.unsqueeze(0),
                tform4x4=_cam_t.unsqueeze(1),
                intr4x4=_intr_n.unsqueeze(1),
            )
            modalities["rgb"] = _draw_kpts2d_on_imgs(
                modalities["rgb"], _kpts2d, mask=kpts_mask,
                radius=max(_H, _W) // 50,
            )
            _valid = (
                (_kpts2d[..., 0] >= 0) & (_kpts2d[..., 0] < _W)
                & (_kpts2d[..., 1] >= 0) & (_kpts2d[..., 1] < _H)
            )
            if kpts_mask is not None:
                _m = (kpts_mask if isinstance(kpts_mask, torch.Tensor)
                      else torch.tensor(kpts_mask))
                _valid = _valid & _m.bool().unsqueeze(0)
            _r = 4
            _rgb_kpts = modalities["rgb"].clone()
            for _n in range(_N):
                for _k in range(_K):
                    if not _valid[_n, _k]:
                        continue
                    _x = int(_kpts2d[_n, _k, 0].round().item())
                    _y = int(_kpts2d[_n, _k, 1].round().item())
                    _c = torch.tensor([
                        (_k * 37  % 256) / 255.0,
                        (_k * 97  % 256) / 255.0,
                        (_k * 153 % 256) / 255.0,
                    ])
                    _y0, _y1 = max(0, _y - _r), min(_H, _y + _r + 1)
                    _x0, _x1 = max(0, _x - _r), min(_W, _x + _r + 1)
                    _rgb_kpts[_n, :, _y0:_y1, _x0:_x1] = _c[:, None, None]
            modalities["rgb_kpts"] = _rgb_kpts

        # convert to uint8 horizontal strips for the GUI sidebar
        strips = {}
        for k, v in modalities.items():
            arr = torch.cat(list(v.clamp(0, 1)), dim=2).permute(1, 2, 0).cpu().numpy()
            strips[k] = (arr * 255).astype(np.uint8)

        entry = {
            "strips":           strips,
            "modalities":       modalities,
            "cam_tform4x4_obj": batch_vp.cam_tform4x4_obj,
            "cam_intr4x4":      _intr,
            "H": H, "W": W,
        }
        _render_cache[i] = entry
        return entry

    def _load(i: int) -> None:
        _clear()
        item = dataset[i]
        item_id = getattr(
            item, "object_id",
            getattr(item, "frame_id",
                    getattr(item, "scene_id", str(i))),
        )
        category = getattr(item, "category", None)
        cat_str  = f"  cat={category}" if category is not None else ""

        result = item.viz(server=server)
        if isinstance(result, list):
            handles.extend(result)

        if render:
            rdata = _compute_render(i)
            if rdata is not None:
                strips      = rdata["strips"]
                modalities  = rdata["modalities"]
                cam_tforms  = rdata["cam_tform4x4_obj"]   # (N, 4, 4)
                cam_intr4x4 = rdata["cam_intr4x4"]        # (4, 4)
                H_r, W_r    = rdata["H"], rdata["W"]

                keys = list(strips.keys())
                # update or create the modality dropdown
                if _mod_dd[0] is None:
                    _mod_dd[0] = server.gui.add_dropdown(
                        "Modality", options=keys, initial_value=keys[0]
                    )

                    @_mod_dd[0].on_update
                    def _(_e):
                        _update_render_img()
                else:
                    _mod_dd[0].options = keys
                    if _mod_dd[0].value not in keys:
                        _mod_dd[0].value = keys[0]

                # update or create the GUI image strip
                mod = _mod_dd[0].value if _mod_dd[0].value in strips else keys[0]
                if _img_handle[0] is None:
                    _img_handle[0] = server.gui.add_image(strips[mod], label="Rendered views")
                else:
                    _img_handle[0].image = strips[mod]

                # add a camera frustum + rendered RGB panel per viewpoint
                from o3b.data.viz_viser import (
                    _add_frustum_to_scene, _add_rgb_image_to_scene, _rot3x3_to_wxyz,
                )
                N_views = cam_tforms.shape[0]
                # axis-gizmo length scaled to the object (verts normalised to ~[-1,1])
                _cam_axis_len = 0.4
                for vi in range(N_views):
                    world_tform4x4_cam = torch.linalg.inv(cam_tforms[vi].float())
                    hs = _add_frustum_to_scene(
                        server, cam_intr4x4, H_r, W_r,
                        name=f"/render/cam{vi}/frustum",
                        world_tform4x4_cam=world_tform4x4_cam,
                    )
                    handles.extend(hs)
                    # 3D axis gizmo at the camera origin (X=red, Y=green, Z=blue)
                    try:
                        _wxyz = _rot3x3_to_wxyz(world_tform4x4_cam[:3, :3].float())
                        _pos  = tuple(float(v) for v in world_tform4x4_cam[:3, 3].float().cpu())
                        handles.append(server.scene.add_frame(
                            f"/render/cam{vi}/axes",
                            wxyz=_wxyz,
                            position=_pos,
                            axes_length=_cam_axis_len,
                            axes_radius=_cam_axis_len * 0.025,
                            origin_radius=_cam_axis_len * 0.04,
                        ))
                    except Exception:
                        pass
                    rgb_vi   = modalities["rgb"][vi]              # (3, H, W)
                    depth_vi = (modalities["depth"][vi, 0]
                                if "depth" in modalities else None)  # (H, W) or None
                    h = _add_rgb_image_to_scene(
                        server, rgb_vi, cam_intr4x4, depth_vi,
                        name=f"/render/cam{vi}/rgb",
                        world_tform4x4_cam=world_tform4x4_cam,
                    )
                    if h is not None:
                        handles.append(h)

        label.value = f"[{i + 1}/{n}]  {item_id}{cat_str}"
        print(f"  [{i + 1}/{n}] {item_id}{cat_str}")

    def _update_render_img() -> None:
        if _mod_dd[0] is None or _img_handle[0] is None:
            return
        rdata = _render_cache.get(idx[0])
        if rdata is None:
            return
        mod = _mod_dd[0].value
        strips = rdata["strips"]
        if mod in strips:
            _img_handle[0].image = strips[mod]

    with server.gui.add_folder("Navigation"):
        label    = server.gui.add_text("Item",   initial_value="loading…")
        btn_prev = server.gui.add_button("← Prev")
        btn_next = server.gui.add_button("Next →")

    @btn_prev.on_click
    def _(_):
        idx[0] = (idx[0] - 1) % n
        _load(idx[0])

    @btn_next.on_click
    def _(_):
        idx[0] = (idx[0] + 1) % n
        _load(idx[0])

    _load(0)
    print(f"\nViser running at http://localhost:{server.get_port()}")
    print("Use Prev / Next in the panel to browse. Press Ctrl+C to exit.\n")
    if render:
        print("Rendered views appear in the GUI sidebar (Modality dropdown to switch).\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")

def get_front_top_right_viewpoints(dist: float = 5.0, mesh=None) -> "FrameObjectBatch":
    return _get_render_viewpoints(n_views=3, dist=dist, mesh=mesh)

def _get_render_viewpoints(n_views: int, dist: float = 5.0, mesh=None) -> "FrameObjectBatch":
    """Dispatch to the right semantic viewpoint function based on n_views.

    1  → get_front_viewpoint
    2  → get_front_top_viewpoints
    3  → get_front_top_right_viewpoints
    >3 → get_cam_tform4x4_obj_for_viewpoints_count
    """
    from o3b.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
    from o3b.data.datatypes import FrameObjectBatch
    cam = get_cam_tform4x4_obj_for_viewpoints_count(
        viewpoints_count=n_views, dist=dist
    )
    return FrameObjectBatch(cam_tform4x4_obj=cam, mesh=mesh)


def sample_uniform_viewpoints(
    n: int,
    dist: float = 5.0,
    mesh: "Optional[Mesh]" = None,
) -> "FrameObjectBatch":
    """Returns FrameObjectBatch with cam_tform4x4_obj (n, 4, 4) sampled over a sphere."""
    from o3b.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
    from o3b.data.datatypes import FrameObjectBatch
    cam_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=n, dist=dist)
    return FrameObjectBatch(cam_tform4x4_obj=cam_tform4x4_obj, mesh=mesh)


def _render_mesh_pyrender(
    mesh: "Mesh",
    cam_tform4x4_obj: "Tensor",  # (B, 4, 4)
    cam_intr4x4: "Tensor",       # (B, 4, 4)
    H: int,
    W: int,
) -> "dict[str, Tensor]":
    """Render mesh from B viewpoints using pyrender. Returns dict of (B, 3, H, W) in [0, 1]."""
    import torch
    import numpy as np
    import trimesh as _tm
    from o3b.io import _mesh_to_trimesh
    from o3b.cv.visual.show import render_trimesh_to_tensor

    mesh_tm = _mesh_to_trimesh(mesh)

    # normals mesh: vertex colors encode world-space normals mapped to [0, 1]
    vn = mesh_tm.vertex_normals.astype(np.float32)
    vc_rgba = np.concatenate(
        [((vn * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8),
         np.full((len(vn), 1), 255, np.uint8)], axis=1
    )
    normals_tm = _tm.Trimesh(vertices=mesh_tm.vertices, faces=mesh_tm.faces,
                              vertex_colors=vc_rgba, process=False)

    from o3b.cv.visual.show import OPEN3D_CAM_TFORM_CAM

    B = cam_tform4x4_obj.shape[0]
    rgb_frames, depth_frames, normal_frames = [], [], []
    for b in range(B):
        intr, tform = cam_intr4x4[b], cam_tform4x4_obj[b]
        # following OpenGL
        tform_od = tform
        rgb, depth = render_trimesh_to_tensor(mesh_tm, intr, tform_od, H=H, W=W)
        normal_rgb, _ = render_trimesh_to_tensor(normals_tm, intr, tform_od, H=H, W=W)
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        normal_frames.append(normal_rgb)

    rgb    = torch.stack(rgb_frames,    dim=0)  # (B, 3, H, W)
    depth  = torch.stack(depth_frames,  dim=0)  # (B, 1, H, W)
    normals = torch.stack(normal_frames, dim=0)  # (B, 3, H, W)

    d = depth[:, 0]
    d_max = d[d > 0].max() if (d > 0).any() else torch.tensor(1.0)
    depth_rgb = (d / d_max).clamp(0, 1).unsqueeze(1).expand(-1, 3, -1, -1)

    return {"rgb": rgb, "depth": depth_rgb, "normals": normals}


def _render_mesh_nvdiffrast(
    mesh: "Mesh",
    cam_tform4x4_obj: "Tensor",  # (B, 4, 4)
    cam_intr4x4: "Tensor",       # (B, 4, 4)
    H: int,
    W: int,
) -> "Tensor":
    """Render mesh from B viewpoints using nvdiffrast. Returns (B, 3, H, W) in [0, 1]."""
    import os
    import torch
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import nvdiffrast.torch as dr
    from o3b.cv.visual.show import OPEN3D_CAM_TFORM_CAM

    device = torch.device("cuda")
    cam_tform4x4_obj = cam_tform4x4_obj.to(device)
    cam_intr4x4 = cam_intr4x4.to(device)
    B = cam_tform4x4_obj.shape[0]

    znear, zfar = 0.001, 10000.0

    cams_persp4x4 = cam_intr4x4.clone().to(device=device, dtype=torch.float32)
    cams_persp4x4[:, 0, 2] = -cams_persp4x4[:, 0, 2]
    cams_persp4x4[:, 1, 2] = -cams_persp4x4[:, 1, 2]
    cams_persp4x4[:, 2, 2] = 0.0
    cams_persp4x4[:, 2, 3] = 1.0
    cams_persp4x4[:, 3, 2] = 1.0
    cams_persp4x4[:, 3, 3] = 0.0
    cams_persp4x4[:, 1, 2] = -(H + cams_persp4x4[:, 1, 2])

    top, bottom, left, right = 0, max(H, 1), 0, max(W, 1)
    tx = -(right + left) / (right - left)
    ty = -(top + bottom) / (top - bottom)
    U = -2.0 * znear * zfar / (zfar - znear)
    V_coef = -(zfar + znear) / (zfar - znear)
    ndc_mat = torch.tensor([
        [2.0 / (right - left), 0.0, 0.0, -tx],
        [0.0, 2.0 / (top - bottom), 0.0, -ty],
        [0.0, 0.0, U, V_coef],
        [0.0, 0.0, 0.0, -1.0],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 4)

    open3d_tform = OPEN3D_CAM_TFORM_CAM.clone().unsqueeze(0).to(device=device, dtype=torch.float32)
    cams_proj4x4 = (ndc_mat @ cams_persp4x4) @ (open3d_tform @ cam_tform4x4_obj)  # (B, 4, 4)

    verts = mesh.verts.to(device=device, dtype=torch.float32)  # (V, 3)
    V_count = verts.shape[0]
    verts_h = torch.cat([verts, torch.ones(V_count, 1, device=device, dtype=torch.float32)], dim=-1)
    verts_clip = (cams_proj4x4[:, None] @ verts_h[None, :, :, None])[..., 0]  # (B, V, 4)

    faces = mesh.faces.to(device=device, dtype=torch.int32)  # (F, 3)

    glctx = dr.RasterizeCudaContext(device=device)
    rast_out, _ = dr.rasterize(glctx, verts_clip.contiguous(), faces.contiguous(), resolution=[H, W])

    if mesh.texture is not None and mesh.verts_uvs is not None:
        uv = mesh.verts_uvs.to(device=device, dtype=torch.float32).clone()
        uv[:, 1] = 1.0 - uv[:, 1]
        uv_idx = (mesh.faces_uvs if mesh.faces_uvs is not None else faces).to(device=device, dtype=torch.int32)
        texc, _ = dr.interpolate(uv.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, uv_idx.contiguous())
        tex = mesh.texture.flip(1).permute(1, 2, 0).to(device=device, dtype=torch.float32)
        color = dr.texture(tex.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), texc.contiguous(), filter_mode="linear")
    elif mesh.vert_colors is not None:
        vc = mesh.vert_colors.to(device=device, dtype=torch.float32)
        color, _ = dr.interpolate(vc.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, faces.contiguous())
    else:
        gray = torch.full((V_count, 3), 0.7, device=device, dtype=torch.float32)
        color, _ = dr.interpolate(gray.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, faces.contiguous())

    try:
        color = dr.antialias(color.contiguous(), rast_out, verts_clip.contiguous(), faces.contiguous())
    except Exception:
        pass
    mask = (rast_out[..., 3:4] > 0).float()
    color = color * mask
    rgb = color.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)

    # ── Depth ─────────────────────────────────────────────────────────────────
    verts_cam = (cam_tform4x4_obj[:, None] @ verts_h[None, :, :, None])[..., 0]  # (B, V, 4)
    # In our OpenGL convention +Z is back, so Z is negative for verts in front of camera.
    # Negate to obtain positive depth values.
    verts_z = -verts_cam[:, :, 2:3]  # (B, V, 1)
    depth_interp, _ = dr.interpolate(verts_z.contiguous(), rast_out, faces)  # (B, H, W, 1)
    depth_masked = depth_interp * mask
    d_max = depth_masked.max()
    depth_rgb = (depth_masked / (d_max + 1e-6)).expand(-1, -1, -1, 3).permute(0, 3, 1, 2).contiguous()

    # ── Normals ───────────────────────────────────────────────────────────────
    import numpy as np
    verts_np = mesh.verts.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    v0, v1, v2 = verts_np[faces_np[:, 0]], verts_np[faces_np[:, 1]], verts_np[faces_np[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    vn = np.zeros_like(verts_np)
    for i in range(3):
        np.add.at(vn, faces_np[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8
    vn_t = torch.from_numpy(vn.astype(np.float32)).to(device)  # (V, 3)
    normals_interp, _ = dr.interpolate(
        vn_t.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, faces
    )  # (B, H, W, 3)
    normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1)
    normals_rgb = ((normals_interp * 0.5 + 0.5) * mask).permute(0, 3, 1, 2).contiguous()

    return {"rgb": rgb, "depth": depth_rgb, "normals": normals_rgb}


def render_mesh_from_viewpoints(
    batch: "FrameObjectBatch",
    H: int = 512,
    W: int = 512,
    renderer: str = "pyrender",
) -> "dict[str, Tensor]":
    """
    Render batch.mesh from viewpoints in batch.cam_tform4x4_obj.

    Args:
        batch: FrameObjectBatch with mesh (shared) and cam_tform4x4_obj (B, 4, 4).
               Optionally cam_intr4x4 (B, 4, 4) or (4, 4); defaults to 25° FOV.
        H, W:  Output image size.
        renderer: "pyrender" or "nvdiffrast".

    Returns:
        dict with keys "rgb", "depth", "normals", each (B, 3, H, W) float32 in [0, 1].
    """
    import torch
    from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size

    cam_tform4x4_obj = batch.cam_tform4x4_obj
    B = cam_tform4x4_obj.shape[0]

    cam_intr4x4 = batch.cam_intr4x4
    if cam_intr4x4 is None:
        cam_intr4x4 = get_default_camera_intrinsics_from_img_size(W, H).unsqueeze(0).expand(B, -1, -1)
    elif cam_intr4x4.dim() == 2:
        cam_intr4x4 = cam_intr4x4.unsqueeze(0).expand(B, -1, -1)

    if renderer == "nvdiffrast":
        return _render_mesh_nvdiffrast(batch.mesh, cam_tform4x4_obj, cam_intr4x4, H, W)
    return _render_mesh_pyrender(batch.mesh, cam_tform4x4_obj, cam_intr4x4, H, W)
