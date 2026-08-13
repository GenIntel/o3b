from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

from o3b.task.task import OD3D_Task, register_task
from o3b.data.datatypes.mesh import OBJ_SIZE_NCDS as _OBJ_SIZE_NCDS, verts_ncds
from o3b.data.datatypes.frame_object import FrameObjectPairBatch
from o3b.task.datatypes.frame_object_pair_quant import FrameObjectPairQuantBatch
from o3b.task.datatypes.frame_object_pair_qualit import FrameObjectPairQualitBatch


# ── projection helpers ─────────────────────────────────────────────────────────

def _proj_frame(kpts3d: torch.Tensor, tform4x4: torch.Tensor, intr4x4: torch.Tensor):
    """Project 3D points to 2D pixels via the frame (CV) convention — matches the
    frame-object overlay viewer. Returns an (K, 2) numpy array."""
    from o3b.cv.geometry.transform import proj3d2d_tform4x4_intr4x4_broadcast
    uv = proj3d2d_tform4x4_intr4x4_broadcast(
        pts3d=kpts3d.float().cpu().unsqueeze(0),
        tform4x4=tform4x4.float().cpu().unsqueeze(0).unsqueeze(0),
        intr4x4=intr4x4.float().cpu().unsqueeze(0).unsqueeze(0),
    )[0]
    return uv.numpy()


def _proj_opengl(kpts3d: torch.Tensor, cam_tform4x4_obj: torch.Tensor, intr4x4: torch.Tensor):
    """Project via the OpenGL convention used by render_trimesh_to_tensor
    (-Z forward, -Y up). Returns an (K, 2) numpy array."""
    import numpy as np
    fx, fy = intr4x4[0, 0].item(), intr4x4[1, 1].item()
    cx, cy = intr4x4[0, 2].item(), intr4x4[1, 2].item()
    pts_h = torch.cat([kpts3d.float().cpu(), torch.ones(len(kpts3d), 1)], dim=1)
    pts_cam = (cam_tform4x4_obj.float().cpu() @ pts_h.T).T[:, :3]
    z = (-pts_cam[:, 2]).clamp(min=1e-6).numpy()
    u = fx * pts_cam[:, 0].numpy() / z + cx
    v = fy * (-pts_cam[:, 1]).numpy() / z + cy
    return np.stack([u, v], axis=1)


def _to_uint8_img(rgb: torch.Tensor):
    """(3, H, W) tensor → (H, W, 3) uint8 numpy."""
    import numpy as np
    a = rgb.float().cpu()
    if a.max() > 1.5:
        a = a / 255.0
    return (a.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── qualitative renderers ──────────────────────────────────────────────────────

def _draw_corr_panels(left, right, left_uv, right_gt_uv, right_pred_uv, valid, correct,
                      left_amodal=None, right_amodal=None):
    """Draw a query|target side-by-side panel (left/right are (H, W, 3) uint8):
    each query keypoint (left) linked to its predicted target location (right),
    GT target keypoints shown as hollow white rings.  Returns (3, H, Wl+Wr)."""
    import numpy as np
    from PIL import Image, ImageDraw

    H = left.shape[0]
    if right.shape[0] != H:  # match heights for side-by-side concat
        new_w = max(1, int(round(right.shape[1] * H / right.shape[0])))
        right = np.asarray(Image.fromarray(right).resize((new_w, H)))
    Wl, Wr = left.shape[1], right.shape[1]
    pil = Image.fromarray(np.concatenate([left, right], axis=1))
    draw = ImageDraw.Draw(pil)
    draw.line([(Wl, 0), (Wl, H - 1)], fill=(120, 120, 120), width=2)
    from o3b.cv.visual.draw import get_colors
    K = len(valid)
    kpt_colors_f = get_colors(K)  # (K, 3) float [0, 1], distinct per keypoint
    r = max(3, H // 64)
    for k in range(len(valid)):
        if not bool(valid[k]):
            continue
        col     = (0, 180, 0) if bool(correct[k]) else (0, 100, 220)     # line: green / blue
        kpt_col = tuple((kpt_colors_f[k] * 255).round().int().tolist())   # keypoint: per-index color
        sx, sy = int(np.clip(left_uv[k, 0], 0, Wl - 1)),       int(np.clip(left_uv[k, 1], 0, H - 1))
        gx, gy = int(np.clip(right_gt_uv[k, 0], 0, Wr - 1)),   int(np.clip(right_gt_uv[k, 1], 0, H - 1))
        px, py = int(np.clip(right_pred_uv[k, 0], 0, Wr - 1)), int(np.clip(right_pred_uv[k, 1], 0, H - 1))
        gx += Wl; px += Wl
        rb = max(1, r // 2)  # black centre dot marks an amodal (occluded) keypoint
        draw.line([(sx, sy), (px, py)], fill=col, width=1)                         # query → predicted
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=kpt_col, outline=(0, 0, 0))       # source
        if left_amodal is not None and bool(left_amodal[k]):
            draw.ellipse([sx - rb, sy - rb, sx + rb, sy + rb], fill=(0, 0, 0))
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=kpt_col, width=2)             # GT ring
        if right_amodal is not None and bool(right_amodal[k]):
            draw.ellipse([gx - rb, gy - rb, gx + rb, gy + rb], fill=(0, 0, 0))
        draw.ellipse([px - r, py - r, px + r, py + r], fill=kpt_col, outline=(0, 0, 0))      # predicted
        if right_amodal is not None and bool(right_amodal[k]):
            draw.ellipse([px - rb, py - rb, px + rb, py + rb], fill=(0, 0, 0))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _render_mesh_topdown(mesh, cam_t, cam_k, H, W):
    """Render one mesh from the given (top-down) camera → (H, W, 3) uint8."""
    import numpy as np
    if mesh is None:
        return np.full((H, W, 3), 235, dtype=np.uint8)
    try:
        from o3b.io import _mesh_to_trimesh
        from o3b.cv.visual.show import render_trimesh_to_tensor
        rgb, _ = render_trimesh_to_tensor(
            _mesh_to_trimesh(mesh), cam_k, cam_t, H, W, rgb_bg=[0.92, 0.92, 0.92],
        )
        return (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    except Exception:
        return np.full((H, W, 3), 235, dtype=np.uint8)


def _render_mesh_nocs(mesh, cam_t, cam_k, H, W):
    """Render per-pixel NOCS-0c coordinates of a mesh, then colorise pixelwise.

    Renders the mesh depth, unprojects each pixel to a 3D surface point
    (exact per-pixel coordinates — no color pipeline involved), maps it back
    to object/NOCS-0c space via inv(cam_t), and applies
    o3b.cv.visual.point3d_to_color3d.nocs_0c_to_rgb **pixelwise** (so its
    checkerboard pattern stays sharp instead of being blurred by vertex-color
    interpolation and the renderer's tone mapping).

    Returns (colored (H, W, 3) float32 [0,1], mask (H, W) bool) or (None, None).
    """
    import numpy as np
    if mesh is None:
        return None, None
    try:
        from o3b.io import _mesh_to_trimesh
        from o3b.cv.visual.show import render_trimesh_to_tensor
        from o3b.cv.visual.point3d_to_color3d import nocs_0c_to_rgb
        from o3b.cv.geometry.transform import (
            depth2pts3d_grid, transf3d_broadcast, inv_tform4x4,
        )
        _, depth = render_trimesh_to_tensor(
            _mesh_to_trimesh(mesh), cam_k.float(), cam_t.float(), H, W,
            rgb_bg=[0.0, 0.0, 0.0],
        )
        depth = depth.reshape(H, W).float().cpu()
        mask = (depth > 0)
        # per-pixel surface points: cam space (OpenGL) → object (NOCS-0c) space
        pts_cam = depth2pts3d_grid(depth[None], cam_k.float().cpu(), opengl=True)  # (3, H, W)
        pts_obj = transf3d_broadcast(
            pts_cam.permute(1, 2, 0).reshape(-1, 3), inv_tform4x4(cam_t.float().cpu()),
        ).reshape(H, W, 3)
        colored = nocs_0c_to_rgb(pts_obj).clamp(0, 1)
        colored = torch.where(mask[..., None], colored, torch.zeros_like(colored))
        return colored.numpy().astype(np.float32), mask.numpy()
    except Exception:
        return None, None


def _nocs_on_bg(colored, mask, bg, alpha=1.0):
    """Composite a NOCS render onto a background (H, W, 3) float [0,1] image."""
    import numpy as np
    out = bg.astype(np.float32).copy()
    m = mask.astype(np.float32)[..., None] * alpha
    out = out * (1 - m) + colored * m
    return (out * 255).clip(0, 255).astype(np.uint8)


def _topdown_camera(H=256, W=256):
    """A single top-down camera (looking down the NCDS vertical axis) + intrinsics."""
    from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size
    from o3b.cv.geometry.transform import transf4x4_from_spherical
    cam_t = transf4x4_from_spherical(
        azim=torch.tensor([0.0]), elev=torch.tensor([math.pi / 2 - 0.15]),
        theta=torch.tensor([0.0]), dist=5.0,
    )[0].float()
    cam_k = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()
    return cam_t, cam_k


def _render_topdown_corr_img(
    src_mesh, trgt_mesh, src_kpts_ncds, trgt_gt_kpts_ncds, trgt_pred_kpts_ncds,
    valid, correct, H=256, W=256, src_amodal=None, trgt_amodal=None,
):
    """source|target objects rendered from a top-down camera, with source query
    keypoints linked to their predicted target locations.  Returns (3, H, 2W)
    or None if the renderer is unavailable."""
    try:
        from o3b.io import _mesh_to_trimesh  # noqa: F401  (renderer availability probe)
        from o3b.cv.visual.show import render_trimesh_to_tensor  # noqa: F401
    except ImportError:
        return None
    cam_t, cam_k = _topdown_camera(H, W)
    left  = _render_mesh_topdown(src_mesh,  cam_t, cam_k, H, W)
    right = _render_mesh_topdown(trgt_mesh, cam_t, cam_k, H, W)
    src_uv  = _proj_opengl(src_kpts_ncds,       cam_t, cam_k)
    gt_uv   = _proj_opengl(trgt_gt_kpts_ncds,   cam_t, cam_k)
    pred_uv = _proj_opengl(trgt_pred_kpts_ncds, cam_t, cam_k)
    return _draw_corr_panels(left, right, src_uv, gt_uv, pred_uv, valid, correct,
                             left_amodal=src_amodal, right_amodal=trgt_amodal)


# ── correspondence variants ────────────────────────────────────────────────────

def _img_hw(imgs, b):
    """(H, W) of sample b for a stacked tensor or a per-sample list."""
    im = imgs[b]
    return int(im.shape[-2]), int(im.shape[-1])


def _pred_featmap2d(batch, src_ncds, kpts_valid):
    """Variant b: featmap 2D correspondence.

    Per sample: project src keypoints into the src image (GT pose), sample the
    src featmap at those pixels, find the nearest-neighbour pixel in the trgt
    featmap (search restricted to the trgt object mask ∧ valid depth), and lift
    the matched pixel to 3D camera space via the trgt depth map.

    Returns (pred_kpts_cam_t (B,K,3) w/ NaN where unavailable,
             src_uv (B,K,2), pred_uv (B,K,2)) or (None, None, None).
    """
    import torch.nn.functional as F
    from o3b.cv.geometry.transform import depth2pts3d_grid

    if (batch.src_featmap is None or batch.trgt_featmap is None
            or batch.src_rgb is None or batch.trgt_depth is None
            or batch.src_cam_intr4x4 is None or batch.trgt_cam_intr4x4 is None
            or batch.src_obj_kpts3d is None):
        return None, None, None

    B, K, _ = batch.src_obj_kpts3d.shape
    pred_kpts = torch.full((B, K, 3), float("nan"))
    src_uv_all  = torch.full((B, K, 2), float("nan"))
    pred_uv_all = torch.full((B, K, 2), float("nan"))

    for b in range(B):
        try:
            src_fm  = batch.src_featmap[b].float()   # (F, Hf, Wf)
            trgt_fm = batch.trgt_featmap[b].float()  # (F, Hf, Wf)
            Hf, Wf = src_fm.shape[-2:]
            H_s, W_s = _img_hw(batch.src_rgb, b)
            trgt_depth = batch.trgt_depth[b].float().cpu()  # (H, W)
            H_t, W_t = trgt_depth.shape[-2:]

            # 1. src kpts → 2D (image res, GT pose)
            src_uv = torch.from_numpy(_proj_frame(
                batch.src_obj_kpts3d[b], src_ncds[b], batch.src_cam_intr4x4[b],
            )).float()  # (K, 2)
            src_uv_all[b] = src_uv

            # 2. sample src feats at kpt pixels (featmap res)
            uf = ((src_uv[:, 0] + 0.5) * Wf / W_s - 0.5).round().long().clamp(0, Wf - 1)
            vf = ((src_uv[:, 1] + 0.5) * Hf / H_s - 0.5).round().long().clamp(0, Hf - 1)
            query_feats = src_fm[:, vf, uf].T  # (K, F)

            # 3. search mask on the trgt featmap: object mask ∧ valid depth
            valid_t = (trgt_depth > 0)
            if batch.trgt_fo_mask is not None:
                valid_t = valid_t & batch.trgt_fo_mask[b].bool().cpu()
            mask_f = F.interpolate(
                valid_t[None, None].float(), size=(Hf, Wf), mode="nearest",
            )[0, 0] > 0.5  # (Hf, Wf)
            if not mask_f.any():
                mask_f = torch.ones_like(mask_f)

            # 4. NN in feature space (feats are L2-normalised → cosine sim)
            sim = torch.einsum("kf,fhw->khw", query_feats, trgt_fm)  # (K, Hf, Wf)
            sim = sim.masked_fill(~mask_f[None], -float("inf"))
            flat_idx = sim.reshape(K, -1).argmax(dim=1)  # (K,)
            pv, pu = flat_idx // Wf, flat_idx % Wf

            # 5. featmap pixel → image res
            pu_img = (pu.float() + 0.5) * W_t / Wf - 0.5
            pv_img = (pv.float() + 0.5) * H_t / Hf - 0.5
            pred_uv_all[b] = torch.stack([pu_img, pv_img], dim=1)

            # 6. lift via trgt depth (OpenGL cam space, matching the GT kpts)
            pts3d = depth2pts3d_grid(
                trgt_depth[None], batch.trgt_cam_intr4x4[b].float().cpu(), opengl=True,
            )  # (3, H, W)
            iu = pu_img.round().long().clamp(0, W_t - 1)
            iv = pv_img.round().long().clamp(0, H_t - 1)
            pred_kpts[b] = pts3d[:, iv, iu].T  # (K, 3)
        except Exception:
            continue

    return pred_kpts, src_uv_all, pred_uv_all


_WARNED_MESH_CORRESP = False


def _mesh_ncds(mesh, rescale: bool = False, recenter: bool = False):
    """``mesh`` with :func:`verts_ncds`-normalised verts (colors/UVs/texture kept)."""
    from dataclasses import replace
    if mesh is None or mesh.verts is None:
        return mesh
    return replace(mesh, verts=verts_ncds(mesh.verts, rescale, recenter))


def _pred_mesh_corresp(batch, query_kpts_cam_q, pred_src, pred_trgt,
                       ncds_rescale: bool = False, ncds_recenter: bool = False):
    """Variant c: mesh (barycentric) correspondence.

    Per sample: pose the src/trgt meshes into their camera spaces with the
    predicted poses, project each query keypoint onto the src mesh surface,
    express it in barycentric coordinates of its triangle, and evaluate the same
    triangle + barycentric coordinates on the trgt mesh.

    Requires src and trgt meshes with identical topology (same category mesh).
    Returns pred_kpts_cam_t (B, K, 3) with NaN where unavailable, or None.
    """
    src_meshes  = batch.src_pred_mesh  if batch.src_pred_mesh  is not None else batch.src_meshes
    trgt_meshes = batch.trgt_pred_mesh if batch.trgt_pred_mesh is not None else batch.trgt_meshes
    if src_meshes is None or trgt_meshes is None:
        return None

    import numpy as np
    from o3b.cv.geometry.transform import transf3d_broadcast

    B, K, _ = query_kpts_cam_q.shape
    pred_kpts = torch.full((B, K, 3), float("nan"))

    for b in range(B):
        try:
            m_src, m_trgt = src_meshes[b], trgt_meshes[b]
            if m_src is None or m_trgt is None:
                continue
            f_src  = m_src.faces.long().cpu()
            f_trgt = m_trgt.faces.long().cpu()
            if f_src.shape != f_trgt.shape or len(m_src.verts) != len(m_trgt.verts):
                continue  # barycentric transfer needs identical topology

            # optionally normalise both meshes to NCDS before posing: the poses
            # are NCDS poses, and src/trgt must be normalised the same way for
            # the barycentric transfer to be a correspondence between comparable
            # shapes.  Off by default — see the class docstring.
            v_src  = verts_ncds(m_src.verts,  ncds_rescale, ncds_recenter).cpu()
            v_trgt = verts_ncds(m_trgt.verts, ncds_rescale, ncds_recenter).cpu()
            v_src_cam  = transf3d_broadcast(v_src,  pred_src[b].float().cpu())
            v_trgt_cam = transf3d_broadcast(v_trgt, pred_trgt[b].float().cpu())

            import trimesh
            from trimesh.triangles import points_to_barycentric, barycentric_to_points
            tm = trimesh.Trimesh(v_src_cam.numpy(), f_src.numpy(), process=False)
            closest, _dists, face_ids = tm.nearest.on_surface(
                query_kpts_cam_q[b].detach().cpu().numpy(),
            )
            bary = points_to_barycentric(
                triangles=v_src_cam.numpy()[f_src.numpy()[face_ids]], points=closest,
            )
            pred = barycentric_to_points(
                triangles=v_trgt_cam.numpy()[f_trgt.numpy()[face_ids]], barycentric=bary,
            )
            pred_kpts[b] = torch.from_numpy(np.asarray(pred, dtype=np.float32))
        except Exception as e:
            # All-NaN samples are dropped from the nan-safe metric means, so a
            # systematic failure here (e.g. missing rtree for
            # trimesh.nearest.on_surface) would silently erase the mesh variant.
            global _WARNED_MESH_CORRESP
            if not _WARNED_MESH_CORRESP:
                _WARNED_MESH_CORRESP = True
                logger.warning(
                    f"mesh correspondence failed ({type(e).__name__}: {e}); "
                    f"the 'mesh' variant will be NaN for such samples and "
                    f"dropped from the metric means. Warning only shown once."
                )
            continue

    return pred_kpts


# ── keypoint symmetry (discrete + continuous rotational) ──────────────────────

def _kpts_sym_candidates(trgt_kpts, trgt_ncds, obj_tform4x4_obj_syms):
    """Discrete-symmetry target keypoint candidates, in target camera space.

    trgt_kpts: (B, K, 3) target keypoints in NCDS object space.
    trgt_ncds: (B, 4, 4) NCDS→cam (GT) pose.
    obj_tform4x4_obj_syms: (B, S, 4, 4) discrete object-space symmetry transforms,
      from o3b.cv.metric.pose.get_obj_tform4x4_obj_sym (candidate 0 is always the
      identity, so this is a strict superset of the un-symmetrised keypoints).

    Returns (B, K, S, 3).
    """
    from o3b.cv.geometry.transform import transf3d_broadcast
    trgt_obj_kpts3d_syms = transf3d_broadcast(
        trgt_kpts[:, :, None, :], obj_tform4x4_obj_syms[:, None, :, :, :],
    )  # (B, K, S, 3)
    return transf3d_broadcast(trgt_obj_kpts3d_syms, trgt_ncds[:, None, None, :, :])


def _kpts_sym_min_dist(gt_cam_kpts_syms, axis6d_syms, axis6d_syms_mask, pred_cam_kpts):
    """Per-keypoint distance from a prediction to its nearest symmetric target.

    gt_cam_kpts_syms: (B, K, S, 3) discrete symmetric target candidates (cam space),
      from _kpts_sym_candidates.
    axis6d_syms: (B, 3, 6) per-axis-slot (offset, direction) in cam space, from
      transf_axis6d_broadcast(get_obj_axis6d_with_mask(obj_syms), trgt_ncds).
    axis6d_syms_mask: (B, 3) bool — True where that axis slot is continuously
      (rotationally) symmetric, from o3b.cv.metric.pose.get_obj_axis6d_with_mask.
    pred_cam_kpts: (B, K, 3) predicted target keypoints (cam space); NaN rows are
      supported (variant unavailable for that sample) and propagate to NaN dist.

    For a sample with exactly one continuous rotational axis, each discrete
    candidate is first rotated to its closest point (around that axis) to the
    prediction — see get_closest_B_to_A_rotated_around_axis6d — before taking the
    arg-min over candidates; samples with more than one such axis are not
    supported and fall back to the discrete candidates only.

    Returns (dist (B, K), gt_sel (B, K, 3)) — dist to, and the winning candidate.
    """
    from o3b.cv.metric.pose import get_closest_B_to_A_rotated_around_axis6d

    B, K, S, _ = gt_cam_kpts_syms.shape
    gt_syms = gt_cam_kpts_syms.clone()
    pred_safe = torch.nan_to_num(pred_cam_kpts, nan=0.0)
    for b in range(B):
        mask_b = axis6d_syms_mask[b]
        if mask_b.sum() != 1:
            if mask_b.sum() > 1:
                logger.warning("keypoint symmetry: >1 continuous rotational axis is not supported, "
                               "falling back to discrete candidates only.")
            continue
        axis6d_b = axis6d_syms[b][mask_b][0]                              # (6,)
        A = pred_safe[b][:, None, :].expand(K, S, 3).reshape(-1, 3)
        B_ = gt_syms[b].reshape(-1, 3)
        gt_syms[b] = get_closest_B_to_A_rotated_around_axis6d(
            A=A, B=B_, axis6d=axis6d_b[None].expand(K * S, 6),
        ).reshape(K, S, 3)

    dist_all = (gt_syms - pred_cam_kpts[:, :, None, :]).norm(dim=-1)              # (B, K, S)
    dist_finite = torch.where(torch.isfinite(dist_all), dist_all, torch.full_like(dist_all, float("inf")))
    sym_ids = dist_finite.argmin(dim=-1)                                          # (B, K)
    gt_sel = torch.gather(gt_syms, dim=2,
                          index=sym_ids[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)
    dist = (pred_cam_kpts - gt_sel).norm(dim=-1)                                  # (B, K), NaN-preserving
    return dist, gt_sel


def _variant_metrics(dist, kpts_valid, threshold, src_vis, trgt_vis):
    """PCK@0.1 / modal / amodal PCK given a precomputed (symmetry-aware) per-keypoint
    Euclidean distance (see _kpts_sym_min_dist).

    NaN distances (variant unavailable for that sample) yield NaN aggregates so
    nan-safe means skip them.  Returns a dict of tensors + per-kpt is_correct.
    """
    dev = dist.device
    sample_ok = torch.isfinite(dist).all(dim=1)                      # (B,)
    n_valid = kpts_valid.float().sum(dim=1).clamp(min=1)

    is_correct = (dist < threshold.unsqueeze(1)) & torch.isfinite(dist)
    correct = is_correct.float() * kpts_valid.float()

    dist_masked = torch.where(kpts_valid, dist, torch.zeros_like(dist))
    dist_mean = torch.where(sample_ok, dist_masked.sum(dim=1) / n_valid,
                            torch.full_like(n_valid, float("nan")))
    pck = torch.where(sample_ok, correct.sum(dim=1) / n_valid,
                      torch.full_like(n_valid, float("nan")))

    modal_pck = amodal_pck = None
    if src_vis is not None and trgt_vis is not None:
        both_vis    = src_vis.bool().to(dev) & trgt_vis.bool().to(dev)
        modal_mask  = kpts_valid & both_vis
        amodal_mask = kpts_valid & ~both_vis
        n_modal, n_amodal = modal_mask.float().sum(dim=1), amodal_mask.float().sum(dim=1)
        nan = torch.full_like(n_valid, float("nan"))
        modal_pck  = torch.where((n_modal  > 0) & sample_ok,
                                 (correct * modal_mask.float()).sum(dim=1)  / n_modal.clamp(min=1),  nan)
        amodal_pck = torch.where((n_amodal > 0) & sample_ok,
                                 (correct * amodal_mask.float()).sum(dim=1) / n_amodal.clamp(min=1), nan)

    return {"dist": dist, "dist_mean": dist_mean, "pck": pck,
            "modal": modal_pck, "amodal": amodal_pck, "is_correct": is_correct}


# ── task ───────────────────────────────────────────────────────────────────────

@register_task("CamCrsp3DNNTask")
class CamCrsp3DNNTask(OD3D_Task):
    """Camera-space 3D correspondence driven by predicted object poses.

    For a frame-object pair (query = src, target = trgt) the GT keypoints live
    in NCDS object space.  The metric goes:

      1. Transform query and target ``obj_kpts3d`` into their respective camera
         spaces via the GT ``cam_tform4x4_obj_ncds``.
      2. From the *predicted* per-frame poses build the relative camera
         transform::

             cam_query_tform4x4_cam_target = pred_cam_q_tform_obj @ inv(pred_cam_t_tform_obj)
             cam_target_tform4x4_cam_query = inv(cam_query_tform4x4_cam_target)

         (the relative cam↔cam transform is rigid, so any uniform object scale
         embedded in the poses cancels).
      3. Map the query keypoints (query-camera space) into the target camera
         space and compare against the GT target keypoints.
      4. Euclidean error, thresholded at ``0.1 * trgt_obj_size`` for PCK.

    Qualitative: correspondences drawn on the query|target frame RGB, and on a
    top-down render of the target object.  Predicted poses fall back to GT when
    unavailable (oracle upper bound).

    ``mesh_ncds_rescale`` / ``mesh_ncds_recenter`` control how the meshes of the
    ``mesh`` variant are normalised into NCDS before being posed — see
    :func:`o3b.data.datatypes.mesh.verts_ncds`.  Both default to off: GT meshes
    are already NCDS, and a method that emits a predicted mesh is expected to
    normalise it itself (as MorpheusMethod's own ``mesh_ncds_rescale`` /
    ``mesh_ncds_recenter`` do) so that its mesh and pose stay consistent.  Turn
    them on to normalise here instead, for a method that does not.
    """

    def __init__(
        self,
        qualit: bool = True,
        mesh_ncds_rescale: bool = False,
        mesh_ncds_recenter: bool = False,
        **kwargs,
    ):
        self.qualit = qualit
        self.mesh_ncds_rescale  = mesh_ncds_rescale
        self.mesh_ncds_recenter = mesh_ncds_recenter
        self._warned_missing: set[tuple] = set()

    def forward(self, batch: FrameObjectPairBatch, return_qualit: bool = True) -> Tuple[FrameObjectPairQuantBatch, FrameObjectPairQualitBatch]:
        from o3b.cv.geometry.transform import (
            transf3d_broadcast, tform4x4_broadcast, inv_tform4x4,
        )

        src_kpts       = batch.src_obj_kpts3d
        trgt_kpts      = batch.trgt_obj_kpts3d
        src_kpts_mask  = batch.src_obj_kpts3d_mask
        trgt_kpts_mask = batch.trgt_obj_kpts3d_mask
        src_ncds       = batch.src_cam_tform4x4_obj_ncds
        trgt_ncds      = batch.trgt_cam_tform4x4_obj_ncds
        trgt_obj_size  = batch.trgt_obj_size

        # Predicted poses, falling back to GT (oracle) when unavailable. The oracle
        # uses cam_tform4x4_obj_ncds (NCDS→cam), which embeds the 1D size (largest
        # object dimension) so cross-instance correspondences transfer in a
        # size-normalised frame rather than in absolute metric pose.
        pred_src  = batch.src_pred_cam_tform4x4_obj  if batch.src_pred_cam_tform4x4_obj  is not None else src_ncds
        pred_trgt = batch.trgt_pred_cam_tform4x4_obj if batch.trgt_pred_cam_tform4x4_obj is not None else trgt_ncds

        required = {
            "src_obj_kpts3d":             src_kpts,
            "trgt_obj_kpts3d":            trgt_kpts,
            "src_cam_tform4x4_obj_ncds":  src_ncds,
            "trgt_cam_tform4x4_obj_ncds": trgt_ncds,
            "pred_cam_tform4x4_obj (src)":  pred_src,
            "pred_cam_tform4x4_obj (trgt)": pred_trgt,
            "trgt_obj_size":              trgt_obj_size,
        }
        missing = tuple(name for name, x in required.items() if x is None)
        if missing:
            if missing not in self._warned_missing:
                self._warned_missing.add(missing)
                print(f"WARNING: CamCrsp3DNNTask skipping batch — required batch fields "
                      f"are None: {', '.join(missing)}. No metrics will be produced for "
                      f"such batches (check dataset modalities / sharded cache contents).")
            return FrameObjectPairQuantBatch(), FrameObjectPairQualitBatch()

        B, K, _ = src_kpts.shape
        dev = src_kpts.device

        # ── valid keypoint mask ───────────────────────────────────────────────
        if src_kpts_mask is not None and trgt_kpts_mask is not None:
            kpts_valid = src_kpts_mask & trgt_kpts_mask
        elif src_kpts_mask is not None:
            kpts_valid = src_kpts_mask
        elif trgt_kpts_mask is not None:
            kpts_valid = trgt_kpts_mask
        else:
            kpts_valid = torch.ones(B, K, dtype=torch.bool, device=dev)
        n_valid = kpts_valid.float().sum(dim=1).clamp(min=1)

        # ── step 1: kpts (NCDS) → camera space (metric, GT poses) ─────────────
        query_kpts_cam_q   = transf3d_broadcast(src_kpts.float(),  src_ncds.float().unsqueeze(1))   # (B, K, 3)
        gt_trgt_kpts_cam_t = transf3d_broadcast(trgt_kpts.float(), trgt_ncds.float().unsqueeze(1))   # (B, K, 3)

        # ── target keypoint symmetry: discrete candidates + continuous rotational ──
        # obj_syms axis codes: -1 continuous rotational, 1 none, 2/4 discrete rotational
        # (see o3b.cv.metric.pose). Candidate 0 is always the identity, so this is a
        # strict superset of gt_trgt_kpts_cam_t when no symmetry is available/annotated.
        from o3b.cv.geometry.transform import transf_axis6d_broadcast
        from o3b.cv.metric.pose import get_obj_tform4x4_obj_sym, get_obj_axis6d_with_mask

        trgt_obj_syms = batch.trgt_obj_syms.float().to(dev) if batch.trgt_obj_syms is not None \
            else torch.ones(B, 3, device=dev)
        obj_tform4x4_obj_syms = get_obj_tform4x4_obj_sym(trgt_obj_syms)                       # (B, S, 4, 4)
        obj_axis6d_syms, obj_axis6d_syms_mask = get_obj_axis6d_with_mask(trgt_obj_syms)       # (B, 3, 6), (B, 3) bool

        gt_trgt_kpts_cam_t_syms = _kpts_sym_candidates(
            trgt_kpts.float(), trgt_ncds.float(), obj_tform4x4_obj_syms,
        )  # (B, K, S, 3)
        trgt_cam_obj_axis6d_syms = transf_axis6d_broadcast(
            axis6d=obj_axis6d_syms, transf4x4=trgt_ncds.float()[:, None, :, :],
        )  # (B, 3, 6)

        # ── step 2: predicted relative camera transform (target ← query) ──────
        cam_query_tform4x4_cam_target = tform4x4_broadcast(pred_src.float(), inv_tform4x4(pred_trgt.float()))
        cam_target_tform4x4_cam_query = inv_tform4x4(cam_query_tform4x4_cam_target)  # (B, 4, 4)

        # ── step 3: query kpts → predicted target camera space ────────────────
        pred_trgt_kpts_cam_t = transf3d_broadcast(
            query_kpts_cam_q, cam_target_tform4x4_cam_query.unsqueeze(1),
        )  # (B, K, 3)

        # ── variant b: featmap 2D correspondence (NN in feature space + depth) ─
        pred_feat2d_kpts, feat2d_src_uv, feat2d_pred_uv = _pred_featmap2d(
            batch, src_ncds, kpts_valid,
        )

        # ── variant c: mesh (barycentric) correspondence ───────────────────────
        pred_mesh_kpts = _pred_mesh_corresp(
            batch, query_kpts_cam_q, pred_src, pred_trgt,
            self.mesh_ncds_rescale, self.mesh_ncds_recenter,
        )

        # ── metrics: euclidean error + PCK @ 0.1 * trgt_obj_size, per variant ──
        threshold = 0.1 * trgt_obj_size.float().to(dev)  # (B,)
        src_vis, trgt_vis = batch.src_obj_kpts2d_mask, batch.trgt_obj_kpts2d_mask

        pose_dist, _ = _kpts_sym_min_dist(
            gt_trgt_kpts_cam_t_syms, trgt_cam_obj_axis6d_syms, obj_axis6d_syms_mask,
            pred_trgt_kpts_cam_t,
        )
        m_pose = _variant_metrics(pose_dist, kpts_valid, threshold, src_vis, trgt_vis)

        quant = FrameObjectPairQuantBatch(
            cam_kpts_trgt_euc_dist      = m_pose["dist"],
            kpts_mask                   = kpts_valid,
            cam_kpts_trgt_euc_dist_mean = m_pose["dist_mean"],
            cam_kpts_trgt_pck01         = m_pose["pck"],
            cam_kpts_modal_trgt_pck01   = m_pose["modal"],
            cam_kpts_amodal_trgt_pck01  = m_pose["amodal"],
        )

        variant_correct = {"pose": m_pose["is_correct"]}
        for name, pred in (("featmap2d", pred_feat2d_kpts), ("mesh", pred_mesh_kpts)):
            if pred is None:
                continue
            dist, _ = _kpts_sym_min_dist(
                gt_trgt_kpts_cam_t_syms, trgt_cam_obj_axis6d_syms, obj_axis6d_syms_mask,
                pred.to(dev),
            )
            m = _variant_metrics(dist, kpts_valid, threshold, src_vis, trgt_vis)
            variant_correct[name] = m["is_correct"]
            quant.extra[f"cam_kpts_trgt_euc_dist_mean_{name}"] = m["dist_mean"]
            quant.extra[f"cam_kpts_trgt_pck01_{name}"]         = m["pck"]
            if m["modal"] is not None:
                quant.extra[f"cam_kpts_modal_trgt_pck01_{name}"]  = m["modal"]
                quant.extra[f"cam_kpts_amodal_trgt_pck01_{name}"] = m["amodal"]

        if self.qualit and return_qualit:
            qualit = self._render_qualit(
                batch, query_kpts_cam_q, gt_trgt_kpts_cam_t, pred_trgt_kpts_cam_t,
                trgt_ncds, kpts_valid, m_pose["is_correct"],
            )
            self._render_qualit_featmap2d(
                batch, qualit, src_ncds, feat2d_src_uv, feat2d_pred_uv,
                kpts_valid, variant_correct.get("featmap2d"),
            )
            self._render_qualit_mesh(
                batch, qualit, src_ncds, trgt_ncds, pred_src, pred_trgt,
                pred_mesh_kpts, kpts_valid, variant_correct.get("mesh"),
            )
        else:
            qualit = None

        return quant, qualit

    # ── qualitative ────────────────────────────────────────────────────────────

    def _render_qualit(
        self, batch, query_kpts_cam_q, gt_trgt_kpts_cam_t, pred_trgt_kpts_cam_t,
        trgt_ncds, kpts_valid, is_correct,
    ) -> FrameObjectPairQualitBatch:
        import torch.nn.functional as F
        from o3b.cv.geometry.transform import transf3d_broadcast, inv_tform4x4

        B = query_kpts_cam_q.shape[0]
        eye = torch.eye(4)

        # predicted target keypoints back in target NCDS space (for the render)
        trgt_ncds_inv = inv_tform4x4(trgt_ncds.float())
        pred_kpts_ncds = transf3d_broadcast(pred_trgt_kpts_cam_t, trgt_ncds_inv.unsqueeze(1))  # (B, K, 3)

        # draw the predicted 3-D bounding box on each frame (camera transforms +
        # predicted obj sizes in 3D), via batch_draw_bbox3d (OpenGL convention by
        # default), before the correspondence overlay. The box is expressed in
        # NCDS units (longest side → obj_size_ncds) and posed with the predicted
        # cam_tform4x4_obj_ncds (the GT NCDS pose/size is used as the oracle
        # fallback when no prediction is available).

        def _frames_with_bbox(rgb, bbox3d, intr, pose_ncds):
            frames = list(rgb) if rgb is not None else None
            if frames is None or bbox3d is None or intr is None:
                return frames
            bbox3d_f = bbox3d.float()
            if bbox3d_f.dim() == 3 and bbox3d_f.shape[1:] == (8, 3):
                # (B, 8, 3) cam-space corners — project and draw directly
                try:
                    from o3b.cv.visual.draw import draw_bbox3d_corners
                    for b_i in range(len(frames)):
                        frames[b_i] = draw_bbox3d_corners(
                            frames[b_i], bbox3d_f[b_i], intr[b_i], thickness=2,
                        ).float().div(255.0)
                except Exception:
                    pass
            elif pose_ncds is not None:
                # (B, 3) NCDS side lengths — normalize and use batch_draw_bbox3d
                try:
                    from o3b.cv.visual.draw import batch_draw_bbox3d
                    bbox3d_r = bbox3d_f.reshape(-1, 3)
                    denom = bbox3d_r.max(dim=1, keepdim=True).values.clamp(min=1e-6)
                    bbox3d_ncds = bbox3d_r * (_OBJ_SIZE_NCDS / denom)
                    frames = batch_draw_bbox3d(frames, bbox3d_ncds, intr, pose_ncds.float(), thickness=2)
                except Exception:
                    frames = list(rgb)
            return frames

        def _pick(pred, gt):
            return pred if pred is not None else gt

        src_frames  = _frames_with_bbox(
            batch.src_rgb,
            _pick(batch.src_pred_obj_size3d, batch.src_cam_bbox3d),
            batch.src_cam_intr4x4,
            _pick(batch.src_pred_cam_tform4x4_obj, batch.src_cam_tform4x4_obj_ncds))
        trgt_frames = _frames_with_bbox(
            batch.trgt_rgb,
            _pick(batch.trgt_pred_obj_size3d, batch.trgt_cam_bbox3d),
            batch.trgt_cam_intr4x4,
            _pick(batch.trgt_pred_cam_tform4x4_obj, batch.trgt_cam_tform4x4_obj_ncds))

        def _vstack(rows):
            """Pad row widths to the max and stack vertically → (3, sum H, W)."""
            rows = [r for r in rows if r is not None]
            if not rows:
                return None
            W = max(r.shape[2] for r in rows)
            out = []
            for r in rows:
                if r.shape[2] != W:
                    new_h = max(1, int(round(r.shape[1] * W / r.shape[2])))
                    r = F.interpolate(r.unsqueeze(0), size=(new_h, W), mode="bilinear", align_corners=False)[0]
                out.append(r)
            return torch.cat(out, dim=1)

        # amodal = annotated (valid) but NOT visible in the frame (obj_kpts2d_mask=False)
        def _amodal(valid_b, vis2d, b):
            if vis2d is None:
                return None
            return valid_b & (~vis2d[b].bool().cpu())

        imgs = []
        for b in range(B):
            valid   = kpts_valid[b].cpu()
            correct = is_correct[b].cpu()
            src_amodal  = _amodal(valid, batch.src_obj_kpts2d_mask, b)
            trgt_amodal = _amodal(valid, batch.trgt_obj_kpts2d_mask, b)

            # ── top row: correspondences on the frame RGB ─────────────────────
            frame_row = None
            if (batch.src_rgb is not None and batch.trgt_rgb is not None
                    and batch.src_cam_intr4x4 is not None and batch.trgt_cam_intr4x4 is not None):
                try:
                    src_uv  = _proj_frame(batch.src_obj_kpts3d[b], batch.src_cam_tform4x4_obj_ncds[b], batch.src_cam_intr4x4[b])
                    gt_uv   = _proj_frame(batch.trgt_obj_kpts3d[b], batch.trgt_cam_tform4x4_obj_ncds[b], batch.trgt_cam_intr4x4[b])
                    pred_uv = _proj_frame(pred_trgt_kpts_cam_t[b], eye, batch.trgt_cam_intr4x4[b])
                    frame_row = _draw_corr_panels(
                        _to_uint8_img(src_frames[b]), _to_uint8_img(trgt_frames[b]),
                        src_uv, gt_uv, pred_uv, valid, correct,
                        left_amodal=src_amodal, right_amodal=trgt_amodal,
                    )
                except Exception:
                    frame_row = None

            # ── bottom row: top-down renders of source + target objects ───────
            src_mesh  = batch.src_meshes[b]  if batch.src_meshes  is not None else None
            trgt_mesh = batch.trgt_meshes[b] if batch.trgt_meshes is not None else None
            top_row = _render_topdown_corr_img(
                src_mesh, trgt_mesh,
                batch.src_obj_kpts3d[b], batch.trgt_obj_kpts3d[b], pred_kpts_ncds[b],
                valid, correct, src_amodal=src_amodal, trgt_amodal=trgt_amodal,
            )

            combined = _vstack([frame_row, top_row])
            if combined is not None:
                imgs.append(combined)

        if not imgs:
            return FrameObjectPairQualitBatch()
        H = max(i.shape[1] for i in imgs)
        W = max(i.shape[2] for i in imgs)
        imgs = [
            i if (i.shape[1] == H and i.shape[2] == W)
            else F.interpolate(i.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
            for i in imgs
        ]
        return FrameObjectPairQualitBatch(imgs=torch.stack(imgs))

    # ── qualitative: variant b — featmap PCA correspondence panels ─────────────

    @staticmethod
    def _featmap_pca_imgs(src_fm, trgt_fm, out_hw_src, out_hw_trgt):
        """Joint PCA over both featmaps → two (H, W, 3) uint8 images."""
        import numpy as np
        import torch.nn.functional as F
        Fd, Hf, Wf = src_fm.shape
        flat = torch.cat([
            src_fm.reshape(Fd, -1).T, trgt_fm.reshape(Fd, -1).T,
        ], dim=0)  # (2*Hf*Wf, F)
        with torch.no_grad():
            _, _, Vt = torch.pca_lowrank(flat, q=3, center=True)
            proj = flat @ Vt  # (2*Hf*Wf, 3)
        lo, hi = proj.min(dim=0).values, proj.max(dim=0).values
        proj = (proj - lo) / (hi - lo + 1e-6)
        src_pca  = proj[: Hf * Wf].T.reshape(3, Hf, Wf)
        trgt_pca = proj[Hf * Wf:].T.reshape(3, Hf, Wf)
        out = []
        for pca, (H, W) in ((src_pca, out_hw_src), (trgt_pca, out_hw_trgt)):
            up = F.interpolate(pca[None], size=(H, W), mode="nearest")[0]
            out.append((up.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        return out

    def _render_qualit_featmap2d(
        self, batch, qualit, src_ncds, feat2d_src_uv, feat2d_pred_uv,
        kpts_valid, is_correct,
    ) -> None:
        """query|target joint-PCA featmap panels with the featmap-NN
        correspondences drawn on top → qualit.extra['correspondences_featmap2d']."""
        if (qualit is None or is_correct is None or feat2d_src_uv is None
                or batch.src_featmap is None or batch.trgt_featmap is None
                or batch.trgt_cam_intr4x4 is None):
            return
        B = kpts_valid.shape[0]
        imgs = []
        for b in range(B):
            try:
                hw_s = _img_hw(batch.src_rgb, b)
                hw_t = _img_hw(batch.trgt_rgb, b) if batch.trgt_rgb is not None else hw_s
                left, right = self._featmap_pca_imgs(
                    batch.src_featmap[b].float().cpu(),
                    batch.trgt_featmap[b].float().cpu(), hw_s, hw_t,
                )
                gt_uv = _proj_frame(
                    batch.trgt_obj_kpts3d[b], batch.trgt_cam_tform4x4_obj_ncds[b],
                    batch.trgt_cam_intr4x4[b],
                )
                row = _draw_corr_panels(
                    left, right,
                    feat2d_src_uv[b].numpy(), gt_uv, feat2d_pred_uv[b].numpy(),
                    kpts_valid[b].cpu(), is_correct[b].cpu(),
                )
                imgs.append(row)
            except Exception:
                continue
        if imgs:
            qualit.extra["correspondences_featmap2d"] = imgs

    # ── qualitative: variant c — NOCS-colored mesh correspondence panels ───────

    def _render_qualit_mesh(
        self, batch, qualit, src_ncds, trgt_ncds, pred_src, pred_trgt,
        pred_mesh_kpts, kpts_valid, is_correct,
    ) -> None:
        """Predicted meshes rendered as per-pixel NOCS-0c coordinates colorised
        pixelwise via nocs_0c_to_rgb, posed with the predicted poses and overlaid
        on the frame RGB (row 1) and top-down (row 2), with the mesh-barycentric
        correspondences drawn on top → qualit.extra['correspondences_mesh']."""
        if qualit is None or is_correct is None or pred_mesh_kpts is None:
            return
        import numpy as np
        import torch.nn.functional as F
        from o3b.cv.geometry.transform import transf3d_broadcast, inv_tform4x4

        src_meshes  = batch.src_pred_mesh  if batch.src_pred_mesh  is not None else batch.src_meshes
        trgt_meshes = batch.trgt_pred_mesh if batch.trgt_pred_mesh is not None else batch.trgt_meshes
        if src_meshes is None or trgt_meshes is None:
            return

        def _vstack(rows):
            rows = [r for r in rows if r is not None]
            if not rows:
                return None
            W = max(r.shape[2] for r in rows)
            out = []
            for r in rows:
                if r.shape[2] != W:
                    new_h = max(1, int(round(r.shape[1] * W / r.shape[2])))
                    r = F.interpolate(r.unsqueeze(0), size=(new_h, W), mode="bilinear", align_corners=False)[0]
                out.append(r)
            return torch.cat(out, dim=1)

        def _rgb_np(imgs, b, H, W):
            """Frame RGB as (H, W, 3) float [0,1]; grey fallback."""
            if imgs is None:
                return np.full((H, W, 3), 0.92, dtype=np.float32)
            im = imgs[b].float().cpu()
            if im.max() > 1.5:
                im = im / 255.0
            return im.clamp(0, 1).permute(1, 2, 0).numpy()

        _GREY = 0.92
        eye = torch.eye(4)
        B = kpts_valid.shape[0]
        imgs = []
        for b in range(B):
            try:
                m_src, m_trgt = src_meshes[b], trgt_meshes[b]
                if m_src is None or m_trgt is None:
                    continue
                # same NCDS normalisation as _pred_mesh_corresp, so the rendered
                # surface (and its NOCS-0c colors) match the geometry the
                # correspondences were computed on
                m_src  = _mesh_ncds(m_src,  self.mesh_ncds_rescale, self.mesh_ncds_recenter)
                m_trgt = _mesh_ncds(m_trgt, self.mesh_ncds_rescale, self.mesh_ncds_recenter)
                valid, correct = kpts_valid[b].cpu(), is_correct[b].cpu()

                # ── row 1: GT viewpoints — NOCS render overlaid on the frame RGB ─
                gt_row = None
                if batch.src_cam_intr4x4 is not None and batch.trgt_cam_intr4x4 is not None:
                    H_s, W_s = _img_hw(batch.src_rgb, b) if batch.src_rgb is not None else (256, 256)
                    H_t, W_t = _img_hw(batch.trgt_rgb, b) if batch.trgt_rgb is not None else (256, 256)
                    src_k,  trgt_k  = batch.src_cam_intr4x4[b].float().cpu(), batch.trgt_cam_intr4x4[b].float().cpu()
                    src_t,  trgt_t  = src_ncds[b].float().cpu(), trgt_ncds[b].float().cpu()
                    # meshes are posed with the predicted poses (matching
                    # _pred_mesh_corresp, so the drawn pred correspondences lie on
                    # the rendered surface); GT keypoints stay projected with the
                    # GT poses so they sit at their true image locations.
                    c_s, msk_s = _render_mesh_nocs(m_src,  pred_src[b].float().cpu(),  src_k,  H_s, W_s)
                    c_t, msk_t = _render_mesh_nocs(m_trgt, pred_trgt[b].float().cpu(), trgt_k, H_t, W_t)
                    if c_s is not None and c_t is not None:
                        left  = _nocs_on_bg(c_s, msk_s, _rgb_np(batch.src_rgb,  b, H_s, W_s), alpha=0.65)
                        right = _nocs_on_bg(c_t, msk_t, _rgb_np(batch.trgt_rgb, b, H_t, W_t), alpha=0.65)
                        src_uv  = _proj_opengl(batch.src_obj_kpts3d[b],  src_t,  src_k)
                        gt_uv   = _proj_opengl(batch.trgt_obj_kpts3d[b], trgt_t, trgt_k)
                        pred_uv = _proj_opengl(pred_mesh_kpts[b], eye, trgt_k)  # already trgt-cam space
                        gt_row = _draw_corr_panels(left, right, src_uv, gt_uv, pred_uv, valid, correct)

                # ── row 2: top-down — NOCS render on grey ─────────────────────
                top_row = None
                cam_t, cam_k = _topdown_camera()
                c_s, msk_s = _render_mesh_nocs(m_src,  cam_t, cam_k, 256, 256)
                c_t, msk_t = _render_mesh_nocs(m_trgt, cam_t, cam_k, 256, 256)
                if c_s is not None and c_t is not None:
                    grey = np.full((256, 256, 3), _GREY, dtype=np.float32)
                    left  = _nocs_on_bg(c_s, msk_s, grey)
                    right = _nocs_on_bg(c_t, msk_t, grey)
                    pred_kpts_ncds = transf3d_broadcast(
                        pred_mesh_kpts[b].float(), inv_tform4x4(pred_trgt[b].float().cpu()),
                    )
                    src_uv  = _proj_opengl(batch.src_obj_kpts3d[b],  cam_t, cam_k)
                    gt_uv   = _proj_opengl(batch.trgt_obj_kpts3d[b], cam_t, cam_k)
                    pred_uv = _proj_opengl(pred_kpts_ncds, cam_t, cam_k)
                    top_row = _draw_corr_panels(left, right, src_uv, gt_uv, pred_uv, valid, correct)

                combined = _vstack([gt_row, top_row])
                if combined is not None:
                    imgs.append(combined)
            except Exception:
                continue
        if imgs:
            qualit.extra["correspondences_mesh"] = imgs
