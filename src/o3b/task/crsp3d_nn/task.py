from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

from o3b.task.task import OD3D_Task, register_task
from o3b.data.datatypes.object import ObjectPairBatch
from o3b.task.datatypes.object_pair_quant import ObjectPairQuantBatch
from o3b.task.datatypes.object_pair_qualit import ObjectPairQualitBatch


# ── target keypoint symmetry (discrete + continuous rotational) ───────────────
# Mirrors CamCrsp3DNNTask's keypoint symmetry evaluation (o3b.task.cam_crsp3d_nn.task),
# adapted for this task's vertex-anchored GT: every symmetric candidate is snapped to
# its nearest target vertex (needed for the geodesic-distance metric below) before
# picking the candidate closest to the prediction.

def _kpts_sym_candidates(trgt_kpts: Tensor, obj_tform4x4_obj_syms: Tensor) -> Tensor:
    """(B, K, 3) target keypoints (object space) + (B, S, 4, 4) discrete symmetry
    transforms (see o3b.cv.metric.pose.get_obj_tform4x4_obj_sym) -> (B, K, S, 3)
    discrete-symmetric keypoint candidates, same object space (candidate 0 is
    always the identity)."""
    from o3b.cv.geometry.transform import transf3d_broadcast
    return transf3d_broadcast(
        trgt_kpts[:, :, None, :], obj_tform4x4_obj_syms[:, None, :, :, :],
    )


def _kpts_sym_min_dist_to_verts(
    kpts_syms: Tensor, axis6d_syms: Tensor, axis6d_syms_mask: Tensor,
    pred_pos: Tensor, trgt_verts: Tensor, trgt_verts_mask: "Optional[Tensor]",
) -> "Tuple[Tensor, Tensor, Tensor]":
    """For each keypoint, pick the discrete/continuous symmetric target candidate
    whose nearest-target-vertex position is closest to the prediction.

    kpts_syms: (B, K, S, 3) discrete-symmetric candidates (object space, pre
      vertex-snap; candidate 0 is the identity) — see _kpts_sym_candidates.
    axis6d_syms: (B, 3, 6) per-axis-slot (offset, direction), object space, from
      o3b.cv.metric.pose.get_obj_axis6d_with_mask.
    axis6d_syms_mask: (B, 3) bool — True where that axis slot is continuously
      (rotationally) symmetric.
    pred_pos: (B, K, 3) predicted target vertex positions.
    trgt_verts: (B, V, 3); trgt_verts_mask: (B, V) bool or None.

    For a sample with exactly one continuous rotational axis, each discrete
    candidate is first rotated to its closest point (around that axis) to the
    prediction — see get_closest_B_to_A_rotated_around_axis6d — before being
    snapped to its nearest target vertex; samples with more than one such axis
    fall back to the discrete candidates only.

    Returns (gt_vert (B, K) long, gt_pos (B, K, 3), dist (B, K)).
    """
    from o3b.cv.metric.pose import get_closest_B_to_A_rotated_around_axis6d

    B, K, S, _ = kpts_syms.shape
    cand = kpts_syms.clone()
    for b in range(B):
        mask_b = axis6d_syms_mask[b]
        if mask_b.sum() != 1:
            if mask_b.sum() > 1:
                logger.warning("keypoint symmetry: >1 continuous rotational axis is not "
                               "supported, falling back to discrete candidates only.")
            continue
        axis6d_b = axis6d_syms[b][mask_b][0]                              # (6,)
        A = pred_pos[b][:, None, :].expand(K, S, 3).reshape(-1, 3)
        Bp = cand[b].reshape(-1, 3)
        cand[b] = get_closest_B_to_A_rotated_around_axis6d(
            A=A, B=Bp, axis6d=axis6d_b[None].expand(K * S, 6),
        ).reshape(K, S, 3)

    # nearest target vertex for every candidate, vectorized over the batch
    cand_flat = cand.reshape(B, K * S, 3)
    dists = torch.cdist(cand_flat, trgt_verts)                           # (B, K*S, V)
    if trgt_verts_mask is not None:
        dists = dists.masked_fill(~trgt_verts_mask.unsqueeze(1), float("inf"))
    gt_vert_flat = dists.argmin(dim=2)                                   # (B, K*S)
    gt_pos_flat  = torch.gather(
        trgt_verts, dim=1, index=gt_vert_flat.unsqueeze(-1).expand(-1, -1, 3),
    )                                                                     # (B, K*S, 3)
    gt_vert = gt_vert_flat.reshape(B, K, S)
    gt_pos  = gt_pos_flat.reshape(B, K, S, 3)

    dist_all = (gt_pos - pred_pos[:, :, None, :]).norm(dim=-1)           # (B, K, S)
    sym_ids  = dist_all.argmin(dim=-1)                                   # (B, K)

    gt_vert_sel = torch.gather(gt_vert, dim=2, index=sym_ids.unsqueeze(-1)).squeeze(-1)
    gt_pos_sel  = torch.gather(
        gt_pos, dim=2, index=sym_ids[:, :, None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    dist = torch.gather(dist_all, dim=2, index=sym_ids.unsqueeze(-1)).squeeze(-1)

    return gt_vert_sel, gt_pos_sel, dist


# ── geodesic helper ───────────────────────────────────────────────────────────

def _mesh_sqrt_area(mesh) -> float:
    """Return sqrt of total surface area of the mesh."""
    import math
    verts = mesh.verts.float().cpu()
    faces = mesh.faces.long().cpu()
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1).sum().item()
    return math.sqrt(max(area, 1e-12))


def _geodesic_pairwise(mesh, src_indices, dst_indices) -> "np.ndarray":
    """Geodesic distance for each (src_indices[i], dst_indices[i]) pair on mesh.

    Returns float32 numpy array of shape (N,).
    """
    import numpy as np
    from torch_geometric.utils import geodesic_distance

    pos  = mesh.verts.float().cpu()
    face = mesh.faces.t().contiguous().long().cpu()
    dists = geodesic_distance(
        pos, face,
        src=torch.tensor(src_indices, dtype=torch.long),
        dst=torch.tensor(dst_indices, dtype=torch.long),
        norm=True,
    )
    return dists.cpu().numpy().astype(np.float32)


# ── part correspondence rendering ─────────────────────────────────────────────

def _render_part_imgs(
    src_feats:       "Tensor",            # (B, V_src, F)
    trgt_feats:      "Tensor",            # (B, V_trgt, F)
    src_verts_mask:  "Optional[Tensor]",  # (B, V_src)  bool or None
    trgt_verts_mask: "Optional[Tensor]",  # (B, V_trgt) bool or None
    src_part_id:     "Tensor",            # (B, V_src) int64
    src_meshes:      list,
    trgt_meshes:     list,
    H: int = 256,
    W: int = 256,
) -> "Optional[Tensor]":
    """Return (B, 3, 2H, 2W) images.

    Left panel:  source mesh colored by source part label.
    Right panel: target mesh where each vertex is colored by the part label of
                 its nearest source vertex in feature space.
    """
    try:
        import numpy as np
        from dataclasses import replace as _dc_replace
        from o3b.io import _mesh_to_trimesh
        from o3b.cv.visual.show import (
            render_trimesh_to_tensor,
            get_default_camera_intrinsics_from_img_size,
        )
        from o3b.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
        from o3b.data.datatypes.object import _part_id_to_vert_colors
    except ImportError:
        return None

    B = src_feats.shape[0]
    imgs_out = []
    cam_ts = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=2, dist=5.0).float()
    cam_k  = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()

    def render_panel(mesh, cam_t):
        if mesh is None:
            return np.full((H, W, 3), 235, dtype=np.uint8)
        try:
            rgb, _ = render_trimesh_to_tensor(
                _mesh_to_trimesh(mesh), cam_k, cam_t, H, W, rgb_bg=[0.92, 0.92, 0.92],
            )
            return (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        except Exception:
            return np.full((H, W, 3), 235, dtype=np.uint8)

    for b in range(B):
        sm = src_meshes[b]
        tm = trgt_meshes[b]
        if sm is None or tm is None:
            continue

        V_src_real  = sm.verts.shape[0]
        V_trgt_real = tm.verts.shape[0]
        src_pid_b   = src_part_id[b, :V_src_real].cpu()

        src_colors = _part_id_to_vert_colors(src_pid_b)

        sf = src_feats[b, :V_src_real].float()
        tf = trgt_feats[b, :V_trgt_real].float()
        feat_dists = torch.cdist(tf, sf)  # (V_trgt, V_src)
        if src_verts_mask is not None:
            feat_dists = feat_dists.masked_fill(~src_verts_mask[b, :V_src_real].unsqueeze(0), float("inf"))
        nn_src      = feat_dists.argmin(dim=1).cpu()
        trgt_colors = _part_id_to_vert_colors(src_pid_b[nn_src], reference_part_id=src_pid_b)

        src_colored  = _dc_replace(sm, vert_colors=src_colors)
        trgt_colored = _dc_replace(tm, vert_colors=trgt_colors)

        rows = []
        for v_idx in range(cam_ts.shape[0]):
            cam_t = cam_ts[v_idx]
            rows.append(np.concatenate([render_panel(src_colored, cam_t),
                                        render_panel(trgt_colored, cam_t)], axis=1))
        canvas = np.concatenate(rows, axis=0).astype(np.float32) / 255.0
        imgs_out.append(torch.from_numpy(canvas).permute(2, 0, 1))

    return torch.stack(imgs_out) if imgs_out else None


# ── feature-colored mesh rendering ─────────────────────────────────────────────

def _feat_vert_colors(feats: Tensor) -> Tensor:
    """(V, F) → (V, 3) float32 in [0, 1].

    F <= 3: per-channel min-max normalized features directly (zero-padded to 3
            channels if F < 3).
    F > 3:  first 3 PCA components, per-channel min-max normalized.
    """
    V, F = feats.shape
    flat = torch.nan_to_num(feats.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if F > 3:
        centered = flat - flat.mean(dim=0)
        _, _, Vmat = torch.pca_lowrank(centered, q=3, niter=4)
        colors = centered @ Vmat[:, :3]
    else:
        colors = flat if F == 3 else torch.cat([flat, torch.zeros(V, 3 - F)], dim=1)
    colors = colors.clone()
    for c in range(3):
        ch  = colors[:, c]
        rng = ch.max() - ch.min()
        colors[:, c] = (ch - ch.min()) / rng if rng > 0 else 0.5
    return colors.clamp(0, 1)


def _feat_colored_mesh_pairs(
    src_feats:   Tensor,  # (B, V_src, F)
    trgt_feats:  Tensor,  # (B, V_trgt, F)
    src_meshes:  list,
    trgt_meshes: list,
) -> "Tuple[list, list]":
    """Return (src_meshes, trgt_meshes) copies with vert_colors replaced by
    per-vertex feature colors (see _feat_vert_colors), computed jointly over
    src+trgt features so the two meshes stay color-comparable."""
    from dataclasses import replace as _dc_replace

    B = src_feats.shape[0]
    src_out, trgt_out = [], []
    for b in range(B):
        sm, tm = src_meshes[b], trgt_meshes[b]
        if sm is None or tm is None:
            src_out.append(sm)
            trgt_out.append(tm)
            continue
        V_src, V_trgt = sm.verts.shape[0], tm.verts.shape[0]
        sf = src_feats[b, :V_src].float().cpu()
        tf = trgt_feats[b, :V_trgt].float().cpu()
        combined = _feat_vert_colors(torch.cat([sf, tf], dim=0))
        src_out.append(_dc_replace(sm, vert_colors=combined[:V_src]))
        trgt_out.append(_dc_replace(tm, vert_colors=combined[V_src:]))
    return src_out, trgt_out


# ── qualitative rendering ─────────────────────────────────────────────────────

def _render_corr_imgs(
    left_verts:      Tensor,            # (B, V_left, 3)   source verts
    right_verts:     Tensor,            # (B, V_right, 3)  target verts
    left_kpts:       Tensor,            # (B, K, 3)  source keypoints (GT)
    right_kpts:      Tensor,            # (B, K, 3)  predicted target positions
    euc_dist:        Tensor,            # (B, K)     Euclidean error
    trgt_max_dim:    Tensor,            # (B,)
    kpts_valid:      Tensor,            # (B, K) bool
    left_verts_mask:  Optional[Tensor],
    right_verts_mask: Optional[Tensor],
    right_kpts_gt: Optional[Tensor] = None,  # (B, K, 3)  GT target positions
    left_meshes:  Optional[list] = None,
    right_meshes: Optional[list] = None,
    H: int = 256,
    W: int = 256,
) -> Optional[Tensor]:
    """Return (B, 3, 2H, 2W) float32 [0,1] images.

    Left panel:  source mesh with GT source keypoints (filled dot, per-keypoint color).
    Right panel: target mesh with the GT target position (hollow ring, same color)
                 and the predicted target position (filled dot, same color).
    Line color: green = Euclidean error < 0.1 * trgt_max_dim, blue = wrong — matches
    CamCrsp3DNNTask's _draw_corr_panels styling.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        from o3b.io import _mesh_to_trimesh
        from o3b.cv.visual.show import (
            render_trimesh_to_tensor,
            get_default_camera_intrinsics_from_img_size,
        )
        from o3b.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
        from o3b.cv.visual.draw import get_colors
    except ImportError:
        return None

    B, K, _ = left_kpts.shape
    imgs_out = []
    r_dot = max(4, H // 52)
    kpt_colors = (get_colors(K) * 255).round().int().tolist()  # (K, 3) distinct per-keypoint

    cam_ts = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=2, dist=5.0).float()
    cam_k  = get_default_camera_intrinsics_from_img_size(W, H, fov_x=25).float()
    fx, fy  = cam_k[0, 0].item(), cam_k[1, 1].item()
    cx, cy  = cam_k[0, 2].item(), cam_k[1, 2].item()

    def project(pts_3d: Tensor, cam_t: Tensor):
        pts_h   = torch.cat([pts_3d.float().cpu(), torch.ones(len(pts_3d), 1)], dim=1)
        pts_cam = (cam_t @ pts_h.T).T[:, :3]
        z = (-pts_cam[:, 2]).clamp(min=1e-6).numpy()
        return fx * pts_cam[:, 0].numpy() / z + cx, fy * (-pts_cam[:, 1]).numpy() / z + cy

    def render_panel(mesh, cam_t):
        if mesh is not None:
            try:
                rgb, _ = render_trimesh_to_tensor(
                    _mesh_to_trimesh(mesh), cam_k, cam_t, H, W, rgb_bg=[0.92, 0.92, 0.92],
                )
                return (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            except Exception:
                pass
        return np.full((H, W, 3), 235, dtype=np.uint8)

    for b in range(B):
        lk  = left_kpts[b].float().cpu()
        rk  = right_kpts[b].float().cpu()
        rk_gt = right_kpts_gt[b].float().cpu() if right_kpts_gt is not None else None
        vm  = kpts_valid[b].cpu()
        threshold  = 0.1 * trgt_max_dim[b].item()
        is_correct = euc_dist[b].cpu() < threshold

        lm = left_meshes[b]  if left_meshes  is not None else None
        rm = right_meshes[b] if right_meshes is not None else None

        rows = []
        for v_idx in range(cam_ts.shape[0]):
            cam_t = cam_ts[v_idx]
            canvas = np.concatenate([render_panel(lm, cam_t), render_panel(rm, cam_t)], axis=1)
            pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil)
            draw.line([(W, 0), (W, H - 1)], fill=(120, 120, 120), width=2)

            lu, lv = project(lk, cam_t)
            ru, rv = project(rk, cam_t)
            if rk_gt is not None:
                gu, gv = project(rk_gt, cam_t)

            for k_i in range(K):
                if not bool(vm[k_i]):
                    continue
                line_color = (0, 180, 0) if bool(is_correct[k_i]) else (0, 100, 220)
                kpt_color  = tuple(kpt_colors[k_i])
                lx = int(np.clip(lu[k_i], 0, W - 1))
                ly = int(np.clip(lv[k_i], 0, H - 1))
                rx = int(np.clip(ru[k_i], 0, W - 1)) + W
                ry = int(np.clip(rv[k_i], 0, H - 1))
                draw.line([(lx, ly), (rx, ry)], fill=line_color, width=1)
                if rk_gt is not None:
                    gx = int(np.clip(gu[k_i], 0, W - 1)) + W
                    gy = int(np.clip(gv[k_i], 0, H - 1))
                    draw.ellipse([(gx - r_dot, gy - r_dot), (gx + r_dot, gy + r_dot)],
                                 outline=kpt_color, width=2)
                for px, py in [(lx, ly), (rx, ry)]:
                    draw.ellipse([(px - r_dot, py - r_dot), (px + r_dot, py + r_dot)],
                                 fill=kpt_color, outline=(0, 0, 0))
            rows.append(np.array(pil))

        canvas = np.concatenate(rows, axis=0).astype(np.float32) / 255.0
        imgs_out.append(torch.from_numpy(canvas).permute(2, 0, 1))

    return torch.stack(imgs_out) if imgs_out else None


# ── task ──────────────────────────────────────────────────────────────────────

@register_task("Crsp3DNNTask")
class Crsp3DNNTask(OD3D_Task):
    """Feature-based nearest-neighbour 3D correspondence.

    Pipeline:
      source (query) feat  →  nearest-neighbour target feat  →  pred target vertex

    Keypoint metric:
      query  = nearest source vertex to each GT source keypoint (3D)
      GT     = nearest target vertex to each GT target keypoint (3D)
      error  = distance between pred target vertex and GT target vertex

    Part metric:
      query  = source vertices with valid part label (src_part_id >= 0)
      GT     = nearest correct-part target vertex to pred target vertex (Euclidean)
      error  = Euclidean / geodesic distance from pred to GT
      PCK    = fraction where pred target vertex has the correct part label
    """

    def __init__(self, feat_metric: str = "euclidean", **kwargs):
        # feat_metric: "euclidean" (default, raw cdist) or "cosine"
        # (L2-normalize features before the nearest-neighbour cdist). Must be an
        # explicit named param so build_task's config forwarding picks it up.
        self.feat_metric = feat_metric or "euclidean"

    def forward(self, batch: ObjectPairBatch, return_qualit: bool = True) -> Tuple[ObjectPairQuantBatch, ObjectPairQualitBatch]:
        src_verts       = batch.src_verts3d             # (B, V_src, 3)
        src_feats       = batch.src_verts3d_feats       # (B, V_src, F)
        trgt_verts      = batch.trgt_verts3d            # (B, V_trgt, 3)
        trgt_feats      = batch.trgt_verts3d_feats      # (B, V_trgt, F)
        src_verts_mask  = batch.src_verts3d_feats_mask  # (B, V_src)  bool or None
        trgt_verts_mask = batch.trgt_verts3d_feats_mask # (B, V_trgt) bool or None
        src_kpts        = batch.src_obj_kpts3d           # (B, K, 3)
        trgt_kpts       = batch.trgt_obj_kpts3d          # (B, K, 3)
        src_kpts_mask   = batch.src_obj_kpts3d_mask
        trgt_kpts_mask  = batch.trgt_obj_kpts3d_mask
        src_part_id     = batch.src_verts3d_part_id     # (B, V_src)  int64 or None
        trgt_part_id    = batch.trgt_verts3d_part_id    # (B, V_trgt) int64 or None

        has_kpts  = all(x is not None for x in (src_kpts, trgt_kpts, src_verts, src_feats, trgt_verts, trgt_feats))
        has_parts = (src_part_id is not None and trgt_part_id is not None
                     and all(x is not None for x in (src_verts, src_feats, trgt_verts, trgt_feats)))
        has_part_viz = (src_part_id is not None and src_feats is not None and trgt_feats is not None
                        and batch.src_meshes is not None and batch.trgt_meshes is not None)

        if not has_kpts and not has_parts:
            return ObjectPairQuantBatch(), ObjectPairQualitBatch()

        B   = (src_verts if src_verts is not None else trgt_verts).shape[0]
        dev = (src_verts if src_verts is not None else trgt_verts).device

        # ── target bounding-box size (PCK threshold) ──────────────────────────
        tv = trgt_verts.float()
        if trgt_verts_mask is not None:
            m3  = trgt_verts_mask.unsqueeze(-1)
            trgt_max_dim = (
                tv.masked_fill(~m3, float("-inf")).amax(dim=1)
                - tv.masked_fill(~m3, float("inf")).amin(dim=1)
            ).amax(dim=1)
        else:
            trgt_max_dim = (tv.amax(dim=1) - tv.amin(dim=1)).amax(dim=1)  # (B,)

        # ── shared: query feats → nearest target vert ─────────────────────────
        def feat_nn_batch(query_feats_b, b):
            """(Q, F) → (Q,) target vert indices via feature nearest-neighbour."""
            qf = query_feats_b.float()
            tf = trgt_feats[b].float()
            if self.feat_metric == "cosine":
                # euclidean NN on L2-normalized features == cosine-similarity NN
                qf = torch.nn.functional.normalize(qf, dim=-1)
                tf = torch.nn.functional.normalize(tf, dim=-1)
            dists = torch.cdist(qf, tf)  # (Q, V_trgt)
            if trgt_verts_mask is not None:
                dists = dists.masked_fill(~trgt_verts_mask[b].unsqueeze(0), float("inf"))
            return dists.argmin(dim=1)  # (Q,)

        # ═════════════════════════════════════════════════════════════════════
        # KEYPOINTS
        # ═════════════════════════════════════════════════════════════════════
        quant_kpts       = {}
        pred_trgt_kpt_pos = None

        if has_kpts:
            B_k, K, _ = src_kpts.shape
            B_idx      = torch.arange(B_k, device=dev)[:, None].expand(B_k, K)

            # valid mask
            if src_kpts_mask is not None and trgt_kpts_mask is not None:
                kpts_valid = src_kpts_mask & trgt_kpts_mask
            else:
                kpts_valid = src_kpts_mask if src_kpts_mask is not None else \
                             trgt_kpts_mask if trgt_kpts_mask is not None else \
                             torch.ones(B_k, K, dtype=torch.bool, device=dev)
            n_valid = kpts_valid.float().sum(dim=1).clamp(min=1)

            # step 1: src_kpt → nearest src_vert (3D) → query vert
            src_kpt_to_vert = torch.cdist(src_kpts.float(), src_verts.float())  # (B, K, V_src)
            if src_verts_mask is not None:
                src_kpt_to_vert = src_kpt_to_vert.masked_fill(~src_verts_mask.unsqueeze(1), float("inf"))
            nn_src_vert = src_kpt_to_vert.argmin(dim=2)  # (B, K)

            # step 2: query src feat → nearest trgt vert (feature space)
            query_feats = src_feats[B_idx, nn_src_vert]  # (B, K, F)
            pred_trgt_kpt_vert = torch.stack([
                feat_nn_batch(query_feats[b], b) for b in range(B_k)
            ])  # (B, K)
            pred_trgt_kpt_pos = trgt_verts[B_idx, pred_trgt_kpt_vert]  # (B, K, 3)

            # GT: nearest trgt_vert to trgt_kpt (3D), extended over target keypoint
            # symmetry — the discrete/continuous symmetric candidate whose nearest
            # target vertex is closest to the prediction wins (see
            # _kpts_sym_min_dist_to_verts / CamCrsp3DNNTask's keypoint symmetry eval).
            from o3b.cv.metric.pose import get_obj_tform4x4_obj_sym, get_obj_axis6d_with_mask

            trgt_obj_syms = batch.trgt_obj_syms.float().to(dev) if batch.trgt_obj_syms is not None \
                else torch.ones(B_k, 3, device=dev)
            obj_tform4x4_obj_syms = get_obj_tform4x4_obj_sym(trgt_obj_syms)                     # (B, S, 4, 4)
            obj_axis6d_syms, obj_axis6d_syms_mask = get_obj_axis6d_with_mask(trgt_obj_syms)     # (B, 3, 6), (B, 3) bool

            trgt_kpts_syms = _kpts_sym_candidates(trgt_kpts.float(), obj_tform4x4_obj_syms.to(dev))  # (B, K, S, 3)

            gt_trgt_kpt_vert, gt_trgt_kpt_pos, kpts_trgt_euc_dist = _kpts_sym_min_dist_to_verts(
                trgt_kpts_syms, obj_axis6d_syms, obj_axis6d_syms_mask,
                pred_trgt_kpt_pos, trgt_verts.float(), trgt_verts_mask,
            )  # (B, K) long, (B, K, 3), (B, K)
            kpts_trgt_euc_dist_mean = (kpts_trgt_euc_dist * kpts_valid.float()).sum(dim=1) / n_valid

            # PCK @ 0.1 * trgt_max_dim
            kpts_trgt_pck01 = (
                (kpts_trgt_euc_dist < trgt_max_dim.unsqueeze(1) * 0.1).float()
                * kpts_valid.float()
            ).sum(dim=1) / n_valid

            # Geodesic error (pred → GT vert on target mesh), normalized by sqrt(surface area)
            kpts_trgt_geo_dist      = torch.zeros(B_k, K, dtype=torch.float32)
            kpts_trgt_geo_dist_norm = torch.zeros(B_k, K, dtype=torch.float32)
            if batch.trgt_meshes is not None:
                import numpy as np
                for b in range(B_k):
                    if batch.trgt_meshes[b] is None:
                        continue
                    valid_idx = kpts_valid[b].cpu().numpy().nonzero()[0]
                    if len(valid_idx) == 0:
                        continue
                    geo = _geodesic_pairwise(
                        batch.trgt_meshes[b],
                        pred_trgt_kpt_vert[b].cpu().numpy()[valid_idx],
                        gt_trgt_kpt_vert[b].cpu().numpy()[valid_idx],
                    )
                    sqrt_area = _mesh_sqrt_area(batch.trgt_meshes[b])
                    for j, k_i in enumerate(valid_idx):
                        d = float(geo[j])
                        if np.isfinite(d):
                            kpts_trgt_geo_dist[b, k_i]      = d
                            kpts_trgt_geo_dist_norm[b, k_i] = d / sqrt_area

            geo_t      = kpts_trgt_geo_dist.to(dev)
            geo_norm_t = kpts_trgt_geo_dist_norm.to(dev)
            kpts_trgt_geo_dist_mean      = (geo_t      * kpts_valid.float()).sum(dim=1) / n_valid
            kpts_trgt_geo_dist_norm_mean = (geo_norm_t * kpts_valid.float()).sum(dim=1) / n_valid

            # AUC: integrate PCK over normalized geodesic thresholds [0, 0.1]
            thresholds = torch.linspace(0, 0.1, 1000, device=dev)
            pck_per_thresh = (
                (geo_norm_t.unsqueeze(-1) < thresholds).float() * kpts_valid.float().unsqueeze(-1)
            ).sum(dim=1) / n_valid.unsqueeze(-1)
            kpts_trgt_geo_auc01 = pck_per_thresh.mean(dim=1)

            quant_kpts = dict(
                kpts_trgt_euc_dist           = kpts_trgt_euc_dist,
                kpts_trgt_geo_dist           = geo_t,
                kpts_mask                    = kpts_valid,
                kpts_trgt_pck01              = kpts_trgt_pck01,
                kpts_trgt_euc_dist_mean      = kpts_trgt_euc_dist_mean,
                kpts_trgt_geo_dist_mean      = kpts_trgt_geo_dist_mean,
                kpts_trgt_geo_dist_norm_mean = kpts_trgt_geo_dist_norm_mean,
                kpts_trgt_geo_auc01            = kpts_trgt_geo_auc01,
            )

        # ═════════════════════════════════════════════════════════════════════
        # PARTS
        # ═════════════════════════════════════════════════════════════════════
        quant_parts = {}

        if has_parts:
            import numpy as np

            parts_trgt_euc_dist_mean_list      = []
            parts_trgt_geo_dist_mean_list      = []
            parts_trgt_geo_dist_norm_mean_list = []
            parts_trgt_geo_dist_norm_raw_list  = []  # normalized, for AUC
            parts_trgt_pck_list                = []

            for b in range(B):
                src_pid_b  = src_part_id[b]   # (V_src,)
                trgt_pid_b = trgt_part_id[b]  # (V_trgt,)

                # query = src verts with valid part label
                valid_mask = src_pid_b >= 0
                if src_verts_mask is not None:
                    valid_mask = valid_mask & src_verts_mask[b]
                query_idx = valid_mask.nonzero(as_tuple=True)[0]  # (Q,)

                if query_idx.numel() == 0:
                    parts_trgt_euc_dist_mean_list.append(torch.tensor(0.0))
                    parts_trgt_geo_dist_mean_list.append(torch.tensor(float("nan")))
                    parts_trgt_geo_dist_norm_mean_list.append(torch.tensor(float("nan")))
                    parts_trgt_geo_dist_norm_raw_list.append(None)
                    parts_trgt_pck_list.append(torch.tensor(0.0))
                    continue

                query_parts_b = src_pid_b[query_idx]              # (Q,)
                query_feats_b = src_feats[b, query_idx]           # (Q, F)

                # step 1: query src feat → nearest trgt vert
                pred_trgt_vert = feat_nn_batch(query_feats_b, b)  # (Q,)
                pred_trgt_pid  = trgt_pid_b[pred_trgt_vert]       # (Q,)

                # PCK: pred target vert has same part label as query source vert
                parts_trgt_pck_list.append((pred_trgt_pid == query_parts_b).float().mean())

                # Euclidean GT: nearest correct-part target vert to pred (per unique part)
                pred_pos     = trgt_verts[b].float()[pred_trgt_vert]  # (Q, 3)
                trgt_verts_b = trgt_verts[b].float()                  # (V_trgt, 3)
                trgt_mask_b  = trgt_verts_mask[b] if trgt_verts_mask is not None else None

                euc_errors_t   = torch.full((query_idx.numel(),), float("nan"))
                geo_gt_vert_t  = torch.full((query_idx.numel(),), -1, dtype=torch.long)

                for part_p in query_parts_b.unique():
                    qi_mask = (query_parts_b == part_p)
                    correct  = (trgt_pid_b == part_p)
                    if trgt_mask_b is not None:
                        correct = correct & trgt_mask_b
                    correct_idx = correct.nonzero(as_tuple=True)[0]
                    if correct_idx.numel() == 0:
                        continue
                    d = torch.cdist(pred_pos[qi_mask], trgt_verts_b[correct_idx])  # (Q_p, C)
                    min_d, min_i = d.min(dim=1)
                    euc_errors_t[qi_mask]  = min_d
                    geo_gt_vert_t[qi_mask] = correct_idx[min_i]

                finite_euc = euc_errors_t[torch.isfinite(euc_errors_t)]
                parts_trgt_euc_dist_mean_list.append(
                    finite_euc.mean() if finite_euc.numel() > 0 else torch.tensor(0.0)
                )

                # Geodesic: pred → euclidean-GT vert on target mesh, normalized by sqrt(surface area)
                if batch.trgt_meshes is not None and batch.trgt_meshes[b] is not None:
                    valid_geo = (geo_gt_vert_t >= 0).nonzero(as_tuple=True)[0]
                    if valid_geo.numel() > 0:
                        geo_np = _geodesic_pairwise(
                            batch.trgt_meshes[b],
                            pred_trgt_vert[valid_geo].cpu().numpy(),
                            geo_gt_vert_t[valid_geo].cpu().numpy(),
                        )
                        sqrt_area  = _mesh_sqrt_area(batch.trgt_meshes[b])
                        geo_raw    = torch.from_numpy(geo_np)
                        geo_norm   = geo_raw / sqrt_area
                        finite_geo      = geo_raw [torch.isfinite(geo_raw)]
                        finite_geo_norm = geo_norm[torch.isfinite(geo_norm)]
                        parts_trgt_geo_dist_mean_list.append(
                            finite_geo.mean() if finite_geo.numel() > 0 else torch.tensor(float("nan"))
                        )
                        parts_trgt_geo_dist_norm_mean_list.append(
                            finite_geo_norm.mean() if finite_geo_norm.numel() > 0 else torch.tensor(float("nan"))
                        )
                        parts_trgt_geo_dist_norm_raw_list.append(
                            finite_geo_norm if finite_geo_norm.numel() > 0 else None
                        )
                        continue
                parts_trgt_geo_dist_mean_list.append(torch.tensor(float("nan")))
                parts_trgt_geo_dist_norm_mean_list.append(torch.tensor(float("nan")))
                parts_trgt_geo_dist_norm_raw_list.append(None)

            # AUC: integrate PCK over normalized geodesic thresholds [0, 0.1]
            thresholds_parts = torch.linspace(0, 0.1, 1000)
            parts_trgt_geo_auc01_list = [
                (geo_norm_raw.unsqueeze(0) < thresholds_parts.unsqueeze(1)).float().mean(dim=1).mean()
                if geo_norm_raw is not None else torch.tensor(float("nan"))
                for geo_norm_raw in parts_trgt_geo_dist_norm_raw_list
            ]

            quant_parts = dict(
                parts_trgt_euc_dist_mean      = torch.stack(parts_trgt_euc_dist_mean_list).to(dev),
                parts_trgt_geo_dist_mean      = torch.stack(parts_trgt_geo_dist_mean_list).to(dev),
                parts_trgt_geo_dist_norm_mean = torch.stack(parts_trgt_geo_dist_norm_mean_list).to(dev),
                parts_trgt_geo_auc01            = torch.stack(parts_trgt_geo_auc01_list).to(dev),
                parts_trgt_pck                = torch.stack(parts_trgt_pck_list).to(dev),
            )

        # ── qualitative: keypoint correspondence images ───────────────────────
        if not return_qualit:
            return ObjectPairQuantBatch(**quant_kpts, **quant_parts), None

        qualit_imgs = None
        if has_kpts and pred_trgt_kpt_pos is not None:
            qualit_imgs = _render_corr_imgs(
                left_verts       = src_verts,
                right_verts      = trgt_verts,
                left_kpts        = src_kpts.float(),
                right_kpts       = pred_trgt_kpt_pos,
                euc_dist         = kpts_trgt_euc_dist,
                trgt_max_dim     = trgt_max_dim,
                kpts_valid       = kpts_valid,
                left_verts_mask  = src_verts_mask,
                right_verts_mask = trgt_verts_mask,
                right_kpts_gt    = gt_trgt_kpt_pos,
                left_meshes      = batch.src_meshes,
                right_meshes     = batch.trgt_meshes,
            )

        # ── qualitative: part correspondence images ───────────────────────────
        part_imgs = None
        if has_part_viz:
            part_imgs = _render_part_imgs(
                src_feats, trgt_feats,
                src_verts_mask, trgt_verts_mask,
                src_part_id,
                batch.src_meshes, batch.trgt_meshes,
            )

        # ── qualitative: keypoint correspondences on feature-colored meshes ───
        feat_imgs = None
        if (has_kpts and pred_trgt_kpt_pos is not None
                and batch.src_meshes is not None and batch.trgt_meshes is not None):
            feat_src_meshes, feat_trgt_meshes = _feat_colored_mesh_pairs(
                src_feats, trgt_feats, batch.src_meshes, batch.trgt_meshes,
            )
            feat_imgs = _render_corr_imgs(
                left_verts       = src_verts,
                right_verts      = trgt_verts,
                left_kpts        = src_kpts.float(),
                right_kpts       = pred_trgt_kpt_pos,
                euc_dist         = kpts_trgt_euc_dist,
                trgt_max_dim     = trgt_max_dim,
                kpts_valid       = kpts_valid,
                left_verts_mask  = src_verts_mask,
                right_verts_mask = trgt_verts_mask,
                right_kpts_gt    = gt_trgt_kpt_pos,
                left_meshes      = feat_src_meshes,
                right_meshes     = feat_trgt_meshes,
            )

        quant  = ObjectPairQuantBatch(**quant_kpts, **quant_parts)
        qualit = ObjectPairQualitBatch(imgs=qualit_imgs, part_imgs=part_imgs, feat_imgs=feat_imgs)
        if has_kpts:
            qualit.trgt_src_vert_corr      = pred_trgt_kpt_vert
            qualit.trgt_src_vert_corr_mask = kpts_valid
        return quant, qualit
