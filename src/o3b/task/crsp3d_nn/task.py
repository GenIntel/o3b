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


# ── multi-view feature distances (H03) ────────────────────────────────────────

def _multiview_dists(query: Tensor, trgt: Tensor, view_dim: int, mode: str, chunk: int = 256,
                     n_views: Optional[int] = None) -> Tensor:
    """(Q, N*C) query and (V, N*C) target per-view features, NaN = view does
    not see the vertex → (Q, V) distances; inf where a side has no view.

    With ``n_views``, channels past N*C are one feature per vertex (a
    concatenated single-view part); they add as sqrt(D^2 + |x_a - x_b|^2).

    view_mean averages the visible views and takes the Euclidean distance;
    mean_of_min / min_of_min compare every visible view of the query with
    every visible view of each target vertex (see Crsp3DNNTask.__init__).
    """
    Q, V = query.shape[0], trgt.shape[0]
    if n_views is not None and query.shape[1] > n_views * view_dim:
        k = n_views * view_dim
        d = _multiview_dists(query[:, :k], trgt[:, :k], view_dim, mode, chunk)
        return (d ** 2 + torch.cdist(query[:, k:].float(), trgt[:, k:].float()) ** 2).sqrt()
    q = query.float().view(Q, -1, view_dim)                       # (Q, N, C)
    t = trgt.float().view(V, -1, view_dim)                        # (V, N, C)
    q_ok = ~torch.isnan(q).any(-1)                                # (Q, N)
    t_ok = ~torch.isnan(t).any(-1)                                # (V, N)
    q, t = q.nan_to_num(0.0), t.nan_to_num(0.0)
    out = torch.full((Q, V), float("inf"), device=trgt.device)

    if mode == "view_mean":
        qm = (q * q_ok[..., None]).sum(1) / q_ok.sum(1, keepdim=True).clamp(min=1)
        tm = (t * t_ok[..., None]).sum(1) / t_ok.sum(1, keepdim=True).clamp(min=1)
        d = torch.cdist(qm, tm)
        d[~q_ok.any(1)] = float("inf")
        d[:, ~t_ok.any(1)] = float("inf")
        return d

    t_sq = (t ** 2).sum(-1)                                       # (V, N)
    for k in range(Q):
        qk = q[k][q_ok[k]]                                        # (n_k, C) visible views of the query
        if qk.shape[0] == 0:
            continue                                              # never seen: stays inf
        q_sq = (qk ** 2).sum(-1)                                  # (n_k,)
        for v0 in range(0, V, chunk):
            tc, tc_ok = t[v0:v0 + chunk], t_ok[v0:v0 + chunk]     # (c, N, C), (c, N)
            d2 = q_sq[:, None, None] + t_sq[None, v0:v0 + chunk] - 2 * torch.einsum("ic,vjc->ivj", qk, tc)
            d = d2.clamp(min=0).sqrt().masked_fill(~tc_ok[None], float("inf"))  # (n_k, c, N)
            nearest = d.min(dim=2).values                         # (n_k, c): per query view, closest target view
            out[k, v0:v0 + chunk] = nearest.mean(0) if mode == "mean_of_min" else nearest.min(0).values
    return out


# ── task ──────────────────────────────────────────────────────────────────────

def _log_sinkhorn_uot(Z: Tensor, iters: int, rho: float) -> Tensor:
    """Unbalanced log-domain Sinkhorn with uniform (log 1) marginals and KL
    relaxation ``rho``, as Shape-of-You's ``log_sinkhorn_uot`` (stabilised)."""
    u = Z.new_zeros(Z.shape[0])
    v = Z.new_zeros(Z.shape[1])
    for _ in range(iters):
        u = rho * -torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = rho * -torch.logsumexp(Z + u.unsqueeze(1), dim=0)
        u = u - u.max()
        v = v - v.max()
    return Z + u.unsqueeze(1) + v.unsqueeze(0)


def _fgw_match(src_feats: Tensor, trgt_feats: Tensor, src_verts: Tensor, trgt_verts: Tensor, *,
               alpha: float = 0.7, n_anchors: int = 64, rounds: int = 5, iters: int = 100,
               rho: float = 0.75, anchor_q: float = 0.01, temp: float = 1.0,
               normalize_scale: bool = True) -> Tensor:
    """Fused Gromov-Wasserstein matching, Shape-of-You's zero-shot recipe
    (Im et al., CVPR 2026, github.com/SpatialAILab/shapeofyou,
    src/eval/evaluation.py), on mesh vertices instead of image patches.

    Semantic reward R_sem = -|f_i - f_j|^2, so its argmax is the plain feature
    nearest neighbour. A first unbalanced Sinkhorn on R_sem gives the plan Z.
    Then ``rounds`` times:

    - anchors: source vertices whose argmax match maps back (target -> source
      argmax) to within the ``anchor_q`` quantile of 3-D distance, best
      ``n_anchors`` by Z;
    - structure: R_str[i, j] = -sum_a |d_s(i, a_s) - d_t(j, a_t)|, the L1
      distance between the vertices' distance profiles to the anchors;
    - Z = Sinkhorn(alpha * minmax(R_sem) + (1 - alpha) * minmax(R_str)).

    d is the 3-D Euclidean distance on each shape, divided by the shape's
    largest bbox side when ``normalize_scale`` (the repo's VGGT points share
    one frame, two meshes need not). ``temp`` divides the rewards before each
    Sinkhorn; 1 is the repo's. Returns (V_src,) target vertex indices, argmax Z.
    """
    def minmax(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    fs, ft = src_feats.float(), trgt_feats.float()
    vs, vt = src_verts.float(), trgt_verts.float()
    if normalize_scale:
        vs = vs / (vs.amax(0) - vs.amin(0)).max().clamp(min=1e-8)
        vt = vt / (vt.amax(0) - vt.amin(0)).max().clamp(min=1e-8)
    r_sem = minmax(-torch.cdist(fs, ft) ** 2)
    Z = _log_sinkhorn_uot(r_sem / temp, iters, rho)
    for _ in range(rounds):
        fwd = Z.argmax(dim=1)                  # (V_s,)
        bwd = Z.argmax(dim=0)                  # (V_t,)
        cyc = (vs - vs[bwd[fwd]]).norm(dim=1)  # source -> target -> source drift
        cand = None
        for q in (anchor_q, min(2 * anchor_q, 0.5)):
            m = cyc <= torch.quantile(cyc, q)
            if m.any():
                cand = m.nonzero(as_tuple=True)[0]
                break
        if cand is None:
            flat = torch.topk(Z.flatten(), k=n_anchors).indices
            a_s, a_t = flat // Z.shape[1], flat % Z.shape[1]
        else:
            order = Z[cand, fwd[cand]].argsort(descending=True)[:n_anchors]
            a_s, a_t = cand[order], fwd[cand[order]]
        # distance profiles to the anchors, compared in L1 -> (V_s, V_t)
        r_str = minmax(-torch.cdist(torch.cdist(vs, vs[a_s]), torch.cdist(vt, vt[a_t]), p=1))
        Z = _log_sinkhorn_uot((alpha * r_sem + (1 - alpha) * r_str) / temp, iters, rho)
    return Z.argmax(dim=1)


def _fmap_match(src_feats: Tensor, trgt_feats: Tensor, src_verts: Tensor, trgt_verts: Tensor,
                src_faces: Tensor, trgt_faces: Tensor, *, k: int = 50, w_descr: float = 1.0,
                w_lap: float = 1e-2, w_dcomm: float = 1e-1, w_orient: float = 0.0,
                zoomout: int = 0, zoomout_step: int = 5) -> Tensor:
    """Functional-map matching with the features as descriptors, the recipe
    od3d's DenseMatcher / Diff3F code used (``compute_surface_map`` in
    od3d/methods/densematcher/diff3f/functional_map.py): pyFM, k = 50
    Laplacian eigenfunctions per mesh, fit weights descriptors 1, Laplacian
    commutativity 0.01, descriptor commutativity 0.1, no orientation term,
    then the spectral nearest-neighbour point map. ``zoomout`` > 0 adds that
    many ZoomOut steps of ``zoomout_step`` eigenfunctions (not in the old
    recipe). Needs pyfmaps (``pip install pyfmaps``, imports as pyFM).

    The cached remeshes duplicate vertices along UV seams (an fc16 mesh is
    ~110 pieces), which leaves the Laplacian with one zero eigenvalue per
    piece and the eigenbasis useless. So each mesh is welded first
    (``weld_mesh``; features averaged per welded vertex) and the map is
    computed between the largest connected pieces. The few source vertices
    off it (floating bits of 4-8 vertices) fall back to the feature nearest
    neighbour. Vertices are centred and divided by their largest bbox side,
    as the old code meant to. Returns (V_src,) target vertex indices.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from pyFM.functional import FunctionalMapping
    from pyFM.mesh import TriMesh

    from o3b.data.datatypes.mesh import mean_by_index, weld_mesh

    def prep(v, f, feats):
        v = v.detach().double().cpu().numpy()
        f = f.detach().long().cpu().numpy()
        vw, fw, inv, rep = weld_mesh(v, f)
        fw = fw[(fw[:, 0] != fw[:, 1]) & (fw[:, 1] != fw[:, 2]) & (fw[:, 0] != fw[:, 2])]
        n = len(vw)
        adj = sp.coo_matrix((np.ones(3 * len(fw)), (fw.ravel(), np.roll(fw, 1, axis=1).ravel())), shape=(n, n))
        _, lab = connected_components(adj, directed=False)
        main = np.flatnonzero(lab == np.bincount(lab).argmax())          # welded ids of the largest piece
        sub = np.full(n, -1)
        sub[main] = np.arange(len(main))
        fm = sub[fw]
        fm = fm[(fm >= 0).all(1)]
        vm = vw[main]
        lo, hi = vm.min(0), vm.max(0)
        vm = (vm - 0.5 * (lo + hi)) / max((hi - lo).max(), 1e-12)
        fmain = mean_by_index(feats.detach().double().cpu().numpy(), inv, n)[main]
        return TriMesh(vm, fm), fmain, inv, rep, main, sub

    m1, d1, _, rep_t, main_t, _ = prep(trgt_verts, trgt_faces, trgt_feats)
    m2, d2, inv_s, _, _, sub_s = prep(src_verts, src_faces, src_feats)
    model = FunctionalMapping(m1, m2)
    model.preprocess(descr_type=None, k_process=k + zoomout * zoomout_step)
    model.add_descriptors(d1, d2, normalize=True)
    model.fit(K=(k, k), w_descr=w_descr, w_lap=w_lap, w_dcomm=w_dcomm, w_orient=w_orient)
    if zoomout > 0:
        model.zoomout_refine(nit=zoomout, step=zoomout_step)
    p2p = np.asarray(model.get_p2p(n_jobs=1))          # (n_src_main,) index into the target's main piece
    trgt_orig = rep_t[main_t[p2p]]                      # -> an original target vertex

    pred = torch.cdist(src_feats.float(), trgt_feats.float()).argmin(dim=1).cpu().numpy()  # fallback
    on_main = sub_s[inv_s] >= 0
    pred[on_main] = trgt_orig[sub_s[inv_s[on_main]]]
    return torch.as_tensor(pred, dtype=torch.long, device=src_feats.device)


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

    FEAT_DISTS = ("euclidean", "view_mean", "mean_of_min", "min_of_min")

    MATCHINGS = ("nn", "uot", "fgw", "fmap")

    def __init__(self, feat_dist: str = "euclidean", feat_view_dim: Optional[int] = None,
                 feat_views: Optional[int] = None, matching: str = "nn", fgw: Optional[dict] = None,
                 fmap: Optional[dict] = None, **kwargs):
        """``feat_dist`` -- how a source vertex is compared with target vertices:

        euclidean    one feature per vertex, Euclidean NN (the default).
        The others read per-view features, (V, N * feat_view_dim) with NaN for
        views that do not see the vertex (``..._f<model>_mv`` / ``_mvvis``):
        view_mean    average the visible views, then Euclidean (reproduces the
                     view-averaged feature; a sanity check).
        mean_of_min  d(a, b) = mean over a's views i of min over b's views j of
                     |f_a^i - f_b^j|.
        min_of_min   d(a, b) = min over i, j of |f_a^i - f_b^j|.

        ``feat_views`` (N): only the first N * feat_view_dim channels are
        per-view; the rest is one feature per vertex (a concatenated part,
        e.g. PartField), and d = sqrt(D_views^2 + |rest_a - rest_b|^2).
        Unset: every channel is per-view.

        ``matching`` -- how a query vertex gets its target vertex:

        nn           feature nearest neighbour (the default).
        uot          one unbalanced Sinkhorn on the feature reward, argmax of
                     the plan (``_fgw_match`` with no structure rounds).
        fgw          Shape-of-You's fused Gromov-Wasserstein matching
                     (``_fgw_match``): Sinkhorn plans re-solved with a
                     structural term from anchors' 3-D distances.
        fmap         functional map with the features as descriptors
                     (``_fmap_match``, the old DenseMatcher recipe; needs the
                     meshes' faces, i.e. the ``mesh`` modality).

        ``uot`` / ``fgw`` solve one plan per pair over all vertices, so they need
        ``feat_dist: euclidean``; ``fgw`` takes ``_fgw_match``'s keyword
        arguments (alpha, n_anchors, rounds, iters, rho, anchor_q, temp,
        normalize_scale), ``fmap`` ``_fmap_match``'s (k, w_descr, w_lap,
        w_dcomm, w_orient, zoomout, zoomout_step).
        """
        if feat_dist not in self.FEAT_DISTS:
            raise ValueError(f"feat_dist {feat_dist!r} not in {self.FEAT_DISTS}")
        if feat_dist != "euclidean" and not feat_view_dim:
            raise ValueError(f"feat_dist {feat_dist!r} needs feat_view_dim (per-view channels)")
        self.feat_dist = feat_dist
        self.feat_view_dim = int(feat_view_dim) if feat_view_dim else None
        self.feat_views = int(feat_views) if feat_views else None
        if matching not in self.MATCHINGS:
            raise ValueError(f"matching {matching!r} not in {self.MATCHINGS}")
        if matching != "nn" and feat_dist != "euclidean":
            raise ValueError(f"matching {matching!r} needs feat_dist 'euclidean'")
        self.matching = matching
        self.fgw = dict(fgw or {})
        self.fmap = dict(fmap or {})
        if matching == "uot":
            self.fgw["rounds"] = 0

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

        # mesh resolution, (src + trgt) / 2 per pair: the prediction is a target
        # vertex, so a denser remesh lowers the errors by itself -- logged so a
        # geometry comparison (mc / dmtet / fc) can tell density from fit
        def _n_verts(verts, mask):
            if verts is None:
                return None
            return mask.sum(dim=1).float() if mask is not None else \
                torch.full((verts.shape[0],), float(verts.shape[1]), device=dev)
        n_counts = [n for n in (_n_verts(src_verts, src_verts_mask), _n_verts(trgt_verts, trgt_verts_mask))
                    if n is not None]
        quant_extra = dict(n_verts=torch.stack(n_counts).mean(dim=0))

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
        plans = {}  # b -> (V_src,) target vertex per source vertex (uot / fgw)

        def feat_nn_batch(query_feats_b, b, query_idx_b=None):
            """(Q, F) → (Q,) target vert indices via feature nearest-neighbour,
            or via the pair's transport plan (``matching``), indexed by the
            query vertices ``query_idx_b`` (Q,)."""
            if self.matching != "nn":
                if b not in plans:
                    sm = src_verts_mask[b] if src_verts_mask is not None else \
                        torch.ones(src_feats.shape[1], dtype=torch.bool, device=dev)
                    tm = trgt_verts_mask[b] if trgt_verts_mask is not None else \
                        torch.ones(trgt_feats.shape[1], dtype=torch.bool, device=dev)
                    s_idx, t_idx = sm.nonzero(as_tuple=True)[0], tm.nonzero(as_tuple=True)[0]
                    if self.matching == "fmap":
                        if batch.src_meshes is None or batch.trgt_meshes is None:
                            raise ValueError("matching 'fmap' needs the meshes (object modality 'mesh')")
                        sf, tf = batch.src_meshes[b].faces, batch.trgt_meshes[b].faces
                        # faces index the mesh's own vertices, i.e. the unpadded prefix
                        if not (torch.equal(s_idx.cpu(), torch.arange(len(s_idx)))
                                and torch.equal(t_idx.cpu(), torch.arange(len(t_idx)))
                                and int(sf.max()) < len(s_idx) and int(tf.max()) < len(t_idx)):
                            raise ValueError("matching 'fmap': mesh faces do not index the unpadded vertices")
                        pred = _fmap_match(src_feats[b, s_idx], trgt_feats[b, t_idx],
                                           src_verts[b, s_idx], trgt_verts[b, t_idx], sf, tf, **self.fmap)
                    else:
                        pred = _fgw_match(src_feats[b, s_idx], trgt_feats[b, t_idx],
                                          src_verts[b, s_idx], trgt_verts[b, t_idx], **self.fgw)
                    full = torch.zeros(src_feats.shape[1], dtype=torch.long, device=dev)
                    full[s_idx] = t_idx[pred]
                    plans[b] = full
                return plans[b][query_idx_b]
            if self.feat_dist == "euclidean":
                dists = torch.cdist(query_feats_b.float(), trgt_feats[b].float())  # (Q, V_trgt)
            else:
                dists = _multiview_dists(query_feats_b, trgt_feats[b], self.feat_view_dim, self.feat_dist,
                                         n_views=self.feat_views)
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
                feat_nn_batch(query_feats[b], b, nn_src_vert[b]) for b in range(B_k)
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

            # PCK @ 0.1 / 0.05 * trgt_max_dim
            def _pck(frac):
                return (
                    (kpts_trgt_euc_dist < trgt_max_dim.unsqueeze(1) * frac).float()
                    * kpts_valid.float()
                ).sum(dim=1) / n_valid
            kpts_trgt_pck01  = _pck(0.1)
            kpts_trgt_pck005 = _pck(0.05)

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
                kpts_trgt_pck005             = kpts_trgt_pck005,
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
                pred_trgt_vert = feat_nn_batch(query_feats_b, b, query_idx)  # (Q,)
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
            return ObjectPairQuantBatch(**quant_kpts, **quant_parts, extra=quant_extra), None

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

        quant  = ObjectPairQuantBatch(**quant_kpts, **quant_parts, extra=quant_extra)
        qualit = ObjectPairQualitBatch(imgs=qualit_imgs, part_imgs=part_imgs, feat_imgs=feat_imgs)
        if has_kpts:
            qualit.trgt_src_vert_corr      = pred_trgt_kpt_vert
            qualit.trgt_src_vert_corr_mask = kpts_valid
        return quant, qualit
