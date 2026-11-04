"""AffXAttn: per-point affordance logits from 3-D queries over LitePT's tokens (H12).

A side branch after ShapeXAttn, written for HANDAL's handle segmentation. Every
query is a 3-D point in the normalised cloud frame, embedded and cross-attended
over LitePT's stage-4 tokens (``tokens`` / ``tokens_coord`` / ``tokens_mask``,
LitePT with ``out_tokens: true``) with PointROPE, as ShapeXAttn's surface
tokens are. A head reads one logit per query and class. Two kinds of query:

- **pixels**, ``aff_query_pxl2d`` (B, Q, 2) on ``frames_gt``: image pixels (x, y)
  of the current (cropped) frame, lifted with the frame's depth and intrinsics
  exactly as Frames2FeatPCL lifts its cloud, then mapped into the cloud frame.
  Each also carries its DINOv2 feature, sampled bilinearly from
  ``frames_pred.featmap``. Logits land in ``aff_logit_pxl`` (B, Q[, C]): the 2-D
  affordance mask, at whatever pixels are asked for (in training a sample of
  the object mask; at test every pixel of it).
- **surface**, the predicted fine surface ``pts3d_shape_fine`` (+ its
  ``feat_shape_fine`` when ``surface_feat_dim``): logits in
  ``aff_logit_shape_fine`` (B, N[, C]), the amodal 3-D affordance a grasp needs.

Queries have no self-attention, so each one's logit depends only on its own
point and the tokens. That makes the answer independent of how many pixels are
asked at once, and lets ``chunk`` bound the memory of a full-mask query.
Positions and the surface inputs are detached: the affordance loss reaches the
encoder through the tokens, not by moving the predicted surface.

``frames_pred.feat`` (the global feature, ``in_dim``) is read and left alone.
"""

import logging

import torch
import torch.nn as nn

from o3b.model.model import OD3D_Model, register_model
from o3b.model.shapemlp.model import _mlp
from o3b.model.xattnpool.model import _Layer

logger = logging.getLogger(__name__)


@register_model("AffXAttn")
class AffXAttn(OD3D_Model):
    def __init__(
        self,
        in_dim: int,
        dim: int = 504,
        img_feat_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 28,
        mlp_ratio: int = 2,
        rope_freq: float = 100.0,
        rope_unit: float = 0.08,
        num_classes: int = 1,
        surface: bool = True,
        surface_feat_dim: int = 0,
        pts3d_opengl: bool = True,
        chunk: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.rope_unit = rope_unit
        self.num_classes = num_classes
        self.surface = surface
        self.surface_feat_dim = surface_feat_dim
        self.pts3d_opengl = pts3d_opengl
        self.chunk = chunk
        self.embed_pt = _mlp([3, dim // 2, dim])
        self.embed_img = _mlp([img_feat_dim, dim])
        self.embed_glob = nn.Linear(in_dim, dim)
        self.embed_surf = _mlp([surface_feat_dim, dim]) if surface_feat_dim else None
        # 0 = pixel query, 1 = surface query
        self.query_type = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.query_type, std=0.02)
        self.layers = nn.ModuleList(
            [_Layer(dim, num_heads, mlp_ratio, rope_freq, dropout, self_attn=False)
             for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = _mlp([dim, dim // 2, num_classes])

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cam_to_cloud(frames_pred, pts):
        from o3b.cv.geometry.transform import inv_tform4x4, transf3d_broadcast
        return transf3d_broadcast(pts3d=pts, transf4x4=inv_tform4x4(
            frames_pred.obj_tform4x4_obj_nocs_0c.detach().float())[:, None])

    def _lift_pixels(self, frames_gt, pxl):
        """(B, Q, 2) pixels -> (B, Q, 3) camera-space points, as Frames2FeatPCL.

        Depth is read at the nearest pixel. A pixel with no depth (0, or outside
        depth_mask) takes the median depth of the valid object pixels, so the
        query still lands near the object instead of at the camera centre.
        """
        from o3b.cv.geometry.transform import depth2pts3d_grid

        depth = frames_gt.depth.float()                                 # (B, 1, H, W)
        B, _, H, W = depth.shape
        valid = depth > 1e-5
        if frames_gt.depth_mask is not None:
            valid = valid & (frames_gt.depth_mask > 0.5)
        if frames_gt.mask is not None:
            obj = valid & (frames_gt.mask > 0.5)
        else:
            obj = valid
        fill = torch.stack([
            depth[b][obj[b]].median() if bool(obj[b].any()) else
            (depth[b][valid[b]].median() if bool(valid[b].any()) else depth.new_tensor(1.0))
            for b in range(B)])
        depth = torch.where(valid, depth, fill[:, None, None, None])
        grid = depth2pts3d_grid(depth=depth, cam_intr4x4=frames_gt.cam_intr4x4.float(),
                                opengl=self.pts3d_opengl)                # (B, 3, H, W)
        x = pxl[..., 0].round().long().clamp(0, W - 1)
        y = pxl[..., 1].round().long().clamp(0, H - 1)
        idx = (y * W + x)                                               # (B, Q)
        flat = grid.flatten(2)                                          # (B, 3, H*W)
        return torch.gather(flat, 2, idx[:, None].expand(B, 3, idx.shape[1])).permute(0, 2, 1)

    @staticmethod
    def _sample_featmap(frames_gt, frames_pred, pxl):
        """(B, Q, C) DINOv2 features at the pixels, bilinear, L2-normalised."""
        import torch.nn.functional as F
        featmap = frames_pred.featmap.float()                           # (B, C, fH, fW)
        H, W = int(frames_gt.size[0]), int(frames_gt.size[1])
        g = torch.stack([(pxl[..., 0] + 0.5) / W * 2 - 1,
                         (pxl[..., 1] + 0.5) / H * 2 - 1], dim=-1)      # (B, Q, 2)
        f = F.grid_sample(featmap, g[:, :, None], mode="bilinear", align_corners=False)
        return F.normalize(f[..., 0].permute(0, 2, 1), dim=-1)

    def _run(self, q, q_pos, tok, tok_pos, tok_mask):
        outs = []
        for s in range(0, q.shape[1], self.chunk):
            h, p = q[:, s:s + self.chunk], q_pos[:, s:s + self.chunk]
            for layer in self.layers:
                h = layer(h, p, tok, tok_pos, tok_mask)
            outs.append(self.head(self.norm_out(h)))
        out = torch.cat(outs, dim=1)
        return out[..., 0] if self.num_classes == 1 else out

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, frames_gt, frames_pred=None):
        tok, tok_coord, tok_mask = frames_pred.tokens, frames_pred.tokens_coord, frames_pred.tokens_mask
        if tok is None:
            raise RuntimeError("AffXAttn: no frames_pred.tokens — run LitePT with out_tokens: true")
        tok, tok_pos = tok.float(), tok_coord.float() / self.rope_unit
        glob = self.embed_glob(frames_pred.feat.float())[:, None]       # (B, 1, dim)

        pxl = getattr(frames_gt, "aff_query_pxl2d", None)
        if pxl is not None and frames_gt.depth is not None:
            pxl = pxl.float()
            xyz = self._cam_to_cloud(frames_pred, self._lift_pixels(frames_gt, pxl)).detach()
            q = (self.embed_pt(xyz) + self.embed_img(self._sample_featmap(frames_gt, frames_pred, pxl))
                 + glob + self.query_type[0])
            frames_pred.aff_logit_pxl = self._run(q, xyz / self.rope_unit, tok, tok_pos, tok_mask)

        fine = frames_pred.pts3d_shape_fine
        if self.surface and fine is not None:
            xyz = fine.detach().float()
            q = self.embed_pt(xyz) + glob + self.query_type[1]
            if self.embed_surf is not None and frames_pred.feat_shape_fine is not None:
                q = q + self.embed_surf(frames_pred.feat_shape_fine.detach().float())
            frames_pred.aff_logit_shape_fine = self._run(q, xyz / self.rope_unit, tok, tok_pos, tok_mask)
        return frames_gt, frames_pred
