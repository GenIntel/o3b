"""ShapeHier: a coarse-to-fine surface decoder over all of LitePT's stages.

The decoder mirrors LitePT's encoder the other way round, like a U-Net's
decoder. It has one level per encoder stage. Level ``l`` has
``init_count * fold**l`` tokens of width ``dims[l]`` and cross-attends to the
encoder stage of the same width: with LitePT's default widths
(36, 72, 144, 252, 504) and ``init_count`` 8, ``fold`` 4 that is

    L0    8 tokens x 504  <->  stage 4
    L1   32 tokens x 252  <->  stage 3
    L2  128 tokens x 144  <->  stage 2
    L3  512 tokens x  72  <->  stage 1
    L4 2048 tokens x  36  <->  stage 0   (output)

so the keys and values need no projection between widths.

* **Start (L0).** An MLP on the global feature (``frames_pred.feat``, the chain's
  MLP output, as ShapeXAttn's coarse MLP reads it) gives the ``init_count`` L0
  points. A token is a learned per-slot query, plus the projected global feature,
  plus its embedded point.
* **Each level** runs ``layers_per_level`` layers. Each layer is self-attention
  among the level's tokens, then cross-attention to its encoder stage, then an
  FFN (XAttnPool's layer), so self- and cross-attention alternate. The PointROPE
  positions are the tokens' points / ``rope_units[l]``. By default
  ``rope_units[l]`` is the matching stage's cell (``grid_size * 2**stage``), which
  is LitePT's own unit at that stage.
* **Upfold to the next level.** For every token, a fold MLP reads the token and
  one of ``fold`` 2-D seeds. It gives each child an offset,
  ``tanh(.) * offset_scale * offset_decay**l`` (bounded and shrinking with depth,
  so children stay near their parent), plus a feature at the next width, added to
  a linear map of the parent's feature. A child's point is its parent's point plus
  its offset.
* **Output (L4).** A head on the last level's tokens gives a final point offset
  (zero at init), the NOCS and the per-point features. Its 2048 points are
  ``pts3d_shape_fine``. L3's points, with features from a per-level head, are
  ``pts3d_shape_coarse``, so TrinityMethod's metrics and tables apply unchanged.

Every output point is the L0 point plus the offsets along its branch, so the
output Chamfer loss reaches the offsets of every level without intermediate
losses. ``level_aux: true`` adds them anyway: L0-L2's points (and per-level
features / NOCS) are handed to TrinityMethod as ``pts3d_shape_aux`` /
``feat_shape_aux`` / ``nocs_shape_aux``, whose ``shape_*_l1_aux`` weights
(averaged over the list) then apply. Whether L3 (``*_coarse``) counts is the
config's loss weight.

``knn: k`` makes the cross-attention of the levels in ``knn_levels`` local: each
query attends to its ``k`` nearest tokens of the stage, by distance from its
(detached) point, instead of to all of them. ``rope_pos_grad: true`` keeps the
gradient on the query positions, as in ShapeXAttn.

The module is a side branch: it reads ``feat`` and LitePT's
``tokens_stages`` (``out_tokens_stages: true``) and leaves ``feat`` alone, so it
goes before the PoseMLP. The pose comes from the chain's own XAttnPool, MLP and
PoseMLP.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from o3b.model.model import OD3D_Model, register_model
from o3b.model.shapemlp.model import _mlp
from o3b.model.xattnpool.model import _Layer, rope3d

logger = logging.getLogger(__name__)


class _LocalLayer(nn.Module):
    """XAttnPool's layer (self-attention, cross-attention, FFN) with the
    cross-attention restricted to each query's ``k`` nearest keys."""

    def __init__(self, dim, num_heads, mlp_ratio, rope_freq, dropout, k):
        super().__init__()
        self.base = _Layer(dim, num_heads, mlp_ratio, rope_freq, dropout, self_attn=True)
        self.k = k

    def forward(self, q, q_pos, q_pts, tok, tok_pos, tok_pts, tok_mask):
        b = self.base
        h = b.norm_sa(q)
        q = q + b.self_attn(h, q_pos, h, q_pos)
        q = q + self._cross(b.cross, b.norm_q(q), q_pos, q_pts, b.norm_kv(tok), tok_pos, tok_pts, tok_mask)
        return q + b.ffn(b.norm_ffn(q))

    def _cross(self, att, x, x_pos, x_pts, ctx, ctx_pos, ctx_pts, ctx_mask):
        B, N, C = x.shape
        S, H = ctx.shape[1], att.h
        k = min(self.k, S)
        with torch.no_grad():
            d = torch.cdist(x_pts.detach().float(), ctx_pts.detach().float())        # (B, N, S)
            if ctx_mask is not None:
                d = d.masked_fill(~ctx_mask[:, None, :], float("inf"))
            dist, idx = d.topk(k, dim=-1, largest=False)                             # (B, N, k)
            valid = torch.isfinite(dist)
        q = rope3d(att.q(x).reshape(B, N, H, -1).transpose(1, 2), x_pos, att.rope_freq)   # (B, H, N, d)
        kk, v = att.kv(ctx).reshape(B, S, 2, H, -1).permute(2, 0, 3, 1, 4)              # (B, H, S, d)
        kk = rope3d(kk, ctx_pos, att.rope_freq)
        gi = idx[:, None, :, :, None].expand(B, H, N, k, kk.shape[-1]).reshape(B, H, N * k, -1)
        kk = kk.gather(2, gi).reshape(B, H, N, k, -1)
        v = v.gather(2, gi).reshape(B, H, N, k, -1)
        w = torch.einsum("bhnd,bhnkd->bhnk", q, kk) / kk.shape[-1] ** 0.5
        w = w.masked_fill(~valid[:, None], float("-inf")).softmax(-1)
        w = torch.nan_to_num(w)                                   # a query with no valid key
        if att.dropout and self.training:
            w = F.dropout(w, att.dropout)
        out = torch.einsum("bhnk,bhnkd->bhnd", w, v)
        return att.proj(out.transpose(1, 2).reshape(B, N, C))


@register_model("ShapeHier")
class ShapeHier(OD3D_Model):
    def __init__(
        self,
        in_dim: int,
        dims=(504, 252, 144, 72, 36),
        num_heads=(28, 14, 8, 4, 2),
        init_count: int = 8,
        fold: int = 4,
        layers_per_level: int = 2,
        mlp_ratio: int = 2,
        rope_freq: float = 100.0,
        rope_units=None,
        grid_size: float = 0.005,
        init_hidden_dim: int = 1024,
        fold_hidden_dim: int = 256,
        fold_grid_scale: float = 1.0,
        offset_scale: float = 0.5,
        offset_decay: float = 0.5,
        nocs: bool = False,
        point_feat_dim: int = 0,
        level_aux: bool = False,
        knn: int = 0,
        knn_levels=(3, 4),
        rope_pos_grad: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims, num_heads = list(dims), list(num_heads)
        L = len(dims)
        if len(num_heads) != L:
            raise ValueError("ShapeHier: dims and num_heads must have one entry per level")
        if L < 2:
            raise ValueError("ShapeHier: needs at least two levels")
        side = int(round(fold ** 0.5))
        if side * side != fold:
            raise ValueError(f"ShapeHier: fold must be a square, got {fold}")
        self.dims, self.L, self.fold, self.init_count = dims, L, fold, init_count
        # level l reads encoder stage L-1-l; LitePT's cell there is grid_size * 2**stage
        self.rope_units = (list(rope_units) if rope_units is not None
                           else [grid_size * 2 ** (L - 1 - l) for l in range(L)])
        if len(self.rope_units) != L:
            raise ValueError("ShapeHier: rope_units must have one entry per level")
        self.offset_scale, self.offset_decay = offset_scale, offset_decay
        self.nocs, self.point_feat_dim = nocs, point_feat_dim
        self.extra_ch = (3 if nocs else 0) + point_feat_dim
        self.level_aux = level_aux
        self.knn, self.knn_levels = knn, set(knn_levels) if knn else set()
        self.rope_pos_grad = rope_pos_grad
        self._logged_stages = False

        g = torch.Generator().manual_seed(0)
        self.init = _mlp([in_dim, init_hidden_dim, init_hidden_dim, init_count * 3])
        self.slot = nn.Parameter(torch.randn(init_count, dims[0], generator=g) * 0.02)
        self.embed_glob = nn.Linear(in_dim, dims[0])
        self.embed_pt = _mlp([3, dims[0] // 2, dims[0]])

        self.levels = nn.ModuleList()
        for l in range(L):
            if l in self.knn_levels:
                layers = [_LocalLayer(dims[l], num_heads[l], mlp_ratio, rope_freq, dropout, knn)
                          for _ in range(layers_per_level)]
            else:
                layers = [_Layer(dims[l], num_heads[l], mlp_ratio, rope_freq, dropout, self_attn=True)
                          for _ in range(layers_per_level)]
            self.levels.append(nn.ModuleList(layers))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])

        lin = torch.linspace(-fold_grid_scale, fold_grid_scale, side)
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(-1, 2)
        self.register_buffer("fold_grid", grid, persistent=False)
        # upfold l -> l+1: offset (3) + child feature (dims[l+1]) from token + seed
        self.folds = nn.ModuleList([_mlp([dims[l] + 2, fold_hidden_dim, fold_hidden_dim, 3 + dims[l + 1]])
                                    for l in range(L - 1)])
        self.skips = nn.ModuleList([nn.Linear(dims[l], dims[l + 1]) for l in range(L - 1)])
        self.embed_off = nn.ModuleList([nn.Linear(3, dims[l + 1]) for l in range(L - 1)])
        # per-level NOCS / feature heads for L0 .. L(L-2) (coarse = L(L-2); the rest aux)
        self.level_heads = nn.ModuleList([nn.Linear(dims[l], self.extra_ch) for l in range(L - 1)]
                                         if self.extra_ch else [])
        # output head: final point offset (zero at init) + NOCS / features
        self.out_head = nn.Linear(dims[-1], 3 + self.extra_ch)
        with torch.no_grad():
            self.out_head.weight[:3].zero_()
            self.out_head.bias[:3].zero_()

    def _rp(self, x):
        return x if self.rope_pos_grad else x.detach()

    def forward(self, frames_gt, frames_pred=None):
        feat = frames_pred.feat                                   # (B, in_dim) global
        stages = frames_pred.tokens_stages
        if stages is None:
            raise RuntimeError("ShapeHier: no frames_pred.tokens_stages — run LitePT with out_tokens_stages: true")
        if len(stages) != self.L:
            raise RuntimeError(f"ShapeHier: {len(stages)} encoder stages for {self.L} levels")
        B = feat.shape[0]
        if not self._logged_stages:
            # once per process: real tokens per sample in each LitePT stage (mean / max)
            n = [mk.sum(1).float() for _, _, mk in stages]
            logger.info("ShapeHier: LitePT tokens per sample, stage 0..%d: %s", len(n) - 1,
                        ", ".join(f"{x.mean():.0f} / {x.max():.0f}" for x in n))
            self._logged_stages = True

        pts = self.init(feat).reshape(B, self.init_count, 3)      # L0 points, cloud frame
        q = self.slot[None] + self.embed_glob(feat)[:, None] + self.embed_pt(pts)
        level_pts, level_h = [], []
        for l in range(self.L):
            tok, tok_coord, tok_mask = stages[self.L - 1 - l]
            tok, tok_coord = tok.float(), tok_coord.float()
            u = self.rope_units[l]
            q_pos, tok_pos = self._rp(pts) / u, tok_coord / u
            for layer in self.levels[l]:
                if isinstance(layer, _LocalLayer):
                    q = layer(q, q_pos, pts, tok, tok_pos, tok_coord, tok_mask)
                else:
                    q = layer(q, q_pos, tok, tok_pos, tok_mask)
            h = self.norms[l](q)
            level_pts.append(pts)
            level_h.append(h)
            if l == self.L - 1:
                break
            n = h.shape[1]
            seed = self.fold_grid[None, None].expand(B, n, self.fold, 2)
            out = self.folds[l](torch.cat([h[:, :, None].expand(B, n, self.fold, h.shape[-1]), seed], dim=-1))
            off = torch.tanh(out[..., :3]) * (self.offset_scale * self.offset_decay ** l)
            pts = (pts[:, :, None] + off).reshape(B, n * self.fold, 3)
            q = (self.skips[l](h)[:, :, None] + out[..., 3:] + self.embed_off[l](off)).reshape(B, n * self.fold, -1)

        o = self.out_head(level_h[-1])
        fine = level_pts[-1] + o[..., :3]
        extra_fine = o[..., 3:]
        extra_lvl = [self.level_heads[l](level_h[l]) for l in range(self.L - 1)] if self.extra_ch else None

        k0 = 3 if self.nocs else 0
        coarse_l = self.L - 2
        if self.nocs:
            frames_pred.nocs_shape_fine = extra_fine[..., :3]
            frames_pred.nocs_shape_coarse = extra_lvl[coarse_l][..., :3]
        if self.point_feat_dim:
            frames_pred.feat_shape_fine = extra_fine[..., k0:]
            frames_pred.feat_shape_coarse = extra_lvl[coarse_l][..., k0:]
        if self.level_aux and coarse_l > 0:
            frames_pred.pts3d_shape_aux = level_pts[:coarse_l]
            if self.nocs:
                frames_pred.nocs_shape_aux = [e[..., :3] for e in extra_lvl[:coarse_l]]
            if self.point_feat_dim:
                frames_pred.feat_shape_aux = [e[..., k0:] for e in extra_lvl[:coarse_l]]

        frames_pred.pts3d_shape_coarse = level_pts[coarse_l]
        frames_pred.pts3d_shape_fine = fine
        frames_pred.cam_tform4x4_pts3d_shape = frames_pred.obj_tform4x4_obj_nocs_0c
        return frames_gt, frames_pred
