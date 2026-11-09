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

``pose_levels: true`` (H08-T03) moves the pose into the decoder, refined level by
level. One pose token of width ``pose_dim`` (a learned parameter at L0) runs
through ``pose_layers`` cross-attention layers per level (XAttnPool's layer, no
self-attention) over that level's decoder tokens after the level's layers,
projected to ``pose_dim``. The keys sit at the level's points, the query at the
previous level's predicted centre (the cloud origin at L0), both in the level's
RoPE unit. A head on the token gives PoseMLP's 12 raw outputs (translation, 6D
rotation, size before the softplus): L0 predicts them, every later level adds a
residual (zero-initialised, so all levels start at L0's pose). From L1 on the token
also gets the embedded previous raw pose (detached; the residual sum is not). The
surface never reads the pose token. The last level's pose is written as
``cam_tform4x4_obj`` / ``obj_size3d`` (no PoseMLP after this module); the earlier
ones are ``pose_aux``, which TrinityMethod supervises only if ``train.loss`` weighs
the ``pose_*_aux`` terms.

``init_queries: true`` (H08-T03) starts L0 from learned queries instead: one learned
query per slot, plus the projected global feature, runs through ``init_layers``
layers (self-attention, cross-attention to the coarsest stage, FFN; the queries
carry no position yet, the keys sit at the stage's points), and a linear head on each
query gives that slot's point. The refined query plus its embedded point is the L0
token. Each point is then read from the observed cloud by its own query, rather than
all of them from one pooled vector.

``fold`` may be a list with one entry per upfold (H08-T03), e.g. ``init_count`` 128
and ``fold: [4, 1, 1, 4]`` for 128 / 512 / 512 / 512 / 2048 tokens. A fold of 1 gives
each token one child at a bounded offset (seed 0): the level refines the points
without adding any.

``pose_pool: true`` (H08-T03) reads the pose once, after the decoder, from all of its
levels at the same time instead of refining it level by level. A pose token (learned,
plus the projected global feature, so it starts from what XAttnPool -> MLP pooled)
runs through ``pose_layers`` cross-attention layers over the concatenated tokens of
every level, each level's projected to ``pose_dim`` plus a learned level embedding,
with RoPE at its points in its own level's unit (the query sits at the origin). A
head gives PoseMLP's 12 raw outputs; ``cam_tform4x4_obj`` / ``obj_size3d`` come from it
and no PoseMLP follows. ``pose_pool_detach: true`` stops the pose gradient at the
decoder tokens, so the pose objective cannot change the surface.

``pose_umeyama: true`` (H08-T03, after DROID-SLAM's predict -> measure -> correct)
reads the pose out of the NOCS instead of regressing it, at every level in
``umeyama_levels``. A level's tokens give a NOCS each (the level's NOCS head; at the
last level the output head and the output points) and a confidence (a zero-initialised
linear head, softmax over the level's tokens, so uniform at the start). A weighted
Umeyama fit of the NOCS onto the level's points (detached: the pose never moves the
surface) gives ``s R``, ``t`` in the cloud frame (``o3b.cv.geometry.umeyama``, a
gradient that stays finite at equal singular values). The size is ``2 s`` times the
NOCS's trimmed extent per axis (``size_quantile``) over its longest one, so it agrees
with the NCDS pose (longest side 2). The fit's residual per token, ``point - (s R nocs
+ t)``, with the NOCS and the confidence (all detached), is embedded (zero-initialised)
into the token's children: the next level sees where the current pose and its NOCS
disagree. The pose loss reaches the NOCS and the confidences through the fit; the
NOCS also have their own objective (``shape_nocs_l1_*``). Needs ``nocs: true``; the
last fitted level is ``cam_tform4x4_obj`` / ``obj_size3d``, the earlier ones
``pose_aux``. No PoseMLP after this module.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from o3b.model.model import OD3D_Model, register_model
from o3b.model.shapemlp.model import _mlp
from o3b.model.xattnpool.model import RopePos, _Layer, rope3d

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
        fold=4,
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
        pose_levels: bool = False,
        pose_dim: int = 252,
        pose_heads: int = 14,
        pose_layers: int = 3,
        pose_mlp_ratio: int = 4,
        pose_umeyama: bool = False,
        umeyama_levels=(1, 2, 3, 4),
        size_quantile: float = 0.02,
        pose_pool: bool = False,
        pose_pool_detach: bool = False,
        init_queries: bool = False,
        init_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims, num_heads = list(dims), list(num_heads)
        L = len(dims)
        if len(num_heads) != L:
            raise ValueError("ShapeHier: dims and num_heads must have one entry per level")
        if L < 2:
            raise ValueError("ShapeHier: needs at least two levels")
        folds = [int(fold)] * (L - 1) if isinstance(fold, int) else [int(f) for f in fold]
        if len(folds) != L - 1:
            raise ValueError(f"ShapeHier: fold needs one entry per upfold ({L - 1}), got {folds}")
        for f in folds:
            if f < 1 or int(round(f ** 0.5)) ** 2 != f:
                raise ValueError(f"ShapeHier: every fold must be a square >= 1, got {folds}")
        self.dims, self.L, self.fold_l, self.init_count = dims, L, folds, init_count
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
        self.init_queries = init_queries
        if init_queries:
            if init_layers < 1:
                raise ValueError("ShapeHier: init_queries needs init_layers >= 1")
            self.init_blocks = nn.ModuleList([_Layer(dims[0], num_heads[0], mlp_ratio, rope_freq, dropout, self_attn=True)
                                              for _ in range(init_layers)])
            self.init_norm = nn.LayerNorm(dims[0])
            self.init_pt = nn.Linear(dims[0], 3)
        else:
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

        for f in sorted(set(folds)):
            side = int(round(f ** 0.5))
            lin = (torch.linspace(-fold_grid_scale, fold_grid_scale, side) if side > 1
                   else torch.zeros(1))                               # fold 1: one child, seed 0
            grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(-1, 2)
            self.register_buffer(f"fold_grid_{f}", grid, persistent=False)
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

        self.pose_levels = pose_levels
        if pose_levels:
            if pose_dim % pose_heads or (pose_dim // pose_heads) % 6:
                raise ValueError("ShapeHier: pose_dim / pose_heads must be a multiple of 6 (PointROPE)")
            self.pose_token = nn.Parameter(torch.randn(1, pose_dim, generator=g) * 0.02)
            self.pose_kv = nn.ModuleList([nn.Linear(d, pose_dim) for d in dims])
            self.pose_blocks = nn.ModuleList([
                nn.ModuleList([_Layer(pose_dim, pose_heads, pose_mlp_ratio, rope_freq, dropout, self_attn=False)
                               for _ in range(pose_layers)]) for _ in range(L)])
            self.pose_embed = nn.ModuleList([_mlp([12, pose_dim, pose_dim]) for _ in range(L - 1)])
            self.pose_norm = nn.LayerNorm(pose_dim)
            self.pose_out = nn.ModuleList([nn.Linear(pose_dim, 12) for _ in range(L)])
            with torch.no_grad():
                for head in self.pose_out[1:]:              # residual updates start at zero
                    head.weight.zero_()
                    head.bias.zero_()

        self.pose_pool, self.pose_pool_detach = pose_pool, pose_pool_detach
        if pose_pool:
            if pose_levels or pose_umeyama:
                raise ValueError("ShapeHier: pose_pool excludes pose_levels and pose_umeyama")
            if pose_dim % pose_heads or (pose_dim // pose_heads) % 6:
                raise ValueError("ShapeHier: pose_dim / pose_heads must be a multiple of 6 (PointROPE)")
            self.pp_token = nn.Parameter(torch.randn(1, pose_dim, generator=g) * 0.02)
            self.pp_glob = nn.Linear(in_dim, pose_dim)
            self.pp_kv = nn.ModuleList([nn.Linear(d, pose_dim) for d in dims])
            self.pp_level = nn.Parameter(torch.randn(L, pose_dim, generator=g) * 0.02)
            self.pp_blocks = nn.ModuleList([_Layer(pose_dim, pose_heads, pose_mlp_ratio, rope_freq, dropout,
                                                   self_attn=False) for _ in range(pose_layers)])
            self.pp_norm = nn.LayerNorm(pose_dim)
            self.pp_out = nn.Linear(pose_dim, 12)
        elif pose_pool_detach:
            raise ValueError("ShapeHier: pose_pool_detach needs pose_pool: true")

        self.pose_umeyama = pose_umeyama
        if pose_umeyama:
            if not nocs:
                raise ValueError("ShapeHier: pose_umeyama needs nocs: true")
            if pose_levels:
                raise ValueError("ShapeHier: pose_umeyama and pose_levels exclude each other")
            self.um_levels = sorted(int(l) for l in umeyama_levels)
            if not self.um_levels or not all(0 <= l < L for l in self.um_levels):
                raise ValueError(f"ShapeHier: umeyama_levels must lie in 0..{L - 1}")
            self.size_quantile = size_quantile
            self.um_conf = nn.ModuleDict({str(l): nn.Linear(dims[l], 1) for l in self.um_levels})
            # residual (3) + NOCS (3) + confidence x n (1) of a parent -> its children
            self.um_embed = nn.ModuleDict({str(l): _mlp([7, dims[l + 1], dims[l + 1]])
                                           for l in self.um_levels if l < L - 1})
            with torch.no_grad():
                for m in self.um_conf.values():
                    m.weight.zero_()
                    m.bias.zero_()
                for m in self.um_embed.values():
                    m[-1].weight.zero_()
                    m[-1].bias.zero_()

    def _umeyama(self, l, h, pts):
        """Weighted Umeyama fit of level ``l``'s NOCS onto its points (cloud frame).

        Returns the pose (``pose_from_parts``'s cloud-frame inputs) and the feedback
        features (B, n, 7) for the children."""
        from o3b.cv.geometry.umeyama import weighted_umeyama
        if l == self.L - 1:
            o = self.out_head(h)
            pts, nocs = pts + o[..., :3], o[..., 3:6]
        else:
            nocs = self.level_heads[l](h)[..., :3]
        w = torch.softmax(self.um_conf[str(l)](h)[..., 0].float(), dim=1)
        nocs = nocs.float()
        s, R, t = weighted_umeyama(nocs, pts.detach().float(), w)
        q = torch.tensor([self.size_quantile, 1 - self.size_quantile], device=nocs.device)
        lo, hi = torch.quantile(nocs, q, dim=1)                       # (B, 3) each
        ext = (hi - lo).clamp_min(1e-6)
        size = 2 * s.abs()[:, None] * ext / ext.max(dim=1, keepdim=True).values
        fit = s[:, None, None] * (nocs @ R.transpose(1, 2)) + t[:, None]
        fb = torch.cat([pts.float() - fit, nocs, w[..., None] * w.shape[1]], dim=-1).detach()
        return (t, R, size), fb

    def _rp(self, x):
        return x if self.rope_pos_grad else x.detach()

    def _pose(self, frames_pred, level_pts, level_h, B):
        """The pose token through every level, refining PoseMLP's raw pose."""
        from o3b.model.posemlp.model import pose_from_raw
        obj_tform = frames_pred.obj_tform4x4_obj_nocs_0c
        t = self.pose_token[None].expand(B, 1, -1)
        raw, poses = None, []
        for l in range(self.L):
            u = self.rope_units[l]
            if raw is None:
                q_pos = level_pts[l].new_zeros(B, 1, 3)
            else:
                t = t + self.pose_embed[l - 1](raw.detach())[:, None]
                q_pos = self._rp(raw[:, None, :3]) / u
            kv = self.pose_kv[l](level_h[l])
            kv_pos, q_pos = RopePos(self._rp(level_pts[l]) / u), RopePos(q_pos)
            for layer in self.pose_blocks[l]:
                t = layer(t, q_pos, kv, kv_pos, None)
            d = self.pose_out[l](self.pose_norm(t)[:, 0])
            raw = d if raw is None else raw + d
            poses.append(pose_from_raw(raw, obj_tform))
        frames_pred.cam_tform4x4_obj, frames_pred.obj_size3d = poses[-1]
        frames_pred.pose_aux = poses[:-1]

    def _pose_pool(self, frames_pred, feat, level_pts, level_h, B):
        """One pose token over the tokens of all levels at once."""
        from o3b.model.posemlp.model import pose_from_raw
        kv, kv_pos = [], []
        for l in range(self.L):
            h = level_h[l].detach() if self.pose_pool_detach else level_h[l]
            kv.append(self.pp_kv[l](h) + self.pp_level[l])
            kv_pos.append(self._rp(level_pts[l]) / self.rope_units[l])
        kv, kv_pos = torch.cat(kv, dim=1), torch.cat(kv_pos, dim=1)
        t = self.pp_token[None].expand(B, 1, -1) + self.pp_glob(feat)[:, None]
        q_pos, kv_pos = RopePos(kv_pos.new_zeros(B, 1, 3)), RopePos(kv_pos)
        for layer in self.pp_blocks:
            t = layer(t, q_pos, kv, kv_pos, None)
        raw = self.pp_out(self.pp_norm(t)[:, 0])
        frames_pred.cam_tform4x4_obj, frames_pred.obj_size3d = pose_from_raw(raw, frames_pred.obj_tform4x4_obj_nocs_0c)

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
            logger.warning("ShapeHier: LitePT tokens per sample, stage 0..%d: %s", len(n) - 1,
                        ", ".join(f"{x.mean():.0f} / {x.max():.0f}" for x in n))
            self._logged_stages = True

        if self.init_queries:
            tok, tok_coord, tok_mask = stages[-1]                 # coarsest stage
            u = self.rope_units[0]
            q = self.slot[None] + self.embed_glob(feat)[:, None]
            q_pos = RopePos(q.new_zeros(B, self.init_count, 3))
            tok_pos = RopePos(tok_coord.float() / u)
            for layer in self.init_blocks:
                q = layer(q, q_pos, tok.float(), tok_pos, tok_mask)
            pts = self.init_pt(self.init_norm(q))                 # L0 points, one per query
            q = q + self.embed_pt(pts)
        else:
            pts = self.init(feat).reshape(B, self.init_count, 3)  # L0 points, cloud frame
            q = self.slot[None] + self.embed_glob(feat)[:, None] + self.embed_pt(pts)
        level_pts, level_h = [], []
        um_poses = []
        for l in range(self.L):
            tok, tok_coord, tok_mask = stages[self.L - 1 - l]
            tok, tok_coord = tok.float(), tok_coord.float()
            u = self.rope_units[l]
            q_pos, tok_pos = self._rp(pts) / u, tok_coord / u
            if l not in self.knn_levels:                          # one cos / sin table per level
                q_pos, tok_pos = RopePos(q_pos), RopePos(tok_pos)
            for layer in self.levels[l]:
                if isinstance(layer, _LocalLayer):
                    q = layer(q, q_pos, pts, tok, tok_pos, tok_coord, tok_mask)
                else:
                    q = layer(q, q_pos, tok, tok_pos, tok_mask)
            h = self.norms[l](q)
            level_pts.append(pts)
            level_h.append(h)
            fb = None
            if self.pose_umeyama and l in self.um_levels:
                pose_l, fb = self._umeyama(l, h, pts)
                um_poses.append(pose_l)
            if l == self.L - 1:
                break
            n, f = h.shape[1], self.fold_l[l]
            seed = getattr(self, f"fold_grid_{f}")[None, None].expand(B, n, f, 2)
            out = self.folds[l](torch.cat([h[:, :, None].expand(B, n, f, h.shape[-1]), seed], dim=-1))
            off = torch.tanh(out[..., :3]) * (self.offset_scale * self.offset_decay ** l)
            pts = (pts[:, :, None] + off).reshape(B, n * f, 3)
            q = self.skips[l](h)[:, :, None] + out[..., 3:] + self.embed_off[l](off)
            if fb is not None and str(l) in self.um_embed:
                q = q + self.um_embed[str(l)](fb.to(q.dtype))[:, :, None]
            q = q.reshape(B, n * f, -1)

        if self.pose_pool:
            self._pose_pool(frames_pred, feat, level_pts, level_h, B)
        if self.pose_levels:
            self._pose(frames_pred, level_pts, level_h, B)
        if self.pose_umeyama:
            from o3b.model.posemlp.model import pose_from_parts
            obj_tform = frames_pred.obj_tform4x4_obj_nocs_0c
            poses = [pose_from_parts(t, R, size, obj_tform) for t, R, size in um_poses]
            frames_pred.cam_tform4x4_obj, frames_pred.obj_size3d = poses[-1]
            frames_pred.pose_aux = poses[:-1]

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
