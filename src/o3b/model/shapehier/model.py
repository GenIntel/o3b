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
  FFN (XAttnPool's layer), so self- and cross-attention alternate. A list gives one
  count per level; 0 (H08-T03, e.g. ``[4, 4, 4, 4, 0]``) leaves that level without
  attention: its tokens are the fold's output (skip + fold feature + embedded offset)
  and go straight to the norm and the heads, so the level still has its points and
  features but never reads its encoder stage or its neighbours. The PointROPE
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

``num_registers: r`` (H08-T03) gives every level ``r`` learned register tokens of the
level's width (Darcet et al., "Vision Transformers Need Registers"; DINOv2-reg uses 4).
They join the level's tokens in every layer (self-attention among all of them,
cross-attention to the stage, FFN) at the RoPE origin, are new at every level (not
carried through an upfold) and are dropped after the level's layers: no head, upfold
or pose pooling reads them. Not with ``knn`` levels. The ``init_queries`` stage has none.

``mask_tokens: true`` (H07 on ShapeHier, needs ``point_feat_dim``) adds ShapeXAttn's
training-only masked tokens: the method puts GT surface points on ``frames_gt``
(``mask_pts3d_cam``, ``mask_feat``); the first half know their XYZ (embedded, PointROPE
at the point in each level's unit) and predict the features, the second half know the
features (embedded, at the RoPE origin) and predict the XYZ (normalised cloud frame).
They enter at level ``mask_start_level`` (0: all levels, ``L - 1``: the last only) with
their own embeddings of that level's width plus the projected global feature, run
through the same layers as the surface tokens, change width at each upfold through
their own linear map (no fold: K tokens at every level) and are read at the last
level by ``mask_head`` (``mask_pred_ppf`` / ``mask_pred_xyz`` with ``mask_tgt_*``).
Each layer runs on them as a separate set after it ran on the surface tokens, so the
surface tokens never see them and their computation is exactly that of a pass without
them; the masked tokens never see each other either. With ``mask_see_surface: false``
they see only themselves (identity self-attention mask); with ``true`` their
self-attention keys are that layer's input surface tokens (registers included) plus
themselves, as ShapeXAttn's ``mask_see``, so the masked losses also train the surface
tokens through those keys. At test time ``query_pts3d_cam`` / ``query_feat`` give
``query_pred_feat`` / ``query_pred_pts3d_cam`` the same way, as in ShapeXAttn.

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

``init_pos_layers: k`` (H08-T03, needs ``init_queries``) adds a second, posed stage to that
start: after the ``init_layers`` unposed layers have given each query its point, the query
plus its embedded point runs ``k`` more layers (self-attention, cross-attention to the
coarsest stage, FFN) with PointROPE at that point, in L0's unit, so it probes the stage
around where it sits. A zero-initialised head then moves each point by an offset, and the
embedded offset is added to the token, as at an upfold. Those points and tokens are L0's.

``fold`` may be a list with one entry per upfold (H08-T03), e.g. ``init_count`` 128
and ``fold: [4, 1, 1, 4]`` for 128 / 512 / 512 / 512 / 2048 tokens. A fold of 1 gives
each token one child at a bounded offset (seed 0): the level refines the points
without adding any.

``root_fold: k**3`` (H08-T03) replaces the init MLP by a root stage: one token (a learned
slot plus the projected global feature) at the cloud origin runs ``root_layers`` layers
over the coarsest stage (L0's layer type and unit; its self-attention is over itself),
then a fold MLP reads it with each of ``k**3`` 3-D seeds (a ``k x k x k`` grid spaced
``fold_grid_scale``) and gives every child an offset ``tanh(.) * root_offset_scale``
from the origin plus a feature; the children's tokens are a linear map of the root, that
feature and their embedded point. They are L0 (``init_count`` must equal ``root_fold``),
e.g. ``root_fold: 125`` and ``fold: [4, 1, 1, 4]`` for 1 -> 125 / 500 / 500 / 500 / 2000.

``out_fold: k`` (H08-T03) replaces the output head by a fold MLP, as ShapeXAttn's: each
last-level token, with one of ``k`` 2-D seeds (spaced ``out_fold_grid_scale``), gives a
child's point offset (unbounded) plus its NOCS / features, so the last level's ``n``
tokens become ``n * k`` output points. With ``fold: [4, 1, 1, 1]`` and ``out_fold: 4``
the last attention level has 512 tokens and the output still 2048 points.

``pose_pool: true`` (H08-T03) reads the pose once, after the decoder, from all of its
levels at the same time instead of refining it level by level. A pose token (learned,
plus the projected global feature, so it starts from what XAttnPool -> MLP pooled)
runs through ``pose_layers`` cross-attention layers over the concatenated tokens of
every level, each level's projected to ``pose_dim`` plus a learned level embedding,
with RoPE at its points in its own level's unit (the query sits at the origin). A
head gives PoseMLP's 12 raw outputs; ``cam_tform4x4_obj`` / ``obj_size3d`` come from it
and no PoseMLP follows. ``pose_pool_detach: true`` stops the pose gradient at the
decoder tokens, so the pose objective cannot change the surface.
``pose_pool_head_layers: n`` (H18; 0 = the single ``Linear(pose_dim, 12)``) reads the
pose token with PoseMLP's head instead: three separate networks of ``n`` layers (hidden
``pose_pool_head_hidden``, ReLU, no dropout) for translation, 6D rotation and size. At
``n`` = 1 that is three linear layers, the same function class as the single one.

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
        layers_per_level=2,
        mlp_ratio: int = 2,
        rope_freq: float = 100.0,
        rope_units=None,
        grid_size: float = 0.005,
        init_hidden_dim: int = 1024,
        init_mlp_layers: int = 3,
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
        pose_pool_head_layers: int = 0,
        pose_pool_head_hidden: int = 512,
        init_queries: bool = False,
        init_layers: int = 2,
        init_pos_layers: int = 0,
        out_fold: int = 1,
        out_fold_grid_scale: float = 0.05,
        root_fold: int = 0,
        root_layers: int = 4,
        root_offset_scale: float = 1.0,
        num_registers: int = 0,
        mask_tokens: bool = False,
        mask_see_surface: bool = False,
        mask_start_level: int = 0,
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
            self.init_pos_layers = int(init_pos_layers)
            if self.init_pos_layers < 0:
                raise ValueError("ShapeHier: init_pos_layers must be >= 0")
            if self.init_pos_layers:
                # second init stage: the queries at their points (RoPE), then a point offset
                self.init_pos_blocks = nn.ModuleList([
                    _Layer(dims[0], num_heads[0], mlp_ratio, rope_freq, dropout, self_attn=True)
                    for _ in range(self.init_pos_layers)])
                self.init_pos_norm = nn.LayerNorm(dims[0])
                self.init_pos_off = nn.Linear(dims[0], 3)
                self.init_pos_embed_off = nn.Linear(3, dims[0])
                with torch.no_grad():                       # starts at the unposed stage's points
                    self.init_pos_off.weight.zero_()
                    self.init_pos_off.bias.zero_()
        else:
            if init_pos_layers:
                raise ValueError("ShapeHier: init_pos_layers needs init_queries: true")
            self.init_pos_layers = 0
        self.root_fold = int(root_fold)
        if self.root_fold:
            side = int(round(self.root_fold ** (1 / 3)))
            if self.root_fold < 2 or side ** 3 != self.root_fold:
                raise ValueError(f"ShapeHier: root_fold must be a cube >= 8, got {root_fold}")
            if init_queries:
                raise ValueError("ShapeHier: root_fold and init_queries are two different L0 inits")
            if init_count != self.root_fold:
                raise ValueError(f"ShapeHier: root_fold gives L0 its tokens, so init_count must equal it "
                                 f"({init_count} != {root_fold})")
            if root_layers < 0:
                raise ValueError("ShapeHier: root_layers must be >= 0")
            self.root_blocks = nn.ModuleList([_Layer(dims[0], num_heads[0], mlp_ratio, rope_freq, dropout, self_attn=True)
                                              for _ in range(root_layers)])
            self.root_norm = nn.LayerNorm(dims[0])
            lin = torch.linspace(-fold_grid_scale, fold_grid_scale, side)
            grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1).reshape(-1, 3)
            self.register_buffer("root_grid", grid, persistent=False)
            # root token + 3-D seed -> child offset (3) + child feature (dims[0])
            self.root_fold_mlp = _mlp([dims[0] + 3, fold_hidden_dim, fold_hidden_dim, 3 + dims[0]])
            self.root_skip = nn.Linear(dims[0], dims[0])
            self.root_offset_scale = root_offset_scale
        elif not init_queries:
            # init_mlp_layers linear layers (ReLU between): 3 -> in, hidden, hidden, out; 1 -> one Linear
            if init_mlp_layers < 1:
                raise ValueError(f"ShapeHier: init_mlp_layers must be >= 1, got {init_mlp_layers}")
            self.init = _mlp([in_dim] + [init_hidden_dim] * (init_mlp_layers - 1) + [init_count * 3])
        # the root stage has one token, so one slot
        self.slot = nn.Parameter(torch.randn(1 if self.root_fold else init_count, dims[0], generator=g) * 0.02)
        self.embed_glob = nn.Linear(in_dim, dims[0])
        self.embed_pt = _mlp([3, dims[0] // 2, dims[0]])

        lpl = ([int(layers_per_level)] * L if isinstance(layers_per_level, int)
               else [int(n) for n in layers_per_level])
        if len(lpl) != L or min(lpl) < 0:
            raise ValueError(f"ShapeHier: layers_per_level needs one count >= 0 per level ({L}), got {lpl}")
        self.levels = nn.ModuleList()
        for l in range(L):
            if l in self.knn_levels:
                layers = [_LocalLayer(dims[l], num_heads[l], mlp_ratio, rope_freq, dropout, knn)
                          for _ in range(lpl[l])]
            else:
                layers = [_Layer(dims[l], num_heads[l], mlp_ratio, rope_freq, dropout, self_attn=True)
                          for _ in range(lpl[l])]
            self.levels.append(nn.ModuleList(layers))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])

        self.num_registers = int(num_registers)
        if self.num_registers < 0:
            raise ValueError("ShapeHier: num_registers must be >= 0")
        if self.num_registers and self.knn_levels:
            raise ValueError("ShapeHier: num_registers does not work with knn levels")
        self.registers = (nn.ParameterList([nn.Parameter(torch.randn(self.num_registers, d, generator=g) * 0.02)
                                            for d in dims]) if self.num_registers else None)

        self.mask_tokens = mask_tokens
        self.mask_see_surface = mask_see_surface
        self.mask_start_level = int(mask_start_level)
        if mask_tokens:
            if not point_feat_dim:
                raise ValueError("ShapeHier: mask_tokens needs point_feat_dim")
            if not 0 <= self.mask_start_level < L:
                raise ValueError(f"ShapeHier: mask_start_level must lie in 0..{L - 1}")
            if self.knn_levels & set(range(self.mask_start_level, L)):
                raise ValueError("ShapeHier: mask_tokens does not work with knn levels")
            d0 = dims[self.mask_start_level]
            self.mask_embed_pt = _mlp([3, max(d0 // 2, 16), d0])
            self.mask_embed_feat = _mlp([point_feat_dim, max(d0 // 2, 16), d0])
            self.mask_embed_glob = nn.Linear(in_dim, d0)
            self.mask_type = nn.Parameter(torch.randn(2, d0, generator=g) * 0.02)  # 0: XYZ known, 1: features
            self.mask_skips = nn.ModuleList([nn.Linear(dims[l], dims[l + 1])
                                             for l in range(self.mask_start_level, L - 1)])
            self.mask_head = _mlp([dims[-1], fold_hidden_dim, 3 + point_feat_dim])

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
        self.out_fold = int(out_fold)
        if self.out_fold < 1 or int(round(self.out_fold ** 0.5)) ** 2 != self.out_fold:
            raise ValueError(f"ShapeHier: out_fold must be a square >= 1, got {out_fold}")
        if self.out_fold > 1:
            # ShapeXAttn's fold: token + seed -> offset (3) + NOCS / features per child
            side = int(round(self.out_fold ** 0.5))
            lin = torch.linspace(-out_fold_grid_scale, out_fold_grid_scale, side)
            grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(-1, 2)
            self.register_buffer("out_fold_grid", grid, persistent=False)
            self.out_head = _mlp([dims[-1] + 2, fold_hidden_dim, fold_hidden_dim, 3 + self.extra_ch])
        else:
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
        self.pose_pool_head_layers = int(pose_pool_head_layers) if pose_pool else 0
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
            if pose_pool_head_layers:
                # PoseMLP's head (o3b MLP with out_dims [3, 6, 3]): one network each for
                # translation, 6D rotation and size, concatenated to the 12 raw outputs
                def _head(out):
                    if pose_pool_head_layers == 1:
                        return nn.Sequential(nn.Linear(pose_dim, out))
                    net = [nn.Linear(pose_dim, pose_pool_head_hidden)]
                    for _ in range(pose_pool_head_layers - 2):
                        net += [nn.ReLU(inplace=True), nn.Linear(pose_pool_head_hidden, pose_pool_head_hidden)]
                    return nn.Sequential(*net, nn.ReLU(inplace=True), nn.Linear(pose_pool_head_hidden, out))
                self.pp_heads = nn.ModuleList([_head(o) for o in (3, 6, 3)])
            else:
                self.pp_out = nn.Linear(pose_dim, 12)
        elif pose_pool_detach:
            raise ValueError("ShapeHier: pose_pool_detach needs pose_pool: true")

        self.pose_umeyama = pose_umeyama
        if pose_umeyama:
            if not nocs:
                raise ValueError("ShapeHier: pose_umeyama needs nocs: true")
            if pose_levels:
                raise ValueError("ShapeHier: pose_umeyama and pose_levels exclude each other")
            if self.out_fold > 1 and L - 1 in [int(l) for l in umeyama_levels]:
                raise ValueError("ShapeHier: pose_umeyama at the last level needs out_fold: 1")
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
        t = self.pp_norm(t)[:, 0]
        raw = torch.cat([h(t) for h in self.pp_heads], dim=-1) if self.pose_pool_head_layers else self.pp_out(t)
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
            if self.init_pos_layers:                              # probe the stage at those points
                q_pos = RopePos(self._rp(pts) / u)
                for layer in self.init_pos_blocks:
                    q = layer(q, q_pos, tok.float(), tok_pos, tok_mask)
                off = self.init_pos_off(self.init_pos_norm(q))
                pts = pts + off
                q = q + self.init_pos_embed_off(off)
        elif self.root_fold:
            # one root token at the cloud origin reads the coarsest stage, then folds into L0
            tok, tok_coord, tok_mask = stages[-1]
            u = self.rope_units[0]
            r = self.slot[None] + self.embed_glob(feat)[:, None]   # (B, 1, d0)
            r_pos = RopePos(r.new_zeros(B, 1, 3))
            tok_pos = RopePos(tok_coord.float() / u)
            for layer in self.root_blocks:
                r = layer(r, r_pos, tok.float(), tok_pos, tok_mask)
            h = self.root_norm(r)
            n = self.root_fold
            seed = self.root_grid[None].expand(B, n, 3)
            out = self.root_fold_mlp(torch.cat([h.expand(B, n, h.shape[-1]), seed], dim=-1))
            pts = torch.tanh(out[..., :3]) * self.root_offset_scale    # L0 points = root (origin) + offset
            q = self.root_skip(h) + out[..., 3:] + self.embed_pt(pts)
        else:
            pts = self.init(feat).reshape(B, self.init_count, 3)  # L0 points, cloud frame
            q = self.slot[None] + self.embed_glob(feat)[:, None] + self.embed_pt(pts)
        # H07 masked tokens (training) or test-time query tokens: (tokens, xyz or None, ...)
        mset = self._mask_inputs(frames_gt, frames_pred) if self.mask_tokens else None
        qm = None
        level_pts, level_h = [], []
        um_poses = []
        for l in range(self.L):
            tok, tok_coord, tok_mask = stages[self.L - 1 - l]
            tok, tok_coord = tok.float(), tok_coord.float()
            u = self.rope_units[l]
            n_surf = q.shape[1]
            q_pos = self._rp(pts) / u
            if self.num_registers:                                # at the RoPE origin, new per level
                q = torch.cat([q, self.registers[l][None].expand(B, -1, -1).to(q.dtype)], dim=1)
                q_pos = torch.cat([q_pos, q_pos.new_zeros(B, self.num_registers, 3)], dim=1)
            tok_pos = tok_coord / u
            if l not in self.knn_levels:                          # one cos / sin table per level
                q_pos, tok_pos = RopePos(q_pos), RopePos(tok_pos)
            surf_in = []                                          # each layer's input, for mask_see
            for layer in self.levels[l]:
                if mset is not None and self.mask_see_surface:
                    surf_in.append((q, q_pos))
                if isinstance(layer, _LocalLayer):
                    q = layer(q, q_pos, pts, tok, tok_pos, tok_coord, tok_mask)
                else:
                    q = layer(q, q_pos, tok, tok_pos, tok_mask)
            q = q[:, :n_surf]                                     # drop the registers
            if mset is not None and l >= self.mask_start_level:
                qm = self._mask_level(l, mset, qm, feat, tok, tok_pos, tok_mask, B, surf_in)
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

        h = level_h[-1]
        if self.out_fold > 1:
            n, K = h.shape[1], self.out_fold
            seed = self.out_fold_grid[None, None].expand(B, n, K, 2)
            o = self.out_head(torch.cat([h[:, :, None].expand(B, n, K, h.shape[-1]), seed], dim=-1))
            fine = (level_pts[-1][:, :, None] + o[..., :3]).reshape(B, n * K, 3)
            extra_fine = o[..., 3:].reshape(B, n * K, self.extra_ch)
        else:
            o = self.out_head(h)
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
        if qm is not None:
            self._mask_out(frames_pred, self.norms[-1](qm), mset)
        return frames_gt, frames_pred

    # ── H07 masked / query tokens ────────────────────────────────────────────

    @staticmethod
    def _cam_to_cloud(frames_pred, pts):
        """Camera -> the normalised cloud frame the surface points live in."""
        from o3b.cv.geometry.transform import inv_tform4x4, transf3d_broadcast
        return transf3d_broadcast(pts3d=pts, transf4x4=inv_tform4x4(
            frames_pred.obj_tform4x4_obj_nocs_0c.detach().float())[:, None])

    def _mask_inputs(self, frames_gt, frames_pred):
        """The XYZ-known points (cloud frame) and the feature-known features of this
        pass: the H07 masked tokens in training (with their targets), else the
        test-time queries; None when there are neither."""
        pts = getattr(frames_gt, "mask_pts3d_cam", None)
        if pts is not None:
            pts = self._cam_to_cloud(frames_pred, pts.float())
            ppf = frames_gt.mask_feat.float()
            k1 = pts.shape[1] // 2
            return {"xyz": pts[:, :k1], "ppf": ppf[:, k1:], "tgt": (ppf[:, :k1], pts[:, k1:])}
        qry_pts = getattr(frames_gt, "query_pts3d_cam", None)
        qry_feat = getattr(frames_gt, "query_feat", None)
        if qry_pts is None and qry_feat is None:
            return None
        return {"xyz": None if qry_pts is None else self._cam_to_cloud(frames_pred, qry_pts.float()),
                "ppf": None if qry_feat is None else qry_feat.float(), "tgt": None}

    def _mask_level(self, l, mset, qm, feat, tok, tok_pos, tok_mask, B, surf_in=()):
        """Level ``l``'s layers on the masked tokens: each sees itself and, with
        ``mask_see_surface``, the surface tokens at that layer's input (``surf_in``)."""
        xyz, ppf = mset["xyz"], mset["ppf"]
        if l == self.mask_start_level:
            glob = self.mask_embed_glob(feat)[:, None]
            parts = []
            if xyz is not None:
                parts.append(self.mask_embed_pt(xyz) + self.mask_type[0])
            if ppf is not None:
                parts.append(self.mask_embed_feat(ppf) + self.mask_type[1])
            qm = torch.cat(parts, dim=1) + glob
        else:
            qm = self.mask_skips[l - 1 - self.mask_start_level](self.norms[l - 1](qm))
        u = self.rope_units[l]
        pos = []
        if xyz is not None:
            pos.append(xyz / u)
        if ppf is not None:
            pos.append(ppf.new_zeros(B, ppf.shape[1], 3))
        pos = torch.cat(pos, dim=1)
        K = qm.shape[1]
        eye = torch.eye(K, dtype=torch.bool, device=qm.device)
        if not self.mask_see_surface:
            pos = RopePos(pos)
            for layer in self.levels[l]:
                qm = layer(qm, pos, tok, tok_pos, tok_mask, self_mask=eye)
            return qm
        n = surf_in[0][0].shape[1] if surf_in else 0
        allow = torch.ones(K, n + K, dtype=torch.bool, device=qm.device)   # rows: masked tokens
        allow[:, n:] = eye                                                 # the surface + itself
        ctx_pos = RopePos(torch.cat([surf_in[0][1].pos, pos], dim=1)) if surf_in else None
        pos = RopePos(pos)
        for layer, (q_s, _) in zip(self.levels[l], surf_in):
            h = layer.norm_sa(qm)
            ctx = torch.cat([layer.norm_sa(q_s), h], dim=1)
            qm = qm + layer.self_attn(h, pos, ctx, ctx_pos, pair_mask=allow)
            qm = qm + layer.cross(layer.norm_q(qm), pos, layer.norm_kv(tok), tok_pos, tok_mask)
            qm = qm + layer.ffn(layer.norm_ffn(qm))
        return qm

    def _mask_out(self, frames_pred, h, mset):
        """Read the masked tokens: training targets or test-time query answers."""
        out = self.mask_head(h)
        k1 = 0 if mset["xyz"] is None else mset["xyz"].shape[1]
        if mset["tgt"] is not None:
            frames_pred.mask_pred_ppf, frames_pred.mask_tgt_ppf = out[:, :k1, 3:], mset["tgt"][0]
            frames_pred.mask_pred_xyz, frames_pred.mask_tgt_xyz = out[:, k1:, :3], mset["tgt"][1]
            return
        if mset["xyz"] is not None:
            frames_pred.query_pred_feat = out[:, :k1, 3:]
        if mset["ppf"] is not None:
            from o3b.cv.geometry.transform import transf3d_broadcast
            frames_pred.query_pred_pts3d_cam = transf3d_broadcast(
                pts3d=out[:, k1:, :3],
                transf4x4=frames_pred.obj_tform4x4_obj_nocs_0c.float()[:, None])
