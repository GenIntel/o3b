"""XAttnPool: learned queries cross-attending over LitePT's last-stage tokens.

A drop-in for LitePT's mean pooling: run LitePT with ``out_tokens: true`` and
``out_feat_pool: none``, so it hands on its tokens (``tokens``,
``tokens_coord``, ``tokens_mask``) and leaves ``feat`` as the DINO cls feature;
this stage pools the tokens and writes ``feat = cat([pooled, feat])`` — the
layout LitePT's ``out_feat_append_in_feat`` writes, so the MLP / PoseMLP after
it read the same width as with mean pooling.

Each layer is pre-norm cross-attention (queries over the tokens, padded tokens
masked) + FFN, with self-attention among the queries every
``self_attn_every`` layers (0 = never; pointless with one query, where softmax
over a single key is the identity).

Positions enter as PointROPE (LitePT's rotary encoding, same frequency layout
and base), here in torch so positions can be float: a token sits at its pooled
``coord`` divided by ``rope_unit``, the queries at ``query_pos`` (the cloud
frame's origin — the observed cloud's median — by default). ``rope_unit``
defaults to LitePT's stage-4 cell (grid_size 0.005 x stride 2^4), the unit its
own last-stage RoPE counted in. The head dim (dim / num_heads) must be a
multiple of 6, as in PointROPE: 504 / 28 = 18 like LitePT's stage 4.

**Surface queries** (``surface_queries: 512``, H02-T04 variant c): the pooling
also carries 512 surface queries next to the pose query, through the same
layers (set ``self_attn_every: 1`` so pose and surface queries exchange in every
layer). The pose feature is then the *first* query alone, not the mean. Each
refined surface token is decoded by two MLPs: ``coarse`` maps it to one point
(+ its NOCS), and ``fine`` reads the token *and* that output and emits
``surface_fold`` points (offsets from the coarse point, + their NOCS): 512 x 4 =
2048. Outputs are written like ShapeXAttn's (``pts3d_shape_coarse`` /
``_fine``, ``nocs_shape_*``, ``cam_tform4x4_pts3d_shape``), in the normalised
cloud frame the tokens live in, so TrinityMethod's shape metrics apply.

``surface_anchor`` sets what a surface query is before the first layer:

- ``learned``: a free parameter per query, at the pose query's position.
``surface_pose_attn: false`` masks self-attention between the pose queries and
the surface queries (each group still self-attends within itself), so the pose
query sees the surface only through the shared encoder.

- ``observed_sphere``: the same 512 Fibonacci directions u, but placed per
  crop on a sphere around the observed cloud (``frames_pred.pts3d``, in the
  normalised cloud frame): centre = the points' median, radius = the
  ``surface_sphere_quantile`` quantile of their distances to it. The query is
  MLP(u) and its PointROPE position is centre + radius * u. Positions stay
  fixed over the layers (as ShapeXAttn's coarse points do).
``surface_cls_dim: k`` (e.g. 384 for DINOv2-S) also feeds the DINOv2 cls
feature (``frames_pred.feat`` as it reaches this stage) into both point MLPs:
coarse reads [t_i | cls], fine reads [t_i | cls | coarse output].

- ``sphere``: 512 fixed anchors spread uniformly on the unit sphere in NOCS
  units (a Fibonacci lattice); the query is MLP(anchor) and its PointROPE
  position is the anchor.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


def rope3d(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """PointROPE in torch: x (B, H, N, D), pos (B, N, 3) float.

    The head dim is split into x/y/z thirds; in each, the first and second half
    are rotated as (u, v) pairs by pos[axis] * base^(-i / (D/6)) — the layout of
    o3b.model.litept.pointrope's kernel.
    """
    D = x.shape[-1]
    Q = D // 6
    inv = base ** (-torch.arange(Q, device=x.device, dtype=torch.float32) / Q)  # (Q,)
    out = []
    for a in range(3):
        seg = x[..., a * 2 * Q:(a + 1) * 2 * Q]
        u, v = seg[..., :Q], seg[..., Q:]
        f = (pos[..., a, None].float() * inv)[:, None]                         # (B, 1, N, Q)
        cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)
        out += [u * cos - v * sin, v * cos + u * sin]
    return torch.cat(out, dim=-1)


class _Attn(nn.Module):
    def __init__(self, dim, num_heads, rope_freq, dropout):
        super().__init__()
        if (dim // num_heads) % 6 or dim % num_heads:
            raise ValueError(f"XAttnPool: head dim {dim}/{num_heads} must be a multiple of 6")
        self.h = num_heads
        self.rope_freq = rope_freq
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, x_pos, ctx, ctx_pos, ctx_mask=None, pair_mask=None):
        """ctx_mask: (B, S) valid keys; pair_mask: (N, S) allowed query-key pairs."""
        B, N, C = x.shape
        S = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).reshape(B, S, 2, self.h, -1).permute(2, 0, 3, 1, 4)
        q = rope3d(q, x_pos, self.rope_freq)
        k = rope3d(k, ctx_pos, self.rope_freq)
        mask = ctx_mask[:, None, None, :] if ctx_mask is not None else None
        if pair_mask is not None:
            mask = pair_mask[None, None] if mask is None else mask & pair_mask[None, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class _Layer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, rope_freq, dropout, self_attn):
        super().__init__()
        self.self_attn = _Attn(dim, num_heads, rope_freq, dropout) if self_attn else None
        self.norm_sa = nn.LayerNorm(dim) if self_attn else None
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = _Attn(dim, num_heads, rope_freq, dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, q, q_pos, tok, tok_pos, tok_mask, self_mask=None):
        if self.self_attn is not None:
            h = self.norm_sa(q)
            q = q + self.self_attn(h, q_pos, h, q_pos, pair_mask=self_mask)
        q = q + self.cross(self.norm_q(q), q_pos, self.norm_kv(tok), tok_pos, tok_mask)
        return q + self.ffn(self.norm_ffn(q))


def fibonacci_sphere(n: int) -> torch.Tensor:
    """n points spread uniformly on the unit sphere (Fibonacci lattice), (n, 3)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = (1.0 - z * z).sqrt()
    phi = i * torch.pi * (3.0 - 5.0 ** 0.5)
    return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1).float()


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@register_model("XAttnPool")
class XAttnPool(OD3D_Model):
    def __init__(
        self,
        dim: int = 504,
        num_queries: int = 1,
        num_layers: int = 3,
        num_heads: int = 28,
        mlp_ratio: int = 4,
        self_attn_every: int = 0,
        rope_freq: float = 100.0,
        rope_unit: float = 0.08,
        query_pos=(0.0, 0.0, 0.0),
        dropout: float = 0.0,
        out_feat_append_in_feat: bool = True,
        surface_queries: int = 0,
        surface_anchor: str = "learned",
        surface_fold: int = 4,
        surface_hidden_dim: int = 256,
        surface_nocs: bool = False,
        surface_pose_attn: bool = True,
        surface_sphere_quantile: float = 0.9,
        surface_cls_dim: int = 0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.surface_queries = surface_queries
        self.surface_anchor = surface_anchor
        self.surface_fold = surface_fold
        self.surface_nocs = surface_nocs
        self.surface_pose_attn = surface_pose_attn
        self.surface_sphere_quantile = surface_sphere_quantile
        self.surface_cls_dim = surface_cls_dim
        if surface_queries:
            if surface_anchor == "learned":
                self.surface_q = nn.Parameter(torch.randn(1, surface_queries, dim) * 0.02)
            elif surface_anchor in ("sphere", "observed_sphere"):
                self.register_buffer("surface_pos", fibonacci_sphere(surface_queries), persistent=False)
                self.embed_anchor = _mlp([3, dim // 2, dim])
            else:
                raise ValueError("XAttnPool: surface_anchor must be learned, sphere or observed_sphere, "
                                 f"got {surface_anchor}")
            self.out_ch = 3 + (3 if surface_nocs else 0)
            cd = surface_cls_dim
            self.coarse = _mlp([dim + cd, surface_hidden_dim, surface_hidden_dim, self.out_ch])
            self.fine = _mlp([dim + cd + self.out_ch, surface_hidden_dim, surface_hidden_dim,
                              surface_fold * self.out_ch])
        self.rope_unit = rope_unit
        self.out_feat_append_in_feat = out_feat_append_in_feat
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.register_buffer("query_pos", torch.tensor(query_pos, dtype=torch.float32),
                             persistent=False)
        self.layers = nn.ModuleList([
            _Layer(dim, num_heads, mlp_ratio, rope_freq, dropout,
                   self_attn=bool(self_attn_every) and (i + 1) % self_attn_every == 0)
            for i in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, frames_gt, frames_pred=None):
        tok, tok_coord, tok_mask = frames_pred.tokens, frames_pred.tokens_coord, frames_pred.tokens_mask
        if tok is None:
            raise RuntimeError("XAttnPool: no frames_pred.tokens — run LitePT with out_tokens: true")
        # LitePT's attention runs in half precision; its tokens may come out fp16
        tok, tok_coord = tok.float(), tok_coord.float()
        B = tok.shape[0]
        q = self.queries.expand(B, -1, -1)
        q_pos = (self.query_pos / self.rope_unit).expand(B, q.shape[1], 3)
        if self.surface_queries:
            S = self.surface_queries
            if self.surface_anchor == "learned":
                q_s = self.surface_q.expand(B, -1, -1)
                pos_s = q_pos[:, :1].expand(B, S, 3)
            elif self.surface_anchor == "sphere":
                q_s = self.embed_anchor(self.surface_pos)[None].expand(B, -1, -1)
                pos_s = (self.surface_pos / self.rope_unit)[None].expand(B, S, 3)
            else:
                q_s = self.embed_anchor(self.surface_pos)[None].expand(B, -1, -1)
                pos_s = self.observed_sphere(frames_pred.pts3d) / self.rope_unit
            q = torch.cat([q, q_s], dim=1)
            q_pos = torch.cat([q_pos, pos_s], dim=1)
        tok_pos = tok_coord / self.rope_unit
        self_mask = None
        if self.surface_queries and not self.surface_pose_attn:
            # pose and surface queries do not see each other in self-attention;
            # each group still attends within itself
            g = torch.arange(q.shape[1], device=q.device) >= self.num_queries
            self_mask = g[:, None] == g[None, :]
        for layer in self.layers:
            q = layer(q, q_pos, tok, tok_pos, tok_mask, self_mask)
        q = self.norm_out(q)
        pooled = q[:, :self.num_queries].mean(dim=1)                # (B, dim)

        if self.surface_queries:
            self._surface(q[:, self.num_queries:], frames_pred)

        if self.out_feat_append_in_feat and frames_pred.feat is not None:
            frames_pred.feat = torch.cat([pooled, frames_pred.feat], dim=1)
        else:
            frames_pred.feat = pooled
        return frames_gt, frames_pred

    def observed_sphere(self, pts3d):
        """(B, S, 3) anchors on a sphere around the observed cloud (B, N, 3)."""
        if pts3d is None:
            raise RuntimeError("XAttnPool: surface_anchor observed_sphere needs frames_pred.pts3d")
        pts3d = pts3d.detach().float()
        center = pts3d.median(dim=1).values                        # (B, 3)
        radius = (pts3d - center[:, None]).norm(dim=-1).quantile(
            self.surface_sphere_quantile, dim=1)                    # (B,)
        return center[:, None] + radius[:, None, None] * self.surface_pos[None]

    def _surface(self, t, frames_pred):
        """Two MLPs per refined surface token: coarse point, then K fine points."""
        B, S, _ = t.shape
        K = self.surface_fold
        if self.surface_cls_dim:
            # the DINOv2 cls feature, still in frames_pred.feat here (the pooled
            # feature is appended to it only after this), joins every token
            cls = frames_pred.feat
            if cls is None or cls.shape[-1] != self.surface_cls_dim:
                raise RuntimeError(f"XAttnPool: surface_cls_dim {self.surface_cls_dim} but frames_pred.feat is "
                                   f"{None if cls is None else tuple(cls.shape)}")
            t = torch.cat([t, cls.float()[:, None].expand(B, S, -1)], dim=-1)
        c = self.coarse(t)                                          # (B, S, out_ch)
        f = self.fine(torch.cat([t, c], dim=-1)).reshape(B, S, K, self.out_ch)
        coarse = c[..., :3]
        frames_pred.pts3d_shape_coarse = coarse
        frames_pred.pts3d_shape_fine = (coarse[:, :, None] + f[..., :3]).reshape(B, S * K, 3)
        if self.surface_nocs:
            frames_pred.nocs_shape_coarse = c[..., 3:6]
            frames_pred.nocs_shape_fine = f[..., 3:6].reshape(B, S * K, 3)
        frames_pred.cam_tform4x4_pts3d_shape = frames_pred.obj_tform4x4_obj_nocs_0c
