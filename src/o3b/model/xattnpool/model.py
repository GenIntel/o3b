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

    def forward(self, x, x_pos, ctx, ctx_pos, ctx_mask=None):
        B, N, C = x.shape
        S = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).reshape(B, S, 2, self.h, -1).permute(2, 0, 3, 1, 4)
        q = rope3d(q, x_pos, self.rope_freq)
        k = rope3d(k, ctx_pos, self.rope_freq)
        mask = ctx_mask[:, None, None, :] if ctx_mask is not None else None
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

    def forward(self, q, q_pos, tok, tok_pos, tok_mask):
        if self.self_attn is not None:
            h = self.norm_sa(q)
            q = q + self.self_attn(h, q_pos, h, q_pos)
        q = q + self.cross(self.norm_q(q), q_pos, self.norm_kv(tok), tok_pos, tok_mask)
        return q + self.ffn(self.norm_ffn(q))


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
    ):
        super().__init__()
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
        B = tok.shape[0]
        q = self.queries.expand(B, -1, -1)
        q_pos = (self.query_pos / self.rope_unit).expand(B, q.shape[1], 3)
        tok_pos = tok_coord / self.rope_unit
        for layer in self.layers:
            q = layer(q, q_pos, tok, tok_pos, tok_mask)
        pooled = self.norm_out(q).mean(dim=1)                       # (B, dim)

        if self.out_feat_append_in_feat and frames_pred.feat is not None:
            frames_pred.feat = torch.cat([pooled, frames_pred.feat], dim=1)
        else:
            frames_pred.feat = pooled
        return frames_gt, frames_pred
