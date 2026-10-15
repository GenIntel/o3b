"""ShapeXAttn: surface tokens cross-attending over LitePT's last-stage tokens.

The shape branch of ShapeMLP with a decoder in the middle. From the global
feature an MLP regresses ``coarse_count`` coarse points (supervised as before);
each becomes a surface token — embedded from its point and the global feature —
which cross-attends over LitePT's stage-4 tokens (``tokens``, ``tokens_coord``,
``tokens_mask``, from LitePT with ``out_tokens: true``) with self-attention among
the surface tokens every ``self_attn_every`` layers. Every refined token is
then folded into ``fold_count`` fine points (its point + offsets read from the
token and a 2x2 grid seed): 512 x 4 = 2048.

So the shape loss reaches the encoder per token (through the cross-attention)
and not only through the pooled global feature, which is the difference to
ShapeMLP that H02-T04 tests.

Positions are PointROPE as in XAttnPool (same layers, o3b.model.xattnpool):
a surface token sits at its coarse point / ``rope_unit``, a stage-4 token at
its pooled coord / ``rope_unit``, both in the normalised cloud frame. The
coarse points are detached for the positions, so the rotary angles carry no
gradient.

Writes ``pts3d_shape_coarse`` / ``pts3d_shape_fine`` /
``cam_tform4x4_pts3d_shape`` like ShapeMLP, so TrinityMethod's shape metrics
apply unchanged.

With ``nocs: true`` every coarse and fine point also carries its normalised
object coordinates (``nocs_shape_coarse`` / ``nocs_shape_fine``, NCDS units:
the object's longest side spans 2): the coarse MLP and the fold MLP each emit
3 more numbers per point. TrinityMethod supervises them on the Chamfer
matching (camera-space nearest neighbours) — see ``_shape_metrics``. A side branch like ShapeMLP: ``feat`` is left alone, and it
has to sit before PoseMLP.
"""

import logging

import torch
import torch.nn as nn

from o3b.model.model import OD3D_Model, register_model
from o3b.model.shapemlp.model import _mlp
from o3b.model.xattnpool.model import _Layer

logger = logging.getLogger(__name__)


@register_model("ShapeXAttn")
class ShapeXAttn(OD3D_Model):
    def __init__(
        self,
        in_dim: int,
        dim: int = 504,
        hidden_dim: int = 1024,
        coarse_count: int = 512,
        fold_count: int = 4,
        fold_hidden_dim: int = 256,
        fold_grid_scale: float = 0.05,
        num_layers: int = 4,
        num_heads: int = 28,
        mlp_ratio: int = 2,
        self_attn_every: int = 2,
        rope_freq: float = 100.0,
        rope_unit: float = 0.08,
        dropout: float = 0.0,
        nocs: bool = False,
    ):
        super().__init__()
        self.nocs = nocs
        self.coarse_count = coarse_count
        self.fold_count = fold_count
        self.rope_unit = rope_unit

        self.out_ch = 6 if nocs else 3          # point (+ its NOCS)
        self.coarse = _mlp([in_dim, hidden_dim, hidden_dim, coarse_count * self.out_ch])

        # surface token = embed(point) + embed(global feature)
        self.embed_pt = _mlp([3, dim // 2, dim])
        self.embed_glob = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            _Layer(dim, num_heads, mlp_ratio, rope_freq, dropout,
                   self_attn=bool(self_attn_every) and (i + 1) % self_attn_every == 0)
            for i in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim)

        side = int(round(fold_count ** 0.5))
        if side * side != fold_count:
            raise ValueError(f"ShapeXAttn: fold_count must be a square, got {fold_count}")
        lin = torch.linspace(-fold_grid_scale, fold_grid_scale, side)
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(-1, 2)
        self.register_buffer("fold_grid", grid, persistent=False)
        self.fold = _mlp([dim + 2, fold_hidden_dim, fold_hidden_dim, self.out_ch])

    def forward(self, frames_gt, frames_pred=None):
        feat = frames_pred.feat                                   # (B, F) global
        tok, tok_coord, tok_mask = frames_pred.tokens, frames_pred.tokens_coord, frames_pred.tokens_mask
        if tok is None:
            raise RuntimeError("ShapeXAttn: no frames_pred.tokens — run LitePT with out_tokens: true")
        tok, tok_coord = tok.float(), tok_coord.float()
        B = feat.shape[0]
        C, K = self.coarse_count, self.fold_count

        coarse_out = self.coarse(feat).reshape(B, C, self.out_ch)
        coarse = coarse_out[..., :3]                              # (B, C, 3)

        q = self.embed_pt(coarse) + self.embed_glob(feat)[:, None]
        q_pos = coarse.detach() / self.rope_unit
        tok_pos = tok_coord / self.rope_unit
        for layer in self.layers:
            q = layer(q, q_pos, tok, tok_pos, tok_mask)
        q = self.norm_out(q)                                      # (B, C, dim)

        seed = self.fold_grid[None, None].expand(B, C, K, 2)
        fold_out = self.fold(torch.cat([q[:, :, None].expand(B, C, K, q.shape[-1]), seed], dim=-1))
        fine = (coarse[:, :, None] + fold_out[..., :3]).reshape(B, C * K, 3)
        if self.nocs:
            frames_pred.nocs_shape_coarse = coarse_out[..., 3:]
            frames_pred.nocs_shape_fine = fold_out[..., 3:].reshape(B, C * K, 3)

        frames_pred.pts3d_shape_coarse = coarse
        frames_pred.pts3d_shape_fine = fine
        frames_pred.cam_tform4x4_pts3d_shape = frames_pred.obj_tform4x4_obj_nocs_0c
        return frames_gt, frames_pred
