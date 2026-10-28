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

With ``pose_query: true`` the decoder also refines the pose (H02-T04 variant
b, option 2): a pose token, projected from the global feature and placed at
the cloud centre, joins the surface tokens as one more query. It cross-attends
over the stage-4 tokens like them and exchanges with them in the self-attention
layers. ``feat`` then becomes ``feat + pose_out(refined pose token)``, which is
what PoseMLP reads. So the pose is read *after* the surface tokens have
informed it, and this module is no longer a side branch. ``pose_out`` is
zero-initialised, so the chain starts out as the pose-only one.

With ``point_feat_dim: k`` every coarse and fine point also carries k
learned feature channels (``feat_shape_coarse`` / ``feat_shape_fine``), e.g.
PCA-projected DINOv2 / PartField surface features; supervised like the NOCS on
the Chamfer matching (``TrinityMethod._shape_metrics``, ``shape_feat_l1_*``).

With ``nocs: true`` every coarse and fine point also carries its normalised
object coordinates (``nocs_shape_coarse`` / ``nocs_shape_fine``, NCDS units:
the object's longest side spans 2): the coarse MLP and the fold MLP each emit
3 more numbers per point. TrinityMethod supervises them on the Chamfer
matching (camera-space nearest neighbours) — see ``_shape_metrics``. A side branch like ShapeMLP: ``feat`` is left alone, and it
has to sit before PoseMLP.

With ``mask_tokens: true`` (H07, needs ``point_feat_dim``) the decoder also
carries training-only *masked tokens* built from GT surface points, which the
method puts on ``frames_gt`` (``mask_pts3d_cam`` (B, K, 3) in camera space,
``mask_feat`` (B, K, point_feat_dim)). The first half knows a point's XYZ and
predicts its features (``mask_pred_ppf``); the second half knows the features
and predicts the XYZ (``mask_pred_xyz``, normalised cloud frame, at the cloud
origin for PointROPE). Targets go to ``mask_tgt_ppf`` / ``mask_tgt_xyz``. They
run through the same layers next to the surface tokens, with a self-attention
mask: surface tokens never see them (so their outputs equal a pass without
them) and masked tokens never see each other; with ``mask_see_surface: true``
they see the surface tokens and themselves, otherwise only themselves. Without
``mask_pts3d_cam`` (evaluation) nothing changes.
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
        pose_query: bool = False,
        point_feat_dim: int = 0,
        moe_experts: int = 0,
        moe_top_k: int = 2,
        mask_tokens: bool = False,
        mask_see_surface: bool = True,
    ):
        super().__init__()
        self.nocs = nocs
        self.pose_query = pose_query
        self.coarse_count = coarse_count
        self.fold_count = fold_count
        self.rope_unit = rope_unit

        self.point_feat_dim = point_feat_dim
        # point (+ its NOCS) (+ point_feat_dim feature channels)
        self.out_ch = 3 + (3 if nocs else 0) + point_feat_dim
        self.coarse = _mlp([in_dim, hidden_dim, hidden_dim, coarse_count * self.out_ch])

        # surface token = embed(point) + embed(global feature)
        self.embed_pt = _mlp([3, dim // 2, dim])
        self.embed_glob = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            _Layer(dim, num_heads, mlp_ratio, rope_freq, dropout,
                   self_attn=bool(self_attn_every) and (i + 1) % self_attn_every == 0,
                   moe_experts=moe_experts, moe_top_k=moe_top_k)   # H02-T21: MoE FFNs (o3b/model/moe.py)
            for i in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim)
        if pose_query:
            self.embed_pose = nn.Linear(in_dim, dim)
            self.pose_out = nn.Linear(dim, in_dim)
            nn.init.zeros_(self.pose_out.weight)
            nn.init.zeros_(self.pose_out.bias)

        self.mask_tokens = mask_tokens
        self.mask_see_surface = mask_see_surface
        if mask_tokens:
            if not point_feat_dim:
                raise ValueError("ShapeXAttn: mask_tokens needs point_feat_dim")
            self.embed_mask_feat = _mlp([point_feat_dim, dim // 2, dim])
            self.mask_type = nn.Parameter(torch.zeros(2, dim))   # 0: XYZ known, 1: features known
            nn.init.normal_(self.mask_type, std=0.02)
            self.mask_head = _mlp([dim, fold_hidden_dim, 3 + point_feat_dim])

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
        if self.pose_query:
            # pose token first, at the cloud frame's origin (the observed
            # cloud's median), like XAttnPool's pose query
            q = torch.cat([self.embed_pose(feat)[:, None], q], dim=1)
            q_pos = torch.cat([q_pos.new_zeros(B, 1, 3), q_pos], dim=1)
        tok_pos = tok_coord / self.rope_unit
        n_main, self_mask = q.shape[1], None
        mask_pts = getattr(frames_gt, "mask_pts3d_cam", None) if self.mask_tokens else None
        if mask_pts is not None:
            q, q_pos, self_mask, mask_tgt = self._add_mask_tokens(frames_gt, frames_pred, feat, q, q_pos)
        for layer in self.layers:
            q = layer(q, q_pos, tok, tok_pos, tok_mask, self_mask=self_mask)
        q = self.norm_out(q)                                      # (B, [1 +] C [+ K], dim)
        if mask_pts is not None:
            out = self.mask_head(q[:, n_main:])
            k1 = mask_tgt[0].shape[1]
            frames_pred.mask_pred_ppf, frames_pred.mask_tgt_ppf = out[:, :k1, 3:], mask_tgt[0]
            frames_pred.mask_pred_xyz, frames_pred.mask_tgt_xyz = out[:, k1:, :3], mask_tgt[1]
            q = q[:, :n_main]
        if self.pose_query:
            frames_pred.feat = feat + self.pose_out(q[:, 0])
            q = q[:, 1:]

        seed = self.fold_grid[None, None].expand(B, C, K, 2)
        fold_out = self.fold(torch.cat([q[:, :, None].expand(B, C, K, q.shape[-1]), seed], dim=-1))
        fine = (coarse[:, :, None] + fold_out[..., :3]).reshape(B, C * K, 3)
        o = 3
        if self.nocs:
            frames_pred.nocs_shape_coarse = coarse_out[..., o:o + 3]
            frames_pred.nocs_shape_fine = fold_out[..., o:o + 3].reshape(B, C * K, 3)
            o += 3
        if self.point_feat_dim:
            frames_pred.feat_shape_coarse = coarse_out[..., o:]
            frames_pred.feat_shape_fine = fold_out[..., o:].reshape(B, C * K, self.point_feat_dim)

        frames_pred.pts3d_shape_coarse = coarse
        frames_pred.pts3d_shape_fine = fine
        frames_pred.cam_tform4x4_pts3d_shape = frames_pred.obj_tform4x4_obj_nocs_0c
        return frames_gt, frames_pred

    def _add_mask_tokens(self, frames_gt, frames_pred, feat, q, q_pos):
        """Append the H07 masked tokens to the queries; returns the queries, their
        positions, the self-attention mask and the (features, XYZ) targets."""
        from o3b.cv.geometry.transform import inv_tform4x4, transf3d_broadcast
        pts = frames_gt.mask_pts3d_cam.float()
        ppf = frames_gt.mask_feat.float()
        B, K = pts.shape[:2]
        k1 = K // 2
        # camera -> the normalised cloud frame the surface points live in
        pts = transf3d_broadcast(pts3d=pts, transf4x4=inv_tform4x4(
            frames_pred.obj_tform4x4_obj_nocs_0c.detach().float())[:, None])
        glob = self.embed_glob(feat)[:, None]
        m_xyz = self.embed_pt(pts[:, :k1]) + glob + self.mask_type[0]
        m_ppf = self.embed_mask_feat(ppf[:, k1:]) + glob + self.mask_type[1]
        m_pos = torch.cat([pts[:, :k1] / self.rope_unit, pts.new_zeros(B, K - k1, 3)], dim=1)

        n, dev = q.shape[1], q.device
        allow = torch.zeros(n + K, n + K, dtype=torch.bool, device=dev)
        allow[:n, :n] = True                                  # main tokens: each other only
        if self.mask_see_surface:
            allow[n:, :n] = True                              # masked tokens: the main tokens ...
        allow[n:, n:] = torch.eye(K, dtype=torch.bool, device=dev)   # ... and themselves
        return (torch.cat([q, m_xyz, m_ppf], dim=1), torch.cat([q_pos, m_pos], dim=1),
                allow, (ppf[:, :k1], pts[:, k1:]))
