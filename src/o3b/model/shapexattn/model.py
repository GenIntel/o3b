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

The same tokens answer queries at test time: ``query_pts3d_cam`` (B, Q, 3,
camera space) on ``frames_gt`` gives XYZ-known tokens whose predicted
features land in ``query_pred_feat`` (B, Q, point_feat_dim), and
``query_feat`` (B, Q, point_feat_dim) gives feature-known tokens whose
predicted XYZ lands in ``query_pred_pts3d_cam`` (B, Q, 3, camera space). Either
may be given alone. TrinityMethod chains them for 3-D correspondence: the
query frame turns keypoints into features, the target frame turns those
features into its keypoints.

With ``learned_queries: true`` (H08) there is no coarse MLP and no global
feature MLP: ``coarse_count`` learned surface queries start at learned 3-D
anchors and a learned pose query at the cloud origin, each plus a projection of
``feat`` (the DINOv2 cls feature, ``in_dim``). ``num_passes`` runs the same
layers that many times; after every pass a point head moves each coarse point by
an offset (and gives its NOCS / features) and an own PoseMLP reads the pose
token, and the next pass takes those points (and the predicted object centre)
as the queries' PointROPE positions. Earlier passes are deep supervision
(``pts3d_shape_aux`` / ``feat_shape_aux`` / ``nocs_shape_aux`` / ``pose_aux``).
The pose is written as ``cam_tform4x4_obj`` / ``obj_size3d`` directly, so the
chain needs no MLP or PoseMLP after this module.
"""

import logging

import torch
import torch.nn as nn

from o3b.model.model import OD3D_Model, register_model
from o3b.model.shapemlp.model import _mlp
from o3b.model.xattnpool.model import _Layer

logger = logging.getLogger(__name__)


class _Bag:
    """Attribute bag reading unset fields as None (for the internal PoseMLP)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        return None


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
        learned_queries: bool = False,
        num_passes: int = 1,
        anchor_extent: float = 0.5,
        learned_anchors: bool = True,
    ):
        super().__init__()
        self.nocs = nocs
        self.pose_query = pose_query
        self.coarse_count = coarse_count
        self.fold_count = fold_count
        self.rope_unit = rope_unit
        self.learned_queries = learned_queries
        self.num_passes = num_passes
        if num_passes > 1 and not learned_queries:
            raise ValueError("ShapeXAttn: num_passes > 1 needs learned_queries")

        self.point_feat_dim = point_feat_dim
        # point (+ its NOCS) (+ point_feat_dim feature channels)
        self.out_ch = 3 + (3 if nocs else 0) + point_feat_dim
        if learned_queries:
            # H08: no coarse MLP. Learned surface queries at learned anchors in the
            # cloud frame (uniform in the cube +-anchor_extent at init), a learned
            # pose query, and a per-pass embedding; each pass ends in a point head
            # (offset + NOCS / features per surface token) and a pose head.
            g = torch.Generator().manual_seed(0)
            self.query = nn.Parameter(torch.randn(coarse_count, dim, generator=g) * 0.02)
            anchor = (torch.rand(coarse_count, 3, generator=g) * 2 - 1) * anchor_extent
            if learned_anchors:
                self.anchor = nn.Parameter(anchor)
            else:                                     # all queries start at the cloud origin
                self.register_buffer("anchor", torch.zeros_like(anchor), persistent=False)
            self.pass_embed = nn.Parameter(torch.randn(num_passes, dim, generator=g) * 0.02)
            self.point_head = nn.Linear(dim, self.out_ch)
            if pose_query:
                from o3b.model.posemlp.model import PoseMLP
                self.pose_query_embed = nn.Parameter(torch.randn(1, dim, generator=g) * 0.02)
                self.pose_head = PoseMLP(num_layers=1, hidden_dim=0, dropout=0., in_dim=dim, activation="none",
                                         bias=True, out_dims=[3, 6, 3])
        else:
            self.coarse = _mlp([in_dim, hidden_dim, hidden_dim, coarse_count * self.out_ch])

        # surface token = embed(point) + embed(global feature); with learned queries
        # only the masked XYZ tokens embed a point
        if not learned_queries or mask_tokens:
            self.embed_pt = _mlp([3, dim // 2, dim])
        self.embed_glob = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            _Layer(dim, num_heads, mlp_ratio, rope_freq, dropout,
                   self_attn=bool(self_attn_every) and (i + 1) % self_attn_every == 0,
                   moe_experts=moe_experts, moe_top_k=moe_top_k)   # H02-T21: MoE FFNs (o3b/model/moe.py)
            for i in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim)
        if pose_query and not learned_queries:
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
        if self.learned_queries:
            return self._forward_learned(frames_gt, frames_pred)
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
        qry_pts = getattr(frames_gt, "query_pts3d_cam", None) if self.mask_tokens else None
        qry_feat = getattr(frames_gt, "query_feat", None) if self.mask_tokens else None
        if mask_pts is not None:
            q, q_pos, self_mask, mask_tgt = self._add_mask_tokens(frames_gt, frames_pred, feat, q, q_pos)
        elif qry_pts is not None or qry_feat is not None:
            xyz = None if qry_pts is None else self._cam_to_cloud(frames_pred, qry_pts.float())
            q, q_pos, self_mask = self._mask_tokens(feat, q, q_pos, xyz,
                                                    None if qry_feat is None else qry_feat.float())
        for layer in self.layers:
            q = layer(q, q_pos, tok, tok_pos, tok_mask, self_mask=self_mask)
        q = self.norm_out(q)                                      # (B, [1 +] C [+ K], dim)
        if mask_pts is not None or qry_pts is not None or qry_feat is not None:
            self._mask_out(frames_pred, q[:, n_main:], mask_tgt if mask_pts is not None else None,
                           qry_pts, qry_feat)
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

    def _forward_learned(self, frames_gt, frames_pred):
        """H08: learned pose / surface queries, ``num_passes`` passes through the same
        layers. Pass 1 starts the surface queries at their learned anchors and the
        pose query at the cloud origin; every pass adds its point head's offset to
        the coarse points and reads the pose from the pose token, and the next pass
        moves the queries' PointROPE positions there (the pose token to its predicted
        centre). The token features carry over. Every pass but the last is handed to
        TrinityMethod as deep supervision (``pts3d_shape_aux`` / ``feat_shape_aux`` /
        ``nocs_shape_aux`` / ``pose_aux``); the fold reads the last pass only."""
        feat = frames_pred.feat                                   # (B, F): the DINOv2 cls feature
        tok, tok_coord, tok_mask = frames_pred.tokens, frames_pred.tokens_coord, frames_pred.tokens_mask
        if tok is None:
            raise RuntimeError("ShapeXAttn: no frames_pred.tokens — run LitePT with out_tokens: true")
        tok, tok_coord = tok.float(), tok_coord.float()
        B = feat.shape[0]
        C, K = self.coarse_count, self.fold_count
        obj_tform = frames_pred.obj_tform4x4_obj_nocs_0c

        glob = self.embed_glob(feat)[:, None]
        q = self.query[None].expand(B, -1, -1) + glob
        coarse = self.anchor[None].expand(B, -1, -1)
        q_pos = coarse.detach() / self.rope_unit
        p0 = 1 if self.pose_query else 0
        if self.pose_query:
            q = torch.cat([self.pose_query_embed[None].expand(B, -1, -1) + glob, q], dim=1)
            q_pos = torch.cat([q_pos.new_zeros(B, 1, 3), q_pos], dim=1)
        tok_pos = tok_coord / self.rope_unit
        n_main, self_mask = q.shape[1], None
        mask_pts = getattr(frames_gt, "mask_pts3d_cam", None) if self.mask_tokens else None
        qry_pts = getattr(frames_gt, "query_pts3d_cam", None) if self.mask_tokens else None
        qry_feat = getattr(frames_gt, "query_feat", None) if self.mask_tokens else None
        mask_tgt = None
        if mask_pts is not None:
            q, q_pos, self_mask, mask_tgt = self._add_mask_tokens(frames_gt, frames_pred, feat, q, q_pos)
        elif qry_pts is not None or qry_feat is not None:
            xyz = None if qry_pts is None else self._cam_to_cloud(frames_pred, qry_pts.float())
            q, q_pos, self_mask = self._mask_tokens(feat, q, q_pos, xyz,
                                                    None if qry_feat is None else qry_feat.float())

        aux_pts, aux_extra, aux_pose = [], [], []
        for p in range(self.num_passes):
            q = q + self.pass_embed[p]
            for layer in self.layers:
                q = layer(q, q_pos, tok, tok_pos, tok_mask, self_mask=self_mask)
            h = self.norm_out(q)
            out = self.point_head(h[:, p0:n_main])
            coarse = coarse + out[..., :3]
            extra = out[..., 3:]
            pose = None
            if self.pose_query:
                bag = _Bag(feat=h[:, 0], obj_tform4x4_obj_nocs_0c=obj_tform)
                _, bag = self.pose_head(None, bag)
                pose = (bag.cam_tform4x4_obj, bag.obj_size3d)
            if p < self.num_passes - 1:
                aux_pts.append(coarse)
                aux_extra.append(extra)
                if pose is not None:
                    aux_pose.append(pose)
                    centre = self.pose_head.nets[0](h[:, 0]).detach()      # cloud-frame translation
                    q_pos = torch.cat([centre[:, None] / self.rope_unit, coarse.detach() / self.rope_unit,
                                       q_pos[:, n_main:]], dim=1)
                else:
                    q_pos = torch.cat([coarse.detach() / self.rope_unit, q_pos[:, n_main:]], dim=1)

        if mask_pts is not None or qry_pts is not None or qry_feat is not None:
            self._mask_out(frames_pred, h[:, n_main:], mask_tgt, qry_pts, qry_feat)
        h = h[:, p0:n_main]
        if pose is not None:
            frames_pred.cam_tform4x4_obj, frames_pred.obj_size3d = pose

        seed = self.fold_grid[None, None].expand(B, C, K, 2)
        fold_out = self.fold(torch.cat([h[:, :, None].expand(B, C, K, h.shape[-1]), seed], dim=-1))
        fine = (coarse[:, :, None] + fold_out[..., :3]).reshape(B, C * K, 3)
        o = 0
        if self.nocs:
            frames_pred.nocs_shape_coarse = extra[..., :3]
            frames_pred.nocs_shape_fine = fold_out[..., 3:6].reshape(B, C * K, 3)
            o = 3
        if self.point_feat_dim:
            frames_pred.feat_shape_coarse = extra[..., o:]
            frames_pred.feat_shape_fine = fold_out[..., 3 + o:].reshape(B, C * K, self.point_feat_dim)
        if aux_pts:
            frames_pred.pts3d_shape_aux = aux_pts
            if self.nocs:
                frames_pred.nocs_shape_aux = [e[..., :3] for e in aux_extra]
            if self.point_feat_dim:
                frames_pred.feat_shape_aux = [e[..., o:] for e in aux_extra]
        if aux_pose:
            frames_pred.pose_aux = aux_pose

        frames_pred.pts3d_shape_coarse = coarse
        frames_pred.pts3d_shape_fine = fine
        frames_pred.cam_tform4x4_pts3d_shape = obj_tform
        return frames_gt, frames_pred

    def _mask_out(self, frames_pred, h, mask_tgt, qry_pts, qry_feat):
        """Read the masked tokens (``h``, normalised): training targets (``mask_tgt``)
        or test-time query answers."""
        out = self.mask_head(h)
        if mask_tgt is not None:
            k1 = mask_tgt[0].shape[1]
            frames_pred.mask_pred_ppf, frames_pred.mask_tgt_ppf = out[:, :k1, 3:], mask_tgt[0]
            frames_pred.mask_pred_xyz, frames_pred.mask_tgt_xyz = out[:, k1:, :3], mask_tgt[1]
            return
        k1 = 0 if qry_pts is None else qry_pts.shape[1]
        if qry_pts is not None:
            frames_pred.query_pred_feat = out[:, :k1, 3:]
        if qry_feat is not None:
            from o3b.cv.geometry.transform import transf3d_broadcast
            frames_pred.query_pred_pts3d_cam = transf3d_broadcast(
                pts3d=out[:, k1:, :3],
                transf4x4=frames_pred.obj_tform4x4_obj_nocs_0c.float()[:, None])

    def _add_mask_tokens(self, frames_gt, frames_pred, feat, q, q_pos):
        """Append the H07 masked tokens to the queries; returns the queries, their
        positions, the self-attention mask and the (features, XYZ) targets."""
        pts = self._cam_to_cloud(frames_pred, frames_gt.mask_pts3d_cam.float())
        ppf = frames_gt.mask_feat.float()
        k1 = pts.shape[1] // 2
        q, q_pos, allow = self._mask_tokens(feat, q, q_pos, pts[:, :k1], ppf[:, k1:])
        return q, q_pos, allow, (ppf[:, :k1], pts[:, k1:])

    @staticmethod
    def _cam_to_cloud(frames_pred, pts):
        """Camera -> the normalised cloud frame the surface points live in."""
        from o3b.cv.geometry.transform import inv_tform4x4, transf3d_broadcast
        return transf3d_broadcast(pts3d=pts, transf4x4=inv_tform4x4(
            frames_pred.obj_tform4x4_obj_nocs_0c.detach().float())[:, None])

    def _mask_tokens(self, feat, q, q_pos, xyz, ppf):
        """Append XYZ-known tokens (``xyz``, cloud frame) then feature-known tokens
        (``ppf``) after the queries; either may be None. Returns the queries,
        their positions and the self-attention mask."""
        B = q.shape[0]
        glob = self.embed_glob(feat)[:, None]
        toks, pos = [q], [q_pos]
        if xyz is not None:
            toks.append(self.embed_pt(xyz) + glob + self.mask_type[0])
            pos.append(xyz / self.rope_unit)
        if ppf is not None:
            toks.append(self.embed_mask_feat(ppf) + glob + self.mask_type[1])
            pos.append(ppf.new_zeros(B, ppf.shape[1], 3))

        n, dev = q.shape[1], q.device
        K = sum(t.shape[1] for t in toks[1:])
        allow = torch.zeros(n + K, n + K, dtype=torch.bool, device=dev)
        allow[:n, :n] = True                                  # main tokens: each other only
        if self.mask_see_surface:
            allow[n:, :n] = True                              # masked tokens: the main tokens ...
        allow[n:, n:] = torch.eye(K, dtype=torch.bool, device=dev)   # ... and themselves
        return torch.cat(toks, dim=1), torch.cat(pos, dim=1), allow
