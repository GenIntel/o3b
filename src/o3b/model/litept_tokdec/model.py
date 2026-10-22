"""LitePTTokenDecoder: LitePT-S* with learnable pose / surface tokens inserted into its decoder.

H02-T13 follow-up. The point cloud runs through LitePT-S* (the paper's variant
whose decoder mirrors the encoder: D3 two attention blocks, D2-D0 two sparse-
convolution blocks each). Learnable tokens ride along:

- 1 pose token and 32 surface tokens start at the bottleneck. The surface
  tokens start on a sphere around the observed cloud (median centre, quantile
  radius, Fibonacci directions).
- Stages and token counts: bottleneck E4 (32), D3 (128), D2 (512), D1 (2048),
  D0 (2048, read once more). Between stages every surface token is folded into
  4 children (shared fold MLP over a 2x2 grid seed): 32 -> 128 -> 512 -> 2048.

Per stage the surface tokens are **inserted into the stage's own voxels**:

1. unpool the real points with LitePT's GridUnpooling, untouched (tokens never
   pass through an unpooling, so the encoder's pooling maps stay valid);
2. voxelise every token at its current 3-D point on this stage's grid. A token
   on an occupied voxel is *hosted*: its feature is added (mean over the tokens
   there) to that voxel and it reads that voxel's output back. Tokens on empty
   voxels create new voxels (shared by the tokens that land together);
3. run the stage's blocks (an extra attention block at the bottleneck, LitePT's
   decoder blocks at D3-D0) on real + token voxels; a fresh sparse tensor and
   serialisation, so convolution rule tables and patches include the tokens;
4. pull the token features back out (residual into each token's own state) and
   write the real points' features back for the next unpooling;
5. per-stage heads: a point MLP refines each token's 3-D point and predicts its
   NOCS; the pose token (off the grid: one attention read over the stage's
   real voxels) feeds a per-stage PoseMLP.

Tokens keep their own width (``token_dim``) between stages; per stage a
projection maps them to the stage's width (504 / 252 / 144 / 72 / 72) and back.
A token-type embedding and the token's sub-voxel offset mark token voxels.

Outputs: ``pts3d_shape_coarse`` / ``nocs_shape_coarse`` = D2's 512 points,
``pts3d_shape_fine`` / ``nocs_shape_fine`` = D0's 2048, the other stages in
``pts3d_shape_aux`` / ``nocs_shape_aux`` (E4 32, D3 128, D1 2048), the final
pose (D0) via ``cam_tform4x4_obj`` / ``obj_size3d`` and the earlier stages'
poses in ``pose_aux`` = [(cam_tform4x4_obj, obj_size3d), ...] for deep
supervision (TrinityMethod). Coordinates as Frames2FeatPCL leaves them (the
normalised cloud frame), ``cam_tform4x4_pts3d_shape`` = its inverse transform.
"""

import logging
from functools import partial

import torch
import torch.nn as nn

from o3b.model.model import OD3D_Model, build_model, register_model

logger = logging.getLogger(__name__)

STAGES = ("e4", "dec3", "dec2", "dec1", "dec0")
STRIDE = {"e4": 16, "dec3": 8, "dec2": 4, "dec1": 2, "dec0": 1}


class _Bag:
    """Attribute bag reading unset fields as None (for the per-stage PoseMLPs)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, name):
        return None


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _fibonacci_sphere(n):
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = (1.0 - z * z).sqrt()
    phi = i * torch.pi * (3.0 - 5.0 ** 0.5)
    return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1).float()


def _keys(batch, grid):
    """(N,) int64 keys of (batch, x, y, z) voxel coordinates (each < 2^16)."""
    g = grid.long()
    return (batch.long() << 48) | (g[:, 0] << 32) | (g[:, 1] << 16) | g[:, 2]


@register_model("LitePTTokenDecoder")
class LitePTTokenDecoder(OD3D_Model):
    def __init__(
        self,
        in_channels: int = 384,
        grid_size: float = 0.005,
        grid_margin: int = 256,
        enc_channels=(36, 72, 144, 252, 504),
        enc_depths=(2, 2, 2, 6, 2),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        dec_channels=(72, 72, 144, 252),
        dec_depths=(2, 2, 2, 2),
        dec_num_head=(4, 4, 8, 14),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        drop_path: float = 0.3,
        bottleneck_blocks: int = 1,
        token_dim: int = 256,
        surface_tokens: int = 32,
        fold: int = 4,
        fold_grid_scale: float = 0.05,
        sphere_quantile: float = 0.9,
        cls_dim: int = 384,
        head_hidden: int = 256,
        nocs: bool = True,
        pose_hidden: int = 512,
        pose_heads: int = 4,
    ):
        super().__init__()
        from o3b.model.litept.litept import LitePT as LitePT_Orig, Block
        self.grid_size = grid_size
        self.grid_margin = grid_margin
        self.nocs = nocs
        self.fold = fold
        self.out_ch = 3 + (3 if nocs else 0)
        self.lit = LitePT_Orig(
            in_channels=in_channels, enc_depths=enc_depths, enc_channels=enc_channels, enc_num_head=enc_num_head,
            enc_conv=enc_conv, enc_attn=enc_attn,
            dec_depths=dec_depths, dec_channels=dec_channels, dec_num_head=dec_num_head, dec_conv=dec_conv,
            dec_attn=dec_attn, drop_path=drop_path, enc_mode=False,
        )
        order = self.lit.order if hasattr(self.lit, "order") else ("z", "z-trans", "hilbert", "hilbert-trans")
        self.order = order
        self.bottleneck = nn.ModuleList([
            Block(channels=enc_channels[-1], num_heads=enc_num_head[-1], patch_size=1024, mlp_ratio=4,
                  qkv_bias=True, drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
                  order_index=i % len(order), cpe_indice_key="tokdec_e4", enable_conv=False, enable_attn=True,
                  rope_freq=100.0)
            for i in range(bottleneck_blocks)])
        width = {"e4": enc_channels[-1], "dec3": dec_channels[3], "dec2": dec_channels[2],
                 "dec1": dec_channels[1], "dec0": dec_channels[0]}
        self.width = width
        self.n_tokens = {"e4": surface_tokens, "dec3": surface_tokens * fold, "dec2": surface_tokens * fold ** 2,
                         "dec1": surface_tokens * fold ** 3, "dec0": surface_tokens * fold ** 3}
        W = token_dim
        self.surface_init = nn.Parameter(torch.randn(1, surface_tokens, W) * 0.02)
        self.pose_init = nn.Parameter(torch.randn(1, W) * 0.02)
        self.register_buffer("sphere_dirs", _fibonacci_sphere(surface_tokens), persistent=False)
        self.sphere_quantile = sphere_quantile
        self.proj_in = nn.ModuleDict({s: nn.Linear(W, width[s]) for s in STAGES})
        self.offset_in = nn.ModuleDict({s: nn.Linear(3, width[s]) for s in STAGES})
        self.type_emb = nn.ParameterDict({s: nn.Parameter(torch.zeros(width[s])) for s in STAGES})
        self.proj_out = nn.ModuleDict({s: nn.Linear(width[s], W) for s in STAGES})
        self.norm_tok = nn.ModuleDict({s: nn.LayerNorm(W) for s in STAGES})
        self.pose_kv = nn.ModuleDict({s: nn.Linear(width[s], W) for s in STAGES})
        self.pose_attn = nn.ModuleDict({s: nn.MultiheadAttention(W, pose_heads, batch_first=True) for s in STAGES})
        self.point_head = nn.ModuleDict({s: _mlp([W + cls_dim, head_hidden, head_hidden, self.out_ch]) for s in STAGES})
        from omegaconf import OmegaConf
        self.pose_head = nn.ModuleDict({s: build_model(OmegaConf.create(dict(
            class_name="PoseMLP", in_dim=W + cls_dim, out_dims=[3, 6, 3], num_layers=1, hidden_dim=pose_hidden,
            dropout=0.5, activation="none", bias=True, rot6d_othant=[False] * 6))) for s in STAGES})
        side = int(round(fold ** 0.5))
        if side * side != fold:
            raise ValueError(f"LitePTTokenDecoder: fold must be a square, got {fold}")
        lin = torch.linspace(-fold_grid_scale, fold_grid_scale, side)
        self.register_buffer("fold_grid", torch.stack(torch.meshgrid(lin, lin, indexing="ij"), -1).reshape(-1, 2),
                             persistent=False)
        self.fold_seed = nn.ModuleDict({s: nn.Linear(2, W) for s in STAGES[:3]})
        self.fold_mlp = nn.ModuleDict({s: _mlp([W + 2, head_hidden, head_hidden, 3]) for s in STAGES[:3]})

    # ── token insertion ──────────────────────────────────────────────────────
    def _insert_run_extract(self, stage, point, blocks, h, p, origin):
        """Insert tokens (B, T, W) at points p (B, T, 3) into `point`, run `blocks`, return token outputs."""
        from o3b.model.litept.litept import Point
        B, T, W = h.shape
        dev = h.device
        cell = self.grid_size * STRIDE[stage]
        # A token's cell is bounded by the real voxels' box plus the margin the origin already
        # leaves below it: a point that drifts (or goes NaN) far outside the cloud would otherwise
        # blow up the sparse tensor's spatial shape and the serialisation depth.
        pf = torch.nan_to_num(p.reshape(-1, 3).float(), nan=0.0, posinf=0.0, neginf=0.0)
        g0 = torch.floor((pf.detach() - origin) / self.grid_size).clamp(min=0, max=(1 << 16) - 1).long()
        g_max = point.grid_coord.long().max(0).values + max(1, self.grid_margin // STRIDE[stage])
        g = torch.minimum(torch.div(g0, STRIDE[stage], rounding_mode="floor"), g_max)
        tb = torch.arange(B, device=dev).repeat_interleave(T)
        sub = ((pf - origin) / cell - g.float() - 0.5).clamp(-0.5, 0.5)                 # in [-0.5, 0.5]
        x = self.proj_in[stage](h.reshape(-1, W)) + self.offset_in[stage](sub) + self.type_emb[stage]
        x = x.to(point.feat.dtype)

        n_real = point.feat.shape[0]
        k_real = _keys(point.batch, point.grid_coord)
        k_sorted, k_order = torch.sort(k_real)
        k_tok = _keys(tb, g)
        pos = torch.searchsorted(k_sorted, k_tok).clamp(max=n_real - 1)
        hit = k_sorted[pos] == k_tok
        host = torch.where(hit, k_order[pos], torch.full_like(pos, -1))

        feat_real = point.feat
        if hit.any():
            add = torch.zeros_like(feat_real).index_add_(0, host[hit], x[hit])
            cnt = torch.zeros(n_real, device=dev, dtype=feat_real.dtype).index_add_(
                0, host[hit], torch.ones_like(host[hit], dtype=feat_real.dtype))
            feat_real = feat_real + add / cnt.clamp(min=1)[:, None]
        new = ~hit
        if new.any():
            k_new, inv_new = torch.unique(k_tok[new], return_inverse=True)
            n_new = k_new.shape[0]
            feat_new = torch.zeros(n_new, x.shape[1], device=dev, dtype=x.dtype).index_add_(0, inv_new, x[new])
            cnt_new = torch.zeros(n_new, device=dev, dtype=x.dtype).index_add_(
                0, inv_new, torch.ones_like(inv_new, dtype=x.dtype))
            feat_new = feat_new / cnt_new[:, None]
            b_new = (k_new >> 48)
            g_new = torch.stack([(k_new >> 32) & 0xFFFF, (k_new >> 16) & 0xFFFF, k_new & 0xFFFF], -1)
            c_new = origin + (g_new.float() + 0.5) * cell
        else:
            n_new = 0
            inv_new = torch.zeros(0, dtype=torch.long, device=dev)
            feat_new = feat_real.new_zeros(0, feat_real.shape[1])
            b_new = point.batch.new_zeros(0)
            g_new = point.grid_coord.new_zeros(0, 3)
            c_new = point.coord.new_zeros(0, 3)

        batch = torch.cat([point.batch.long(), b_new.long()])
        perm = torch.argsort(batch, stable=True)                       # LitePT needs batch-contiguous points
        aug = Point({
            "feat": torch.cat([feat_real, feat_new.to(feat_real.dtype)])[perm],
            "coord": torch.cat([point.coord.float(), c_new.float()])[perm],
            # int64: LitePT's PointROPE kernel reads grid_coord as Long (its pooled stages are int64)
            "grid_coord": torch.cat([point.grid_coord.long(), g_new.long()])[perm],
            "batch": batch[perm],
        })
        aug.serialization(order=self.order, shuffle_orders=self.training)
        aug.sparsify()
        for blk in blocks:
            aug = blk(aug)
        out = torch.empty_like(aug.feat)
        out[perm] = aug.feat
        point.feat = out[:n_real]
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        tok_out = torch.empty(B * T, out.shape[1], device=dev, dtype=out.dtype)
        tok_out[hit] = out[host[hit]]
        if n_new:
            tok_out[new] = out[n_real + inv_new]
        h = h + self.proj_out[stage](tok_out.float()).reshape(B, T, W)
        return self.norm_tok[stage](h), point

    def _pose_read(self, stage, point, hp):
        """Pose token update: one attention read over this stage's real voxels (per sample)."""
        from torch.nn.utils.rnn import pad_sequence
        B = hp.shape[0]
        counts = torch.bincount(point.batch.long(), minlength=B).tolist()
        kv = pad_sequence(torch.split(self.pose_kv[stage](point.feat.float()), counts), batch_first=True)
        mask = pad_sequence([torch.zeros(c, dtype=torch.bool, device=hp.device) for c in counts],
                            batch_first=True, padding_value=True)
        upd, _ = self.pose_attn[stage](hp[:, None], kv, kv, key_padding_mask=mask, need_weights=False)
        return hp + upd[:, 0]

    def _heads(self, stage, h, p, hp, cls, obj_tform):
        B, T, _ = h.shape
        o = self.point_head[stage](torch.cat([h, cls[:, None].expand(B, T, -1)], -1))
        p = p + o[..., :3]
        nocs = o[..., 3:6] if self.nocs else None
        bag = _Bag(feat=torch.cat([hp, cls], -1), obj_tform4x4_obj_nocs_0c=obj_tform)
        _, bag = self.pose_head[stage](None, bag)
        return p, nocs, (bag.cam_tform4x4_obj, bag.obj_size3d)

    def _fold(self, stage, h, p):
        B, T, W = h.shape
        K = self.fold
        seed = self.fold_grid.to(h.dtype)[None, None].expand(B, T, K, 2)
        hc = h[:, :, None].expand(B, T, K, W) + self.fold_seed[stage](seed)
        off = self.fold_mlp[stage](torch.cat([h[:, :, None].expand(B, T, K, W), seed], -1))
        pc = p[:, :, None] + off
        return hc.reshape(B, T * K, W), pc.reshape(B, T * K, 3)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, frames_gt, frames_pred=None):
        from o3b.model.litept.litept import Point
        pts, feats, cls = frames_pred.pts3d, frames_pred.feats, frames_pred.feat
        if cls is None:
            raise RuntimeError("LitePTTokenDecoder: frames_pred.feat (the DINOv2 cls feature) is unset")
        B, N, _ = pts.shape
        dev = pts.device
        coord = pts.reshape(-1, 3).float()
        origin = coord.min(0).values - self.grid_margin * self.grid_size
        grid0 = torch.floor((coord - origin) / self.grid_size).int()
        point = Point({"feat": feats.reshape(B * N, -1), "coord": coord, "grid_coord": grid0,
                       "grid_size": self.grid_size, "batch": torch.arange(B, device=dev).repeat_interleave(N)})
        if self.lit.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.training)
        point.sparsify()
        point = self.lit.embedding(point)
        point = self.lit.enc(point)

        # surface tokens on a sphere around the observed cloud
        c = pts.detach().float().median(dim=1).values
        r = (pts.detach().float() - c[:, None]).norm(dim=-1).quantile(self.sphere_quantile, dim=1)
        p = c[:, None] + r[:, None, None] * self.sphere_dirs[None]
        h = self.surface_init.expand(B, -1, -1)
        hp = self.pose_init.expand(B, -1)
        cls = cls.float()
        obj_tform = frames_pred.obj_tform4x4_obj_nocs_0c

        decs = dict(self.lit.dec.named_children())
        pts_out, nocs_out, poses = {}, {}, {}
        for i, stage in enumerate(STAGES):
            if stage == "e4":
                blocks = list(self.bottleneck)
            else:
                dec = decs[stage]
                point = dec.up(point)
                blocks = [m for n, m in dec.named_children() if n != "up"]
            h, point = self._insert_run_extract(stage, point, blocks, h, p, origin)
            hp = self._pose_read(stage, point, hp)
            p, nocs, pose = self._heads(stage, h, p, hp, cls, obj_tform)
            pts_out[stage], nocs_out[stage], poses[stage] = p, nocs, pose
            if stage in self.fold_mlp:
                h, p = self._fold(stage, h, p)

        frames_pred.pts3d_shape_coarse = pts_out["dec2"]
        frames_pred.pts3d_shape_fine = pts_out["dec0"]
        frames_pred.pts3d_shape_aux = [pts_out[s] for s in ("e4", "dec3", "dec1")]
        if self.nocs:
            frames_pred.nocs_shape_coarse = nocs_out["dec2"]
            frames_pred.nocs_shape_fine = nocs_out["dec0"]
            frames_pred.nocs_shape_aux = [nocs_out[s] for s in ("e4", "dec3", "dec1")]
        frames_pred.cam_tform4x4_pts3d_shape = obj_tform
        frames_pred.pose_aux = [poses[s] for s in ("e4", "dec3", "dec2", "dec1")]
        frames_pred.cam_tform4x4_obj, frames_pred.obj_size3d = poses["dec0"]
        frames_pred.feat = torch.cat([hp, cls], -1)
        return frames_gt, frames_pred
