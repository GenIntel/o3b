"""ShapeMLP: coarse-to-fine shape completion from the global feature.

The shape branch of Beyond 'Templates' (arXiv 2510.11687, §3.3 / App. 6.1),
without its input-point fusion: from the global feature an MLP regresses a
coarse cloud of ``coarse_count`` points, and every coarse point is folded into
``fold_count`` fine points by a shared MLP that reads the point, a fixed 2-D
grid seed and the global feature (PCN / PoinTr folding) — 512 x 4 = 2048.

It is a *side branch*: it reads ``frames_pred.feat`` and writes
``pts3d_shape_coarse`` / ``pts3d_shape_fine``, but leaves ``feat`` untouched,
so a chain ``... -> MLP -> ShapeMLP -> PoseMLP`` hands PoseMLP exactly the
input it had without this stage. It has to sit *before* PoseMLP, which
overwrites ``feat`` with its own output and clears ``obj_tform4x4_obj_nocs_0c``.

Points are predicted in the frame of ``frames_pred.pts3d``: with
Frames2FeatPCL's ``normalize_pts3d`` the camera-aligned, median-centred,
quantile-scaled cloud frame — the paper's camera frame with the per-sample
scale divided out. That frame -> camera is written as
``cam_tform4x4_pts3d_shape``, for the caller to bring GT points into it.
"""

import logging

import torch
import torch.nn as nn

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


def _mlp(dims, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@register_model("ShapeMLP")
class ShapeMLP(OD3D_Model):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        coarse_count: int = 512,
        fold_count: int = 4,
        fold_hidden_dim: int = 512,
        fold_grid_scale: float = 0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.coarse_count = coarse_count
        self.fold_count = fold_count

        self.coarse = _mlp([in_dim, hidden_dim, hidden_dim, coarse_count * 3])

        # fold_count 2-D seeds on a square grid (4 -> 2x2), centred on 0
        side = int(round(fold_count ** 0.5))
        if side * side != fold_count:
            raise ValueError(f"ShapeMLP: fold_count must be a square, got {fold_count}")
        lin = torch.linspace(-fold_grid_scale, fold_grid_scale, side)
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(-1, 2)
        self.register_buffer("fold_grid", grid, persistent=False)  # (fold_count, 2)

        # per fine point: global feature + its coarse point + its seed -> offset
        self.fold = _mlp([in_dim + 3 + 2, fold_hidden_dim, fold_hidden_dim, 3])

    def forward(self, frames_gt, frames_pred=None):
        feat = frames_pred.feat                                   # (B, F)
        B = feat.shape[0]
        C, K = self.coarse_count, self.fold_count

        coarse = self.coarse(feat).reshape(B, C, 3)               # (B, C, 3)

        pts = coarse[:, :, None].expand(B, C, K, 3)
        seed = self.fold_grid[None, None].expand(B, C, K, 2)
        glob = feat[:, None, None].expand(B, C, K, feat.shape[-1])
        offset = self.fold(torch.cat([glob, pts, seed], dim=-1))  # (B, C, K, 3)
        fine = (pts + offset).reshape(B, C * K, 3)

        frames_pred.pts3d_shape_coarse = coarse
        frames_pred.pts3d_shape_fine = fine
        # the points' frame -> camera; PoseMLP clears obj_tform4x4_obj_nocs_0c
        # after reading it, so keep it under a name of our own. None = the
        # cloud was not normalised and the points are in camera space already.
        frames_pred.cam_tform4x4_pts3d_shape = frames_pred.obj_tform4x4_obj_nocs_0c
        return frames_gt, frames_pred
