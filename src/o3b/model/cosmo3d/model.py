"""CoSMo3D per-vertex feature extractor (o3b OD3D_Model adapter).

Wraps the official PointTransformerV3 backbone from JinLi998/CoSMo3D so it can
bake features into a mesh via ``mesh_type: mc32_vuni100_r256_fcosmo3d``.

The feature pipeline is a direct port of
``src/housecorr3dv2/encoder/cosmo3d/encoder.py::_run_pipeline`` in the
HouseCorr3Dv2 repo, so the o3b number is measured on exactly the same
descriptors as the ``eval_encoders.py`` number. Only the I/O boundary differs:
o3b hands an ``o3b.data.datatypes.mesh.Mesh`` where the encoder took a
``trimesh.Trimesh``.

CoSMo3D consumes the surface directly, so it is registered in ``_SELF_RENDERING``
(``o3b/data/datatypes/mesh.py``) — the ``vuni100`` multi-view rig is skipped and
``n_views`` / ``resolution`` in the mesh type are inert for this model.

forward(ObjectBatch) -> ObjectBatch with verts3d_feats (B, V, 64).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)

# Built backbone cached per (ckpt_path, device) across forward calls, so the
# checkpoint is loaded once per process rather than once per mesh.
_MODULE_CACHE: dict = {}

_COLOR_MEAN = 0.5
_COLOR_STD = 0.5


def _inject_flash_attn_stub() -> None:
    """Expose a PyTorch-native ``flash_attn`` shim when the real one is absent.

    CUDA 13+ cannot compile flash_attn yet, but PyTorch 2.0+ ships an equivalent
    through ``F.scaled_dot_product_attention``. PointTransformerV3 imports
    flash_attn unconditionally, so stub the two entry points it uses.
    """
    try:
        import flash_attn  # noqa: F401 — already installed, nothing to do

        return
    except ImportError:
        pass

    import types

    import torch.nn.functional as F

    def _varlen_qkvpacked(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kw,
    ):
        # qkv: (total_tokens, 3, nheads, head_dim)
        q, k, v = qkv.unbind(1)  # each (total, nh, hd)
        scale = softmax_scale or q.shape[-1] ** -0.5
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs, offset = [], 0
        for slen in seqlens:
            qi = q[offset : offset + slen].unsqueeze(0).transpose(1, 2)
            ki = k[offset : offset + slen].unsqueeze(0).transpose(1, 2)
            vi = v[offset : offset + slen].unsqueeze(0).transpose(1, 2)
            oi = F.scaled_dot_product_attention(
                qi, ki, vi, scale=scale, is_causal=causal
            )
            outs.append(oi.transpose(1, 2).squeeze(0))
            offset += slen
        return torch.cat(outs, dim=0)

    def _flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw
    ):
        scale = softmax_scale or q.shape[-1] ** -0.5
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=scale,
            is_causal=causal,
        )
        return o.transpose(1, 2)

    stub = types.ModuleType("flash_attn")
    stub.flash_attn_varlen_qkvpacked_func = _varlen_qkvpacked
    stub.flash_attn_func = _flash_attn_func

    iface = types.ModuleType("flash_attn.flash_attn_interface")
    iface.flash_attn_varlen_qkvpacked_func = _varlen_qkvpacked
    iface.flash_attn_func = _flash_attn_func

    sys.modules["flash_attn"] = stub
    sys.modules["flash_attn.flash_attn_interface"] = iface


@register_model("CoSMo3D")
class CoSMo3DModel(OD3D_Model):
    """Per-vertex CoSMo3D features (Li et al., CVPR 2026)."""

    def __init__(
        self,
        repo_path: str = "third_party/cosmo3d",
        ckpt_path: str = "checkpoints/cosmo3d/ours_final.pth",
        n_sample_pts: int = 5000,
        grid_size: float = 0.02,
        freeze: bool = True,
        use_texture: bool = False,
    ):
        super().__init__()
        # env overrides win, so the model also works from a CWD without the
        # symlinked repo/checkpoint tree.
        self.repo_path = os.environ.get("COSMO3D_ROOT", repo_path)
        self.ckpt_path = os.environ.get("COSMO3D_CKPT", ckpt_path)
        self.n_sample_pts = n_sample_pts
        self.grid_size = grid_size
        self.freeze = freeze
        # Colour is an ablation axis, not a fixed choice — see the `use_texture`
        # branch in _forward_object_batch. Flip it in configs/model/cosmo3d.yaml
        # to reproduce either number.
        self.use_texture = use_texture
        self.out_dim = 64

    def _build_backbone(self, device: torch.device):
        cache_key = (str(self.ckpt_path), str(device))
        cached = _MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        repo = Path(self.repo_path).resolve()
        if not repo.exists():
            raise RuntimeError(
                f"[CoSMo3D] repository not found at {repo}. Symlink or clone "
                "JinLi998/CoSMo3D there, or set COSMO3D_ROOT."
            )

        _inject_flash_attn_stub()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from release_module.network.canoncolor_bbox_pre import (
                PointSemSegWithDecoder,
            )
        except ImportError as exc:
            raise RuntimeError(
                "[CoSMo3D] cannot import the CoSMo3D model — spconv, torch-scatter, "
                "addict or timm may be missing."
            ) from exc

        ckpt_file = Path(self.ckpt_path).resolve()
        if not ckpt_file.exists():
            raise RuntimeError(f"[CoSMo3D] checkpoint not found: {ckpt_file}")

        model = PointSemSegWithDecoder(args=None)
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "CoSMo3D: loaded checkpoint (missing=%d, unexpected=%d)",
            len(missing),
            len(unexpected),
        )

        model = model.to(device).eval()
        _MODULE_CACHE[cache_key] = model
        return model

    @staticmethod
    def _sample_colored_surface(mesh, n_pts: int):
        """n_pts surface samples with texture RGB in [0,1] and face normals."""
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        try:
            pts, face_idx, colors = trimesh.sample.sample_surface(
                mesh, n_pts, sample_color=True
            )
        except Exception:
            pts, face_idx = trimesh.sample.sample_surface(mesh, n_pts)
            colors = None

        if colors is not None:
            rgb = np.asarray(colors, dtype=np.float32)[:, :3]
            if rgb.max() > 1.0:
                rgb /= 255.0
        else:
            try:
                vert_rgb = (
                    np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.float32)[
                        :, :3
                    ]
                    / 255.0
                )
                _, vidx = cKDTree(mesh.vertices).query(pts, k=1)
                rgb = vert_rgb[vidx]
            except Exception:
                # untextured mesh — mid-grey matches the encoder's flat fallback
                rgb = np.full((len(pts), 3), 0.5, dtype=np.float32)

        nrm = np.asarray(mesh.face_normals, dtype=np.float32)[face_idx]
        return np.asarray(pts, dtype=np.float32), rgb.astype(np.float32), nrm

    @torch.no_grad()
    def _forward_object_batch(self, batch):
        """Extract per-vertex CoSMo3D features for a batch sharing a single Mesh."""
        import gc

        import numpy as np
        from o3b.io import _mesh_to_trimesh
        from scipy.spatial import cKDTree

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tri = _mesh_to_trimesh(mesh)

        pts, rgb, nrm = self._sample_colored_surface(tri, self.n_sample_pts)

        # Coordinate-frame conversion: (x,y,z) → (−x, z, y)  [upstream app/segment/_data.py]
        coord = np.stack([-pts[:, 0], pts[:, 2], pts[:, 1]], axis=1).astype(np.float32)

        # CenterShift(apply_z=True): XY bbox-center, floor Z to 0. No scaling — model is metric.
        lo, hi = coord.min(0), coord.max(0)
        coord -= np.array(
            [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], dtype=np.float32
        )

        # GridSample: non-negative grid coords, one representative point per voxel
        grid_coord = np.floor(coord / self.grid_size).astype(np.int32)
        grid_coord -= grid_coord.min(0)
        _, keep = np.unique(grid_coord, axis=0, return_index=True)
        pts, coord, grid_coord = pts[keep], coord[keep], grid_coord[keep]
        rgb, nrm = rgb[keep], nrm[keep]

        # CenterShift(apply_z=False): re-center XY after voxel sampling
        lo, hi = coord.min(0), coord.max(0)
        coord -= np.array(
            [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0], dtype=np.float32
        )

        if self.use_texture:
            color_norm = (rgb - _COLOR_MEAN) / _COLOR_STD  # ≡ upstream rgb255/127.5 − 1
        else:
            # Neutral grey — bit-identical to the rgb=0.5 fallback this model
            # already uses for untextured meshes, so it is a supported input
            # mode rather than an out-of-distribution one. Two instances of a
            # category rarely share a paint job, so texture pushes the
            # descriptor toward appearance and away from cross-instance
            # correspondence; `housecorr3dv2`'s encoder holds grey for the same
            # reason. Measured both ways — see the README table.
            color_norm = np.zeros_like(nrm)
        feat = np.concatenate([color_norm, nrm], axis=1).astype(np.float32)  # (M, 6)

        data = {
            "coord": torch.from_numpy(coord).to(device),
            "feat": torch.from_numpy(feat).to(device),
            "grid_coord": torch.from_numpy(grid_coord).to(device),
            "offset": torch.tensor([len(coord)], dtype=torch.long, device=device),
        }

        model = self._build_backbone(device)
        # PTv3 point features, NOT the distillation head. `backbone(data)` alone
        # returns the 768-D head, which is a pointwise MLP of these 64-D features
        # trained to sit in SigLIP text-embedding space (see the upstream
        # `forward(x, return_pts_feat)` in model/backbone/pt3/model.py). That
        # reshaping serves text queries we never issue, and costs -0.05 PCK@0.1
        # under point-to-point NN matching. `housecorr3dv2`'s encoder defaults to
        # the backbone for the same reason; this keeps o3b consistent with it.
        _distill_feat, backbone_feat = model.backbone(data, return_pts_feat=True)

        feat_np = backbone_feat.float().cpu().numpy()  # (M, 64)
        feat_np -= feat_np.mean(axis=0)  # remove per-mesh common mode
        feat_np /= np.linalg.norm(feat_np, axis=1, keepdims=True) + 1e-8

        # scatter the voxel-representative features back onto the mesh vertices
        verts = mesh.verts.detach().cpu().float().numpy()
        _, nn_idx = cKDTree(pts).query(verts, k=1)
        feats = torch.from_numpy(feat_np[nn_idx]).float()  # (V, 64)

        del data, backbone_feat, _distill_feat
        torch.cuda.empty_cache()
        gc.collect()

        V = feats.shape[0]
        B = batch.verts3d.shape[0] if batch.verts3d is not None else 1
        batch.verts3d_feats = feats.unsqueeze(0).expand(B, V, -1).contiguous()
        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch

        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)
        raise NotImplementedError(
            "CoSMo3D only supports ObjectBatch (mesh) inputs; it has no 2-D frame path."
        )
