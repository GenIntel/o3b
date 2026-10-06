from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


@register_model("MeshFM")
class MeshFMModel(OD3D_Model):
    """Per-vertex feature extractor based on MeshFM (threedle/MeshFM).

    PartField's architecture (PVCNN point encoder → triplane transformer)
    trained to distil DINOv2 + SAM features into a 1536-d triplane field, with
    SO(3) augmentation. Encodes a surface-sampled point cloud, squashes the
    planes with tanh and queries them at the mesh vertices. Replicates the
    upstream ``MeshFM.extract_mesh_features(mode='vertex')``; weights are the
    released Objaverse checkpoint.

    forward(ObjectBatch) -> ObjectBatch with verts3d_feats (B, V, 1536).
    Frame batches are not supported (MeshFM is 3D-only).
    """

    def __init__(
        self,
        ckpt_path: str = "checkpoints/meshfm/meshfm.pth",
        ckpt_gdrive_id: str = "1ONUMs3Ji_VlmyflO74No5rk5bAHM6qgc",
        n_pc_points: int = 100000,
        n_sample_each: int = 200000,
        triplane_channels_low: int = 128,
        triplane_channels_high: int = 1536,
        triplane_resolution: int = 128,
        pvcnn_z_triplane_channels: int = 256,
        pvcnn_z_triplane_resolution: int = 128,
        apply_tanh: bool = True,
        seed: int | None = 0,
        freeze: bool = True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.ckpt_gdrive_id = ckpt_gdrive_id
        self.n_pc_points = n_pc_points
        self.n_sample_each = n_sample_each
        self.triplane_channels_low = triplane_channels_low
        self.triplane_channels_high = triplane_channels_high
        self.triplane_resolution = triplane_resolution
        self.pvcnn_z_triplane_channels = pvcnn_z_triplane_channels
        self.pvcnn_z_triplane_resolution = pvcnn_z_triplane_resolution
        self.apply_tanh = apply_tanh
        self.seed = seed
        self.freeze = freeze
        self.out_dim = triplane_channels_high  # 1536

    def _ensure_ckpt(self) -> Path:
        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            import gdown

            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("MeshFM: downloading checkpoint gdrive:%s → %s", self.ckpt_gdrive_id, ckpt_path)
            # per-process temp name, as for PartField: concurrent first-use
            # jobs would otherwise rename one shared .part file from under
            # each other
            tmp = ckpt_path.with_suffix(f".pth.part{os.getpid()}")
            gdown.download(id=self.ckpt_gdrive_id, output=str(tmp), quiet=True)
            tmp.replace(ckpt_path)
        return ckpt_path

    def _build_model(self, device: torch.device):
        from o3b.model.meshfm.meshfm.model.simple_triplane_model import SimpleTriplaneModel

        # upstream configs/meshfm_1536d.yaml; SimpleTriplaneModel reads it by
        # attribute, missing keys as None (upstream's DotDict)
        class _Cfg(dict):
            __getattr__ = dict.get

        cfg = _Cfg(
            triplane_channels_low=self.triplane_channels_low,
            triplane_channels_high=self.triplane_channels_high,
            triplane_resolution=self.triplane_resolution,
            pvcnn=_Cfg(
                point_encoder_type="pvcnn",
                use_point_scatter=True,
                z_triplane_channels=self.pvcnn_z_triplane_channels,
                z_triplane_resolution=self.pvcnn_z_triplane_resolution,
                unet_cfg=_Cfg(
                    depth=3,
                    enabled=True,
                    rolled=True,
                    use_3d_aware=True,
                    start_hidden_channels=32,
                    use_initial_conv=False,
                ),
            ),
        )
        model = SimpleTriplaneModel(cfg=cfg, apply_tanh=self.apply_tanh)
        state = torch.load(self._ensure_ckpt(), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model.eval().to(device)

    @torch.no_grad()
    def _forward_object_batch(self, batch):
        """Extract per-vertex MeshFM features for a batch sharing a single Mesh."""
        import gc

        from o3b.model.meshfm.meshfm.sampling import sample_surface_points

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        verts = mesh.verts.detach().cpu().double()
        faces = mesh.faces.detach().cpu().long()

        # centre on the bbox, scale to the unit sphere (upstream mesh_io.normalize_vertices)
        center = 0.5 * (verts.min(0).values + verts.max(0).values)
        scale = (verts - center).norm(dim=1).max()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("MeshFM: degenerate mesh, all vertices coincide")
        verts_normed = ((verts - center) / scale).float()

        generator = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else None
        pc = sample_surface_points(verts_normed, faces, self.n_pc_points, generator=generator)

        model = self._build_model(device)
        planes = model(pc.unsqueeze(0).to(device))  # (1, 3, C, H, W)

        # query the field at the mesh vertices, chunked to bound memory
        query = verts_normed.clamp(-1.0, 1.0).unsqueeze(0)
        feats = torch.cat(
            [
                model.sample_triplane_features_from_points(
                    planes, query[:, i : i + self.n_sample_each].to(device)
                ).cpu()
                for i in range(0, query.shape[1], self.n_sample_each)
            ],
            dim=1,
        )[0].float()  # (V, out_dim)

        del model, planes, pc, query
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
            "MeshFM only supports ObjectBatch (mesh) inputs; it has no 2-D frame path."
        )
