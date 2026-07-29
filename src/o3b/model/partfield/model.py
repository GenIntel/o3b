from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)

# Built encoder modules cached per (ckpt_path, device) across forward calls, so
# the 1.2 GB checkpoint is loaded once per process rather than once per mesh.
_MODULE_CACHE: dict = {}


@register_model("PartField")
class PartFieldModel(OD3D_Model):
    """Per-vertex feature extractor based on PartField (nv-tlabs/PartField).

    Encodes a surface-sampled point cloud with a PVCNN encoder into a triplane
    feature field, refines it with a triplane transformer, and queries the
    field at the mesh vertices. Replicates the upstream ``predict_step``
    (``model_trainer_pvcnn_only_demo.py``) without the PyTorch Lightning
    dependency; weights are loaded from the released Objaverse checkpoint.

    forward(ObjectBatch) -> ObjectBatch with verts3d_feats (B, V, 448).
    Frame batches are not supported (PartField is 3D-only).
    """

    def __init__(
        self,
        ckpt_path: str = "checkpoints/partfield/model_objaverse.ckpt",
        ckpt_url: str = "https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt",
        n_pc_points: int = 100000,
        normalize_extent: float = 0.9,
        n_sample_each: int = 10000,
        triplane_channels_low: int = 128,
        triplane_channels_high: int = 512,
        sdf_channels: int = 64,
        pvcnn_z_triplane_channels: int = 256,
        pvcnn_z_triplane_resolution: int = 128,
        freeze: bool = True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.ckpt_url = ckpt_url
        self.n_pc_points = n_pc_points
        self.normalize_extent = normalize_extent
        self.n_sample_each = n_sample_each
        self.triplane_channels_low = triplane_channels_low
        self.triplane_channels_high = triplane_channels_high
        self.sdf_channels = sdf_channels
        self.pvcnn_z_triplane_channels = pvcnn_z_triplane_channels
        self.pvcnn_z_triplane_resolution = pvcnn_z_triplane_resolution
        self.freeze = freeze
        self.out_dim = triplane_channels_high - sdf_channels  # 448

    def _ensure_ckpt(self) -> Path:
        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("PartField: downloading checkpoint %s → %s", self.ckpt_url, ckpt_path)

            def _reporthook(count, block_size, total_size):
                if total_size > 0 and count % 500 == 0:
                    pct = min(100, count * block_size * 100 // total_size)
                    logger.info("PartField: download %d%%", pct)

            tmp = ckpt_path.with_suffix(".ckpt.part")
            urllib.request.urlretrieve(self.ckpt_url, tmp, reporthook=_reporthook)
            tmp.rename(ckpt_path)
        return ckpt_path

    def _build_modules(self, device: torch.device):
        # Cache the built (pvcnn, triplane_transformer) per (ckpt, device) so we
        # don't reload the 1.2 GB checkpoint on every mesh during generation.
        cache_key = (str(self.ckpt_path), str(device))
        cached = _MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        from omegaconf import OmegaConf

        from o3b.model.partfield.partfield.PVCNN.encoder_pc import TriPlanePC2Encoder
        from o3b.model.partfield.partfield.triplane import TriplaneTransformer

        # matches upstream configs/final/demo.yaml + partfield/config/defaults.py
        pvcnn_cfg = OmegaConf.create(
            {
                "point_encoder_type": "pvcnn",
                "use_point_scatter": True,
                "z_triplane_channels": self.pvcnn_z_triplane_channels,
                "z_triplane_resolution": self.pvcnn_z_triplane_resolution,
                "unet_cfg": {
                    "depth": 3,
                    "enabled": True,
                    "rolled": True,
                    "use_3d_aware": True,
                    "start_hidden_channels": 32,
                    "use_initial_conv": False,
                },
            }
        )
        pvcnn = TriPlanePC2Encoder(
            pvcnn_cfg, device=device, shape_min=-1, shape_length=2, use_2d_feat=False
        )
        triplane_transformer = TriplaneTransformer(
            input_dim=self.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=self.triplane_channels_high,
        )

        ckpt = torch.load(self._ensure_ckpt(), map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        pvcnn.load_state_dict(
            {k[len("pvcnn."):]: v for k, v in state_dict.items() if k.startswith("pvcnn.")},
            strict=True,
        )
        triplane_transformer.load_state_dict(
            {
                k[len("triplane_transformer."):]: v
                for k, v in state_dict.items()
                if k.startswith("triplane_transformer.")
            },
            strict=True,
        )
        pvcnn.eval().to(device)
        triplane_transformer.eval().to(device)
        _MODULE_CACHE[cache_key] = (pvcnn, triplane_transformer)
        return pvcnn, triplane_transformer

    @torch.no_grad()
    def _forward_object_batch(self, batch):
        """Extract per-vertex PartField features for a batch sharing a single Mesh."""
        import gc

        import numpy as np
        import trimesh

        from o3b.model.partfield.partfield.PVCNN.encoder_pc import sample_triplane_feat

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        verts = mesh.verts.detach().cpu().float().numpy()
        faces = mesh.faces.detach().cpu().long().numpy()

        # normalize longest bbox axis to [-normalize_extent, normalize_extent]
        # (upstream partfield/dataloader.py)
        bbmin, bbmax = verts.min(0), verts.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * self.normalize_extent / (bbmax - bbmin).max()
        verts_normed = (verts - center) * scale

        tri = trimesh.Trimesh(vertices=verts_normed, faces=faces, process=False)
        pc, _ = trimesh.sample.sample_surface(tri, self.n_pc_points)
        pc = torch.tensor(np.asarray(pc), dtype=torch.float32, device=device).unsqueeze(0)

        pvcnn, triplane_transformer = self._build_modules(device)

        planes = triplane_transformer(pvcnn(pc, pc))
        _, part_planes = torch.split(
            planes, [self.sdf_channels, planes.shape[2] - self.sdf_channels], dim=2
        )

        # query the triplane field at the mesh vertices, chunked to avoid OOM
        query = torch.tensor(verts_normed, dtype=torch.float32, device=device).unsqueeze(0)
        feats = torch.cat(
            [
                sample_triplane_feat(part_planes, query[:, i : i + self.n_sample_each])
                for i in range(0, query.shape[1], self.n_sample_each)
            ],
            dim=1,
        )[0].cpu().float()  # (V, out_dim)

        del pvcnn, triplane_transformer, planes, part_planes, pc, query
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
            "PartField only supports ObjectBatch (mesh) inputs; it has no 2-D frame path."
        )
