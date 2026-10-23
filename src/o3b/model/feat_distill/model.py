from __future__ import annotations

import logging
from pathlib import Path

import torch

from o3b.model.meshfm.model import MeshFMModel
from o3b.model.model import register_model

logger = logging.getLogger(__name__)


def normalize_verts(verts: torch.Tensor) -> torch.Tensor:
    """Centre on the bbox, scale to the unit sphere: MeshFM's input normalisation,
    shared by training (train.py) and inference so the two see the same frame."""
    v = verts.detach().double()
    center = 0.5 * (v.min(0).values + v.max(0).values)
    scale = (v - center).norm(dim=1).max()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate mesh, all vertices coincide")
    return ((v - center) / scale).float()


@register_model("TriplaneDistill")
class TriplaneDistillModel(MeshFMModel):
    """A PartField-like per-vertex feature field, distilled to ``out_dim`` channels.

    MeshFM's architecture (o3b/model/meshfm), which is PartField's: surface
    point cloud -> PVCNN encoder -> triplane transformer -> features sampled
    from the three planes at the mesh vertices. The triplane head emits
    ``out_dim`` channels instead of 1536 and no tanh is applied, since the
    features regress distances (see train.py, which trains it to preserve a
    teacher feature's distances between vertex pairs of one instance).

    The trunk is initialised from MeshFM's checkpoint for training; inference
    loads ``ckpt_path``, a state dict written by train.py.
    """

    def __init__(self, ckpt_path: str = "checkpoints/feat_distill/model.pth", out_dim: int = 32,
                 init_ckpt_path: str = "checkpoints/meshfm/meshfm.pth",
                 init_gdrive_id: str = "1ONUMs3Ji_VlmyflO74No5rk5bAHM6qgc",
                 n_pc_points: int = 100000, seed: int | None = 0, freeze: bool = True, **kwargs):
        super().__init__(ckpt_path=init_ckpt_path, ckpt_gdrive_id=init_gdrive_id, n_pc_points=n_pc_points,
                         triplane_channels_high=out_dim, apply_tanh=False, seed=seed, freeze=freeze, **kwargs)
        self.distill_ckpt_path = ckpt_path
        self.out_dim = int(out_dim)

    def build_network(self, device: torch.device, init_from_meshfm: bool = True, init_state_path: str | None = None):
        """The network with the out_dim head. ``init_from_meshfm``: copy every
        MeshFM weight whose shape matches (all but the two head layers).
        ``init_state_path``: instead copy every matching tensor of that plain
        state dict, e.g. PartField's trunk (191 / 197 tensors match; its
        triplane MLP has one layer fewer, so mlp.2, mlp.4 and the upsampler
        stay new). ``net.init_loaded`` holds the names that were copied."""
        from o3b.model.meshfm.meshfm.model.simple_triplane_model import SimpleTriplaneModel

        class _Cfg(dict):
            __getattr__ = dict.get

        cfg = _Cfg(
            triplane_channels_low=self.triplane_channels_low,
            triplane_channels_high=self.out_dim,
            triplane_resolution=self.triplane_resolution,
            pvcnn=_Cfg(point_encoder_type="pvcnn", use_point_scatter=True,
                       z_triplane_channels=self.pvcnn_z_triplane_channels,
                       z_triplane_resolution=self.pvcnn_z_triplane_resolution,
                       unet_cfg=_Cfg(depth=3, enabled=True, rolled=True, use_3d_aware=True,
                                     start_hidden_channels=32, use_initial_conv=False)),
        )
        net = SimpleTriplaneModel(cfg=cfg, apply_tanh=False)
        net.init_loaded = set()
        if init_state_path or init_from_meshfm:
            src = Path(init_state_path) if init_state_path else self._ensure_ckpt()
            state = torch.load(src, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
            own = net.state_dict()
            keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            net.load_state_dict(keep, strict=False)
            net.init_loaded = set(keep)
            logger.info("TriplaneDistill: %d / %d tensors from %s; new: %s", len(keep), len(own), src.name,
                        sorted(set(own) - set(keep)))
        return net.to(device)

    def _build_model(self, device: torch.device):
        net = self.build_network(device, init_from_meshfm=False)
        path = Path(self.distill_ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"TriplaneDistill: no trained checkpoint at {path} (train.py writes it)")
        state = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"] if "model" in state else state)
        return net.eval()
