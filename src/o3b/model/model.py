from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY_MODELS: dict[str, type["OD3D_Model"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "MLP":             "o3b.model.mlp.model",
    "CoordMLP":        "o3b.model.coordmlp.model",
    "PoseMLP":         "o3b.model.posemlp.model",
    "Feat2Pose":       "o3b.model.feat2pose.model",
    "DINOv2":          "o3b.model.dino.model",
    "ViT":             "o3b.model.vit.model",
    "SequentialModel": "o3b.model.sequential.model",
    "Frames2FeatPCL":  "o3b.model.frames2featpcl.model",
    "SAM3":            "o3b.model.sam3.model",
    "VGGT":            "o3b.model.vggt.model",
    "VGGTCameraHead":  "o3b.model.vggt_camera_head.model",
    "PointNet2":       "o3b.model.pointnet2.model",
    "LitePT":          "o3b.model.litept.model",
    "Diff3F":          "o3b.model.diff3f.model",
    "DenseMatcher":    "o3b.model.densematcher.model",
    "PartField":       "o3b.model.partfield.model",
    "CoSMo3D":         "o3b.model.cosmo3d.model",
    "GenPose2":        "o3b.model.obj_pose.genpose2.model",
    "ResNetHead":      "o3b.model.resnet.model",
}


def _ensure_model_imported(name: str) -> None:
    if name not in _REGISTRY_MODELS and name in _CLASS_TO_MODULE:
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_model(name: str):
    """Class decorator: @register_model("DINOv2")"""
    def decorator(cls):
        _REGISTRY_MODELS[name] = cls
        return cls
    return decorator


def build_model(cfg: DictConfig) -> "OD3D_Model":
    name = cfg.class_name
    _ensure_model_imported(name)
    if name not in _REGISTRY_MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(_REGISTRY_MODELS)}"
        )
    return _REGISTRY_MODELS[name].create_from_config(cfg)


# ── Base class ────────────────────────────────────────────────────────────────

class OD3D_Model(nn.Module):

    def __init__(self, **kwargs):
        self.down_sample_rate = 1
        super().__init__()

    def forward(self, frames_gt, frames_pred=None):
        return frames_gt, frames_pred

    def get_down_sample_rate(self):
        return 1.0

    @classmethod
    def create_by_name(cls, name: str, config: Optional[dict] = None) -> "OD3D_Model":
        import yaml
        configs_dir = Path(__file__).parent.parent.parent / "configs" / "model"
        config_path = configs_dir / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}\n"
                f"Available: {[p.stem for p in configs_dir.glob('*.yaml')]}"
            )
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        if config:
            cfg.update(config)
        _ensure_model_imported(cfg.get("class_name", ""))
        return cls.create_from_config(OmegaConf.create(cfg))

    @classmethod
    def create_from_config_or_name(cls, config: dict) -> "OD3D_Model":
        """Like ``create_from_config``, but if ``config`` has no ``class_name``
        and instead has ``model_name``, load ``configs/model/<model_name>.yaml``
        via ``create_by_name`` and apply the remaining keys as overrides.
        """
        config = dict(config)
        model_name = config.pop("model_name", None)
        if model_name is not None:
            return cls.create_by_name(model_name, config=config)
        return cls.create_from_config(OmegaConf.create(config))

    @classmethod
    def create_from_config(cls, config: DictConfig) -> "OD3D_Model":
        name = config.class_name
        _ensure_model_imported(name)
        if name not in _REGISTRY_MODELS:
            raise KeyError(
                f"Unknown model '{name}'. Registered: {sorted(_REGISTRY_MODELS)}"
            )
        model_cls = _REGISTRY_MODELS[name]
        keys = inspect.getfullargspec(model_cls.__init__)[0][1:]
        return model_cls(
            **{k: config.get(k) for k in keys if config.get(k, None) is not None}
        )
