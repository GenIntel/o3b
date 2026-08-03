from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY_TASKS: dict[str, type["OD3D_Task"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "ObjectTask":       "o3b.task.object.task",
    "Crsp3DNNTask":     "o3b.task.crsp3d_nn.task",
    "CamCrsp3DNNTask":  "o3b.task.cam_crsp3d_nn.task",
    "CamApp2D3D":       "o3b.task.cam_app2d3d.task",
    "CamGeo2D3D":       "o3b.task.cam_geo2d3d.task",
}


def _ensure_task_imported(name: str) -> None:
    if name not in _REGISTRY_TASKS and name in _CLASS_TO_MODULE:
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_task(name: str):
    """Class decorator: @register_task("ObjectTask")"""
    def decorator(cls):
        _REGISTRY_TASKS[name] = cls
        return cls
    return decorator


def build_task(cfg: DictConfig) -> "OD3D_Task":
    name = cfg.class_name
    _ensure_task_imported(name)
    if name not in _REGISTRY_TASKS:
        raise KeyError(
            f"Unknown task '{name}'. Registered: {sorted(_REGISTRY_TASKS)}"
        )
    return _REGISTRY_TASKS[name].create_from_config(cfg)


# ── Base class ────────────────────────────────────────────────────────────────

class OD3D_Task:

    def forward(self, batch, return_qualit: bool = True):
        raise NotImplementedError

    def __call__(self, batch, return_qualit: bool = True):
        return self.forward(batch, return_qualit=return_qualit)

    @classmethod
    def create_by_name(cls, name: str, config: Optional[dict] = None) -> "OD3D_Task":
        import yaml
        configs_dir = Path(__file__).parent.parent.parent / "configs" / "task"
        config_path = configs_dir / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Task config not found: {config_path}\n"
                f"Available: {[p.stem for p in configs_dir.glob('*.yaml')]}"
            )
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        if config:
            cfg.update(config)
        _ensure_task_imported(cfg.get("class_name", ""))
        return cls.create_from_config(OmegaConf.create(cfg))

    @classmethod
    def create_from_config(cls, config: DictConfig) -> "OD3D_Task":
        name = config.class_name
        _ensure_task_imported(name)
        if name not in _REGISTRY_TASKS:
            raise KeyError(
                f"Unknown task '{name}'. Registered: {sorted(_REGISTRY_TASKS)}"
            )
        task_cls = _REGISTRY_TASKS[name]
        keys = inspect.getfullargspec(task_cls.__init__)[0][1:]
        return task_cls(
            **{k: config.get(k) for k in keys if config.get(k, None) is not None}
        )
