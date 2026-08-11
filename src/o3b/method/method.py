"""Method base class and registry.

A *method* is the thing under evaluation: it runs on each batch before the task
(e.g. a pose estimator that writes predicted poses).  This module holds only the
plumbing — ``o3b`` ships no concrete methods.  Downstream packages provide them
and make them discoverable in one of three ways:

1. **Entry points** (preferred).  Declare them in the downstream package's
   ``pyproject.toml``::

       [project.entry-points."o3b.methods"]
       MorpheusMethod = "housecorr3dv2.method.morpheus.method:MorpheusMethod"

   ``build_method("MorpheusMethod")`` then imports that module on demand, with no
   import of the downstream package anywhere in ``o3b``.  Entry points are baked
   into the install metadata, so a newly added one needs a re-install
   (``pip install -e .``) before it is visible.

2. **A ``method_module`` key in the method config**, naming the module that
   defines the class::

       method:
         class_name: MorpheusMethod
         method_module: housecorr3dv2.method.morpheus.method

   Handy for a method that is not installed as an entry point (a scratch
   experiment, a stale editable install).

3. **``register_method_module(name, module)``** at import time, for a package
   that is already imported by the time ``build_method`` runs.

In every case the class itself registers under its name via the
``@register_method`` decorator when its module is imported.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field


@dataclass
class MethodConfig:
    class_name: str
    extra: dict = field(default_factory=dict)
    # optional module that defines ``class_name``, see (2) above
    method_module: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "MethodConfig":
        d = dict(d)
        class_name = d.pop("class_name")
        method_module = d.pop("method_module", None)
        extra = d.pop("extra", {})
        extra.update(d)
        return cls(class_name=class_name, extra=extra, method_module=method_module)


# ── Method registry ────────────────────────────────────────────────────────────

_REGISTRY_METHODS: dict[str, type["Method"]] = {}

# name → module that defines it, filled by register_method_module() and by the
# ``o3b.methods`` entry points.  Empty by default: o3b has no methods of its own.
_CLASS_TO_MODULE: dict[str, str] = {}

_ENTRY_POINT_GROUP = "o3b.methods"
_entry_points_scanned = False


def register_method_module(name: str, module: str) -> None:
    """Declare that method *name* is defined in *module* (imported on demand)."""
    _CLASS_TO_MODULE[name] = module


def _scan_entry_points() -> None:
    """Fold the ``o3b.methods`` entry points into ``_CLASS_TO_MODULE`` (once).

    An entry point's value may be ``module:ClassName`` or a bare ``module``; only
    the module part is used, since importing it is what triggers the class's
    ``@register_method``.  Explicit ``register_method_module`` calls win.
    """
    global _entry_points_scanned
    if _entry_points_scanned:
        return
    _entry_points_scanned = True
    from importlib.metadata import entry_points
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        _CLASS_TO_MODULE.setdefault(ep.name, ep.value.split(":", 1)[0])


def _ensure_method_imported(name: str, module: str | None = None) -> None:
    if name in _REGISTRY_METHODS:
        return
    if module is None:
        _scan_entry_points()
        module = _CLASS_TO_MODULE.get(name)
    if module is not None:
        importlib.import_module(module)


def _unknown_method_error(name: str) -> KeyError:
    _scan_entry_points()
    return KeyError(
        f"Unknown method '{name}'. Registered: {sorted(_REGISTRY_METHODS)}. "
        f"Discoverable via the '{_ENTRY_POINT_GROUP}' entry points: "
        f"{sorted(_CLASS_TO_MODULE)}. A method added to a package's "
        f"entry points only becomes visible after re-installing it "
        f"(pip install -e .); alternatively name its module in the config's "
        f"'method_module' key."
    )


def register_method(name: str):
    """Class decorator: @register_method("DefaultMethod")"""
    def decorator(cls):
        _REGISTRY_METHODS[name] = cls
        return cls
    return decorator


def build_method(cfg: MethodConfig) -> "Method":
    _ensure_method_imported(cfg.class_name, cfg.method_module)
    if cfg.class_name not in _REGISTRY_METHODS:
        raise _unknown_method_error(cfg.class_name)
    # Under torchrun every rank of a node sees all of its GPUs, so the configs'
    # `device: cuda:0` has to become this rank's cuda:LOCAL_RANK — otherwise the
    # whole node's train/eval ranks pile onto GPU 0. Outside a torchrun launch
    # this returns the configured device unchanged.
    if "device" in cfg.extra:
        from o3b import ddp
        cfg.extra["device"] = ddp.resolve_device(cfg.extra["device"])
    return _REGISTRY_METHODS[cfg.class_name].create_from_config(cfg)


# ── Base class ─────────────────────────────────────────────────────────────────

class Method:
    def forward(self, batch, return_qualit: bool = False):
        raise NotImplementedError

    def __call__(self, batch, return_qualit: bool = False):
        return self.forward(batch, return_qualit=return_qualit)

    @classmethod
    def create_from_config(cls, cfg: MethodConfig) -> "Method":
        name = cfg.class_name
        _ensure_method_imported(name, cfg.method_module)
        if name not in _REGISTRY_METHODS:
            raise _unknown_method_error(name)
        method_cls = _REGISTRY_METHODS[name]
        spec = inspect.getfullargspec(method_cls.__init__)
        if spec.varkw is not None:
            return method_cls(**cfg.extra)
        keys = spec.args[1:]
        return method_cls(**{k: cfg.extra[k] for k in keys if k in cfg.extra})
