import copy
import dataclasses
import logging

logger = logging.getLogger(__name__)
from typing import Any, Optional
from omegaconf import DictConfig
from o3b.model.model import OD3D_Model, register_model
import torch

MERGE_INPUT_OR_FIRST_NOT_NONE = "input_or_first_not_none"
MERGE_ADD = "add"
MERGE_CONCAT = "concat"
MERGE_TYPES = (MERGE_INPUT_OR_FIRST_NOT_NONE, MERGE_ADD, MERGE_CONCAT)

# modalities whose channel count defines `out_dim`
_FEAT_MODALITIES = (
    "feat",
    "featmap",
    "featmap_lvls",
    "pts3d_feats",
    "verts3d_feats",
    "src_pts3d_feats",
    "src_verts3d_feats",
    "trgt_pts3d_feats",
    "trgt_verts3d_feats",
)


def _copy(obj):
    """Shallow copy so each branch may re-assign fields without affecting others."""
    return copy.copy(obj) if obj is not None else None


def _modalities(obj) -> list[str]:
    """Field names of a batch dataclass / SimpleNamespace, including attributes
    set after construction."""
    if obj is None:
        return []
    names = []
    if dataclasses.is_dataclass(obj):
        names += [f.name for f in dataclasses.fields(obj)]
    names += [k for k in getattr(obj, "__dict__", {}) if k not in names]
    return names


def _merge_add(values: list, modality: str):
    out = values[0]
    for v in values[1:]:
        out = out + v
    return out


def _merge_concat(values: list, modality: str, dim: int):
    if isinstance(values[0], (list, tuple)):
        lengths = {len(v) for v in values}
        if len(lengths) != 1:
            raise ValueError(
                f"ParallelModel: cannot concat modality '{modality}', "
                f"branches returned lists of different lengths {sorted(lengths)}."
            )
        return [
            torch.cat([v[l] for v in values], dim=dim) for l in range(len(values[0]))
        ]
    return torch.cat(values, dim=dim)


@register_model("ParallelModel")
class ParallelModel(OD3D_Model):
    """Runs its models on copies of the same input and merges their outputs.

    Each branch receives its own shallow copy of ``frames_gt`` / ``frames_pred``,
    so branches never observe each other's writes. Afterwards the outputs are
    merged modality by modality (one dataclass field / attribute at a time).

    A modality counts as *produced* by a branch when the branch's value is not
    ``None`` and is not the very object that was passed in; untouched
    pass-through modalities (``rgb``, ``pts3d``, ...) are therefore taken from
    the input instead of being added or concatenated B times.

    Merge types:
      * ``input_or_first_not_none`` -- the input value if it is not None,
        otherwise the first branch (in config order) with a non-None value.
      * ``add``    -- element-wise sum of the produced values.
      * ``concat`` -- ``torch.cat`` of the produced values along ``concat_dim``
        (level-wise for lists such as ``featmap_lvls``).

    ``merge_type`` sets the default, ``merge_types`` overrides it per modality.
    Likewise ``concat_dim`` sets the default concat axis and ``concat_dims``
    overrides it per modality -- needed for channel-first modalities such as
    ``featmap`` (B, F, H, W), where the feature axis is 1 rather than -1.
    """

    def __init__(
        self,
        models: torch.nn.ModuleList,
        merge_type: str = MERGE_INPUT_OR_FIRST_NOT_NONE,
        merge_types: Optional[dict] = None,
        concat_dim: int = -1,
        concat_dims: Optional[dict] = None,
    ):
        super().__init__()
        self.models: torch.nn.ModuleList = models
        self.merge_type: str = self._check_merge_type(merge_type, "merge_type")
        self.merge_types: dict[str, str] = {
            k: self._check_merge_type(v, f"merge_types.{k}")
            for k, v in dict(merge_types or {}).items()
        }
        self.concat_dim: int = int(concat_dim)
        self.concat_dims: dict[str, int] = {
            k: int(v) for k, v in dict(concat_dims or {}).items()
        }

    @staticmethod
    def _check_merge_type(merge_type: str, where: str) -> str:
        if merge_type not in MERGE_TYPES:
            raise ValueError(
                f"ParallelModel: unknown {where}='{merge_type}'. "
                f"Available: {list(MERGE_TYPES)}."
            )
        return merge_type

    def get_merge_type(self, modality: str) -> str:
        return self.merge_types.get(modality, self.merge_type)

    def get_concat_dim(self, modality: str) -> int:
        return self.concat_dims.get(modality, self.concat_dim)

    @property
    def out_dim(self):
        dims = [
            d
            for d in (getattr(m, "out_dim", None) for m in self.models if m is not None)
            if d is not None
        ]
        if not dims:
            return None
        if any(self.get_merge_type(k) == MERGE_CONCAT for k in _FEAT_MODALITIES):
            return sum(dims)
        return dims[0]

    def forward(self, frames_gt, frames_pred=None):
        outs_gt, outs_pred = [], []
        for model in self.models:
            if model is None:
                logger.warning(f"Model is None. {self.models}")
                continue
            out = model.forward(frames_gt=_copy(frames_gt), frames_pred=_copy(frames_pred))
            if isinstance(out, (tuple, list)) and len(out) == 2:
                out_gt, out_pred = out
            else:
                # models such as Diff3F return the batch itself on the mesh path
                out_gt, out_pred = out, frames_pred
            outs_gt.append(out_gt)
            outs_pred.append(out_pred)
        return (
            self._merge(frames_gt, outs_gt),
            self._merge(frames_pred, outs_pred),
        )

    def _merge(self, frames_in, outs: list):
        outs = [o for o in outs if o is not None]
        if not outs:
            return frames_in
        merged = _copy(frames_in) if frames_in is not None else _copy(outs[0])

        modalities = _modalities(frames_in)
        for out in outs:
            modalities += [k for k in _modalities(out) if k not in modalities]

        for modality in modalities:
            val_in = getattr(frames_in, modality, None) if frames_in is not None else None
            produced = [
                v
                for v in (getattr(out, modality, None) for out in outs)
                if v is not None and v is not val_in
            ]
            setattr(merged, modality, self._merge_modality(modality, val_in, produced))
        return merged

    def _merge_modality(self, modality: str, val_in, produced: list):
        if not produced:
            return val_in

        merge_type = self.get_merge_type(modality)
        if merge_type == MERGE_INPUT_OR_FIRST_NOT_NONE:
            return val_in if val_in is not None else produced[0]

        if len(produced) == 1:
            # nothing to combine: the single branch already saw the input
            return produced[0]

        mergeable = all(
            isinstance(v, torch.Tensor)
            or (
                isinstance(v, (list, tuple))
                and all(isinstance(e, torch.Tensor) for e in v)
            )
            for v in produced
        )
        if not mergeable:
            logger.warning(
                f"ParallelModel: modality '{modality}' is not tensor-valued "
                f"({[type(v).__name__ for v in produced]}), falling back to "
                f"'{MERGE_INPUT_OR_FIRST_NOT_NONE}' instead of '{merge_type}'."
            )
            return val_in if val_in is not None else produced[0]

        try:
            if merge_type == MERGE_ADD:
                return _merge_add(produced, modality)
            return _merge_concat(produced, modality, self.get_concat_dim(modality))
        except RuntimeError as e:
            shapes = [
                tuple(v.shape) if isinstance(v, torch.Tensor) else [tuple(e.shape) for e in v]
                for v in produced
            ]
            raise RuntimeError(
                f"ParallelModel: '{merge_type}' failed for modality '{modality}' "
                f"with branch shapes {shapes} "
                f"(concat_dim={self.get_concat_dim(modality)}): {e}"
            ) from e

    def get_down_sample_rate(self):
        rates = [
            m.get_down_sample_rate() for m in self.models if m is not None
        ]
        if not rates:
            logger.warning(f"All models are None. {self.models}")
            return 1.0
        if len(set(rates)) > 1:
            logger.warning(
                f"ParallelModel: branches disagree on down sample rate {rates}, "
                f"using {rates[0]}."
            )
        return rates[0]

    @classmethod
    def create_from_config(cls, config: DictConfig):
        from o3b.model.model import build_model

        models = torch.nn.ModuleList([build_model(m) for m in config.models])
        merge_types = config.get("merge_types", None)
        if merge_types is not None:
            merge_types = dict(merge_types)
        concat_dims = config.get("concat_dims", None)
        if concat_dims is not None:
            concat_dims = dict(concat_dims)
        return ParallelModel(
            models=models,
            merge_type=config.get("merge_type", MERGE_INPUT_OR_FIRST_NOT_NONE),
            merge_types=merge_types,
            concat_dim=config.get("concat_dim", -1),
            concat_dims=concat_dims,
        )
