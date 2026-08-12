from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import yaml

import torch
from torch.utils.data import Dataset as _TorchDataset, DataLoader
from o3b.data.modalities import (
    FrameObject, SceneObject,
    FrameObjectBatch, SceneObjectBatch,
    ObjectPair,
    collate_frame_objects, collate_scene_objects,
)


# ── Item / batch type enums ───────────────────────────────────────────────────

class ItemType(str, Enum):
    OBJECT            = "object"
    OBJECT_PAIR       = "object_pair"
    FRAME_OBJECT      = "frame_object"
    FRAME_OBJECT_PAIR = "frame_object_pair"
    SCENE_OBJECT      = "scene_object"

class BatchType(str, Enum):
    OBJECT            = "object"
    OBJECT_PAIR       = "object_pair"
    FRAME_OBJECT      = "frame_object"
    FRAME_OBJECT_PAIR = "frame_object_pair"
    SCENE_OBJECT      = "scene_object"


# ── Dataset config ────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    # what and where
    class_name:      str                   # registered dataset name, e.g. "DenseMatcher"
    root:            Path                  = Path("data")
    path_raw:        Optional[Path]        = None   # raw (downloaded) data root
    path_preprocess: Optional[Path]        = None   # preprocessed data root
    split:           str                   = "train"

    # item / batch shape
    item_type:     ItemType                = ItemType.OBJECT
    batch_type:    BatchType               = BatchType.FRAME_OBJECT
    scene_length:  int                     = 8      # T; ignored for OBJECT / FRAME_OBJECT

    # mesh variant: "default" uses the raw mesh path; others load/create a
    # marching-cubes remesh at path_preprocess/mesh/<mesh_type>/<obj_id>.glb
    mesh_type:     str                     = "default"

    # object-frame variant: "raw" keeps the dataset's own object frame; any other
    # value names a per-object rigid/similarity transform stored under
    # path_preprocess/tform_obj/<tform_obj_type>/… that maps it into a canonical
    # frame (UCO3D's canonical-pose labels are the case this exists for).
    tform_obj_type: str                    = "raw"

    # which modalities to load (None = all available)
    modalities:        Optional[set[str]]  = None
    object_modalities: Optional[set[str]]  = None

    # filtering
    categories:        Optional[list[str]]   = None   # None = all
    subsets:           Optional[list[str]]   = None   # None = all; e.g. ["train"] or ["train","val"]
    filter_count_max:  Optional[int]         = None   # None = all; max number of samples to load
    filter_has_kpts:   bool                  = False
    # frame_object_pair: how many viewpoints (frames) to sample per object instance
    # when forming cross-instance pairs (1 = one representative frame per instance,
    # -1 = use all available frames of each instance)
    frame_pair_views_per_instance: int       = 1
    # how to combine the sampled viewpoints of two instances:
    #   "aligned" → index-aligned (view i of A with view i of B), ~n_views pairs
    #   "cross"   → full cross-product (every view of A with every view of B), ~n_views^2
    frame_pair_view_mode: str                = "aligned"
    filter_is_real:    Optional[bool]        = None   # None = all, True = real only, False = synthetic only
    filter_score_zero: bool                  = False  # OpenTT: drop clips where both scores == 0
    # dataset-wide rigid transform mapping the raw object frame → canonical (GL) object
    # frame, applied to every loaded object (4x4, R|t convention)
    obj_gl_tform4x4_obj_raw: Optional[list]   = None   # [[r00,r01,r02,tx],[...],[...],[0,0,0,1]]
    # camera-convention transform: left-multiplies cam_tform4x4_obj_raw (e.g. CV→OpenGL flip)
    cam_tform4x4_cam_raw:  Optional[list]    = None   # [[r00,...],[...],[...],[0,0,0,1]]

    # O3B_Transform applied to every loaded item (dict with class_name + kwargs)
    transform:             Optional[dict]    = None
    # O3B_Transform baked into the shards at build time (applied once per item
    # in the shard-build workers); items read back from the shards are already
    # transformed, so no per-access cost remains
    sharded_transform:     Optional[dict]    = None

    # HuggingFace sharding: when set, items are materialised once into
    # path_preprocess/sharded/<sharded_name> and loaded from there on subsequent
    # runs.  sharded_override=True rebuilds the shards even if they already exist.
    sharded_name:      Optional[str]        = None
    sharded_override:  bool                 = False
    sharded_shard_size: int                 = 1000
    # worker processes used to load+encode items while building the shards
    # (0 = build sequentially in the main process)
    sharded_num_workers: int                = 0

    # extra per-dataset kwargs passed through to the implementation
    extra: dict                            = field(default_factory=dict)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_yaml(self, path: Path) -> None:
        d = {
            "class_name":      self.class_name,
            "root":            str(self.root),
            "path_raw":        str(self.path_raw) if self.path_raw else None,
            "path_preprocess": str(self.path_preprocess) if self.path_preprocess else None,
            "split":           self.split,
            "item_type":       self.item_type.value,
            "batch_type":      self.batch_type.value,
            "scene_length":    self.scene_length,
            "mesh_type":       self.mesh_type,
            "tform_obj_type":  self.tform_obj_type,
            "modalities":        sorted(self.modalities)        if self.modalities        else None,
            "object_modalities": sorted(self.object_modalities) if self.object_modalities else None,
            "categories":        self.categories,
            "subsets":           self.subsets,
            "filter_count_max":  self.filter_count_max,
            "filter_has_kpts":    self.filter_has_kpts,
            "frame_pair_views_per_instance": self.frame_pair_views_per_instance,
            "frame_pair_view_mode":          self.frame_pair_view_mode,
            "filter_is_real":     self.filter_is_real,
            "filter_score_zero":  self.filter_score_zero,
            "obj_gl_tform4x4_obj_raw": self.obj_gl_tform4x4_obj_raw,
            "cam_tform4x4_cam_raw": self.cam_tform4x4_cam_raw,
            "transform":            self.transform,
            "sharded_transform":    self.sharded_transform,
            "sharded_name":       self.sharded_name,
            "sharded_override":   self.sharded_override,
            "sharded_shard_size": self.sharded_shard_size,
            "sharded_num_workers": self.sharded_num_workers,
            "extra":           self.extra,
        }
        path.write_text(yaml.safe_dump(d, sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            class_name      = d["class_name"],
            root            = Path(d["root"]) if d.get("root") else Path("data"),
            path_raw        = Path(d["path_raw"]) if d.get("path_raw") else None,
            path_preprocess = Path(d["path_preprocess"]) if d.get("path_preprocess") else None,
            split           = d.get("split", "train"),
            item_type       = ItemType(d.get("item_type", "object")),
            batch_type      = BatchType(d.get("batch_type", "frame_object")),
            scene_length    = d.get("scene_length", 8),
            mesh_type       = d.get("mesh_type", "default"),
            tform_obj_type  = d.get("tform_obj_type", "raw"),
            modalities        = set(d["modalities"])        if d.get("modalities")        else None,
            object_modalities = set(d["object_modalities"]) if d.get("object_modalities") else None,
            categories       = d.get("categories"),
            subsets          = d.get("subsets"),
            filter_count_max = d.get("filter_count_max") or d.get("max_samples"),
            filter_has_kpts   = bool(d.get("filter_has_kpts",   False)),
            frame_pair_views_per_instance = int(d.get("frame_pair_views_per_instance", 1)),
            frame_pair_view_mode = d.get("frame_pair_view_mode", "aligned"),
            filter_is_real    = None if "filter_is_real" not in d or d["filter_is_real"] is None
                                 else bool(d["filter_is_real"]),
            filter_score_zero = bool(d.get("filter_score_zero", False)),
            obj_gl_tform4x4_obj_raw = d.get("obj_gl_tform4x4_obj_raw", d.get("obj_tform4x4")),
            cam_tform4x4_cam_raw = d.get("cam_tform4x4_cam_raw"),
            transform            = d.get("transform"),
            sharded_transform    = d.get("sharded_transform"),
            sharded_name         = d.get("sharded_name"),
            sharded_override     = bool(d.get("sharded_override", False)),
            sharded_shard_size   = int(d.get("sharded_shard_size", 1000)),
            sharded_num_workers  = int(d.get("sharded_num_workers", 0)),
            extra           = d.get("extra", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path, overrides: "list[str] | None" = None) -> "DatasetConfig":
        return cls.from_dict(_load_yaml_with_defaults(Path(path), overrides=overrides))


def _load_yaml_with_defaults(
    path: Path, overrides: "list[str] | None" = None, resolve: bool = True
) -> dict:
    """Load a YAML config, resolving defaults and (unless ``resolve=False``) ${}."""
    from o3b.io import _load_yaml_with_defaults as _impl
    return _impl(path, overrides=overrides, resolve=resolve)


# ── Dataset registry ──────────────────────────────────────────────────────────

_REGISTRY_DATASETS: dict[str, type["ConfigurableDataset"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "HouseCorr3D":  "o3b.dataset.housecorr3d.dataset",
    "DenseMatcher": "o3b.dataset.densematcher.dataset",
    "OpenTT":       "o3b.dataset.opentt.dataset",
    "UCO3D":        "o3b.dataset.uco3d.dataset",
}


def _ensure_dataset_imported(name: str) -> None:
    if name not in _REGISTRY_DATASETS and name in _CLASS_TO_MODULE:
        import importlib
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_dataset(name: str):
    """Class decorator: @register_dataset("DenseMatcher")"""
    def decorator(cls):
        _REGISTRY_DATASETS[name] = cls
        return cls
    return decorator


def build_dataset(cfg: DatasetConfig) -> "ConfigurableDataset":
    _ensure_dataset_imported(cfg.class_name)
    if cfg.class_name not in _REGISTRY_DATASETS:
        raise KeyError(
            f"Unknown dataset '{cfg.class_name}'. "
            f"Registered: {sorted(_REGISTRY_DATASETS)}"
        )
    return _REGISTRY_DATASETS[cfg.class_name](cfg)


def build_dataset_from_config_or_name(config: dict) -> "ConfigurableDataset":
    """Like ``build_dataset``, but if ``config`` has ``dataset_name``, load
    ``configs/dataset/<dataset_name>.yaml`` (defaults chain) as the base and
    merge the remaining keys on top — mirroring
    ``OD3D_Model.create_from_config_or_name``.

    ``path_datasets_raw`` / ``path_datasets_preprocess`` keys in ``config``
    are applied as overrides while loading the named config, so its path
    interpolations resolve against them (the bench CLI injects the platform's
    values there).

    The named config is loaded *unresolved* and only resolved after ``config``
    is merged on top, so overriding a key that the config interpolates
    propagates: ``{"dataset_name": ..., "category": "bottle"}`` also updates a
    ``sharded_name: ..._c${category}`` defined in that config.
    """
    from omegaconf import OmegaConf
    config = dict(config)
    dataset_name = config.pop("dataset_name", None)
    if dataset_name is not None:
        from o3b.dataset.cli import _resolve_dataset_config
        overrides = [
            f"{key}={config.pop(key)}"
            for key in ("path_datasets_raw", "path_datasets_preprocess")
            if config.get(key) is not None
        ]
        base = _load_yaml_with_defaults(
            _resolve_dataset_config(dataset_name), overrides=overrides, resolve=False
        )
        config = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(config)),
            resolve=True,
        )
    return build_dataset(DatasetConfig.from_dict(config))


# ── Transform helper ─────────────────────────────────────────────────────────

def _build_transform(transform_cfg: Optional[dict]):
    """Instantiate an O3B_Transform from a config dict, or return None."""
    if transform_cfg is None:
        return None
    from omegaconf import OmegaConf
    from o3b.data.transforms.transform import O3B_Transform
    # import all known transform modules so subclasses register themselves
    _TRANSFORM_MODULES = [
        "o3b.data.transforms.frame_object.crop_cam_bbox2d",
    ]
    import importlib
    for mod in _TRANSFORM_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    cfg = OmegaConf.create(transform_cfg)
    return O3B_Transform.create_from_config(cfg)


# ── Base class ────────────────────────────────────────────────────────────────

class ConfigurableDataset(_TorchDataset):
    """
    Subclass this and implement _load_frame_object / _load_scene_object.
    Everything else — collation, DataLoader wiring — is handled here.
    """

    categories: tuple[str, ...] = ()  # override in subclasses with dataset-specific names

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._index: list = []
        self._sharded = None       # HuggingFace Dataset when sharding is active
        self._sharded_meshes = None      # mesh sidecar HF dataset (dedup'd meshes)
        self._sharded_mesh_rows = None   # {object_id: sidecar row index}
        self._transform = _build_transform(cfg.transform)
        self._sharded_transform = _build_transform(cfg.sharded_transform)
        self._setup()
        if cfg.sharded_name:
            self._setup_sharded()

    def _setup(self) -> None:
        pass

    # ── HuggingFace sharding ──────────────────────────────────────────────────

    _ITEM_TYPE_TO_CLS = {
        ItemType.OBJECT:       "Object",
        ItemType.OBJECT_PAIR:  "ObjectPair",
        ItemType.FRAME_OBJECT: "FrameObject",
        ItemType.SCENE_OBJECT: "SceneObject",
    }

    def _item_type_cls(self):
        from o3b.data.modalities import (  # noqa: F401
            FrameObject, SceneObject, Object, ObjectPair, FrameObjectPair,
        )
        return {
            ItemType.OBJECT: Object,
            ItemType.OBJECT_PAIR: ObjectPair,
            ItemType.FRAME_OBJECT: FrameObject,
            ItemType.FRAME_OBJECT_PAIR: FrameObjectPair,
            ItemType.SCENE_OBJECT: SceneObject,
        }[self.cfg.item_type]

    def _sharded_dir(self) -> Path:
        if self.cfg.path_preprocess is None:
            raise ValueError(
                "sharded_name is set but path_preprocess is None; "
                "cannot resolve the sharded dataset location."
            )
        return Path(self.cfg.path_preprocess) / "sharded" / self.cfg.sharded_name

    def _setup_sharded(self) -> None:
        """Load the sharded dataset, building it from raw items if necessary."""
        from o3b.dataset.sharding import (
            build_sharded_dataset_from_generator, drop_reason, iter_records,
            read_sharded_dataset, write_sharded_dataset,
            read_mesh_sidecar, strip_mesh_from_record, write_mesh_sidecar,
        )

        path = self._sharded_dir()
        if path.exists() and not self.cfg.sharded_override:
            print(f"Loading sharded dataset from {path}")
            self._sharded = read_sharded_dataset(path)
            self._sharded_meshes, self._sharded_mesh_rows = read_mesh_sidecar(path)
            return

        from tqdm import tqdm

        action = "Overriding" if path.exists() else "Building"
        n = len(self)
        num_workers = self.cfg.sharded_num_workers
        workers_note = f", {num_workers} workers" if num_workers > 0 else ""
        print(f"{action} sharded dataset at {path} ({n} items{workers_note})…")

        pbar = tqdm(total=n, desc="Sharding", unit="item")
        meshes: dict = {}   # object_id → encoded mesh, deduplicated across frames
        from collections import Counter
        drops: Counter = Counter()   # drop reason → count (why n_out < n)

        def _gen():
            for record in iter_records(self._load_sharded_item, n,
                                       num_workers=num_workers,
                                       reason_fn=self._sharded_drop_reason):
                pbar.update(1)
                reason = drop_reason(record)
                if reason is not None:
                    drops[reason] += 1
                    continue
                yield strip_mesh_from_record(record, meshes)

        hf = build_sharded_dataset_from_generator(
            _gen, writer_batch_size=self.cfg.sharded_shard_size,
            item_cls=self._item_type_cls(), probe=self._sharded_schema_probe(n),
        )
        pbar.close()
        if drops:
            print(f"Skipped {sum(drops.values())}/{n} items:")
            for reason, count in drops.most_common():
                print(f"  {count:>6}  {reason}")
        write_sharded_dataset(hf, path, shard_size=self.cfg.sharded_shard_size)
        if meshes:
            write_mesh_sidecar(meshes, path)
        self._sharded = read_sharded_dataset(path)
        self._sharded_meshes, self._sharded_mesh_rows = read_mesh_sidecar(path)
        print(f"Done. Wrote {len(self._sharded)} items → {path}")

    def _sharded_schema_probe(self, n: int, max_tries: int = 32):
        """One encoded record, used to type the shards' scalar columns.

        Taken from the first index that yields an item, outside the build stream
        so the progress bar and the drop tally stay honest.  Returns None when
        nothing loads, which leaves the schema to be inferred.

        It goes through ``iter_records`` rather than loading in-process because
        loading an item renders fo_mask_amodal: on a machine with a DISPLAY that
        binds pyrender to the pyglet/Xlib backend, and DataLoader workers forked
        afterwards then die with ``pyglet.gl.ContextException``.  A worker keeps
        the first GL context out of the parent (with sharded_num_workers: 0 the
        build renders in-process anyway, so there is nothing to protect).
        """
        from o3b.dataset.sharding import drop_reason, iter_records, strip_mesh_from_record

        for record in iter_records(self._load_sharded_item, min(n, max_tries),
                                   num_workers=min(1, self.cfg.sharded_num_workers)):
            if drop_reason(record) is None:
                # a throwaway mesh store: only the record's shape matters here
                return strip_mesh_from_record(record, {})
        return None

    # ── item loading dispatch ─────────────────────────────────────────────────

    def _load_sharded_item(self, idx: int):
        """Load a raw item with ``sharded_transform`` baked in (shard-build path)."""
        return self._apply_transform(self._load_item(idx), self._sharded_transform)

    def _sharded_drop_reason(self, idx: int) -> str:
        """Why ``_load_sharded_item(idx)`` yielded nothing — for the build tally.

        Called only for items that were dropped, so it may re-do work.  The
        generic answer is uninformative; subclasses override it to name the
        missing input (each loader overrides ``_sharded_drop_reason``).
        """
        if self._load_item(idx) is None:
            return "item could not be loaded"
        return f"{type(self._sharded_transform).__name__} returned None"

    def _apply_transform(self, item, transform):
        if transform is None or item is None:
            return item
        from o3b.data.datatypes.object import ObjectPair
        from o3b.data.datatypes.frame_object import FrameObjectPair
        if isinstance(item, (ObjectPair, FrameObjectPair)):
            # per-frame transforms (e.g. CropCamBBox2D) apply to each side
            from dataclasses import replace as _replace
            return _replace(
                item,
                src_object=transform(item.src_object),
                trgt_object=transform(item.trgt_object),
            )
        return transform(item)

    def _load_item(self, idx: int):
        if self.cfg.item_type == ItemType.OBJECT:
            return self._load_object(idx)
        elif self.cfg.item_type == ItemType.OBJECT_PAIR:
            return self._load_object_pair(idx)
        elif self.cfg.item_type == ItemType.FRAME_OBJECT:
            return self._load_frame_object(idx)
        elif self.cfg.item_type == ItemType.FRAME_OBJECT_PAIR:
            return self._load_frame_object_pair(idx)
        else:
            return self._load_scene_object(idx)

    # ── item loading (implement the one(s) matching your item_type) ──────────

    def _load_object(self, idx: int):
        raise NotImplementedError

    def _load_object_pair(self, idx: int) -> ObjectPair:
        raise NotImplementedError

    def _load_frame_object(self, idx: int) -> FrameObject:
        raise NotImplementedError

    def _load_frame_object_pair(self, idx: int) -> ObjectPair:
        raise NotImplementedError

    def _load_scene_object(self, idx: int) -> SceneObject:
        raise NotImplementedError

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self._sharded is not None:
            return len(self._sharded)
        raise NotImplementedError

    def __getitem__(self, idx: int):
        if self._sharded is not None:
            from o3b.dataset.sharding import record_to_item
            item = record_to_item(self._sharded[int(idx)], self._item_type_cls())
            self._attach_sharded_mesh(item)
        else:
            item = self._load_item(idx)
        return self._apply_transform(item, self._transform)

    def _attach_sharded_mesh(self, item) -> None:
        """Re-attach a mesh that was deduplicated into the sidecar at build time
        (see strip_mesh_from_record). Decodes fresh per access so items never
        share a Mesh instance."""
        if self._sharded_meshes is None or item is None:
            return
        if getattr(item, "mesh", None) is not None:
            return
        row = self._sharded_mesh_rows.get(getattr(item, "object_id", None))
        if row is None:
            return
        from o3b.dataset.sharding import decode_sidecar_mesh
        item.mesh = decode_sidecar_mesh(self._sharded_meshes, row)

    # ── Collation ─────────────────────────────────────────────────────────────

    def collate_fn(
        self, samples: list[Union[FrameObject, SceneObject]]
    ) -> Union[FrameObjectBatch, SceneObjectBatch]:
        if self.cfg.batch_type == BatchType.FRAME_OBJECT:
            return collate_frame_objects(samples, include=self.cfg.modalities)
        else:
            return collate_scene_objects(samples, include=self.cfg.modalities)

    # ── DataLoader factory ────────────────────────────────────────────────────

    def build_loader(self, batch_size: int = 8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    # ── Dataset-level CLI hooks (override in subclasses) ──────────────────────

    @classmethod
    def fetch(cls, cfg: "DatasetConfig", *, url: Optional[str] = None) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement fetch()")

    @classmethod
    def index(cls, cfg: "DatasetConfig", *, db: Optional[Path] = None, **kwargs) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement index()")

    @classmethod
    def init(cls, cfg: "DatasetConfig", *, limit: int = 0, override: bool = False, **_) -> None:
        """Instantiate the dataset without visualising it.

        Same construction path as `visualize`, so any one-off setup work the
        constructor performs — most notably building the sharded cache under
        <path_preprocess>/sharded/<sharded_name> — happens here.  With
        ``limit > 0`` the first N items are additionally loaded, which
        exercises the read path (and, for a sharded config, the cache that was
        just written).  ``override=True`` forces ``cfg.sharded_override``, so
        an existing sharded cache is rebuilt instead of loaded.
        """
        if override and not cfg.sharded_override:
            print("Overriding sharded_override=True (sharded cache will be rebuilt)")
            cfg.sharded_override = True
        dataset = cls(cfg)
        n = len(dataset)
        print(f"Initialised {cls.__name__} (item_type={ItemType(cfg.item_type).value}) — {n} items")
        if getattr(dataset, "_sharded", None) is not None:
            print(f"  sharded cache: {dataset._sharded_dir()}")

        if limit > 0 and n:
            from tqdm import tqdm
            for i in tqdm(range(min(limit, n)), desc="Loading", unit="item"):
                dataset[i]

    @classmethod
    def visualize(
        cls,
        cfg: "DatasetConfig",
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        debug: bool = False,
        **_,
    ) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement visualize()")
