"""HuggingFace-sharding helpers for ConfigurableDataset.

A "sharded" dataset is a materialised cache of the items a dataset would
otherwise build lazily from raw files.  Each item (FrameObject, Object,
ObjectPair, …) is serialised into a flat, Arrow-friendly record, stored via
``datasets.Dataset.save_to_disk`` (sharded), and read back with
``load_from_disk``.

Serialisation is generic and recursive:
  - ``torch.Tensor``        → {"__t__": 3, "shape", "dtype", "data": bytes, …} (see below)
  - dataclass instance      → {"__dc__": "ClassName", "fields": {name: encoded, ...}}
  - list / tuple            → {"__l__": [encoded, ...]}
  - str / int / float / None → stored as-is

Tensor codec (v3; v1 list-of-floats and v2 raw-bytes records still decode):
  - bool tensors are bit-packed (``np.packbits``, 8× smaller)
  - selected float32 fields are quantised to integers (``_QUANT_FIELDS``):
    rgb / texture / vert_colors → uint8, depth → uint16 millimetres; skipped
    automatically when the value range doesn't fit the scheme
  - the resulting bytes are zstd-compressed (via pyarrow, a hard dependency of
    ``datasets``), falling back to raw bytes when compression doesn't help

Mesh deduplication: FrameObject records repeat the same object mesh in every
frame, which dominates shard size.  ``strip_mesh_from_record`` moves each
encoded mesh out of the stream (first occurrence per object_id wins) and
``write_mesh_sidecar`` stores them once in ``<shards>/meshes``;
ConfigurableDataset re-attaches them on read via ``read_mesh_sidecar`` /
``decode_sidecar_mesh``.

This keeps nested structures (e.g. an Object's Mesh) fully self-contained in
the shards.  ``datasets`` is imported lazily so the dependency is only required
when sharding is actually used.
"""
from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data


# ── dataclass registry (for reconstructing nested values) ──────────────────────

def _dataclass_registry() -> dict:
    from o3b.data.modalities import FrameObject, SceneObject, Object, ObjectPair
    from o3b.data.datatypes.mesh import Mesh
    return {c.__name__: c for c in (FrameObject, SceneObject, Object, ObjectPair, Mesh)}


# ── (de)serialisation ──────────────────────────────────────────────────────────

# Float32 tensors that can safely be stored in integer form (field name → scheme):
#   u8  — colour-like values: [0, 1] range → uint8 with scale 255,
#         [0, 255] range → uint8 with scale 1 (thresholds tolerate small
#         bicubic-interpolation overshoot, which gets clipped)
#   u16 — non-negative metric depth in metres → uint16 millimetres (max 65.535 m)
# Quantisation is skipped (raw float bytes kept) whenever the value range does
# not fit the scheme.
_QUANT_FIELDS = {
    "rgb":         "u8",
    "texture":     "u8",
    "vert_colors": "u8",
    "depth":       "u16",
    # normalised mask distance transform in [0, 1] → uint16 at 1/1000 resolution
    # (~0.5 px at 512²), instead of 1 MB of float32 per item
    "fo_mask_dt":  "u16",
    "fo_mask_amodal_dt": "u16",
}
_U16_SCALE  = 1000.0   # metres → millimetres
_ZSTD_LEVEL = 3

_zstd_codec = None


def _zstd():
    global _zstd_codec
    if _zstd_codec is None:
        import pyarrow as pa
        _zstd_codec = pa.Codec("zstd", compression_level=_ZSTD_LEVEL)
    return _zstd_codec


def _drop_zstd_cache() -> None:
    """Forget the cached codec.

    ``Dataset.from_generator`` fingerprints the generator by deep-pickling the
    functions it reaches and *their* module globals, which lands on this one —
    and ``pyarrow.lib.Codec`` cannot be pickled ("self.wrapped cannot be
    converted to a Python object").  It is only a cache, so anything that
    encodes before handing a generator to ``datasets`` clears it again.
    """
    global _zstd_codec
    _zstd_codec = None


def _encode_tensor(value: torch.Tensor, field: str | None) -> dict:
    t   = value.detach().cpu().contiguous()
    arr = t.numpy()
    q, scale, pack = "", 1.0, ""

    quant = _QUANT_FIELDS.get(field or "")
    if arr.dtype == np.bool_:
        pack = "bits"
        arr  = np.packbits(arr.reshape(-1))
    elif (quant == "u8" and arr.dtype == np.float32
          and arr.size and float(arr.min()) >= -0.5):
        vmax = float(arr.max())
        if vmax <= 1.5:
            q, scale = "u8", 255.0
        elif vmax <= 383.0:
            q, scale = "u8", 1.0
        if q:
            arr = np.clip(np.rint(arr * scale), 0, 255).astype(np.uint8)
    elif (quant == "u16" and arr.dtype == np.float32
          and arr.size and float(arr.min()) >= 0.0
          and float(arr.max()) * _U16_SCALE <= 65535.0):
        q, scale = "u16", _U16_SCALE
        arr = np.clip(np.rint(arr * scale), 0, 65535).astype(np.uint16)

    raw  = arr.tobytes()
    data = _zstd().compress(raw, asbytes=True)
    z    = "zstd"
    if len(data) >= len(raw):            # incompressible → keep raw bytes
        data, z = raw, ""
    return {
        "__t__":  3,
        "shape":  list(t.shape),
        "dtype":  str(t.dtype).replace("torch.", ""),
        "data":   data,
        "q":      q,        # quantisation scheme ("" = none)
        "scale":  scale,    # decode: float32(stored) / scale
        "pack":   pack,     # "bits" = np.packbits'ed bool tensor
        "z":      z,        # compression codec ("" = raw)
        "nbytes": len(raw), # pre-compression size (needed for zstd decompress)
    }


def _decode_tensor_v3(value: dict) -> torch.Tensor:
    shape = list(value["shape"])
    raw   = value["data"]
    if value.get("z") == "zstd":
        raw = _zstd().decompress(raw, decompressed_size=int(value["nbytes"]), asbytes=True)
    if value.get("pack") == "bits":
        n   = int(np.prod(shape)) if shape else 1
        arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=n).astype(np.bool_)
    elif value.get("q"):
        src = np.uint8 if value["q"] == "u8" else np.uint16
        arr = np.frombuffer(raw, dtype=src).astype(np.float32)
        if value["scale"] != 1.0:
            arr /= np.float32(value["scale"])
    else:
        arr = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy()
    return torch.from_numpy(arr.reshape(shape))


def _encode(value, field: str | None = None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _encode_tensor(value, field)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dc__":  type(value).__name__,
            "fields":  {f.name: _encode(getattr(value, f.name), f.name) for f in fields(value)},
        }
    if isinstance(value, (list, tuple)):
        return {"__l__": [_encode(v, field) for v in value]}
    return value


def _decode(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "__t__" in value:
            dtype_str = value["dtype"]
            shape     = value["shape"]
            data      = value["data"]
            if value["__t__"] == 3:
                return _decode_tensor_v3(value)
            elif value["__t__"] == 2 and isinstance(data, (bytes, bytearray, memoryview)):
                # v2: raw bytes → numpy → torch (fast path)
                arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).copy()
                return torch.from_numpy(arr.reshape(shape))
            else:
                # v1 legacy: Python list (slow path, kept for backward compatibility)
                torch_dtype = getattr(torch, dtype_str)
                if len(data) == 0:
                    return torch.zeros(shape, dtype=torch_dtype)
                return torch.tensor(data, dtype=torch_dtype).reshape(shape)
        if "__dc__" in value:
            cls    = _dataclass_registry()[value["__dc__"]]
            kwargs = {k: _decode(v) for k, v in value["fields"].items()}
            return cls(**kwargs)
        if "__l__" in value:
            return [_decode(v) for v in value["__l__"]]
    return value


def item_to_record(item) -> dict:
    """Serialise a dataclass item into a flat Arrow-friendly record dict."""
    return {f.name: _encode(getattr(item, f.name), f.name) for f in fields(item)}


# ── explicit Arrow schema ─────────────────────────────────────────────────────
#
# Without one, ``Dataset.from_generator`` infers the schema from the first
# writer batch: a modality that is None throughout those records freezes its
# column to Arrow ``null``, and the first later record that does carry a value
# kills the build with "Couldn't cast array of type struct<__t__: …> to null".
# (Seen on the toy_train shard build, whose first keypoint-annotated frame sits
# at row 166 with a writer batch of 100.)  Declaring the composite columns up
# front removes the dependency on what the first batch happens to contain.

def _tensor_feature():
    from datasets import Value
    return {
        "__t__":  Value("int64"),
        "shape":  [Value("int64")],
        "dtype":  Value("string"),
        "data":   Value("binary"),
        "q":      Value("string"),
        "scale":  Value("float64"),
        "pack":   Value("string"),
        "z":      Value("string"),
        "nbytes": Value("int64"),
    }


_SCALAR_ARROW_TYPE = {str: "string", bool: "bool", int: "int64", float: "float64", bytes: "binary"}


def _strip_optional(tp):
    """``Optional[X]`` / ``X | None`` → ``X``; other unions → None (untypable)."""
    import types
    import typing
    if typing.get_origin(tp) in (typing.Union, getattr(types, "UnionType", None)):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else None
    return tp


def _hints(cls) -> dict:
    import typing
    return typing.get_type_hints(cls)


class _Untypable(Exception):
    """A record holds something the schema deriver cannot name a type for."""


def _scalar_feature(probe):
    """Type a leaf from a sample value — None gives Arrow ``null``, as inference would."""
    from datasets import Value
    if probe is None:
        return Value("null")
    if type(probe) in _SCALAR_ARROW_TYPE:
        return Value(_SCALAR_ARROW_TYPE[type(probe)])
    raise _Untypable(f"no Arrow type for {type(probe).__name__}")


def _feature(tp, probe, seen: tuple = ()):
    """Feature for one encoded field.

    The struct shape comes from the annotation *tp* (that is the point: it holds
    even where *probe* has None), the leaves from the sample value *probe* —
    annotations are not reliable down there, e.g. ``Object.category`` is declared
    ``int`` while the HouseCorr3D loaders store the category name as a string.
    """
    import typing
    from datasets import Value

    tp = _strip_optional(tp)
    if isinstance(tp, type) and issubclass(tp, torch.Tensor):
        return _tensor_feature()
    if typing.get_origin(tp) in (list, tuple):
        args   = typing.get_args(tp)
        items  = probe.get("__l__") if isinstance(probe, dict) else None
        return {"__l__": [_feature(args[0] if args else None,
                                   items[0] if items else None, seen)]}
    if is_dataclass(tp) and isinstance(tp, type) and tp not in seen:
        try:
            hints = _hints(tp)
        except Exception:
            hints = {}
        sub = probe.get("fields") if isinstance(probe, dict) else None
        return {
            "__dc__": Value("string"),
            "fields": {f.name: _feature(hints.get(f.name), (sub or {}).get(f.name), seen + (tp,))
                       for f in fields(tp)},
        }
    return _scalar_feature(probe)


def record_features(item_cls, probe: dict):
    """Arrow schema for the records of ``item_cls``, or None if underivable.

    *probe* is one real record, used for the leaf types (see ``_feature``).  A
    field that is None in *probe* and carries no composite annotation stays
    Arrow ``null``, exactly as inference would have typed it.
    """
    from datasets import Features

    try:
        hints = _hints(item_cls)
        return Features({f.name: _feature(hints.get(f.name), probe.get(f.name))
                         for f in fields(item_cls)})
    except Exception as e:
        print(f"WARNING: could not derive an Arrow schema for {item_cls.__name__} "
              f"({type(e).__name__}: {e}); falling back to inference from the first batch.")
        return None


def record_to_item(record: dict, item_cls):
    """Reconstruct a dataclass item of type ``item_cls`` from a stored record."""
    kwargs = {k: _decode(v) for k, v in record.items()}
    return item_cls(**kwargs)


# ── record iteration (optionally parallel) ─────────────────────────────────────

_DROP_KEY = "__drop__"


def _drop(reason_fn, i: int) -> dict:
    """Marker yielded in place of a record when ``load_fn(i)`` produced no item.

    The reason is resolved in the same (worker) process that failed to load, so
    it can inspect the index row / files that caused the drop.
    """
    reason = "unknown"
    if reason_fn is not None:
        try:
            reason = reason_fn(i)
        except Exception as e:                      # diagnostics must never fail a build
            reason = f"unknown (reason lookup failed: {type(e).__name__})"
    return {_DROP_KEY: reason}


def drop_reason(record) -> str | None:
    """Return the drop reason if *record* is a drop marker, else None."""
    if record is None:
        return "unknown"
    if isinstance(record, dict) and _DROP_KEY in record and len(record) == 1:
        return record[_DROP_KEY]
    return None


class _RecordDataset(torch.utils.data.Dataset):
    """Adapter: index → encoded record (or drop marker), so loading *and*
    encoding run inside DataLoader worker processes."""

    def __init__(self, load_fn, n: int, reason_fn=None):
        self._load_fn = load_fn
        self._n = n
        self._reason_fn = reason_fn

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        item = self._load_fn(i)
        return item_to_record(item) if item is not None else _drop(self._reason_fn, i)


def _passthrough(x):
    return x


def iter_records(load_fn, n: int, num_workers: int = 0, reason_fn=None):
    """Yield the encoded record for ``load_fn(i)`` for i in 0..n-1.

    Where the item is None (filtered / unavailable) a drop marker
    ``{"__drop__": reason}`` is yielded instead — see ``drop_reason`` — so the
    caller can report *why* a build produced fewer items than it started with.
    ``reason_fn(i) -> str`` supplies that reason; without it drops are reported
    as ``"unknown"``.

    With ``num_workers > 0`` the items are loaded and serialised in torch
    DataLoader worker processes while preserving index order.  Records cross
    the process boundary as raw bytes (not tensors), so torch's shared-memory
    (/dev/shm) tensor transport is never involved.
    """
    if num_workers <= 0:
        for i in range(n):
            item = load_fn(i)
            yield item_to_record(item) if item is not None else _drop(reason_fn, i)
        return
    loader = torch.utils.data.DataLoader(
        _RecordDataset(load_fn, n, reason_fn),
        batch_size=None,           # one record at a time, no collation
        num_workers=num_workers,
        collate_fn=_passthrough,   # records are plain dicts/bytes; keep as-is
    )
    yield from loader


# ── disk I/O ────────────────────────────────────────────────────────────────────

_DEFAULT_SHARD_SIZE = 1048  # fallback items-per-shard when write_sharded_dataset gets none


def build_sharded_dataset(records: list[dict]):
    """Build an in-memory HuggingFace Dataset from a list of records."""
    from datasets import Dataset as HFDataset
    return HFDataset.from_list(records)


def build_sharded_dataset_from_generator(gen_fn, writer_batch_size: int = 1000,
                                         item_cls=None, probe: dict | None = None):
    """Build a HuggingFace Dataset by streaming records from a generator.

    Processes ``writer_batch_size`` records at a time so peak memory is
    bounded to that many items rather than the full dataset.  ``gen_fn``
    is a zero-argument callable that returns an iterator of record dicts.

    Given ``item_cls`` (the dataclass the records were built from) and ``probe``
    (one sample record), the Arrow schema is declared up front via
    ``record_features`` rather than inferred from the first batch — see the note
    there.  A schema that a real record does not fit is discarded in favour of
    inference, so a stale annotation degrades the cache instead of failing the
    build after hours of work.
    """
    from datasets import Dataset as HFDataset

    features = None
    if item_cls is not None and probe is not None:
        features = record_features(item_cls, probe)
        if features is not None:
            try:
                HFDataset.from_list([probe], features=features)
            except Exception as e:
                print(f"WARNING: derived Arrow schema rejects a real {item_cls.__name__} "
                      f"record ({type(e).__name__}: {e}); falling back to inference.")
                features = None

    # building the probe record encoded tensors, so the codec is live by now
    _drop_zstd_cache()
    return HFDataset.from_generator(
        gen_fn, num_proc=1, writer_batch_size=writer_batch_size, features=features,
    )


def _remove_dir(path: Path) -> None:
    """Remove a directory robustly, tolerating NFS ``.nfsXXXX`` leftovers.

    On NFS, deleting a file that is still held open (e.g. memory-mapped by a
    prior ``load_from_disk``, or a stale handle from an interrupted run) leaves
    a ``.nfsXXXX`` placeholder, so a plain ``rmtree`` fails the final ``rmdir``
    with ``OSError: [Errno 39] Directory not empty``.

    To avoid blocking on the live path, the directory is first *renamed* out of
    the way (an atomic metadata op that works even with open files), then the
    renamed copy is deleted best-effort; any ``.nfs`` leftovers there are
    harmless and get cleaned up once the holding process exits.
    """
    if not path.exists():
        return
    trash = path.with_name(f"{path.name}.trash-{os.getpid()}-{time.time_ns()}")
    try:
        os.rename(str(path), str(trash))
    except OSError:
        # rename failed (e.g. cross-device); fall back to in-place rmtree
        shutil.rmtree(str(path), ignore_errors=True)
        return
    shutil.rmtree(str(trash), ignore_errors=True)


def write_sharded_dataset(hf_dataset, path, shard_size: int | None = None) -> None:
    """Save a HuggingFace Dataset to ``path`` as Arrow shards (overwrites).

    ``shard_size`` is the number of items per on-disk shard file; it defaults
    to ``_DEFAULT_SHARD_SIZE`` when not given (e.g. callers that don't tie it
    to ``sharded_shard_size``).
    """
    path = Path(path)
    _remove_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, int(math.ceil(len(hf_dataset) / (shard_size or _DEFAULT_SHARD_SIZE))))
    hf_dataset.save_to_disk(str(path), num_shards=num_shards)


def read_sharded_dataset(path):
    """Load a sharded HuggingFace Dataset from ``path``."""
    from datasets import load_from_disk
    return load_from_disk(str(path))


# ── mesh deduplication sidecar ─────────────────────────────────────────────────

_MESH_SIDECAR_DIRNAME = "meshes"


def strip_mesh_from_record(record: dict, meshes: dict) -> dict:
    """Move a record's top-level encoded mesh into ``meshes`` (build path).

    The mesh is stashed under the record's ``object_id`` (first occurrence per
    object wins — frames of one object share the same mesh) and the record's
    ``mesh`` entry is set to None.  Records without a top-level object_id/mesh
    (e.g. pair items with nested src/trgt objects) pass through unchanged.
    """
    oid = record.get("object_id")
    if not isinstance(oid, str) or not oid or "mesh" not in record:
        return record
    if record["mesh"] is not None and oid not in meshes:
        meshes[oid] = record["mesh"]
    record = dict(record)
    record["mesh"] = None
    return record


def write_mesh_sidecar(meshes: dict, path) -> None:
    """Save {object_id: encoded mesh} as a single-shard HF dataset at
    ``<path>/meshes`` (overwrites)."""
    from datasets import Dataset as HFDataset
    side = Path(path) / _MESH_SIDECAR_DIRNAME
    _remove_dir(side)
    hf = HFDataset.from_list(
        [{"object_id": oid, "mesh": mesh} for oid, mesh in meshes.items()]
    )
    hf.save_to_disk(str(side), num_shards=1)


def read_mesh_sidecar(path):
    """Load the mesh sidecar at ``<path>/meshes``.

    Returns ``(hf_dataset, {object_id: row_index})``, or ``(None, None)`` when
    the sidecar doesn't exist (pre-dedup caches, pair items, meshless data).
    """
    side = Path(path) / _MESH_SIDECAR_DIRNAME
    if not side.exists():
        return None, None
    from datasets import load_from_disk
    ds = load_from_disk(str(side))
    return ds, {oid: i for i, oid in enumerate(ds["object_id"])}


def decode_sidecar_mesh(mesh_sidecar, row: int):
    """Decode the Mesh stored at ``row`` of a mesh sidecar dataset."""
    return _decode(mesh_sidecar[int(row)]["mesh"])
