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


def record_to_item(record: dict, item_cls):
    """Reconstruct a dataclass item of type ``item_cls`` from a stored record."""
    kwargs = {k: _decode(v) for k, v in record.items()}
    return item_cls(**kwargs)


# ── record iteration (optionally parallel) ─────────────────────────────────────

class _RecordDataset(torch.utils.data.Dataset):
    """Adapter: index → encoded record (or None), so loading *and* encoding run
    inside DataLoader worker processes."""

    def __init__(self, load_fn, n: int):
        self._load_fn = load_fn
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        item = self._load_fn(i)
        return item_to_record(item) if item is not None else None


def _passthrough(x):
    return x


def iter_records(load_fn, n: int, num_workers: int = 0):
    """Yield the encoded record for ``load_fn(i)`` for i in 0..n-1, yielding
    None where the item is None (filtered / unavailable).

    With ``num_workers > 0`` the items are loaded and serialised in torch
    DataLoader worker processes while preserving index order.  Records cross
    the process boundary as raw bytes (not tensors), so torch's shared-memory
    (/dev/shm) tensor transport is never involved.
    """
    if num_workers <= 0:
        for i in range(n):
            item = load_fn(i)
            yield item_to_record(item) if item is not None else None
        return
    loader = torch.utils.data.DataLoader(
        _RecordDataset(load_fn, n),
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


def build_sharded_dataset_from_generator(gen_fn, writer_batch_size: int = 1000):
    """Build a HuggingFace Dataset by streaming records from a generator.

    Processes ``writer_batch_size`` records at a time so peak memory is
    bounded to that many items rather than the full dataset.  ``gen_fn``
    is a zero-argument callable that returns an iterator of record dicts.
    """
    from datasets import Dataset as HFDataset
    return HFDataset.from_generator(gen_fn, num_proc=1, writer_batch_size=writer_batch_size)


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
