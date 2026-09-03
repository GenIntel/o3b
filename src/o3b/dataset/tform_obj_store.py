"""SQLite storage for per-sequence ``tform_obj`` transforms.

od3d writes one ``tform_obj.pt`` per sequence, which for UCO3D means 114k files
holding 64 bytes each: 7 MiB of numbers spread over 454 MB and 230k inodes on
NFS.  Everything that touches them then pays a network round trip per sequence —
reading all of them takes ~43 s, copying the tree ~6 min, and the dataset walk's
one ``.exists()`` per sequence is what makes indexing over an sshfs mount slow.

The same data in one SQLite file is 10.3 MB and loads into a dict in 0.12 s
(measured on all 114,368), so the existence check becomes free and forking a
labelling round becomes a single-file copy.  Per-row updates keep the property
the directory layout had for free: a labelling tool can write one sequence
without rewriting everything, and SQLite's locking makes concurrent writers safe.

    <path_preprocess>/tform_obj/<tform_obj_type>.db      preferred
    <path_preprocess>/tform_obj/<tform_obj_type>/...     legacy, still read

Both are supported and the ``.db`` wins when present, so existing directories
keep working untouched.  ``o3b dataset tform-obj-to-db`` converts one.
"""
from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)

_TFORM_OBJ_PARENT = "tform_obj"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tform_obj (
    category TEXT NOT NULL,
    sequence TEXT NOT NULL,
    tform    BLOB NOT NULL,          -- 64 bytes: 4x4 float32, row-major, LE
    PRIMARY KEY (category, sequence)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def dir_path(path_preprocess, tform_obj_type: str) -> Path:
    """Legacy per-sequence directory for a tform_obj_type."""
    return Path(path_preprocess) / _TFORM_OBJ_PARENT / tform_obj_type


def db_path(path_preprocess, tform_obj_type: str) -> Path:
    """SQLite file for a tform_obj_type, alongside the directory of the same name."""
    return Path(path_preprocess) / _TFORM_OBJ_PARENT / f"{tform_obj_type}.db"


def key(category: str, sequence: str) -> str:
    return f"{category}/{sequence}"


# ── encoding ──────────────────────────────────────────────────────────────────

def encode(tform: torch.Tensor) -> bytes:
    """(4, 4) tensor → the 64-byte blob stored in the table."""
    t = tform.detach().to(torch.float32).cpu().contiguous()
    if tuple(t.shape) != (4, 4):
        raise ValueError(f"tform_obj must be 4x4, got {tuple(t.shape)}")
    return t.numpy().tobytes()


def decode(blob: bytes) -> torch.Tensor:
    """The 64-byte blob → a (4, 4) float32 tensor (its own storage)."""
    return torch.frombuffer(bytearray(blob), dtype=torch.float32).reshape(4, 4)


# ── reading ───────────────────────────────────────────────────────────────────

# The last few tables read, by (path, mtime, size).  A grid editor builds a
# dataset per navigation and up to `prefetch` more on its background thread, and
# each build read the whole table again — 13 MB over sshfs, seconds when the
# file is cold.  Local writes clear this (see _forget), and a write from
# elsewhere is picked up when the file's mtime or size changes, which is the
# same guarantee a fresh read has over a mount held with kernel_cache and
# attr_timeout: the kernel serves cached pages until the attributes change too.
_TABLES: dict = {}
_TABLES_MAX = 4


def _forget(path: Path) -> None:
    """Drop memoised copies of *path* — called by everything that writes it."""
    for k in [k for k in _TABLES if k[0] == str(path)]:
        _TABLES.pop(k, None)


def load_all(path_preprocess, tform_obj_type: str) -> Optional[dict[str, torch.Tensor]]:
    """Whole table as ``{"category/sequence": (4, 4)}``, or None when there is no db.

    Read eagerly into a plain dict rather than holding a connection: it is 0.12 s
    for the full UCO3D table, and a dict survives the fork into DataLoader
    workers, which a sqlite3 connection does not.

    The dict is memoised and shared between callers, so treat it as read-only —
    every reader here only looks things up in it.
    """
    path = db_path(path_preprocess, tform_obj_type)
    if not path.exists():
        return None
    try:
        st = path.stat()
        memo_key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        memo_key = None
    if memo_key is not None and memo_key in _TABLES:
        return _TABLES[memo_key]
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute("SELECT category, sequence, tform FROM tform_obj").fetchall()
    except sqlite3.DatabaseError as e:
        logger.warning(f"could not read {path}: {e} — falling back to the directory")
        return None
    finally:
        con.close()
    if not rows:
        # A tform_obj_type with zero transforms is not a dataset with no
        # sequences, it is a db that was created and never filled (an
        # interrupted conversion, say).  Letting it win over the directory
        # would silently empty the dataset, so treat it as absent.
        logger.warning(f"{path} has no rows — ignoring it and using the directory. "
                       f"Delete it, or rebuild with 'o3b dataset tform-obj-to-db "
                       f"--override'.")
        return None
    table = {key(c, s): decode(b) for c, s, b in rows}
    if memo_key is not None:
        if len(_TABLES) >= _TABLES_MAX:
            _TABLES.clear()
        _TABLES[memo_key] = table
    return table


def read_one(path_preprocess, tform_obj_type: str,
             category: str, sequence: str) -> Optional[torch.Tensor]:
    """One transform, for callers that do not want the whole table."""
    path = db_path(path_preprocess, tform_obj_type)
    if not path.exists():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        row = con.execute(
            "SELECT tform FROM tform_obj WHERE category = ? AND sequence = ?",
            (category, sequence),
        ).fetchone()
    finally:
        con.close()
    return decode(row[0]) if row else None


# ── writing ───────────────────────────────────────────────────────────────────

def _connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.executescript(_SCHEMA)
    return con


def write_many(path_preprocess, tform_obj_type: str,
               rows: Iterable[tuple[str, str, torch.Tensor]], *,
               meta: Optional[dict] = None) -> int:
    """Upsert ``(category, sequence, tform)`` rows. Returns the number written."""
    path = db_path(path_preprocess, tform_obj_type)
    _forget(path)
    con = _connect_rw(path)
    n = 0
    try:
        with con:
            for category, sequence, tform in rows:
                con.execute(
                    "INSERT INTO tform_obj (category, sequence, tform) VALUES (?, ?, ?) "
                    "ON CONFLICT(category, sequence) DO UPDATE SET tform = excluded.tform",
                    (category, sequence, encode(tform)),
                )
                n += 1
            for k, v in (meta or {}).items():
                con.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, str(v)),
                )
    finally:
        con.close()
    return n


def write_one(path_preprocess, tform_obj_type: str,
              category: str, sequence: str, tform: torch.Tensor) -> None:
    """Write a single sequence's transform — what a labelling tool calls."""
    write_many(path_preprocess, tform_obj_type, [(category, sequence, tform)])


def delete_one(path_preprocess, tform_obj_type: str, category: str, sequence: str) -> bool:
    """Drop a sequence, i.e. remove it from the dataset. True if a row went."""
    path = db_path(path_preprocess, tform_obj_type)
    if not path.exists():
        return False
    _forget(path)
    con = sqlite3.connect(path, timeout=60)
    try:
        with con:
            cur = con.execute(
                "DELETE FROM tform_obj WHERE category = ? AND sequence = ?",
                (category, sequence),
            )
        return cur.rowcount > 0
    finally:
        con.close()


def count(path_preprocess, tform_obj_type: str) -> Optional[int]:
    path = db_path(path_preprocess, tform_obj_type)
    if not path.exists():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        return con.execute("SELECT COUNT(*) FROM tform_obj").fetchone()[0]
    finally:
        con.close()


# ── conversion from the per-sequence directory ────────────────────────────────

def _read_pt_fast(path: Path) -> Optional[bytes]:
    """The 64-byte payload of a torch.save'd 4x4 float32, without importing torch.

    A torch.save file is a zip whose ``*/data/<key>`` member is the raw storage,
    so a 4x4 float32 is one 64-byte blob.  Reading it directly is what makes
    converting 114k files take ~43 s instead of minutes of unpickling — and it
    lets the conversion run anywhere python does.  Returns None when the file is
    not that shape, and the caller falls back to torch.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            blobs = [i for i in zf.infolist()
                     if "/data/" in i.filename and i.file_size == 64]
            if len(blobs) != 1:
                return None
            raw = zf.read(blobs[0])
    except Exception:
        return None
    if len(raw) != 64:
        return None
    vals = struct.unpack("<16f", raw)
    # bottom row of an affine 4x4 — cheap guard against a mis-shaped storage
    if vals[12:] != (0.0, 0.0, 0.0, 1.0):
        return None
    return raw


def _read_tform_file(path: Path) -> Optional[torch.Tensor]:
    raw = _read_pt_fast(path)
    if raw is not None:
        return decode(raw)
    try:
        t = torch.load(path, map_location="cpu", weights_only=True).float()
    except Exception as e:
        logger.warning(f"could not read {path}: {e}")
        return None
    return t if tuple(t.shape) == (4, 4) else None


def iter_dir(path_preprocess, tform_obj_type: str, pcl_sfm=("meta_mask", "meta"),
             workers: int = 32):
    """Yield ``(category, sequence, tform)`` from the per-sequence directory."""
    from concurrent.futures import ThreadPoolExecutor

    root = dir_path(path_preprocess, tform_obj_type) / pcl_sfm[0] / pcl_sfm[1]
    if not root.is_dir():
        return
    paths = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for seq_dir in sorted(cat_dir.iterdir()):
            f = seq_dir / "tform_obj.pt"
            paths.append((cat_dir.name, seq_dir.name, f))

    # latency-bound on NFS, so read them concurrently
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (cat, seq, _), tform in zip(
            paths, pool.map(lambda it: _read_tform_file(it[2]), paths, chunksize=64)
        ):
            if tform is not None:
                yield cat, seq, tform


def convert_dir_to_db(path_preprocess, tform_obj_type: str, *,
                      override: bool = False, workers: int = 32) -> tuple[int, Path]:
    """Build ``<tform_obj_type>.db`` from ``<tform_obj_type>/``. Returns (rows, path)."""
    from datetime import date

    out = db_path(path_preprocess, tform_obj_type)
    src = dir_path(path_preprocess, tform_obj_type)
    if not src.is_dir():
        raise FileNotFoundError(f"no per-sequence directory to convert: {src}")
    if out.exists() and not override:
        raise FileExistsError(f"{out} already exists (pass --override to rebuild)")
    if out.exists():
        out.unlink()

    n = write_many(
        path_preprocess, tform_obj_type, iter_dir(path_preprocess, tform_obj_type, workers=workers),
        meta={"source": str(src), "converted": date.today().isoformat(),
              "tform_obj_type": tform_obj_type},
    )
    return n, out
