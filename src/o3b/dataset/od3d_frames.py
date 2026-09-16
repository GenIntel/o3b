"""Shared walk / index machinery for the datasets migrated from od3d.

PASCAL3D, ImageNet3D, HANDAL and Objectron were all preprocessed by od3d into
the same shape of tree, so they share one walk, one ``frames.db`` schema and one
``index()``.  Only the depth of the meta tree differs:

    flat       meta/frames/<split>/<category>/<name>.yaml            PASCAL3D, ImageNet3D
    sequenced  meta/frames/<split>/<category>/<sequence>/<frame>.yaml  HANDAL, Objectron

A flat dataset's "sequence" is the frame itself, so one table with a `sequence`
column covers both and ``frame_id`` is built the same way for all four.

All four are ``frame_object``: one object per frame.  What makes that true
differs per dataset and is recorded in each subclass.

Indexing never reads a sharded cache
------------------------------------
``index()`` must produce the index *from the source tree*, and a sharded config
makes that easy to get wrong: ``ConfigurableDataset.__init__`` calls
``_setup_sharded()`` whenever ``sharded_name`` is set, after which ``__len__``
and ``__getitem__`` serve the cache rather than the tree.  Indexing through such
an instance would read back whatever a previous build happened to contain — so a
cache built from a thinned or stale walk would perpetuate itself, and a cache
built *from* this index would become its own input.

Two independent guards, because either alone is a construction detail someone
can undo later:

1. :meth:`index_walk_cfg` strips ``sharded_name``, ``use_huggingface``,
   ``transform`` and ``sharded_transform`` from the config, so even a fully
   initialised instance could not reach a cache.
2. :meth:`_walker` builds the instance with ``__new__``, bypassing ``__init__``
   entirely, so ``_setup_sharded`` is never called and no cache is opened.

The same call also drops ``filter_count_max``, ``subset_name`` and the
per-sequence caps: ``frames.db`` stores the *unthinned* walk and the caps are
re-applied on read, so a config yields the same items whether or not the cache
exists.  A subset-thinned index would starve every other config sharing the file.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from dataclasses import replace as _replace
from pathlib import Path
from typing import Iterator, Optional

from o3b.dataset.dataset import ConfigurableDataset, ItemType

logger = logging.getLogger(__name__)


def load_meta_yaml(path: Path) -> Optional[dict]:
    """Read one od3d meta YAML (re-exported from the UCO3D loader).

    Those files carry ``!!python/object/apply:pathlib.PosixPath`` tags, which
    SafeLoader rejects; the helper whitelists that one tag rather than reaching
    for ``unsafe_load``, which would execute whatever else a file contained.
    """
    from o3b.dataset.uco3d.dataset import _load_meta_yaml

    return _load_meta_yaml(path)


class Od3dFrameDataset(ConfigurableDataset):
    """Base for the four od3d-preprocessed ``frame_object`` datasets."""

    # ── subclass configuration ───────────────────────────────────────────────
    #: meta/frames/<split>/<category>/<sequence>/<frame>.yaml when True,
    #: meta/frames/<split>/<category>/<name>.yaml when False.
    has_sequence_level: bool = False
    #: category vocabulary and the maps onto UCO3D's, set by each subclass
    all_categories: tuple = ()
    map_categories_to_uco3d = None
    map_categories_obj_orient_to_uco3d = None

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, cfg):
        if cfg.item_type != ItemType.FRAME_OBJECT:
            raise ValueError(
                f"{type(self).__name__} supports item_type 'frame_object', "
                f"got {cfg.item_type}"
            )
        super().__init__(cfg)

    def _init_fields(self) -> None:
        cfg = self.cfg
        self.path_raw = Path(cfg.path_raw or cfg.root)
        self.path_preprocess = Path(cfg.path_preprocess or cfg.root)
        self.path_meta_frames = self.path_preprocess / "meta" / "frames"

    @classmethod
    def _walker(cls, cfg) -> "Od3dFrameDataset":
        """An instance able to walk the tree, with no index and no shard cache.

        Built with ``__new__`` so ``__init__`` — and with it ``_setup`` and
        ``_setup_sharded`` — never runs: walking is all this is for, and going
        through the constructor would build an index only to discard it, or open
        a sharded cache this must not read.  See the module docstring.
        """
        self = cls.__new__(cls)
        self.cfg = cfg
        self._sharded = None
        self._sharded_meshes = None
        self._sharded_mesh_rows = None
        self._transform = None
        self._sharded_transform = None
        self._init_fields()
        return self

    @staticmethod
    def index_walk_cfg(cfg):
        """The config ``index()`` walks with: source tree only, nothing thinned.

        Strips, and why:

        ``sharded_name`` / ``use_huggingface``
            the index is built *from* the tree; reading a cache here would let a
            stale or thinned build define the index that a later build reads.
        ``transform`` / ``sharded_transform``
            a walk enumerates rows, it does not load pixels — a crop configured
            for training has no meaning here and only costs time.
        ``filter_count_max`` / ``subset_name`` / the per-sequence caps
            frames.db holds the unthinned walk and every config re-applies its
            own caps on read, so a thinned index would silently shrink every
            other config sharing the file.
        """
        extra = dict(cfg.extra or {})
        extra.pop("subset_ids", None)
        extra["frames_count_max_per_sequence"] = None
        extra["sequences_count_max_per_category"] = None
        return _replace(
            cfg,
            sharded_name=None,
            use_huggingface=False,
            transform=None,
            sharded_transform=None,
            filter_count_max=None,
            subset_name=None,
            extra=extra,
        )

    # ── walking the meta tree ────────────────────────────────────────────────

    def _splits(self) -> list[str]:
        """Split directories to walk. ``split: all`` (or None) takes every one."""
        if self.cfg.split in (None, "all"):
            if not self.path_meta_frames.is_dir():
                return []
            return sorted(p.name for p in self.path_meta_frames.iterdir() if p.is_dir())
        return [self.cfg.split]

    def _iter_categories(self) -> list[str]:
        """Categories present on disk, restricted to ``cfg.categories`` if set."""
        wanted = set(self.cfg.categories) if self.cfg.categories else None
        found: set[str] = set()
        for split in self._splits():
            d = self.path_meta_frames / split
            if not d.is_dir():
                continue
            found |= {p.name for p in d.iterdir() if p.is_dir()}
        if wanted is not None:
            found &= wanted
        return sorted(found)

    def _walk_rows(self) -> Iterator[dict]:
        """Yield one row per (split, category, sequence, frame) meta file."""
        for split in self._splits():
            for category in self._iter_categories():
                cat_dir = self.path_meta_frames / split / category
                if not cat_dir.is_dir():
                    continue
                if self.has_sequence_level:
                    for seq_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
                        for meta in sorted(seq_dir.glob("*.yaml")):
                            yield self._row(split, category, seq_dir.name, meta.stem)
                else:
                    for meta in sorted(cat_dir.glob("*.yaml")):
                        # flat tree: the frame is its own sequence
                        yield self._row(split, category, meta.stem, meta.stem)

    def _row(self, split: str, category: str, sequence: str, frame: str) -> dict:
        return {
            "frame_id": f"{split}/{category}/{sequence}/{frame}",
            "split": split,
            "category": category,
            "sequence": sequence,
            "frame": frame,
        }

    def meta_path(self, row) -> Path:
        base = self.path_meta_frames / row["split"] / row["category"]
        if self.has_sequence_level:
            return base / row["sequence"] / f"{row['frame']}.yaml"
        return base / f"{row['frame']}.yaml"

    # ── index ────────────────────────────────────────────────────────────────

    _DB_NAME = "frames.db"

    @classmethod
    def index(cls, cfg, *, db: Optional[Path] = None, remove: bool = False,
              max_index: Optional[int] = None, **_) -> None:
        """Cache the meta-tree walk into ``<path_preprocess>/frames.db``.

        Reads the source tree only — never a sharded cache; see the module
        docstring for the two guards that enforce it.
        """
        walk_cfg = cls.index_walk_cfg(cfg)
        self = cls._walker(walk_cfg)

        db_path = Path(db) if db else self.path_preprocess / cls._DB_NAME
        db_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"path_preprocess : {self.path_preprocess}")
        print(f"meta tree       : {self.path_meta_frames}")
        print(f"db              : {db_path}")
        if cfg.sharded_name:
            print(f"note            : ignoring sharded_name={cfg.sharded_name!r} — "
                  f"the index is built from the meta tree, not from a cache")

        if remove and db_path.exists():
            print(f"Removing existing index: {db_path}")
            db_path.unlink()

        con = sqlite3.connect(db_path, timeout=60)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id   TEXT PRIMARY KEY,
                split      TEXT NOT NULL,
                category   TEXT NOT NULL,
                sequence   TEXT NOT NULL,
                frame      TEXT NOT NULL,
                -- which annotated object of the frame this row is. Always 0 for
                -- the three datasets whose meta is already per object; kept so
                -- emitting every object of a PASCAL3D image later is a walk
                -- change rather than a schema migration.
                object_idx INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS frames_cat ON frames (split, category)")

        categories = self._iter_categories()
        if not categories:
            print(f"ERROR: no categories found under {self.path_meta_frames}",
                  file=sys.stderr)
            sys.exit(1)

        from tqdm import tqdm

        print(f"Indexing {len(categories)} categor"
              f"{'y' if len(categories) == 1 else 'ies'} → {db_path}")
        total = 0
        batch: list[tuple] = []
        pbar = tqdm(desc="Indexing", unit="row")
        for category in categories:
            pbar.set_postfix_str(category)
            # Re-insert this category wholesale, so a resumed run redoes only
            # the category it was interrupted in and other categories survive.
            cur.execute("DELETE FROM frames WHERE category = ?", (category,))
            self.cfg = _replace(walk_cfg, categories=[category])
            for row in self._walk_rows():
                batch.append((row["frame_id"], row["split"], row["category"],
                              row["sequence"], row["frame"], 0))
                total += 1
                pbar.update(1)
                if len(batch) >= 5000:
                    cur.executemany(
                        "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?)", batch)
                    con.commit()
                    batch.clear()
                if max_index is not None and total >= max_index:
                    break
            if batch:
                cur.executemany(
                    "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
            con.commit()
            if max_index is not None and total >= max_index:
                break
        pbar.close()
        self.cfg = walk_cfg

        # Report what actually landed, rather than what the walk thinks it sent:
        # every coverage bug in this migration so far has been a check trusting
        # its own success line over the filesystem.
        n_rows = cur.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        n_cats = cur.execute("SELECT COUNT(DISTINCT category) FROM frames").fetchone()[0]
        con.close()
        print(f"Done. {total} rows walked; {n_rows} rows over {n_cats} categories "
              f"now in {db_path}")
