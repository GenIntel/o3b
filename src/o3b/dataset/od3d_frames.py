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

#: read_depth_image decodes uint16 millimetres, so this is the largest value the
#: encoding can represent; anything at it was clipped, not measured.
_DEPTH_U16_CEILING_M = 65.535


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

    # ── UCO3D axis alignment ─────────────────────────────────────────────────
    # Applied on *read*, never baked into the shards — matching od3d, whose
    # shard build saves use_map_obj_orient_uco3d / use_map_obj_syms_uco3d, sets
    # both False for the duration, and restores them afterwards (see
    # OD3D_Dataset around the save_to_sharded_dataset_path call).
    #
    # The split matters. The crop *is* baked in, because it is expensive and
    # fixed; the alignment is not, because it is a labelling. Re-deriving a
    # category's canonical axes then costs nothing, where baking it in would
    # mean rebuilding every shard of every dataset that category appears in.
    #
    # o3b's two paths already separate exactly here: the shard build calls
    # _load_sharded_item -> _load_item + sharded_transform, while reads go
    # through __getitem__. So overriding __getitem__ covers items loaded raw and
    # items read back from a cache, and is unreachable from the build.

    def _uco3d_category(self, category: Optional[str]) -> Optional[str]:
        """This dataset's category name in UCO3D's vocabulary."""
        if category is None:
            return None
        m = self.map_categories_to_uco3d
        if not m:
            return category
        mapped = m.get(category)
        return str(mapped) if mapped is not None else None

    def _obj_orient_tform(self, category: Optional[str]):
        """(4, 4) rotation taking this category's object axes onto UCO3D's.

        None when the dataset needs no re-orientation (PASCAL3D, Objectron) or
        the category is unmapped.
        """
        import torch

        from o3b.cv.geometry.transform import transf4x4_from_rot3x3

        m = self.map_categories_obj_orient_to_uco3d
        if not m or category is None:
            return None
        rot = m.get(category)
        if rot is None:
            return None
        return transf4x4_from_rot3x3(torch.tensor(rot, dtype=torch.float32))

    def _apply_uco3d_alignment(self, item):
        """Re-express one item's object frame and symmetry in UCO3D's convention.

        Mirrors od3d's OD3D_Dataset.get_frames bookkeeping: the object frame is
        rotated by T, so the pose is post-multiplied by inv(T) and the mesh and
        the NCDS transform follow, leaving the object projecting onto exactly the
        same pixels. obj_size3d is permuted by |R| because rotating the axes
        permutes which side length belongs to which.
        """
        import torch

        from o3b.cv.geometry.transform import inv_tform4x4

        if item is None or not (self.cfg.extra or {}).get("map_to_uco3d"):
            return item

        category = getattr(item, "category", None)
        if isinstance(category, (list, tuple)):      # defensive: never batched here
            return item
        uco3d_cat = self._uco3d_category(category)
        T = self._obj_orient_tform(category)

        if T is not None:
            R_abs = T[:3, :3].abs()
            if item.cam_tform4x4_obj is not None:
                item.cam_tform4x4_obj = item.cam_tform4x4_obj @ inv_tform4x4(T)
            if item.obj_ncds0c_tform4x4_obj is not None:
                item.obj_ncds0c_tform4x4_obj = T @ item.obj_ncds0c_tform4x4_obj @ inv_tform4x4(T)
            if item.obj_size3d is not None:
                item.obj_size3d = (item.obj_size3d[..., None, :] * R_abs).sum(dim=-1)
            if item.mesh is not None:
                item.mesh.transf3d(T)
            if item.obj_kpts3d is not None:
                item.obj_kpts3d = item.obj_kpts3d @ T[:3, :3].T
            if item.cam_tform4x4_obj is not None and item.obj_ncds0c_tform4x4_obj is not None:
                item.cam_tform4x4_obj_ncds = (
                    item.cam_tform4x4_obj @ inv_tform4x4(item.obj_ncds0c_tform4x4_obj)
                )

        # Symmetry comes from UCO3D's orientation tree, which is stated in
        # UCO3D's axes. With T applied the item is already in those axes; without
        # it (an unmapped category, or a dataset with no orientation table) the
        # code has to be rotated back into the item's own frame, or the pose
        # metric would quotient out the wrong axis.
        if uco3d_cat is not None:
            from o3b.dataset.uco3d.obj_syms import obj_syms_for_category

            try:
                syms = obj_syms_for_category(uco3d_cat)
            except KeyError:
                syms = None
            if syms is not None:
                if T is None:
                    T_back = self._obj_orient_tform(category)
                    if T_back is not None:
                        R_abs = inv_tform4x4(T_back)[:3, :3].abs()
                        syms = (syms[..., None, :].float() * R_abs).sum(dim=-1).long()
                item.obj_syms = syms
        return item

    def __getitem__(self, idx: int):
        # super() serves either a shard record or a freshly loaded item, then
        # applies cfg.transform; the alignment goes on top of both. The shard
        # build does not come through here (see the note above), so what is
        # written to disk stays in the dataset's own object frame.
        return self._apply_uco3d_alignment(super().__getitem__(idx))

    # ── viz ──────────────────────────────────────────────────────────────────

    @classmethod
    def visualize(cls, cfg, *, db: Optional[Path] = None, limit: int = 20,
                  object_id: Optional[str] = None, render: bool = False,
                  debug: bool = False, obj_centric: bool = False, **_) -> None:
        """Browse items in viser: mesh, frustum, rgb panel, depth cloud, axes.

        Shares HouseCorr3D's frame-object viewer, so all six evaluation sets are
        inspected through one tool and a pose convention that disagrees between
        them is visible side by side rather than inferred from a metric.

        Worth doing before sharding rather than after: an object frame that is
        wrong by a rotation indexes and shards perfectly happily, and only shows
        up as a disappointing number several phases later.
        """
        from dataclasses import replace as _r

        from o3b.dataset.housecorr3d.frame_dataset import _visualize_frame_objects_viser

        path_preprocess = cls._path_preprocess(cfg)
        if not (path_preprocess / "meta" / "frames").is_dir():
            print(
                f"No meta tree at {path_preprocess / 'meta' / 'frames'}.\n"
                f"  Fetch it:              o3b dataset fetch -d <config> -p slurm_lmbl40\n"
                f"  or run on the cluster: o3b dataset viz -d <config> -p slurm_lmbl40 --remote",
                file=sys.stderr,
            )
            sys.exit(1)

        # modalities=None so the viewer gets everything the loader can produce,
        # not just what a training config asked for.
        viz_cfg = _r(cfg, modalities=None, filter_count_max=limit)
        dataset = cls(viz_cfg)
        if len(dataset) == 0:
            print("No frames found matching the current config filters.")
            return
        print(f"Showing up to {limit} of {len(dataset)} frames  {path_preprocess}\n")
        _visualize_frame_objects_viser(dataset, debug=debug, obj_centric=obj_centric)

    # ── item loading ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._init_fields()
        db_path = self.path_preprocess / self._DB_NAME
        rows = None
        if db_path.exists() and not (self.cfg.extra or {}).get("ignore_frames_db"):
            rows = self._rows_from_db(db_path)
        if rows is None:
            rows = list(self._walk_rows())

        # Caps are applied here, on read, never at index time — frames.db holds
        # the unthinned walk so every config sharing it sees all of it.
        sub = self.subset()
        if sub is not None:
            rows = [r for r in rows
                    if self.in_subset(object_id=f"{r['category']}/{r['sequence']}",
                                      frame_id=r["frame_id"])]
        per_seq = (self.cfg.extra or {}).get("frames_count_max_per_sequence")
        if per_seq:
            seen: dict = {}
            kept = []
            for r in rows:
                key = (r["category"], r["sequence"])
                if seen.get(key, 0) >= per_seq:
                    continue
                seen[key] = seen.get(key, 0) + 1
                kept.append(r)
            rows = kept
        if self.cfg.filter_count_max:
            if self.cfg.categories:
                # per category, so a rare one is not starved by a common one
                seen = {}
                kept = []
                for r in rows:
                    c = r["category"]
                    if seen.get(c, 0) >= self.cfg.filter_count_max:
                        continue
                    seen[c] = seen.get(c, 0) + 1
                    kept.append(r)
                rows = kept
            else:
                rows = rows[: self.cfg.filter_count_max]
        self._frame_rows = rows

    def _rows_from_db(self, db_path: Path) -> Optional[list[dict]]:
        """Read the cached walk, or None when it does not cover this config.

        Returning None rather than a partial answer is deliberate: a cache
        missing some of the requested categories would silently shrink the
        dataset, which is the failure that looks like a bad result rather than
        like a bug.
        """
        cats = list(self.cfg.categories) if self.cfg.categories else None
        splits = self._splits()
        con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()
            wanted = set(cats) if cats is not None else set(self._iter_categories())
            if not wanted:
                return None
            missing = [c for c in wanted if not cur.execute(
                "SELECT 1 FROM frames WHERE category = ? LIMIT 1", (c,)).fetchone()]
            if missing:
                logger.info(
                    f"frames.db is missing {len(missing)}/{len(wanted)} requested "
                    f"categor{'y' if len(missing) == 1 else 'ies'} "
                    f"({', '.join(sorted(missing)[:5])}"
                    f"{'…' if len(missing) > 5 else ''}); walking the tree instead")
                return None
            q = "SELECT * FROM frames WHERE split IN ({})".format(
                ", ".join("?" * len(splits)))
            params = list(splits)
            if cats is not None:
                q += " AND category IN ({})".format(", ".join("?" * len(cats)))
                params += cats
            q += " ORDER BY category, sequence, frame"
            return [dict(r) for r in cur.execute(q, params)]
        finally:
            con.close()

    def __len__(self) -> int:
        if self._sharded is not None:
            return len(self._sharded)
        return len(getattr(self, "_frame_rows", ()))

    # ── per-dataset path hooks ───────────────────────────────────────────────

    def _rel_key(self, row) -> str:
        """``<split>/<category>[/<sequence>]/<frame>`` — the parallel trees' key."""
        base = f"{row['split']}/{row['category']}"
        if self.has_sequence_level:
            base += f"/{row['sequence']}"
        return f"{base}/{row['frame']}"

    def _mask_path(self, row, meta) -> Optional[Path]:
        """Where this row's instance mask lives.

        Two conventions among the four: HANDAL names it in the meta (relative to
        path_raw), the others keep parallel trees under path_preprocess.
        """
        if meta.get("rfpath_mask"):
            return self.path_raw / meta["rfpath_mask"]
        mask_type = (self.cfg.extra or {}).get("mask_type")
        if not mask_type:
            return None
        return self.path_preprocess / "mask" / mask_type / f"{self._rel_key(row)}.png"

    def _depth_path(self, row, meta) -> Optional[Path]:
        if meta.get("rfpath_depth"):
            return self.path_raw / meta["rfpath_depth"]
        extra = self.cfg.extra or {}
        depth_type = extra.get("depth_type")
        if not depth_type:
            return None
        base = self.path_preprocess / "depth" / depth_type
        # PASCAL3D / ImageNet3D render depth from a mesh, so their tree carries a
        # further <mesh_type> level (depth/mesh/meta/...); Objectron's estimator
        # output does not.
        if extra.get("depth_mesh_type"):
            base = base / extra["depth_mesh_type"]
        return base / f"{self._rel_key(row)}.png"

    def _load_frame_object(self, idx: int):
        import torch

        from o3b.cv.io import read_depth_image, read_image
        from o3b.dataset.utils import want as _want

        row = self._frame_rows[idx]
        mods = self.cfg.modalities
        meta = load_meta_yaml(self.meta_path(row))
        if meta is None:
            return None

        rgb = None
        if _want("rgb", mods) and meta.get("rfpath_rgb"):
            p = self.path_raw / meta["rfpath_rgb"]
            if p.exists():
                img = read_image(p)
                rgb = img[:3].float() / 255.0 if img is not None else None
        if rgb is None:
            return None

        fo_mask = None
        if _want("fo_mask", mods):
            p = self._mask_path(row, meta)
            if p is not None and p.exists():
                m = read_image(p)
                if m is not None:
                    fo_mask = (m[0] > 127) if m.dtype == torch.uint8 else (m[0] > 0.5)

        depth, depth_mask = None, None
        if _want("depth", mods) or _want("depth_mask", mods):
            p = self._depth_path(row, meta)
            if p is not None and p.exists():
                d = read_depth_image(p, factor=float(meta.get("depth_scale", 1000.0)))
                if d is not None:
                    depth = d[0] if d.dim() == 3 else d
                    # Depth is stored as uint16 millimetres, so anything past
                    # 65.535 m saturates at exactly that value rather than
                    # clipping to something obviously wrong. Monocular estimates
                    # run past it on open scenes — 42% of the in-mask pixels of a
                    # PASCAL3D aeroplane sit on the ceiling — and a saturated
                    # pixel lifted as a real 65 m point would drag the encoder's
                    # point cloud with it. Excluded from depth_mask so nothing
                    # downstream treats it as a measurement.
                    valid = (depth > 0) & (depth < _DEPTH_U16_CEILING_M)
                    depth_mask = valid if _want("depth_mask", mods) else None
                    if not _want("depth", mods):
                        depth = None

        cam_intr4x4 = None
        if meta.get("l_cam_intr4x4"):
            cam_intr4x4 = torch.tensor(meta["l_cam_intr4x4"], dtype=torch.float32)

        cam_tform4x4_obj = None
        if meta.get("l_cam_tform4x4_obj"):
            M = torch.tensor(meta["l_cam_tform4x4_obj"], dtype=torch.float32)
            # od3d writes cam<-obj in an OpenCV-style frame (+Y down, +Z forward);
            # o3b is OpenGL. Same flip UCO3D applies.
            if self.cfg.cam_tform4x4_cam_raw is not None:
                M = torch.tensor(self.cfg.cam_tform4x4_cam_raw, dtype=torch.float32) @ M
            scale = float((self.cfg.extra or {}).get("scale_to_m") or 1.0)
            if scale != 1.0:
                M = M.clone()
                M[:3, 3] = M[:3, 3] * scale   # HANDAL's poses are millimetres
            cam_tform4x4_obj = M

        cam_bbox2d = None
        if _want("cam_bbox2d", mods):
            if meta.get("l_bbox"):
                cam_bbox2d = torch.tensor(meta["l_bbox"], dtype=torch.float32)
            elif fo_mask is not None and bool(fo_mask.any()):
                from o3b.cv.visual.draw import get_bboxs_from_masks
                cam_bbox2d = get_bboxs_from_masks(fo_mask[None])[0].float()

        mesh, obj_size3d, obj_bbox3d, obj_ncds0c = self._object_geometry(row, meta)

        # obj_kpts3d must be in the SAME space as mesh.verts, i.e. NCDS: the
        # viewer (and o3b's Object.transform) map the whole item by
        # cam_tform4x4_obj_ncds and use that one transform for mesh and
        # keypoints alike. The metas store them in raw object space — PASCAL3D's
        # l_kpts3d is its CAD bounding-box corners — so leaving them unnormalised
        # projects them onto the wrong pixels while the mesh looks correct.
        obj_kpts3d = None
        if meta.get("l_kpts3d") and _want("obj_kpts3d", mods):
            obj_kpts3d = torch.tensor(meta["l_kpts3d"], dtype=torch.float32)
            if obj_ncds0c is not None:
                from o3b.cv.geometry.transform import inv_tform4x4
                inv = inv_tform4x4(obj_ncds0c)
                obj_kpts3d = obj_kpts3d @ inv[:3, :3].T + inv[:3, 3]

        cam_tform4x4_obj_ncds = None
        if cam_tform4x4_obj is not None and obj_ncds0c is not None:
            cam_tform4x4_obj_ncds = cam_tform4x4_obj @ obj_ncds0c

        # Two different boxes, exactly as HouseCorr3D distinguishes them:
        #   obj_bbox3d  (8, 3) corners in OBJECT space — a property of the object
        #   cam_bbox3d  (8, 3) the same corners in CAMERA space, i.e. per frame
        # FrameObject.viz draws its 3-D box overlay from cam_bbox3d, so a loader
        # that fills only obj_bbox3d leaves the viewer nothing to draw and the
        # box silently never appears.
        cam_bbox3d = None
        if (_want("cam_bbox3d", mods) and obj_bbox3d is not None
                and cam_tform4x4_obj is not None):
            R, tr = cam_tform4x4_obj[:3, :3], cam_tform4x4_obj[:3, 3]
            cam_bbox3d = obj_bbox3d.float() @ R.t() + tr
        if not _want("obj_bbox3d", mods):
            obj_bbox3d = None

        # The mesh rasterised under the GT pose. Worth having as a modality
        # rather than only in a viewer: overlaid on the rgb it shows directly
        # whether the pose and the object frame agree with the image, which is
        # the check the whole cross-dataset comparison rests on and which no
        # scalar metric makes visible.
        fo_mask_amodal, fo_mask_amodal_dt = None, None
        if (_want("fo_mask_amodal", mods) or _want("fo_mask_amodal_dt", mods)) \
                and mesh is not None and cam_tform4x4_obj_ncds is not None \
                and cam_intr4x4 is not None and rgb is not None:
            fo_mask_amodal = self._render_mesh_mask(
                mesh, cam_tform4x4_obj_ncds, cam_intr4x4,
                H=rgb.shape[-2], W=rgb.shape[-1])
            if fo_mask_amodal is not None and _want("fo_mask_amodal_dt", mods):
                from o3b.cv.visual.mask import get_mask_distance_transform_norm
                fo_mask_amodal_dt = get_mask_distance_transform_norm(
                    fo_mask_amodal.cpu()).float()
            if not _want("fo_mask_amodal", mods):
                fo_mask_amodal = None

        from o3b.data.datatypes.frame_object import FrameObject

        object_id = f"{row['category']}/{row['sequence']}"
        return FrameObject(
            frame_id=row["frame_id"],
            frame_object_id=row["frame_id"],
            object_id=object_id,
            category=row["category"],
            rgb=rgb,
            depth=depth,
            depth_mask=depth_mask,
            fo_mask=fo_mask,
            cam_intr4x4=cam_intr4x4,
            cam_tform4x4_obj=cam_tform4x4_obj,
            cam_bbox2d=cam_bbox2d,
            obj_kpts3d=obj_kpts3d,
            mesh=mesh,
            obj_size3d=obj_size3d,
            obj_bbox3d=obj_bbox3d,
            cam_bbox3d=cam_bbox3d,
            obj_ncds0c_tform4x4_obj=obj_ncds0c,
            cam_tform4x4_obj_ncds=cam_tform4x4_obj_ncds,
            fo_mask_amodal=fo_mask_amodal,
            fo_mask_amodal_dt=fo_mask_amodal_dt,
        )

    # ── object geometry ──────────────────────────────────────────────────────
    # obj_ncds0c_tform4x4_obj follows UCO3D's convention exactly: the mesh verts
    # are centred and divided by half their longest extent (so they span [-1, 1]
    # on that axis), and the transform maps NCDS coordinates back into object
    # space.  The benchmark compares cam_tform4x4_obj_ncds, so a dataset that
    # normalised differently would be scored against a differently-shaped target.
    #
    # Note for PASCAL3D / ImageNet3D: their poses are in normalised CAD units,
    # not metres, and no per-category table recovers that reliably.  NCDS is
    # scale-free by construction, so the rotation metrics those two datasets are
    # evaluated on are unaffected; metric translation and 3-D IoU are simply not
    # measurable there, which is why the published table reports only 30/10
    # degree for both.

    _MESH_CACHE_MAX = 64

    def _mesh_path(self, row, meta) -> Optional[Path]:
        """Where this row's object mesh lives; None when the dataset has none.

        PASCAL3D and ImageNet3D name a CAD model in the meta, relative to
        path_raw.  Subclasses with another arrangement override this.
        """
        if meta.get("rfpath_mesh"):
            return self.path_raw / meta["rfpath_mesh"]
        return None

    def _object_geometry(self, row, meta):
        """``(mesh, obj_size3d, obj_bbox3d, obj_ncds0c_tform4x4_obj)``.

        Geometry comes from the mesh where there is one, and otherwise from the
        meta's 3-D box corners — which is what Objectron has instead of a scan,
        its "mesh" being a cuboid fitted to the annotation anyway.

        Cached per object: the frames of one object arrive together, so a small
        cache turns N mesh reads into one, while keeping every object would grow
        without bound across an epoch.
        """
        import torch

        from o3b.data.datatypes.mesh import Mesh

        key = f"{row['category']}/{row['sequence']}"
        cache = getattr(self, "_mesh_cache", None)
        if cache is None:
            cache = self._mesh_cache = {}
        if key in cache:
            cached = cache[key]
            if cached is None:
                return None, None, None, None
            from dataclasses import replace as _r
            mesh, size3d, bbox3d, tform = cached
            return (_r(mesh) if mesh is not None else None), size3d, bbox3d, tform

        if len(cache) >= self._MESH_CACHE_MAX:
            cache.clear()

        verts, faces = None, None
        path = self._mesh_path(row, meta)
        if path is not None and path.exists():
            try:
                import trimesh
                loaded = trimesh.load(path, process=False)
                verts = torch.tensor(loaded.vertices, dtype=torch.float32)
                faces = torch.tensor(loaded.faces, dtype=torch.int64)
            except Exception as e:
                logger.warning(f"could not read mesh {path}: {e}")

        pts = verts
        if pts is None and meta.get("l_kpts3d"):
            pts = torch.tensor(meta["l_kpts3d"], dtype=torch.float32)
        if pts is None or pts.numel() == 0:
            cache[key] = None
            return None, None, None, None

        v_min, v_max = pts.min(dim=0).values, pts.max(dim=0).values
        center = (v_min + v_max) * 0.5
        size3d = (v_max - v_min)
        half_scale = float(size3d.max().clamp(min=1e-8)) * 0.5

        tform = torch.eye(4, dtype=torch.float32)
        tform[:3, :3] = torch.eye(3) * half_scale
        tform[:3, 3] = center

        # the 8 corners, in object space
        bbox3d = torch.stack([
            torch.tensor([x, y, z], dtype=torch.float32)
            for x in (v_min[0], v_max[0])
            for y in (v_min[1], v_max[1])
            for z in (v_min[2], v_max[2])
        ])

        mesh = None
        if verts is not None and faces is not None:
            mesh = Mesh(verts=(verts - center) / half_scale, faces=faces)

        cache[key] = (mesh, size3d, bbox3d, tform)
        from dataclasses import replace as _r
        return (_r(mesh) if mesh is not None else None), size3d, bbox3d, tform

    def _render_mesh_mask(self, mesh, cam_tform4x4_obj_ncds, cam_intr4x4, H: int, W: int):
        """(H, W) bool silhouette of `mesh` rasterised under the given pose.

        Shares HouseCorr3D's rasteriser, so the silhouette lands on the same
        pixels its fo_mask_amodal would. Only this object is drawn, so occluders
        leave no hole: the result is the full silhouette, cut off only by the
        image border.
        """
        import torch

        from o3b.dataset.housecorr3d.frame_dataset import render_scene_depth

        try:
            M = cam_tform4x4_obj_ncds.float()
            verts_cam = (M[:3, :3] @ mesh.verts.float().t()).t() + M[:3, 3]
            depth = render_scene_depth([(verts_cam, mesh.faces)], cam_intr4x4.float(), H, W)
        except Exception as e:                       # a bad mesh must not kill the item
            logger.warning(f"could not render mesh mask: {e}")
            return None
        if depth is None:
            return None
        return depth > 0        # 0 = no hit = background
