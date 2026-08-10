from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import register_dataset, ConfigurableDataset, ItemType

import torch
from o3b.data.modalities import (
    FrameObject, SceneObject, Object, ObjectPair,
)

from o3b.dataset.housecorr3d.enum import OMNI6DPOSE_CATEGORIES

_MESH_EXTS = {".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"}


@register_dataset("HouseCorr3D")
class HouseCorr3D(ConfigurableDataset):

    categories: tuple = tuple(OMNI6DPOSE_CATEGORIES)

    @property
    def path_raw(self) -> Path:
        return self.cfg.path_raw

    @property
    def path_preprocess(self) -> Path:
        return self.cfg.path_preprocess

    @property
    def path_object_meshes(self) -> Path:
        return self.path_raw / "PAM" / "object_meshes"

    @property
    def path_raw_meta(self) -> Path:
        return self.path_raw / "Meta"

    def __len__(self):
        if self._sharded is not None:
            return len(self._sharded)
        if self.cfg.item_type == ItemType.FRAME_OBJECT:
            return len(self._frame_rows_id)
        if self.cfg.item_type == ItemType.FRAME_OBJECT_PAIR:
            return len(self._frame_pairs_id)
        return len(self._object_rows_id)

    def _setup(self) -> None:
        self._object_rows: list[dict] = []
        self._object_rows_id: list = []
        self._obj_by_id: dict[str, dict] = {}
        self._frame_rows: list[dict] = []
        self._frame_rows_id: list[int] = []
        self._frame_pairs_id: list[tuple[int, int]] = []

        # Always load the objects table from index.db — used for mesh lookup in all item types.
        db_path = self.path_preprocess / "index.db"
        if db_path.exists():
            con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
            con.row_factory = sqlite3.Row
            try:
                cur = con.cursor()
                self._object_rows = [dict(r) for r in cur.execute("SELECT * FROM objects").fetchall()]
                _id_to_idx: dict[str, int] = {r["object_id"]: i for i, r in enumerate(self._object_rows)}
                self._obj_by_id = {r["object_id"]: r for r in self._object_rows}

                cats = self.cfg.categories  # Optional[list[str]]

                if self.cfg.item_type == ItemType.OBJECT:
                    kpts_clause = " AND obj_kpts3d IS NOT NULL" if self.cfg.filter_has_kpts else ""
                    real_clause = (
                        " AND object_id LIKE '%real%'" if self.cfg.filter_is_real is True
                        else " AND object_id NOT LIKE '%real%'" if self.cfg.filter_is_real is False
                        else ""
                    )
                    limit_clause = f" LIMIT {self.cfg.filter_count_max}" if self.cfg.filter_count_max else ""
                    if cats:
                        placeholders = ", ".join("?" * len(cats))
                        cat_clause   = f" AND category IN ({placeholders})"
                        params: list = list(cats)
                    else:
                        cat_clause, params = "", []
                    rows = cur.execute(
                        f"SELECT object_id FROM objects WHERE 1=1{kpts_clause}{real_clause}{cat_clause}{limit_clause}",
                        params,
                    ).fetchall()
                    self._object_rows_id = [_id_to_idx[r["object_id"]] for r in rows]

                elif self.cfg.item_type == ItemType.OBJECT_PAIR:
                    # Pairs are derived on the fly (all same-category combinations);
                    # no separate `object_pairs` index table is required.
                    kpts_clause = (
                        " AND a.obj_kpts3d IS NOT NULL AND b.obj_kpts3d IS NOT NULL"
                        if self.cfg.filter_has_kpts else ""
                    )
                    real_clause = (
                        " AND a.object_id LIKE '%real%' AND b.object_id LIKE '%real%'"
                        if self.cfg.filter_is_real is True
                        else " AND a.object_id NOT LIKE '%real%' AND b.object_id NOT LIKE '%real%'"
                        if self.cfg.filter_is_real is False
                        else ""
                    )
                    limit_clause = f" LIMIT {self.cfg.filter_count_max}" if self.cfg.filter_count_max else ""
                    if cats:
                        placeholders = ", ".join("?" * len(cats))
                        cat_clause   = f" AND a.category IN ({placeholders})"
                        params       = list(cats)
                    else:
                        cat_clause, params = "", []
                    rows = cur.execute(f"""
                        SELECT a.object_id AS src_object_id, b.object_id AS trgt_object_id
                        FROM objects a
                        JOIN objects b ON a.category IS b.category AND a.object_id != b.object_id
                        WHERE a.category IS NOT NULL{kpts_clause}{real_clause}{cat_clause}
                        ORDER BY a.object_id, b.object_id{limit_clause}
                    """, params).fetchall()
                    self._object_rows_id = [
                        (_id_to_idx[r["src_object_id"]], _id_to_idx[r["trgt_object_id"]])
                        for r in rows
                    ]
            finally:
                con.close()

        if self.cfg.item_type in (ItemType.FRAME_OBJECT, ItemType.FRAME_OBJECT_PAIR):
            frames_db = self.path_preprocess / "frames.db"
            if not frames_db.exists():
                return
            con2 = sqlite3.connect(f"file:{frames_db}?immutable=1", uri=True, timeout=30)
            con2.row_factory = sqlite3.Row
            try:
                cur2 = con2.cursor()
                self._frame_rows = [dict(r) for r in cur2.execute("SELECT * FROM frames").fetchall()]

                split = self.cfg.split or "train"
                is_real = self.cfg.filter_is_real
                cats2 = self.cfg.categories
                limit = self.cfg.filter_count_max
                has_kpts = self.cfg.filter_has_kpts

                split_clause = "" if split == "all" else f" AND split = '{split}'"
                if is_real is None:
                    real_clause2 = ""
                elif is_real:
                    real_clause2 = " AND data_type = 'real'"
                else:
                    real_clause2 = " AND data_type = 'synthetic'"
                kpts_clause2  = " AND has_kpts = 1" if has_kpts else ""
                # In pair mode filter_count_max caps the number of *pairs* (applied in
                # _build_frame_pairs), so the full frame pool must stay available here.
                limit_clause2 = (
                    f" LIMIT {limit}"
                    if limit and self.cfg.item_type == ItemType.FRAME_OBJECT else ""
                )

                if cats2:
                    placeholders2 = ", ".join("?" * len(cats2))
                    cat_clause2   = f" AND category IN ({placeholders2})"
                    params2: list = list(cats2)
                else:
                    cat_clause2, params2 = "", []

                frows = cur2.execute(
                    f"SELECT rowid FROM frames"
                    f" WHERE is_valid = 1{split_clause}{real_clause2}{kpts_clause2}{cat_clause2}{limit_clause2}",
                    params2,
                ).fetchall()
                # rowid is 1-based; our list is 0-based
                self._frame_rows_id = [r[0] - 1 for r in frows]
            finally:
                con2.close()

            if self.cfg.item_type == ItemType.FRAME_OBJECT_PAIR:
                self._frame_pairs_id = self._build_frame_pairs()

    def _build_frame_pairs(self) -> list[tuple[int, int]]:
        """Build cross-instance frame pairs within each category.

        Frames are grouped by category, then by object instance (object_id).  Up
        to ``frame_pair_views_per_instance`` evenly-spaced viewpoints are sampled
        per instance (-1 = all available frames), and each instance is paired with
        every *other* distinct instance of the same category as ordered (query,
        target) pairs.  This is independent of row ordering — unlike adjacent
        pairing it does not collapse when a category has long same-instance runs
        in frames.db.

        ``frame_pair_view_mode`` selects how the two instances' viewpoints combine:
          "aligned" → view i of A with view i of B (~n_views pairs per instance pair)
          "cross"   → every view of A with every view of B (~n_views^2)

        Viewpoint-index combinations are emitted broad-coverage-first so that under
        the filter_count_max cap every instance pair gets its earliest combinations
        before any gets a later one.
        """
        from itertools import permutations

        views_cfg = self.cfg.frame_pair_views_per_instance
        use_all   = views_cfg is not None and views_cfg < 0
        n_views   = None if use_all else max(1, views_cfg)
        cross     = self.cfg.frame_pair_view_mode == "cross"

        def _sample_views(rids: list[int]) -> list[int]:
            """Up to n_views evenly-spaced viewpoints (all of them when use_all)."""
            if use_all or len(rids) <= n_views:
                return rids
            step = len(rids) / n_views
            return [rids[int(i * step)] for i in range(n_views)]

        def _view_combos(max_v: int) -> list[tuple[int, int]]:
            """Viewpoint-index combinations, broad-coverage first."""
            if cross:
                return sorted(
                    ((i, j) for i in range(max_v) for j in range(max_v)),
                    key=lambda ij: (max(ij), ij[0], ij[1]),
                )
            return [(i, i) for i in range(max_v)]

        # category -> { object_id -> [all frame row indices for that instance] }
        by_cat: dict = {}
        for rid in self._frame_rows_id:
            row = self._frame_rows[rid]
            by_cat.setdefault(row.get("category"), {}).setdefault(row.get("object_id"), []).append(rid)

        pairs: list[tuple[int, int]] = []
        for instances in by_cat.values():
            views = {oid: _sample_views(rids) for oid, rids in instances.items()}
            oids = list(views)
            max_v = max((len(v) for v in views.values()), default=0)
            for vi, vj in _view_combos(max_v):
                for a_oid, b_oid in permutations(oids, 2):
                    av, bv = views[a_oid], views[b_oid]
                    if vi < len(av) and vj < len(bv):
                        pairs.append((av[vi], bv[vj]))

        limit = self.cfg.filter_count_max
        if limit:
            pairs = pairs[:limit]
        return pairs

    def _load_object(self, idx: int) -> Object:
        return self._load_object_from_row(self._object_rows[self._object_rows_id[idx]])

    def _load_object_from_row(self, row: dict) -> Object:
        oid = row["object_id"]
        mods = self.cfg.object_modalities

        need_mesh        = _want("mesh",                    mods)
        need_verts3d     = _want("verts3d",                 mods)
        need_verts3d_feats = _want("verts3d_feats",         mods)
        need_tform       = _want("obj_ncds0c_tform4x4_obj", mods)
        need_kpts        = _want("obj_kpts3d",              mods)
        need_category    = _want("category",                mods)
        need_syms        = _want("obj_syms",                mods)

        mesh, tform = None, None
        if need_mesh or need_verts3d or need_verts3d_feats or need_tform or need_kpts:
            mesh_rel = row.get("mesh_path")
            mesh_entry = self.path_raw / mesh_rel if mesh_rel else self.path_object_meshes / oid
            from o3b.io import _load_mesh
            mesh, tform = _load_mesh(mesh_entry)

            if (need_mesh or need_verts3d or need_verts3d_feats) and \
                    self.cfg.mesh_type != "default" and mesh is not None:
                from o3b.data.datatypes.mesh import Mesh
                converted_path = (
                    self.path_preprocess / "mesh" / self.cfg.mesh_type / f"{oid}.glb"
                )
                mesh = Mesh.load_or_convert(converted_path, mesh_entry, self.cfg.mesh_type)

        kpts3d, kpts3d_mask = None, None
        if need_kpts:
            kpts3d, kpts3d_mask = _load_kpts3d_by_id(oid, self.path_preprocess, tform)

        obj_syms, obj_kpts3d_syms, obj_axis6d_sym = None, None, None
        if need_syms:
            obj_syms = _load_obj_syms_from_row(row)
            obj_kpts3d_syms, obj_axis6d_sym = _compute_obj_sym_geometry(obj_syms, kpts3d)

        category_id = None
        category = None
        if need_category:
            cat_str = row.get("category")
            if cat_str is not None:
                try:
                    category_id = list(self.categories).index(cat_str)
                except ValueError:
                    pass
            category = row.get("category")

        obj_size_ncds = 2.0 if tform is not None else None
        obj_size      = float(2.0 * tform[0, 0].item()) if tform is not None else None

        obj = Object(
            object_id               = oid,
            mesh                    = mesh if need_mesh else None,
            verts3d                 = mesh.verts if (need_verts3d and mesh is not None) else None,
            verts3d_feats           = mesh.vert_feats if (need_verts3d_feats and mesh is not None) else None,
            obj_ncds0c_tform4x4_obj = tform if (need_tform or need_mesh) else None,
            obj_size_ncds           = obj_size_ncds,
            obj_size                = obj_size,
            obj_kpts3d              = kpts3d,
            obj_kpts3d_mask         = kpts3d_mask,
            obj_syms                = obj_syms,
            obj_kpts3d_syms         = obj_kpts3d_syms,
            obj_axis6d_sym          = obj_axis6d_sym,
            category                = category,
            category_id             = category_id,
        )

        if self.cfg.obj_gl_tform4x4_obj_raw is not None:
            import torch as _torch
            from dataclasses import replace as _r
            T = _torch.tensor(self.cfg.obj_gl_tform4x4_obj_raw, dtype=_torch.float32)
            # transform() rotates verts/kpts by T and updates M → M @ inv(T).
            obj = obj.transform(T)
            # Apply T on the *other* side too, so M → T @ M @ inv(T) (both sides).
            # This rotates the metric object frame itself by T into the canonical
            # (right/top/back → X/Y/Z) orientation, instead of leaving metric invariant.
            # The matching pose correction is cam_tform4x4_obj @ inv(T) in the frame loader.
            if obj.obj_ncds0c_tform4x4_obj is not None:
                obj = _r(
                    obj,
                    obj_ncds0c_tform4x4_obj=T @ obj.obj_ncds0c_tform4x4_obj.float(),
                )

        # Compute obj_size3d / obj_bbox3d in canonical metric object space.
        # tform = obj_ncds0c_tform4x4_obj maps NCDS-space verts → metric object space.
        if obj.mesh is not None and obj.mesh.verts is not None and obj.obj_ncds0c_tform4x4_obj is not None:
            import torch as _torch
            from dataclasses import replace as _r
            from o3b.cv.geometry.transform import transf3d_broadcast as _t3d
            verts_ncds = obj.mesh.verts.float()  # NCDS-normalized verts
            _tform = obj.obj_ncds0c_tform4x4_obj.float()  # NCDS → metric
            vmin = verts_ncds.min(0).values
            vmax = verts_ncds.max(0).values
            cx, cy, cz = ((vmin + vmax) / 2).tolist()
            hx, hy, hz = ((vmax - vmin) / 2).tolist()
            corners_ncds = _torch.tensor([
                [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
                [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
                [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
                [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
            ], dtype=_torch.float32)
            _bbox3d = _t3d(corners_ncds, _tform)  # (8, 3) metric object space
            _size3d = _bbox3d.max(0).values - _bbox3d.min(0).values  # (3,) metric
            obj = _r(obj, obj_size3d=_size3d, obj_bbox3d=_bbox3d)

        return obj

    def _load_object_pair(self, idx: int) -> ObjectPair:
        src_idx, trgt_idx = self._object_rows_id[idx]
        src_row  = self._object_rows[src_idx]
        trgt_row = self._object_rows[trgt_idx]
        return ObjectPair(
            src_object_id  = src_row["object_id"],
            trgt_object_id = trgt_row["object_id"],
            src_object     = self._load_object_from_row(src_row),
            trgt_object    = self._load_object_from_row(trgt_row),
        )

    # ── CLI hooks ─────────────────────────────────────────────────────────────

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return cfg.path_raw or cfg.root

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return cfg.path_preprocess or cfg.root

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None) -> None:
        path_raw = cls._path_raw(cfg)
        print(f"Target directory: {path_raw}")
        path_raw.mkdir(parents=True, exist_ok=True)
        if not url:
            print(
                "No --url provided.\n"
                "Expected on-disk layout after download:\n"
                f"  {path_raw}/PAM/object_meshes/<object_id>/mesh.obj\n"
                f"  {path_raw}/Meta/<object_id>.json\n"
                "Provide --url <zip-url> to download automatically."
            )
            return
        import tempfile, urllib.request, zipfile
        print(f"Downloading {url} …")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urllib.request.urlretrieve(url, tmp_path, _download_progress)
            print(f"\nExtracting to {path_raw} …")
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(path_raw)
            print("Done.")
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def index(cls, cfg, *, db: Optional[Path] = None, **kwargs) -> None:
        if cfg.item_type == ItemType.FRAME_OBJECT:
            cls._index_frames(cfg, db=db, **kwargs)
            return
        if cfg.item_type in (ItemType.OBJECT_PAIR, ItemType.FRAME_OBJECT_PAIR):
            base = "object" if cfg.item_type == ItemType.OBJECT_PAIR else "frame_object"
            print(
                f"'{cfg.item_type.value}' needs no separate indexing — pairs are derived "
                f"at load time. Run 'index -d hc3d_{base}' to build the base index.",
            )
            return
        if cfg.item_type != ItemType.OBJECT:
            print(f"ERROR: item_type '{cfg.item_type}' indexing is not implemented.", file=sys.stderr)
            sys.exit(1)
        path_raw = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)

        db_path = db or path_preprocess / "index.db"
        mesh_root = path_raw / "PAM" / "object_meshes"
        meta_root = path_raw / "Meta"

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"mesh_root       : {mesh_root}")
        print(f"meta_root       : {meta_root}")
        print(f"db              : {db_path}")

        if not mesh_root.exists():
            print(f"ERROR: {mesh_root} does not exist. Run 'fetch' first.", file=sys.stderr)
            sys.exit(1)

        meta_by_id: dict[str, dict] = {}
        if meta_root.exists():
            meta_files = [f for f in sorted(meta_root.iterdir())
                          if f.suffix in (".json", ".yaml", ".yml")]
            print(f"\nReading meta   : {len(meta_files)} file(s) in {meta_root}")
            for f in meta_files:
                _load_meta_file(f, meta_by_id, fmt="json" if f.suffix == ".json" else "yaml")
            print(f"  loaded meta for {len(meta_by_id)} object(s)")
        else:
            print(f"\nNote: {meta_root} not found — no meta columns will be added.")

        print(f"\nScanning meshes : {mesh_root}")
        mesh_entries: list[dict] = []
        no_mesh: list[str] = []
        for entry in sorted(mesh_root.iterdir()):
            if entry.is_dir():
                mesh_file = _first_mesh_in_dir(entry)
                if mesh_file is None:
                    no_mesh.append(entry.name)
                mesh_entries.append({
                    "object_id": entry.name,
                    "mesh_path": str(mesh_file.relative_to(path_raw)) if mesh_file else None,
                })
            elif entry.suffix.lower() in _MESH_EXTS:
                mesh_entries.append({
                    "object_id": entry.stem,
                    "mesh_path": str(entry.relative_to(path_raw)),
                })
        print(f"  found {len(mesh_entries)} object(s)"
              + (f", {len(no_mesh)} without a mesh file" if no_mesh else ""))

        kpts_root = path_preprocess / "obj_kpts3d"
        print(f"\nScanning kpts   : {kpts_root}")
        kpts_by_id: dict[str, tuple] = {}
        if kpts_root.exists():
            for entry in sorted(kpts_root.iterdir()):
                if not entry.is_dir():
                    continue
                kpts_file = entry / "kpts3d.pt"
                if not kpts_file.exists():
                    continue
                try:
                    t = torch.load(kpts_file, map_location="cpu")  # (K, 4)
                    kpts3d = t[:, :3].tolist()
                    mask   = t[:, 3].bool().tolist()
                    kpts_by_id[entry.name] = (json.dumps(kpts3d), json.dumps(mask))
                except Exception as exc:
                    print(f"  WARN: could not load {kpts_file}: {exc}")
            print(f"  loaded kpts for {len(kpts_by_id)} object(s)")
        else:
            print(f"  Note: {kpts_root} not found — obj_kpts3d/obj_kpts3d_mask will be NULL.")

        _META_SKIP = {"object_id", "class_name"}
        all_meta_keys: list[str] = []
        for meta in meta_by_id.values():
            for k in meta:
                if k not in all_meta_keys and k not in _META_SKIP:
                    all_meta_keys.append(k)
        print(f"\nMeta columns    : {all_meta_keys if all_meta_keys else '(none)'}")

        extra_col_defs = (
            ", " + ", ".join(f'"{k}" TEXT' for k in all_meta_keys)
            if all_meta_keys else ""
        )

        print(f"\nWriting index   : {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS objects")
        cur.execute(f"""
            CREATE TABLE objects (
                object_id       TEXT PRIMARY KEY,
                mesh_path       TEXT,
                obj_kpts3d      TEXT,
                obj_kpts3d_mask TEXT,
                category        TEXT{extra_col_defs}
            )
        """)
        for i, entry in enumerate(mesh_entries):
            oid = entry["object_id"]
            meta = meta_by_id.get(oid, {})
            kpts_json, mask_json = kpts_by_id.get(oid, (None, None))
            category = _to_text(meta.get("class_name"))
            cols = ["object_id", "mesh_path", "obj_kpts3d", "obj_kpts3d_mask", "category"] + all_meta_keys
            vals = [oid, entry["mesh_path"], kpts_json, mask_json, category] + [_to_text(meta.get(k)) for k in all_meta_keys]
            placeholders = ", ".join("?" * len(cols))
            col_names = ", ".join(f'"{c}"' for c in cols)
            cur.execute(f"INSERT INTO objects ({col_names}) VALUES ({placeholders})", vals)
            if (i + 1) % 100 == 0 or (i + 1) == len(mesh_entries):
                sys.stdout.write(f"\r  inserted {i + 1}/{len(mesh_entries)}")
                sys.stdout.flush()
        print()
        con.commit()

        con.row_factory = sqlite3.Row
        first = con.execute("SELECT * FROM objects LIMIT 1").fetchone()
        if first:
            print("\nFirst row:")
            for col in first.keys():
                val = first[col]
                display = val if val is None or len(str(val)) <= 80 else str(val)[:77] + "…"
                print(f"  {col:<30} = {display!r}")

        con.close()
        print(f"\nDone. {len(mesh_entries)} object(s) indexed -> {db_path}")

    @classmethod
    def _index_frames(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        remove: bool = False,
        max_index: Optional[int] = None,
    ) -> None:
        """Index frame-object data from ROPE/SOPE into frames.db."""
        import sqlite3 as _sq
        from o3b.dataset.housecorr3d.frame_dataset import _index_scene

        path_raw        = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)
        db_path         = db or path_preprocess / "frames.db"
        limit       = max_index or cfg.filter_count_max
        filter_kpts = cfg.filter_has_kpts
        is_real     = cfg.filter_is_real
        kpts_preprocess = path_preprocess / "obj_kpts3d"
        # When categories are requested, filter_count_max is applied PER category
        # (indexing only stops once each requested category has that many rows),
        # so a rare category isn't starved by commoner ones filling a global cap.
        categories  = set(cfg.categories) if cfg.categories else None
        cat_counts: dict = {}
        # Honour the load-time split filter while indexing, so the count_max quota
        # fills with rows the loader will actually keep (e.g. split='test' must not
        # be exhausted by 'train' scenes that are processed first).
        split_filter = None if cfg.split in (None, "all") else cfg.split

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"db              : {db_path}")

        rope_root = path_raw / "ROPE"
        sope_root = path_raw / "SOPE"
        if not rope_root.exists() and not sope_root.exists():
            print(f"ERROR: neither {rope_root} nor {sope_root} found.", file=sys.stderr)
            sys.exit(1)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        if remove and db_path.exists():
            db_path.unlink()
            print(f"Removed existing index: {db_path}")
        for suffix in ("-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)

        con = _sq.connect(db_path, timeout=60)
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        # Append to an existing index (INSERT OR IGNORE dedupes on frame_id) so
        # indexing a new category keeps previously-indexed ones; pass --remove to
        # rebuild from scratch.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id         TEXT PRIMARY KEY,
                scene_name       TEXT    NOT NULL,
                object_idx       INTEGER NOT NULL,
                mask_id          INTEGER,
                split            TEXT    NOT NULL,
                data_type        TEXT    NOT NULL,
                category         TEXT,
                object_id        TEXT,
                rgb_path         TEXT,
                depth_path       TEXT,
                mask_path        TEXT,
                cam_intr4x4      TEXT,
                cam_tform4x4_obj TEXT,
                obj_size3d       TEXT,
                obj_scale        REAL,
                has_kpts         INTEGER DEFAULT 0,
                is_valid         INTEGER DEFAULT 1
            )
        """)

        from tqdm import tqdm

        COMMIT_EVERY = 50
        rows_written = rows_loadable = scenes_since_commit = 0
        done = False

        # Legend for the two counters shown on the progress bars below, so the
        # numbers are self-explanatory in logs.
        loadable_note = "is_valid=1" + (" and has_kpts=1" if filter_kpts else "")
        print(f"\nrows_written  : frame-object rows inserted into {db_path.name}")
        print(f"rows_loadable : subset of those the loader will keep ({loadable_note}) — "
              f"this is what filter_count_max caps")

        def _scene_limit():
            # per-category cap when categories requested, else global remaining
            if categories is not None:
                return limit
            return (limit - rows_loadable) if limit else None

        def _all_categories_full() -> bool:
            return (categories is not None and limit is not None
                    and all(cat_counts.get(c, 0) >= limit for c in categories))

        def _update(n_total: int, n_match: int) -> bool:
            nonlocal rows_written, rows_loadable, scenes_since_commit
            rows_written  += n_total
            rows_loadable += n_match
            scenes_since_commit += 1
            if scenes_since_commit >= COMMIT_EVERY:
                con.commit()
                scenes_since_commit = 0
            if categories is not None:
                return _all_categories_full()
            return bool(limit and rows_loadable >= limit)

        if rope_root.exists() and not done and is_real is not False and split_filter in (None, "test"):
            scene_dirs = sorted(d for d in rope_root.iterdir() if d.is_dir())
            print(f"\nIndexing ROPE ({len(scene_dirs)} scenes, split=test, data_type=real)")
            bar = tqdm(scene_dirs, unit="scene", desc="ROPE")
            for scene_dir in bar:
                n_total, n_match = _index_scene(
                    cur=cur, scene_dir=scene_dir, scene_name=scene_dir.name,
                    split="test", data_type="real", path_raw=path_raw,
                    kpts_preprocess=kpts_preprocess, limit=_scene_limit(), filter_kpts=filter_kpts,
                    categories=categories, cat_counts=cat_counts,
                )
                bar.set_postfix(rows_written=rows_written, rows_loadable=rows_loadable)
                if _update(n_total, n_match):
                    done = True; bar.close(); break

        if sope_root.exists() and not done and is_real is not True:
            patch_dirs = sorted(d for d in sope_root.iterdir() if d.is_dir())

            def _sope_scenes():
                for patch_dir in patch_dirs:
                    for split_name in ("train", "test"):
                        split_dir = patch_dir / split_name
                        if not split_dir.exists():
                            continue
                        for kind_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
                            for scene_dir in sorted(d for d in kind_dir.iterdir() if d.is_dir()):
                                yield scene_dir, f"{patch_dir.name}_{scene_dir.name}", split_name

            # materialise so the progress bar can show total / remaining scenes
            # (and drop scenes whose split the loader would filter out)
            sope_scenes = [s for s in _sope_scenes() if split_filter is None or s[2] == split_filter]
            print(f"\nIndexing SOPE ({len(patch_dirs)} patches, {len(sope_scenes)} scenes, "
                  f"split={split_filter or 'all'}, data_type=synthetic)")
            bar = tqdm(sope_scenes, unit="scene", desc="SOPE")
            for scene_dir, scene_name, split_name in bar:
                n_total, n_match = _index_scene(
                    cur=cur, scene_dir=scene_dir, scene_name=scene_name,
                    split=split_name, data_type="synthetic", path_raw=path_raw,
                    kpts_preprocess=kpts_preprocess, limit=_scene_limit(), filter_kpts=filter_kpts,
                    categories=categories, cat_counts=cat_counts,
                )
                bar.set_postfix(rows_written=rows_written, rows_loadable=rows_loadable)
                if _update(n_total, n_match):
                    bar.close(); break

        con.commit()
        con.close()
        print(f"\nDone. {rows_written} frame-object rows written, "
              f"{rows_loadable} of them loadable ({loadable_note}) → {db_path}")

    @classmethod
    def visualize(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        render_frames: int = 0,
        renderer: str = "pyrender",
        debug: bool = False,
        obj_centric: bool = False,
        **_,
    ) -> None:
        if cfg.item_type == ItemType.FRAME_OBJECT:
            cls._visualize_frame_objects(cfg, db=db, limit=limit, object_id=object_id,
                                          render=render, debug=debug, obj_centric=obj_centric)
            return

        if cfg.item_type == ItemType.FRAME_OBJECT_PAIR:
            cls._visualize_frame_object_pairs(cfg, db=db, limit=limit, object_id=object_id,
                                              debug=debug, obj_centric=obj_centric)
            return

        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "index.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        dataset = cls(cfg)
        if not dataset._object_rows_id:
            print("No objects found in index"
                  + (" with keypoints available." if cfg.filter_has_kpts else "."))
            return

        total = len(dataset._object_rows_id)

        if object_id:
            if cfg.item_type == ItemType.OBJECT_PAIR:
                dataset._object_rows_id = [
                    (si, ti) for si, ti in dataset._object_rows_id
                    if dataset._object_rows[si]["object_id"] == object_id
                    or dataset._object_rows[ti]["object_id"] == object_id
                ]
            else:
                dataset._object_rows_id = [
                    i for i in dataset._object_rows_id
                    if dataset._object_rows[i]["object_id"] == object_id
                ]
        if limit < len(dataset._object_rows_id):
            dataset._object_rows_id = dataset._object_rows_id[:limit]

        filter_note = "  [filtered: has kpts]" if cfg.filter_has_kpts else ""
        print(f"Showing {len(dataset._object_rows_id)} / {total} objects{filter_note}  {db_path}\n")
        for entry in dataset._object_rows_id:
            if cfg.item_type == ItemType.OBJECT_PAIR:
                src_row  = dataset._object_rows[entry[0]]
                trgt_row = dataset._object_rows[entry[1]]
                mask_json = src_row.get("obj_kpts3d_mask")
                if mask_json:
                    mask = json.loads(mask_json)
                    kpts_info = f"  kpts={sum(mask)}/{len(mask)}"
                else:
                    kpts_info = "  kpts=none"
                print(f"{src_row['object_id']}  <--->  {trgt_row['object_id']}{kpts_info}")
            else:
                row = dataset._object_rows[entry]
                mask_json = row.get("obj_kpts3d_mask")
                if mask_json:
                    mask = json.loads(mask_json)
                    kpts_info = f"  kpts={sum(mask)}/{len(mask)}"
                else:
                    kpts_info = "  kpts=none"
                print(f"{row['object_id']}{kpts_info}")

        from o3b.data.viz import visualize_dataset
        visualize_dataset(dataset, render=render, render_frames=render_frames, renderer=renderer, debug=debug)

    @classmethod
    def _visualize_frame_objects(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        debug: bool = False,
        obj_centric: bool = False,
    ) -> None:
        from o3b.dataset.housecorr3d.frame_dataset import _visualize_frame_objects_viser
        from dataclasses import replace as _r

        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "frames.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        # Load dataset with all modalities for visualization
        viz_cfg = _r(cfg, modalities=None, object_modalities=None)
        dataset = cls(viz_cfg)
        if not dataset._frame_rows_id:
            print("No frames found matching the current config filters.")
            return

        total = len(dataset._frame_rows_id)
        if object_id:
            dataset._frame_rows_id = [
                i for i in dataset._frame_rows_id
                if dataset._frame_rows[i].get("object_id") == object_id
            ]
        if limit < len(dataset._frame_rows_id):
            dataset._frame_rows_id = dataset._frame_rows_id[:limit]

        print(f"Showing {len(dataset._frame_rows_id)} / {total} frames  {db_path}\n")

        _visualize_frame_objects_viser(dataset, debug=debug, obj_centric=obj_centric)

    @classmethod
    def _visualize_frame_object_pairs(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        debug: bool = False,
        obj_centric: bool = False,
    ) -> None:
        from o3b.dataset.housecorr3d.frame_dataset import _visualize_frame_object_pairs_viser
        from dataclasses import replace as _r

        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "frames.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        # Load all modalities for visualization
        viz_cfg = _r(cfg, modalities=None, object_modalities=None)
        dataset = cls(viz_cfg)
        if not dataset._frame_pairs_id:
            print("No frame-object pairs found matching the current config filters.")
            return

        total = len(dataset._frame_pairs_id)
        if object_id:
            dataset._frame_pairs_id = [
                (a, b) for (a, b) in dataset._frame_pairs_id
                if dataset._frame_rows[a].get("object_id") == object_id
                or dataset._frame_rows[b].get("object_id") == object_id
            ]
        if limit < len(dataset._frame_pairs_id):
            dataset._frame_pairs_id = dataset._frame_pairs_id[:limit]

        print(f"Showing {len(dataset._frame_pairs_id)} / {total} frame pairs  {db_path}\n")

        _visualize_frame_object_pairs_viser(dataset, debug=debug, obj_centric=obj_centric)

    # ── item loading ──────────────────────────────────────────────────────────

    def _sharded_drop_reason(self, idx: int) -> str:
        """Name the missing input behind a dropped frame-object.

        Frame items are dropped by CropCamBBox2D when cam_bbox2d is None, which
        traces back to the instance mask — absent from the index, missing on
        disk, undecodable (cv2 without EXR support), or empty (occluded).
        Categories, not paths, so the build tally aggregates.
        """
        if self.cfg.item_type != ItemType.FRAME_OBJECT:
            return super()._sharded_drop_reason(idx)

        row = self._frame_rows[self._frame_rows_id[idx]]
        if not row.get("mask_path") or row.get("mask_id") is None:
            return "no mask_path/mask_id in frames.db"
        mask_path = self.path_raw / row["mask_path"]
        if not mask_path.exists():
            return "mask file missing on disk"

        from o3b.dataset.housecorr3d.frame_dataset import _load_mask_tensor
        mask = _load_mask_tensor(mask_path, int(row["mask_id"]))
        if mask is None:
            return "mask file unreadable (cv2 EXR decode failed?)"
        if not bool(mask.any()):
            return "object not visible in frame (empty mask)"

        from o3b.cv.visual.draw import get_bboxs_from_masks
        bbox = get_bboxs_from_masks(mask[None])[0].float()
        if float(bbox[2] - bbox[0]) < 1.0 or float(bbox[3] - bbox[1]) < 1.0:
            return "object smaller than 1 px in frame"
        return super()._sharded_drop_reason(idx)

    def _load_frame_object(self, idx: int) -> FrameObject:
        """Load one frame-object from frames.db.

        Object geometry (mesh, kpts, tform) is delegated to _load_object_from_row,
        which applies obj_gl_tform4x4_obj_raw, mesh_type conversion, and normalises kpts to
        NCDS space.  obj_size is rescaled from mesh units to metres via obj_scale.
        cam_tform4x4_obj_ncds = cam_tform4x4_obj_metric @ obj_ncds0c_tform4x4_obj
        maps NCDS → camera space in metric units.
        """
        return self._load_frame_object_by_rowidx(self._frame_rows_id[idx])

    def _load_frame_object_pair(self, idx: int) -> "FrameObjectPair":
        """Load a frame-object pair: query (src) and target (trgt) FrameObjects."""
        from o3b.data.datatypes.frame_object import FrameObjectPair
        src_rowidx, trgt_rowidx = self._frame_pairs_id[idx]
        src_fo  = self._load_frame_object_by_rowidx(src_rowidx)
        trgt_fo = self._load_frame_object_by_rowidx(trgt_rowidx)
        return FrameObjectPair(
            src_object_id  = src_fo.object_id,
            trgt_object_id = trgt_fo.object_id,
            src_object     = src_fo,
            trgt_object    = trgt_fo,
        )

    def _load_frame_object_by_rowidx(self, row_idx: int) -> FrameObject:
        from o3b.dataset.housecorr3d.frame_dataset import (
            _load_image_tensor, _load_depth_tensor, _load_mask_tensor,
        )

        row  = self._frame_rows[row_idx]
        mods = self.cfg.modalities
        oid       = row.get("object_id", "")
        obj_scale = float(row.get("obj_scale") or 1.0)

        # ── Object first: mesh, tform, kpts, bbox3d ───────────────────────────
        obj_row = self._obj_by_id.get(oid)
        obj     = self._load_object_from_row(obj_row) if obj_row is not None else None
        tform   = obj.obj_ncds0c_tform4x4_obj if obj is not None else None

        # ── frame modalities ──────────────────────────────────────────────────
        rgb = _load_image_tensor(self.path_raw / row["rgb_path"]) \
            if _want("rgb", mods) and row.get("rgb_path") else None
        depth = _load_depth_tensor(self.path_raw / row["depth_path"]) \
            if _want("depth", mods) and row.get("depth_path") else None
        depth_mask = (depth > 0) if (_want("depth_mask", mods) and depth is not None) else None
        mask = _load_mask_tensor(self.path_raw / row["mask_path"], int(row["mask_id"])) \
            if _want("fo_mask", mods) and row.get("mask_path") and row.get("mask_id") is not None else None

        # Distance transform of the object mask, normalised to [0, 1] by the larger
        # image side.  Computed here so it can be baked into the shards; when a
        # crop transform follows, CropCamBBox2D recomputes it from the cropped mask
        # (distances live in the pixel grid of the image they describe).
        fo_mask_dt = None
        if _want("fo_mask_dt", mods) and mask is not None:
            from o3b.cv.visual.mask import get_mask_distance_transform_norm
            fo_mask_dt = get_mask_distance_transform_norm(mask.bool().cpu()).float()

        cam_intr4x4 = (torch.tensor(json.loads(row["cam_intr4x4"]), dtype=torch.float32)
                       if _want("cam_intr4x4", mods) and row.get("cam_intr4x4") else None)

        cam_bbox2d = None
        # mask.any() guard: an object fully occluded in this frame has no pixels,
        # and get_bboxs_from_masks answers that with the *whole image* box (its
        # except-branch fallback), which would silently bake a full-scene crop
        # into the shards.  Leaving cam_bbox2d None drops the frame instead.
        if _want("cam_bbox2d", mods) and mask is not None and bool(mask.any()):
            from o3b.cv.visual.draw import get_bboxs_from_masks
            cam_bbox2d = get_bboxs_from_masks(mask[None])[0].float()

        # ── camera pose ───────────────────────────────────────────────────────
        # Always load — needed for cam_tform4x4_obj_ncds even when not directly requested.
        cam_tform4x4_obj_raw = (torch.tensor(json.loads(row["cam_tform4x4_obj"]), dtype=torch.float32)
                                if row.get("cam_tform4x4_obj") else None)

        # Left-multiply: camera-frame convention correction (e.g. CV → OpenGL flip).
        if cam_tform4x4_obj_raw is not None and self.cfg.cam_tform4x4_cam_raw is not None:
            C = torch.tensor(self.cfg.cam_tform4x4_cam_raw, dtype=torch.float32)
            cam_tform4x4_obj_raw = C @ cam_tform4x4_obj_raw

        # SVD-normalise rotation, embed obj_scale.
        cam_tform4x4_obj_metric = None
        if cam_tform4x4_obj_raw is not None:
            U, _, Vh = torch.linalg.svd(cam_tform4x4_obj_raw[:3, :3].float())
            cam_tform4x4_obj_metric = cam_tform4x4_obj_raw.clone().float()
            cam_tform4x4_obj_metric[:3, :3] = (U @ Vh) * obj_scale

        # obj_gl_tform4x4_obj_raw rotates the metric object frame by T into the canonical
        # orientation (applied on both sides of obj_ncds0c_tform4x4_obj in
        # _load_object_from_row).  To keep the object projecting to the same pixels,
        # the pose must map canonical-metric → cam, i.e. cam_tform4x4_obj @ inv(T).
        if cam_tform4x4_obj_metric is not None and self.cfg.obj_gl_tform4x4_obj_raw is not None:
            from o3b.cv.geometry.transform import inv_tform4x4
            T = torch.tensor(self.cfg.obj_gl_tform4x4_obj_raw, dtype=torch.float32)
            cam_tform4x4_obj_metric = cam_tform4x4_obj_metric @ inv_tform4x4(T)

        cam_tform4x4_obj = cam_tform4x4_obj_metric if _want("cam_tform4x4_obj", mods) else None

        cam_tform4x4_obj_ncds = None
        obj_size_ncds = None
        obj_size      = None
        if tform is not None:
            obj_size_ncds = 2.0
            # _load_object_from_row gives obj_size in mesh units; scale to metres
            obj_size = (obj.obj_size * obj_scale) if obj.obj_size is not None else None
            if cam_tform4x4_obj_metric is not None and _want("cam_tform4x4_obj_ncds", mods):
                cam_tform4x4_obj_ncds = cam_tform4x4_obj_metric @ tform.float()

        # cam_bbox3d: transform obj_bbox3d (metric object space) to camera space.
        cam_bbox3d = None
        if (_want("cam_bbox3d", mods) and obj is not None and obj.obj_bbox3d is not None
                and cam_tform4x4_obj_metric is not None):
            from o3b.cv.geometry.transform import transf3d_broadcast as _t3d
            cam_bbox3d = _t3d(obj.obj_bbox3d.float(), cam_tform4x4_obj_metric)  # (8, 3)

        # ── amodal object mask (+ its DT) ─────────────────────────────────────
        # Rendered from the GT mesh under the GT pose, so unlike fo_mask it also
        # covers what other objects occlude.  Baked in here (and recomputed by
        # CropCamBBox2D after a crop) so training reads it instead of rasterising
        # the GT mesh every batch — see CamGeo2D3D.use_mask_amodal.
        fo_mask_amodal, fo_mask_amodal_dt = None, None
        want_amodal    = _want("fo_mask_amodal", mods)
        want_amodal_dt = _want("fo_mask_amodal_dt", mods)
        if ((want_amodal or want_amodal_dt) and obj is not None and obj.mesh is not None
                and cam_tform4x4_obj_metric is not None and tform is not None
                and row.get("cam_intr4x4")):
            fo_mask_amodal = self._render_fo_mask_amodal(
                row, obj, cam_tform4x4_obj_metric, tform, rgb=rgb, mask=mask,
            )
            if fo_mask_amodal is not None and want_amodal_dt:
                from o3b.cv.visual.mask import get_mask_distance_transform_norm
                fo_mask_amodal_dt = get_mask_distance_transform_norm(
                    fo_mask_amodal,
                ).float()
            if not want_amodal:
                fo_mask_amodal = None

        # ── occlusion-aware 2-D keypoint visibility ───────────────────────────
        obj_kpts2d_mask = None
        if (_want("obj_kpts2d_mask", mods) and obj is not None and obj.mesh is not None
                and obj.obj_kpts3d is not None and cam_tform4x4_obj_metric is not None
                and tform is not None and row.get("cam_intr4x4")):
            obj_kpts2d_mask = self._compute_obj_kpts2d_mask(
                row, obj, cam_tform4x4_obj_metric, tform, rgb=rgb, mask=mask,
            )

        return FrameObject(
            frame_id                = row.get("frame_id", ""),
            frame_object_id         = row.get("frame_id", ""),
            object_id               = oid,
            rgb                     = rgb,
            depth                   = depth,
            depth_mask              = depth_mask,
            fo_mask                 = mask,
            fo_mask_dt              = fo_mask_dt,
            fo_mask_amodal          = fo_mask_amodal,
            fo_mask_amodal_dt       = fo_mask_amodal_dt,
            cam_intr4x4             = cam_intr4x4,
            cam_tform4x4_obj        = cam_tform4x4_obj,
            cam_tform4x4_obj_ncds   = cam_tform4x4_obj_ncds,
            cam_bbox2d              = cam_bbox2d,
            cam_bbox3d              = cam_bbox3d,
            obj_bbox3d              = (obj.obj_bbox3d if (obj is not None and _want("obj_bbox3d", mods)) else None),
            mesh                    = obj.mesh if obj is not None else None,
            obj_ncds0c_tform4x4_obj = tform if _want("obj_ncds0c_tform4x4_obj", mods) else None,
            obj_size_ncds           = obj_size_ncds,
            obj_size                = obj_size,
            obj_kpts3d              = obj.obj_kpts3d      if obj is not None else None,
            obj_kpts3d_mask         = obj.obj_kpts3d_mask if obj is not None else None,
            obj_kpts2d_mask         = obj_kpts2d_mask,
            obj_syms                = (obj.obj_syms        if (obj is not None and _want("obj_syms", mods)) else None),
            obj_kpts3d_syms         = (obj.obj_kpts3d_syms if (obj is not None and _want("obj_syms", mods)) else None),
            obj_axis6d_sym          = (obj.obj_axis6d_sym  if (obj is not None and _want("obj_syms", mods)) else None),
            category                = row.get("category") if _want("category", mods) else None,
        )

    # ── occlusion-aware keypoint visibility helpers ───────────────────────────

    def _cam_tform4x4_obj_metric(self, row) -> "Optional[torch.Tensor]":
        """Canonical-metric cam←obj SE(3) for a frame row (SVD-normalised R, obj_scale
        embedded, obj_gl_tform4x4_obj_raw applied as @ inv(T) to match _load_object_from_row)."""
        raw = row.get("cam_tform4x4_obj")
        if not raw:
            return None
        M = torch.tensor(json.loads(raw), dtype=torch.float32)
        if self.cfg.cam_tform4x4_cam_raw is not None:
            C = torch.tensor(self.cfg.cam_tform4x4_cam_raw, dtype=torch.float32)
            M = C @ M
        obj_scale = float(row.get("obj_scale") or 1.0)
        U, _, Vh = torch.linalg.svd(M[:3, :3].float())
        out = M.clone().float()
        out[:3, :3] = (U @ Vh) * obj_scale
        if self.cfg.obj_gl_tform4x4_obj_raw is not None:
            from o3b.cv.geometry.transform import inv_tform4x4
            T = torch.tensor(self.cfg.obj_gl_tform4x4_obj_raw, dtype=torch.float32)
            out = out @ inv_tform4x4(T)
        return out

    def _obj_cam_mesh(self, row) -> "Optional[tuple]":
        """Return (verts_cam (N,3), faces (F,3)) for a frame row's object in camera space."""
        oid = row.get("object_id", "")
        obj_row = self._obj_by_id.get(oid)
        if obj_row is None:
            return None
        metric = self._cam_tform4x4_obj_metric(row)
        if metric is None:
            return None
        obj = self._load_object_from_row(obj_row)
        if obj is None or obj.mesh is None:
            return None
        tform = obj.obj_ncds0c_tform4x4_obj
        ncds = (metric @ tform.float()) if tform is not None else metric
        R, t = ncds[:3, :3], ncds[:3, 3]
        verts_cam = (R @ obj.mesh.verts.float().t()).t() + t
        return verts_cam, obj.mesh.faces

    def _frame_siblings(self, row) -> list:
        """All valid object rows sharing the same scene frame as *row* (incl. itself)."""
        if getattr(self, "_frame_to_rows", None) is None:
            m: dict[str, list] = {}
            for r in self._frame_rows:
                if not int(r.get("is_valid", 1) or 0):
                    continue
                key = r.get("frame_id", "").rsplit("/", 1)[0]
                m.setdefault(key, []).append(r)
            self._frame_to_rows = m
        key = row.get("frame_id", "").rsplit("/", 1)[0]
        return self._frame_to_rows.get(key, [row])

    def _frame_image_hw(self, rgb, mask, intr) -> tuple:
        """Full-image (H, W): prefer a loaded modality, then cfg.extra, then intrinsics."""
        if rgb is not None:
            return int(rgb.shape[-2]), int(rgb.shape[-1])
        if mask is not None:
            return int(mask.shape[-2]), int(mask.shape[-1])
        img_size = (self.cfg.extra or {}).get("image_size")
        if img_size:
            return int(img_size[0]), int(img_size[1])
        if intr is not None:
            return int(round(2 * float(intr[1, 2]))), int(round(2 * float(intr[0, 2])))
        return None, None

    def _render_fo_mask_amodal(self, row, obj, cam_tform4x4_obj_metric, tform,
                               rgb=None, mask=None) -> "Optional[torch.Tensor]":
        """Amodal instance mask: this object's GT mesh alone, under the GT pose.

        Only the target object is rasterised, so occluders leave no hole — the
        result is the object's full silhouette, cut off only by the image
        border.  Same transform chain as the scene render used for keypoint
        visibility (metric cam←obj composed with obj_ncds0c_tform4x4_obj), so
        the mask lands on the same pixels as fo_mask.

        Returns (H, W) bool at full-frame resolution, or None when the frame
        has no usable intrinsics / image size or the render fails.
        """
        from o3b.dataset.housecorr3d.frame_dataset import render_scene_depth

        intr_full = torch.tensor(json.loads(row["cam_intr4x4"]), dtype=torch.float32)
        H, W = self._frame_image_hw(rgb, mask, intr_full)
        if H is None:
            return None

        ncds = cam_tform4x4_obj_metric @ tform.float()
        R, t = ncds[:3, :3], ncds[:3, 3]
        verts_cam = (R @ obj.mesh.verts.float().t()).t() + t

        depth = render_scene_depth([(verts_cam, obj.mesh.faces)], intr_full, H, W)
        if depth is None:
            return None
        return depth > 0  # 0 = no hit → background

    def _compute_obj_kpts2d_mask(self, row, obj, cam_tform4x4_obj_metric, tform,
                                 rgb=None, mask=None) -> "Optional[torch.Tensor]":
        """Occlusion-aware per-keypoint 2-D visibility for the target object.

        Renders all objects in the frame to a depth buffer (cached per frame) to
        determine which of the target mesh's vertices are the front-most surface
        (visible). A keypoint is then visible iff a visible vertex lies within
        0.1 * normalised object size of it (NCDS space), and only where the
        keypoint annotation (obj_kpts3d_mask) is valid.
        """
        from o3b.dataset.housecorr3d.frame_dataset import (
            render_scene_depth, visible_vertices_from_render,
            kpts2d_mask_from_visible_verts,
        )

        intr_full = torch.tensor(json.loads(row["cam_intr4x4"]), dtype=torch.float32)
        H, W = self._frame_image_hw(rgb, mask, intr_full)
        if H is None:
            return None

        # rendered scene depth for this frame, shared across its objects
        key = row.get("frame_id", "").rsplit("/", 1)[0]
        cache = getattr(self, "_scene_depth_cache", None)
        if cache is None:
            cache = self._scene_depth_cache = {}
        if key in cache:
            depth_render = cache[key]
        else:
            meshes_cam = [g for g in (self._obj_cam_mesh(r) for r in self._frame_siblings(row))
                          if g is not None]
            depth_render = render_scene_depth(meshes_cam, intr_full, H, W)
            if len(cache) >= 8:
                cache.clear()
            cache[key] = depth_render
        if depth_render is None:
            return None

        # target mesh verts in camera space (same transform chain as the scene render)
        ncds = cam_tform4x4_obj_metric @ tform.float()
        R, t = ncds[:3, :3], ncds[:3, 3]
        verts_cam = (R @ obj.mesh.verts.float().t()).t() + t

        obj_scale = float(row.get("obj_scale") or 1.0)
        obj_size  = (obj.obj_size * obj_scale) if obj.obj_size is not None else None

        # which vertices are visible (front-most surface in the all-objects render)
        visible_verts = visible_vertices_from_render(
            verts_cam, depth_render, intr_full, H, W, obj_size=obj_size,
        )
        # keypoint visible iff a visible vertex is nearby (NCDS), and annotated
        return kpts2d_mask_from_visible_verts(
            kpts_ncds       = obj.obj_kpts3d,
            verts_ncds      = obj.mesh.verts,
            visible_verts   = visible_verts,
            obj_kpts3d_mask = obj.obj_kpts3d_mask,
            norm_size       = 2.0,   # obj_size_ncds
            rel_radius      = 0.05,
        )

    def _load_scene_object(self, idx: int) -> SceneObject:
        row = self._object_rows[self._object_rows_id[idx]]
        obj_id = row["object_id"]
        mesh_rel = row.get("mesh_path")
        entry = self.path_raw / mesh_rel if mesh_rel else self.path_object_meshes / obj_id
        T = self.cfg.scene_length
        frame_indices = _sample_frame_indices(entry, T)

        fo_list = [self._load_frame(entry, i) for i in frame_indices]
        return SceneObject.from_frame_objects(
            fo_list,
            scene_object_id = obj_id,
            scene_id        = obj_id,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()


def _load_meta_file(path: Path, store: dict, fmt: str = "json") -> None:
    try:
        if fmt == "json":
            data = json.loads(path.read_text())
        else:
            import yaml
            data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and "instance_dict" in data:
            for obj_id, obj_meta in data["instance_dict"].items():
                store[obj_id] = obj_meta if isinstance(obj_meta, dict) else {"value": str(obj_meta)}
        else:
            store[path.stem] = data if isinstance(data, dict) else {"value": str(data)}
    except Exception as exc:
        print(f"  WARN: could not parse {path}: {exc}")


def _first_mesh_in_dir(directory: Path) -> Optional[Path]:
    for ext in (".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"):
        hits = sorted(directory.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def _to_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _want(name: str, mods: Optional[set]) -> bool:
    return mods is None or name in mods


def _n_frames(entry: Path) -> int:
    if entry.is_dir():
        return max(1, len(list(entry.glob("*.png"))) + len(list(entry.glob("*.jpg"))))
    return 1


def _sample_frame_indices(entry: Path, T: int) -> list[int]:
    n = _n_frames(entry)
    step = max(1, n // T)
    return list(range(0, min(n, T * step), step))[:T]


def _load_rgb(_entry: Path, _frame_idx: int):
    return None

def _load_depth(_entry: Path, _frame_idx: int):
    return None

def _load_intr(_entry: Path):
    return None

def _load_bbox2d(_entry: Path, _frame_idx: int):
    return None

def _load_pose(_entry: Path, _frame_idx: int):
    return None

def _load_category(_entry: Path):
    return None


_SYM_AXIS_CODE = {"none": 1.0, "any": -1.0, "half": 2.0, "quarter": 4.0}


def _load_obj_syms_from_row(row: dict) -> Optional[torch.Tensor]:
    """(3,) symmetry code, in the mesh's on-disk axis order, from the object's
    Omni6DPose/cutoop `tag.symmetry` meta column (populated generically by the
    Meta/*.json -> "tag" index column already built by `index()`), or None when
    absent. Codes: -1 continuous, 1 none, 2 half (180 deg), 4 quarter (90 deg).
    This is the *raw* mesh-frame order — obj_gl_tform4x4_obj_raw's permutation is
    applied to the derived obj_kpts3d_syms / obj_axis6d_sym (not to this code
    itself) via Object.transform(), same as verts/kpts.
    """
    tag_json = row.get("tag")
    if not tag_json:
        return None
    try:
        sym = json.loads(tag_json).get("symmetry")
    except Exception:
        return None
    if not isinstance(sym, dict):
        return None
    if sym.get("any"):
        return torch.tensor([-1.0, -1.0, -1.0])
    return torch.tensor([_SYM_AXIS_CODE.get(sym.get(ax), 1.0) for ax in ("x", "y", "z")])


def _compute_obj_sym_geometry(
    obj_syms: "Optional[torch.Tensor]",
    obj_kpts3d: "Optional[torch.Tensor]",
):
    """Derive visualization/evaluation-ready symmetry geometry from a raw (3,)
    obj_syms code, in the same space as obj_kpts3d:

      - obj_kpts3d_syms (K, S, 3): discrete-symmetric keypoint candidates
        (candidate 0 is always the identity).
      - obj_axis6d_sym (6,): continuous-rotation axis (offset, direction), when
        exactly one continuous ("-1") axis is annotated.

    Both are plain geometric quantities (points / offset+direction) so
    Object.transform() can keep them correctly posed under any subsequent
    rigid transform. Returns (None, None) when obj_syms is None.
    """
    if obj_syms is None:
        return None, None
    from o3b.cv.metric.pose import get_obj_tform4x4_obj_sym, get_obj_axis6d_with_mask

    kpts3d_syms = None
    if obj_kpts3d is not None:
        sym_tforms = get_obj_tform4x4_obj_sym(obj_syms.float())          # (S, 4, 4)
        R, t = sym_tforms[:, :3, :3], sym_tforms[:, :3, 3]
        kpts3d_syms = torch.einsum("sij,kj->ksi", R, obj_kpts3d.float()) + t[None]  # (K, S, 3)

    axis6d_sym = None
    axis6d, mask = get_obj_axis6d_with_mask(obj_syms.float())            # (3, 6), (3,) bool
    if mask.sum() == 1:
        axis6d_sym = axis6d[mask][0].clone()                              # (6,)

    return kpts3d_syms, axis6d_sym


def _load_kpts3d_by_id(
    obj_id: str,
    path_preprocess: Path,
    tform: Optional[torch.Tensor],
):
    kpts_file = path_preprocess / "obj_kpts3d" / obj_id / "kpts3d.pt"
    if not kpts_file.exists():
        return None, None
    try:
        t = torch.load(kpts_file, map_location="cpu")  # (K, 4)
        kpts3d = t[:, :3].float()
        mask   = t[:, 3].bool()
        if tform is not None:
            scale  = tform[0, 0]
            center = tform[:3, 3]
            kpts3d = (kpts3d - center) / scale
        return kpts3d, mask
    except Exception:
        return None, None
