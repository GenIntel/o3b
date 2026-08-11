from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch
from o3b.dataset.dataset import register_dataset, ConfigurableDataset, ItemType
from o3b.data.modalities import Object, ObjectPair
from o3b.dataset.densematcher.enum import DENSEMATCHER_CATEGORIES

_MESH_EXTS = {".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"}

_DEFAULT_URL = "https://uc5bd5477daf057f29a9be114ebe.dl.dropboxusercontent.com/zip_download_get/Cl3YTCzPess6IGcEaFENNRsKiVD3tlplOVlH5WciuMYDo5ZPPB1-fC1BvJllNFGOUkkL0oPz9C-MwtQMyxToNZ-vPeKeJJ4-eVZ62tA_CTxkMQ?_download_id=568191518838943791998429792307572148146857030313156947310723491&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1"
# note: could be that this url changes all the time for densematcher. 

@register_dataset("DenseMatcher")
class DenseMatcher(ConfigurableDataset):

    categories: tuple = tuple(DENSEMATCHER_CATEGORIES)

    @property
    def path_raw(self) -> Path:
        return self.cfg.path_raw

    @property
    def path_preprocess(self) -> Path:
        return self.cfg.path_preprocess

    @property
    def path_object_meshes(self) -> Path:
        return self.path_raw / "object_meshes"

    def __len__(self):
        return len(self._object_rows_id)

    def _setup(self) -> None:
        self._object_rows: list[dict] = []
        self._object_rows_id: list = []
        db_path = self.path_preprocess / "index.db"
        if not db_path.exists():
            return
        con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()
            self._object_rows = [dict(r) for r in cur.execute("SELECT * FROM objects").fetchall()]
            _id_to_idx: dict[str, int] = {r["object_id"]: i for i, r in enumerate(self._object_rows)}
            cats    = self.cfg.categories
            subsets = self.cfg.subsets

            def _build_clauses(prefix: str = "") -> tuple[str, list]:
                """Return (WHERE clauses string, params list) for optional filters."""
                clauses, params = "", []
                kpts_col    = f"{prefix}obj_kpts3d"
                cat_col     = f"{prefix}category"
                subset_col  = f"{prefix}subset"
                if self.cfg.filter_has_kpts:
                    clauses += f" AND {kpts_col} IS NOT NULL"
                if cats:
                    ph = ", ".join("?" * len(cats))
                    clauses += f" AND {cat_col} IN ({ph})"
                    params   += list(cats)
                if subsets:
                    ph = ", ".join("?" * len(subsets))
                    clauses += f" AND {subset_col} IN ({ph})"
                    params   += list(subsets)
                return clauses, params

            if self.cfg.item_type == ItemType.OBJECT:
                clauses, params = _build_clauses()
                limit_clause = f" LIMIT {self.cfg.filter_count_max}" if self.cfg.filter_count_max else ""
                rows = cur.execute(
                    f"SELECT object_id FROM objects WHERE 1=1{clauses}{limit_clause}",
                    params,
                ).fetchall()
                self._object_rows_id = [_id_to_idx[r["object_id"]] for r in rows]

            elif self.cfg.item_type == ItemType.OBJECT_PAIR:
                # Pairs are derived on the fly (all same-category combinations);
                # no separate `object_pairs` index table is required.
                src_clauses, src_params   = _build_clauses(prefix="a.")
                trgt_clauses, trgt_params = _build_clauses(prefix="b.")
                all_clauses = src_clauses + trgt_clauses
                all_params  = src_params  + trgt_params
                limit_clause = f" LIMIT {self.cfg.filter_count_max}" if self.cfg.filter_count_max else ""
                rows = cur.execute(f"""
                    SELECT a.object_id AS src_object_id, b.object_id AS trgt_object_id
                    FROM objects a
                    JOIN objects b ON a.category IS b.category AND a.object_id != b.object_id
                    WHERE a.category IS NOT NULL{all_clauses}
                    ORDER BY a.object_id, b.object_id{limit_clause}
                """, all_params).fetchall()
                self._object_rows_id = [
                    (_id_to_idx[r["src_object_id"]], _id_to_idx[r["trgt_object_id"]])
                    for r in rows
                ]
        finally:
            con.close()

    def _load_object(self, idx: int) -> Object:
        return self._load_object_from_row(self._object_rows[self._object_rows_id[idx]])

    def _load_object_from_row(self, row: dict) -> Object:
        from o3b.io import _load_mesh
        from o3b.dataset.utils import want as _want, load_kpts3d_by_id as _load_kpts3d_by_id

        oid  = row["object_id"]
        mods = self.cfg.object_modalities

        need_mesh           = _want("mesh",                    mods)
        need_verts3d        = _want("verts3d",                 mods)
        need_verts3d_feats  = _want("verts3d_feats",           mods)
        need_tform          = _want("obj_ncds0c_tform4x4_obj", mods)
        need_kpts           = _want("obj_kpts3d",              mods)
        need_category       = _want("category",                mods)
        need_part_id        = _want("obj_verts_part_id",       mods)

        mesh, tform = None, None
        if need_mesh or need_verts3d or need_verts3d_feats or need_tform or need_kpts or need_part_id:
            mesh_rel   = row.get("mesh_path")
            mesh_entry = self.path_raw / mesh_rel if mesh_rel else self.path_object_meshes / oid
            mesh, tform = _load_mesh(mesh_entry)

            if (need_mesh or need_verts3d or need_verts3d_feats) and \
                    self.cfg.mesh_type != "default" and mesh is not None:
                from o3b.data.datatypes.mesh import Mesh
                converted_path = self.path_preprocess / "mesh" / self.cfg.mesh_type / f"{oid}.glb"
                mesh = Mesh.load_or_convert(converted_path, mesh_entry, self.cfg.mesh_type)

        obj_verts_part_id = None
        if need_part_id and mesh is not None:
            mesh_rel = row.get("mesh_path")
            obj_dir  = (self.path_raw / mesh_rel).parent if mesh_rel else self.path_object_meshes / oid
            if self.cfg.mesh_type == "default":
                # simple_mesh.obj is the loaded mesh — groups.txt indices apply directly
                obj_verts_part_id = _load_groups_txt(obj_dir, n_verts=len(mesh.verts))
            else:
                # Preprocessed mesh has different vertices; remap via nearest simple_mesh vertex
                simple_file = obj_dir / "simple_mesh.obj"
                if simple_file.exists():
                    from o3b.io import _load_mesh as _lm
                    simple_mesh, _ = _lm(simple_file)
                    if simple_mesh is not None:
                        simple_part_id = _load_groups_txt(obj_dir, n_verts=len(simple_mesh.verts))
                        if simple_part_id is not None:
                            nn = torch.cdist(mesh.verts.float().cpu(),
                                             simple_mesh.verts.float().cpu()).argmin(dim=1)
                            obj_verts_part_id = simple_part_id[nn]

        kpts3d, kpts3d_mask = None, None
        if need_kpts:
            kpts3d, kpts3d_mask = _load_kpts3d_by_id(oid, self.path_preprocess, tform)

        category, category_id = None, None
        if need_category:
            cat_str = row.get("category")
            category = cat_str
            if cat_str is not None:
                try:
                    category_id = list(DENSEMATCHER_CATEGORIES).index(cat_str)
                except ValueError:
                    pass

        obj = Object(
            object_id               = oid,
            mesh                    = mesh if need_mesh else None,
            verts3d                 = mesh.verts if (need_verts3d and mesh is not None) else None,
            verts3d_feats           = mesh.vert_feats if (need_verts3d_feats and mesh is not None) else None,
            obj_ncds0c_tform4x4_obj = tform if (need_tform or need_mesh) else None,
            obj_kpts3d              = kpts3d,
            obj_kpts3d_mask         = kpts3d_mask,
            obj_verts_part_id       = obj_verts_part_id,
            category                = category,
            category_id             = category_id,
        )
        if self.cfg.obj_gl_tform4x4_obj_raw is not None:
            import torch as _torch
            from dataclasses import replace as _r
            T = _torch.tensor(self.cfg.obj_gl_tform4x4_obj_raw, dtype=_torch.float32)
            # transform() rotates verts/kpts by T and updates M → M @ inv(T).
            obj = obj.transform(T)
            # Apply T on the *other* side too, so M → T @ M @ inv(T) (both sides),
            # rotating the metric object frame into the canonical orientation.
            # Matches HouseCorr3D._load_object_from_row.
            if obj.obj_ncds0c_tform4x4_obj is not None:
                obj = _r(
                    obj,
                    obj_ncds0c_tform4x4_obj=T @ obj.obj_ncds0c_tform4x4_obj.float(),
                )
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
        import tempfile, urllib.request, zipfile
        from o3b.dataset.utils import download_progress as _download_progress

        path_raw = cls._path_raw(cfg)
        path_raw.mkdir(parents=True, exist_ok=True)
        download_url = url or _DEFAULT_URL
        print(f"Downloading {download_url} …")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urllib.request.urlretrieve(download_url, tmp_path, _download_progress)
            print(f"\nExtracting to {path_raw} …")
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(path_raw)
            print("Done.")
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def index(cls, cfg, *, db: Optional[Path] = None, remove: bool = False, max_index: Optional[int] = None, **_) -> None:
        if cfg.item_type == ItemType.OBJECT_PAIR:
            print(
                "'object_pair' needs no separate indexing — pairs are derived at load "
                "time. Run 'index -d dm_object' to build the base index.",
            )
            return
        if cfg.item_type != ItemType.OBJECT:
            print(f"ERROR: item_type '{cfg.item_type}' indexing is not implemented.", file=sys.stderr)
            sys.exit(1)

        path_raw        = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)
        db_path         = db or path_preprocess / "index.db"

        if remove and db_path.exists():
            print(f"Removing existing index: {db_path}")
            db_path.unlink()

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"db              : {db_path}")

        # Read subset membership from train/val/test txt files.
        # Each line has the format  <category>/<obj_dir_name>
        # which maps to object_id = <category>__<obj_dir_name>.
        print(f"\nReading subsets : {path_raw}")
        subset_by_id: dict[str, str] = {}
        for subset_name in ("train", "val", "test"):
            txt = path_raw / f"{subset_name}_files.txt"
            if not txt.exists():
                continue
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                object_id = line.replace("/", "__")
                subset_by_id[object_id] = subset_name
        print(f"  {len(subset_by_id)} object(s) assigned to a subset")

        # Layout: path_raw/<category>/<object_id>/simple_mesh.obj
        print(f"\nScanning meshes : {path_raw}")
        mesh_entries: list[dict] = []
        for cat_dir in sorted(path_raw.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for obj_dir in sorted(cat_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                mesh_file = _dm_mesh_in_dir(obj_dir)
                oid = f"{category}__{obj_dir.name}"
                mesh_entries.append({
                    "object_id": oid,
                    "mesh_path": str(mesh_file.relative_to(path_raw)) if mesh_file else None,
                    "category":  category,
                    "subset":    subset_by_id.get(oid),
                })
        if max_index is not None:
            mesh_entries = mesh_entries[:max_index]
        print(f"  found {len(mesh_entries)} object(s)")

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
                    t = torch.load(kpts_file, map_location="cpu")
                    kpts_by_id[entry.name] = (
                        json.dumps(t[:, :3].tolist()),
                        json.dumps(t[:, 3].bool().tolist()),
                    )
                except Exception as exc:
                    print(f"  WARN: could not load {kpts_file}: {exc}")
            print(f"  loaded kpts for {len(kpts_by_id)} object(s)")
        else:
            print(f"  Note: {kpts_root} not found — obj_kpts3d will be NULL.")

        print(f"\nWriting index   : {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS objects")
        cur.execute("""
            CREATE TABLE objects (
                object_id       TEXT PRIMARY KEY,
                mesh_path       TEXT,
                obj_kpts3d      TEXT,
                obj_kpts3d_mask TEXT,
                category        TEXT,
                subset          TEXT
            )
        """)
        for i, entry in enumerate(mesh_entries):
            oid = entry["object_id"]
            kpts_json, mask_json = kpts_by_id.get(oid, (None, None))
            cur.execute(
                "INSERT INTO objects (object_id, mesh_path, obj_kpts3d, obj_kpts3d_mask, category, subset) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (oid, entry["mesh_path"], kpts_json, mask_json, entry.get("category"), entry.get("subset")),
            )
            if (i + 1) % 100 == 0 or (i + 1) == len(mesh_entries):
                sys.stdout.write(f"\r  inserted {i + 1}/{len(mesh_entries)}")
                sys.stdout.flush()
        print()
        con.commit()
        con.close()
        print(f"\nDone. {len(mesh_entries)} object(s) indexed -> {db_path}")

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
        **_,
    ) -> None:
        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "index.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        dataset = cls(cfg)
        if not dataset._object_rows_id:
            print("No objects found in index.")
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

        print(f"Showing {len(dataset._object_rows_id)} / {total} objects  {db_path}\n")
        for entry in dataset._object_rows_id:
            if cfg.item_type == ItemType.OBJECT_PAIR:
                src_row  = dataset._object_rows[entry[0]]
                trgt_row = dataset._object_rows[entry[1]]
                print(f"{src_row['object_id']}  <--->  {trgt_row['object_id']}")
            else:
                row = dataset._object_rows[entry]
                mesh_rel = row.get("mesh_path")
                mesh_path = (dataset.path_raw / mesh_rel) if mesh_rel else (dataset.path_object_meshes / row["object_id"])
                print(f"{row['object_id']}  {mesh_path}")

        from o3b.data.viz import visualize_dataset
        visualize_dataset(dataset, render=render, render_frames=render_frames, renderer=renderer, debug=debug)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dm_mesh_in_dir(obj_dir: Path) -> Optional[Path]:
    """Return the best mesh file in a DenseMatcher object directory.

    Preference order: simple_mesh.obj → color_mesh.obj → any other .obj/.ply/etc.
    simple_mesh.obj is preferred because groups.txt vertex indices reference it.
    """
    for name in ("simple_mesh.obj", "color_mesh.obj"):
        candidate = obj_dir / name
        if candidate.exists():
            return candidate
    for ext in (".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"):
        hits = sorted(obj_dir.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def _load_groups_txt(obj_dir: Path, n_verts: int) -> Optional[torch.Tensor]:
    """Parse groups.txt into a (V,) int64 tensor of part IDs (0-indexed by line).

    Each line in groups.txt lists space-separated vertex indices belonging to one part.
    Vertices not listed in any group are assigned -1.
    """
    groups_file = obj_dir / "groups.txt"
    if not groups_file.exists():
        return None
    part_id = torch.full((n_verts,), -1, dtype=torch.int64)
    for group_idx, line in enumerate(groups_file.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        for idx_str in line.split():
            v_idx = int(idx_str)
            if 0 <= v_idx < n_verts:
                part_id[v_idx] = group_idx
    return part_id
