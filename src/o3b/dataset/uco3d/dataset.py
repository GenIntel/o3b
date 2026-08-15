"""uCO3D loader — frame-object items from the od3d-preprocessed UCO3D tree.

Raw data (``path_raw``, ``UCO3D/``) is the download from facebookresearch/uco3d:
one directory per sequence holding ``rgb_video.mp4``, ``mask_video.mkv`` and
``depth_maps.h5``.  Everything else comes from the od3d preprocessing
(``path_preprocess``, ``UCO3D_Preprocess/``), which this loader reads but does
not produce::

    meta/sequences/<category>/<sequence>.yaml            per-sequence annotation
    meta/frames/<category>/<sequence>/<frame>.yaml       per-frame annotation
    mesh/<mesh_type>/meta_mask/meta/<category>/<sequence>/mesh.ply
    tform_obj/<tform_obj_type>/meta_mask/meta/<category>/<sequence>/tform_obj.pt

The ``meta_mask/meta`` path segments are od3d's ``<pcl_type>/<sfm_type>`` pair,
fixed for UCO3D (segmented-point-cloud masks, metadata SfM).

**tform_obj** rotates a sequence's raw (SfM) object frame into the category's
canonical frame; ``tform_obj_type: raw`` skips it and keeps the SfM frame.  It
is a similarity, so it also carries a scale ``s``; o3b's metric object space is
defined as the canonical frame *divided* by ``s``, which is the raw
reconstruction's scale — see ``_load_frame_object_by_rowidx``.

The dataset is 1076 categories x ~100k sequences x ~200 frames, so there is no
eager scan: ``_setup`` walks only the categories asked for, and stops as soon as
``filter_count_max`` frames have been collected.  ``o3b dataset index`` caches
that walk into ``<path_preprocess>/frames.db`` for repeat runs (worth it over an
sshfs mount, where each directory listing is a round trip).
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch

from o3b.data.modalities import FrameObject
from o3b.dataset.dataset import ConfigurableDataset, ItemType, register_dataset
from o3b.dataset.uco3d.enum import UCO3D_CATEGORIES, MAP_UCO3D_CATEGORIES_TO_SUPER
from o3b.dataset.utils import want as _want

logger = logging.getLogger(__name__)

# od3d's <pcl_type>/<sfm_type> pair, constant for UCO3D (see UCO3D in od3d:
# pcl_type = META_MASK, sfm_type = META).
_PCL_SFM = ("meta_mask", "meta")

# How many sequences' meshes / tform_objs to keep decoded (see _sequence_mesh).
_MESH_CACHE_MAX = 64


def _load_meta_yaml(path: Path) -> Optional[dict]:
    """Read one od3d meta YAML.

    The files were written with ``yaml.dump`` of a dataclass holding ``Path``
    fields, so they carry ``!!python/object/apply:pathlib.PosixPath`` tags that
    SafeLoader rejects.  Only that one tag is added back — not ``unsafe_load``,
    which would execute whatever else a file happened to contain.
    """
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _posix_path(loader, node):
        return Path(*[str(p) for p in loader.construct_sequence(node)])

    _Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", _posix_path)
    _Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.WindowsPath", _posix_path)

    try:
        with open(path, "r") as f:
            return yaml.load(f, Loader=_Loader)
    except Exception as e:
        logger.warning(f"could not read meta {path}: {e}")
        return None


@register_dataset("UCO3D")
class UCO3D(ConfigurableDataset):

    categories: tuple = tuple(c.value for c in UCO3D_CATEGORIES)

    # ── paths ─────────────────────────────────────────────────────────────────

    @property
    def path_raw(self) -> Path:
        return Path(self.cfg.path_raw)

    @property
    def path_preprocess(self) -> Path:
        return Path(self.cfg.path_preprocess)

    @property
    def path_meta_frames(self) -> Path:
        return self.path_preprocess / "meta" / "frames"

    @property
    def path_meta_sequences(self) -> Path:
        return self.path_preprocess / "meta" / "sequences"

    def path_mesh(self, category: str, sequence: str) -> Path:
        return (self.path_preprocess / "mesh" / self.cfg.mesh_type / _PCL_SFM[0] / _PCL_SFM[1]
                / category / sequence / "mesh.ply")

    def path_tform_obj(self, category: str, sequence: str) -> Path:
        return (self.path_preprocess / "tform_obj" / self.cfg.tform_obj_type
                / _PCL_SFM[0] / _PCL_SFM[1] / category / sequence / "tform_obj.pt")

    # ── index ─────────────────────────────────────────────────────────────────

    def __len__(self):
        if self._sharded is not None:
            return len(self._sharded)
        return len(self._frame_rows_id)

    def _init_fields(self) -> None:
        # Same attribute names as HouseCorr3D's frame index, so the shared viser
        # browser (o3b.dataset.housecorr3d.frame_dataset) works unchanged.
        self._frame_rows: list[dict] = []
        self._frame_rows_id: list[int] = []
        self._tform_obj_cache: dict[str, Optional[torch.Tensor]] = {}
        self._mesh_cache: dict[str, Optional[tuple]] = {}

        # <tform_obj_type>.db, when it exists, replaces both the per-sequence
        # torch.load *and* the per-sequence .exists() the walk would otherwise
        # do — the whole table is one 0.12 s read (see tform_obj_store).  A plain
        # dict, not a live connection, so it survives the fork into DataLoader
        # workers.  None means "no db, use the directory".
        self._tform_obj_store: Optional[dict] = None
        if self.cfg.tform_obj_type != "raw" and self.cfg.path_preprocess is not None:
            from o3b.dataset import tform_obj_store
            self._tform_obj_store = tform_obj_store.load_all(
                self.cfg.path_preprocess, self.cfg.tform_obj_type,
            )
            if self._tform_obj_store is not None:
                logger.info(
                    f"tform_obj: {len(self._tform_obj_store)} transforms from "
                    f"{tform_obj_store.db_path(self.cfg.path_preprocess, self.cfg.tform_obj_type).name}"
                )

    def _subset_frames(self, category: str, sequence: str, frames: list) -> list:
        """The frames of a sequence a subset leaves, before any thinning.

        A subset listing whole sequences (``apple/1-2-3``) keeps all of them; one
        listing frame ids (``apple/1-2-3/66``, which is how od3d's own subsets in
        ``subset/`` are written) keeps exactly the frames it names.
        """
        sub = self.subset()
        if sub is None or sub.has_item(object_id=f"{category}/{sequence}"):
            return frames
        return [f for f in frames
                if sub.has_item(frame_id=f"{category}/{sequence}/{f}")]

    def _has_tform_obj(self, category: str, sequence: str) -> bool:
        """Is this sequence in the dataset? (a dict lookup when the db is present)"""
        if self.cfg.tform_obj_type == "raw":
            return True
        if self._tform_obj_store is not None:
            return f"{category}/{sequence}" in self._tform_obj_store
        return self.path_tform_obj(category, sequence).exists()

    @classmethod
    def _walker(cls, cfg) -> "UCO3D":
        """An instance set up for ``_walk_rows`` but with no index built yet.

        ``index`` walks one category at a time so it can commit as it goes;
        going through ``__init__`` would walk everything up front and then throw
        that away.  Everything the walk touches is derived from ``cfg``.
        """
        self = cls.__new__(cls)
        self.cfg = cfg
        self._sharded = None
        self._init_fields()
        return self

    def _setup(self) -> None:
        self._init_fields()

        if self.cfg.item_type != ItemType.FRAME_OBJECT:
            raise NotImplementedError(
                f"UCO3D supports item_type 'frame_object', got {self.cfg.item_type}"
            )

        db_path = self.path_preprocess / "frames.db"
        rows = self._rows_from_db(db_path) if db_path.exists() else None
        if rows is None:
            rows = list(self._walk_rows())
        self._frame_rows = rows
        self._frame_rows_id = list(range(len(rows)))

    def _rows_from_db(self, db_path: Path) -> Optional[list[dict]]:
        """Read the cached walk from frames.db, filtered by the current config.

        Returns None when the cache does not cover this config (a category that
        was never indexed), so the caller falls back to walking the tree.
        """
        cats = list(self.cfg.categories) if self.cfg.categories else None
        con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()
            indexed = {r[0] for r in cur.execute(
                "SELECT DISTINCT category FROM frames WHERE tform_obj_type = ?",
                (self.cfg.tform_obj_type,),
            ).fetchall()}
            if not indexed:
                return None
            # A cache covering only some of the wanted categories would silently
            # shrink the dataset, so fall back to the walk unless it covers all
            # of them.  With no `categories` set that means every category on
            # disk — one directory listing to check.
            wanted = set(cats) if cats is not None else set(self._iter_categories())
            if not wanted.issubset(indexed):
                missing = sorted(wanted - indexed)
                logger.info(
                    f"frames.db is missing {len(missing)} categor{'y' if len(missing) == 1 else 'ies'} "
                    f"({', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}) for "
                    f"tform_obj_type={self.cfg.tform_obj_type}; walking the tree instead"
                )
                return None
            if cats is not None:
                placeholders = ", ".join("?" * len(cats))
                where, params = f" AND category IN ({placeholders})", list(cats)
            else:
                where, params = "", []
            db_rows = cur.execute(
                f"SELECT category, sequence, frame FROM frames"
                f" WHERE tform_obj_type = ?{where}"
                f" ORDER BY category, sequence, rowid",
                [self.cfg.tform_obj_type] + params,
            ).fetchall()
        finally:
            con.close()

        # frames.db holds the *unthinned* walk, so the caps apply here exactly as
        # they do in _walk_rows — otherwise the same config would yield different
        # items depending on whether the cache happened to exist.
        extra      = self.cfg.extra or {}
        frames_max = extra.get("frames_count_max_per_sequence")
        seqs_max   = extra.get("sequences_count_max_per_category")
        limit      = self.cfg.filter_count_max

        by_sequence: dict[tuple[str, str], list[str]] = {}
        for r in db_rows:  # ordered by category, sequence, insertion
            by_sequence.setdefault((r["category"], r["sequence"]), []).append(r["frame"])

        rows: list[dict] = []
        n_frames: dict[str, int] = {}
        n_seqs: dict[str, int] = {}
        for (cat, seq), frames in by_sequence.items():
            # before the caps, exactly as in _walk_rows: a subset thins the pool
            # the caps then count, it does not carve items out of a capped pool.
            if not self.in_subset(f"{cat}/{seq}"):
                continue
            frames = self._subset_frames(cat, seq, frames)
            if not frames:
                continue
            if seqs_max is not None and n_seqs.get(cat, 0) >= seqs_max:
                continue
            if limit is not None and n_frames.get(cat, 0) >= limit:
                continue
            n_seqs[cat] = n_seqs.get(cat, 0) + 1
            for frame in _subsample(frames, frames_max):
                if limit is not None and n_frames.get(cat, 0) >= limit:
                    break
                rows.append(self._row(cat, seq, frame))
                n_frames[cat] = n_frames.get(cat, 0) + 1
        return rows

    @staticmethod
    def _row(category: str, sequence: str, frame: str) -> dict:
        return {
            "category":  category,
            "sequence":  sequence,
            "frame":     frame,
            "frame_id":  f"{category}/{sequence}/{frame}",
            "object_id": f"{category}/{sequence}",
        }

    def _iter_categories(self) -> list[str]:
        if self.cfg.categories:
            return list(self.cfg.categories)
        if not self.path_meta_sequences.is_dir():
            return []
        return sorted(p.name for p in self.path_meta_sequences.iterdir() if p.is_dir())

    def _walk_rows(self):
        """Enumerate (category, sequence, frame) triples from the meta tree.

        Only sequences that have both a ``tform_obj`` for the configured type and
        a mesh survive — od3d applies the same filter, and a sequence missing
        either cannot produce a posed item.  With ``subset_name`` set, only the
        sequences that subset lists survive as well.  Directory listings only; no
        YAML is read here.
        """
        extra = self.cfg.extra or {}
        frames_max   = extra.get("frames_count_max_per_sequence")
        seqs_max     = extra.get("sequences_count_max_per_category")
        limit        = self.cfg.filter_count_max

        if not self.path_meta_frames.is_dir():
            logger.warning(
                f"UCO3D meta not found at {self.path_meta_frames} — "
                f"mount it with 'o3b dataset sshfs -d uco3d -p slurm' or pass -p <platform>."
            )
            return

        for category in self._iter_categories():
            cat_dir = self.path_meta_frames / category
            if not cat_dir.is_dir():
                logger.warning(f"no frame meta for category {category!r} at {cat_dir}")
                continue

            n_cat = 0
            n_seq = 0
            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                if seqs_max is not None and n_seq >= seqs_max:
                    break
                sequence = seq_dir.name
                # first: a subset lookup is a set membership test, while the two
                # checks below are a stat() each (a network round trip over sshfs)
                if not self.in_subset(f"{category}/{sequence}"):
                    continue
                if not self._has_tform_obj(category, sequence):
                    continue
                if not self.path_mesh(category, sequence).exists():
                    continue

                frames = sorted(
                    (p.stem for p in seq_dir.iterdir() if p.suffix == ".yaml"),
                    key=lambda s: (len(s), s),  # numeric frame names, zero-padding-free
                )
                frames = self._subset_frames(category, sequence, frames)
                if not frames:
                    continue
                n_seq += 1
                for frame in _subsample(frames, frames_max):
                    yield self._row(category, sequence, frame)
                    n_cat += 1
                    if limit is not None and n_cat >= limit:
                        break
                if limit is not None and n_cat >= limit:
                    break

    # ── item loading ──────────────────────────────────────────────────────────

    def _load_frame_object(self, idx: int) -> Optional[FrameObject]:
        return self._load_frame_object_by_rowidx(self._frame_rows_id[idx])

    def _sequence_tform_obj(self, category: str, sequence: str) -> Optional[torch.Tensor]:
        """(4, 4) raw-object → canonical-object similarity, or None for 'raw'."""
        if self.cfg.tform_obj_type == "raw":
            return None
        key = f"{category}/{sequence}"

        # db present: the whole table is already in memory, no caching needed
        if self._tform_obj_store is not None:
            tform = self._tform_obj_store.get(key)
            if tform is None:
                logger.warning(f"no tform_obj for {key} in "
                               f"{self.cfg.tform_obj_type}.db")
                return None
            return tform.float()

        if key in self._tform_obj_cache:
            return self._tform_obj_cache[key]
        if len(self._tform_obj_cache) >= _MESH_CACHE_MAX:
            self._tform_obj_cache.clear()
        tform = None
        path = self.path_tform_obj(category, sequence)
        if path.exists():
            try:
                tform = torch.load(path, map_location="cpu", weights_only=True).float()
            except Exception as e:
                logger.warning(f"could not read tform_obj {path}: {e}")
        else:
            logger.warning(f"no tform_obj of type {self.cfg.tform_obj_type} at {path}")
        self._tform_obj_cache[key] = tform
        return tform

    def _sequence_mesh(self, category: str, sequence: str, tform_obj):
        """Load the sequence mesh and normalise it to NCDS space.

        Returns ``(mesh, obj_ncds0c_tform4x4_obj)`` where the mesh's verts span
        [-1, 1] on their longest axis and the transform maps them back into
        metric object space (canonical divided by tform_obj's scale).
        """
        key = f"{category}/{sequence}"
        if key in self._mesh_cache:
            cached = self._mesh_cache[key]
            if cached is None:
                return None, None
            mesh, tform = cached
            from dataclasses import replace as _r
            return _r(mesh), tform  # fresh Mesh instance, shared tensors

        from o3b.data.datatypes.mesh import Mesh

        # Frames of one sequence arrive together, so a small cache already turns
        # ~frames_count_max_per_sequence mesh reads into one; keeping every
        # sequence would instead grow without bound across a full epoch.
        if len(self._mesh_cache) >= _MESH_CACHE_MAX:
            self._mesh_cache.clear()

        path = self.path_mesh(category, sequence)
        if not path.exists():
            self._mesh_cache[key] = None
            return None, None
        try:
            import trimesh
            loaded = trimesh.load(path, process=False)
            verts = torch.tensor(loaded.vertices, dtype=torch.float32)
            faces = torch.tensor(loaded.faces, dtype=torch.int64)
        except Exception as e:
            logger.warning(f"could not read mesh {path}: {e}")
            self._mesh_cache[key] = None
            return None, None

        # raw SfM object frame → canonical object frame
        s = 1.0
        if tform_obj is not None:
            verts = verts @ tform_obj[:3, :3].t() + tform_obj[:3, 3]
            s = float(tform_obj[:3, :3].norm(dim=0).mean())

        # canonical → NCDS (centred, longest axis in [-1, 1])
        v_min, v_max = verts.min(dim=0).values, verts.max(dim=0).values
        center     = (v_min + v_max) * 0.5
        half_scale = float((v_max - v_min).max().clamp(min=1e-8)) * 0.5
        verts_ncds = (verts - center) / half_scale

        # NCDS → metric object space (= canonical / s)
        tform = torch.eye(4, dtype=torch.float32)
        tform[:3, :3] = torch.eye(3) * (half_scale / s)
        tform[:3,  3] = center / s

        mesh = Mesh(verts=verts_ncds, faces=faces)
        self._mesh_cache[key] = (mesh, tform)
        from dataclasses import replace as _r
        return _r(mesh), tform

    def _load_frame_object_by_rowidx(self, row_idx: int) -> Optional[FrameObject]:
        from o3b.cv.geometry.transform import inv_tform4x4, transf3d_broadcast
        from o3b.cv.io import read_depth_from_video, read_image_from_video, read_mask_from_video

        row  = self._frame_rows[row_idx]
        mods = self.cfg.modalities
        category, sequence, frame = row["category"], row["sequence"], row["frame"]

        meta = _load_meta_yaml(self.path_meta_frames / category / sequence / f"{frame}.yaml")
        if meta is None:
            return None

        # ── object geometry ───────────────────────────────────────────────────
        tform_obj = self._sequence_tform_obj(category, sequence)
        if self.cfg.tform_obj_type != "raw" and tform_obj is None:
            return None
        s = float(tform_obj[:3, :3].norm(dim=0).mean()) if tform_obj is not None else 1.0
        mesh, obj_ncds0c_tform4x4_obj = self._sequence_mesh(category, sequence, tform_obj)

        # ── frame modalities ──────────────────────────────────────────────────
        timestamp = float(meta.get("timestamp", 0.0))
        rgb = None
        if _want("rgb", mods) and meta.get("rfpath_rgb"):
            rgb_u8 = read_image_from_video(self.path_raw / meta["rfpath_rgb"], timestamp)
            if rgb_u8 is not None:
                rgb = rgb_u8[:3].float() / 255.0

        mask = None
        if _want("fo_mask", mods) and meta.get("rfpath_mask"):
            m = read_mask_from_video(self.path_raw / meta["rfpath_mask"], timestamp)
            if m is not None:
                mask = m[0].bool()   # (1, H, W) → (H, W)

        depth, depth_mask = None, None
        if (_want("depth", mods) or _want("depth_mask", mods)) and meta.get("rfpath_depth"):
            try:
                d = read_depth_from_video(
                    self.path_raw / meta["rfpath_depth"],
                    int(meta.get("timestep", 0)),
                    float(meta.get("depth_scale", 1.0)),
                )
                depth = d[0] if d is not None else None   # (1, H, W) → (H, W)
            except Exception as e:
                logger.warning(f"could not read depth for {row['frame_id']}: {e}")
            if depth is not None:
                depth_mask = (depth > 0) if _want("depth_mask", mods) else None
                if not _want("depth", mods):
                    depth = None

        fo_mask_dt = None
        if _want("fo_mask_dt", mods) and mask is not None:
            from o3b.cv.visual.mask import get_mask_distance_transform_norm
            fo_mask_dt = get_mask_distance_transform_norm(mask.cpu()).float()

        cam_intr4x4 = None
        if meta.get("l_cam_intr4x4"):
            cam_intr4x4 = torch.tensor(meta["l_cam_intr4x4"], dtype=torch.float32)

        cam_bbox2d = None
        # An empty mask makes get_bboxs_from_masks fall back to the whole image,
        # which would bake a full-frame crop into the shards — leave it None so
        # the frame is dropped instead.
        if _want("cam_bbox2d", mods) and mask is not None and bool(mask.any()):
            from o3b.cv.visual.draw import get_bboxs_from_masks
            cam_bbox2d = get_bboxs_from_masks(mask[None])[0].float()

        # ── camera pose ───────────────────────────────────────────────────────
        # meta holds cam←obj in the raw SfM object frame, in od3d's OpenCV-style
        # camera convention; cam_tform4x4_cam_raw flips it to o3b's OpenGL one.
        cam_tform4x4_obj_metric = None
        if meta.get("l_cam_tform4x4_obj"):
            M = torch.tensor(meta["l_cam_tform4x4_obj"], dtype=torch.float32)
            if self.cfg.cam_tform4x4_cam_raw is not None:
                M = torch.tensor(self.cfg.cam_tform4x4_cam_raw, dtype=torch.float32) @ M
            if tform_obj is not None:
                M = M @ inv_tform4x4(tform_obj)
            # inv(tform_obj) shrank the rotation block by 1/s; metric object
            # space is the canonical frame divided by s, so scaling back by s
            # leaves a rigid cam←obj.
            M = M.clone()
            M[:3, :3] = M[:3, :3] * s
            cam_tform4x4_obj_metric = M

        # obj_gl_tform4x4_obj_raw rotates the object frame itself into o3b's
        # canonical axis order. Same bookkeeping as HouseCorr3D's
        # _load_object_from_row: verts by T, obj_ncds0c → T @ M @ inv(T), and the
        # pose by @ inv(T), so the object still projects onto the same pixels.
        if self.cfg.obj_gl_tform4x4_obj_raw is not None:
            T = torch.tensor(self.cfg.obj_gl_tform4x4_obj_raw, dtype=torch.float32)
            if cam_tform4x4_obj_metric is not None:
                cam_tform4x4_obj_metric = cam_tform4x4_obj_metric @ inv_tform4x4(T)
            if obj_ncds0c_tform4x4_obj is not None:
                obj_ncds0c_tform4x4_obj = T @ obj_ncds0c_tform4x4_obj @ inv_tform4x4(T)
            if mesh is not None:
                mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=T)

        cam_tform4x4_obj_ncds = None
        obj_size, obj_size_ncds, obj_size3d, obj_bbox3d = None, None, None, None
        if obj_ncds0c_tform4x4_obj is not None:
            obj_size_ncds = 2.0
            # isotropic, but read it as a norm so an axis-permuting T is fine
            obj_size = float(2.0 * obj_ncds0c_tform4x4_obj[:3, :3].norm(dim=0).mean())
            if cam_tform4x4_obj_metric is not None:
                cam_tform4x4_obj_ncds = cam_tform4x4_obj_metric @ obj_ncds0c_tform4x4_obj
        if mesh is not None and obj_ncds0c_tform4x4_obj is not None:
            v_min, v_max = mesh.verts.min(dim=0).values, mesh.verts.max(dim=0).values
            x0, y0, z0 = v_min.tolist()
            x1, y1, z1 = v_max.tolist()
            # Corner order is load-bearing, not cosmetic: the viser wireframe
            # walks 0-1-2-3 and 4-5-6-7 as two cyclic quads joined by 0-4/1-5/
            # 2-6/3-7, and _corners8_to_size_tform reads +x off 0→1, +y off 0→3,
            # +z off 0→4.  Emitting the corners in lexicographic (binary-count)
            # order instead draws four face diagonals.  Same layout as
            # HouseCorr3D's _load_object_from_row.
            corners_ncds = torch.tensor([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # -z face, cyclic
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # +z face, cyclic
            ], dtype=torch.float32)
            obj_bbox3d = transf3d_broadcast(corners_ncds, obj_ncds0c_tform4x4_obj)  # (8, 3) metric
            obj_size3d = obj_bbox3d.max(0).values - obj_bbox3d.min(0).values

        cam_bbox3d = None
        if (_want("cam_bbox3d", mods) and obj_bbox3d is not None
                and cam_tform4x4_obj_metric is not None):
            cam_bbox3d = transf3d_broadcast(obj_bbox3d, cam_tform4x4_obj_metric)  # (8, 3)

        # ── amodal object mask (+ its DT) ─────────────────────────────────────
        # UCO3D has no artist mesh: this rasterises the *reconstructed* mesh
        # (mesh_type, e.g. alpha500) under the GT pose.  It fills in what the
        # frame's mask video leaves out, at that mesh's fidelity.
        fo_mask_amodal, fo_mask_amodal_dt = None, None
        want_amodal    = _want("fo_mask_amodal", mods)
        want_amodal_dt = _want("fo_mask_amodal_dt", mods)
        if ((want_amodal or want_amodal_dt) and mesh is not None
                and cam_tform4x4_obj_ncds is not None and cam_intr4x4 is not None):
            fo_mask_amodal = self._render_fo_mask_amodal(
                mesh, cam_tform4x4_obj_ncds, cam_intr4x4, rgb=rgb, mask=mask, meta=meta,
            )
            if fo_mask_amodal is not None and want_amodal_dt:
                from o3b.cv.visual.mask import get_mask_distance_transform_norm
                fo_mask_amodal_dt = get_mask_distance_transform_norm(fo_mask_amodal).float()
            if not want_amodal:
                fo_mask_amodal = None

        # ── symmetry ──────────────────────────────────────────────────────────
        obj_syms, obj_kpts3d_syms, obj_axis6d_sym = None, None, None
        if _want("obj_syms", mods):
            from o3b.dataset.housecorr3d.dataset import _compute_obj_sym_geometry
            from o3b.dataset.uco3d.obj_syms import obj_syms_for_category
            obj_syms = obj_syms_for_category(category)
            obj_kpts3d_syms, obj_axis6d_sym = _compute_obj_sym_geometry(obj_syms, None)

        category_id = None
        if _want("category", mods):
            try:
                category_id = list(self.categories).index(category)
            except ValueError:
                pass

        return FrameObject(
            frame_id                = row["frame_id"],
            frame_object_id         = row["frame_id"],
            object_id               = row["object_id"],
            rgb                     = rgb,
            depth                   = depth,
            depth_mask              = depth_mask,
            fo_mask                 = mask,
            fo_mask_dt              = fo_mask_dt,
            fo_mask_amodal          = fo_mask_amodal,
            fo_mask_amodal_dt       = fo_mask_amodal_dt,
            cam_intr4x4             = cam_intr4x4 if _want("cam_intr4x4", mods) else None,
            cam_tform4x4_obj        = cam_tform4x4_obj_metric if _want("cam_tform4x4_obj", mods) else None,
            cam_tform4x4_obj_ncds   = cam_tform4x4_obj_ncds if _want("cam_tform4x4_obj_ncds", mods) else None,
            cam_bbox2d              = cam_bbox2d,
            cam_bbox3d              = cam_bbox3d,
            obj_bbox3d              = obj_bbox3d if _want("obj_bbox3d", mods) else None,
            obj_size3d              = obj_size3d,
            mesh                    = mesh if _want("mesh", mods) else None,
            obj_ncds0c_tform4x4_obj = obj_ncds0c_tform4x4_obj,
            obj_size_ncds           = obj_size_ncds,
            obj_size                = obj_size,
            obj_syms                = obj_syms,
            obj_kpts3d_syms         = obj_kpts3d_syms,
            obj_axis6d_sym          = obj_axis6d_sym,
            category                = category if _want("category", mods) else None,
            category_id             = category_id,
        )

    def _render_fo_mask_amodal(self, mesh, cam_tform4x4_obj_ncds, cam_intr4x4,
                               rgb=None, mask=None, meta=None) -> "Optional[torch.Tensor]":
        """(H, W) bool silhouette of *mesh* under the GT pose, or None."""
        from o3b.dataset.housecorr3d.frame_dataset import render_scene_depth

        H, W = self._frame_image_hw(rgb, mask, cam_intr4x4, meta)
        if H is None or W is None:
            return None
        R, t = cam_tform4x4_obj_ncds[:3, :3], cam_tform4x4_obj_ncds[:3, 3]
        verts_cam = (R @ mesh.verts.float().t()).t() + t
        try:
            depth = render_scene_depth([(verts_cam, mesh.faces)], cam_intr4x4.float(), H, W)
        except Exception as e:
            logger.warning(f"amodal mask render failed: {e}")
            return None
        return None if depth is None else (depth > 0)

    @staticmethod
    def _frame_image_hw(rgb, mask, intr, meta) -> tuple:
        """Full-image (H, W): prefer a loaded modality, then the meta's l_size."""
        if rgb is not None:
            return int(rgb.shape[-2]), int(rgb.shape[-1])
        if mask is not None:
            return int(mask.shape[-2]), int(mask.shape[-1])
        # l_size is the raw uco3d `_image_size` blob, which is stored (H, W)
        # despite od3d unpacking it under the names (width, height) — the
        # principal point it derives lands on the H-major image either way.
        size = (meta or {}).get("l_size")
        if size and len(size) == 2:
            return int(size[0]), int(size[1])
        if intr is not None:
            return int(round(2 * float(intr[1, 2]))), int(round(2 * float(intr[0, 2])))
        return None, None

    def _sharded_drop_reason(self, idx: int) -> str:
        row = self._frame_rows[self._frame_rows_id[idx]]
        category, sequence, frame = row["category"], row["sequence"], row["frame"]
        if not (self.path_meta_frames / category / sequence / f"{frame}.yaml").exists():
            return "frame meta yaml missing"
        if not self._has_tform_obj(category, sequence):
            return f"no tform_obj of type {self.cfg.tform_obj_type}"
        if not self.path_mesh(category, sequence).exists():
            return f"no {self.cfg.mesh_type} mesh"
        return super()._sharded_drop_reason(idx)

    # ── CLI hooks ─────────────────────────────────────────────────────────────

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None) -> None:
        raise NotImplementedError(
            "UCO3D is not downloaded by o3b. Get the raw videos with\n"
            "  python dataset_download/download_dataset.py --use_huggingface "
            "--download_folder <path_raw> \\\n"
            "    --download_modalities rgb_videos,mask_videos,depth_maps,point_clouds,"
            "segmented_point_clouds\n"
            "from https://github.com/facebookresearch/uco3d, then produce "
            "<path_preprocess> with od3d\n(meta / mesh / tform_obj), or mount an "
            "existing tree: o3b dataset sshfs -d uco3d -p slurm"
        )

    @classmethod
    def index(cls, cfg, *, db: Optional[Path] = None, **kwargs) -> None:
        """Cache the meta-tree walk into <path_preprocess>/frames.db.

        Rows are keyed by tform_obj_type, so every9d_v3 and every9d_v4 coexist in
        one file.  Re-indexing a (tform_obj_type, category) pair replaces its rows.
        """
        from dataclasses import replace as _r

        db_path = db or cls._path_preprocess(cfg) / "frames.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Walk unrestricted: the cache should not inherit this run's frame cap,
        # per-sequence thinning or subset selection — all three apply again when
        # it is read, and frames.db is keyed by tform_obj_type alone, so a
        # subset-thinned walk written into it would starve every other config.
        extra = dict(cfg.extra or {})
        extra["frames_count_max_per_sequence"] = None
        extra.pop("subset_ids", None)
        walk_cfg = _r(cfg, filter_count_max=None, subset_name=None, extra=extra)
        dataset = cls._walker(walk_cfg)

        con = sqlite3.connect(db_path, timeout=60)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                tform_obj_type TEXT NOT NULL,
                category       TEXT NOT NULL,
                sequence       TEXT NOT NULL,
                frame          TEXT NOT NULL,
                UNIQUE (tform_obj_type, category, sequence, frame)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS frames_cat ON frames (tform_obj_type, category)")

        from tqdm import tqdm

        categories = dataset._iter_categories()
        print(f"Indexing {len(categories)} categor{'y' if len(categories) == 1 else 'ies'} "
              f"(tform_obj_type={cfg.tform_obj_type}) → {db_path}")
        # The walk is one directory listing per sequence, so over an sshfs mount
        # this is minutes per category: stream it into the table in batches and
        # commit as we go, so progress is visible and an interrupt keeps what is
        # already indexed (each category is deleted before it is re-inserted, so
        # a resumed run just redoes the one it was in the middle of).
        total = 0
        batch: list[tuple] = []
        pbar = tqdm(desc="Indexing", unit="row")
        for category in categories:
            pbar.set_postfix_str(category)
            cur.execute("DELETE FROM frames WHERE tform_obj_type = ? AND category = ?",
                        (cfg.tform_obj_type, category))
            dataset.cfg = _r(walk_cfg, categories=[category])
            for row in dataset._walk_rows():
                batch.append((cfg.tform_obj_type, row["category"], row["sequence"], row["frame"]))
                total += 1
                pbar.update(1)
                if len(batch) >= 5000:
                    cur.executemany("INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?)", batch)
                    con.commit()
                    batch.clear()
            if batch:
                cur.executemany("INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?)", batch)
                batch.clear()
            con.commit()
        pbar.close()
        con.close()
        print(f"Done. {total} frame rows → {db_path}")

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
        from dataclasses import replace as _r

        from o3b.dataset.housecorr3d.frame_dataset import _visualize_frame_objects_viser

        path_preprocess = cls._path_preprocess(cfg)
        if not (path_preprocess / "meta" / "frames").is_dir():
            print(
                f"No UCO3D meta at {path_preprocess}.\n"
                f"  Mount the remote tree:  o3b dataset sshfs -d uco3d -p slurm\n"
                f"  or run on the cluster:  o3b dataset viz -d uco3d -p slurm",
                file=sys.stderr,
            )
            sys.exit(1)

        # Cap the walk at what is actually going to be browsed, and thin the
        # frames-per-sequence so `limit` items span several objects rather than
        # a single sequence's first N viewpoints.
        viz_extra = dict(cfg.extra or {})
        per_seq = viz_extra.get("frames_count_max_per_sequence")
        viz_extra["frames_count_max_per_sequence"] = min(per_seq or 4, 4)
        viz_cfg = _r(cfg, modalities=None, filter_count_max=limit, extra=viz_extra)
        dataset = cls(viz_cfg)
        if not dataset._frame_rows_id:
            print("No frames found matching the current config filters.")
            return

        total = len(dataset._frame_rows_id)
        if object_id:
            dataset._frame_rows_id = [
                i for i in dataset._frame_rows_id
                if dataset._frame_rows[i]["object_id"] == object_id
            ]
        if limit < len(dataset._frame_rows_id):
            dataset._frame_rows_id = dataset._frame_rows_id[:limit]

        print(f"Showing {len(dataset._frame_rows_id)} / {total} frames  {path_preprocess}\n")
        _visualize_frame_objects_viser(dataset, debug=debug, obj_centric=obj_centric)


def _subsample(items: list, count_max: Optional[int]) -> list:
    """Evenly-spaced subset of *items* of length <= count_max (None = all)."""
    if not count_max or count_max >= len(items) or count_max <= 0:
        return list(items)
    step = len(items) / count_max
    return [items[int(i * step)] for i in range(count_max)]


def uco3d_category_to_super(category: str) -> Optional[str]:
    """Super-category of a UCO3D category (the raw tree's top directory level)."""
    return MAP_UCO3D_CATEGORIES_TO_SUPER.get(category)
