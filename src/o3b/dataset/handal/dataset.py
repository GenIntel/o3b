"""HANDAL — frame-object items from the od3d-preprocessed tree.

One object per frame, and the raw data says so directly: HANDAL's masks are
``mask_visib/<frame>_<object>.png``, so od3d's meta is already keyed by
(frame, object).  It is the largest of the four migrated sets — 277,016 frame
metas — which is why every sharded config over it carries a per-sequence cap.

Unlike PASCAL3D / ImageNet3D, the depth here is real (the ``_with_depth``
archives, plus od3d's ``depth_nerf`` renders), so no CAD-rendered stand-in is
involved.

fetch() downloads a Google Drive *folder*, which is rate-limited and regularly
fails part-way.  That is the case ``fetch_or_copy`` exists for: on the cluster
the tree is already present and the copy source settles it without touching
Drive at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import ItemType, register_dataset
from o3b.dataset.od3d_frames import Od3dFrameDataset
import json
import logging

from o3b.dataset.handal.enum import (
    HANDAL_CATEGORIES,
    MAP_CATEGORIES_HANDAL_TO_RPATH,
    MAP_CATEGORIES_HANDAL_TO_UCO3D,
    MAP_CATEGORIES_OBJ_ORIENT_HANDAL_TO_UCO3D,
)

logger = logging.getLogger(__name__)

_GDRIVE_FOLDER = "https://drive.google.com/drive/folders/10mDNZnYrg55ZiP9GV4upKWnxlxay1OwM"


@register_dataset("HANDAL")
class HANDAL(Od3dFrameDataset):
    """HANDAL as ``frame_object`` items."""

    all_categories = tuple(c.value for c in HANDAL_CATEGORIES)
    map_categories_to_uco3d = MAP_CATEGORIES_HANDAL_TO_UCO3D
    map_categories_obj_orient_to_uco3d = MAP_CATEGORIES_OBJ_ORIENT_HANDAL_TO_UCO3D

    # sequenced meta tree — see Od3dFrameDataset
    has_sequence_level = True

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return Path(cfg.path_raw or cfg.root)

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    @classmethod
    def _expected_dirs(cls, cfg) -> tuple[str, ...]:
        """Per-category raw directory names, for the completeness check.

        Restricted to ``cfg.categories`` when set, so fetching one category does
        not report the other sixteen as missing.
        """
        cats = list(cfg.categories) if cfg.categories else list(cls.all_categories)
        return tuple(
            MAP_CATEGORIES_HANDAL_TO_RPATH[c]
            for c in cats
            if c in MAP_CATEGORIES_HANDAL_TO_RPATH
        )

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None, dry_run: bool = False) -> None:
        from o3b.dataset.od3d_fetch import fetch_or_copy

        extra = dict(cfg.extra or {})
        copy_from = extra.get("fetch_copy_from")
        path_raw = cls._path_raw(cfg)
        folder_url = url or extra.get("url_raw") or _GDRIVE_FOLDER

        def _gdown() -> None:
            import gdown  # imported here: only the download path needs it

            print(f"  gdown folder {folder_url}")
            gdown.download_folder(
                folder_url, output=str(path_raw), quiet=False,
                use_cookies=False, remaining_ok=True,
            )

        fetch_or_copy(
            "HANDAL",
            path_raw,
            expect=cls._expected_dirs(cfg),
            copy_from=Path(copy_from) if copy_from else None,
            download=None if dry_run else _gdown,
            download_hint=(
                "Google Drive rate-limits folder downloads. Either retry\n"
                f"  gdown --folder {folder_url} -O {path_raw}\n"
                "or point extra.fetch_copy_from at a machine-local copy."
            ),
            dry_run=dry_run,
        )

    # ── object geometry ──────────────────────────────────────────────────────

    def _obj_id(self, row) -> Optional[int]:
        """BOP object id for this sequence, from its scene_gt.json.

        Every HANDAL sequence holds exactly one object (verified across 90
        sequences of three categories: no frame has more than one), so the id is
        a property of the sequence rather than of the frame.

        The sequence name encodes it — `004001` is object 4 — and that held on
        every sequence checked, but scene_gt.json is the authority and the name
        is only the fallback for a sequence that has none.
        """
        cache = getattr(self, "_obj_id_cache", None)
        if cache is None:
            cache = self._obj_id_cache = {}
        key = (row["category"], row["sequence"])
        if key in cache:
            return cache[key]

        rpath = MAP_CATEGORIES_HANDAL_TO_RPATH.get(row["category"])
        obj_id = None
        if rpath:
            gt = self.path_raw / rpath / row["split"] / row["sequence"] / "scene_gt.json"
            if gt.exists():
                try:
                    d = json.loads(gt.read_text())
                    first = d[sorted(d, key=lambda k: int(k))[0]]
                    obj_id = int(first[0]["obj_id"])
                except Exception as e:
                    logger.warning(f"could not read {gt}: {e}")
        if obj_id is None:
            try:
                obj_id = int(str(row["sequence"])[:3])
            except ValueError:
                obj_id = None
        cache[key] = obj_id
        return obj_id

    def _mesh_path(self, row, meta) -> Optional[Path]:
        """BOP model for this sequence: <raw>/<rpath>/models/obj_NNNNNN.ply.

        HANDAL names no mesh in its meta — unlike PASCAL3D and ImageNet3D, whose
        metas carry rfpath_mesh — so it is resolved from the scene's object id.
        The models are in millimetres, which extra.scale_to_m converts along with
        the poses.
        """
        rpath = MAP_CATEGORIES_HANDAL_TO_RPATH.get(row["category"])
        obj_id = self._obj_id(row)
        if not rpath or obj_id is None:
            return None
        return self.path_raw / rpath / "models" / f"obj_{obj_id:06d}.ply"

    # ── bbox correction ──────────────────────────────────────────────────────

    def _cam_bbox2d_from_meta(self, meta):
        """HANDAL's ``l_bbox`` is mislabelled at the source; reorder it here.

        od3d's extract_meta wrote

            l_bbox = [bbox_visib[2], bbox_visib[3], bbox_visib[0], bbox_visib[1]]
            ...
            l_bbox=l_bbox,   # x0, y0, x1, y1

        but BOP's ``bbox_visib`` is ``[x, y, w, h]``, so what is stored is
        ``[w, h, x, y]`` while the comment claims xyxy. Verified against the
        source JSON: bbox_visib [925, 496, 199, 258] is stored as
        [199.0, 258.0, 925.0, 496.0].

        Read as xyxy that gives width ``x - w`` and height ``y - h``, which is
        negative whenever the box is wider than its left offset — 41% of a
        1,646-meta sample, and those frames were dropped outright by
        CropCamBBox2D (2,962 of 8,320 in the first whole-dataset build). The
        other 59% were *worse*: a plausible-looking box in the wrong place, which
        crops to the wrong pixels and reports nothing.

        Corrected rather than re-extracted because the fix is exact and local;
        re-running od3d's extract_meta over 277k frames to rewrite the same
        numbers would be slower and would leave the loader trusting a field it
        has been shown cannot be trusted.
        """
        import torch

        b = meta.get("l_bbox")
        if not b or len(b) != 4:
            return None
        w, h, x, y = (float(v) for v in b)
        if w <= 0 or h <= 0:
            return None
        return torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
