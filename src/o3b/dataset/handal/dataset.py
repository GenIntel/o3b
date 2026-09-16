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
from o3b.dataset.handal.enum import (
    HANDAL_CATEGORIES,
    MAP_CATEGORIES_HANDAL_TO_RPATH,
    MAP_CATEGORIES_HANDAL_TO_UCO3D,
    MAP_CATEGORIES_OBJ_ORIENT_HANDAL_TO_UCO3D,
)

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
