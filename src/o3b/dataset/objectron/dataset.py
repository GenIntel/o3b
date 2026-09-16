"""Objectron — frame-object items from the od3d-preprocessed tree.

The smallest of the four: 2,250 items, 9 categories x 50 sequences x 5 frames
evenly spaced through each video.  That sampling is not re-derived here — it is
recorded in the preprocessed tree itself, as the filenames under
``mask/sam_bbox/test/<category>/<sequence>/<frame>.png`` — so the walk reads it
back rather than guessing which sequences and frames od3d drew.

This dataset needs no protobuf decoding, despite the raw data being
``video.MOV`` + ``annotation.pbdata`` per sequence: od3d already resolved both
into ``meta/frames/<subset>/<category>/<sequence>/<frame>.yaml`` (poses,
intrinsics, the 8 3-D box corners) and extracted the rgb into
``<path_raw>/frames/``.  Those two are what ``fetch`` has to secure.

Depth is ``depth_anything_v3`` — a *monocular estimate*, not sensed and not
CAD-rendered.  That makes three different depth provenances across the six
evaluation sets, which is worth stating whenever the Objectron column is
compared against the others.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import ConfigurableDataset, ItemType, register_dataset
from o3b.dataset.objectron.enum import (
    MAP_CATEGORIES_OBJECTRON_TO_UCO3D,
    OBJECTRON_CATEGORIES,
)


@register_dataset("Objectron")
class Objectron(ConfigurableDataset):
    """Objectron as ``frame_object`` items."""

    all_categories = tuple(c.value for c in OBJECTRON_CATEGORIES)
    map_categories_to_uco3d = MAP_CATEGORIES_OBJECTRON_TO_UCO3D
    # No per-category re-orientation: Objectron's box annotation already uses a
    # consistent right/up/front frame across its nine categories.
    map_categories_obj_orient_to_uco3d = None

    def __init__(self, cfg):
        if cfg.item_type != ItemType.FRAME_OBJECT:
            raise ValueError(
                f"Objectron supports item_type 'frame_object', got {cfg.item_type}"
            )
        super().__init__(cfg)

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return Path(cfg.path_raw or cfg.root)

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None, dry_run: bool = False) -> None:
        """Secure the extracted frames and the od3d meta tree.

        Deliberately not a download.  od3d's own Objectron setup fetched from
        ``storage.googleapis.com/objectron`` and its comment records that the
        bucket stopped serving anonymous callers ("first call gcloud auth
        login").  Re-implementing that would produce a fetch that fails for
        anyone without credentials, to rebuild something the copy source already
        holds in resolved form.  So both halves come from ``fetch_copy_from``,
        and the hint names the manual route for a machine that has neither.
        """
        from o3b.dataset.od3d_fetch import fetch_or_copy

        extra = dict(cfg.extra or {})
        copy_raw = extra.get("fetch_copy_from")
        copy_pre = extra.get("fetch_copy_from_preprocess")
        path_raw = cls._path_raw(cfg)
        path_pre = cls._path_preprocess(cfg)

        hint = (
            "Objectron's GCS bucket no longer serves anonymous callers; od3d's\n"
            "downloader needs `gcloud auth login`. Point extra.fetch_copy_from /\n"
            "extra.fetch_copy_from_preprocess at an existing tree instead."
        )

        # rgb: <path_raw>/frames/<category>/<sequence>/<frame>.jpg
        fetch_or_copy(
            "Objectron (frames)",
            path_raw,
            expect=("frames",),
            copy_from=Path(copy_raw) if copy_raw else None,
            download=None,
            download_hint=hint,
            dry_run=dry_run,
        )

        # poses + intrinsics + the sampling itself
        fetch_or_copy(
            "Objectron (meta)",
            path_pre,
            expect=("meta/frames", "mask/sam_bbox", "depth/depth_anything_v3"),
            copy_from=Path(copy_pre) if copy_pre else None,
            download=None,
            download_hint=hint,
            dry_run=dry_run,
        )
