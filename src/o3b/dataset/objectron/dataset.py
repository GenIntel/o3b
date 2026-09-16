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
from o3b.dataset.od3d_fetch import FetchSkipped
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
    def _sampled_frames(cls, path_pre: Path, mask_type: str, split: str):
        """The (category, sequence, frame) triples the benchmark actually uses.

        Read off the mask tree's filenames rather than parsed out of the metas:
        ``mask/<mask_type>/<split>/<category>/<sequence>/<frame>.png`` *is* the
        record of which 5 frames of which 50 sequences od3d drew, so no yaml
        parsing (and no PosixPath-tagged unsafe_load) is needed just to find out
        what to copy.
        """
        root = path_pre / "mask" / mask_type / split
        if not root.is_dir():
            return []
        out = []
        for cat in sorted(p for p in root.iterdir() if p.is_dir()):
            for seq in sorted(p for p in cat.iterdir() if p.is_dir()):
                for frame in sorted(seq.glob("*.png")):
                    out.append((cat.name, seq.name, frame.stem))
        return out

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None, dry_run: bool = False) -> None:
        """Secure the od3d meta tree and exactly the rgb frames it references.

        Deliberately not a download.  od3d's own Objectron setup fetched from
        ``storage.googleapis.com/objectron`` and its comment records that the
        bucket stopped serving anonymous callers ("first call gcloud auth
        login").  Re-implementing that would produce a fetch that fails for
        anyone without credentials, to rebuild something the copy source already
        holds in resolved form.

        The rgb copy is per-file, not per-directory.  ``<path_raw>/frames`` holds
        910,660 jpgs across the nine categories; the benchmark samples 2,250 of
        them.  Syncing the directory would move ~350 GB to obtain ~900 MB, so the
        sampled set is derived from the mask tree and passed to rsync directly.
        """
        from o3b.dataset.od3d_fetch import fetch_or_copy, rsync_files

        extra = dict(cfg.extra or {})
        copy_raw = extra.get("fetch_copy_from")
        copy_pre = extra.get("fetch_copy_from_preprocess")
        mask_type = extra.get("mask_type", "sam_bbox")
        depth_type = extra.get("depth_type", "depth_anything_v3")
        mesh_type = extra.get("mesh_type", "cuboid500")
        split = cfg.split or "test"
        path_raw = cls._path_raw(cfg)
        path_pre = cls._path_preprocess(cfg)

        hint = (
            "Objectron's GCS bucket no longer serves anonymous callers; od3d's\n"
            "downloader needs `gcloud auth login`. Point extra.fetch_copy_from /\n"
            "extra.fetch_copy_from_preprocess at an existing tree instead."
        )

        # ── preprocessed subtrees, one at a time ─────────────────────────────
        # Per-subtree rather than one sync of path_preprocess: that tree is 8 GB
        # (it carries pcl/ and other od3d by-products this dataset never reads),
        # and on the cluster only meta/frames is actually missing.
        for sub in (
            "meta/frames",                    # poses, intrinsics, 3-D box corners
            f"mask/{mask_type}",
            f"depth/{depth_type}",
            f"mesh/{mesh_type}",
        ):
            fetch_or_copy(
                f"Objectron ({sub})",
                path_pre / sub,
                copy_from=(Path(copy_pre) / sub) if copy_pre else None,
                download=None,
                download_hint=hint,
                dry_run=dry_run,
            )

        # ── rgb: only the sampled frames ─────────────────────────────────────
        sampled = cls._sampled_frames(path_pre, mask_type, split)
        if not sampled:
            print("[Objectron (frames)] no mask tree to read the sampling from — "
                  "skipping the rgb copy")
            return

        missing = [
            f"frames/{cat}/{seq}/{frame}.jpg"
            for cat, seq, frame in sampled
            if not (path_raw / "frames" / cat / seq / f"{frame}.jpg").exists()
        ]
        print(f"[Objectron (frames)] {len(sampled)} sampled, {len(missing)} missing")
        if missing and copy_raw:
            rsync_files(Path(copy_raw), path_raw, missing,
                        label="Objectron rgb", dry_run=dry_run)
        elif missing:
            raise FetchSkipped(f"Objectron: {len(missing)} rgb frames missing.\n{hint}")
        else:
            print("[Objectron (frames)] already present — nothing to do.")
