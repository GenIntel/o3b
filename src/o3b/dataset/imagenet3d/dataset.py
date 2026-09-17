"""ImageNet3D — frame-object items from the od3d-preprocessed tree.

One object per frame, and here that is literal rather than truncated: od3d's
``extract-meta`` iterates ``for obj_id, annotation in enumerate(annotations)``
and writes ``<image>_<obj_id>.yaml``, so every annotated object of an image
becomes its own frame.  Contrast PASCAL3D, which keeps only the first.

Each meta additionally carries ``object_status`` (``status_good`` /
``status_partially`` / ``status_barely`` / ``status_bad``).  od3d's default
allows all four; ``extra.object_status_allow`` narrows it, which is the only
knob that changes how many items a category yields.

Depth is ``depth_anything_v3``, a monocular estimate — the only depth this
tree has; unlike PASCAL3D there is no CAD-mesh render to fall back on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import ItemType, register_dataset
from o3b.dataset.od3d_frames import Od3dFrameDataset
from o3b.dataset.imagenet3d.enum import (
    IMAGENET3D_CATEGORIES,
    MAP_CATEGORIES_IMAGENET3D_TO_UCO3D,
    MAP_CATEGORIES_OBJ_ORIENT_IMAGENET3D_TO_UCO3D,
)

_DEFAULT_URL = "https://huggingface.co/datasets/ccvl/ImageNet3D/resolve/main/imagenet3d_v1.zip"


@register_dataset("ImageNet3D")
class ImageNet3D(Od3dFrameDataset):
    """ImageNet3D as ``frame_object`` items."""

    all_categories = tuple(c.value for c in IMAGENET3D_CATEGORIES)
    map_categories_to_uco3d = MAP_CATEGORIES_IMAGENET3D_TO_UCO3D
    # Per-category rotation into UCO3D's canonical right/top/back axes.  Without
    # it a model trained on Every9D is judged against a different axis
    # convention per category and scores near chance on the ones that differ.
    map_categories_obj_orient_to_uco3d = MAP_CATEGORIES_OBJ_ORIENT_IMAGENET3D_TO_UCO3D

    # flat meta tree — see Od3dFrameDataset
    has_sequence_level = False

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return Path(cfg.path_raw or cfg.root)

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None, dry_run: bool = False) -> None:
        from o3b.dataset.od3d_fetch import download_zip, fetch_or_copy

        extra = dict(cfg.extra or {})
        copy_from = extra.get("fetch_copy_from")
        path_raw = cls._path_raw(cfg)
        download_url = url or extra.get("url_raw") or _DEFAULT_URL

        fetch_or_copy(
            "ImageNet3D",
            path_raw,
            expect=("imagenet3d_v1/annotations", "imagenet3d_v1/images"),
            copy_from=Path(copy_from) if copy_from else None,
            download=lambda: download_zip(download_url, path_raw),
            download_hint=(
                f"Download {_DEFAULT_URL} by hand and unpack it so that\n"
                f"  {path_raw}/imagenet3d_v1/annotations exists."
            ),
            dry_run=dry_run,
        )
