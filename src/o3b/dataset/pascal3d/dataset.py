"""PASCAL3D+ — frame-object items from the od3d-preprocessed tree.

One object per frame.  od3d's ``extract-meta`` wrote one yaml per image under
``meta/frames/<subset>/<category>/<name>.yaml`` holding a single
``l_cam_tform4x4_obj`` / ``l_cam_intr4x4`` / ``l_bbox`` / ``l_kpts2d_annot``,
and this loader reads those directly.

Note what that single object means: od3d kept ``objects[0]`` of each PASCAL3D
annotation and dropped the rest (the ``assert len(objects) == 1`` in its
``Pascal3DFrameMeta.load_from_raw`` is commented out).  So an image with two
annotated aeroplanes contributes one item, not two.  That is reproduced here on
purpose — the published numbers were measured on exactly this set — and the
index carries an ``object_idx`` column so emitting every annotated object later
is a walk change rather than a schema change.

Depth defaults to ``depth_anything_v3`` — a monocular estimate.  The tree also
holds a CAD-mesh render (``depth/mesh/<mesh_type>/``) and ``depth_pro``, and
``extra.depth_type`` selects between them.  The estimate is the default because
ImageNet3D has no mesh-rendered depth at all, and a benchmark whose two
image-only columns disagreed about where depth came from would not be comparing
like with like.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import ItemType, register_dataset
from o3b.dataset.od3d_frames import Od3dFrameDataset
from o3b.dataset.pascal3d.enum import (
    MAP_CATEGORIES_PASCAL3D_TO_UCO3D,
    PASCAL3D_CATEGORIES,
)

# ftp:// is what od3d used; it is slow and frequently blocked by campus
# firewalls, which is precisely why fetch() prefers a local copy over it.
_DEFAULT_URL = "ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip"
_ZIP_TOP_LEVEL = "PASCAL3D+_release1.1"


@register_dataset("Pascal3D")
class Pascal3D(Od3dFrameDataset):
    """PASCAL3D+ as ``frame_object`` items."""

    all_categories = tuple(c.value for c in PASCAL3D_CATEGORIES)
    map_categories_to_uco3d = MAP_CATEGORIES_PASCAL3D_TO_UCO3D
    # PASCAL3D's object axes already agree with UCO3D's canonical right/top/back,
    # so there is no per-category re-orientation to apply — od3d defined none for
    # it either (unlike ImageNet3D and HANDAL, which both carry one).
    map_categories_obj_orient_to_uco3d = None

    # flat meta tree — see Od3dFrameDataset
    has_sequence_level = False

    # ── paths ────────────────────────────────────────────────────────────────

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return Path(cfg.path_raw or cfg.root)

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    # ── CLI hooks ────────────────────────────────────────────────────────────

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None, dry_run: bool = False) -> None:
        from o3b.dataset.od3d_fetch import download_zip, fetch_or_copy

        extra = dict(cfg.extra or {})
        copy_from = extra.get("fetch_copy_from")
        path_raw = cls._path_raw(cfg)
        download_url = url or extra.get("url_raw") or _DEFAULT_URL

        fetch_or_copy(
            "PASCAL3D+",
            path_raw,
            # Images/ and Annotations/ are what od3d's extract-meta reads; a tree
            # with only CAD/ present is a partial unzip, not a usable dataset.
            expect=("Images", "Annotations"),
            copy_from=Path(copy_from) if copy_from else None,
            download=lambda: download_zip(
                download_url, path_raw, strip_top_level=_ZIP_TOP_LEVEL,
            ),
            download_hint=(
                "Download PASCAL3D+_release1.1.zip by hand from\n"
                "  https://cvgl.stanford.edu/projects/pascal3d.html\n"
                f"and unpack it so that {path_raw}/Images exists."
            ),
            dry_run=dry_run,
        )
