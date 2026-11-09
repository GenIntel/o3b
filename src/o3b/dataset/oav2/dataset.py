"""Orient Anything V2's training renders — frame-object items from od3d's tree.

The raw data is the Hugging Face dataset ``Viglong/OriAnyV2_Train_Render``:
3,720 category directories, one directory per asset, each with 5 canonical
renders (``cano_dark{0..4}.jpg``: left / right / back / front / top) and N
random views (``random_dark{i}.jpg``) on a black background, ~1.2M frames.
od3d's ``extract-meta`` (od3d a61b67d0 "fix oav2 train data" and later) wrote
one meta per render under ``meta/frames/train/<category>/<asset>_<view>.yaml``
and od3d's preprocess a Depth Anything V3 estimate under
``depth/depth_anything_v3/train/<category>/<asset>_<view>.png``. This loader
reads that tree exactly as od3d's ``OrientAnythingV2`` dataset did:

* pose: the meta's ``l_cam_tform4x4_obj``, od3d's ``transf4x4_from_spherical``
  at a fixed distance 2 (azimuth / elevation only, no in-plane angle). Its
  object frame is od3d's x = left, y = back, z = top — UCO3D's raw labelling —
  so the shared ``obj_gl_tform4x4_obj_raw`` alone takes it to o3b's
  right / top / back. No per-category re-orientation: Orient Anything's own
  front / up convention is the label.
* intrinsics: fixed, 40 degree field of view on 512 x 512.
* mask: od3d's ``OrientAnythingV2Frame.read_mask`` — any RGB channel above 5
  (of 255), since the renders sit on black. No mask tree is read.
* geometry: none. Every asset is normalised into a unit cube before rendering,
  so od3d gives every object ``obj_size3d = (1, 1, 1)``, and so does this
  loader. Size and translation therefore carry no metric information; train
  rotation only (as for ImageNet3D).
* symmetry: none labelled (od3d used none), ``obj_syms = (1, 1, 1)``.

The object behind a frame is its asset, so the walk groups the flat meta tree
by asset (``sequence``), which makes ``object_id`` = ``<category>/<asset>``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import register_dataset
from o3b.dataset.od3d_frames import Od3dFrameDataset

#: od3d's read_mask threshold: foreground is any channel above this (0..255)
_MASK_RGB_THRESHOLD = 5.0
#: every asset is rendered normalised into a unit cube (od3d's OBJ_SIZE3D)
_OBJ_SIZE3D = (1.0, 1.0, 1.0)
#: <asset>_<cano|random>_dark<i>; asset ids may themselves contain underscores
_NAME_RE = re.compile(r"^(?P<asset>.+)_(?:cano|random)_dark\d+$")


@register_dataset("OrientAnythingV2")
class OrientAnythingV2(Od3dFrameDataset):
    """Orient Anything V2's training renders as ``frame_object`` items."""

    # 3,720 free-form category names, discovered on disk by the walk; none of
    # them is mapped onto UCO3D's vocabulary (see _uco3d_category)
    all_categories = ()
    has_sequence_level = False

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return Path(cfg.path_raw or cfg.root)

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    # ── walk: group the flat tree by asset ───────────────────────────────────

    def _row(self, split: str, category: str, sequence: str, frame: str) -> dict:
        # the flat walk passes sequence == frame; the asset is the object
        m = _NAME_RE.match(frame)
        return super()._row(split, category, m.group("asset") if m else frame, frame)

    # ── per item ─────────────────────────────────────────────────────────────

    def _uco3d_category(self, category: Optional[str]) -> Optional[str]:
        # No UCO3D category: the base class would otherwise treat an unmapped
        # name as UCO3D's own and hand e.g. "apple" UCO3D's symmetries, while
        # most categories get none. Only the shared T_gl is applied.
        return None

    def _object_geometry(self, row, meta):
        import torch

        from o3b.dataset.obj_frame import _corners_canonical

        size3d = torch.tensor(_OBJ_SIZE3D, dtype=torch.float32)
        half = size3d / 2
        tform = torch.eye(4, dtype=torch.float32)
        tform[:3, :3] = torch.eye(3) * float(size3d.max()) * 0.5
        return None, size3d, _corners_canonical(-half, half), tform

    def _load_frame_object(self, idx: int):
        import torch

        from o3b.dataset.utils import want as _want

        item = super()._load_frame_object(idx)
        if item is None:
            return None
        mods = self.cfg.modalities
        fo_mask = (item.rgb * 255.0 > _MASK_RGB_THRESHOLD).any(dim=0)
        if not bool(fo_mask.any()):
            return None                       # an empty render: nothing to pose
        if _want("fo_mask", mods):
            item.fo_mask = fo_mask
        if _want("cam_bbox2d", mods) and item.cam_bbox2d is None:
            from o3b.cv.visual.draw import get_bboxs_from_masks
            item.cam_bbox2d = get_bboxs_from_masks(fo_mask[None])[0].float()
        if _want("obj_syms", mods):
            item.obj_syms = torch.ones(3, dtype=torch.long)    # 1 = no symmetry
        return item
