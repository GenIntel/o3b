"""SPair-71k (Min et al., 2019) as ``frame_object_pair`` items, for 2-D semantic correspondence.

Raw tree (``path_raw``, the official release unpacked)::

    JPEGImages/<category>/<image>.jpg
    Segmentation/<category>/<image>.png       PASCAL instance ids (0 bg, 255 void)
    PairAnnotation/<trn|val|test>/<id>-<src>-<trg>:<category>.json

Preprocess tree (``path_preprocess``)::

    depth/<depth_type>/<category>/<image>.npz   depth (H, W) metric, intr (3, 3) pixels

SPair has no depth and no intrinsics; both come from a monocular estimate
(``o3b/dataset/spair/depth_da3.py``, Depth Anything 3), so a model that reads
RGB-D sees the depth a deployment without a sensor would give it.

The 2-D keypoints are carried as 3-D ones so that ``CamCrsp3DNNTask`` scores
them unchanged: each annotated pixel is lifted on its camera ray to the
estimated depth there, and the GT pose is the identity (``cam_tform4x4_obj`` =
``cam_tform4x4_obj_ncds`` = I, i.e. "object space" *is* camera space). The task
projects them back through the intrinsics, which lands exactly on the annotated
pixel whatever the depth, so its 2-D metrics (``cam_kpts2d_*``) are SPair's
PCK; its 3-D metrics are on the estimated depth and are not SPair's. With
``task.bbox2d_from: cam_bbox2d`` the threshold is 0.1 * max side of SPair's
annotated target box, the benchmark's PCK@0.1_bbox. All SPair keypoints are
annotated in both images, so ``obj_kpts2d_mask`` is True for the valid ones and
there is no amodal split.

The object mask is the instance of the segmentation that covers most of the
annotated box (``cam_bbox2d``), which is also what the crop is taken around.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from o3b.dataset.dataset import ConfigurableDataset, register_dataset

SPAIR_CATEGORIES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train",
    "tvmonitor",
)

# keypoints are padded to this many per pair (SPair's largest category has 25 ids;
# a pair annotates at most 20 of them in both images)
K_MAX = 32

_SPLIT_DIRS = {"train": "trn", "trn": "trn", "val": "val", "test": "test"}


def instance_mask(seg: np.ndarray, bbox) -> np.ndarray:
    """(H, W) bool: the segmentation instance covering most of the box ``bbox``
    (x0, y0, x1, y1, inclusive pixels). Empty when the box holds no instance."""
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    win = seg[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    ids, counts = np.unique(win[(win != 0) & (win != 255)], return_counts=True)
    if len(ids) == 0:
        return np.zeros(seg.shape, dtype=bool)
    return seg == ids[np.argmax(counts)]


def lift_px(uv: torch.Tensor, depth: torch.Tensor, intr3x3: torch.Tensor) -> torch.Tensor:
    """(K, 2) pixels -> (K, 3) OpenGL camera points at the depth under each pixel.

    Inverse of o3b's projection (proj3d2d_tform4x4_intr4x4_broadcast): u = fx X / -Z + cx,
    v = fy (-Y) / -Z + cy, with depth d = -Z > 0."""
    H, W = depth.shape
    iu = uv[:, 0].round().long().clamp(0, W - 1)
    iv = uv[:, 1].round().long().clamp(0, H - 1)
    d = depth[iv, iu].float()
    fx, fy, cx, cy = intr3x3[0, 0], intr3x3[1, 1], intr3x3[0, 2], intr3x3[1, 2]
    x = (uv[:, 0] - cx) / fx * d
    y = -(uv[:, 1] - cy) / fy * d
    return torch.stack([x, y, -d], dim=-1)


@register_dataset("SPair71K")
class SPair71K(ConfigurableDataset):
    """SPair-71k image pairs (``item_type: frame_object_pair``)."""

    categories = SPAIR_CATEGORIES

    def _setup(self) -> None:
        self.path_raw = Path(self.cfg.path_raw or self.cfg.root)
        self.path_preprocess = Path(self.cfg.path_preprocess or self.cfg.root)
        extra = dict(self.cfg.extra or {})
        self.depth_type = extra.get("depth_type", "da3nested_giant_large_1.1")
        split_dir = _SPLIT_DIRS.get(self.cfg.split, self.cfg.split)
        cats = set(self.cfg.categories) if self.cfg.categories else set(SPAIR_CATEGORIES)
        files = sorted((self.path_raw / "PairAnnotation" / split_dir).glob("*.json"))
        self._pairs = [f for f in files if f.stem.rsplit(":", 1)[-1] in cats]
        if self.cfg.filter_count_max:
            self._pairs = self._pairs[: int(self.cfg.filter_count_max)]

    def __len__(self) -> int:
        if self._sharded is not None:
            return len(self._sharded)
        return len(self._pairs)

    def _load_side(self, ann: dict, side: str):
        from o3b.data.datatypes.frame_object import FrameObject

        cat = ann["category"]
        imname = ann[f"{side}_imname"]
        stem = Path(imname).stem
        import torchvision.io as tio
        rgb = tio.read_image(str(self.path_raw / "JPEGImages" / cat / imname)).float() / 255.0
        if rgb.shape[0] == 1:
            rgb = rgb.expand(3, -1, -1)
        H, W = rgb.shape[-2:]

        dz = np.load(self.path_preprocess / "depth" / self.depth_type / cat / f"{stem}.npz")
        depth = torch.from_numpy(dz["depth"].astype(np.float32))
        intr3x3 = torch.from_numpy(dz["intr"].astype(np.float32))
        assert tuple(depth.shape) == (H, W), (imname, tuple(depth.shape), (H, W))
        intr = torch.eye(4)
        intr[:3, :3] = intr3x3
        depth_mask = torch.isfinite(depth) & (depth > 1e-6)
        depth = torch.where(depth_mask, depth, torch.zeros_like(depth))

        from PIL import Image
        seg = np.array(Image.open(self.path_raw / "Segmentation" / cat / f"{stem}.png"))
        bbox = ann["src_bndbox" if side == "src" else "trg_bndbox"]
        fo_mask = torch.from_numpy(instance_mask(seg, bbox))

        kps = torch.tensor(ann["src_kps" if side == "src" else "trg_kps"], dtype=torch.float32).reshape(-1, 2)
        K = kps.shape[0]
        kpts3d = torch.zeros(K_MAX, 3)
        kmask = torch.zeros(K_MAX, dtype=torch.bool)
        kpts3d[:K] = lift_px(kps, depth, intr3x3)
        kmask[:K] = True

        # metric extent of the object's lifted mask pixels (the 3-D PCK threshold;
        # not SPair's metric, which is 2-D)
        obj_size = 1.0
        m = fo_mask & depth_mask
        if m.any():
            vv, uu = torch.nonzero(m, as_tuple=True)
            pts = lift_px(torch.stack([uu, vv], -1).float(), depth, intr3x3)
            ext = pts.quantile(0.98, dim=0) - pts.quantile(0.02, dim=0)
            obj_size = float(ext.max().clamp(min=1e-3))

        eye = torch.eye(4)
        return FrameObject(
            frame_id=f"{cat}/{stem}",
            object_id=f"{cat}/{stem}",
            frame_object_id=f"{cat}/{stem}",
            cam_intr4x4=intr,
            rgb=rgb,
            depth=depth,
            depth_mask=depth_mask,
            fo_mask=fo_mask,
            cam_bbox2d=torch.tensor(bbox, dtype=torch.float32),
            cam_tform4x4_obj=eye.clone(),
            cam_tform4x4_obj_ncds=eye.clone(),
            obj_ncds0c_tform4x4_obj=eye.clone(),
            obj_kpts3d=kpts3d,
            obj_kpts3d_mask=kmask,
            obj_kpts2d_mask=kmask.clone(),
            obj_size=obj_size,
            obj_syms=torch.ones(3),
            category=SPAIR_CATEGORIES.index(cat),
            category_id=SPAIR_CATEGORIES.index(cat),
        )

    def _load_frame_object_pair(self, idx: int):
        from o3b.data.datatypes.frame_object import FrameObjectPair

        ann = json.loads(self._pairs[idx].read_text())
        src, trgt = self._load_side(ann, "src"), self._load_side(ann, "trg")
        return FrameObjectPair(src_object=src, trgt_object=trgt,
                               src_object_id=src.object_id, trgt_object_id=trgt.object_id)
