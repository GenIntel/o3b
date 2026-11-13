"""HouseCat6D (Jung et al., CVPR 2024) — frame-object items, test scenes (H16-T02).

Raw tree (https://sites.google.com/view/housecat6d/dataset, ``test_scene.zip`` +
``obj_models.zip``)::

    test_scene<N>/rgb/<frame>.png              1096 x 852
    test_scene<N>/depth/<frame>.png            uint16 millimetres (D435-style sensor depth)
    test_scene<N>/instance/<frame>_<obj>.png   one mask per object: object = 1, background = 255
    test_scene<N>/labels/<frame>_label.pkl     model_list, rotations, translations, gt_scales, ...
    test_scene<N>/intrinsics.txt               3 x 3
    obj_models_small_size_final/<cat>/<obj>.obj

The poses are NOCS-style (``scales`` is the box diagonal), but the meshes are metric, centred and
their bounding box equals ``gt_scales``, so ``cam <- R @ mesh + t`` is a rigid metric pose in an
OpenCV camera; checked on test_scene1 (93-98 % of projected mesh points inside the mask, depth
agrees).

:func:`prepare` turns that into od3d's sequenced meta tree, so the shared
:class:`Od3dFrameDataset` walk / index / crops / sharding apply unchanged:

    meta/frames/test/<category>/<scene>__<object>/<frame>.yaml
    mask/gt/test/<category>/<scene>__<object>/<frame>.png   (object = 255)

One sequence is one object in one scene. Objects with fewer than ``min_mask_px`` visible pixels
in a frame are skipped there.

Object frame: HouseCat6D's own (NOCS convention, y up), not UCO3D's (``map_to_uco3d: false``).
Symmetry per category (``obj_syms``, o3b's per-axis code: -1 continuous, 1 none) follows the NOCS /
HouseCat6D evaluation: bottle, can and glass are continuous about y, everything else has none.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import register_dataset
from o3b.dataset.od3d_frames import Od3dFrameDataset

logger = logging.getLogger(__name__)

HOUSECAT6D_CATEGORIES = ("bottle", "box", "can", "cup", "remote", "teapot", "cutlery", "glass", "shoe", "tube")

#: per-axis symmetry code in HouseCat6D's object frame (x, y, z)
HOUSECAT6D_OBJ_SYMS = {c: (1, 1, 1) for c in HOUSECAT6D_CATEGORIES}
HOUSECAT6D_OBJ_SYMS.update({c: (1, -1, 1) for c in ("bottle", "can", "glass")})

MESH_DIR = "obj_models_small_size_final"


@register_dataset("HouseCat6D")
class HouseCat6D(Od3dFrameDataset):
    """HouseCat6D test scenes as ``frame_object`` items."""

    all_categories = HOUSECAT6D_CATEGORIES
    has_sequence_level = True

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return Path(cfg.path_raw or cfg.root)

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return Path(cfg.path_preprocess or cfg.root)

    def __getitem__(self, idx: int):
        import torch

        item = super().__getitem__(idx)
        if item is not None and isinstance(getattr(item, "category", None), str):
            syms = HOUSECAT6D_OBJ_SYMS.get(item.category)
            if syms is not None:
                item.obj_syms = torch.tensor(syms, dtype=torch.long)
        return item


def prepare(path_raw: Path, path_preprocess: Path, *, split: str = "test", min_mask_px: int = 50,
            scenes: Optional[list[str]] = None) -> dict:
    """Write the od3d meta tree + inverted GT masks for every (scene, frame, object).

    Returns counts per category. Idempotent: existing files are overwritten.
    """
    import pickle

    import cv2
    import numpy as np
    import yaml

    path_raw, path_preprocess = Path(path_raw), Path(path_preprocess)
    scene_dirs = sorted(p for p in path_raw.glob(f"{split}_scene*") if p.is_dir())
    if scenes:
        scene_dirs = [p for p in scene_dirs if p.name in scenes]
    if not scene_dirs:
        raise FileNotFoundError(f"no {split}_scene* under {path_raw}")
    counts: dict = {}
    for sd in scene_dirs:
        K = np.loadtxt(sd / "intrinsics.txt")
        K4 = np.eye(4)
        K4[:3, :3] = K
        for lp in sorted((sd / "labels").glob("*_label.pkl")):
            frame = lp.name.split("_")[0]
            with open(lp, "rb") as f:
                lab = pickle.load(f)
            rgb = sd / "rgb" / f"{frame}.png"
            depth = sd / "depth" / f"{frame}.png"
            if not rgb.exists() or not depth.exists():
                continue
            for i, name in enumerate(lab["model_list"]):
                name = str(name)
                cat = name.split("-")[0]
                mp = sd / "instance" / f"{frame}_{name}.png"
                if cat not in HOUSECAT6D_CATEGORIES or not mp.exists():
                    continue
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                obj = m < 128
                n = int(obj.sum())
                if n < min_mask_px:
                    continue
                ys, xs = np.nonzero(obj)
                seq = f"{sd.name}__{name}"
                rel = Path(split) / cat / seq / frame
                out_mask = path_preprocess / "mask" / "gt" / f"{rel}.png"
                out_mask.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_mask), obj.astype(np.uint8) * 255)
                T = np.eye(4)
                T[:3, :3] = np.asarray(lab["rotations"][i], dtype=np.float64)
                T[:3, 3] = np.asarray(lab["translations"][i], dtype=np.float64)
                meta = {
                    "name": frame,
                    "sequence_name": seq,
                    "subset": split,
                    "category": cat,
                    "l_size": [float(m.shape[0]), float(m.shape[1])],
                    "rfpath_rgb": str(rgb.relative_to(path_raw)),
                    "rfpath_depth": str(depth.relative_to(path_raw)),
                    "rfpath_mesh": f"{MESH_DIR}/{cat}/{name}.obj",
                    "l_bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    "l_cam_tform4x4_obj": T.tolist(),
                    "l_cam_intr4x4": K4.tolist(),
                    "depth_scale": 1000.0,
                    "l_obj_size3d": [float(s) for s in lab["gt_scales"][i]],
                    "mask_px": n,
                }
                out_meta = path_preprocess / "meta" / "frames" / f"{rel}.yaml"
                out_meta.parent.mkdir(parents=True, exist_ok=True)
                out_meta.write_text(yaml.safe_dump(meta, sort_keys=False))
                counts[cat] = counts.get(cat, 0) + 1
        logger.info(f"{sd.name}: {sum(counts.values())} items so far")
    return counts
