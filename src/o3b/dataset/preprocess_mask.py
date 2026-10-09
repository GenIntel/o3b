"""`o3b dataset preprocess-mask` — write an instance-mask tree the od3d way.

Only ``sam3_bbox`` is implemented: od3d's mask for Objectron since February
2026 (od3d ``od3d_datasets/objectron`` ``mask_type=SAM3``; the evaluation config
``objectron_test_sharded`` reads ``mask_type: "sam3_bbox"``). It is a port of
od3d's ``models_v2/sam3/model.py`` ``SAM3.forward_tracker`` — the path od3d's
``preprocess_mask`` takes for SAM3 with a box and no category — kept identical
in every choice that changes the mask:

* the SAM2-style ``Sam3TrackerModel`` of ``facebook/sam3``, not the text model
  (a box without a text concept goes to the tracker in od3d);
* prompts: the frame's 2-D box (``l_bbox`` from the meta, clamped to the
  image), 8 *negative* points on its corners and edge midpoints, and 1
  *positive* point at its centre;
* ``multimask_output=True``, bfloat16 autocast, and the candidate with the
  highest predicted IoU, thresholded at > 0;
* the full uint8 RGB frame, untransformed (od3d's SAM3 transform is empty);
* written as an 8-bit 0/255 PNG at ``mask/sam3_bbox/<name_unique>.png``, which
  is ``<split>/<category>[/<sequence>]/<frame>`` — the key both packages use.

Needs the ``sam3`` dependency group (``transformers`` 5, see setup_slurm.sh) and
the gated ``facebook/sam3`` weights in the HF cache; ``slurm_lmbl40_sam3`` sets
both. Run it where the data is:

    o3b dataset preprocess-mask -d objectron_test -p slurm_lmbl40_sam3 --remote
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MASK_TYPES = ("sam3_bbox",)


class Sam3BoxSegmenter:
    """od3d's ``SAM3.forward_tracker``: one box (+ 9 points) per image."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        # as od3d (following facebookresearch/sam3's batched-inference example)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        self.model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self.device).eval()
        self.processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

    @staticmethod
    def _points(bbox):
        """od3d's prompt points: 8 negatives on the box outline, 1 positive centre."""
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        pts = [[x1, y1], [x2, y1], [x1, y2], [x2, y2],
               [cx, y1], [x2, cy], [x1, cy], [cx, y2],
               [cx, cy]]
        labels = [0] * 8 + [1]
        return pts, labels

    def __call__(self, images, bboxs):
        """images: list of HxWx3 uint8 arrays; bboxs: list of [x1, y1, x2, y2].

        Returns a list of HxW bool masks and a list of float scores.
        """
        import torch

        boxes, points, labels = [], [], []
        for img, bbox in zip(images, bboxs):
            H, W = img.shape[:2]
            b = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            # od3d clamps x to [0, W-1] and y to [0, H-1] before prompting
            b = [min(max(b[0], 0), W - 1), min(max(b[1], 0), H - 1),
                 min(max(b[2], 0), W - 1), min(max(b[3], 0), H - 1)]
            p, l = self._points(b)
            boxes.append([b]); points.append([p]); labels.append([l])

        inputs = self.processor(images=images, input_boxes=boxes, input_points=points,
                                input_labels=labels, return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                                                    dtype=torch.bfloat16,
                                                    enabled=self.device.type == "cuda"):
            out = self.model(**inputs, multimask_output=True)
        masks_batch = self.processor.post_process_masks(
            out.pred_masks.float().detach().cpu(), inputs["original_sizes"].detach().cpu())
        scores_batch = out.iou_scores.float().detach().cpu()

        masks, scores = [], []
        for b in range(len(images)):
            lvl = scores_batch[b, 0].argmax()
            masks.append(masks_batch[b][0, lvl] > 0)
            scores.append(float(scores_batch[b, 0, lvl]))
        return masks, scores


def run_preprocess_mask(config_path: Path, *, platform: str = "default",
                        mask_type: str = "sam3_bbox", override: bool = False,
                        limit: int | None = None, batch_size: int = 1) -> None:
    """Write ``mask/<mask_type>/<rel_key>.png`` for every frame the config walks.

    batch_size 1 by default, as od3d segments (its DataLoader runs batch 1): it
    keeps the numerics identical and mixed image sizes out of one forward.
    """
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    from o3b.dataset.cli import _platform_to_dataset_overrides
    from o3b.dataset.dataset import DatasetConfig, build_dataset
    from o3b.dataset.od3d_frames import Od3dFrameDataset, load_meta_yaml

    if mask_type not in MASK_TYPES:
        print(f"mask type {mask_type!r} not implemented (have: {', '.join(MASK_TYPES)})",
              file=sys.stderr)
        sys.exit(1)

    # The walk only: no shard (it would hand out crops), no transform, and no
    # modality that would read the mask being written.
    overrides = _platform_to_dataset_overrides(platform) + [
        "sharded_name=null", "transform=null", "modalities=[rgb]"]
    cfg = DatasetConfig.from_yaml(Path(config_path), overrides=overrides)
    ds = build_dataset(cfg)
    if not isinstance(ds, Od3dFrameDataset):
        print(f"{type(ds).__name__} is not an od3d frame dataset — preprocess-mask "
              f"reads od3d's metas", file=sys.stderr)
        sys.exit(1)

    rows = list(ds._frame_rows)
    if limit:
        rows = rows[:limit]
    out_root = Path(ds.path_preprocess) / "mask" / mask_type
    todo = [r for r in rows
            if override or not (out_root / f"{ds._rel_key(r)}.png").exists()]
    print(f"{len(rows)} frames, {len(rows) - len(todo)} already written → "
          f"{len(todo)} to segment into {out_root}/")
    if not todo:
        return

    seg = Sam3BoxSegmenter()
    n_written = n_skipped = 0
    for i in tqdm(range(0, len(todo), batch_size), unit="batch"):
        imgs, bboxs, outs = [], [], []
        for row in todo[i:i + batch_size]:
            meta = load_meta_yaml(ds.meta_path(row))
            bbox = None
            if meta is not None:
                bbox = ds._cam_bbox2d_from_meta(meta)
                if bbox is None and meta.get("l_bbox"):
                    bbox = meta["l_bbox"]
            if meta is None or bbox is None or not meta.get("rfpath_rgb"):
                n_skipped += 1
                continue
            p = Path(ds.path_raw) / meta["rfpath_rgb"]
            if not p.exists():
                n_skipped += 1
                continue
            with Image.open(p) as im:
                imgs.append(np.asarray(im.convert("RGB")).copy())
            bboxs.append([float(v) for v in (bbox.tolist() if hasattr(bbox, "tolist") else bbox)])
            outs.append(out_root / f"{ds._rel_key(row)}.png")
        if not imgs:
            continue
        masks, _ = seg(imgs, bboxs)
        for mask, path in zip(masks, outs):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.numpy().astype(np.uint8) * 255).save(path)
            n_written += 1
    print(f"Done. {n_written} masks written, {n_skipped} frames skipped "
          f"(no meta, box or rgb) → {out_root}/")
