"""`o3b dataset preprocess-mask` — write an instance-mask tree the od3d way.

Only ``sam3_bbox`` is implemented: od3d's mask for Objectron since February
2026 (od3d ``od3d_datasets/objectron`` ``mask_type=SAM3``; the evaluation config
``objectron_test_sharded`` reads ``mask_type: "sam3_bbox"``). It is a port of
od3d's ``models_v2/sam3/model.py`` ``SAM3.forward_tracker`` — the path od3d's
``preprocess_mask`` takes for SAM3 with a box and no category — kept identical
in every choice that changes the mask:

* the SAM2-style ``Sam3TrackerModel`` of ``facebook/sam3``, not the text model
  (a box without a text concept goes to the tracker in od3d);
* prompts: the frame's 2-D box as od3d derives it
  (``_mask_prompt_bbox_from_meta`` — for Objectron the min/max of the
  projected box corners), clamped to the image; 8 *negative* points on its
  corners and edge midpoints, and 1 *positive* point at its centre;
* ``multimask_output=True``, bfloat16 autocast, and the candidate with the
  highest predicted IoU, thresholded at > 0;
* the full uint8 RGB frame, untransformed (od3d's SAM3 transform is empty);
* written as an 8-bit 0/255 PNG at ``mask/sam3_bbox/<name_unique>.png``, which
  is ``<split>/<category>[/<sequence>]/<frame>`` — the key both packages use.

``sam3_text`` (H12) prompts SAM 3's text model (``Sam3Model``, promptable
concept segmentation) with the frame's category name and nothing else — no box,
no point — and keeps the highest-scoring instance (an empty mask when none
passes the score threshold, 0 by default — see the class). The name is ``_mask_text_prompt(row)``: for
HANDAL the category directory as AffordanceNet words its prompt ("power
drills"), so the baselines and this mask see the same text. It is the test-time
object mask for a "no ground truth at test time" protocol, which a box prompt
from the annotated box would not be.

``gdino_sam2_text`` (H12, SAM 2 instead of SAM 3) is the same text-only setting
for SAM 2, which takes no text: Grounding DINO (``IDEA-Research/grounding-dino-base``)
finds the category's highest-scoring box from the same text, and SAM 2.1
(``facebook/sam2.1-hiera-large``) segments that box, single-mask output —
"Grounded SAM 2". Both are in transformers 5, so the ``sam3`` env runs it.

``sam3p1_text`` (H12) is ``sam3_text`` with SAM 3.1's detector weights and
nothing else changed (same ``Sam3Model`` code, processor, prompt and top-instance
rule). SAM 3.1 (``facebook/sam3.1``) ships only the native
``sam3.1_multiplex.pt``, whose detector is a retrained SAM 3 detector (every
tensor differs by 1-3 %; the neck's unused fourth level and the tracker necks
are gone). transformers has no port, so the weights are converted once into a
``Sam3Model`` directory — the native->HF key map read off SAM 3, where both
formats hold the same tensors — and ``$O3B_SAM3P1_HF`` points at it.

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

MASK_TYPES = ("sam3_bbox", "sam3_text", "sam3p1_text", "gdino_sam2_text")


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


class Sam3TextSegmenter:
    """SAM 3's text model: the instance with the highest score for a concept."""

    # score_threshold 0: always the best instance. Sam3's post-processing gates
    # every query's score by one per-image presence score, which on HANDAL frames
    # sits low enough that a wrench whose best query scores 0.81-0.88 returned no
    # instance even at 0.1. The frames are known to show the category (the
    # protocol's premise, which every method shares), and the gate is the same for
    # all queries of an image, so the top instance is the top query either way.
    def __init__(self, device: str = "cuda", score_threshold: float = 0.0,
                 mask_threshold: float = 0.5, weights: str = "facebook/sam3"):
        import torch
        from transformers import Sam3Model, Sam3Processor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        self.model = Sam3Model.from_pretrained(weights).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(weights)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

    def __call__(self, images, texts):
        """images: list of HxWx3 uint8 arrays; texts: one concept per image.

        Returns a list of HxW bool masks and a list of float scores (0 and an
        empty mask where nothing passes the threshold).
        """
        import torch

        masks, scores = [], []
        for img, text in zip(images, texts):
            inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                out, threshold=self.score_threshold, mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            if len(res["masks"]) == 0:
                masks.append(torch.zeros(img.shape[:2], dtype=torch.bool))
                scores.append(0.0)
                continue
            i = int(torch.as_tensor(res["scores"]).argmax())
            masks.append(torch.as_tensor(res["masks"][i]).cpu() > 0)
            scores.append(float(res["scores"][i]))
        return masks, scores


class GroundedSam2Segmenter:
    """Grounding DINO's top box for a concept, segmented by SAM 2.1 (box prompt).

    Always the top box (no score threshold), for the reason Sam3TextSegmenter
    keeps its top instance: the frames are known to show the category. No box at
    all (no query scores above zero) gives an empty mask.
    """

    def __init__(self, device: str = "cuda",
                 detector: str = "IDEA-Research/grounding-dino-base",
                 segmenter: str = "facebook/sam2.1-hiera-large"):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(self.device).eval()
        self.det_processor = AutoProcessor.from_pretrained(detector)
        self.sam = Sam2Model.from_pretrained(segmenter).to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(segmenter)

    def _box(self, img, text):
        import torch
        from PIL import Image
        H, W = img.shape[:2]
        # Grounding DINO wants lower-case phrases ending with a period
        prompt = text.lower().strip().rstrip(".") + "."
        inputs = self.det_processor(images=Image.fromarray(img), text=prompt,
                                    return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.det(**inputs)
        res = self.det_processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.0, text_threshold=0.0, target_sizes=[(H, W)])[0]
        if len(res["scores"]) == 0:
            return None, 0.0
        i = int(res["scores"].argmax())
        b = res["boxes"][i].tolist()
        b = [min(max(b[0], 0), W - 1), min(max(b[1], 0), H - 1),
             min(max(b[2], 0), W - 1), min(max(b[3], 0), H - 1)]
        return (b if b[2] > b[0] and b[3] > b[1] else None), float(res["scores"][i])

    def __call__(self, images, texts):
        """images: list of HxWx3 uint8 arrays; texts: one concept per image."""
        import torch

        masks, scores = [], []
        for img, text in zip(images, texts):
            box, score = self._box(img, text)
            if box is None:
                masks.append(torch.zeros(img.shape[:2], dtype=torch.bool))
                scores.append(0.0)
                continue
            inputs = self.sam_processor(images=img, input_boxes=[[box]],
                                        return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.sam(**inputs, multimask_output=False)
            m = self.sam_processor.post_process_masks(
                out.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
            masks.append(m.reshape(-1, *m.shape[-2:])[0] > 0)
            scores.append(score)
        return masks, scores


def run_preprocess_mask(config_path: Path, *, platform: str = "default",
                        mask_type: str = "sam3_bbox", override: bool = False,
                        limit: int | None = None, batch_size: int = 1,
                        shard: int = 0, num_shards: int = 1) -> None:
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
    if num_shards > 1:
        rows = rows[shard::num_shards]
    out_root = Path(ds.path_preprocess) / "mask" / mask_type
    todo = [r for r in rows
            if override or not (out_root / f"{ds._rel_key(r)}.png").exists()]
    print(f"{len(rows)} frames, {len(rows) - len(todo)} already written → "
          f"{len(todo)} to segment into {out_root}/")
    if not todo:
        return

    text_mode = mask_type in ("sam3_text", "sam3p1_text", "gdino_sam2_text")
    if mask_type == "sam3p1_text":
        import os
        weights = os.environ.get("O3B_SAM3P1_HF")
        if not weights or not (Path(weights) / "model.safetensors").exists():
            print("sam3p1_text needs $O3B_SAM3P1_HF = a Sam3Model directory holding "
                  "SAM 3.1's converted weights (see the module docstring)", file=sys.stderr)
            sys.exit(1)
    seg = (Sam3TextSegmenter() if mask_type == "sam3_text" else
           Sam3TextSegmenter(weights=weights) if mask_type == "sam3p1_text" else
           GroundedSam2Segmenter() if mask_type == "gdino_sam2_text" else Sam3BoxSegmenter())
    n_written = n_skipped = 0
    for i in tqdm(range(0, len(todo), batch_size), unit="batch"):
        imgs, bboxs, outs = [], [], []
        for row in todo[i:i + batch_size]:
            meta = load_meta_yaml(ds.meta_path(row))
            if text_mode:
                bbox = ds._mask_text_prompt(row) if meta is not None else None
            else:
                bbox = ds._mask_prompt_bbox_from_meta(meta) if meta is not None else None
            if meta is None or bbox is None or not meta.get("rfpath_rgb"):
                n_skipped += 1
                continue
            p = Path(ds.path_raw) / meta["rfpath_rgb"]
            if not p.exists():
                n_skipped += 1
                continue
            with Image.open(p) as im:
                imgs.append(np.asarray(im.convert("RGB")).copy())
            bboxs.append(bbox if text_mode else
                         [float(v) for v in (bbox.tolist() if hasattr(bbox, "tolist") else bbox)])
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
