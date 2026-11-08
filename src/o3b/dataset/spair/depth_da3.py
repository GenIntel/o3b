"""Monocular depth + intrinsics for SPair-71k images with Depth Anything 3.

Standalone (numpy, PIL, torch and ``depth_anything_3`` only), so it runs in its
own env::

    python depth_da3.py --raw <SPair71K> --out <SPair71K_Preprocess>/depth/da3nested_giant_large_1.1 \
        --splits test [--model depth-anything/da3nested-giant-large-1.1]

Writes ``<out>/<category>/<image stem>.npz`` with ``depth`` (H, W) float32 in
metres at the image's own resolution and ``intr`` (3, 3), the model's estimated
intrinsics rescaled to that resolution, plus ``conf``. The same model od3d used
for ImageNet3D / PASCAL3D (od3d_datasets/dataset.py), but without given
intrinsics: SPair has none, so DA3 estimates them as well.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    images = set()
    for split in args.splits.split(","):
        for f in sorted((args.raw / "PairAnnotation" / split).glob("*.json")):
            a = json.loads(f.read_text())
            images.add((a["category"], a["src_imname"]))
            images.add((a["category"], a["trg_imname"]))
    images = sorted(images)
    print(f"{len(images)} images", flush=True)

    import torch
    from PIL import Image
    from depth_anything_3.api import DepthAnything3

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model).to(device=dev).eval()

    for i, (cat, imname) in enumerate(images):
        out = args.out / cat / f"{Path(imname).stem}.npz"
        if out.exists() and not args.overwrite:
            continue
        img = Image.open(args.raw / "JPEGImages" / cat / imname).convert("RGB")
        W, H = img.size
        with torch.no_grad():
            pred = model.inference([img])
        d = np.asarray(pred.depth[0], dtype=np.float32)             # (h, w) processed res
        K = np.asarray(pred.intrinsics[0], dtype=np.float64).copy()  # (3, 3) at (h, w)
        h, w = d.shape
        sx, sy = W / w, H / h
        # pixel-centre-consistent rescale (u + 0.5) -> (u + 0.5) * s
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        dt = torch.from_numpy(d)[None, None]
        depth = torch.nn.functional.interpolate(dt, size=(H, W), mode="nearest")[0, 0].numpy()
        conf = None
        if getattr(pred, "conf", None) is not None:
            c = torch.from_numpy(np.asarray(pred.conf[0], dtype=np.float32))[None, None]
            conf = torch.nn.functional.interpolate(c, size=(H, W), mode="nearest")[0, 0].numpy()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, depth=depth.astype(np.float32), intr=K.astype(np.float32),
                            **({"conf": conf.astype(np.float16)} if conf is not None else {}))
        if i < 3 or i % 50 == 0:
            print(f"{i}/{len(images)} {cat}/{imname} {W}x{H} proc {w}x{h} "
                  f"f=({K[0,0]:.1f},{K[1,1]:.1f}) c=({K[0,2]:.1f},{K[1,2]:.1f}) "
                  f"depth med {np.median(depth):.2f} m", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
