"""Small dataset-agnostic helpers shared by the dataset loaders.

These used to live in ``o3b.dataset.housecorr3d.dataset``, which meant every
other loader that wanted them (DenseMatcher does) had to reach into a sibling
dataset's module.  Nothing here knows about a particular dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """``urllib.request.urlretrieve`` reporthook drawing a one-line progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()


def want(name: str, mods: Optional[set]) -> bool:
    """True if modality *name* should be loaded (``mods=None`` means "all")."""
    return mods is None or name in mods


def load_kpts3d_by_id(
    obj_id: str,
    path_preprocess: Path,
    tform: Optional[torch.Tensor],
):
    """Load ``<path_preprocess>/obj_kpts3d/<obj_id>/kpts3d.pt`` → (kpts3d, mask).

    The file holds a (K, 4) tensor: xyz plus a validity flag.  *tform* is an
    NCDS-0c normalisation transform (isotropic scale + centre shift); when given,
    the keypoints are mapped into that normalised frame.  Returns ``(None, None)``
    if the file is missing or unreadable — an object without annotated keypoints
    is normal, not an error.
    """
    kpts_file = path_preprocess / "obj_kpts3d" / obj_id / "kpts3d.pt"
    if not kpts_file.exists():
        return None, None
    try:
        t = torch.load(kpts_file, map_location="cpu")  # (K, 4)
        kpts3d = t[:, :3].float()
        mask   = t[:, 3].bool()
        if tform is not None:
            scale  = tform[0, 0]
            center = tform[:3, 3]
            kpts3d = (kpts3d - center) / scale
        return kpts3d, mask
    except Exception:
        return None, None
