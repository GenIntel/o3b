"""Draw the objects to annotate: one sequence from each of N different categories.

    o3b dataset annotate-obb-sample -d uco3d --source every9d_v7_test --source every9d_v7_train

Categories holding an already labelled sequence (``annotations_old``, with a box)
come first and keep that sequence, so existing work is reused; the remaining
categories are drawn at random, each contributing one random sequence.  An
existing list is only replaced on request: once annotation has started, a changed
list no longer is the sample.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from o3b.dataset import subset_store

SEED = 9
COUNT = 100


def sample_name(source: str, count: int = COUNT) -> str:
    return f"{source}_verify{count}"


def draw(ids, count: int = COUNT, seed: int = SEED, preferred=()) -> list:
    """``count`` ids from ``count`` different categories, ``preferred`` ids first.

    Independent of the input order.  A category with several preferred ids keeps
    one of them at random.
    """
    rng = np.random.default_rng(seed)
    by_category: dict = {}
    for i in sorted(set(ids)):
        by_category.setdefault(i.split("/")[0], []).append(i)
    if len(by_category) < count:
        raise ValueError(f"cannot draw {count} categories from {len(by_category)}")
    preferred = set(preferred) & set(ids)
    labelled = sorted({i.split("/")[0] for i in preferred})
    rest = [c for c in sorted(by_category) if c not in preferred and c not in labelled]
    picked_rest = [rest[int(i)] for i in rng.permutation(len(rest))]
    categories = (labelled + picked_rest)[:count]
    out = []
    for c in categories:
        pool = sorted(i for i in by_category[c] if i in preferred) or by_category[c]
        out.append(pool[int(rng.integers(len(pool)))])
    return sorted(out)


def run_sample(path_preprocess, root, sources, count: int = COUNT, seed: int = SEED,
               replace: bool = False) -> list:
    """Write ``<source>_verify<count>`` for each source. Returns the sample names."""
    from o3b.dataset.verify_labels import store

    if replace and any(store.iter_all(root, store.ANNOTATIONS)):
        raise FileExistsError(f"{root} already holds annotations; not replacing the sample under them")
    preferred = {r["sequence_name"] for r in store.iter_all(root, store.OLD) if r.get("center_raw")}
    names, drawn = [], {}
    for source in sources:
        name = sample_name(source, count)
        if subset_store.exists(path_preprocess, name) and not replace:
            raise FileExistsError(f"subset {name!r} already exists; pass --replace to redraw it")
        drawn[name] = draw(subset_store.load(path_preprocess, source).ids, count, seed, preferred)
        names.append(name)
    seen = set()
    for name, ids in drawn.items():
        if seen & set(ids):
            raise ValueError(f"{name} overlaps another sample: {sorted(seen & set(ids))[:3]}")
        seen |= set(ids)
    out = Path(root) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for source, (name, ids) in zip(sources, drawn.items()):
        p = subset_store.save(path_preprocess, name, ids,
                              header={"source": source, "seed": seed, "per_category": 1,
                                      "old_labels": len(preferred & set(ids))})
        shutil.copyfile(p, out / p.name)
        print(f"{name}: {len(ids)} sequences over "
              f"{len({i.split('/')[0] for i in ids})} categories, "
              f"{len(preferred & set(ids))} with an old label -> {p}")
    return names
