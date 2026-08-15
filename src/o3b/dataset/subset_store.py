"""Named subsets — a hand-picked list of ids stored next to the dataset.

    <path_preprocess>/subset/<name>.yaml

A YAML list, one id per entry, sorted on write::

    # o3b subset "every9d_v3_test"
    # dataset: every9d_v3   tform_obj_type: every9d_v3
    - apple/1-2-3
    - apple/4-5-6

A dataset config names one through ``subset_name:`` and then contains only what
the file lists.  An entry is matched against an item's **object id** or its
**frame id**, so the same mechanism expresses both granularities: for UCO3D
``apple/1-2-3`` selects a whole sequence, ``apple/1-2-3/66`` selects one frame
of it.

This is the directory od3d already writes its own subsets into, as YAML lists of
frame ids in files named ``<name>yaml`` — no dot, from an f-string that forgot
one.  Those are read as they are (``load`` tries ``<name>.yaml``, then
``<name>yaml``, then ``<name>.txt``, then the literal name), so an od3d subset is
usable under ``subset_name`` without being renamed; o3b writes the dotted form.
Plain one-id-per-line text is accepted too, for a file written by hand.

``o3b dataset select-subset -d every9d_v3 -s every9d_v3_test`` is the GUI that
writes these; nothing stops a script or an editor from writing one instead.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_SUBSET_PARENT = "subset"

# Tried in order for `-s <name>`; "" is the name used verbatim.
_SUFFIXES = (".yaml", "yaml", ".txt", "")


def dir_path(path_preprocess) -> Path:
    return Path(path_preprocess) / _SUBSET_PARENT


def _check_name(name: str) -> str:
    if "/" in name or name in ("", ".", ".."):
        raise ValueError(f"invalid subset name {name!r}: it names one file under "
                         f"{_SUBSET_PARENT}/, not a path")
    return name


def path(path_preprocess, name: str) -> Path:
    """Where o3b *writes* the subset — the dotted .yaml form."""
    return dir_path(path_preprocess) / f"{_check_name(name)}.yaml"


def find(path_preprocess, name: str) -> Optional[Path]:
    """The file backing *name*, whichever spelling it has on disk."""
    d = dir_path(path_preprocess)
    for suffix in _SUFFIXES:
        candidate = d / f"{_check_name(name)}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def exists(path_preprocess, name: str) -> bool:
    return find(path_preprocess, name) is not None


def list_names(path_preprocess) -> list[str]:
    """Names usable with ``-s`` / ``subset_name``, from whatever is in the dir."""
    d = dir_path(path_preprocess)
    if not d.is_dir():
        return []
    names = set()
    for p in d.iterdir():
        if not p.is_file():
            continue
        n = p.name
        for suffix in (".yaml", "yaml", ".txt"):
            if n.endswith(suffix) and len(n) > len(suffix):
                n = n[: -len(suffix)]
                break
        names.add(n)
    return sorted(names)


# ── the selection itself ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Subset:
    """The ids a subset lists, and the two questions a loader asks of them."""

    name: str
    ids: frozenset
    # Each entry's parent id — 'category/sequence' for a
    # 'category/sequence/frame' one — which is what lets a walk skip a whole
    # sequence without first enumerating its frames.  An entry that is already
    # an object id contributes its category here, and no object id is a bare
    # category, so that extra never matches anything.
    object_ids: frozenset

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, item: str) -> bool:
        return item in self.ids

    def has_object(self, object_id: Optional[str]) -> bool:
        """Is any part of this object selected? (the walk-level question)"""
        return object_id is not None and (object_id in self.ids
                                          or object_id in self.object_ids)

    def has_item(self, object_id: Optional[str] = None,
                 frame_id: Optional[str] = None) -> bool:
        """Is *this* item selected?

        True when the whole object is listed, or when this exact frame is.  A
        file listing only object ids therefore keeps every frame of them, and one
        listing frame ids keeps exactly those frames.
        """
        return ((object_id is not None and object_id in self.ids)
                or (frame_id is not None and frame_id in self.ids))


def _subset(name: str, ids: Iterable[str]) -> Subset:
    ids = frozenset(ids)
    return Subset(
        name=name,
        ids=ids,
        object_ids=frozenset(i.rsplit("/", 1)[0] for i in ids if "/" in i),
    )


def _parse(text: str) -> list[str]:
    """Ids out of a YAML list, or out of one-id-per-line text."""
    import yaml

    try:
        loaded = yaml.safe_load(text)
    except Exception:
        loaded = None
    if isinstance(loaded, list):
        return [str(i).strip() for i in loaded if str(i).strip()]
    if isinstance(loaded, dict):          # a mapping keyed by id is a subset too
        return [str(k).strip() for k in loaded]
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip().lstrip("- ").strip()
        if line:
            out.append(line)
    return out


# ── reading ───────────────────────────────────────────────────────────────────

def load(path_preprocess, name: str) -> Subset:
    """The subset called *name*. Raises FileNotFoundError when there is no file.

    Loud rather than empty-by-default: a config naming a subset that is not
    there would otherwise silently yield the *whole* dataset, which is the one
    failure mode that does not look like a failure.
    """
    p = find(path_preprocess, name)
    if p is None:
        raise FileNotFoundError(
            f"no subset {name!r} under {dir_path(path_preprocess)}\n"
            f"  existing subsets: {', '.join(list_names(path_preprocess)) or '(none)'}\n"
            f"  write one with: o3b dataset select-subset -d <dataset> -s {name}"
        )
    return _subset(name, _parse(p.read_text()))


def load_if_exists(path_preprocess, name: str) -> Optional[Subset]:
    """Like ``load``, but None when the file is missing (for tools that create it)."""
    return load(path_preprocess, name) if exists(path_preprocess, name) else None


def from_ids(ids: Iterable[str], name: str = "extra.subset_ids") -> Subset:
    """A subset out of ids held in memory — no file involved."""
    return _subset(name, (str(i) for i in ids))


# ── writing ───────────────────────────────────────────────────────────────────

def save(path_preprocess, name: str, ids: Iterable[str], *,
         header: Optional[dict] = None) -> Path:
    """Write a subset, replacing whatever was there. Returns the path.

    Through a temp file + rename so an interrupted write (or a second tool
    saving at the same time) cannot leave a half-written selection behind.
    """
    from datetime import date

    p = path(path_preprocess, name)
    p.parent.mkdir(parents=True, exist_ok=True)

    ids = sorted(set(ids))
    lines = [f'# o3b subset "{name}"']
    for k, v in (header or {}).items():
        lines.append(f"# {k}: {v}")
    lines.append(f"# written: {date.today().isoformat()}   {len(ids)} ids")
    lines += [f"- {i}" for i in ids]

    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, p)
    return p
