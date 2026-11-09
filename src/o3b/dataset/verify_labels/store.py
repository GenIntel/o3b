"""One JSON per annotated object, in the raw SfM frame of its sequence.

    <root>/annotations/<category>/<sequence>.json

``R_obj_raw`` maps raw points into the box's object frame (o3b's axes: X right,
Y top, Z back), so a point is ``x_obj = R_obj_raw @ (x_raw - center_raw)`` and
the box is ``|x_obj| <= extents_raw / 2``.  Raw coordinates need no per-config
bridge: they are what ``mesh.ply`` and the meta YAMLs are written in.

A save never destroys the previous version, it moves to ``history``.  That is
what lets ``compare`` score the last version made before the official box was
shown, whatever was edited after.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Rotation block of uco3d.yaml's obj_gl_tform4x4_obj_raw: od3d's labelling axes
# (x left, y back, z top) -> o3b's. Pinned against the config by a test.
G3 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# Same corner order as the UCO3D loader: two cyclic quads joined by 0-4/1-5/2-6/3-7.
_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
# Which object axis each edge of BOX_EDGES runs along.
BOX_EDGE_AXIS = [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2]

ANNOTATIONS = "annotations"
REPEATS = "annotations_repeat"
OLD = "annotations_old"
EMIL = "annotations_emil"
PRELABELS = "prelabels_claude"   # model pre-labels, blind to tform_obj; start boxes only


def corners(R: np.ndarray, center: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """(8, 3) raw-frame corners of a box."""
    return np.asarray(center) + (_SIGNS * np.asarray(extents) / 2.0) @ np.asarray(R)


def proper_rotation(M: np.ndarray) -> np.ndarray:
    """Nearest rotation (det = +1) to a 3x3 matrix."""
    U, _, Vh = np.linalg.svd(np.asarray(M, dtype=np.float64))
    if np.linalg.det(U @ Vh) < 0:
        U[:, -1] *= -1
    return U @ Vh


def path_of(root, sequence_name: str, kind: str = ANNOTATIONS) -> Path:
    category, sequence = sequence_name.split("/", 1)
    return Path(root) / kind / category / f"{sequence}.json"


def load(root, sequence_name: str, kind: str = ANNOTATIONS) -> Optional[dict]:
    p = path_of(root, sequence_name, kind)
    return json.loads(p.read_text()) if p.exists() else None


def _write(p: Path, record: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, p)


def save(root, record: dict, kind: str = ANNOTATIONS) -> dict:
    """Write one annotation, keeping whatever it replaces in ``history``."""
    p = path_of(root, record["sequence_name"], kind)
    record = dict(record)
    previous = json.loads(p.read_text()) if p.exists() else None
    if previous is not None:
        history = previous.pop("history", [])
        record["history"] = history + [previous]
        # Seen once is seen for good: the flag survives into every later version.
        if previous.get("revealed"):
            record["revealed"] = True
            record["revealed_before_save"] = True
    record.setdefault("history", [])
    record.setdefault("revealed", False)
    record.setdefault("revealed_before_save", False)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    _write(p, record)
    return record


def mark_revealed(root, sequence_name: str, kind: str = ANNOTATIONS) -> None:
    """Note that the official box of a saved annotation has been shown."""
    p = path_of(root, sequence_name, kind)
    record = json.loads(p.read_text())
    if not record.get("revealed"):
        record["revealed"] = True
        _write(p, record)


def blind_version(record: dict) -> Optional[dict]:
    """The newest version saved without having seen the official box, or None."""
    for version in [record] + list(reversed(record.get("history", []))):
        if not version.get("revealed_before_save"):
            return version
    return None


def iter_all(root, kind: str = ANNOTATIONS) -> Iterator[dict]:
    for p in sorted((Path(root) / kind).glob("*/*.json")):
        yield json.loads(p.read_text())
