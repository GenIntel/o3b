"""Bring od3d's earlier pose labels in as read-only comparison cohorts.

    o3b dataset annotate-obb-import-old -d uco3d --source .../pose_labeling_share

``labels/`` were made by editing the pipeline's pose and on a filtered sample,
so they are never mixed with the blind random samples; ``emil_labels3`` is a
second annotator on the same objects, which makes the pair a human-vs-human
floor.  Their ``axis_pts3d`` rows are od3d's x, y, z axes in the raw frame.

Their boxes live in a per-sequence normalised frame (``bbox_pack``), brought back
to raw through the camera both frames share.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from o3b.dataset.verify_labels import store


def rotation_from_axes(axis_pts3d) -> np.ndarray:
    """od3d's rot_from_axes: raw -> od3d axes, rows being the labelled axes."""
    a = np.asarray(axis_pts3d, dtype=np.float64)
    d = a[:, 0, :] - a[:, 1, :]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    return store.proper_rotation(d)


def box_from_pack(box: dict, R_manual: np.ndarray, task: dict, R_pipeline: np.ndarray,
                  M_raw: np.ndarray) -> Optional[tuple]:
    """(center_raw, extents_raw) of a box drawn in the manual frame of the pack.

    The pack's camera maps ``x_pack`` to ``A x_pack + t``, the sequence's maps
    ``x_raw`` to ``M x_raw + m``; both land in the same camera, so
    ``x_raw = inv(M) (A x_pack + t - m)``.
    """
    view = task["views"][0]
    T = np.asarray(view["cam_T_canonical"], dtype=np.float64)
    A, t = T[:3, :3], T[:3, 3]
    scale = float(np.linalg.norm(A, axis=0).mean())
    lo, hi = np.asarray(box["lo"], float), np.asarray(box["hi"], float)
    R_rel = R_manual @ R_pipeline.T                      # pack canonical -> manual
    center_pack = ((lo + hi) / 2) @ R_rel                # R_rel.T @ c
    camera = A @ center_pack + t
    M, m = M_raw[:3, :3], M_raw[:3, 3]
    return np.linalg.solve(M, camera - m), (hi - lo) * scale


def run_import(root, source, cfg=None) -> dict:
    """Write the cohorts under ``root``. Boxes need ``cfg`` to reach the meta tree."""
    from o3b.dataset.uco3d.dataset import _load_meta_yaml

    source = Path(source)
    pipeline = json.loads((source / "pipeline_reference.json").read_text())
    tasks = {t["sequence_name"]: t for t in json.loads((source / "bbox_pack" / "tasks.json").read_text())}

    def box_of(boxes, seq, R_od3d):
        if cfg is None or seq not in boxes or seq not in tasks or seq not in pipeline:
            return None
        task = tasks[seq]
        frame_index = task["views"][0]["frame_index"]
        meta_dir = Path(cfg.path_preprocess) / "meta" / "frames" / seq
        names = sorted((p.stem for p in meta_dir.glob("*.yaml")), key=lambda s: (len(s), s))
        if not names or frame_index >= len(names):
            return None
        meta = _load_meta_yaml(meta_dir / f"{names[frame_index]}.yaml")
        if not meta or not meta.get("l_cam_tform4x4_obj"):
            return None
        center, ext = box_from_pack(boxes[seq], R_od3d, task, np.asarray(pipeline[seq]["R"], float),
                                    np.asarray(meta["l_cam_tform4x4_obj"], float))
        return center.tolist(), (np.abs(store.G3) @ ext).tolist()

    counts = {}
    for folder, boxes_file, kind in (("labels", "bbox3d_boxes.json", store.OLD),
                                     ("emil_labels3", "emil_bbox3d_boxes.json", store.EMIL)):
        boxes = json.loads((source / boxes_file).read_text())
        n = n_box = 0
        for p in sorted((source / folder).glob("*/*.json")):
            label = json.loads(p.read_text())
            seq = label["sequence_name"]
            R_od3d = rotation_from_axes(label["axis_pts3d"])
            box = box_of(boxes, seq, R_od3d)
            n_box += box is not None
            store._write(store.path_of(root, seq, kind), {
                "sequence_name": seq, "status": "done", "source": f"{folder}/{p.name}",
                "R_obj_raw": (store.G3 @ R_od3d).tolist(),
                "center_raw": box[0] if box else None, "extents_raw": box[1] if box else None,
                "annotator": label.get("user"), "created_at": label.get("created_at"),
                "method": label.get("method"), "history": [],
            })
            n += 1
        counts[kind] = (n, n_box)
        print(f"{folder}: {n} labels, {n_box} with a box -> {Path(root) / kind}")
    return counts
