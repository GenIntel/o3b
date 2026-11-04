"""What annotating an object needs, in one local directory, and no official pose.

    <pack>/<category>/<sequence>/meta.npz   verts, faces, K (F,3,3), M (F,3,4)
    <pack>/<category>/<sequence>/pcl.ply    the coloured point cloud the mesh was cut from
    <pack>/<category>/<sequence>/000.jpg …

Read straight from the od3d meta tree rather than through a dataset class, so no
``tform_obj`` is ever opened: blindness holds by construction, not by discipline.
``M`` is cam←raw as od3d wrote it (OpenCV camera: +x right, +y down, +z forward).
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

FRAMES = 16
LONG_SIDE = 720


def default_dir() -> Path:
    return Path.home() / ".o3b" / "cache" / "verify_labels"


def pack_object(cfg, sequence_name: str, out_root, frames: int = FRAMES,
                long_side: int = LONG_SIDE) -> Optional[Path]:
    """Build one object's pack. Returns its directory, or None if unusable."""
    import cv2
    import trimesh

    from o3b.dataset.uco3d.dataset import _load_meta_yaml, _subsample

    category, sequence = sequence_name.split("/", 1)
    out = Path(out_root) / category / sequence
    pre, raw = Path(cfg.path_preprocess), Path(cfg.path_raw)
    pcl_path = pre / "pcl" / "meta_mask" / "meta" / category / sequence / "pcl.ply"
    if (out / "meta.npz").exists():
        if not (out / "pcl.ply").exists() and pcl_path.exists():     # packs from before the pcl
            shutil.copyfile(pcl_path, out / "pcl.ply")
        return out

    mesh_path = pre / "mesh" / cfg.mesh_type / "meta_mask" / "meta" / category / sequence / "mesh.ply"
    if not mesh_path.exists():
        print(f"  {sequence_name}: no mesh")
        return None
    mesh = trimesh.load(mesh_path, process=False)

    meta_dir = pre / "meta" / "frames" / category / sequence
    names = sorted((p.stem for p in meta_dir.glob("*.yaml")),
                   key=lambda s: (len(s), s))
    Ks, Ms, kept = [], [], []
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    capture, video = None, None
    for name in _subsample(names, frames):
        meta = _load_meta_yaml(meta_dir / f"{name}.yaml")
        if not meta or not meta.get("l_cam_tform4x4_obj") or not meta.get("l_cam_intr4x4"):
            continue
        if video != meta.get("rfpath_rgb"):
            video = meta.get("rfpath_rgb")
            capture = cv2.VideoCapture(str(raw / video))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(meta.get("timestamp", 0.0)) * 1000)
        ok, bgr = capture.read()
        if not ok:
            continue
        K = np.array(meta["l_cam_intr4x4"], dtype=np.float64)[:3, :3]
        scale = long_side / max(bgr.shape[:2])
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            K[:2] *= scale
        cv2.imwrite(str(tmp / f"{len(kept):03d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        Ks.append(K)
        Ms.append(np.array(meta["l_cam_tform4x4_obj"], dtype=np.float64)[:3, :4])
        kept.append(name)
    if not kept:
        print(f"  {sequence_name}: no readable frame")
        return None
    if pcl_path.exists():
        shutil.copyfile(pcl_path, tmp / "pcl.ply")
    np.savez_compressed(tmp / "meta.npz", verts=np.asarray(mesh.vertices, dtype=np.float32),
                        faces=np.asarray(mesh.faces, dtype=np.int64),
                        K=np.stack(Ks), M=np.stack(Ms), frames=np.array(kept))
    tmp.rename(out)
    return out


def run_pack(cfg, ids, out_root=None, workers: int = 4) -> list:
    """Pack every id; existing ones are kept. Returns the ids that could not be."""
    out_root = Path(out_root) if out_root else default_dir()

    def one(sequence_name):
        try:
            return sequence_name, pack_object(cfg, sequence_name, out_root)
        except Exception as e:      # one bad video must not lose a long copy
            print(f"  {sequence_name}: {e}")
            return sequence_name, None

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (name, result) in enumerate(ex.map(one, ids)):
            if result is None:
                failed.append(name)
            print(f"[{i + 1}/{len(ids)}] {name}" + (" FAILED" if result is None else ""), flush=True)
    return failed


def load_object(out_root, sequence_name: str) -> dict:
    """The pack of one object, with lazily decoded frames listed by path.

    ``pcl_xyz`` / ``pcl_rgb`` are the coloured point cloud, in the mesh's (raw)
    frame, or absent for a pack that has none.
    """
    d = Path(out_root) / sequence_name
    with np.load(d / "meta.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["images"] = sorted(d.glob("*.jpg"))
    if (d / "pcl.ply").exists():
        import trimesh

        pcl = trimesh.load(d / "pcl.ply", process=False)
        data["pcl_xyz"] = np.asarray(pcl.vertices, dtype=np.float32)
        data["pcl_rgb"] = np.asarray(pcl.colors, dtype=np.uint8)[:, :3]
    return data
