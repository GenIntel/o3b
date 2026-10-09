"""Re-express one item in a rotated object frame — the bookkeeping shared by the
loaders that relabel object axes on read (od3d_frames: per-category tables onto
UCO3D's axes; housecorr3d: od3d's Omni6DPose table).

For a rotation T of the object frame the object lands on exactly the same
pixels; only the labels move: pose @ inv(T); NCDS map T @ M @ inv(T), where
M = obj_ncds0c_tform4x4_obj maps NCDS -> object; mesh verts and keypoints by T;
sizes and per-axis symmetry codes permuted by |R|; boxes rebuilt in canonical
corner order; and cam_tform4x4_obj_ncds recomposed as pose @ M — NOT
pose @ inv(M), which an earlier version used and which corrupted the NCDS pose
for every re-oriented item.
"""
from __future__ import annotations


def _corners_canonical(v_min, v_max):
    """The 8 box corners in the order o3b's box drawing requires.

    _corners8_to_size_tform documents it as 0-3 bottom, 4-7 top with 0->1 = +x,
    0->3 = +y, 0->4 = +z, and draw_bbox3d_corners walks (0,1),(1,2),(2,3),(3,0)
    then (4,5),(5,6),(6,7),(7,4) then the four verticals. A plain x/y/z triple
    loop yields a different permutation, and the wireframe is then drawn across
    the diagonals — right size, right place, visibly not a box.
    """
    import torch

    x0, y0, z0 = v_min.tolist()
    x1, y1, z1 = v_max.tolist()
    return torch.tensor([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],   # bottom
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],   # top
    ], dtype=torch.float32)


def rotate_object_frame(item, T, *, rotate_syms: bool = False):
    """Rotate ``item``'s object frame by the (4, 4) rotation ``T``, in place.

    ``rotate_syms`` also permutes ``obj_syms`` (and turns the keypoint and axis
    forms derived from it) — for symmetry annotated per instance in the item's
    own frame (Omni6DPose). Callers that assign symmetry from a table *after*
    the rotation (od3d_frames, from UCO3D's tree) leave it False.
    """
    from dataclasses import replace

    import torch

    from o3b.cv.geometry.transform import inv_tform4x4

    R = T[:3, :3]
    R_abs = R.abs()
    T_inv = inv_tform4x4(T)
    had_ncds_map = item.obj_ncds0c_tform4x4_obj is not None
    if item.cam_tform4x4_obj is not None:
        item.cam_tform4x4_obj = item.cam_tform4x4_obj @ T_inv
    if had_ncds_map:
        item.obj_ncds0c_tform4x4_obj = T @ item.obj_ncds0c_tform4x4_obj @ T_inv
    if item.obj_size3d is not None:
        item.obj_size3d = R_abs @ item.obj_size3d.float()
    if item.mesh is not None:
        # replace, never mutate: a loader's cached Mesh can SHARE its verts
        # tensor with the item, so rotating in place would turn the cache entry
        # too, and every later frame of the object would be rotated again.
        item.mesh = replace(item.mesh, verts=item.mesh.verts.float() @ R.T)
    if item.obj_kpts3d is not None:
        item.obj_kpts3d = item.obj_kpts3d.float() @ R.T
    if item.obj_bbox3d is not None:
        # rotate, then REBUILD in canonical corner order: rotating alone keeps
        # the box in place but permutes which corner is index 0..7, which
        # draw_bbox3d_corners / _corners8_to_size_tform rely on
        rot = item.obj_bbox3d.float() @ R.T
        item.obj_bbox3d = _corners_canonical(rot.min(dim=0).values, rot.max(dim=0).values)
    if item.cam_bbox3d is not None and item.cam_tform4x4_obj is not None \
            and item.obj_bbox3d is not None:
        # camera-space, so unmoved by the rotation, but its corner order
        # followed obj_bbox3d's — recompose through the rotated pose
        R2, t2 = item.cam_tform4x4_obj[:3, :3], item.cam_tform4x4_obj[:3, 3]
        item.cam_bbox3d = item.obj_bbox3d.float() @ R2.t() + t2
    if item.cam_tform4x4_obj is not None and had_ncds_map:
        item.cam_tform4x4_obj_ncds = item.cam_tform4x4_obj @ item.obj_ncds0c_tform4x4_obj
    elif getattr(item, "cam_tform4x4_obj_ncds", None) is not None:
        # no NCDS map on the item: the NCDS frame turns with the object, so
        # pose @ M becomes pose @ inv(T) @ T @ M @ inv(T) = (pose @ M) @ inv(T)
        item.cam_tform4x4_obj_ncds = item.cam_tform4x4_obj_ncds @ T_inv
    if rotate_syms:
        if getattr(item, "obj_syms", None) is not None:
            item.obj_syms = (R_abs @ item.obj_syms.float()).round().to(item.obj_syms.dtype)
        if getattr(item, "obj_kpts3d_syms", None) is not None:
            item.obj_kpts3d_syms = item.obj_kpts3d_syms.float() @ R.T
        if getattr(item, "obj_axis6d_sym", None) is not None:
            a = item.obj_axis6d_sym.float()
            item.obj_axis6d_sym = torch.cat([R @ a[:3], R @ a[3:]])
    return item
