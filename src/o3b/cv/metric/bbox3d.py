"""3-D IoU of oriented bounding boxes given by their 8 corners.

Exact, for arbitrary relative pose: the intersection of two boxes is the convex
polytope cut out by both boxes' 12 face planes, so it is computed as a halfspace
intersection (the interior point comes from a Chebyshev-centre LP) and measured
by its convex hull. No corner ordering is assumed — each box's planes are read
off the convex hull of its corners — so cam_bbox3d from any loader works as is.

Per-pair cost is a small LP plus two hulls (~ms); evaluation-time only.
"""
from __future__ import annotations

import numpy as np
import torch


def _hull(points: np.ndarray):
    """ConvexHull of ``points``, or None when they span no volume (a flat box)."""
    from scipy.spatial import ConvexHull, QhullError
    try:
        return ConvexHull(points)
    except (QhullError, ValueError):
        return None


def _interior_point(halfspaces: np.ndarray) -> "np.ndarray | None":
    """A point strictly inside ``A x + b <= 0`` (rows [A, b]), or None if the
    feasible set has no interior — the Chebyshev centre, by linear programming."""
    from scipy.optimize import linprog
    A, b = halfspaces[:, :-1], halfspaces[:, -1]
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    # maximise r  s.t.  A x + r |A_i| <= -b
    res = linprog(c=np.r_[np.zeros(A.shape[1]), -1.0],
                  A_ub=np.hstack([A, norm]), b_ub=-b,
                  bounds=[(None, None)] * A.shape[1] + [(0, None)], method="highs")
    if not res.success or res.x[-1] <= 1e-9:
        return None
    return res.x[:-1]


def bbox3d_iou_pair(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """IoU of two (8, 3) corner sets. A box without volume scores 0."""
    from scipy.spatial import HalfspaceIntersection, QhullError

    ha, hb = _hull(corners_a), _hull(corners_b)
    if ha is None or hb is None or ha.volume <= 0 or hb.volume <= 0:
        return 0.0
    halfspaces = np.vstack([ha.equations, hb.equations])
    inner = _interior_point(halfspaces)
    if inner is None:
        return 0.0
    try:
        hi = _hull(HalfspaceIntersection(halfspaces, inner).intersections)
    except QhullError:
        return 0.0
    inter = 0.0 if hi is None else float(hi.volume)
    union = ha.volume + hb.volume - inter
    return inter / union if union > 0 else 0.0


def bbox3d_iou(corners_a: torch.Tensor, corners_b: torch.Tensor) -> torch.Tensor:
    """Batched IoU of oriented boxes, (B, 8, 3) x (B, 8, 3) -> (B,)."""
    a = corners_a.detach().double().cpu().numpy()
    b = corners_b.detach().double().cpu().numpy()
    out = [bbox3d_iou_pair(x, y) if np.isfinite(x).all() and np.isfinite(y).all() else 0.0
           for x, y in zip(a, b)]
    return torch.tensor(out, dtype=torch.float32, device=corners_a.device)


def bbox3d_corners_from_pose(cam_tform4x4_obj: torch.Tensor,
                             obj_size3d: torch.Tensor) -> torch.Tensor:
    """(B, 8, 3) camera-space corners of a box of side lengths ``obj_size3d``
    centred on the pose's origin and aligned with its axes.

    The rotation block may carry a scale (the NCDS pose's is max(size)/2); its
    columns are normalised first, so only the orientation and the translation
    are used and the extent comes from ``obj_size3d`` alone.
    """
    R = cam_tform4x4_obj[..., :3, :3].float()
    R = R / R.norm(dim=-2, keepdim=True).clamp(min=1e-12)
    t = cam_tform4x4_obj[..., :3, 3].float()
    signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=torch.float32, device=R.device)          # (8, 3)
    half = obj_size3d.float()[..., None, :] / 2 * signs                 # (B, 8, 3)
    return half @ R.transpose(-1, -2) + t[..., None, :]
