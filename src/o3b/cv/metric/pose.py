import logging

logger = logging.getLogger(__name__)
import torch
from o3b.cv.geometry.transform import rot3x3, rot3x3_broadcast, inv_tform4x4, inv_rot3x3, tform4x4_broadcast, so3_exp_map_tform4x4
import math
from o3b.cv.metric.pt3d_so3 import so3_log_map, so3_exp_map

_SYM_CACHE = {}


def _obj_rot3x3_obj_syms_padded(obj_rot3d_obj_syms, reflective, device):
    """The symmetry rotations of every symmetry code in ``obj_rot3d_obj_syms`` (...x3),
    as (N, S, 3, 3) padded with identities to the largest group S in the batch.

    Same result as building ``get_obj_rot3x3_obj_sym`` per sample, but each distinct
    code is built once per process (``_SYM_CACHE``) and the batch is one gather: the
    per-sample loop cost ~90 ms per call at batch 32, paid by every pose metric."""
    codes = obj_rot3d_obj_syms.reshape(-1, 3)
    uniq, inv = torch.unique(codes, dim=0, return_inverse=True)
    mats = []
    for code in uniq.tolist():
        key = (tuple(code), bool(reflective), str(device))
        m = _SYM_CACHE.get(key)
        if m is None:
            m = get_obj_rot3x3_obj_sym(torch.tensor(code, device=device, dtype=codes.dtype),
                                       reflective=reflective).float()
            _SYM_CACHE[key] = m
        mats.append(m)
    S = max(1, max(m.shape[0] for m in mats))
    table = torch.eye(3, device=device)[None, None].expand(len(mats), S, 3, 3).clone()
    for u, m in enumerate(mats):
        table[u, :m.shape[0]] = m
    return table[inv.to(device)]


def get_pose_diff3x3_in_rad(diff_rot3x3: torch.Tensor, obj_rot3d_obj_syms=None, reflective=False, lp_norm=2):
    """
    Args:
        pose_diff3x3 (torch.Tensor): ...x3x3
        obj_rot3d_obj_syms (torch.Tensor): ...x3 # negative means continous symmetric around this axes
    Returns:
        rot_diff_in_rad (torch.Tensor): ...
    """
    try:
        # diff_so3_log = rotation_matrix_to_axis_angle(diff_rot3x3.reshape(-1, 3, 3)).reshape(*diff_rot3x3.shape[:-2], 3)
        # diff_so3_log = So3.from_matrix(diff_rot3x3.reshape(-1, 3, 3)).log().reshape(*diff_rot3x3.shape[:-2], 3)
        # import pytorch3d
        # import pytorch3d.transforms
        # diff_so3_log = pytorch3d.transforms.so3_log_map(
        #    diff_rot3x3.reshape(-1, 3, 3),
        # ).reshape(*diff_rot3x3.shape[:-2], 3)
        device = diff_rot3x3.device
        if obj_rot3d_obj_syms is not None:
            # TODO: doublecheck
            obj_rot3x3_obj_syms = _obj_rot3x3_obj_syms_padded(obj_rot3d_obj_syms, reflective, device)
            S = obj_rot3x3_obj_syms.shape[1]
            obj_rot3x3_obj_syms = obj_rot3x3_obj_syms.reshape(*(diff_rot3x3.shape[:-2] + (S,) + diff_rot3x3.shape[-2:]))

            diff_rot3x3 = rot3x3_broadcast(diff_rot3x3[..., None, :, :], obj_rot3x3_obj_syms)

        # NOTE COULD BE PROBLEMATIC NOW THAT SYMMETRIES COULD LEAD TO ROT3x3 DET=-1
        diff_so3_log = so3_log_map(
            diff_rot3x3.reshape(-1, 3, 3),
        ).reshape(*diff_rot3x3.shape[:-2], 3)

        if obj_rot3d_obj_syms is not None:
            diff_so3_log = diff_so3_log * (obj_rot3d_obj_syms[..., None, :] > 0.)
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1, p=lp_norm)
            diff_rot_angle_rad = diff_rot_angle_rad.min(dim=-1).values
        else:
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1, p=lp_norm)

    except ValueError:
        logger.warning(
            f"Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to PI.",
        )
        diff_rot_angle_rad = torch.ones_like(diff_rot3x3[..., 0, 0]) * math.pi

    if not torch.isfinite(diff_rot_angle_rad).all():
        logger.warning(f"Nan or Inf in diff {diff_rot_angle_rad}")
        # toytruck rotation 0 0 0 0 0 0 00 0 -> nan for :  ref_mesh_id: 1, src_mesh_id: (4,)
        diff_rot_angle_rad[~diff_rot_angle_rad.isfinite()] = math.pi

    return diff_rot_angle_rad


def get_pose_diff_in_rad(pred_tform4x4: torch.Tensor, gt_tform4x4: torch.Tensor, obj_rot3d_obj_syms=None, reflective=False, lp_norm=2):
    """
    Args:
        pred_tform4x4 (torch.Tensor): ...x4x4
        gt_tform4x4 (torch.Tensor): ...x4x4
    Returns:
        rot_diff_in_rad (torch.Tensor): ...
    """

    pred_ref_rot3x3_src_scaled = pred_tform4x4[..., :3, :3].clone()
    gt_ref_rot3x3_src_scaled = gt_tform4x4[..., :3, :3].clone()

    pred_ref_rot3x3_src_scaled = pred_ref_rot3x3_src_scaled / torch.linalg.norm(
        pred_ref_rot3x3_src_scaled,
        dim=-1,
        keepdim=True,
    )

    gt_ref_rot3x3_src_scaled = gt_ref_rot3x3_src_scaled / torch.linalg.norm(
        gt_ref_rot3x3_src_scaled,
        dim=-1,
        keepdim=True,
    )

    diff_rot3x3 = rot3x3(
        inv_rot3x3(gt_ref_rot3x3_src_scaled),
        pred_ref_rot3x3_src_scaled[..., :3, :3],
    )
    # gt_obj_rot3x3_pred_obj

    return get_pose_diff3x3_in_rad(diff_rot3x3, obj_rot3d_obj_syms=obj_rot3d_obj_syms, reflective=reflective, lp_norm=lp_norm)

def get_obj_tform4x4_obj_sym(obj_rot3d_obj_syms, reflective=False):
    """
    Args:
        obj_rot3d_obj_syms (torch.Tensor): ...x3 , values: -1, 1, 2 (reflective), 4
    Returns:
        obj_tform4x4_obj_syms (torch.Tensor): ...xSx4x4
    """

    obj_rot3d_obj_syms_reshaped = obj_rot3d_obj_syms.reshape(-1, 3)
    S = 1

    N = obj_rot3d_obj_syms[..., 0].numel()
    device = obj_rot3d_obj_syms.device

    l_obj_tform4x4_obj_syms = [] #  torch.eye(4).to(device=device)
    #obj_tform4x4_obj_syms = obj_tform4x4_obj_syms[None, None].repeat(N, S, 1, 1).clone()

    for n in range(N):
        n_obj_rot3d_obj_syms = obj_rot3d_obj_syms_reshaped[n]

        n_obj_rot3d_obj_syms_step_lengths = torch.pi * 2 / n_obj_rot3d_obj_syms.abs()
        obj_rot1d_obj_syms_x = (torch.arange(n_obj_rot3d_obj_syms[..., 0].abs(), device=device) *
                                n_obj_rot3d_obj_syms_step_lengths[..., 0])
        obj_rot1d_obj_syms_y = torch.arange(n_obj_rot3d_obj_syms[..., 1].abs(), device=device) * \
                               n_obj_rot3d_obj_syms_step_lengths[..., 1]
        obj_rot1d_obj_syms_z = torch.arange(n_obj_rot3d_obj_syms[..., 2].abs(), device=device) * \
                               n_obj_rot3d_obj_syms_step_lengths[..., 2]

        obj_rot3d_obj_syms_x = torch.zeros(obj_rot1d_obj_syms_x.shape + (3,), device=device)
        obj_rot3d_obj_syms_x[..., 0] = obj_rot1d_obj_syms_x
        obj_rot3d_obj_syms_y = torch.zeros(obj_rot1d_obj_syms_y.shape + (3,), device=device)
        obj_rot3d_obj_syms_y[..., 1] = obj_rot1d_obj_syms_y
        obj_rot3d_obj_syms_z = torch.zeros(obj_rot1d_obj_syms_z.shape + (3,), device=device)
        obj_rot3d_obj_syms_z[..., 2] = obj_rot1d_obj_syms_z

        obj_tform4x4_obj_syms_x = so3_exp_map_tform4x4(obj_rot3d_obj_syms_x)
        obj_tform4x4_obj_syms_y = so3_exp_map_tform4x4(obj_rot3d_obj_syms_y)
        obj_tform4x4_obj_syms_z = so3_exp_map_tform4x4(obj_rot3d_obj_syms_z)

        ##### START NOTE
        if reflective:
            # for 2 modes along one axis use reflective symmetry instead of rotational (flipping instead of rotation)
            if n_obj_rot3d_obj_syms[0].abs() == 2:
                obj_tform4x4_obj_syms_x[1, :3, :3] = -1. * obj_tform4x4_obj_syms_x[1, :3, :3]
            if n_obj_rot3d_obj_syms[1].abs() == 2:
                obj_tform4x4_obj_syms_y[1, :3, :3] = -1. * obj_tform4x4_obj_syms_y[1, :3, :3]
            if n_obj_rot3d_obj_syms[2].abs() == 2:
                obj_tform4x4_obj_syms_z[1, :3, :3] = -1. * obj_tform4x4_obj_syms_z[1, :3, :3]
        #### END NOTE

        n_obj_tform4x4_obj_syms = tform4x4_broadcast(
            tform4x4_broadcast(obj_tform4x4_obj_syms_x[:, None,],
                               obj_tform4x4_obj_syms_y[None, :,])[:, :, None,],
            obj_tform4x4_obj_syms_z[None, None, :])

        n_obj_tform4x4_obj_syms = n_obj_tform4x4_obj_syms.reshape(-1, 4, 4)

        # logger.info(f"found {len(obj_tform4x4_obj_syms)} symmetries.")

        n_obj_rot3d_obj_syms_dupl_bool = (tform4x4_broadcast(n_obj_tform4x4_obj_syms[:, None],
                                                           inv_tform4x4(n_obj_tform4x4_obj_syms)[None, :])
                                        - torch.eye(4, device=device)[None, None]).abs().max(dim=-1)[0].max(dim=-1)[0] < 1e-5

        n_obj_rot3d_obj_syms_dupl_bool = torch.triu(n_obj_rot3d_obj_syms_dupl_bool, diagonal=1).any(dim=0)

        n_obj_tform4x4_obj_syms = n_obj_tform4x4_obj_syms[~n_obj_rot3d_obj_syms_dupl_bool]

        if len(n_obj_tform4x4_obj_syms) > S:
            S = len(n_obj_tform4x4_obj_syms)

        l_obj_tform4x4_obj_syms.append(n_obj_tform4x4_obj_syms.clone())

    obj_tform4x4_obj_syms = torch.eye(4).to(device=device)
    obj_tform4x4_obj_syms = obj_tform4x4_obj_syms[None, None].repeat(N, S, 1, 1).clone()
    for n in range(N):
        obj_tform4x4_obj_syms[n, :len(l_obj_tform4x4_obj_syms[n])] = l_obj_tform4x4_obj_syms[n]
    obj_tform4x4_obj_syms = obj_tform4x4_obj_syms.reshape(*obj_rot3d_obj_syms.shape[:-1], S, 4, 4)

    return obj_tform4x4_obj_syms

def get_obj_rot3x3_obj_sym(obj_rot3d_obj_syms, reflective=False):
    """
    Args:
        obj_rot3d_obj_syms (torch.Tensor): 3 , values: -1, 1, 2, 4
    Returns:
        obj_tform3x3_obj_syms (torch.Tensor): Sx3x3
    """
    return get_obj_tform4x4_obj_sym(obj_rot3d_obj_syms, reflective=reflective)[..., :3, :3]

def get_obj_axis6d_with_mask(obj_rot3d_obj_syms):
    """
    Args:
        obj_rot3d_obj_syms (torch.Tensor): ...x3 , values: -1, 1, 2 (reflective), 4
    Returns:
        obj_axis6d_syms (torch.Tensor): ...x3x6  offset 0,1,2 direction 3,4,5
        obj_axis6d_syms_mask (torch.Tensor): ...x3
    """

    device = obj_rot3d_obj_syms.device

    obj_axis6d_syms = torch.zeros((*obj_rot3d_obj_syms.shape[:-1], 3, 6)).to(device=device)
    # obj_axis3d_syms = torch.eye(3).to(device=device)
    obj_axis6d_syms[..., :, 3:] = torch.eye(3).to(device=device).clone()
    # obj_axis6d_syms_mask = torch.zeros((*obj_rot3d_obj_syms.shape[:-1], 3)).to(device=device, dtype=torch.bool)
    obj_axis6d_syms_mask = (obj_rot3d_obj_syms == -1)

    return obj_axis6d_syms, obj_axis6d_syms_mask


def radius_and_offset_torch(D, C, v):
    """
    Compute radius and axial offset for multiple points and multiple axes.

    Args:
        D (torch.Tensor): shape (N, 3), points
        C (torch.Tensor): shape (N, 3), axis origins
        v (torch.Tensor): shape (N, 3), axis directions

    Returns:
        r (torch.Tensor): shape (N,), perpendicular distances (radii)
        s (torch.Tensor): shape (N,), axial offsets
    """
    # Normalize axes
    v_hat = v / v.norm(dim=1, keepdim=True)

    # Vector from axis point to D
    d = D - C  # (N, 3)

    # Axial offsets
    s = (d * v_hat).sum(dim=1)  # (N,)

    # Perpendicular component
    d_perp = d - s.unsqueeze(1) * v_hat  # (N, 3)

    # Radii
    r = d_perp.norm(dim=1)  # (N,)

    return r, s


def min_distance_rotating_B_multi_torch(A, C, v, r, s):
    """
    Compute minimal distances between N fixed points A_i and rotating points B_i(θ)
    where each has its own axis (C_i, v_i, r_i, s_i).

    Args:
        A (torch.Tensor): shape (N, 3), fixed points
        C (torch.Tensor): shape (N, 3), axis points
        v (torch.Tensor): shape (N, 3), axis directions
        r (torch.Tensor): shape (N,), rotation radii
        s (torch.Tensor): shape (N,), offsets along axes

    Returns:
        d_min (torch.Tensor): shape (N,), minimal distances
        theta_min (torch.Tensor): shape (N,), minimizing angles in radians
    """

    # Normalize v_i
    v_hat = v / v.norm(dim=1, keepdim=True)

    # Pick an arbitrary vector not parallel to v_i to build u_i
    t = torch.tensor([1.0, 0.0, 0.0], dtype=v.dtype, device=v.device).expand_as(v).clone()
    mask = (torch.abs((v_hat * t).sum(dim=1)) > 0.99)
    t[mask] = torch.tensor([0.0, 1.0, 0.0], dtype=v.dtype, device=v.device).clone()

    # Build orthonormal bases
    u = t - (t * v_hat).sum(dim=1, keepdim=True) * v_hat
    u = u / u.norm(dim=1, keepdim=True)
    w = torch.cross(v_hat, u, dim=1)

    # Circle centers
    o = C + s.unsqueeze(1) * v_hat

    # Vectors from circle centers to A_i
    p = A - o  # (N, 3)

    # Project p onto u and w
    Pu = (p * u).sum(dim=1)
    Pw = (p * w).sum(dim=1)
    R = torch.sqrt(Pu**2 + Pw**2)

    # Compute min distances
    p_norm2 = (p * p).sum(dim=1)
    d_min = torch.sqrt(p_norm2 + r**2 - 2 * r * R)

    # Minimizing angles
    theta_min = torch.atan2(Pw, Pu)

    return d_min, theta_min

def min_distance_rotating_B_multi_with_point_torch(A, C, v, r, s):
    """
    Compute minimal distances and closest points between N fixed points A_i
    and rotating points B_i(θ) around axes (C_i, v_i, r_i, s_i).

    Args:
        A (torch.Tensor): shape (N, 3), fixed points
        C (torch.Tensor): shape (N, 3), axis points
        v (torch.Tensor): shape (N, 3), axis directions
        r (torch.Tensor): shape (N,), rotation radii
        s (torch.Tensor): shape (N,), offsets along axes

    Returns:
        d_min (torch.Tensor): shape (N,), minimal distances
        theta_min (torch.Tensor): shape (N,), minimizing angles (radians)
        B_min (torch.Tensor): shape (N, 3), coordinates of closest points
    """

    # Normalize direction vectors
    v_hat = v / v.norm(dim=1, keepdim=True)

    # Choose an arbitrary vector t not parallel to v_hat
    t = torch.tensor([1.0, 0.0, 0.0], dtype=v.dtype, device=v.device).expand_as(v).clone()
    mask = (torch.abs((v_hat * t).sum(dim=1)) > 0.99)
    t[mask] = torch.tensor([0.0, 1.0, 0.0], dtype=v.dtype, device=v.device).clone()

    # Build orthonormal bases (u, w)
    u = t - (t * v_hat).sum(dim=1, keepdim=True) * v_hat
    u = u / u.norm(dim=1, keepdim=True)
    w = torch.cross(v_hat, u, dim=1)

    # Circle centers
    o = C + s.unsqueeze(1) * v_hat  # (N, 3)

    # Vectors from centers to A_i
    p = A - o  # (N, 3)

    # Projections
    Pu = (p * u).sum(dim=1)
    Pw = (p * w).sum(dim=1)
    R = torch.sqrt(Pu**2 + Pw**2)
    p_norm2 = (p * p).sum(dim=1)

    # Compute distances and angles
    d_min = torch.sqrt(p_norm2 + r**2 - 2 * r * R)
    theta_min = torch.atan2(Pw, Pu)

    # Compute closest points B_min
    cos_t = torch.cos(theta_min).unsqueeze(1)
    sin_t = torch.sin(theta_min).unsqueeze(1)
    B_min = o + r.unsqueeze(1) * (u * cos_t + w * sin_t)

    return d_min, theta_min, B_min


def get_closest_B_to_A_rotated_around_axis6d(A, B, axis6d):
    """
    Compute minimal distances and closest points between N fixed points A_i
    and rotating points B_i(θ) around axes (C_i, v_i, r_i, s_i).

    Args:
        A (torch.Tensor): shape (N, 3), fixed points
        C (torch.Tensor): shape (N, 3), axis points
        axis6d (torch.Tensor): shape (N, 6 axis directions
        r (torch.Tensor): shape (N,), rotation radii
        s (torch.Tensor): shape (N,), offsets along axes

    Returns:
        d_min (torch.Tensor): shape (N,), minimal distances
        theta_min (torch.Tensor): shape (N,), minimizing angles (radians)
        B_min (torch.Tensor): shape (N, 3), coordinates of closest points
    """

    C = axis6d[..., :3].clone()
    v = axis6d[..., 3:].clone()
    r, s = radius_and_offset_torch(B, C, v)

    d_min, theta_min, B_min =  min_distance_rotating_B_multi_with_point_torch(A, C, v, r, s)

    return B_min



def get_pose_diff3x3_with_syms(diff_rot3x3: torch.Tensor, obj_rot3d_obj_syms=None, reflective=False, lp_norm=2):
    """
    Args:
        pose_diff3x3 (torch.Tensor): ...x3x3
        obj_rot3d_obj_syms (torch.Tensor): ...x3 # negative means continous symmetric around this axes
    Returns:
        pose_diff3x3 (torch.Tensor): ...
        pose_diff_error (torch.Tensor): ...
    """
    
    try:
        # diff_so3_log = rotation_matrix_to_axis_angle(diff_rot3x3.reshape(-1, 3, 3)).reshape(*diff_rot3x3.shape[:-2], 3)
        # diff_so3_log = So3.from_matrix(diff_rot3x3.reshape(-1, 3, 3)).log().reshape(*diff_rot3x3.shape[:-2], 3)
        # import pytorch3d
        # import pytorch3d.transforms
        # diff_so3_log = pytorch3d.transforms.so3_log_map(
        #    diff_rot3x3.reshape(-1, 3, 3),
        # ).reshape(*diff_rot3x3.shape[:-2], 3)
        device = diff_rot3x3.device
        if obj_rot3d_obj_syms is not None:
            # TODO: doublecheck
            obj_rot3x3_obj_syms = _obj_rot3x3_obj_syms_padded(obj_rot3d_obj_syms, reflective, device)
            S = obj_rot3x3_obj_syms.shape[1]
            obj_rot3x3_obj_syms = obj_rot3x3_obj_syms.reshape(*(diff_rot3x3.shape[:-2] + (S,) + diff_rot3x3.shape[-2:]))

            diff_rot3x3 = rot3x3_broadcast(diff_rot3x3[..., None, :, :], obj_rot3x3_obj_syms)
        else:
            diff_rot3x3 = diff_rot3x3[..., None, :, :]
            
        
        # NOTE COULD BE PROBLEMATIC NOW THAT SYMMETRIES COULD LEAD TO ROT3x3 DET=-1
        diff_so3_log = so3_log_map(
            diff_rot3x3.reshape(-1, 3, 3),
        ).reshape(*diff_rot3x3.shape[:-2], 3)

        if obj_rot3d_obj_syms is not None:
            diff_so3_log = diff_so3_log * (obj_rot3d_obj_syms[..., None, :] > 0.)
        
        if lp_norm == "l1_smooth":
            diff_rot_errors_l1 = diff_so3_log.abs()
            diff_rot_errors_l2 = diff_so3_log ** 2
            diff_rot_errors = diff_rot_errors_l1 * (diff_rot_errors_l1 >= 1.0).float() + diff_rot_errors_l2 * (diff_rot_errors_l1 < 1.0).float()
        else:
            diff_rot_errors = torch.norm(diff_so3_log, dim=-1, p=lp_norm)
        
        diff_rot_errors_min = diff_rot_errors.min(dim=-1)
        diff_rot_errors = diff_rot_errors_min.values
        diff_rot3x3_id = diff_rot_errors_min.indices

        from o3b.cv.select import batched_index_select

        diff_so3_log = batched_index_select(input=diff_so3_log, index=diff_rot3x3_id[..., None], dim=diff_so3_log.dim() -2)
        diff_so3_log = diff_so3_log.squeeze(-2)

        #diff_rot3x3 = batched_index_select(input=diff_rot3x3, index=diff_rot3x3_id[..., None], dim=-3)
        #diff_rot3x3 = diff_rot3x3.squeeze(-3)

        diff_rot3x3 = so3_exp_map(diff_so3_log.reshape(-1, 3)).reshape(*diff_so3_log.shape[:-1], 3, 3)

        
    except ValueError:
        logger.warning(
            f"Cannot calculate deviation in rotation angle due to rot3x3 trace being too small.",
        )
        diff_rot3x3_id = torch.zeros(diff_rot3x3.shape[:-3], dtype=torch.long, device=diff_rot3x3.device)
        diff_rot_errors = torch.ones(diff_rot3x3.shape[:-3], device=diff_rot3x3.device) * math.pi
        diff_rot3x3 = diff_rot3x3.squeeze(-3)
    
    return diff_rot3x3, diff_rot_errors
    
def get_pose_diff(pred_tform4x4: torch.Tensor, gt_tform4x4: torch.Tensor, obj_rot3d_obj_syms=None, reflective=False, lp_norm=2):
    """
    Args:
        pred_tform4x4 (torch.Tensor): ...x4x4
        gt_tform4x4 (torch.Tensor): ...x4x4
    Returns:
        rot_diff3x3 (torch.Tensor): ...x3x3
        rot_diff (torch.Tensor): ...
    """

    pred_ref_rot3x3_src_scaled = pred_tform4x4[..., :3, :3].clone()
    gt_ref_rot3x3_src_scaled = gt_tform4x4[..., :3, :3].clone()

    pred_ref_rot3x3_src_scaled = pred_ref_rot3x3_src_scaled / torch.linalg.norm(
        pred_ref_rot3x3_src_scaled,
        dim=-1,
        keepdim=True,
    )

    gt_ref_rot3x3_src_scaled = gt_ref_rot3x3_src_scaled / torch.linalg.norm(
        gt_ref_rot3x3_src_scaled,
        dim=-1,
        keepdim=True,
    )

    gt_src_scaled_rot_diff3x3_pred_src_scaled = rot3x3(
        inv_rot3x3(gt_ref_rot3x3_src_scaled),
        pred_ref_rot3x3_src_scaled,
    )
    # gt_obj_rot3x3_pred_obj

    syms_gt_src_scaled_rot_diff3x3_pred_src_scaled, rot_diff = get_pose_diff3x3_with_syms(gt_src_scaled_rot_diff3x3_pred_src_scaled, obj_rot3d_obj_syms=obj_rot3d_obj_syms, reflective=reflective, lp_norm=lp_norm)

    gt_src_scaled_rot_diff3x3_pred_src_scaled = gt_src_scaled_rot_diff3x3_pred_src_scaled.detach()
    syms_gt_src_scaled_rot_diff3x3_pred_src_scaled = syms_gt_src_scaled_rot_diff3x3_pred_src_scaled.detach()
    gt_tform4x4 = gt_tform4x4.clone()
    gt_tform4x4[..., :3, :3] = rot3x3(gt_tform4x4[..., :3, :3], rot3x3(gt_src_scaled_rot_diff3x3_pred_src_scaled, inv_rot3x3(syms_gt_src_scaled_rot_diff3x3_pred_src_scaled)))
    gt_tform4x4 = gt_tform4x4.clone()

    return gt_tform4x4, rot_diff
