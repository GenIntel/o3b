import logging

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


def skew(vector: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [torch.zeros_like(vector[..., 0]), -vector[..., 2], vector[..., 1]],
                dim=-1,
            ),
            torch.stack(
                [vector[..., 2], torch.zeros_like(vector[..., 0]), -vector[..., 0]],
                dim=-1,
            ),
            torch.stack(
                [-vector[..., 1], vector[..., 0], torch.zeros_like(vector[..., 0])],
                dim=-1,
            ),
        ],
        dim=-1,
    )


def logarithmic_map(rotation: torch.Tensor) -> torch.Tensor:
    # rotation torch.Tensor of shape (B, 3, 3)
    angle = torch.arccos(
        (torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1) / 2.0,
    ).unsqueeze(-1)
    axis = (
        1
        / (2 * torch.sin(angle))
        * torch.stack(
            [
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ],
            dim=-1,
        )
    )
    return angle * axis


def exponential_map(so3: torch.Tensor) -> torch.Tensor:
    # so3 torch.Tensor of shape (B, 3)
    theta = torch.norm(so3, dim=-1, keepdim=True)
    axis = so3 / theta
    axis_skew = skew(axis)
    return (
        torch.eye(3, device=so3.device).unsqueeze(0).expand(so3.shape[0], -1, -1)
        + torch.sin(theta).unsqueeze(-1) * axis_skew
        + (1 - torch.cos(theta).unsqueeze(-1)) * axis_skew @ axis_skew
    )


def calculate_rotational_offset(co3d_transforms, colmap_transforms):
    # co3d_transforms torch.Tensor of shape (B, 4, 4)
    # colmap_transforms torch.Tensor of shape (B, 4, 4)
    return exponential_map(
        torch.mean(
            logarithmic_map(
                co3d_transforms[..., :3, :3]
                @ colmap_transforms[:, :3, :3].transpose(-2, -1),
            ),
            dim=0,
            keepdim=True,
        ),
    )


def calulate_scale_offset(co3d_transforms, colmap_transforms):
    # co3d_transforms torch.Tensor of shape (B, 3, 3)
    # colmap_transforms torch.Tensor of shape (B, 3, 3)
    co3d_translations = co3d_transforms[:, :3, 3]
    colmap_translations = colmap_transforms[:, :3, 3]
    co3d_mean = torch.mean(co3d_translations, dim=0, keepdim=True)
    colmap_mean = torch.mean(colmap_translations, dim=0, keepdim=True)

    # version 1
    # s = (torch.mean(torch.norm(co3d_translations - co3d_mean, dim=-1))) / (
    #     (torch.mean(torch.norm(colmap_translations - colmap_mean, dim=-1)))
    # )

    # # version 2
    s = torch.mean(
        torch.norm(co3d_translations - co3d_mean)
        / torch.norm(colmap_translations - colmap_mean),
    )

    return s


def calculate_translational_offset(co3d_transforms, colmap_transforms):
    # co3d_transforms torch.Tensor of shape (B, 4, 4)
    # colmap_transforms torch.Tensor of shape (B, 4, 4)
    return torch.mean(
        co3d_transforms[:, :3, 3] - colmap_transforms[:, :3, 3],
        dim=0,
        keepdim=True,
    )


def calculate_offset(co3d_transforms, colmap_transforms):
    rotational_offset = calculate_rotational_offset(co3d_transforms, colmap_transforms)
    new_transforms = colmap_transforms.clone()
    new_transforms[:, :3, :3] = (
        rotational_offset.transpose(-2, -1) @ new_transforms[:, :3, :3]
    )
    rotated_co3d = co3d_transforms.clone()
    rotated_co3d[:, :3, :3] = rotational_offset @ rotated_co3d[:, :3, :3]
    rotated_co3d[:, :3, 3] = (rotational_offset @ rotated_co3d[:, :3, 3].unsqueeze(-1))[
        ...,
        0,
    ]
    scale_offset = calulate_scale_offset(co3d_transforms, new_transforms)
    new_transforms[:, :3, 3] = scale_offset * new_transforms[:, :3, 3]
    translational_offset = -calculate_translational_offset(rotated_co3d, new_transforms)
    new_transforms[:, :3, 3] = translational_offset + new_transforms[:, :3, 3]

    return new_transforms, rotational_offset, 1 / scale_offset, translational_offset


def get_se3s_alignment(a_tform4x4_src, a_tform4x4_ref):
    """
    Args:
        a_tform4x4_src (torch.Tensor): ...xTx4x4 # cam1_tform_obj
        a_tform4x4_ref (torch.Tensor): ...xTx4x4 # cam2_tform_obj
    Returns:
        src_tform4x4_ref_mean (torch.Tensor): ...x4x4
    """
    from o3b.cv.geometry.transform import (
        rot3d_broadcast,
        inv_tform4x4,
        rot3x3,
        tform4x4,
        tform4x4_broadcast,
        so3_log_map,
        so3_exp_map,
        transf4x4_from_rot3x3,
    )

    # inv_tform4x4
    rot_cov = rot3x3(
        a_tform4x4_src.transpose(-2, -1)[..., :3, :3],
        a_tform4x4_ref[..., :3, :3],
    ).mean(dim=-3)
    U, _, V = torch.svd(rot_cov)
    align_rot = (V @ U.t()).transpose(-2, -1)

    pts_src_rot_src = rot3d_broadcast(
        rot3x3=a_tform4x4_src.transpose(-2, -1)[..., :3, :3],
        pts3d=a_tform4x4_src[..., :3, 3],
    )
    pts_ref_rot_src = rot3d_broadcast(
        rot3x3=a_tform4x4_src.transpose(-2, -1)[..., :3, :3],
        pts3d=a_tform4x4_ref[..., :3, 3],
    )

    pts_src_rot_src_mean = pts_src_rot_src.mean(dim=-2)
    pts_ref_rot_src_mean = pts_ref_rot_src.mean(dim=-2)

    pts_src_rot_src_rel = pts_src_rot_src - pts_src_rot_src_mean[..., None, :]
    pts_ref_rot_src_rel = pts_ref_rot_src - pts_ref_rot_src_mean[..., None, :]
    align_scale = (pts_src_rot_src_rel * pts_ref_rot_src_rel).flatten(-2).mean(
        dim=-1,
    ) / (pts_src_rot_src_rel**2).flatten(-2).mean(dim=-1).clamp(1e-10)
    # align_scale = ((pts_src_rot_src_rel * pts_ref_rot_src_rel) / (pts_src_rot_src_rel ** 2).clamp(1e-10)).flatten(-2).mean(dim=-1)

    align_transl = pts_ref_rot_src_mean - pts_src_rot_src_mean * align_scale

    align_tform = transf4x4_from_rot3x3(align_rot)
    align_tform[..., :3, :3] /= align_scale
    align_tform[..., :3, 3] = align_transl / align_scale

    src_tform4x4_ref_mean = align_tform

    # from o3b.cv.visual.show import show_scene
    a_tform4x4_src_transf = tform4x4_broadcast(
        a_tform4x4_src,
        src_tform4x4_ref_mean[..., None, :, :],
    )
    # a_tform4x4_src_transf = a_tform4x4_src
    # cams_tform = torch.cat([a_tform4x4_src_transf, a_tform4x4_ref])
    #
    # cams_intr = torch.eye(4)[None, ].expand(*cams_tform.shape).to(cams_tform.device, cams_tform.dtype).clone()
    # cams_intr[:, :2] *= 1000
    # cams_intr[:, 0:2, 2] = 500
    # # cams_tform = cams_tform / cams_tform[:, :3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    # show_scene(pts3d=[cams_tform[:, :3, 3]], cams_tform4x4_world=cams_tform[:], cams_intr4x4=cams_intr[:]) #renderer=OD3D_RENDERER.OPEN3D,)
    #
    # show_scene(cams_tform4x4_world=cams_tform[:], cams_intr4x4=cams_intr[:]) #renderer=OD3D_RENDERER.OPEN3D,)
    # show_scene(pts3d=[inv_tform4x4(cams_tform)[:, :3, 3]]) #renderer=OD3D_RENDERER.OPEN3D,)
    return src_tform4x4_ref_mean

    #
    # a_tform4x4_src,
    # A = torch.bmm(cameras_src.R, cameras_src.T[:, :, None])[:, :, 0]
    # B = torch.bmm(cameras_src.R, cameras_tgt.T[:, :, None])[:, :, 0]
    # Amu = A.mean(0, keepdim=True)
    # Bmu = B.mean(0, keepdim=True)
    # if estimate_scale and A.shape[0] > 1:
    #     # get the scaling component by matching covariances
    #     # of centered A and centered B
    #     Ac = A - Amu
    #     Bc = B - Bmu
    #     # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    #     align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
    # else:
    #     # set the scale to identity
    #     align_t_s = 1.0
    # # get the translation as the difference between the means of A and B
    # align_t_T = Bmu - align_t_s * Amu

    #
    #
    # a_tform4x4_ref = inv_tform4x4(a_tform4x4_ref)
    # a_tform4x4_src = inv_tform4x4(a_tform4x4_src)
    #
    # # initial
    # src_transf_tform4x4_ref = tform4x4(inv_tform4x4(a_tform4x4_src), a_tform4x4_ref)
    # logger.info('inital')
    # logger.info(f'error transl: {src_transf_tform4x4_ref[..., :3, 3].norm(dim=-1).mean()}')
    # logger.info(f'error rot: {so3_log_map(src_transf_tform4x4_ref[..., :3, :3]).norm(dim=-1).mean()}')
    #
    #
    # src_rot3x3_ref = rot3x3(inv_tform4x4(a_tform4x4_src)[..., :3, :3], a_tform4x4_ref[..., :3, :3]) # ...x3x3
    # src_rot3x3_ref_mean = so3_exp_map(so3_log_map(src_rot3x3_ref).mean(dim=-2)) # ...x3x3
    # src_tform4x4_ref_mean = transf4x4_from_rot3x3(src_rot3x3_ref_mean)
    #
    # # after rotation
    # a_tform4x4_src_transf = tform4x4_broadcast(a_tform4x4_src, src_tform4x4_ref_mean[..., None, :, :])
    # src_transf_tform4x4_ref = tform4x4(inv_tform4x4(a_tform4x4_src_transf), a_tform4x4_ref)
    # logger.info('after rotation')
    # logger.info(f'error transl: {src_transf_tform4x4_ref[..., :3, 3].norm(dim=-1).mean()}')
    # logger.info(f'error rot: {so3_log_map(src_transf_tform4x4_ref[..., :3, :3]).norm(dim=-1).mean()}')
    #
    # # a_tform4x4_src_rot = tform4x4_broadcast(a_tform4x4_src, src_tform4x4_ref_mean[..., None, :, :])
    # #src_rot_transl = a_tform4x4_src_rot[..., :3, 3]
    # src_transl = a_tform4x4_src[..., :3, 3]
    # ref_transl = a_tform4x4_ref[..., :3, 3]
    #
    # #src_scale = (src_rot_transl - src_rot_transl.mean(dim=-2, keepdim=True)).norm(dim=-1)
    # src_scale = (src_transl - src_transl.mean(dim=-2, keepdim=True)).norm(dim=-1)
    # ref_scale = (ref_transl - ref_transl.mean(dim=-2, keepdim=True)).norm(dim=-1)
    # src_scale_ref = (src_scale / ref_scale).mean(dim=-1)
    #
    # src_tform4x4_ref_mean[..., :3, :3] *= 1. / (src_scale_ref[..., None, None])
    #
    # # after scaling
    # a_tform4x4_src_transf = tform4x4_broadcast(a_tform4x4_src, src_tform4x4_ref_mean[..., None, :, :])
    # src_transf_tform4x4_ref = tform4x4(inv_tform4x4(a_tform4x4_src_transf), a_tform4x4_ref)
    # logger.info('after scaling')
    # logger.info(f'error transl: {src_transf_tform4x4_ref[..., :3, 3].norm(dim=-1).mean()}')
    # logger.info(f'error rot: {so3_log_map(src_transf_tform4x4_ref[..., :3, :3]).norm(dim=-1).mean()}')
    #
    #
    # a_tform4x4_src_rot_scaled = tform4x4_broadcast(a_tform4x4_src, src_tform4x4_ref_mean[..., None, :, :])
    # src_rot_scaled_transl = a_tform4x4_src_rot_scaled[..., :3, 3]
    # ref_transl = a_tform4x4_ref[..., :3, 3]
    # src_rot_scaled_transl_mean = (src_rot_scaled_transl - ref_transl).mean(dim=-2)
    # src_tform4x4_ref_mean[..., :3, 3] = -src_rot_scaled_transl_mean / src_scale_ref
    #
    # # after translation
    # a_tform4x4_src_transf = tform4x4_broadcast(a_tform4x4_src, src_tform4x4_ref_mean[..., None, :, :])
    # src_transf_tform4x4_ref = tform4x4(inv_tform4x4(a_tform4x4_src_transf), a_tform4x4_ref)
    # logger.info('after translation')
    # logger.info(f'error transl: {src_transf_tform4x4_ref[..., :3, 3].norm(dim=-1).mean()}')
    # logger.info(f'error rot: {so3_log_map(src_transf_tform4x4_ref[..., :3, :3]).norm(dim=-1).mean()}')
    #
    # from o3b.cv.visual.show import show_scene
    # cams_tform = torch.cat([a_tform4x4_src_transf, a_tform4x4_ref])
    #
    # cams_intr = torch.eye(4)[None, ].expand(*cams_tform.shape).to(cams_tform.device, cams_tform.dtype).clone()
    # cams_intr[:, :2] *= 1000
    # cams_intr[:, 0:2, 2] = 500
    # cams_tform = cams_tform / cams_tform[:, :3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    # show_scene(cams_imgs_depth_scale=5., pts3d=[cams_tform[:, :3, 3]], cams_tform4x4_world=cams_tform[:30], cams_intr4x4=cams_intr[:30]) #renderer=OD3D_RENDERER.OPEN3D,)
    #
    # return src_tform4x4_ref_mean
