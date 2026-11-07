import logging
logger = logging.getLogger(__name__)
import torch.nn as nn

# from od3d.models.heads.head import OD3D_Head
from typing import List
from omegaconf import DictConfig
import torch
from o3b.cv.geometry.transform import rotation_6d_to_matrix, transf4x4_from_rot3x3_and_transl3
from o3b.model.model import register_model
from o3b.model.mlp import MLP
from torch.nn import Softplus
import numpy as np

@register_model("PoseMLP")
class PoseMLP(MLP):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        dropout,
        in_dim = None,
        in_dims: List = None,
        activation = None,
        bias = False,
        out_dim = None,
        out_dims = None,
        rot6d_othant: List = [False, False, False, False, False, False],
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            in_dim=in_dim,
            in_dims=in_dims,
            activation=activation,
            bias=bias,
            out_dim=out_dim,
            out_dims=out_dims,
        )

        self.rot6d_othant = rot6d_othant

        self.softplus = Softplus(beta=2 * np.log(2))

        # in_dim: 1024
        # out_dim:  # 
        # out_dims: [3, 6, 3] # dims: translation, rotation_dim * count_othants, size
        # num_layers: 1 #
        # hidden_dim: 512
        # dropout: 0.5
        # activation: "none" # "tanh", "sigmoid", "norm", "norm_detach", "relu", "lrelu", "none"
        # bias: True # bias=True init_mode='kaiming_normal', init_weight=1, init_bias=0
        # rot6d_othant: [False, False, False, False, False, False]
        # # rot6d_othant: [ True, True, True, False, True, False ]

        
    def forward(self, frames_gt, frames_pred=None):

        frames_gt, frames_pred = super().forward(frames_gt=frames_gt, frames_pred=frames_pred)
        feat = frames_pred.feat
        out_dim_offset = 0
        
        if self.out_dims[0] == 3: # transl
            poses_transl3d = feat[..., out_dim_offset:self.out_dims[0]].clone()
            
            if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
                from o3b.cv.geometry.transform import transf3d_broadcast, tform4x4_broadcast, inv_tform4x4
                poses_transl3d = transf3d_broadcast(pts3d=poses_transl3d, transf4x4=frames_pred.obj_tform4x4_obj_nocs_0c)

            out_dim_offset += 3
        else:
            poses_transl3d = None
            msg = f"PoseMLP out_dims[0] != 3"
            raise ValueError(msg)
        
        if self.out_dims[1] == 6: # rot
            poses_rot6d = feat[..., out_dim_offset:out_dim_offset + self.out_dims[1]].clone()
            out_dim_offset += 6

            cam_rot3x3_obj = rotation_6d_to_matrix(poses_rot6d)
            
            
            if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
                from o3b.cv.geometry.transform import transf3d_broadcast, rot3x3, inv_tform4x4, rem_scale_tform4x4
                cam_rot3x3_obj = rot3x3(rem_scale_tform4x4(frames_pred.obj_tform4x4_obj_nocs_0c)[..., :3, :3], cam_rot3x3_obj)
                
            frames_pred.cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=cam_rot3x3_obj, transl3=poses_transl3d)
        else:
            msg = f"PoseMLP out_dims[1] != 6"
            raise ValueError(msg)
        
        if self.out_dims[2] == 3:
            frames_pred.obj_size3d = self.softplus(feat[..., out_dim_offset:out_dim_offset + self.out_dims[2]].clone())
        
            if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
                from o3b.cv.geometry.transform import rot3d_broadcast, get_scale1d_tform4x4
                scale1d_to_obj_from_obj_nocs0 = get_scale1d_tform4x4(frames_pred.obj_tform4x4_obj_nocs_0c, keepdim=False)
                frames_pred.obj_size3d = frames_pred.obj_size3d * scale1d_to_obj_from_obj_nocs0[:, None]

        else:
            msg = f"PoseMLP out_dims[2] != 3"
            raise ValueError(msg)

        if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
            frames_pred.obj_tform4x4_obj_nocs_0c = None


        #for key, value in vars(frames_pred).items():
        #    if value is not None and isinstance(value, torch.Tensor) and value.requires_grad:
        #        logger.info(key)
                # cam_tform4x4_obj
                # obj_size3d
                # feat
                #frames_pred.__dict__[key] = value.detach()
        
        if frames_pred.feat is not None:
            frames_pred.feat = frames_pred.feat.detach()
        
        if frames_pred.featmap is not None:
            frames_pred.featmap = frames_pred.featmap.detach()
        
        if frames_pred.feats is not None:
            frames_pred.feats = frames_pred.feats.detach()
        
        if frames_pred.featmaps is not None:
            frames_pred.featmaps = [featmap.detach() for featmap in frames_pred.featmaps]

        return frames_gt, frames_pred

        # pred_cam_rot6d_objs = matrix_to_rotation_6d(cam_tform4x4_objs[..., :3, :3].clone())
        # gt_cam_rot6d_objs = matrix_to_rotation_6d(
        #     batch.cam_tform4x4_obj[..., :3, :3].clone(),
        # )

        # pred_cam_transl3d_objs = cam_tform4x4_objs[..., :3, 3].clone()
        
        # pred_othant = pred_cam_rot6d_objs[
        #     ...,
        #     self.net_pose_rotations_othant,
        # ].sign()
        # pred_othant[pred_othant == 0.] = 1.
        # pred_othant = pred_othant.clone()

        # gt_othant = gt_cam_rot6d_objs[..., self.net_pose_rotations_othant].sign()
        # gt_othant[gt_othant == 0.] = 1.
        # gt_othant = gt_othant.clone()
        # mask_othant_equal = (pred_othant == gt_othant[:, None]).all(dim=-1)
        # pred_cam_rot6d_objs_sel = pred_cam_rot6d_objs[mask_othant_equal]
        # pred_cam_transl3d_objs_sel = pred_cam_transl3d_objs[mask_othant_equal]
        # pred_objs_scale3d_sel = pred_objs_scale3d[mask_othant_equal]
        # loss_pose_rot = ((pred_cam_rot6d_objs_sel - gt_cam_rot6d_objs).norm(dim=-1, p=2)).mean()


        # poses_transl3d = poses.feat[..., 0:self.net_pose_transl_dim]
        # poses_rot6d =  (poses.feat[..., self.net_pose_transl_dim:
        #                            self.net_pose_transl_dim + self.net_pose_rotations * self.net_pose_rotation_dim].
        #               reshape(*poses.feat.shape[:-1], self.net_pose_rotations, self.net_pose_rotation_dim ))

        # if self.net_pose_scale_dim == 1:
        #     objs_scale3d = poses.feat[..., -1:].clone()
        #     if self.pose_net_softplus:
        #         objs_scale3d = softplus(objs_scale3d.clone())
        #     elif val:
        #         objs_scale3d = objs_scale3d.clamp(1e-10)

        # elif self.net_pose_scale_dim == 3:
        #     objs_scale3d = poses.feat[..., -3:].clone()
        #     if self.pose_net_softplus:
        #         objs_scale3d = softplus(objs_scale3d)
        #     elif val:
        #         objs_scale3d = objs_scale3d.clamp(1e-10)
        # else:
        #     objs_scale3d = torch.ones_like(poses_transl3d)

        # if poses_rot6d.dim() == 2:
        #     S = 1
        #     R = 1
        #     poses_rot6d = poses_rot6d[:, None, None]
        # elif poses_rot6d.dim() == 3:
        #     if self.net_pose_rotations == 1:
        #         S = poses_rot6d.shape[1]
        #         R = 1
        #         poses_rot6d = poses_rot6d[:, :, None,]
        #     else:
        #         S = 1
        #         R = poses_rot6d.shape[1]
        #         poses_rot6d = poses_rot6d[:, None, ]
        # elif poses_rot6d.dim() == 4:
        #     S = poses_rot6d.shape[1]
        #     R = poses_rot6d.shape[2]
        # else:
        #     msg = f"unknown pose_rot6d dimensions {poses_rot6d.shape}"
        #     raise NotImplementedError(msg)

        # if poses_transl3d.dim() == 2:
        #     poses_transl3d = poses_transl3d[:, None, None].repeat(1, S, R, 1)
        # elif poses_transl3d.dim() == 3:
        #     poses_transl3d = poses_transl3d[:, :, None].repeat(1, 1, R, 1)

        # if objs_scale3d.dim() == 2:
        #     if self.net_pose_scale_dim == 1:
        #         objs_scale3d = objs_scale3d[:, None, None].repeat(1, S, R, 3)
        #     else:
        #         objs_scale3d = objs_scale3d[:, None, None].repeat(1, S, R, 1)
        # elif objs_scale3d.dim() == 3:
        #     if self.net_pose_scale_dim == 1:
        #         objs_scale3d = objs_scale3d[:, :, None].repeat(1, 1, R, 3)
        #     else:
        #         objs_scale3d = objs_scale3d[:, :, None].repeat(1, 1, R, 1)

        # # [B, T, R, 9])
        # # rot 6d to matrix, [x, y, z], x=lookat, y=up
        # # zero dim: [False, False, False, False, False, False]
        # # add othant: [True, True, True, False, True, False]
        # # 2 ** 6 # 64
        # if self.pose_net_softplus:
        #     poses_rot6d[..., self.net_pose_rotations_othant] = softplus(
        #         poses_rot6d[..., self.net_pose_rotations_othant],
        #     ).clone()

        #     # self.net_pose_rotations_othant_signs #R x 9
        #     poses_rot6d = poses_rot6d * self.net_pose_rotations_othant_signs[None, None].to(
        #         device=self.device,
        #     )


_SOFTPLUS = Softplus(beta=2 * np.log(2))


def pose_from_raw(raw: torch.Tensor, obj_tform4x4_obj_nocs_0c: torch.Tensor = None):
    """PoseMLP's decoding of its 12 outputs (translation 3, rotation 6D, size 3 before
    the softplus) into ``(cam_tform4x4_obj, obj_size3d)``, for models that predict
    and refine the raw vector themselves (ShapeHier's per-level pose). With
    ``obj_tform4x4_obj_nocs_0c`` the outputs are in the normalised cloud frame and
    are mapped back as PoseMLP does."""
    from o3b.cv.geometry.transform import (transf3d_broadcast, rot3x3, rem_scale_tform4x4,
                                           get_scale1d_tform4x4)
    transl = raw[..., :3]
    rot = rotation_6d_to_matrix(raw[..., 3:9])
    size = _SOFTPLUS(raw[..., 9:12])
    if obj_tform4x4_obj_nocs_0c is not None:
        transl = transf3d_broadcast(pts3d=transl, transf4x4=obj_tform4x4_obj_nocs_0c)
        rot = rot3x3(rem_scale_tform4x4(obj_tform4x4_obj_nocs_0c)[..., :3, :3], rot)
        size = size * get_scale1d_tform4x4(obj_tform4x4_obj_nocs_0c, keepdim=False)[:, None]
    return transf4x4_from_rot3x3_and_transl3(rot3x3=rot, transl3=transl), size
