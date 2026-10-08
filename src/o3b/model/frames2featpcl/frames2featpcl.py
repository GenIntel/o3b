import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from copy import copy
from o3b.cv.geometry.transform import depth2pts3d_grid, cam_intr4x4_2_center_ray3d, cam_intr4x4_2_rays3d
from o3b.cv.geometry.grid import get_pxl2d_like, get_pxl2d
from o3b.cv.visual.sample import sample_pxl2d_grid, sample_pxl2d_pts
from o3b.cv.visual.sample import sample_pxl2d_pts
from o3b.cv.geometry.grid import get_pxl2d_like
from o3b.cv.geometry.downsample import random_sampling_with_fill

def encode_axes(axes: torch.Tensor, dim: int) -> torch.Tensor:
    ''' axes: Bx... '''
    bs = axes.shape[0]
    axes = axes.reshape(bs, -1, 1)
    embedding = []
    exponent = (2 ** torch.arange(dim, device=axes.device, dtype=torch.float32)).reshape(1, 1, -1)
    for fn in [torch.sin, torch.cos]:
        embedding.append(fn(exponent * axes).reshape(bs, -1))
    return torch.concat(embedding, dim=-1)


def frames2featpcl(frames_gt, frames_pred, 
                   pts_count=1024, 
                   append_rays3d=True,
                   normalize_feats=False,
                   filter_median=False, 
                   featmap_upsample_rate=2,
                   normalize_pts3d=False,
                   training=False,
                   augment_rot=True,
                   augment_scale_dev=0.2,
                   augment_transl_dev=0.1,
                   scale_quantile=0.9,
                   opengl=False,
                   ):
    # opengl: unproject depth into OpenGL camera space (+Y up, -Z forward), the
    # convention o3b's poses are written in. With normalize_pts3d the predicted
    # translation is mapped back through the cloud's normalisation, so it lands
    # in whatever frame the cloud is in: a CV-frame cloud against an OpenGL
    # target leaves the network to regress (C - I) @ centre — a distance-
    # proportional offset the normalisation has just divided out. False keeps
    # od3d's CV frame, which models trained in od3d expect.

    size = frames_gt.size # H, W

    # B x 3 x H x W
    # featmap = backbone_out.featmaps[-1]
    
    # featmap = frames_pred.featmaps[-1]
    featmap = frames_pred.featmap
    fH, fW = featmap.shape[-2:]
    
    featmap_down_sample_rate = size[1] // fW
    down_sample_rate = featmap_down_sample_rate / featmap_upsample_rate
    B = featmap.shape[0]
    dtype = featmap.dtype
    device = featmap.device

    if normalize_feats:
        featmap = torch.nn.functional.normalize(featmap, p=2, dim=1)

    depth = frames_gt.depth
    depth_mask = frames_gt.depth_mask
    mask = frames_gt.mask

    if  depth is None:
        depth = torch.ones(size=(B, 1, int(size[0]), int(size[1]))).to(dtype=dtype, device=device)
    if mask is None:
        mask = torch.ones(size=(B, 1, int(size[0]), int(size[1]))).to(dtype=dtype, device=device)
    if depth_mask is None:
        depth_mask = torch.ones(size=(B, 1, int(size[0]), int(size[1]))).to(dtype=dtype, device=device)
    
    depth_mask = (depth_mask * (depth > 1e-5)) * 1. 
    mask = mask * 1.

    if filter_median:
        depth_quantiles_thresholds = torch.Tensor([0.4, 0.5, 0.6]).to(dtype=depth.dtype, device=depth.device)
        depth_quantiles = depth.flatten(2).quantile(depth_quantiles_thresholds, dim = -1).permute(1, 2, 0)
        depth_min = depth_quantiles[..., 1] - (depth_quantiles[..., 1] - depth_quantiles[..., 0]) * 4. - 1e-10
        depth_max = depth_quantiles[..., 1] + (depth_quantiles[..., 2] - depth_quantiles[..., 1]) * 4. + 1e-10
        pts3d_grid_mask = (mask > 0.999) * (depth_mask > 0.999) * (depth >= depth_min[:, :, None, None]) * (
                    depth <= depth_max[:, :, None, None])
    else:
        pts3d_grid_mask = (mask > 0.999) * (depth_mask > 0.999)

    pts3d_zero_mask = pts3d_grid_mask.flatten(1).sum(dim=-1) == 0
    depth[pts3d_zero_mask] = 1.
    pts3d_grid_mask[pts3d_zero_mask] = 1
    # add nn layer norm or nn batch norm
    #featmap_res_perm = resize(featmap, H_out=depth.shape[-2], W_out=depth.shape[-1], mode="nearest_v2").permute(0, 2, 3, 1)
    pts3d_grid = depth2pts3d_grid(depth=depth, cam_intr4x4=frames_gt.cam_intr4x4, opengl=opengl).permute(0, 2, 3, 1).to(featmap.dtype)

    if append_rays3d:
        rays3d_grid = cam_intr4x4_2_rays3d(frames_gt.cam_intr4x4, size).permute(0, 2, 3, 1).to(featmap.dtype)
        if opengl:
            rays3d_grid = rays3d_grid * rays3d_grid.new_tensor([1., -1., -1.])
    else:
        rays3d_grid = None

    pts3d_grid = depth2pts3d_grid(depth=depth, cam_intr4x4=frames_gt.cam_intr4x4, opengl=opengl).permute(0, 2, 3, 1).to(featmap.dtype)
    pxl2d_grid = get_pxl2d_like(depth[:, 0, ..., None]).to(featmap.dtype)
    
    from o3b.cv.visual.resize import resize
    featmap_res = resize(featmap, scale_factor=featmap_upsample_rate * 1., mode="bilinear")
    featmap_res_H = featmap_res.shape[-2]
    featmap_res_W = featmap_res.shape[-1]
    
    pts3d_grid_mask = resize(pts3d_grid_mask, H_out=featmap_res_H, W_out=featmap_res_W, mode="nearest_v2")

    pts3d_zero_mask = pts3d_grid_mask.flatten(1).sum(dim=-1) == 0
    pts3d_grid[pts3d_zero_mask] = 1.
    pts3d_grid_mask[pts3d_zero_mask] = 1
    
    pts3d_grid = resize(pts3d_grid.permute(0, 3, 1, 2), H_out=featmap_res_H, W_out=featmap_res_W, mode="nearest_v2").permute(0, 2, 3, 1)
    pxl2d_grid = resize(pxl2d_grid.permute(0, 3, 1, 2), H_out=featmap_res_H, W_out=featmap_res_W, mode="nearest_v2").permute(0, 2, 3, 1)
    if rays3d_grid is not None:
        rays3d_grid = resize(rays3d_grid.permute(0, 3, 1, 2), H_out=featmap_res_H, W_out=featmap_res_W, mode="nearest_v2").permute(0, 2, 3, 1)

    pts3d = []
    rays3d = []
    #feats = []
    pxl2d = []
    for b in range(pts3d_grid.shape[0]):
        pts3d_b = pts3d_grid[b, pts3d_grid_mask[b, 0]]
        pxl2d_b = pxl2d_grid[b, pts3d_grid_mask[b, 0]]
        #feats_b = featmap_res_perm[b, pts3d_grid_mask[b, 0]]
        pts3d_b, pts_sampled_ids = random_sampling_with_fill(pts3d_cls=pts3d_b,
                                                            pts3d_count=pts_count,
                                                            return_ids=True)
        pxl2d_b = pxl2d_b[pts_sampled_ids]
        #feats_b = feats_b[pts_sampled_ids]
        pts3d.append(pts3d_b)
        pxl2d.append(pxl2d_b)

        if rays3d_grid is not None:
            rays3d_b = rays3d_grid[b, pts3d_grid_mask[b, 0]]
            rays3d_b = rays3d_b[pts_sampled_ids]
            rays3d.append(rays3d_b)
        #feats.append(feats_b)

    pts3d = torch.stack(pts3d)
    pxl2d = torch.stack(pxl2d)

    feats = sample_pxl2d_pts(x=featmap_res, pxl2d=pxl2d / down_sample_rate)
    feats = feats.contiguous()
    
    if append_rays3d:
        center_ray3d = cam_intr4x4_2_center_ray3d(frames_gt.cam_intr4x4, size)
        if opengl:
            center_ray3d = center_ray3d * center_ray3d.new_tensor([1., -1., -1.])
        center_ray3d_enc = encode_axes(center_ray3d, dim=10) # -> 10 * 6, 6 because 3 x/y/z * 2 sin/cos
    else:
        center_ray3d_enc = None
    
    if rays3d_grid is not None:
        rays3d = torch.stack(rays3d)
        rays3d_enc = encode_axes(rays3d.reshape(-1, 3), dim=10).reshape(*rays3d.shape[:-1], -1)
        feats = torch.cat([feats, rays3d_enc], dim=-1)

    feat_cls = frames_pred.feat
    if normalize_feats:
        feat_cls =  torch.nn.functional.normalize(feat_cls, p=2, dim=1)

    feat = feat_cls
    if append_rays3d:
        feat = torch.cat([feat_cls, center_ray3d_enc], dim=-1)
    
    frames_pred.feat = feat 
    frames_pred.feats = feats

    if normalize_pts3d:
        # pts3d: B x N x 3

        #pts3d_center = (pts3d.max(dim=-2).values + pts3d.min(dim=-2).values) / 2.0 # B x 3
        #pts3d_center = pts3d.mean(dim=1) # B x 3
        pts3d_center = pts3d.median(dim=1).values # B x 3

        transl = -pts3d_center # (pts3d.max(dim=-2).values + pts3d.min(dim=-2).values) / 2.0

        from o3b.cv.geometry.transform import inv_tform4x4, transf3d_broadcast, tform4x4_from_transl3d, get_a_tform4x4_b_scale1d_rot_and_transl

        obj_0c_tform4x4_obj = tform4x4_from_transl3d(transl)
    
        pts3d_c0 = transf3d_broadcast(
            pts3d=pts3d.clone(),
            transf4x4=obj_0c_tform4x4_obj[:, None],
        )

        #pts3d_dev = pts3d_c0.flatten(1).std(dim=-1)
        #pts3d_dev = ((pts3d_c0.abs().flatten(1).max(dim=-1).values))
        pts3d_dev = (pts3d_c0.abs().flatten(1).quantile(scale_quantile, dim=-1))

        scale = (1.0 / (2 * pts3d_dev + 1e-5)) # this would map to -0.5, 0.5, #  -1, 1
        
        if training and augment_scale_dev > 0:
            scale = scale * ((1.0 + torch.randn_like(scale) * augment_scale_dev).clamp(0.1, 10.0)) # between 0.8 and 1.2 times the scale for data augmentation during training
            # scale = scale * (0.8 + 0.4 * torch.rand_like(scale)) # between 0.8 and 1.2 times the scale for data augmentation during training

        scale = scale.clamp(min=1e-2, max=1e2)
                
        obj_nocs_0c_tform4x4_obj = get_a_tform4x4_b_scale1d_rot_and_transl(
            a_tform4x4_b=obj_0c_tform4x4_obj, scale1d = scale
        )

        if training and (augment_rot or augment_transl_dev > 0):
            # add random rotation around camera center ray for data augmentation during training
            from o3b.cv.geometry.transform import rot6d_to_tform4x4, tform4x4
            B = scale.shape[0]
            rand_tform4x4 = torch.eye(4, device=scale.device).unsqueeze(0).expand(B, -1, -1).clone()
            if augment_rot:
                rand_rot3x3 = rot6d_to_tform4x4(rot6d = 2 * torch.rand_like(scale[:, None].expand(-1, 6)) - 1.)
                rand_tform4x4[:, :3, :3] = rand_rot3x3[:, :3, :3]
            rand_tform4x4[:, :3, 3] = torch.randn(B, 3, device=scale.device) * augment_transl_dev
            obj_nocs_0c_tform4x4_obj = tform4x4(rand_tform4x4, obj_nocs_0c_tform4x4_obj)

        pts3d = transf3d_broadcast(
            pts3d=pts3d.clone(),
            transf4x4=obj_nocs_0c_tform4x4_obj[:, None,],
        )

        pts3d = pts3d.clamp(-10, 10)

        #from od3d.cv.visual.show import show_scene
        #show_scene(pts3d=pts3d)

        frames_pred.obj_tform4x4_obj_nocs_0c = inv_tform4x4(obj_nocs_0c_tform4x4_obj)
    
    frames_pred.pts3d = pts3d 

    return frames_gt, frames_pred
