import logging

logger = logging.getLogger(__name__)
import torch.nn as nn
from o3b.model.model import OD3D_Model, register_model
from typing import List
from omegaconf import DictConfig
import torch
try:
    from o3b.model.litept.litept import LitePT as LitePT_Orig
except ImportError as e:
    logger.warning(f"LitePT dependencies not available, skipping registration: {e}")
    LitePT_Orig = None


@register_model("LitePT")
class LitePT(OD3D_Model):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=True,
        grid_size=0.02,
        out_feat_pool = "mean",
        out_feat_append_in_feat = True,
        out_feats_append_in_feats = False,
    ):
        super().__init__()

        self.out_feat_pool = out_feat_pool
        self.out_feat_append_in_feat = out_feat_append_in_feat
        self.out_feats_append_in_feats = out_feats_append_in_feats

        self.grid_size = grid_size
        self.model = LitePT_Orig(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            enc_conv=enc_conv,
            enc_attn=enc_attn,
            enc_rope_freq=enc_rope_freq,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            dec_conv=dec_conv,
            dec_attn=dec_attn,
            dec_rope_freq=dec_rope_freq,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enc_mode=enc_mode,
        )

        
        # self.nets = nn.ModuleList(nets)

    def forward(self, frames_gt, frames_pred=None):

        # feat_out =  self.net(torch.cat([frames_pred.pts3d, frames_pred.feats], dim=-1))

        #if frames_pred.feat is not None:
        #    frames_pred.feat = torch.cat([feat_out, frames_pred.feat], dim=-1)
        #else:
        #    frames_pred.feat = feat_out

        """
        data_dict is the batched input point cloud, it should contain as least:
        1. feat [N, input_dim]: input feature for the point cloud
        2. grid_coord [N, 3]: voxelized coordinate after grid sampling 
           or/and
           coord [N, 3]: original coordinate + grid_size: grid_size used for grid sampling
        3. offset [batch_size]: separator of point clouds in batched data
           or/and
           batch [N]: batch index of each point
        """
        # B and device from the point cloud the previous stage set, not from
        # len(frames_pred) / frames_pred.device: those are od3d's OD3D_Frames API,
        # which a plain frames bag (TrinityMethod's _Frames) does not have.
        B = frames_pred.pts3d.shape[0]
        map_points2batch_id = torch.arange(B, device=frames_pred.pts3d.device)
        map_points2batch_id = map_points2batch_id.unsqueeze(1).repeat(1, frames_pred.pts3d.shape[1]).view(-1)


        data_dict = {
            "feat": frames_pred.feats.view(-1, frames_pred.feats.shape[-1]),
            "coord": frames_pred.pts3d.view(-1, 3),
            "grid_size": self.grid_size, # 0.02 meters? = 2 cm?
            "batch": map_points2batch_id, 
        }

        res = self.model(data_dict=data_dict)
        # res['feat'].shape #  N x F ( B*num_points , F = 72)
        # sparse_conv_feat = res['sparse_conv_feat'] # same only saved different object
        # res.keys() # dict_keys(['feat', 'coord', 'grid_size', 'batch', 'offset', 'grid_coord', 'sparse_shape', 'sparse_conv_feat'])
        # feat_out = res # ["feat_out"]

        B = frames_pred.pts3d.shape[0]

        # if self.model.enc_mode:

        # feats_out = res['feat'].reshape(B, -1, res['feat'].shape[-1])
        # res['batch'] # id batch:  B*num_points
        
        # use onehot encoding for batch id

        batch_onehot = torch.nn.functional.one_hot(res['batch'], num_classes=B).to(torch.float32)
        if self.out_feat_pool == "max":
            feat = ((res['feat'][:, :, None] * batch_onehot[:, None]).max(dim=0).values ).permute(1, 0)  # [B, F]
        elif self.out_feat_pool == "cat":
            feat = res['feat'].reshape(*frames_pred.feats.shape[:2], -1)
            feat = feat.view(B, -1) # [B, num_points * F]
        else:
            feat = ((res['feat'][:, :, None] * batch_onehot[:, None]).sum(dim=0) / batch_onehot[:, None].sum(dim=0)).permute(1, 0)  # [B, F]

        if self.out_feat_append_in_feat and frames_pred.feat is not None:
            frames_pred.feat = torch.cat([feat, frames_pred.feat], dim=1) # 504 + 444  || 1024 * 72 + 444 = 74172
        else:
            frames_pred.feat = feat # 504 || 1024 * 72 = 73728
        
        if self.model.enc_mode: # encode only
            feats = None
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            from torch.nn.utils.rnn import pad_sequence, unpad_sequence
            from torch.nn.utils.rnn import unpack_sequence

            chunks = torch.split(tensor=res['feat'], split_size_or_sections=batch_onehot.sum(dim=0).cpu().long().tolist(), dim=0)  # list of [li, F]
            padded = pad_sequence(chunks, batch_first=True)  # [B, max_S, F]
            frames_pred.feats  = padded             
              
        else:
            feats = res['feat'].view(*frames_pred.feats.shape[:2], -1)

            if self.out_feats_append_in_feats and frames_pred.feats is not None:
                frames_pred.feats = torch.cat([feats, frames_pred.feats], dim=2)
            else:
                frames_pred.feats = feats

        # fuse_feat_pcl
        return frames_gt, frames_pred
