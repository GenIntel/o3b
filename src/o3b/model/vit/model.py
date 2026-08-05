import logging

logger = logging.getLogger(__name__)
import torch
from timm.models.vision_transformer import Block
from typing import List, Type
from omegaconf import DictConfig
from torch import nn
from timm.layers import Mlp
from functools import partial
from o3b.model.model import OD3D_Model, register_model
from torch.nn.init import trunc_normal_
from typing import Sequence, Tuple, Union, Callable

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

@register_model("ViT")
class ViT(OD3D_Model):
    def __init__(
        self,
        in_featmap = False,
        in_featmap_dim = None,
        in_featmaps = False,
        in_featmaps_dims = None,
        in_feat = True,
        in_feat_dim = None,
        in_feats = True,
        in_feats_dim = None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm: bool = False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        blocks_dim = 128,
        blocks_depth = 8,
        blocks_num_heads = 6,
        out_feat = True,
    ):
        super().__init__()


        self.blocks_dim = blocks_dim  # 384, # 384 | 384*2
        self.blocks_depth = blocks_depth  # ,
        self.blocks_num_heads = blocks_num_heads  # 6, # 6 | 12

        self.in_featmap = in_featmap
        self.in_featmaps = in_featmaps
        self.in_feat = in_feat
        self.in_feats = in_feats
        
        self.in_feat_dim = in_feat_dim
        if self.in_feat_dim is None:
            self.in_feat_dim = self.blocks_dim
        
        self.in_feats_dim = in_feats_dim
        if self.in_feats_dim is None:
            self.in_feats_dim = self.blocks_dim

        self.in_featmap_dim = in_featmap_dim
        if self.in_featmap_dim is None:
            self.in_featmap_dim = self.blocks_dim


        self.out_feat = out_feat

        self.pos_embed = None
        if not self.in_feat and self.out_feat:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, blocks_dim), requires_grad=True)
        else:
            self.cls_token = None
        self.register_tokens = None

        norm_layer = partial(
            nn.LayerNorm,
            eps=1e-6,
        )  # dinov2/vggt uses this LayerNorm, prob not important
        act_layer = nn.GELU # dinov2/vggt uses this LayerNorm, prob not important

        
        if self.in_feat and self.in_feat_dim != self.blocks_dim:
            self.linear_in_feat = nn.Linear(self.in_feat_dim, self.blocks_dim)
        else:
            self.linear_in_feat = None
        
        if self.in_feats_dim != self.blocks_dim:
            self.linear_in_feats = nn.Linear(self.in_feats_dim, self.blocks_dim)
        else:
            self.linear_in_feats = None
        
        if self.in_featmap and self.in_featmap_dim != self.blocks_dim:
            self.linear_in_featmap = nn.Linear(self.in_featmap_dim, self.blocks_dim)
        else:
            self.linear_in_featmap = None
        self.linear_in_featmaps = None

        self.norm_dim_map = norm_layer(self.blocks_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.blocks_depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=self.blocks_dim,
                    num_heads=self.blocks_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(self.blocks_depth)
            ],
        )
        
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def forward(self, frames_gt, frames_pred=None):
        #if len(x.featmaps) > 1:
        #    logger.warning("ViT head only process last feature map.")

        x_in = []
        x_in_feat = False
        x_in_feats = False
        x_in_featmap = False
        x_in_featmaps = False
        x_in_cls = False
        x_in_feats_count = 0
        featmap_shape = None

        if self.in_feat and frames_pred.feat is not None:
            if self.linear_in_feat is not None:
                x_in.append(self.linear_in_feat(frames_pred.feat[:, None]))
            else:
                x_in.append(frames_pred.feat[:, None])
            x_in_feat = True
        elif not self.in_feat and self.out_feat:
            x_in_feat = True
            x_in_cls = True

        elif self.in_feat:
            logger.warning("frames_pred.feat is None")

        if self.in_feats and frames_pred.feats is not None:
            if self.linear_in_feats is not None:
                x_in.append(self.linear_in_feats(frames_pred.feats))
            else:
                x_in.append(frames_pred.feats)
            x_in_feats = True
            x_in_feats_count = x_in[-1].shape[-2]
        elif self.in_feats:
            logger.warning("frames_pred.feats is None")

        if self.in_featmap and frames_pred.featmap is not None:
            featmap = frames_pred.featmap  # BxCxHxW
            featmap_shape = featmap.shape
            tokens = featmap.flatten(2).permute(0, 2, 1)  # B x HW x C
            if self.linear_in_featmap is not None:
                tokens = self.linear_in_featmap(tokens)
            x_in.append(tokens)
            x_in_featmap = True
        elif self.in_featmap:
            logger.warning("frames_pred.featmap is None")

        if x_in_cls:
            # prepended, so the cls feature stays token 0 for all read-outs below
            B = x_in[0].shape[0] if len(x_in) > 0 else len(frames_gt)
            x_in.insert(0, self.cls_token.expand(B, -1, -1).clone())

        x_in = torch.cat(x_in, dim=-2)

        x_in = self.norm_dim_map(x_in)

        x_in = self.blocks(x_in)

        offset = 0
        if x_in_feat:
            frames_pred.feat = x_in[..., offset, :].clone()
            offset += 1

        if x_in_feats:
            frames_pred.feats = x_in[..., offset:offset + x_in_feats_count, :].clone()
            offset += x_in_feats_count

        if x_in_featmap:
            B, _, H, W = featmap_shape
            frames_pred.featmap = (
                x_in[..., offset:offset + H * W, :]
                .permute(0, 2, 1)
                .reshape(B, self.blocks_dim, H, W)
                .clone()
            )
            offset += H * W

        return frames_gt, frames_pred
