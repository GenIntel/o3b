import logging

logger = logging.getLogger(__name__)
from torch import nn
from omegaconf import DictConfig
import torch
from o3b.model.model import OD3D_Model, register_model
from o3b.data.ext_enum import ExtEnum
from o3b.cv.visual.resize import resize
from typing import Tuple
import math
import types


from .dinov1 import ViTExtractor  # for selecting keys, querys, values


class DINOv2_WEIGHTS(str, ExtEnum):
    DEFAULT = "default"
    NONE = "none"


@register_model("DINOv2")
class DINOv2(OD3D_Model):
    def __init__(
        self,
        freeze=True,
        downsample_rate=16,
        out_res=None,
        stride=None,
        normalize=False,
        transform_rgb_uint8_to_float=False,
        transform_rgb_normalize=True,
        hub_repo="facebookresearch/dinov2",
        hub_model="dinov2_vits14",  # dino_vits8, dino_vitb8, dino_vits16, dino_vitb16, dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        weights="default",
        layers_returned=[0],  # choose from [0, 1] start with deepest (1)
        pca=False,
        pca_dim=64,
    ):
        super().__init__()

        self.freeze = freeze
        self.normalize = normalize
        self.transform_rgb_uint8_to_float = transform_rgb_uint8_to_float
        self.transform_rgb_normalize = transform_rgb_normalize

        self.layers_returned = layers_returned # choose from [0, 1] start with deepest (1)
        self.layers_count = len(self.layers_returned)

        self.dinov2 = "dinov2" in hub_model
        self.dinov3 = "dinov3" in hub_model

        #import re
        #self.patch_size = int(re.search(r"\d+$", self.config.hub_model.replace("-", "_").split("_")[1]).group())
        #self.patch_size = int(re.search(r"\d+[plus]$", self.config.hub_model.replace("-", "_").split("_")[1]).group())

        import re

        match = re.match(r"dino[v2]*[v3]*_vit[a-z]*([0-9]+)[a-z]*", hub_model, re.I)
        if match and len(match.groups()) == 1:
            self.patch_size = int(match.groups()[0])
        else:
            msg = f"could not retrieve down sample rate dino from model name {hub_model}"
            raise Exception(msg)

        # match = re.match(r"dino[v2]*_vit[a-z]*([0-9]+)", self.config.hub_model, re.I)

        # "dino[v2]*_vit[a-z]*([0-9]+)"

        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = stride
        self.dino_patch_size = int(self.patch_size)
        self.downsample_rate_dino = self.dino_patch_size // (self.dino_patch_size // self.stride)

        if self.dinov2 or self.dinov3:
            if self.dinov2:
                self.extractor = torch.hub.load(
                    hub_repo,
                    hub_model,
                    pretrained=weights == "default")
            elif self.dinov3:
                #  weights[hub_model]
                self.extractor = torch.hub.load(
                    hub_repo,
                    hub_model,
                    source="github",  # github / local
                    weights=weights[hub_model],
                )

            self.out_dims = [self.extractor.embed_dim]
            self.extractor = self.patch_vit_resolution(self.extractor, self.stride)

        else:  # using keys did not show any improvement
            self.extractor = ViTExtractor(model_type=hub_model)
            self.out_dims = [self.extractor.model.embed_dim]

        self.out_downsample_scales = []
        self.downsample_rate = downsample_rate
        self.out_res = out_res

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

            # if not self.dinov2:
            #     for param in self.extractor.model.parameters():
            #         param.requires_grad = False

        if pca:
            self.pca_enabled = False
            self.pca_layer = nn.Linear(self.out_dims[-1], pca_dim, bias=False)
            self.mean_features = nn.Parameter(
                torch.zeros(1, self.out_dims[-1]),
                requires_grad=False,
            )
            for param in self.pca_layer.parameters():
                param.requires_grad = False
            nn.init.eye_(self.pca_layer.weight)
            self.out_dims[-1] = pca_dim
        else:
            self.pca_enabled = False

        self.out_dim = self.out_dims[-1]

    def forward(self, frames_gt, frames_pred=None):
        if frames_pred is None:
            frames_pred = frames_gt

        x = frames_gt.rgb

        if self.transform_rgb_uint8_to_float:
            x = x.float() / 255.0

        if self.transform_rgb_normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std

        if x.dim() == 3:
            C, H, W = x.shape
            B = 1
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise NotImplementedError

        if self.out_res is not None:
            downsample_rate = H // self.out_res
        else:
            downsample_rate = self.downsample_rate
            
        H_out = H // downsample_rate
        W_out = W // downsample_rate
        H_out_expected = (
            ((H_out * self.downsample_rate_dino) - self.dino_patch_size) // self.stride
        ) + 1
        W_out_expected = (
            ((W_out * self.downsample_rate_dino) - self.dino_patch_size) // self.stride
        ) + 1
        offset_H = H_out - H_out_expected
        offset_H = self.round_up_to_even(offset_H)

        offset_W = W_out - W_out_expected
        offset_W = self.round_up_to_even(offset_W)
        H_in = H_out * self.downsample_rate_dino + offset_H * self.downsample_rate_dino
        W_in = W_out * self.downsample_rate_dino + offset_W * self.downsample_rate_dino

        x = resize(x, H_out=H_in, W_out=W_in)

        if self.dinov2 or self.dinov3:
            x_dict = self.extractor.forward_features(x)
            # # note: with layer normalization
            # x_feat_map = x_dict[
            #     "x_norm_patchtokens"
            # ] # 'x_norm_patchtokens', 'x_prenorm'[:, 1:]
            # x_feat_cls = x_dict["x_norm_clstoken"]

            # note: without layer normalization
            x_feat_map = x_dict["x_prenorm"][
                :,
                -((H_out_expected + offset_H) * (W_out_expected + offset_W)):,
            ]  # 'x_norm_patchtokens', 'x_prenorm'[:, 1:]
            x_feat_cls = x_dict["x_prenorm"][:, 0]

        else:
            # x = self.extractor.get_intermediate_layers(x, n=12)[9]  # maximum 12 layers, zsp uses 9
            # x = x[:, 1:] # remove cls token

            x_feat_cat = self.extractor.extract_descriptors(
                batch=x,
                layer=9,
                facet="key",
                bin=False,
                include_cls=True,
            )
            x_feat_map = x_feat_cat[:, :, 1:]
            x_feat_cls = x_feat_cat[:, :, 0]
            # note: key layer 9 outperforms layer 9

        x_feat_map = x_feat_map.reshape(
            B,
            H_out_expected + offset_H,
            W_out_expected + offset_W,
            -1,
        ).permute(0, 3, 1, 2)

        x_feat_cls = x_feat_cls.reshape(
            B,
            1,
            1,
            -1,
        ).permute(0, 3, 1, 2)
        x_feat_map = x_feat_map[:, :, :H_out, :W_out]

        if self.normalize:
            x_feat_cls = x_feat_cls / (x_feat_cls.norm(dim=-3, keepdim=True) + 1e-10)
            x_feat_map = x_feat_map / (x_feat_map.norm(dim=-3, keepdim=True) + 1e-10)

        B, C, H, W = x_feat_map.shape
        if self.pca_enabled:
            x_feat_map = torch.flatten(x_feat_map, 2)
            x_feat_map = x_feat_map.permute(0, 2, 1)
            x_feat_map = (
                x_feat_map - self.mean_features[(None,) * (x_feat_map.dim() - 2)]
            )
            x_feat_map = self.pca_layer(x_feat_map)
            x_feat_map = x_feat_map.permute(0, 2, 1)
            x_feat_map = x_feat_map.view(B, self.out_dims[-1], H, W)

            x_feat_cls = torch.flatten(x_feat_cls, 2)
            x_feat_cls = x_feat_cls.permute(0, 2, 1)
            x_feat_cls = (
                x_feat_cls - self.mean_features[(None,) * (x_feat_cls.dim() - 2)]
            )
            x_feat_cls = self.pca_layer(x_feat_cls)
            x_feat_cls = x_feat_cls.permute(0, 2, 1)
            x_feat_cls = x_feat_cls.view(B, self.out_dims[-1], 1, 1)

            if self.normalize:
                x_feat_cls = x_feat_cls / (
                    x_feat_cls.norm(dim=-3, keepdim=True) + 1e-10
                )
                x_feat_map = x_feat_map / (
                    x_feat_map.norm(dim=-3, keepdim=True) + 1e-10
                )

        #x_layers = []
        #x_layers.append(x_feat_map)
        #x_layers.append(x_feat_cls)
        #x_layers = [x_layers[layer_id] for layer_id in self.layers_returned]

        #frames_pred.featmaps = x_layers
        
        frames_pred.featmap = x_feat_map #  x_layers[-1]
        frames_pred.feat = x_feat_cls[:, :, 0, 0]

        return frames_gt, frames_pred

    @staticmethod
    def round_up_to_even(num: int) -> int:
        return num if num % 2 == 0 else num + 1

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        def interpolate_pos_encoding(self, x, w, h):
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            M = int(math.sqrt(N))  # Recover the number of patches in each dimension
            assert N == M * M
            kwargs = {}
            if self.interpolate_offset:
                # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
                # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
                sx = float(w0 + self.interpolate_offset) / M
                sy = float(h0 + self.interpolate_offset) / M
                kwargs["scale_factor"] = (sx, sy)
            else:
                # Simply specify an output size instead of a scale factor
                kwargs["size"] = (w0, h0)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            assert (w0, h0) == patch_pos_embed.shape[-2:]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
                previous_dtype,
            )

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn.modules.utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride],
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride

        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride),
            model,
        )

        # model.rope_embed

        return model
