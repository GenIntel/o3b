from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from o3b.model.model import OD3D_Model, register_model
from o3b.model.sd_feat.model import _from_pretrained

logger = logging.getLogger(__name__)


@register_model("SAMFeat")
class SAMFeat(OD3D_Model):
    """Image-encoder features of Segment Anything 1 / 2 / 3, via transformers.

    The image is resized to the model's input size (from its image processor:
    1024 for SAM 1 / 2, 1008 for SAM 3) and normalised with the processor's
    mean and std. ``layer`` picks the output:

    - ``backbone``: the image encoder's last layer before any neck. SAM 1: ViT
      (1280-d for ViT-H, stride 16). SAM 2: Hiera's last stage (stride 32).
      SAM 3: the Perception Encoder ViT (1024-d, stride 14).
    - ``neck``: the 256-d map the mask decoder reads. SAM 1: the neck output
      (stride 16). SAM 2: the stride-16 FPN level, without the no-memory term
      ``get_image_embeddings`` adds. SAM 3: the stride-14 FPN level.

    forward(frames) sets frames_pred.featmap (B, C, h, w), float32.
    """

    def __init__(
        self,
        hf_repo: str = "facebook/sam-vit-huge",
        version: int = 1,
        layer: str = "neck",
        batch_size: int = 8,
        dtype: str = "float16",
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoImageProcessor

        if layer not in ("backbone", "neck"):
            raise ValueError(f"SAMFeat layer: {layer!r}, expected 'backbone' or 'neck'")
        self.hf_repo, self.version, self.layer = hf_repo, int(version), layer
        self.batch_size = int(batch_size)
        self.torch_dtype = getattr(torch, dtype)

        proc = _from_pretrained(AutoImageProcessor, hf_repo)
        size = proc.size
        self.image_size = int(size.get("height") or size.get("longest_edge")) if hasattr(size, "get") \
            else int(size["height"] if "height" in size else size["longest_edge"])
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False)

        if self.version == 1:
            from transformers import SamModel as _M
        elif self.version == 2:
            from transformers import Sam2Model as _M
        elif self.version == 3:
            from transformers import Sam3Model as _M
        else:
            raise ValueError(f"SAMFeat version: {version!r}, expected 1, 2 or 3")
        # only the image encoder is used; the prompt / mask decoders are dropped
        self.encoder = _from_pretrained(_M, hf_repo, torch_dtype=self.torch_dtype).vision_encoder

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
        self.eval()

    @staticmethod
    def _to_bchw(h: torch.Tensor) -> torch.Tensor:
        """(B, N, C) token sequence or (B, H, W, C) channels-last map -> (B, C, H, W)."""
        if h.dim() == 3:
            B, N, C = h.shape
            s = int(round(N ** 0.5))
            if s * s != N:
                raise ValueError(f"SAMFeat: {N} tokens are not a square grid")
            return h.transpose(1, 2).reshape(B, C, s, s)
        return h.permute(0, 3, 1, 2)

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.torch_dtype)
        if self.version == 1:
            o = self.encoder(x, output_hidden_states=self.layer == "backbone")
            # last_hidden_state is the neck output (B, 256, 64, 64); hidden_states[-1]
            # the ViT's last block, channels-last (B, 64, 64, 1280)
            f = o.last_hidden_state if self.layer == "neck" else self._to_bchw(o.hidden_states[-1])
        elif self.version == 2:
            o = self.encoder(x, return_dict=True)
            # fpn levels are strides 4 / 8 / 16 (B, 256, h, w); Hiera's last stage
            # is channels-last (B, 32, 32, 1152) for Hiera-L
            f = o.fpn_hidden_states[-1] if self.layer == "neck" else self._to_bchw(o.last_hidden_state)
        else:
            o = self.encoder(x)
            # fpn levels are strides 3.5 / 7 / 14 / 28; level 2 is the backbone's own stride
            f = o.fpn_hidden_states[2] if self.layer == "neck" else self._to_bchw(o.last_hidden_state)
        return f.float()

    @torch.no_grad()
    def forward(self, frames_gt, frames_pred=None):
        if frames_pred is None:
            frames_pred = frames_gt
        x = frames_gt.rgb
        if x.dim() == 3:
            x = x[None]
        x = x.float()
        if x.max() > 1.5:  # uint8 range
            x = x / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        feats = [self._features(x[i:i + self.batch_size]) for i in range(0, x.shape[0], self.batch_size)]
        frames_pred.featmap = torch.cat(feats, dim=0)
        return frames_gt, frames_pred
