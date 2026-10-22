from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


class _StopForward(Exception):
    """Raised by the up-block hook: nothing after the read block is needed."""


def _from_pretrained(cls, repo: str, **kwargs):
    """``cls.from_pretrained``, falling back to the local cache when the hub is
    unreachable or refuses the lookup (gated / taken-down repos that are cached)."""
    try:
        return cls.from_pretrained(repo, **kwargs)
    except Exception as e:  # noqa: BLE001 - any hub failure; the cache decides
        logger.warning("%s: hub lookup failed (%s), retrying from the local cache", repo, type(e).__name__)
        return cls.from_pretrained(repo, local_files_only=True, **kwargs)


@register_model("SDFeat")
class SDFeat(OD3D_Model):
    """Stable Diffusion UNet features, DIFT recipe (Tang et al., NeurIPS 2023).

    Each image is resized to ``image_size``, VAE-encoded (latent mean), noised
    to timestep ``t`` with ``ensemble`` seeded noise samples, and passed through
    the UNet with a fixed text prompt. The output of ``unet.up_blocks[up_ft_index]``
    is averaged over the samples and returned as the feature map. The forward
    pass stops after that block.

    Defaults are DIFT's: t = 261, ensemble 8. ``up_ft_index`` picks the block
    whose output is at half the latent resolution with 1280 channels: 1 for
    SD 1.x / 2.x, 0 for SDXL, whose UNet has one up block fewer.

    ``xl: true`` builds SDXL's conditioning: both text encoders' penultimate
    hidden states concatenated, the pooled embedding and the size time ids.

    forward(frames) sets frames_pred.featmap (B, C, h, w), float32.
    """

    def __init__(
        self,
        hf_repo: str = "runwayml/stable-diffusion-v1-5",
        xl: bool = False,
        image_size: int = 512,
        t: int = 261,
        up_ft_index: int = 1,
        ensemble: int = 8,
        prompt: str = "a photo of an object",
        batch_size: int = 4,
        dtype: str = "float16",
        seed: int = 0,
        freeze: bool = True,
    ):
        super().__init__()
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

        self.hf_repo, self.xl = hf_repo, bool(xl)
        self.image_size, self.t, self.up_ft_index = int(image_size), int(t), int(up_ft_index)
        self.ensemble, self.prompt, self.batch_size = int(ensemble), prompt, int(batch_size)
        self.seed = int(seed)
        self.torch_dtype = getattr(torch, dtype)

        self.vae = self._load(AutoencoderKL, subfolder="vae")
        if getattr(self.vae.config, "force_upcast", False):
            # SDXL's VAE overflows in fp16 (NaN latents); diffusers' pipelines
            # upcast it for encoding, as here
            self.vae.to(torch.float32)
        self.unet = self._load(UNet2DConditionModel, subfolder="unet")
        self.scheduler = _from_pretrained(DDPMScheduler, hf_repo, subfolder="scheduler")
        self._load_text_encoders()
        self.out_dim = self.unet.config.block_out_channels[::-1][self.up_ft_index]
        self._cond = None  # (prompt_embeds, added_cond_kwargs), built on first use on the right device

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
        self.eval()

    def _load(self, cls, **kwargs):
        """A weight-bearing component: the repo's fp16 variant when running in
        float16 (half the download; SDXL's UNet is 10 GB in fp32), else the default."""
        if self.torch_dtype == torch.float16:
            try:
                return _from_pretrained(cls, self.hf_repo, variant="fp16", torch_dtype=self.torch_dtype, **kwargs)
            except Exception as e:  # noqa: BLE001 - no fp16 variant in this repo / cache
                logger.info("%s/%s: no fp16 variant (%s), loading the default weights",
                            self.hf_repo, kwargs.get("subfolder"), type(e).__name__)
        return _from_pretrained(cls, self.hf_repo, torch_dtype=self.torch_dtype, **kwargs)

    def _load_text_encoders(self):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self.tokenizer = _from_pretrained(CLIPTokenizer, self.hf_repo, subfolder="tokenizer")
        self.text_encoder = self._load(CLIPTextModel, subfolder="text_encoder")
        if self.xl:
            self.tokenizer_2 = _from_pretrained(CLIPTokenizer, self.hf_repo, subfolder="tokenizer_2")
            self.text_encoder_2 = self._load(CLIPTextModelWithProjection, subfolder="text_encoder_2")

    @torch.no_grad()
    def _conditioning(self, device):
        if self._cond is not None and self._cond[0].device == device:
            return self._cond

        def ids(tok):
            return tok(self.prompt, padding="max_length", max_length=tok.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(device)

        if not self.xl:
            emb = self.text_encoder(ids(self.tokenizer))[0]
            added = None
        else:
            # as diffusers' StableDiffusionXLPipeline.encode_prompt: penultimate
            # hidden states of both encoders, pooled embedding of the second
            h1 = self.text_encoder(ids(self.tokenizer), output_hidden_states=True).hidden_states[-2]
            o2 = self.text_encoder_2(ids(self.tokenizer_2), output_hidden_states=True)
            emb = torch.cat([h1, o2.hidden_states[-2]], dim=-1)
            S = self.image_size
            added = {"text_embeds": o2.text_embeds,
                     "time_ids": torch.tensor([[S, S, 0, 0, S, S]], device=device, dtype=emb.dtype)}
        self._cond = (emb, added)
        return self._cond

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """(b, 3, S, S) in [-1, 1] -> (b, C, h, w) float32, averaged over the noise ensemble."""
        device = x.device
        emb, added = self._conditioning(device)
        lat = self.vae.encode(x.to(self.vae.dtype)).latent_dist.mean * self.vae.config.scaling_factor
        lat = lat.to(self.torch_dtype)
        b = lat.shape[0]
        g = torch.Generator(device=device).manual_seed(self.seed)
        noise = torch.randn((self.ensemble,) + tuple(lat.shape[1:]), generator=g, device=device, dtype=lat.dtype)
        t = torch.tensor([self.t], device=device)
        # every image gets the same ensemble of noise samples
        lat_e = lat.repeat_interleave(self.ensemble, dim=0)
        noisy = self.scheduler.add_noise(lat_e, noise.repeat(b, 1, 1, 1), t.repeat(lat_e.shape[0]))

        store = {}

        def hook(_module, _inp, out):
            store["ft"] = out
            raise _StopForward

        handle = self.unet.up_blocks[self.up_ft_index].register_forward_hook(hook)
        try:
            kwargs = {}
            if added is not None:
                kwargs["added_cond_kwargs"] = {k: v.expand(lat_e.shape[0], -1) for k, v in added.items()}
            self.unet(noisy, t.repeat(lat_e.shape[0]), encoder_hidden_states=emb.expand(lat_e.shape[0], -1, -1), **kwargs)
        except _StopForward:
            pass
        finally:
            handle.remove()
        ft = store["ft"].float()
        return ft.view(b, self.ensemble, *ft.shape[1:]).mean(1)

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
        x = x * 2.0 - 1.0
        feats = [self._features(x[i:i + self.batch_size]) for i in range(0, x.shape[0], self.batch_size)]
        frames_pred.featmap = torch.cat(feats, dim=0)
        return frames_gt, frames_pred
