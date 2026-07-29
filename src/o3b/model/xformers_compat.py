"""Blackwell (sm_120) compatibility shim for ``xformers.ops.memory_efficient_attention``.

xformers ships no fused kernel that accepts this GPU: ``cutlassF`` and ``fa3F``
both refuse capability (12, 0), and ``fa2F`` caps head_dim at 256 while Stable
Diffusion's VAE self-attention uses 512. Every backend therefore fails to
dispatch and xformers raises ``NotImplementedError``.

``XFORMERS_DISABLED=1`` fixes DINOv2 (it checks the env var itself) but not the
DenseMatcher stack: ``ldm`` and ``odise`` decide at *import* time via a bare
``import xformers`` and then call ``memory_efficient_attention`` directly, so
there is no env var or config to turn it off.

Replacing the function with a ``F.scaled_dot_product_attention`` equivalent
covers every call site at once, whichever module made the decision. PyTorch's
SDPA computes the same thing — it just picks a kernel that supports this GPU —
so features are unchanged, at some memory cost. This mirrors the ``flash_attn``
stub the CoSMo3D encoder installs for the same class of problem.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_xformers_memory_efficient_attention() -> None:
    """Point ``xformers.ops.memory_efficient_attention`` at PyTorch SDPA. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import torch
        import xformers.ops
    except ImportError:
        _PATCHED = True  # nothing to patch — nothing imports it either
        return

    # Only Blackwell and newer need this; leave working setups on the fused path.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        _PATCHED = True
        return

    import torch.nn.functional as F

    def _memory_efficient_attention(query, key, value, attn_bias=None, p=0.0,
                                    scale=None, op=None, **kwargs):
        # xformers accepts two layouts, and both are used in this stack:
        #   4-D (B, M, H, K) — SD's VAE self-attention
        #   3-D (B, M, K)    — the UNet cross-attention, which folds heads into
        #                      the batch dim before calling (ldm attention.py)
        # SDPA always wants (B, H, M, K), so normalise, then restore.
        if attn_bias is not None and not torch.is_tensor(attn_bias):
            raise NotImplementedError(
                "xformers SDPA shim supports only None or tensor attn_bias, "
                f"got {type(attn_bias).__name__}"
            )
        squeeze_heads = query.ndim == 3
        if squeeze_heads:
            q, k, v = (t.unsqueeze(1) for t in (query, key, value))
        else:
            q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale
        )
        out = out.squeeze(1) if squeeze_heads else out.transpose(1, 2)
        return out.contiguous()

    xformers.ops.memory_efficient_attention = _memory_efficient_attention
    # ldm/odise bind the symbol via `import xformers` + `xformers.ops.…`, so
    # patching the attribute above is enough for them. Also patch the fmha
    # module, which some call sites import from directly.
    try:
        import xformers.ops.fmha as _fmha
        _fmha.memory_efficient_attention = _memory_efficient_attention
    except ImportError:
        pass

    logger.info(
        "xformers: memory_efficient_attention replaced with PyTorch SDPA "
        "(no fused kernel for sm_%d%d)", *torch.cuda.get_device_capability()
    )
    _PATCHED = True
