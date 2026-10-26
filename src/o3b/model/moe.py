"""Mixture-of-experts feed-forward layer, as F2G-Pose's (H02-T21).

F2G-Pose (arXiv 2610.31623, Sec. 3.2, Eq. 5–6; github.com/jrryzh/F2G-Pose
``src/f2g_pose/transformer.py`` ``MoeMlp``) replaces the feed-forward layer of
every transformer block with ``num_experts`` copies of it and a linear gate:

- softmax over the gate's scores, top-``top_k`` experts per token, the selected
  probabilities renormalised to sum to one and used to weight the experts'
  outputs;
- a load-balancing loss per layer, ``num_experts * sum_i f_i * P_i`` with
  ``P_i`` the mean gate probability of expert ``i`` over all tokens, in the
  paper's *soft* variant ``f_i = P_i`` (Switch Transformer uses the fraction of
  tokens routed to ``i`` instead). Minimum 1 at uniform probabilities.

Their code runs every selected expert on all tokens and masks the output
(~``num_experts`` x the FFN compute); here each expert sees only its own
tokens (~``top_k`` x). An expert can then get no tokens on a rank, which
``o3b.ddp.average_gradients`` handles (it materialises a zero gradient when
another rank has one).

The layer stores its loss in ``self.aux_loss`` during training (None in eval);
``moe_aux_loss(model)`` averages it over every MoE layer of a model, as
F2G-Pose's ``_collect_moe_aux_loss`` does.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn


class MoeMlp(nn.Module):
    def __init__(self, dim: int, make_expert: Callable[[], nn.Module], num_experts: int = 8, top_k: int = 2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"MoeMlp: top_k must be in [1, num_experts], got {top_k} of {num_experts}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        self.aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        t = x.reshape(-1, shape[-1])                                     # (T, D)
        probs = torch.softmax(self.gate(t).float(), dim=-1)              # (T, E), fp32 under autocast
        top_w, top_i = probs.topk(self.top_k, dim=-1)                    # (T, k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        out = None
        for e, expert in enumerate(self.experts):
            sel = top_i == e                                             # (T, k)
            rows = sel.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            y = expert(t[rows])
            w = (top_w * sel).sum(dim=-1)[rows].to(y.dtype)
            if out is None:
                out = torch.zeros(t.shape[0], y.shape[-1], device=t.device, dtype=y.dtype)
            out = out.index_add(0, rows, y * w[:, None])
        if out is None:                                                  # no tokens at all
            out = torch.zeros_like(t)
        if self.training:
            p = probs.mean(dim=0)                                        # P_i
            self.aux_loss = self.num_experts * (p * p).sum()             # soft f_i = P_i
        else:
            self.aux_loss = None
        return out.reshape(*shape[:-1], out.shape[-1])


def moe_aux_loss(model: nn.Module) -> Optional[torch.Tensor]:
    """Mean of the MoE layers' last load-balancing losses, None without any."""
    losses = [m.aux_loss for m in model.modules() if isinstance(m, MoeMlp) and m.aux_loss is not None]
    if not losses:
        return None
    return torch.stack([l.float() for l in losses]).mean()
