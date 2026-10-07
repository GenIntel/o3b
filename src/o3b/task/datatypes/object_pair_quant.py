from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class ObjectPairQuantBatch:
    """Quantitative metrics for a batch of B object pairs.

    All distances are measured on the **target** mesh (round-trip evaluation):
      target query → nearest target vertex → feature → nearest source vertex
      → nearest target vertex (predicted) → error vs. target GT.
    """

    # ── per-keypoint tensors ──────────────────────────────────────────────────
    kpts_trgt_euc_dist:  Optional[Tensor] = None  # (B, K)  Euclidean dist on target
    kpts_trgt_geo_dist:  Optional[Tensor] = None  # (B, K)  normalized geodesic dist on target
    kpts_mask:           Optional[Tensor] = None  # (B, K)  bool — valid keypoints

    # ── per-sample keypoint aggregates ───────────────────────────────────────
    kpts_trgt_pck01:              Optional[Tensor] = None  # (B,)  PCK @ 0.1 * trgt_max_dim
    kpts_trgt_pck005:             Optional[Tensor] = None  # (B,)  PCK @ 0.05 * trgt_max_dim
    kpts_trgt_euc_dist_mean:      Optional[Tensor] = None  # (B,)  mean Euclidean dist
    kpts_trgt_geo_dist_mean:      Optional[Tensor] = None  # (B,)  mean geodesic dist
    kpts_trgt_geo_dist_norm_mean: Optional[Tensor] = None  # (B,)  mean geodesic dist / sqrt(surface area)
    kpts_trgt_geo_auc01:            Optional[Tensor] = None  # (B,)  AUC of PCK-geo curve [0, 0.1] normalized

    # ── per-sample part aggregates ────────────────────────────────────────────
    parts_trgt_euc_dist_mean:      Optional[Tensor] = None  # (B,)  mean Euclidean dist
    parts_trgt_geo_dist_mean:      Optional[Tensor] = None  # (B,)  mean geodesic dist
    parts_trgt_geo_dist_norm_mean: Optional[Tensor] = None  # (B,)  mean geodesic dist / sqrt(surface area)
    parts_trgt_geo_auc01:            Optional[Tensor] = None  # (B,)  AUC of PCK-geo curve [0, 0.1] normalized
    parts_trgt_pck:                Optional[Tensor] = None  # (B,)  fraction same part predicted

    # extra per-task metrics
    extra: dict = field(default_factory=dict)

    def mean(self) -> dict:
        """Return a flat dict of scalar means over the batch dimension.

        Uses nanmean so that samples where geodesic could not be computed
        (disconnected mesh components → nan) are excluded rather than
        propagating nan/inf into the logged metric.
        """
        import torch as _torch
        out = {}
        for fname in (
            "kpts_trgt_pck01",
            "kpts_trgt_pck005",
            "kpts_trgt_euc_dist_mean",
            "kpts_trgt_geo_dist_mean",
            "kpts_trgt_geo_dist_norm_mean",
            "kpts_trgt_geo_auc01",
            "parts_trgt_euc_dist_mean",
            "parts_trgt_geo_dist_mean",
            "parts_trgt_geo_dist_norm_mean",
            "parts_trgt_geo_auc01",
            "parts_trgt_pck",
        ):
            val = getattr(self, fname)
            if val is not None:
                v = val.float()
                finite = v[_torch.isfinite(v)]
                if len(finite) > 0:
                    out[fname] = finite.mean().item()
        for k, v in self.extra.items():
            if v is not None and hasattr(v, "mean"):
                out[k] = v.mean().item()
        return out

    def to_wandb_log(self, prefix: str = "batch", wb=None) -> dict:
        out: dict = {f"{prefix}/{k}": v for k, v in self.mean().items()}
        if wb is None:
            return out
        for fname in ("kpts_trgt_euc_dist", "kpts_trgt_geo_dist"):
            val = getattr(self, fname)
            if val is None:
                continue
            if self.kpts_mask is not None:
                data = val[self.kpts_mask].detach().cpu().float().numpy()
            else:
                data = val.detach().cpu().float().flatten().numpy()
            if data.size:
                out[f"{prefix}/{fname}"] = wb.Histogram(data)
        return out
