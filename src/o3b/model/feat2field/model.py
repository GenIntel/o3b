import logging

logger = logging.getLogger(__name__)
from types import SimpleNamespace
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from o3b.model.model import OD3D_Model, register_model


class FeatField:
    """An implicit field conditioned on a global feature.

    Callable with query points ``pts3d`` (B, N, 3) and returns the field values
    (B, N, out_dim) -- an SDF for ``out_dim == 1``. The CoordMLP is shared with
    the :class:`Feat2Field` that produced this object, so gradients flow back
    into it; only the conditioning feature is bound here.

    ``sphere_init_radius`` adds the sphere prior ``|p| - r`` to the CoordMLP
    output, which keeps an untrained SDF field a well-behaved sphere.
    """

    def __init__(
        self,
        coordmlp,
        feat: Optional[torch.Tensor],
        out_dim: int = 1,
        sphere_init_radius: Optional[float] = None,
    ):
        self.coordmlp = coordmlp
        self.feat = feat
        self.out_dim = out_dim
        self.sphere_init_radius = sphere_init_radius

    @property
    def batch_size(self) -> Optional[int]:
        return None if self.feat is None else self.feat.shape[0]

    def __call__(self, pts3d: torch.Tensor) -> torch.Tensor:
        """pts3d (B, N, 3) -> (B, N, out_dim)."""
        if pts3d.dim() != 3 or pts3d.shape[-1] != 3:
            raise ValueError(
                f"FeatField: expected query points of shape (B, N, 3), "
                f"got {tuple(pts3d.shape)}.",
            )
        _f = SimpleNamespace(
            pts3d=pts3d,
            feat=self.feat,
            latent=None,
            latent_mu=None,
            latent_logvar=None,
            feat_mu=None,
            feat_logvar=None,
        )
        _, _pred = self.coordmlp.forward(frames_gt=_f, frames_pred=_f)
        vals = _pred.feat
        if self.sphere_init_radius is not None:
            vals = vals + (
                pts3d.detach().norm(dim=-1, keepdim=True) - self.sphere_init_radius
            )
        return vals

    def __repr__(self) -> str:
        return (
            f"FeatField(out_dim={self.out_dim}, batch_size={self.batch_size}, "
            f"sphere_init_radius={self.sphere_init_radius})"
        )


@register_model("Feat2Field")
class Feat2Field(OD3D_Model):
    """Turns a global feature into a coordinate-conditioned implicit field.

    Reads ``frames_pred.feat`` (B, F) and writes ``frames_pred.field``, a
    :class:`FeatField` that can be queried at arbitrary 3-D points. The field
    itself is a :class:`~o3b.model.coordmlp.model.CoordMLP` taking the harmonic
    embedding of the query point concatenated with the conditioning feature.

    With ``out_dim=1`` (default) and ``sphere_init_radius`` set this is an SDF
    field, which :class:`~o3b.model.field2mesh.model.Field2Mesh` turns into a
    triangle mesh.
    """

    def __init__(
        self,
        in_dim: int = None,
        in_dims: List = None,
        out_dim: int = 1,
        config: DictConfig = None,
        num_layers: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        activation: str = None,
        symmetrize: bool = True,
        n_harmonic_functions: int = 8,
        embed_concat_pts: bool = True,
        sphere_init_radius: float = 1.0,
    ):
        super().__init__()

        feat_dim = in_dims[-1] if in_dims is not None else (in_dim or 0)

        # CoordMLP config: explicit kwargs unless `config` supplies the od3d-style block
        if config is None:
            config = OmegaConf.create(
                {
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "out_dim": out_dim,
                    "dropout": dropout,
                    "activation": activation,
                    "symmetrize": symmetrize,
                },
            )
        else:
            out_dim = config.get("out_dim", out_dim)

        from o3b.model.coordmlp.model import CoordMLP

        self.coordmlp = CoordMLP(
            in_dims=[feat_dim],
            config=config,
            n_harmonic_functions=n_harmonic_functions,
            embed_concat_pts=embed_concat_pts,
        )
        self.out_dim = out_dim
        self.sphere_init_radius = sphere_init_radius

    def forward(self, frames_gt, frames_pred=None):
        if frames_pred is None:
            frames_pred = frames_gt

        feat = getattr(frames_pred, "feat", None)
        if feat is None:
            logger.warning(
                "Feat2Field: frames_pred.feat is None, field is unconditioned.",
            )

        frames_pred.field = FeatField(
            coordmlp=self.coordmlp,
            feat=feat,
            out_dim=self.out_dim,
            sphere_init_radius=self.sphere_init_radius,
        )
        return frames_gt, frames_pred
