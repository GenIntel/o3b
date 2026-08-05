import logging

logger = logging.getLogger(__name__)
from typing import List

from omegaconf import DictConfig

from o3b.model.model import OD3D_Model, register_model


@register_model("Feat2Mesh")
class Feat2Mesh(OD3D_Model):
    """Decodes a global feature into a triangle mesh: Feat2Field + Field2Mesh.

    ``frames_pred.feat`` (B, F) conditions a CoordMLP SDF field
    (:class:`~o3b.model.feat2field.model.Feat2Field`), which is then sampled on a
    tetrahedral grid and converted to an explicit mesh by differentiable marching
    tetrahedra (:class:`~o3b.model.field2mesh.model.Field2Mesh`). Both stages are
    usable on their own; this model is the convenience composition and produces
    exactly what running the two in sequence produces -- ``field`` from the first,
    ``meshes`` / ``mesh`` / ``mesh_sdf`` from the second.

    ``field`` / ``mesh`` accept the sub-model configs verbatim; the flat keyword
    arguments below are forwarded to whichever stage owns them.
    """

    def __init__(
        self,
        in_dim: int = None,
        in_dims: List = None,
        config: DictConfig = None,
        num_layers: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        activation: str = None,
        symmetrize: bool = True,
        n_harmonic_functions: int = 8,
        embed_concat_pts: bool = True,
        init_radius: float = 1.0,
        tet_res: int = 16,
        grid_scale_rel: float = 2.2,
        jitter_scale: float = 0.05,
        field: DictConfig = None,
        mesh: DictConfig = None,
    ):
        super().__init__()

        from o3b.model.feat2field.model import Feat2Field
        from o3b.model.field2mesh.model import Field2Mesh

        field_kwargs = dict(
            in_dim=in_dim,
            in_dims=in_dims,
            out_dim=1,  # SDF
            config=config,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            symmetrize=symmetrize,
            n_harmonic_functions=n_harmonic_functions,
            embed_concat_pts=embed_concat_pts,
            sphere_init_radius=init_radius,
        )
        # the tet grid is sized relative to the sphere prior it has to enclose
        mesh_kwargs = dict(
            tet_res=tet_res,
            grid_scale=init_radius * grid_scale_rel,
            jitter_scale=jitter_scale,
        )
        if field is not None:
            field_kwargs.update({k: v for k, v in dict(field).items()})
        if mesh is not None:
            mesh_kwargs.update({k: v for k, v in dict(mesh).items()})

        self.feat2field = Feat2Field(**field_kwargs)
        self.field2mesh = Field2Mesh(**mesh_kwargs)

        if self.feat2field.out_dim != 1:
            raise ValueError(
                f"Feat2Mesh: the field must be a scalar SDF, "
                f"got out_dim={self.feat2field.out_dim}.",
            )

    def forward(self, frames_gt, frames_pred=None):
        frames_gt, frames_pred = self.feat2field.forward(
            frames_gt=frames_gt,
            frames_pred=frames_pred,
        )
        return self.field2mesh.forward(frames_gt=frames_gt, frames_pred=frames_pred)
