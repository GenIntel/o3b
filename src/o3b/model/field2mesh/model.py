import logging

logger = logging.getLogger(__name__)
from pathlib import Path

import numpy as np
import torch

from o3b.model.model import OD3D_Model, register_model


def _tets_path(tet_res: int) -> Path:
    """Tetrahedral grids shipped with the DMTet objects3d implementation."""
    import o3b.cv.geometry.objects3d.dmtet_x_gaussians as _pkg

    path = Path(_pkg.__file__).parent / f"{tet_res}_tets.npz"
    if not path.exists():
        available = sorted(
            int(p.name.split("_")[0]) for p in path.parent.glob("*_tets.npz")
        )
        raise FileNotFoundError(
            f"Field2Mesh: no tet grid for tet_res={tet_res} at {path}. "
            f"Available: {available}.",
        )
    return path


@register_model("Field2Mesh")
class Field2Mesh(OD3D_Model):
    """Extracts a triangle mesh from an implicit SDF field via DMTet.

    Reads ``frames_pred.field`` -- anything callable with query points (B, N, 3)
    returning (B, N, 1), such as the :class:`~o3b.model.feat2field.model.FeatField`
    written by :class:`~o3b.model.feat2field.model.Feat2Field` -- samples it on a
    fixed tetrahedral grid and runs differentiable marching tetrahedra
    (``DMTet_Core``, one object at a time, so the batch is looped over).

    Writes to ``frames_pred``:
      * ``meshes``   -- list of B :class:`o3b.data.datatypes.mesh.Mesh`
      * ``mesh``     -- the single mesh when B == 1 (``None`` otherwise)
      * ``mesh_sdf`` -- (B, T, 1) field values on the tet grid, e.g. for eikonal regularisation
    """

    def __init__(
        self,
        tet_res: int = 16,
        grid_scale: float = 2.2,
        jitter_scale: float = 0.05,
    ):
        super().__init__()

        from o3b.cv.geometry.objects3d.dmtet.dmtet_core import DMTet_Core

        self.marching_tets = DMTet_Core()
        self.grid_scale = grid_scale
        self.jitter_scale = jitter_scale

        tets = np.load(_tets_path(tet_res))
        # grid vertices come in (-0.5, 0.5), rescale to the object extent
        self.register_buffer(
            "tets_verts",
            torch.tensor(tets["vertices"], dtype=torch.get_default_dtype())
            * self.grid_scale,
            persistent=False,
        )
        self.register_buffer(
            "tets_faces",
            torch.tensor(tets["indices"], dtype=torch.long),
            persistent=False,
        )

    def get_tets_verts(self) -> torch.Tensor:
        """Tet-grid vertices, jittered while training so the mesh is not tied to
        the discrete grid (as in DMTet_x_Gaussians.get_uniform_jittered_tets_verts)."""
        pts = self.tets_verts
        if self.training and self.jitter_scale > 0:
            jitter = (
                (torch.rand(3, device=pts.device, dtype=pts.dtype) - 0.5)
                * self.grid_scale
                * self.jitter_scale
            )
            pts = pts + jitter
        return pts

    def forward(self, frames_gt, frames_pred=None):
        if frames_pred is None:
            frames_pred = frames_gt

        field = getattr(frames_pred, "field", None)
        if field is None:
            raise ValueError(
                "Field2Mesh: frames_pred.field is None, nothing to extract a mesh "
                "from (run Feat2Field first).",
            )
        out_dim = getattr(field, "out_dim", 1)
        if out_dim != 1:
            raise ValueError(
                f"Field2Mesh: the field must be a scalar SDF, got out_dim={out_dim}.",
            )

        B = getattr(field, "batch_size", None)
        if B is None:
            B = 1
        tets_verts = self.get_tets_verts()
        pts3d = tets_verts[None].expand(B, *tets_verts.shape)
        sdf = field(pts3d)  # (B, T, 1)

        # DMTet_Core allocates on its own `device` attribute, which nn.Module.to()
        # does not update (it only moves the registered tables).
        self.marching_tets.device = self.tets_faces.device

        from o3b.data.datatypes.mesh import Mesh

        meshes = []
        for b in range(B):
            verts, faces, _uvs, _uv_idx = self.marching_tets(
                pts3d[b],
                sdf[b],
                self.tets_faces,
            )
            if len(faces) == 0:
                logger.warning(
                    f"Field2Mesh: empty mesh for batch element {b} "
                    f"(field sign is constant over the tet grid).",
                )
            meshes.append(Mesh(verts=verts, faces=faces))

        frames_pred.meshes = meshes
        frames_pred.mesh = meshes[0] if B == 1 else None
        frames_pred.mesh_sdf = sdf
        return frames_gt, frames_pred
