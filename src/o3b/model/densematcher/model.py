from __future__ import annotations

import gc
import logging
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)

_CHECKPOINTS_URL = (
    "https://www.dropbox.com/scl/fo/ke8chm5zjostmlwj0x53p/"
    "AFZjfMgfxQAotaQ-f1sOZS8?rlkey=22yrp7m6bp3stlovslp73ryzs&dl=1"
)


def _download_checkpoints(checkpoints_dir: Path) -> None:
    """Download and extract the DenseMatcher checkpoint zip if not present."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger.info("DenseMatcher: downloading checkpoints from Dropbox → %s", checkpoints_dir)

    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            logger.info("DenseMatcher: download %d%%", pct)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp = Path(f.name)
    try:
        urllib.request.urlretrieve(_CHECKPOINTS_URL, tmp, reporthook=_reporthook)
        logger.info("DenseMatcher: extracting checkpoints…")
        with zipfile.ZipFile(tmp) as z:
            for member in z.infolist():
                if member.filename == "/":
                    continue
                target = checkpoints_dir / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())
        logger.info("DenseMatcher: checkpoints extracted to %s", checkpoints_dir)
    finally:
        tmp.unlink(missing_ok=True)


def _visualize_features(pt3d_mesh, mv_features, out_norm, cameras, viz_dir, *, object_id=None):
    """Render the mesh with its original texture and PCA-colored feature maps; save to viz_dir.

    Triggered by setting VIZ_DIR=/path/to/output before running inference.
    Produces three subdirectories per object:
        <viz_dir>/<object_id>/tex/ – original mesh texture
        <viz_dir>/<object_id>/mv/  – PCA of 768-dim SDDINO multiview features
        <viz_dir>/<object_id>/dm/  – PCA of 512-dim DiffusionNet (non-mv) features
    Each subdirectory contains one PNG per camera view.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    from o3b.model.densematcher.densematcher.render import batch_render
    from o3b.data.datatypes.object import _pca_vert_colors

    tag    = str(object_id) if object_id is not None else "object"
    device = mv_features.device

    def _render_and_save(mesh, label: str) -> None:
        renders, _, _, _, _ = batch_render(
            device, mesh, (3, 1), 384, 384, cameras=cameras, center=None
        )  # (N, H, W, 4)
        out_path = Path(viz_dir) / tag / label
        out_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(renders):
            img = frame[..., :3].clamp(0, 1).cpu().numpy()
            plt.imsave(str(out_path / f"view{i:02d}.png"), img)
        logger.info("DenseMatcher: saved %s renders → %s", label, out_path)

    def _pca_mesh(feats: "torch.Tensor"):
        from pytorch3d.structures.meshes import Meshes
        from pytorch3d.renderer.mesh.textures import Textures
        colors = _pca_vert_colors(feats.float().cpu()).to(device)
        return Meshes(
            verts=pt3d_mesh.verts_list(),
            faces=pt3d_mesh.faces_list(),
            textures=Textures(verts_rgb=[colors]),
        )

    with torch.no_grad():
        _render_and_save(pt3d_mesh,           "tex")
        _render_and_save(_pca_mesh(mv_features), "mv")
        _render_and_save(_pca_mesh(out_norm),    "dm")


@register_model("DenseMatcher")
class DenseMatcherModel(OD3D_Model):
    """Per-vertex feature extractor using the full DenseMatcher pipeline.

    Pipeline: SDDINO (2D multi-view) → DiffusionNet (3D geometry) → 512-dim
    normalised per-vertex features.

    forward(ObjectBatch) -> ObjectBatch with verts3d_feats (B, V, width) set.
    Requires the mesh to be present in the ObjectBatch.

    mesh_weld: textured mc* meshes carry xatlas UV-seam vertex duplicates that
    cut the surface into charts, which the DiffusionNet Laplacian sees as
    boundary cuts. If true, the mesh is welded back to watertight
    connectivity (housecorr3dv2.method.mesh_match_utils.weld_mesh, mirroring
    FunctionalMapsMethod) before feature extraction: per-vertex colour is
    resolved (sampled from the texture atlas if needed) and averaged over
    welded duplicates (mean_by_index), the whole pipeline then runs on the
    welded mesh, and the output features are broadcast back to full
    resolution via the weld index map.

    mesh_flip: if true, face winding is flipped before extraction — useful to
    correct/ablate vertex-normal orientation (and thus DiffusionNet operator
    sign) when a mesh's winding convention doesn't match what this pipeline
    expects.
    """

    def __init__(
        self,
        pretrained_upsampler_path: str,
        aggre_net_weights_folder: str,
        diffusionnet_ckpt_path: str = "",
        num_views_azimuth: int = 3,
        num_views_elevation: int = 1,
        width: int = 512,
        num_blocks: int = 8,
        use_mv_features: bool = False,
        freeze: bool = True,
        mesh_weld: bool = False,
        mesh_flip: bool = False,
    ):
        super().__init__()
        self.pretrained_upsampler_path = pretrained_upsampler_path
        self.aggre_net_weights_folder = aggre_net_weights_folder
        self.diffusionnet_ckpt_path = diffusionnet_ckpt_path
        self.num_views_azimuth = num_views_azimuth
        self.num_views_elevation = num_views_elevation
        self.width = width
        self.num_blocks = num_blocks
        self.use_mv_features = use_mv_features
        self.freeze = freeze
        self.mesh_weld = mesh_weld
        self.mesh_flip = mesh_flip
        # out_dim: 768 (raw SDDINO mv_features) or width (DiffusionNet output)
        self.out_dim = 768 if use_mv_features else width
        self._mesh_featurizer = None

    def _ensure_featurizer(self) -> None:
        if self._mesh_featurizer is not None:
            return
        from o3b.model.xformers_compat import patch_xformers_memory_efficient_attention
        patch_xformers_memory_efficient_attention()
        # Auto-download checkpoints if the FeatUp checkpoint is missing.
        upsampler = Path(self.pretrained_upsampler_path)
        if not upsampler.exists():
            checkpoints_dir = Path(upsampler.parts[0])  # e.g. "checkpoints"
            _download_checkpoints(checkpoints_dir)
        from o3b.model.densematcher.densematcher.meshfeaturizer import MeshFeaturizer
        logger.info("DenseMatcher: loading MeshFeaturizer…")
        featurizer = MeshFeaturizer(
            pretrained_upsampler_path=self.pretrained_upsampler_path,
            num_views=(self.num_views_azimuth, self.num_views_elevation),
            num_blocks=self.num_blocks,
            width=self.width,
            aggre_net_weights_folder=self.aggre_net_weights_folder,
        )
        if self.diffusionnet_ckpt_path:
            logger.info(f"DenseMatcher: loading DiffusionNet weights from {self.diffusionnet_ckpt_path}")
            ckpt = torch.load(self.diffusionnet_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = {
                k.removeprefix("model.extractor_3d."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.extractor_3d")
            }
            featurizer.extractor_3d.load_state_dict(state_dict)
        featurizer.to("cuda").half()
        featurizer.extractor_3d.float()  # sparse mm (Laplacian) doesn't support fp16
        featurizer.extractor_2d.featurizer.mem_eff = True
        featurizer.eval()
        self._mesh_featurizer = featurizer

    def _forward_object_batch(self, batch):
        from dataclasses import replace
        from o3b.model.densematcher.densematcher import diffusion_net

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_featurizer()
        featurizer = self._mesh_featurizer

        verts = mesh.verts.float()  # (V, 3)
        faces = mesh.faces.long()   # (F, 3)
        V = verts.shape[0]          # full (pre-weld) vertex count — output resolution

        if self.mesh_flip:
            faces = faces[:, [0, 2, 1]]

        weld_inv = None
        if self.mesh_weld:
            from o3b.data.datatypes.mesh import mean_by_index, weld_mesh

            verts_np, faces_np = verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
            verts_w, faces_w, weld_inv, _ = weld_mesh(verts_np, faces_np)

            vert_colors = None
            if mesh.vert_colors is not None:
                vert_colors = mean_by_index(
                    mesh.vert_colors.detach().cpu().float().numpy(), weld_inv, len(verts_w)
                )
            elif mesh.texture is not None and mesh.verts_uvs is not None:
                # sample per-vertex colour from the texture atlas, then
                # weld-average over seam duplicates
                uvs = mesh.verts_uvs.detach().cpu().float()
                grid = (uvs * 2 - 1).view(1, 1, -1, 2)
                sampled = torch.nn.functional.grid_sample(
                    mesh.texture.detach().cpu().unsqueeze(0).float(), grid,
                    mode="bilinear", align_corners=True, padding_mode="border",
                )  # (1, 3, 1, V)
                vert_rgb = sampled[0, :, 0, :].T.numpy()
                vert_colors = mean_by_index(vert_rgb, weld_inv, len(verts_w))

            from o3b.data.datatypes.mesh import Mesh
            mesh = Mesh(
                verts=torch.from_numpy(verts_w).float(),
                faces=torch.from_numpy(faces_w).long(),
                vert_colors=torch.from_numpy(vert_colors).float() if vert_colors is not None else None,
            )
            verts, faces = mesh.verts, mesh.faces

        # Normalise to [-0.15, 0.15] (DenseMatcher expects scale ≈ 0.3 on largest axis)
        center = verts.mean(dim=0)
        verts_c = verts - center
        scale = verts_c.abs().max()
        if scale > 1e-8:
            verts_c = verts_c * (0.15 / scale)

        # Build PyTorch3D Meshes, preserving any texture from the source mesh.
        mesh_norm = replace(mesh, verts=verts_c)
        pt3d_mesh = mesh_norm.to_pytorch3d(device=device)

        from o3b.model.densematcher.densematcher.utils import get_uniform_SO3_RT

        # Compute DiffusionNet geometric operators (with vertex normals; sparse → dense)
        normals = pt3d_mesh.verts_normals_list()[0]  # (V, 3)
        operators = diffusion_net.geometry.get_operators(
            verts_c.to(device), faces.to(device),
            k_eig=128, op_cache_dir=None,
            normals=normals,
        )
        frames, mass, L, evals, evecs, gradX, gradY = operators
        operators = (frames, mass, L.to_dense(), evals, evecs, gradX.to_dense(), gradY.to_dense())

        # Compute camera extrinsics around the normalised mesh (mirrors get_mesh)
        bb = pt3d_mesh.get_bounding_boxes()   # [1, 3, 2]
        cam_dist = bb.abs().max() * 2.5       # fixed factor for inference
        Rs, ts, _, _ = get_uniform_SO3_RT(
            num_azimuth=self.num_views_azimuth,
            num_elevation=self.num_views_elevation,
            distance=cam_dist,
            center=bb.mean(2),
            device=device,
        )
        cameras = [Rs, ts]

        with torch.autocast("cuda"):
            with torch.no_grad():
                out_norm, _, mv_features = featurizer(
                    pt3d_mesh, pt3d_mesh, operators, cameras=cameras,
                    return_mvfeatures=True,
                )

        viz_dir = os.environ.get("VIZ_DIR", None)
        if viz_dir is not None:
            _visualize_features(
                pt3d_mesh, mv_features, out_norm, cameras, viz_dir,
                object_id=getattr(batch, "object_id", None),
            )

        del pt3d_mesh
        torch.cuda.empty_cache()
        gc.collect()

        raw = mv_features if self.use_mv_features else out_norm
        feats = torch.nan_to_num(raw.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if weld_inv is not None:
            feats = feats[torch.from_numpy(weld_inv).long()]  # welded (Nw) -> full (V)
        B = batch.verts3d.shape[0] if batch.verts3d is not None else 1
        batch.verts3d_feats = feats.unsqueeze(0).expand(B, V, -1).contiguous()
        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch
        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)
        return frames_gt, frames_pred
