import logging

from o3b.cv.select import cumsum_w0, cumsum_w0_and_wEnd

logger = logging.getLogger(__name__)
from enum import Enum
from pathlib import Path
from typing import List

try:
    import nvdiffrast.torch as dr
except ModuleNotFoundError:
    dr = None

import torch
from o3b.cv.geometry.transform import proj3d2d_broadcast

from omegaconf import DictConfig
import inspect
from o3b.cv.geometry.objects3d.objects3d import OD3D_Objects3D_Deform

from o3b.data.batch_datatypes import OD3D_ModelData
from o3b.data.ext_enum import StrEnum

from dataclasses import dataclass
from o3b.cv.geometry.transform import (
    tform4x4,
    tform4x4_broadcast,
    inv_tform4x4,
    add_homog_dim,
    transf3d_broadcast,
)

from o3b.cv.visual.sample import sample_pxl2d_grid
from o3b.cv.geometry.grid import get_pxl2d
from typing import Union
import open3d as o3d
import numpy as np
from o3b.cv.geometry.objects3d.objects3d import (
    PROJECT_MODALITIES,
    OD3D_Objects3D,
    FEATS_DISTR,
    FEATS_ACTIVATION,
)

from typing import Optional

class FACE_BLEND_TYPE(str, Enum):
    HARD = "hard"
    SOFT_SIGMOID_NORMALIZED = "soft_sigmoid_normalized"
    SOFT_GAUSSIAN_CUMULATIVE = "soft_gaussian_cumulative"


class RASTERIZER(str, Enum):
    PYTORCH3D = "pytorch3d"
    NVDIFFRAST = "nvdiffrast"


class MESH_RENDER_MODALITIES(str, Enum):
    DEPTH = "depth"
    MASK = "mask"
    RGB = "rgb"
    RGBA = "rgba"
    FEATS = "feats"
    MASK_VERTS_VSBL = "mask_verts_vsbl"
    VERTS_NCDS = "verts_ncds"
    VERTS_ONEHOT = "verts_onehot"


class VERT_MODALITIES(StrEnum):
    OBJ_ID = "obj_id"
    OBJ_ONEHOT = "obj_onehot"
    OBJ_ONEHOT_RGB = "obj_onehot_rgb"
    OBJ_IN_SCENE_ID = "obj_in_scene_id"
    OBJ_IN_SCENE_ONEHOT = "obj_in_scene_onehot"
    IN_OBJ_ID = "in_obj_id"
    IN_OBJ_COARSE_ID = "in_obj_coarse_id"
    IN_SCENE_ID = "in_scene_id"
    IN_SCENE_COARSE_ID = "in_scene_coarse_id"
    IN_SCENES_ID = "in_scenes_id"
    IN_SCENES_COARSE_ID = "in_scenes_coarse_id"
    PT3D = "pt3d"
    MASK = "mask"
    UV_PXL2D = "uv_pxl2d"
    RGB = "rgb"
    RGBA = "rgba"
    FEAT = "feat"
    NOCS = "nocs"
    CANOCS = "canocs"
    PT3D_NCDS = "pt3d_ncds"
    PT3D_NCDS_AVG = "pt3d_ncds_avg"
    IN_OBJ_ONEHOT = "in_obj_onehot"
    IN_OBJ_COARSE_ONEHOT = "in_obj_coarse_onehot"
    IN_OBJ_COARSE_ONEHOT_RGB = "in_obj_coarse_onehot_rgb"
    IN_SCENE_ONEHOT = "in_scene_onehot"
    IN_SCENE_COARSE_ONEHOT = "in_scene_coarse_onehot"


class FACE_MODALITIES(StrEnum):
    IN_OBJ_ID = "in_obj_id"
    IN_SCENE_ID = "in_scene_id"
    VERTS_IN_OBJ_ID = "verts_in_obj_id"
    VERTS_IN_SCENE_ID = "verts_in_scene_id"
    VERTS_IN_SCENES_ID = "verts_in_scenes_id"
    VERTS_UVS_IN_OBJ_ID = "verts_uvs_in_obj_id"
    VERTS_UVS_IN_SCENE_ID = "verts_uvs_in_scene_id"
    VERTS_UVS_IN_SCENES_ID = "verts_uvs_in_scenes_id"


class MESH_RENDER_MODALITIES_GAUSSIAN_SPLAT(str, Enum):
    RGB = MESH_RENDER_MODALITIES.RGB
    FEATS = MESH_RENDER_MODALITIES.FEATS
    VERTS_NCDS = MESH_RENDER_MODALITIES.VERTS_NCDS


class OD3D_Meshes_Deform(OD3D_Objects3D_Deform):
    def __init__(
        self,
        verts_deform: torch.Tensor,
        latent: torch.Tensor = None,
        latent_mu: torch.Tensor = None,
        latent_logvar: torch.Tensor = None,
    ):
        super().__init__()
        self.verts_deform = verts_deform
        self.device = self.verts_deform.device
        self.length = self.verts_deform.shape[0]
        self.latent = latent
        self.latent_logvar = latent_logvar
        self.latent_mu = latent_mu

    def to(self, *args, **kwargs):
        if self.latent is not None:
            self.latent.to(*args, **kwargs)
        if self.latent_mu is not None:
            self.latent_mu.to(*args, **kwargs)
        if self.latent_logvar is not None:
            self.latent_logvar.to(*args, **kwargs)
        self.verts_deform.to(*args, **kwargs)
        self.device = self.verts_deform.device


    def __getitem__(self, i):
        if isinstance(i, slice):
            verts_deform = self.verts_deform[i]
            latent = self.latent[i] if self.latent is not None else None
            latent_mu = self.latent_mu[i] if self.latent_mu is not None else None
            latent_logvar = (
                self.latent_logvar[i] if self.latent_logvar is not None else None
            )
        else:
            if isinstance(i, list):
                verts_deform = self.verts_deform[[k for k in i]]
                latent = self.latent[[k for k in i]] if self.latent is not None else None
                latent_mu = (
                    self.latent_mu[[k for k in i]] if self.latent_mu is not None else None
                )
                latent_logvar = (
                    self.latent_logvar[[k for k in i]]
                    if self.latent_logvar is not None
                    else None
                )
            else:
                verts_deform = self.verts_deform[i : i + 1]
                latent = self.latent[i : i + 1] if self.latent is not None else None
                latent_mu = (
                    self.latent_mu[i : i + 1] if self.latent_mu is not None else None
                )
                latent_logvar = (
                    self.latent_logvar[i : i + 1]
                    if self.latent_logvar is not None
                    else None
                )

        return OD3D_Meshes_Deform(
            verts_deform=verts_deform,
            latent=latent,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
        )

    def __add__(self, meshes_deform2):

        verts_deform = torch.cat([self.verts_deform, meshes_deform2.verts_deform], dim=0)

        latent = torch.cat([self.latent, meshes_deform2.latent], dim=0) if self.latent is not None else None
        latent_mu = torch.cat([self.latent_mu, meshes_deform2.latent_mu], dim=0)  if self.latent_mu is not None else None
        latent_logvar = torch.cat([self.latent_mu, meshes_deform2.latent_mu], dim=0) if self.latent_logvar is not None else None

        return OD3D_Meshes_Deform(
            verts_deform=verts_deform,
            latent=latent,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
        )


    def get_first_item_repeated(self):
        verts_deform = self.verts_deform[0:1].expand(*self.verts_deform.shape)
        latent = (
            self.latent[0:1].expand(*self.verts_deform.shape)
            if self.latent is not None
            else None
        )
        latent_mu = (
            self.latent_mu[0:1].expand(*self.verts_deform.shape)
            if self.latent_mu is not None
            else None
        )
        latent_logvar = (
            self.latent_logvar[0:1].expand(*self.verts_deform.shape)
            if self.latent_logvar is not None
            else None
        )

        return OD3D_Meshes_Deform(
            verts_deform=verts_deform,
            latent=latent,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
        )

    def __len__(self):
        return self.length

    def broadcast_with_cams(self, cams_count: int):
        verts_deform = (
            self.verts_deform[:, None]
            .expand(-1, cams_count, -1, -1)
            .reshape(-1, *self.verts_deform.shape[1:])
        )
        latent = (
            self.latent[:, None]
            .expand(-1, cams_count, -1)
            .reshape(-1, *self.latent.shape[1:])
            if self.latent is not None
            else None
        )
        latent_mu = (
            self.latent_mu[:, None]
            .expand(-1, cams_count, -1)
            .reshape(-1, *self.latent_mu.shape[1:])
            if self.latent_mu is not None
            else None
        )
        latent_logvar = (
            self.latent_logvar[:, None]
            .expand(-1, cams_count, -1)
            .reshape(-1, *self.latent_logvar.shape[1:])
            if self.latent_logvar is not None
            else None
        )

        return OD3D_Meshes_Deform(
            verts_deform=verts_deform,
            latent=latent,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
        )

    @classmethod
    def from_net_output(cls, net_output: OD3D_ModelData, affine=False):
        """
        Args:
            net_output (OD3D_ModelData): feat: BxF, feats: BxNxF, featmap: BxFxHxW
        """
        B = net_output.feat.shape[0]
        if not affine:
            return cls(
                verts_deform=net_output.feat.reshape(B, -1, 3),
                latent=net_output.latent,
                latent_mu=net_output.latent_mu,
                latent_logvar=net_output.latent_logvar,
            )
        else:
            return cls(
                verts_deform=net_output.feat.reshape(B, -1, 6),
                latent=net_output.latent,
                latent_mu=net_output.latent_mu,
                latent_logvar=net_output.latent_logvar,
            )

    def get_out_dim(self):
        raise NotImplementedError


class Meshes(OD3D_Objects3D):
    instance_deform_class = OD3D_Meshes_Deform
    feats_objects: Optional[torch.Tensor]
    verts_uvs: Optional[torch.Tensor]
    rgbs_uvs: Optional[torch.Tensor]

    def __init__(
        self,
        verts: Union[List[torch.Tensor], torch.Tensor],
        faces: Union[List[torch.Tensor], torch.Tensor],
        feat_dim=128,
        objects_count=0,
        feats_objects: Union[bool, torch.Tensor] = False,
        verts_uvs: Union[List[torch.Tensor], torch.Tensor] = None,
        faces_uvs: Union[List[torch.Tensor], torch.Tensor] = None,
        rgbs_uvs: Union[List[torch.Tensor], torch.Tensor] = None,
        feats_requires_grad=True,
        feats_objects_requires_param=True,
        feat_clutter_requires_param=True,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        feats_activation=FEATS_ACTIVATION.NORM_DETACH,
        rgb: Union[List[torch.Tensor], torch.Tensor] = None,
        verts_requires_grad=False,
        verts_requires_param=True,
        verts_coarse_count: int = 150,
        verts_coarse_prob_sigma: float = 0.01,  # 0.001, 0.01, 0.05, 0.1
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
        rasterizer=RASTERIZER.NVDIFFRAST,
        face_blend_type=FACE_BLEND_TYPE.HARD,
        face_blend_count=8,
        face_opacity=1.0,
        face_opacity_face_sdf_sigma=1e-2,
        face_opacity_face_sdf_gamma=1e-4,
        scale_3D_params: float = 1e2,
        instance_deform_net_config: DictConfig = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        self.feats_objects_requires_param = feats_objects_requires_param
        self.verts_requires_param = verts_requires_param

        if verts is not None and not isinstance(verts, List):
            verts = [verts]
        if rgb is not None and not isinstance(rgb, List):
            rgb = [rgb]
        if faces is not None and not isinstance(faces, List):
            faces = [faces]
        if verts_uvs is not None and not isinstance(verts_uvs, List):
            verts_uvs = [verts_uvs]
        if faces_uvs is not None and not isinstance(faces_uvs, List):
            faces_uvs = [faces_uvs]
        if rgbs_uvs is not None and not isinstance(rgbs_uvs, List):
            rgbs_uvs = [rgbs_uvs]

        self.verts_counts = [_verts.shape[0] for _verts in verts]
        self.verts_counts_max = max(self.verts_counts)

        self.rasterizer = rasterizer  # RASTERIZER.NVDIFFRAST # PYTORCH3D or NVDIFFRAST
        if instance_deform_net_config is not None:
            affine = instance_deform_net_config.get("affine", False)
            if instance_deform_net_config.head.get("class_name", "MLP") == "CoordMLP":
                if not affine:
                    instance_deform_net_config.head.update({"out_dim": 3})
                else:
                    instance_deform_net_config.head.update({"out_dim": 6})
            else:
                if not affine:
                    instance_deform_net_config.head.update(
                        {"out_dim": 3 * self.verts_counts_max},
                    )
                else:
                    instance_deform_net_config.head.update(
                        {"out_dim": 6 * self.verts_counts_max},
                    )

        super().__init__(
            feat_dim=feat_dim,
            feat_clutter=feat_clutter,
            feat_clutter_requires_param=feat_clutter_requires_param,
            feats_requires_grad=feats_requires_grad,
            feats_distribution=feats_distribution,
            feats_activation=feats_activation,
            verts_requires_grad=verts_requires_grad,
            device=device,
            dtype=dtype,
            scale_3D_params=scale_3D_params,
            instance_deform_net_config=instance_deform_net_config,
        )

        self.face_opacity = face_opacity
        self.face_opacity_face_sdf_sigma = face_opacity_face_sdf_sigma
        self.face_opacity_face_sdf_gamma = face_opacity_face_sdf_gamma
        self.face_blend_type = face_blend_type
        self.face_blend_count = face_blend_count

        self.meshes_count = len(verts)

        if self.verts_requires_param:
            self._verts = torch.nn.Parameter(
                torch.cat(
                    [_verts / float(scale_3D_params) for _verts in verts],
                    dim=0,
                ).to(**factory_kwargs),
                requires_grad=self.verts_requires_grad,
            )
        else:
            self._verts = torch.cat([_verts for _verts in verts], dim=0).to(
                **factory_kwargs,
            )

        self.faces = torch.cat([_faces for _faces in faces], dim=0).to(**factory_kwargs)
        self.device = self._verts.device

        self.gaussian_splat_enabled = gaussian_splat_enabled
        self.gaussian_splat_opacity = gaussian_splat_opacity
        self.gaussian_splat_pts3d_size_rel_to_neighbor_dist = (
            gaussian_splat_pts3d_size_rel_to_neighbor_dist
        )
        self.pt3d_raster_perspective_correct = pt3d_raster_perspective_correct

        self.geodesic_prob_sigma = geodesic_prob_sigma
        self._geodesic_dist = None
        self._verts_coarse = None
        self._verts_ids_coarse = None
        self.verts_coarse_count = verts_coarse_count
        self.verts_coarse_prob_sigma = verts_coarse_prob_sigma
        self._verts_label_coarse = None

        self.faces_counts = [_faces.shape[0] for _faces in faces]
        self.verts_counts_acc_from_0 = [0] + [
            sum(self.verts_counts[: i + 1]) for i in range(self.meshes_count)
        ]
        self.faces_counts_acc_from_0 = [0] + [
            sum(self.faces_counts[: i + 1]) for i in range(self.meshes_count)
        ]

        self.verts_count = self.verts_counts_acc_from_0[-1]
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(
            size=[len(self), self.verts_counts_max],
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i] :] = False

        if verts_uvs is not None and len(verts_uvs) > 0:
            self.verts_uvs = torch.nn.Parameter(
                torch.cat([_verts_uvs for _verts_uvs in verts_uvs], dim=0).to(
                    **factory_kwargs,
                ),
                requires_grad=False,
            )

            self.verts_uvs_counts = [_verts_uvs.shape[0] for _verts_uvs in verts_uvs]
            self.verts_uvs_counts_max = max(self.verts_uvs_counts)
            self.verts_uvs_counts_acc_from_0 = [0] + [
                sum(self.verts_uvs_counts[: i + 1]) for i in range(self.meshes_count)
            ]
        else:
            self.register_parameter("verts_uvs", None)

        if faces_uvs is not None and len(faces_uvs) > 0:
            self.faces_uvs = torch.nn.Parameter(
                torch.cat([_faces_uvs for _faces_uvs in faces_uvs], dim=0).to(
                    **factory_kwargs,
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("faces_uvs", None)

        if rgbs_uvs is not None and len(rgbs_uvs) > 0:
            self.rgbs_uvs = torch.nn.Parameter(
                torch.stack([_rgbs_uvs for _rgbs_uvs in rgbs_uvs], dim=0).to(
                    **factory_kwargs,
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("rgbs_uvs", None)

        if rgb is not None and len(rgb) > 0:
            self.rgb = torch.nn.Parameter(
                torch.cat([_rgb for _rgb in rgb], dim=0),
                requires_grad=False,
            )
        else:
            self.register_parameter("rgb", None)

        if self.feats_objects_requires_param:
            if isinstance(feats_objects, torch.Tensor) or feats_objects:
                self._feats_objects = torch.nn.Parameter(
                    torch.cat(
                        [
                            torch.empty(
                                size=[self.verts_counts[i], self.feat_dim],
                                **factory_kwargs,
                            )
                            for i in range(len(self))
                        ],
                        dim=0,
                    ),
                    requires_grad=feats_requires_grad,
                )
            else:
                self.register_parameter("_feats_objects", None)
        else:
            if isinstance(feats_objects, torch.Tensor) or feats_objects:
                self._feats_objects = torch.cat(
                    [_feats_objects for _feats_objects in feats_objects],
                    dim=0,
                ).to(**factory_kwargs)
            else:
                self._feats_objects = None

        self.pre_rendered_feats = None
        self.pre_rendered_modalities = {}

        self.reset_parameters(feats_objects=feats_objects, feat_clutter=feat_clutter)

        import matplotlib.pyplot as plt

        color_ = plt.get_cmap("tab20", len(self))
        self.feats_rgb_object_id = []
        for i in range(len(self)):
            self.feats_rgb_object_id.extend([color_(i)] * self.verts_counts[i])

    def set_verts_requires_grad(self, verts_requires_grad):
        self.verts_requires_grad = verts_requires_grad
        self._verts.requires_grad = verts_requires_grad

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if not self.feats_objects_requires_param:
            self._feats_objects = self._feats_objects.to(*args, **kwargs)
        if not self.verts_requires_param:
            self._verts = self._verts.to(*args, **kwargs)
        self.faces = self.faces.to(*args, **kwargs)
        self.device = self.faces.device

    def detach(self):
        return Meshes.from_o3d(self.to_o3ds())

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        if not self.feats_objects_requires_param:
            self._feats_objects = self._feats_objects.cuda(*args, **kwargs)
        if not self.verts_requires_param:
            self._verts = self._verts.cuda(*args, **kwargs)
        self.faces = self.faces.cuda(*args, **kwargs)
        self.device = self.faces.device

    @property
    def verts(self):
        if self.verts_requires_param:
            return self._verts * self.scale_3D_params
        else:
            return self._verts

    @verts.setter
    def verts(self, value):
        if self.verts_requires_param:
            with torch.no_grad():
                _ = self._verts.copy_(value / self.scale_3D_params)
        else:
            self._verts = value

    def set_verts_with_mesh_id(self, value, mesh_id):
        if self.verts_requires_param:
            with torch.no_grad():
                _ = self._verts[
                    self.verts_counts_acc_from_0[
                        mesh_id
                    ] : self.verts_counts_acc_from_0[mesh_id + 1]
                ].copy_(value.to(device=self._verts.device) / self.scale_3D_params)
        else:
            self._verts[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ] = value.to(device=self._verts.device)

    def get_geo_smooth_loss(
        self,
        objects_ids,
        instance_deform=None,
        detach_objects_verts=False,
    ):
        from pytorch3d.loss import mesh_laplacian_smoothing

        loss_smooth = mesh_laplacian_smoothing(
            self.get_pt3dmeshes_with_deform(
                objects_ids=objects_ids,
                instance_deform=instance_deform,
                detach_objects_verts=detach_objects_verts,
            ),
        )
        return loss_smooth

    def get_geo_sdf_reg_loss(self, objects_ids):
        regs_losses = []
        for object_id in objects_ids:
            regs_losses.append(torch.Tensor([0]).to(device=self.device))
        regs_losses = torch.stack(regs_losses)
        return regs_losses

    def get_geo_deform_smooth_loss(self, objects_ids, instance_deform):
        edges_packed = self.get_edges_cat_with_mesh_ids(
            mesh_ids=objects_ids,
            use_verts_offset=True,
            use_global_verts_ids=False,
        )

        # edges_packed = self.get_pt3dmeshes_with_deform(objects_ids=objects_ids,
        #                                               instance_deform=instance_deform).edges_packed()

        verts_canonical = self.get_verts_cat_with_mesh_ids(
            mesh_ids=objects_ids,
            detach_objects_verts=True,
        )
        verts_deform = self.get_verts_deform_cat_with_mesh_ids(
            mesh_ids=objects_ids,
            instance_deform=instance_deform,
        )

        loss = (
            (verts_deform[edges_packed[:, 0]] - verts_deform[edges_packed[:, 1]]).norm(
                dim=-1,
            )
            / (
                (
                    verts_canonical[edges_packed[:, 0]]
                    - verts_canonical[edges_packed[:, 1]]
                ).norm(dim=-1)
                + 1e-10
            )
        ).mean()
        return loss

    def show_texture_map(self):
        from o3b.cv.visual.draw import draw_lines, draw_pixels
        from o3b.cv.visual.show import show_imgs

        size = 1000
        H = size
        W = size
        B = len(self)
        texture_map = torch.zeros((B, 3, H, W)) * 0.0
        for b in range(B):
            verts_uvs = self.get_verts_uvs_with_mesh_id(mesh_id=b, clone=True)
            faces = self.get_faces_with_mesh_id(mesh_id=b, clone=True)
            texture_map[b] = draw_pixels(
                texture_map[b],
                pxls=verts_uvs * (size - 1),
                radius_in=0,
                radius_out=1,
            )  # colors=(255, 255, 255)
            lines1 = torch.cat(
                [verts_uvs[faces[:, 0][:, None]], verts_uvs[faces[:, 1][:, None]]],
                dim=1,
            )
            lines2 = torch.cat(
                [verts_uvs[faces[:, 1][:, None]], verts_uvs[faces[:, 2][:, None]]],
                dim=1,
            )
            lines3 = torch.cat(
                [verts_uvs[faces[:, 2][:, None]], verts_uvs[faces[:, 0][:, None]]],
                dim=1,
            )
            lines = torch.cat([lines1, lines2, lines3], dim=0)
            texture_map[b] = draw_lines(
                texture_map[b],
                lines=lines * (size - 1),
                thickness=1,
            )  # , colors=(255, 255, 255)

        show_imgs(texture_map)

        # texture_map = texture_map.permute(0, 2, 3, 1)
        # pt3dtextures = TexturesUV(
        #     maps=texture_map, faces_uvs=pt3dmesh.faces_padded(), verts_uvs=verts_uvs[None,]
        # )
        # import matplotlib.pyplot as plt
        # import matplotlib
        #
        # matplotlib.use("TkAgg")
        #
        # # plt.figure(figsize=(7, 7))
        # # texture_image = pt3dtextures.clone().maps_padded()
        # # plt.imshow(texture_image.squeeze().cpu().numpy())
        # # plt.grid("off")
        # # plt.axis("off")
        # # plt.show()
        #
        # from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
        #
        # plt.figure(figsize=(7, 7))
        # texturesuv_image_matplotlib(pt3dtextures, subsample=None)
        # plt.grid("off")
        # plt.axis("off")
        # plt.show()

    @staticmethod
    def create_sphere(verts_count=1000, radius=1.0, device="cpu", ico=False):
        from o3b.cv.geometry.objects3d.meshes import Meshes

        if not ico:
            # verts_count = 2 * resolution * (resolution-1) + 2
            # (verts_count - 2) / 2  = resolution **2 - resolution
            # (verts_count - 2) / 2  <= resolution **2
            import math

            resolution = int(math.floor(math.sqrt((verts_count - 2.0) / 2.0)))
            o3d_mesh = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius,
                resolution=resolution,
                create_uv_map=True,
            )
            return Meshes.from_o3d(o3d_mesh, device=device)
        else:
            from pytorch3d.utils import ico_sphere

            _verts_count = 0
            mesh = None
            level = -1
            while _verts_count < verts_count:
                level += 1
                mesh = ico_sphere(level)
                _verts_count = mesh.verts_list()[0].shape[0]

            level -= 1
            mesh = ico_sphere(level)
            _verts_count = mesh.verts_list()[0].shape[0]

            mesh_verts_list = mesh.verts_list()
            if isinstance(radius, torch.Tensor):
                mesh_verts_list[0] *= radius.item()
            else:
                mesh_verts_list[0] *= radius
            meshes = Meshes(
                verts=mesh_verts_list,
                faces=mesh.faces_list(),
                device=device,
            )
            return meshes

            # from o3b.cv.visual.show import show_scene
            # show_scene(meshes=meshes)

    @staticmethod
    def from_trimesh(
        mesh_trimesh,
        device="cpu",
        load_texts=True,
        **kwargs,
    ):
        # mesh_trimesh.show()
        vertices = []
        faces = []
        verts_uvs = []
        faces_uvs = []
        rgbs_uvs = []
        rgbs = []
        vertices.append(
            torch.from_numpy(mesh_trimesh.vertices).to(
                dtype=torch.float,
                device=device,
            ),
        )
        faces.append(
            torch.from_numpy(mesh_trimesh.faces).to(dtype=torch.long, device=device),
        )

        if load_texts:
            #             if hasattr(mesh_trimesh, 'visual') and hasattr(mesh_trimesh.visual, 'vertex_colors'):
            #                 rgbs.append(
            #                     torch.from_numpy(mesh_trimesh.visual.vertex_colors[:, :3] / 255.).to(torch.float))
            # trimesh.visual.material.SimpleMaterial  visual.material.main_color
            if hasattr(mesh_trimesh, "visual") and hasattr(
                mesh_trimesh.visual,
                "vertex_colors",
            ):
                rgbs.append(
                    torch.from_numpy(
                        mesh_trimesh.visual.vertex_colors[:, :3] / 255.0,
                    ).to(dtype=torch.float, device=device),
                )
            elif hasattr(mesh_trimesh, "visual") and hasattr(mesh_trimesh.visual, "uv"):
                # rgbs.append(torch.from_numpy(mesh_trimesh.visual.to_color().vertex_colors[:, :3] / 255.).to(torch.float))

                # logger.info(len(mesh_trimesh.visual.uv), len(vertices[-1]))
                if mesh_trimesh.visual.uv is not None:
                    verts_uvs.append(
                        torch.from_numpy(mesh_trimesh.visual.uv).to(torch.float),
                    )
                    verts_uvs[-1][:, 1] = 1.0 - verts_uvs[-1][:, 1]
                    faces_uvs.append(faces[-1])

                    from torchvision import transforms

                    transform = transforms.ToTensor()

                    if (
                        hasattr(mesh_trimesh.visual.material, "image")
                        and mesh_trimesh.visual.material.image is not None
                    ):
                        tensor_image = transform(mesh_trimesh.visual.material.image)
                        rgbs_uvs.append(tensor_image.to(device=device))
                    elif (
                        hasattr(mesh_trimesh.visual.material, "baseColorTexture")
                        and mesh_trimesh.visual.material.baseColorTexture is not None
                    ):
                        tensor_image = transform(
                            mesh_trimesh.visual.material.baseColorTexture,
                        )
                        rgbs_uvs.append(tensor_image.to(device=device))
                    elif (
                        hasattr(mesh_trimesh.visual.material, "main_color")
                        and mesh_trimesh.visual.material.main_color is not None
                    ):
                        main_color = torch.from_numpy(
                            mesh_trimesh.visual.material.main_color[:3] / 255.0,
                        ).to(
                            dtype=torch.float,
                        )
                        rgbs_uvs.append(
                            (
                                main_color[:, None, None]
                                * torch.ones(size=(3, 500, 500))
                            ).to(device=device),
                        )
                    else:
                        logger.info(
                            "could not load texture due to missing visual material in trimesh",
                        )
                else:
                    logger.info(
                        "could not load texture due to missing visual material in trimesh",
                    )
                    logger.info("creating uv map")
                    # import xatlas
                    # # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
                    # # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
                    # # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
                    # vmapping, uv_idx, uvs = xatlas.parametrize(mesh_trimesh.vertices, mesh_trimesh.faces)
                    # uvs = torch.from_numpy(uvs).to(dtype=torch.float, device=device)
                    # uv_idx = torch.from_numpy(uv_idx).to(dtype=torch.long, device=device)

                    uv_idx = faces[-1]
                    # uvs = 0.5 * torch.ones((int(uv_idx.max()), 2)).to(
                    #     dtype=torch.float,
                    #     device=device,
                    # )
                    uvs = 0.5 * torch.ones((vertices[-1].shape[0], 2)).to(
                        dtype=torch.float,
                        device=device,
                    )


                    verts_uvs.append(uvs)
                    verts_uvs[-1][:, 1] = 1.0 - verts_uvs[-1][:, 1]
                    faces_uvs.append(uv_idx)

                    main_color = torch.from_numpy(
                        mesh_trimesh.visual.material.main_color[:3] / 255.0,
                    ).to(dtype=torch.float)
                    rgbs_uvs.append(
                        (main_color[:, None, None] * torch.ones(size=(3, 500, 500))).to(
                            device=device,
                        ),
                    )

                # tensor_image = transform(mesh_trimesh.visual.material.image)

                # tensor_image = tensor_image[[0, 1, 2, 3]] # .permute(0, 2, 1)

                #
                # from o3b.cv.visual.draw import draw_pixels
                # from o3b.cv.visual.show import show_img
                # _tensor_image = tensor_image.clone()
                # _verts_uvs = verts_uvs[-1].clone()
                # show_img(_tensor_image[:3])
                # _verts_uvs[:, 0] = _verts_uvs[:, 0] * (_tensor_image.shape[-1] - 1)
                # _verts_uvs[:, 1] = _verts_uvs[:, 1] * (_tensor_image.shape[-2] - 1)
                # _tensor_image = draw_pixels(img=_tensor_image, pxls=_verts_uvs)
                # show_img(_tensor_image[:3])
                # draw_pixels

        # mesh_trimesh.visual.material.baseColorFactor
        # mesh_trimesh.faces
        return Meshes(
            verts=vertices,
            faces=faces,
            rgb=rgbs,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            rgbs_uvs=rgbs_uvs,
            device=device,
            **kwargs,
        )

    def to_bytes(self, meshes_ids=None):
        import io
        buffer = io.BytesIO()
        mesh_trimesh = self.to_trimesh(meshes_ids=meshes_ids)
        mesh_trimesh.export(buffer, file_type='glb')
        data = buffer.getvalue()
        return torch.frombuffer(memoryview(data), dtype=torch.uint8).clone()
    
    @staticmethod
    def from_bytes(data: torch.Tensor, load_texts=True, **kwargs):
        data = data.to(dtype=torch.uint8)
        if data.dtype != torch.uint8:
            raise ValueError("data must be a torch.Tensor of dtype uint8")
        import io
        import trimesh
        buffer = io.BytesIO(data.cpu().numpy().tobytes())
        mesh_trimesh = trimesh.load(buffer, file_type='glb', force='mesh')
        device = data.device
        meshes = Meshes.from_trimesh(mesh_trimesh, device=device, load_texts=load_texts, **kwargs)
        return meshes

    def to_pyrender(self, meshes_ids=None):
        import pyrender

        mesh_trimesh = self.to_trimesh(meshes_ids=meshes_ids)
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
        return pyrender_mesh

    def to_bboxs3d(self):
        from o3b.cv.geometry.primitives import Cuboids

        cuboids = Cuboids.create_dense_from_limits(
            limits=self.get_limits(),
            verts_count=None,
            device=self.device,
        )
        return cuboids

    def to_rgbs_uvs(self, mod=VERT_MODALITIES.RGB, verts3d_deform=None, H=200, W=200, return_verts_map=False):
        import xatlas

        # # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
        # # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
        # # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)

        device = self.device

        # # Create the mesh and unwrap
        # mesh = xatlas.Mesh()
        # mesh.set_vertices(self.verts.detach().cpu().numpy())
        # mesh.set_faces(self.faces.detach().cpu().numpy())
        # atlas = xatlas.Atlas()
        # atlas.add_mesh(mesh)
        # atlas.generate()  # Generates UVs
        #
        # # Retrieve the unwrapped data
        # uvs = atlas.meshes[0].uvs  # (num_indices, 2)
        # indices = atlas.meshes[0].indices  # (num_faces * 3,)
        # original_indices = atlas.meshes[0].original_indices

        device = self.verts.device
        dtype = self.verts.dtype

        verts = self.verts.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()

        vmapping, faces2d, uvs = xatlas.parametrize(
            verts,
            faces,
        ) # vmapping: vertex-2d-id to vertex-3d-id


        from trimesh import Trimesh
        uv_verts_3d = np.hstack([uvs, np.zeros((uvs.shape[0], 1))])
        uv_mesh = Trimesh(vertices=uv_verts_3d, faces=faces2d, process=False)

        from o3b.cv.geometry.grid import get_pxl2d
        uvs_grid = get_pxl2d(H, W, dtype, device) # H x W x 2 -> H*W x 2
        # uvs_grid = uvs_grid.flip(dims=(-1,))
        uvs_grid[..., 0] = uvs_grid[..., 0] / (W-1)
        uvs_grid[..., 1] = uvs_grid[..., 1] / (H-1)
        uvs_grid = uvs_grid.detach().cpu().numpy().reshape(-1, 2)
        uvs_grid_3d = np.hstack([uvs_grid, np.zeros((uvs_grid.shape[0], 1))])

        uvs_grid_closest_points, uvs_grid_distances, uvs_grid_face_idx = uv_mesh.nearest.on_surface(uvs_grid_3d)

        uvs_triangles = uvs[faces2d[uvs_grid_face_idx]] # N x 3 x 2/3 (2D/3D)

        if mod == VERT_MODALITIES.NOCS or mod == VERT_MODALITIES.CANOCS:
            if mod == VERT_MODALITIES.NOCS and verts3d_deform is not None:
                _verts = verts.copy() + verts3d_deform[0].detach().cpu().numpy()
                uvs_vertices = _verts[vmapping[faces2d[uvs_grid_face_idx]]]
            else:
                uvs_vertices = verts[vmapping[faces2d[uvs_grid_face_idx]]]

            from trimesh.triangles import points_to_barycentric, barycentric_to_points
            bary = points_to_barycentric(triangles=uvs_triangles, points=uvs_grid, method='cramer')
            nocs = barycentric_to_points(triangles=uvs_vertices, barycentric=bary)
            from o3b.cv.visual.point3d_to_color3d import nocs_0c_to_rgb
            nocs = (torch.from_numpy(nocs).to(device=device, dtype=dtype)) # .clamp(-1, 1) + 1) / 2.
            nocs = nocs_0c_to_rgb(nocs)
            _rgbs_uvs = nocs.reshape(H, W, 3).permute(2, 0, 1)
        else:
            from o3b.cv.visual.blend import gaussian_scatter_image
            uvs = torch.from_numpy(uvs).to(dtype=torch.float, device=self.device)
            faces2d = torch.from_numpy(faces2d).to(dtype=torch.long, device=self.device)

            verts_mod = self.get_vert_mod(mod=mod, obj_id=0, clone=True, verts3d_deform=verts3d_deform[0] if verts3d_deform is not None else None)

            #texture_sampled_rgb = self.rgb[vmapping].clone()
            texture_sampled_rgb =  verts_mod[vmapping].clone()

            texture_sampled_uv = uvs.clone()
            texture_sampled_uv[..., 0] *= (W-1)
            texture_sampled_uv[..., 1] *= (H-1)

            _rgbs_uvs = gaussian_scatter_image(
                uv=texture_sampled_uv,
                rgb=texture_sampled_rgb,
                H=500,
                W=500,
                sigma=10,
            )
            _rgbs_uvs = _rgbs_uvs.to(self.device)

        uvs = torch.from_numpy(uvs).to(dtype=torch.float, device=self.device)
        faces2d = torch.from_numpy(faces2d).to(dtype=torch.long, device=self.device)
        faces = faces2d  # self.faces.clone()
        mesh = Meshes(
            verts_uvs=[uvs],
            rgbs_uvs=[_rgbs_uvs],
            verts=self.verts[vmapping].clone(),
            faces=faces,
        )  #  faces_uvs=[faces2d],

        if return_verts_map:
            return mesh, vmapping
        else:
            return mesh
        # return Meshes(rgbs_uvs=)
        # self.rgbs_uvs =
        # self.faces_uvs =
        # self.verts_uvs = uv_idx

        # trimesh_mesh = self.to_trimesh()
        # trimesh_mesh = trimesh_mesh.unwrap(image=None)
        # return Meshes.from_trimesh(mesh_trimesh=trimesh_mesh)

    def to_trimesh(self, meshes_ids=None):
        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        vertices = self.verts.detach().cpu().clone().numpy()
        faces = (
            self.get_faces_cat_with_mesh_ids(
                mesh_ids=meshes_ids,
                use_global_verts_ids=True,
            )
            .detach()
            .cpu()
            .clone()
            .numpy()
        )

        import trimesh

        # self.show()
        # import trimesh.visual
        if self.rgbs_uvs is not None:
            if self.rgbs_uvs.dim() == 4:
                rgbs_uvs = self.rgbs_uvs[meshes_ids[-1]].clone().detach().cpu()  #
                # .permute(1, 2, 0) * 255).cpu().contiguous().
                #           numpy().astype(np.uint8))
            elif self.rgbs_uvs.dim() == 5:
                rgbs_uvs = self.rgbs_uvs[meshes_ids[-1]][0].clone().detach().cpu()
                # .permute(1, 2, 0) * 255).cpu().contiguous().
                #            numpy().astype(np.uint8))

            # faces_uvs = self.get_faces_uvs_cat_with_mesh_ids(mesh_ids=meshes_ids,
            #                                                 use_global_verts_ids=True).detach().flatten().clone()
            # triangle_uvs = self.verts_uvs[faces_uvs].clone().detach().cpu().numpy()

            #     uv = np.array([
            #         [0.0, 0.0],  # Bottom-left
            #         [1.0, 0.0],  # Bottom-right
            #         [1.0, 1.0],  # Top-right
            #         [0.0, 1.0],  # Top-left
            #     ])

            if rgbs_uvs.dtype == torch.float:
                rgbs_uvs = (rgbs_uvs.clone().detach().cpu() * 255).to(dtype=torch.uint8)

            from torchvision.transforms.functional import to_pil_image

            # from o3b.cv.visual.show import show_img
            # show_img(rgbs_uvs)
            # image = to_pil_image(rgbs_uvs.clone())

            texture = to_pil_image(
                rgbs_uvs.clone().flip(dims=(1,)).clone().contiguous(),
            )

            texture = trimesh.visual.texture.TextureVisuals(
                uv=self.verts_uvs.clone().detach().cpu().numpy(),  # triangle_uvs,
                image=texture,
            )
            #
            # visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture)  # , material=material)
            # frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)

            # Create the mesh with the texture
            trimesh_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=texture,
                process=False,
            )
            # trimesh_mesh.show()
            # a = self.from_trimesh(trimesh_mesh)
            # a.show()
        elif self.rgb is not None:
            rgb = self.rgb.detach().cpu()
            if rgb.dtype == torch.float:
                rgb = (rgb.clone() * 255).to(dtype=torch.uint8)
            trimesh_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=rgb.numpy(),
            )
        else:
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return trimesh_mesh

    @staticmethod
    def from_o3d(
        mesh_o3d: Union[List, o3d.geometry.TriangleMesh],
        device="cpu",
        load_texts=True,
        **kwargs,
    ):
        if not isinstance(mesh_o3d, List):
            meshes_o3d = [mesh_o3d]
        else:
            meshes_o3d = mesh_o3d

        vertices = []
        faces = []
        verts_uvs = []
        faces_uvs = []
        rgbs_uvs = []
        rgbs = []
        for _mesh_o3d in meshes_o3d:
            vertices.append(
                torch.from_numpy(np.asarray(_mesh_o3d.vertices)).to(
                    dtype=torch.float,
                    device=device,
                ),
            )
            faces.append(
                torch.from_numpy(np.asarray(_mesh_o3d.triangles)).to(
                    dtype=torch.long,
                    device=device,
                ),
            )

            if len(_mesh_o3d.vertex_normals) > 0:
                pass

            if load_texts:
                if _mesh_o3d.textures is not None and len(_mesh_o3d.textures) > 0:
                    try:
                        text_id = torch.where(
                            torch.Tensor(
                                [not text.is_empty() for text in _mesh_o3d.textures],
                            ),
                        )
                        text_id = text_id[0].item()

                        rgbs_uvs.append(
                            torch.from_numpy(np.asarray(_mesh_o3d.textures[text_id]))
                            .to(dtype=torch.float, device=device)
                            .permute(2, 0, 1)
                            / 255.0,
                        )
                    except Exception as e:
                        logger.info(e)
                        import open3d as o3d

                        o3d.visualization.draw_geometries([_mesh_o3d])
                    # TODO: Bug ShapeNet

                if len(_mesh_o3d.triangle_uvs) > 0:
                    triangle_uvs_single = torch.from_numpy(
                        np.asarray(_mesh_o3d.triangle_uvs),
                    ).to(
                        dtype=torch.float,
                    )  # F*3x2
                    uvs, uvs_idx = triangle_uvs_single.unique(
                        dim=0,
                        return_inverse=True,
                    )
                    uvs_idx = uvs_idx.reshape(-1, 3)

                    verts_uvs.append(uvs)
                    faces_uvs.append(uvs_idx)

                    # V = vertices[-1].shape[0]
                    # F = faces[-1].shape[0]
                    # verts_uvs_single = torch.zeros((V, 2))
                    # verts_uvs_single[faces[-1].flatten(), :] = triangle_uvs_single[:, :] # Vx2
                    # verts_uvs.append(verts_uvs_single)

                if _mesh_o3d.vertex_colors is not None:
                    rgbs.append(
                        torch.from_numpy(np.asarray(_mesh_o3d.vertex_colors)).to(
                            dtype=torch.float,
                            device=device,
                        ),
                    )

        return Meshes(
            verts=vertices,
            faces=faces,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            rgbs_uvs=rgbs_uvs,
            rgb=rgbs,
            **kwargs,
        )

    def to_o3ds(self, meshes_ids=None):
        if meshes_ids is None:
            meshes_ids = list(range(len(self)))
        return [self.get_meshes_with_ids([mesh_id]).to_o3d() for mesh_id in meshes_ids]

    def to_o3d(self, meshes_ids=None):
        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        import open3d

        vertices = open3d.utility.Vector3dVector(self.verts.detach().cpu().numpy())
        faces = open3d.utility.Vector3iVector(
            self.get_faces_cat_with_mesh_ids(
                mesh_ids=meshes_ids,
                use_global_verts_ids=True,
            )
            .detach()
            .cpu()
            .numpy(),
        )
        o3d_obj_mesh = open3d.geometry.TriangleMesh(vertices=vertices, triangles=faces)

        if self.rgbs_uvs is not None:
            faces_uvs = (
                self.get_faces_uvs_cat_with_mesh_ids(
                    mesh_ids=meshes_ids,
                    use_global_verts_ids=True,
                )
                .detach()
                .flatten()
            )
            triangle_uvs = self.verts_uvs[faces_uvs]
            o3d_obj_mesh.triangle_uvs = open3d.utility.Vector2dVector(
                triangle_uvs.detach().cpu().numpy(),
            )
            if self.rgbs_uvs.dim() == 4:
                o3d_obj_mesh.textures = [
                    open3d.geometry.Image(
                        (self.rgbs_uvs[meshes_ids[-1]].detach().permute(1, 2, 0) * 255)
                        .cpu()
                        .contiguous()
                        .numpy()
                        .astype(np.uint8),
                    ),
                ]
            elif self.rgbs_uvs.dim() == 5:
                o3d_obj_mesh.textures = [
                    open3d.geometry.Image(
                        (
                            self.rgbs_uvs[meshes_ids[-1]][0].detach().permute(1, 2, 0)
                            * 255
                        )
                        .cpu()
                        .contiguous()
                        .numpy()
                        .astype(
                            np.uint8,
                        ),
                    ),
                ]
            o3d_obj_mesh.triangle_material_ids = o3d.utility.IntVector(
                [0] * len(faces_uvs),
            )

            # open3d.visualization.draw_geometries([o3d_obj_mesh])
        elif self.rgb is not None:
            vertex_colors = open3d.utility.Vector3dVector(
                self.rgb.detach().cpu().numpy(),
            )
            o3d_obj_mesh.vertex_colors = vertex_colors
        return o3d_obj_mesh

    def reset_parameters(
        self,
        feats_objects: Union[bool, torch.Tensor] = False,
        feat_clutter: Union[bool, torch.Tensor] = False,
    ) -> None:
        super().reset_parameters(feat_clutter=feat_clutter)

        if self.feats_objects_requires_param:
            if self._feats_objects is not None:
                if isinstance(feats_objects, torch.Tensor):
                    with torch.no_grad():
                        self._feats_objects.copy_(feats_objects)
                else:
                    torch.nn.init.normal_(self._feats_objects)

    @property
    def feats_objects(self):
        return self.activate_feats(self._feats_objects)

    @feats_objects.setter
    def feats_objects(self, value):
        if self._feats_objects is None or not isinstance(
            self._feats_objects,
            torch.nn.Parameter,
        ):
            self._feats_objects = value
        else:
            with torch.no_grad():
                _ = self._feats_objects.copy_(value)

    def get_limits(self):
        meshes_limits = []
        for i in range(len(self)):
            mesh_verts = self.get_verts_with_mesh_id(mesh_id=i)
            mesh_limits = torch.stack(
                [mesh_verts.min(dim=0)[0], mesh_verts.max(dim=0)[0]],
            )
            meshes_limits.append(mesh_limits)
        meshes_limits = torch.stack(meshes_limits, dim=0)
        return meshes_limits

    def get_ranges3d(self):
        return self.get_ranges()

    def get_ranges(self):
        meshes_limits = self.get_limits()
        meshes_range = meshes_limits[:, 1, :] - meshes_limits[:, 0, :]
        return meshes_range

    def get_range1d(self):
        return self.get_ranges().max()

    def get_objs_size3d(self):
        return self.get_ranges3d()

    def get_objs_size1d(self):
        return self.get_objs_size3d().max(dim=-1).values

    @property
    def pt3dmeshes(self):
        from pytorch3d.structures.meshes import Meshes as PT3DMeshes

        return PT3DMeshes(
            verts=[self.get_verts_with_mesh_id(i) for i in range(self.meshes_count)],
            faces=[self.get_faces_with_mesh_id(i) for i in range(self.meshes_count)],
        )

    def get_pt3dmeshes_with_deform(
        self,
        objects_ids,
        instance_deform=None,
        detach_objects_verts=False,
        detach_deform_verts=False,
    ):
        from pytorch3d.structures.meshes import Meshes as PT3DMeshes

        verts = [self.get_verts_with_mesh_id(i) for i in objects_ids]
        if instance_deform is not None:
            if not detach_objects_verts:
                if not detach_deform_verts:
                    verts = [
                        verts[i] + instance_deform.verts_deform[i, : len(verts[i])]
                        for i in objects_ids
                    ]
                else:
                    verts = [
                        verts[i]
                        + instance_deform.verts_deform[i, : len(verts[i])].detach()
                        for i in objects_ids
                    ]
            else:
                if not detach_deform_verts:
                    verts = [
                        verts[i].detach()
                        + instance_deform.verts_deform[i, : len(verts[i])]
                        for i in objects_ids
                    ]
                else:
                    verts = [
                        verts[i].detach()
                        + instance_deform.verts_deform[i, : len(verts[i])].detach()
                        for i in objects_ids
                    ]

        faces = [self.get_faces_with_mesh_id(i) for i in objects_ids]
        return PT3DMeshes(
            verts=verts,
            faces=faces,
        )

    def write_to_file(self, fpath: Path):
        from pytorch3d.io import save_ply

        fpath.parent.mkdir(parents=True, exist_ok=True)
        save_ply(fpath, verts=self.verts, faces=self.faces.detach())

    @dataclass
    class PreRendered:
        cams_tform4x4_obj: torch.Tensor
        cams_intr4x4: torch.Tensor
        imgs_sizes: torch.Tensor
        broadcast_batch_and_cams: bool
        meshes_ids: torch.Tensor
        down_sample_rate: float
        rendering: torch.Tensor

    @classmethod
    def read_from_ply_file(
        cls,
        fpath: Path,
        device="cpu",
        scale=1.0,
        load_texts=True,
        **kwargs,
    ):
        if fpath is None or str(fpath).lower() == "none":
            return Meshes.create_sphere(
                verts_count=1000,
                radius=scale,
                device=device,
                ico=True,
            )

        if isinstance(fpath, str):
            fpath = Path(fpath)

        if not fpath.exists():
            msg = f"mesh fpath does not exist at {fpath}"
            logger.warning(msg)
            raise Exception(msg)
        try:
            import trimesh

            mesh_trimesh = trimesh.load(fpath, force="mesh")
            # o3d_mesh = o3d.io.read_triangle_mesh(str(fpath))
            mesh = Meshes.from_trimesh(
                mesh_trimesh,
                device=device,
                load_texts=load_texts,
                **kwargs,
            )
            mesh.verts = mesh.verts * scale
        except Exception as e:
            logger.info(e)
            logger.warning(f"non ply file {fpath}")
        return mesh

        # try:
        #     o3d_mesh = o3d.io.read_triangle_mesh(str(fpath))
        #     mesh = Meshes.from_o3d(o3d_mesh, device=device, load_texts=load_texts, **kwargs)
        #     mesh.verts = mesh.verts * scale
        # except Exception:
        #     logger.warning(f'non ply file {fpath}')

        # try:
        #     from pytorch3d.io import load_ply
        #     verts, faces = load_ply(fpath)
        #     verts = verts * scale
        # except Exception:
        #     logger.warning(f'non ply file {fpath}')
        #     from pytorch3d.io import IO
        #     io = IO()
        #     mesh = io.load_mesh(fpath, device=device)
        #     verts = mesh[0].verts_list()[0] * scale
        #     faces = mesh[0].faces_list()[0]

        return mesh  #  Meshes(verts=[verts], faces=[faces], device=device, **kwargs)

    # @staticmethod
    # def load_from_file(fpath: Path, device="cpu", scale=1.0):
    #     io = IO()
    #     mesh = io.load_mesh(fpath, device=device)
    #     verts = mesh[0].verts_list()[0] * scale
    #     faces = mesh[0].faces_list()[0]
    #     if mesh[0].textures is not None:
    #         verts_rgb = Mesh.convert_to_textureVertex(
    #             textures_uv=mesh[0].textures,
    #             meshes=mesh[0],
    #         ).verts_features_list()[0]
    #     else:
    #         verts_rgb = None
    #     return Mesh(verts=verts, faces=faces, rgb=verts_rgb)

    @classmethod
    def cat_meshes(
        cls,
        l_meshes: List,
        device=None,
        dtype=None,
        clone=True,
        **kwargs,
    ):
        verts = []
        faces = []
        feats_objects = []
        rgb = []
        rgbs_uvs = []
        verts_uvs = []
        faces_uvs = []
        scene_objs_ids = []
        objs_count = 0
        for meshes in l_meshes:
            meshes_ids = list(range(len(meshes)))
            for mesh_id in meshes_ids:
                verts.append(
                    meshes.get_vert_mod(
                        mod=VERT_MODALITIES.PT3D,
                        obj_id=mesh_id,
                        clone=clone,
                    ).to(device=device),
                )
                faces.append(
                    meshes.get_face_mod(
                        mod=FACE_MODALITIES.VERTS_IN_OBJ_ID,
                        obj_id=mesh_id,
                        clone=clone,
                    ).to(device=device),
                )
                if meshes.rgb is not None:
                    rgb.append(
                        meshes.get_vert_mod(
                            mod=VERT_MODALITIES.RGB,
                            obj_id=mesh_id,
                            clone=clone,
                        ).to(device=device),
                    )
                if meshes.verts_uvs is not None:
                    verts_uvs.append(
                        meshes.get_vert_mod(
                            mod=VERT_MODALITIES.UV_PXL2D,
                            obj_id=mesh_id,
                            clone=clone,
                        ).to(device=device),
                    )
                    faces_uvs.append(
                        meshes.get_face_mod(
                            mod=FACE_MODALITIES.VERTS_UVS_IN_OBJ_ID,
                            obj_id=mesh_id,
                            clone=clone,
                        ).to(device=device),
                    )
                    if clone:
                        rgbs_uvs.append(
                            meshes.rgbs_uvs[mesh_id].clone().to(device=device),
                        )
                    else:
                        rgbs_uvs.append(meshes.rgbs_uvs[mesh_id].to(device=device))
            scene_objs_ids.append([id + objs_count for id in meshes_ids])
            objs_count = len(scene_objs_ids)

        if len(feats_objects) == 0:
            feats_objects = None
        if len(rgb) == 0:
            rgb = None
        if len(verts_uvs) == 0:
            verts_uvs = None
        if len(faces_uvs) == 0:
            faces_uvs = None
        if len(rgbs_uvs) == 0:
            rgbs_uvs = None
        else:
            rgbs_uvs_heights = torch.Tensor([rgb_uv.shape[-2] for rgb_uv in rgbs_uvs])
            rgbs_uvs_height_max = max(rgbs_uvs_heights)
            rgbs_uvs_widths = torch.Tensor([rgb_uv.shape[-2] for rgb_uv in rgbs_uvs])
            rgbs_uvs_width_max = max(rgbs_uvs_widths)
            # rgbs_uvs_heights_scales = rgbs_uvs_height_max / rgbs_uvs_heights
            # rgbs_uvs_widths_scales = rgbs_uvs_width_max / rgbs_uvs_widths
            from o3b.cv.visual.resize import resize

            rgbs_uvs = [
                resize(
                    rgb_uv,
                    H_out=rgbs_uvs_height_max,
                    W_out=rgbs_uvs_width_max,
                    align_corners=False,
                )
                for rgb_uv in rgbs_uvs
            ]
        return (
            Meshes(
                verts=verts,
                faces=faces,
                rgb=rgb,
                feats_objects=feats_objects,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
                rgbs_uvs=rgbs_uvs,
                **kwargs,
            ),
            scene_objs_ids,
        )

    @classmethod
    def cat_single_meshes(
        cls,
        meshes: List,
        device=None,
        dtype=None,
        **kwargs,
    ):
        if device is None:
            device = meshes[0].verts.device
        verts = [mesh.verts.to(device=device) for mesh in meshes]
        faces = [mesh.faces.to(device=device) for mesh in meshes]

        if meshes[0].rgb is not None:
            rgb = [mesh.rgb.to(device=device) for mesh in meshes]
        else:
            rgb = None

        if meshes[0].rgbs_uvs is not None:
            rgbs_uvs = [mesh.rgbs_uvs.to(device=device) for mesh in meshes]
            rgbs_uvs_heights = torch.Tensor([rgb_uv.shape[-2] for rgb_uv in rgbs_uvs])
            rgbs_uvs_height_max = max(rgbs_uvs_heights)
            rgbs_uvs_widths = torch.Tensor([rgb_uv.shape[-2] for rgb_uv in rgbs_uvs])
            rgbs_uvs_width_max = max(rgbs_uvs_widths)
            # rgbs_uvs_heights_scales = rgbs_uvs_height_max / rgbs_uvs_heights
            # rgbs_uvs_widths_scales = rgbs_uvs_width_max / rgbs_uvs_widths
            from o3b.cv.visual.resize import resize

            rgbs_uvs = [
                resize(
                    rgb_uv,
                    H_out=rgbs_uvs_height_max,
                    W_out=rgbs_uvs_width_max,
                    align_corners=False,
                )
                for rgb_uv in rgbs_uvs
            ]
        else:
            rgbs_uvs = None

        if meshes[0].verts_uvs is not None:
            verts_uvs = [mesh.verts_uvs.to(device=device) for mesh in meshes]
        else:
            verts_uvs = None

        if meshes[0].faces_uvs is not None:
            faces_uvs = [mesh.faces_uvs.to(device=device) for mesh in meshes]
        else:
            faces_uvs = None

        return cls(
            verts=verts,
            faces=faces,
            rgb=rgb,
            rgbs_uvs=rgbs_uvs,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def read_from_meshes(
        cls,
        meshes: List,
        device=None,
        dtype=None,
        **kwargs,
    ):
        return cls.cat_single_meshes(
            meshes=meshes,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def read_from_ply_files(
        cls,
        fpaths_meshes: List[Path],
        fpaths_meshes_tforms: List[Path] = None,
        device=None,
        normalize_scale=False,
        **kwargs,
    ):
        # @staticmethod
        # def load_from_file_ply(fpath: Path):
        #    verts, faces = load_ply(fpath)
        #    return Mesh(verts=verts, faces=faces)

        meshes = []
        for i, fpath_mesh in enumerate(fpaths_meshes):
            mesh = Meshes.read_from_ply_file(fpath=fpath_mesh, device=device)
            if (
                fpaths_meshes_tforms is not None
                and fpaths_meshes_tforms[i] is not None
                and str(fpaths_meshes_tforms[i]).lower() != "none"
            ):
                mesh_tform = torch.load(fpaths_meshes_tforms[i]).to(device)
                mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=mesh_tform)
            if normalize_scale:
                mesh.verts /= mesh.verts.abs().max()
            meshes.append(mesh)
        return cls.read_from_meshes(
            meshes=meshes,
            **kwargs,
        )

    @staticmethod
    def from_objs_size3d(obj_size3d, verts_count=None):
        from o3b.cv.geometry.primitives import Cuboids
        cuboids = Cuboids.create_dense_from_size3d(
            size3d=obj_size3d,
            #limits=torch.stack([-obj_size3d / 2., obj_size3d / 2.], dim=-2),
            verts_count = verts_count,
            # limits=torch.Tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
            device=obj_size3d.device,
        )
        return cuboids

    @staticmethod
    def load_by_name(name: str, device="cpu", faces_count=None):
        if name == "bunny":
            bunny_data = o3d.data.BunnyMesh()
            bunny_mesh_open3d = o3d.io.read_triangle_mesh(bunny_data.path)
            if faces_count is not None:
                bunny_mesh_open3d = bunny_mesh_open3d.simplify_quadric_decimation(
                    faces_count,
                )
            bunny_mesh = Meshes.read_from_meshes(
                [Meshes.from_o3d(bunny_mesh_open3d, device=device)],
            )
            bunny_rot = torch.Tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ).to(device=device)
            bunny_mesh.verts.data = transf3d_broadcast(
                pts3d=bunny_mesh.verts,
                transf4x4=bunny_rot,
            )
            return bunny_mesh
        elif name == "cuboid":
            from o3b.cv.geometry.primitives import Cuboids

            cuboids = Cuboids.create_dense_from_limits(
                limits=torch.Tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
                device=device,
            )
            return cuboids
        elif name == "sphere":
            return Meshes.create_sphere(
                verts_count=1000,
                radius=1.0,
                device=device,
                ico=False,
            )
        else:
            raise ValueError(f"Unknown mesh name: {name}")

    def __add__(self, meshes2):
        meshes = []
        for mesh_id in list(range(len(self))):
            meshes.append(self.get_meshes_with_ids(meshes_ids=[mesh_id]))
        for mesh_id in list(range(len(meshes2))):
            meshes.append(meshes2.get_meshes_with_ids(meshes_ids=[mesh_id]))
        return Meshes.read_from_meshes(meshes=meshes, device=self.device)

    @staticmethod
    def get_faces_from_verts(verts, ball_radius=0.3):
        import open3d
        import numpy as np
        from o3b.cv.geometry.transform import (
            transf3d_broadcast,
            transf4x4_from_spherical,
        )

        verts_rot = transf3d_broadcast(
            pts3d=verts,
            transf4x4=transf4x4_from_spherical(
                azim=torch.Tensor([0.05]),
                elev=torch.Tensor([0.05]),
                theta=torch.Tensor([0.05]),
                dist=torch.Tensor([1.0]),
            ),
        )
        verts_centered = verts_rot - verts_rot.mean(dim=-2, keepdim=True)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(verts_centered)
        pcd.normals = open3d.utility.Vector3dVector(verts_centered)
        pcd.estimate_normals()
        # mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd)
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd=pcd,
            radii=open3d.utility.DoubleVector([ball_radius]),
        )
        faces = torch.from_numpy(np.asarray(mesh.triangles))
        return faces

    def __len__(self):
        return self.meshes_count

    def _apply(self, fn):
        super()._apply(fn)

    def __getitem__(cls, x):
        if isinstance(x, int):
            return cls.get_meshes_with_ids(meshes_ids=[x], clone=True)
        else:
            return cls.get_meshes_with_ids(meshes_ids=x, clone=True)

    def get_simplified_mesh(self, mesh_vertices_count=500):
        from o3b.cv.geometry.downsample import random_sampling, voxel_downsampling

        import open3d

        pts3d = self.verts.clone()

        # #### OPTION 3: ALPHA_SHAPE
        pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
        pts3d_downsample_count = len(pts3d)
        o3d_pcl = open3d.geometry.PointCloud()
        o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())

        quantile = max(0.01, 3.0 / len(pts3d))
        particle_size = (
            torch.cdist(pts3d[None,], pts3d[None,]).quantile(dim=-1, q=quantile).mean()
        )
        alpha = particle_size

        o3d_obj_mesh = None
        while (
            o3d_obj_mesh is None
            or not o3d_obj_mesh.is_watertight()
            or not o3d_obj_mesh.is_vertex_manifold()
        ):
            try:
                while o3d_obj_mesh is None or not o3d_obj_mesh.is_watertight():
                    if o3d_obj_mesh is not None:
                        pts3d_downsample_count = int(pts3d_downsample_count * 0.95)
                        alpha = alpha * 1.1
                        o3d_pcl = open3d.geometry.PointCloud()
                        o3d_pcl.points = open3d.utility.Vector3dVector(
                            pts3d.detach().cpu().numpy(),
                        )

                        o3d_pcl = open3d.geometry.PointCloud.farthest_point_down_sample(
                            o3d_pcl,
                            pts3d_downsample_count,
                        )
                        logger.warning(
                            f"alpha {alpha}, pts3d_downsample_count {pts3d_downsample_count}",
                        )

                    o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        o3d_pcl,
                        alpha,
                    )
            except Exception as e:
                logger.warning(
                    f"alpha {alpha}, pts3d_downsample_count {pts3d_downsample_count} failed with {e}",
                )

            if o3d_obj_mesh is not None:
                logger.info(o3d_obj_mesh)
                o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
                logger.info(o3d_obj_mesh)
                faces_count = mesh_vertices_count * 2

                o3d_obj_mesh_downsampled = o3d_obj_mesh
                vertices_count = len(o3d_obj_mesh_downsampled.vertices)
                while vertices_count > mesh_vertices_count:
                    faces_count = int(faces_count * 0.9)
                    o3d_obj_mesh_downsampled = o3d_obj_mesh.simplify_quadric_decimation(
                        target_number_of_triangles=faces_count,
                    )
                    logger.info(o3d_obj_mesh_downsampled)
                    vertices_count = len(o3d_obj_mesh_downsampled.vertices)

                obj_mesh = Meshes.from_o3d(o3d_obj_mesh_downsampled, device=self.device)
            alpha = alpha * 1.3

        return obj_mesh
        #
        # device = self.verts.device
        # import open3d
        # from o3b.cv.geometry.downsample import random_sampling
        # pts3d = self.verts.detach().clone()
        #
        #
        # pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
        # quantile = max(0.01, 3.0 / len(pts3d))
        # particle_size = (
        #     torch.cdist(pts3d[None,], pts3d[None,])
        #     .quantile(dim=-1, q=quantile)
        #     .mean()
        # )
        # alpha = particle_size
        #
        # o3d_pcl = open3d.geometry.PointCloud()
        # o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
        # #o3d_pcl.normals = open3d.utility.Vector3dVector(
        # #    pts3d_normals.detach().cpu().numpy(),
        # #)  # invalidate existing normals
        # #o3d_pcl.colors = open3d.utility.Vector3dVector(
        # #    pts3d_colors.detach().cpu().numpy(),
        # #)
        #
        # o3d_obj_mesh = None
        # while (
        #         o3d_obj_mesh is None
        #         or not o3d_obj_mesh.is_watertight()
        #         or not o3d_obj_mesh.is_vertex_manifold()
        # ):
        #     try:
        #         o3d_obj_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        #             o3d_pcl,
        #             alpha,
        #         )
        #
        #     except Exception as e:
        #         logger.warning(f"alpha {alpha} failed with {e}")
        #
        #     if o3d_obj_mesh is not None:
        #         logger.info(o3d_obj_mesh)
        #         o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
        #         logger.info(o3d_obj_mesh)
        #         faces_count = mesh_vertices_count * 2
        #
        #         o3d_obj_mesh_downsampled = o3d_obj_mesh
        #         vertices_count = len(o3d_obj_mesh_downsampled.vertices)
        #         while vertices_count > mesh_vertices_count:
        #             faces_count = int(faces_count * 0.9)
        #             o3d_obj_mesh_downsampled = (
        #                 o3d_obj_mesh.simplify_quadric_decimation(
        #                     target_number_of_triangles=faces_count,
        #                 )
        #             )
        #             logger.info(o3d_obj_mesh_downsampled)
        #             vertices_count = len(o3d_obj_mesh_downsampled.vertices)
        #
        #         obj_mesh = Meshes.from_o3d(o3d_obj_mesh_downsampled, device=device)
        #     alpha = alpha * 1.3
        # logger.info(obj_mesh)
        # return obj_mesh

        #         o3d_obj_mesh = self.to_o3d()
        # o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
        # logger.info(o3d_obj_mesh)
        # faces_count = mesh_vertices_count * 2
        #
        # o3d_obj_mesh_downsampled = o3d_obj_mesh
        # vertices_count = len(o3d_obj_mesh_downsampled.vertices)
        # while vertices_count > mesh_vertices_count:
        #     faces_count = int(faces_count * 0.9)
        #     o3d_obj_mesh_downsampled = (
        #         o3d_obj_mesh.simplify_quadric_decimation(
        #             target_number_of_triangles=faces_count,
        #         )
        #     )
        #     logger.info(o3d_obj_mesh_downsampled)
        #     vertices_count = len(o3d_obj_mesh_downsampled.vertices)
        #
        # o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
        # logger.info(o3d_obj_mesh)
        # o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
        # return Meshes.from_o3d(o3d_obj_mesh, device=self.verts.device)


    def get_meshes_with_ids(self, meshes_ids=None, clone=False, instance_deform=None):
        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        verts, verts_counts = self.get_vert_mod_from_objs(
            mod=VERT_MODALITIES.PT3D,
            objs_ids=meshes_ids,
            instance_deform=instance_deform,
            padded=True,
            clone=clone,
        )
        faces, faces_counts = self.get_face_mod_from_objs(
            mod=FACE_MODALITIES.VERTS_IN_SCENE_ID,
            objs_ids=meshes_ids,
            padded=True,
            clone=clone,
        )

        verts = self.unpad_mod(
            objs_mod=verts,
            objs_lengths=verts_counts,
            return_concatenated=False,
        )
        faces = self.unpad_mod(
            objs_mod=faces,
            objs_lengths=faces_counts,
            return_concatenated=False,
        )

        if self.rgb is not None:
            rgb, rgbs_counts = self.get_vert_mod_from_objs(
                mod=VERT_MODALITIES.RGB,
                objs_ids=meshes_ids,
                instance_deform=instance_deform,
                padded=True,
                clone=clone,
            )
            rgb = self.unpad_mod(
                objs_mod=rgb,
                objs_lengths=rgbs_counts,
                return_concatenated=False,
            )
        else:
            rgb = None

        if self.rgbs_uvs is not None and self.verts_uvs is not None:
            from copy import copy
            rgbs_uvs = copy(self.rgbs_uvs[0]) # always passed as list otherwise list will be added inisde init of meshes
            verts_uvs = copy(self.verts_uvs)
        else:
            rgbs_uvs = None
            verts_uvs = None

        if self.feats_objects is not None:
            feats_objects, feats_objects_counts = self.get_vert_mod_from_objs(
                mod=VERT_MODALITIES.FEAT,
                objs_ids=meshes_ids,
                instance_deform=instance_deform,
                padded=True,
                clone=clone,
            )
            feats_objects = self.unpad_mod(
                objs_mod=feats_objects,
                objs_lengths=feats_objects_counts,
                return_concatenated=False,
            )
        else:
            feats_objects = None

        return Meshes(verts=verts, faces=faces, rgb=rgb, rgbs_uvs=rgbs_uvs, verts_uvs=verts_uvs,
                      feats_objects=feats_objects)

    def clone(self):
        return self.get_meshes_with_ids(clone=True)


    def get_geodesic_dist(self, dist_max=99999):
        if self._geodesic_dist is None:
            import gdist

            meshes_ids = list(range(len(self)))
            geodesic_dist = (
                torch.ones(
                    size=(self.verts_count, self.verts_count),
                    device=self.device,
                )
                * torch.inf
            )
            for mesh_id in meshes_ids:
                verts = self.get_verts_with_mesh_id(mesh_id=mesh_id)
                faces = self.get_faces_with_mesh_id(mesh_id=mesh_id)
                mesh_verts_geodesic_dist = gdist.local_gdist_matrix(
                    vertices=verts.detach().cpu().to(torch.float64).numpy(),
                    triangles=faces.cpu().detach().to(torch.int32).numpy(),
                    max_distance=dist_max,
                )
                # convert  scipy.sparse._csc.csc_matrix to torch.Tensor, fill sparse with torch.inf
                mesh_verts_geodesic_dist = torch.from_numpy(
                    mesh_verts_geodesic_dist.toarray(),
                ).to(
                    dtype=torch.float32,
                    device=self.device,
                )
                mesh_verts_geodesic_dist = (
                    mesh_verts_geodesic_dist / mesh_verts_geodesic_dist.max()
                )
                mesh_verts_geodesic_dist[mesh_verts_geodesic_dist == 0] = torch.inf
                mesh_verts_geodesic_dist[
                    torch.arange(mesh_verts_geodesic_dist.shape[0]).to(self.device),
                    torch.arange(
                        mesh_verts_geodesic_dist.shape[0],
                    ).to(self.device),
                ] = 0.0

                # mesh_verts_geodesic_dist = torch.from_numpy(mesh_verts_geodesic_dist.toarray()).to(dtype=torch.float32, device=self.device)

                geodesic_dist[
                    self.verts_counts_acc_from_0[
                        mesh_id
                    ] : self.verts_counts_acc_from_0[mesh_id + 1],
                    self.verts_counts_acc_from_0[
                        mesh_id
                    ] : self.verts_counts_acc_from_0[mesh_id + 1],
                ] = mesh_verts_geodesic_dist

            # euclidean_dist = torch.cdist(self.verts[None,], self.verts[None,])[0]
            # geodesic_dist = torch.ones_like(euclidean_dist) * torch.inf
            # would need to get verts nearest neighbors and iteratively propagating geodesic_dist
            self._geodesic_dist = geodesic_dist

        return self._geodesic_dist

    def get_geodesic_prob(self):
        _geodesic_dist = self.get_geodesic_dist().clone()
        _geodesic_prob = torch.exp(
            input=-0.5 * (_geodesic_dist / (self.geodesic_prob_sigma + 1e-10)) ** 2,
        )
        # replace inf with 0
        _geodesic_prob[torch.isinf(_geodesic_dist)] = 0.0
        _geodesic_prob = _geodesic_prob / _geodesic_prob.sum(dim=-1, keepdim=True)
        return _geodesic_prob

    def get_geodesic_prob_with_noise(self):
        geodesic_prob_with_noise = torch.eye(
            self.verts_count + 1,
            device=self.device,
        )
        geodesic_prob_with_noise[:-1, :-1] = self.get_geodesic_prob()
        return geodesic_prob_with_noise

    @property
    def verts_coarse(self):
        if self._verts_coarse is None:
            self.update_verts_coarse()
        return self._verts_coarse

    @property
    def verts_ids_coarse(self):
        if self._verts_ids_coarse is None:
            self.update_verts_coarse()
        return self._verts_ids_coarse

    @property
    def verts_label_coarse(self):
        if self._verts_label_coarse is None:
            self.update_verts_coarse()
        return self._verts_label_coarse

    def update_verts_coarse(self):
        verts_ids_coarse = []
        verts_coarse = []
        verts_label_coarse = []
        for mesh_id in range(len(self)):
            verts = self.get_verts_with_mesh_id(mesh_id=mesh_id).detach()

            from o3b.cv.geometry.downsample import fps

            pts_inds, pts_vals = fps(pts3d=verts, K=self.verts_coarse_count)

            # gaussian
            dist_pts = torch.cdist(
                verts,
                pts_vals,
            )  # N x K torch.ones(N, N) * torch.inf
            # dist_nearest_std = (dist_pts ** 2).min(dim=1, keepdim=True)[0].mean() ** 0.5
            # dist_nearest_std = dist_nearest_std.clamp(min=1e-10)

            # old version
            # dist_nearest_std_factor = 1
            # dist_nearest_std = (dist_pts**2).min(dim=1, keepdim=True)[0].mean() ** 0.5
            # dist_nearest_std = dist_nearest_std.clamp(min=1e-10)
            # label_smooth = (
            #     -(dist_pts**2) / ((dist_nearest_std_factor * dist_nearest_std) ** 2)
            # ).softmax(
            #     dim=-1,
            # )  # N x K

            # new version
            dist_nearest_std = self.verts_coarse_prob_sigma
            label_smooth = (-dist_pts ** 2 / ((dist_nearest_std) ** 2)).softmax(
                dim=-1)  # N x K

            verts_label_coarse.append(label_smooth)
            verts_coarse.append(pts_vals)
            verts_ids_coarse.append(pts_inds)

        self._verts_ids_coarse = torch.stack(verts_ids_coarse, dim=0)  # M x K
        self._verts_coarse = torch.stack(verts_coarse, dim=0)  # M x K x 3
        self._verts_label_coarse = torch.cat(verts_label_coarse)  # N x K

    def visualize_coarse_labels(self):
        from o3b.cv.visual.show import get_colors, show_scene

        self.verts = self.verts - self.verts.mean(dim=0, keepdim=True)
        self.verts = self.verts / self.verts.max()
        verts_label_coarse_colors = get_colors(
            K=self.verts_coarse_count,
            device=self.verts.device,
        )  # K x 3
        self.rgb = torch.nn.Parameter(
            self.verts_label_coarse @ verts_label_coarse_colors,
        )
        show_scene(
            meshes=self,
            fpath=Path("mesh.webm"),
            viewpoints_count=30,
            fps=3,
            return_visualization=False,
        )

    def set_rgb(self, rgb):
        if self.rgb is not None:
            self.rgb.copy_(rgb.to(self.device))
        else:
            self.rgb = torch.nn.Parameter(rgb.to(self.device))

    def set_rgb_to_canocs(self, instance_deform=None):
        objs_mod, objs_lengths = self.get_vert_mod_from_objs(mod=VERT_MODALITIES.CANOCS,
                                                             instance_deform=instance_deform, clone=True)
        self.set_rgb(rgb = objs_mod )

    def get_verts_ncds_with_mesh_id(self, mesh_id):
        verts3d = self.get_verts_with_mesh_id(mesh_id)
        verts3d_ncds = (verts3d - verts3d.min(dim=0).values[None,]) / (
            verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,]
        )
        return verts3d_ncds

    def get_verts_ncds_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        verts3d_ncds = []
        for mesh_id in mesh_ids:
            verts3d_ncds.append(self.get_verts_ncds_with_mesh_id(mesh_id=mesh_id))
        verts3d_ncds = torch.cat(verts3d_ncds, dim=0)
        return verts3d_ncds

    def get_verts_deform_cat_with_mesh_ids(self, mesh_ids, instance_deform):
        verts3d_deform = []
        for m, mesh_id in enumerate(mesh_ids):
            verts3d_deform.append(
                instance_deform.verts_deform[m][
                    : len(self.get_verts_with_mesh_id(mesh_id=mesh_id))
                ],
            )
        verts3d_deform = torch.cat(verts3d_deform, dim=0)
        return verts3d_deform

    def get_verts_cat_with_mesh_ids(
        self,
        mesh_ids=None,
        instance_deform=None,
        detach_objects_verts=False,
    ):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        verts3d = []
        for mesh_id in mesh_ids:
            verts3d.append(self.get_verts_with_mesh_id(mesh_id=mesh_id))

        if instance_deform is not None:
            verts3d_deform = self.get_verts_deform_cat_with_mesh_ids(
                mesh_ids,
                instance_deform,
            )

        verts3d = torch.cat(verts3d, dim=0)
        if detach_objects_verts:
            verts3d = verts3d.detach()

        if instance_deform is not None:
            verts3d = verts3d + verts3d_deform

        return verts3d

    def get_faces_uvs_cat_with_mesh_ids(
        self,
        mesh_ids=None,
        use_global_verts_ids=False,
        add_verts_offset=False,
    ):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        faces_uv = []
        verts_uv_count = 0
        for mesh_id in mesh_ids:
            faces_uv.append(
                self.get_faces_uvs_with_mesh_id(
                    mesh_id=mesh_id,
                    use_global_verts_ids=use_global_verts_ids,
                    clone=True,
                ),
            )
            if add_verts_offset:
                faces_uv[-1] += verts_uv_count
                verts_uv_count += self.verts_uvs_counts_max
        faces_uv = torch.cat(faces_uv, dim=0)
        return faces_uv

    def get_faces_cat_with_mesh_ids(self, mesh_ids=None, use_global_verts_ids=False):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        faces = []
        for mesh_id in mesh_ids:
            faces.append(
                self.get_faces_with_mesh_id(
                    mesh_id=mesh_id,
                    use_global_verts_ids=use_global_verts_ids,
                ),
            )
        faces = torch.cat(faces, dim=0)
        return faces

    def get_edges_cat_with_mesh_ids(
        self,
        mesh_ids=None,
        use_global_verts_ids=False,
        use_verts_offset=True,
    ):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        edges_cat = []
        verts_offset = 0
        for mesh_id in mesh_ids:
            edges = self.get_edges_with_mesh_id(
                mesh_id=mesh_id,
                use_global_verts_ids=use_global_verts_ids,
            )
            if use_verts_offset and not use_global_verts_ids:
                edges += verts_offset
            edges_cat.append(edges)
            verts_offset += self.verts_counts[mesh_id]
        edges_cat = torch.cat(edges_cat, dim=0)
        return edges_cat

    def get_verts_rgbs_from_faces_with_mesh_id(self, mesh_id):
        # return self.feats_from_faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
        return self.get_rgb_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)]

    def get_verts_ncds_from_faces_with_mesh_id(self, mesh_id):
        verts3d_ncds = self.get_verts_ncds_with_mesh_id(mesh_id)
        feats_from_faces = verts3d_ncds[self.get_faces_with_mesh_id(mesh_id)]
        return feats_from_faces

    def get_feats_from_faces_with_mesh_id(self, mesh_id):
        # return self.feats_from_faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
        return self.get_feats_with_mesh_id(mesh_id)[
            self.get_faces_with_mesh_id(mesh_id)
        ]

    def get_rgb_with_mesh_id(self, mesh_id, clone=False):
        if self.rgb is not None:
            _rgb = self.rgb
        else:
            _rgb = torch.ones_like(self.verts) * 0.5

        if not clone:
            return _rgb[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return _rgb[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_verts_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.verts[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return self.verts[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_verts_uvs_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_uvs_with_pad(
            tensor=self.get_verts_uvs_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_verts_uvs_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.verts_uvs[
                self.verts_uvs_counts_acc_from_0[
                    mesh_id
                ] : self.verts_uvs_counts_acc_from_0[mesh_id + 1]
            ]
        else:
            return self.verts_uvs[
                self.verts_uvs_counts_acc_from_0[
                    mesh_id
                ] : self.verts_uvs_counts_acc_from_0[mesh_id + 1]
            ].clone()

    def get_mesh_ids_for_verts(self):
        mesh_ids = torch.LongTensor(size=(0,)).to(device=self.device)
        for mesh_id in range(self.meshes_count):
            mesh_ids = torch.cat(
                [
                    mesh_ids,
                    torch.LongTensor([mesh_id] * self.verts_counts[mesh_id]).to(
                        device=self.device,
                    ),
                ],
                dim=0,
            )
        return mesh_ids

    def get_feats_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.feats_objects[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return self.feats_objects[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_feats_coarse_with_mesh_id(self, mesh_id, clone=False):
        return self.get_feats_with_mesh_id(mesh_id=mesh_id, clone=clone)[
            self.verts_ids_coarse[mesh_id]
        ]

    def get_faces_with_mesh_id(self, mesh_id, clone=False, use_global_verts_ids=False):
        faces = self.faces[
            self.faces_counts_acc_from_0[mesh_id] : self.faces_counts_acc_from_0[
                mesh_id + 1
            ]
        ]
        if clone:
            faces = faces.clone()

        if use_global_verts_ids:
            faces += self.verts_counts_acc_from_0[mesh_id]

        return faces

    def get_faces_uvs_with_mesh_id(
        self,
        mesh_id,
        clone=False,
        use_global_verts_ids=False,
    ):
        faces = self.faces_uvs[
            self.faces_counts_acc_from_0[mesh_id] : self.faces_counts_acc_from_0[
                mesh_id + 1
            ]
        ]
        if clone:
            faces = faces.clone()

        if use_global_verts_ids:
            faces += self.verts_uvs_counts_acc_from_0[mesh_id]

        return faces

    def get_edges_with_mesh_id(self, mesh_id, clone=False, use_global_verts_ids=False):
        faces = self.get_faces_with_mesh_id(
            mesh_id,
            clone=clone,
            use_global_verts_ids=use_global_verts_ids,
        )
        F = faces.shape[0]
        v0, v1, v2 = faces.chunk(3, dim=1)
        e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
        e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
        e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

        # All edges including duplicates.
        edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)

        # Sort the edges in increasing vertex order to remove duplicates as
        # the same edge may appear in different orientations in different faces.
        # i.e. rows in edges after sorting will be of the form (v0, v1) where v1 > v0.
        # This sorting does not change the order in dim=0.
        edges, _ = edges.sort(dim=1)

        # Remove duplicate edges: convert each edge (v0, v1) into an
        # integer hash = V * v0 + v1; this allows us to use the scalar version of
        # unique which is much faster than edges.unique(dim=1) which is very slow.
        # After finding the unique elements reconstruct the vertex indices as:
        # (v0, v1) = (hash / V, hash % V)
        # The inverse maps from unique_edges back to edges:
        # unique_edges[inverse_idxs] == edges
        # i.e. inverse_idxs[i] == j means that edges[i] == unique_edges[j]

        V = self.verts_counts[mesh_id]
        edges_hash = V * edges[:, 0] + edges[:, 1]
        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

        # # Find indices of unique elements.
        # # TODO (nikhilar) remove following 4 lines when torch.unique has support
        # # for returning unique indices
        # sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
        # unique_mask = torch.ones(
        #     edges_hash.shape[0], dtype=torch.bool, device=self.device
        # )
        # unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
        # unique_idx = sort_idx[unique_mask]

        edges_packed = torch.stack([u // V, u % V], dim=1)

        return edges_packed

    def get_faces_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_faces_with_pad(
            tensor=self.get_faces_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_faces_uvs_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_faces_with_pad(
            tensor=self.get_faces_uvs_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_tensor_verts_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.verts_counts_max - self.verts_counts[mesh_id]])
        return torch.cat(
            [
                tensor,
                torch.zeros(
                    size=pad + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=0,
        )

    def get_tensor_verts_uvs_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.verts_uvs_counts_max - self.verts_uvs_counts[mesh_id]])
        return torch.cat(
            [
                tensor,
                torch.zeros(
                    size=pad + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=0,
        )

    def get_tensor_faces_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.faces_counts_max - self.faces_counts[mesh_id]])
        return torch.cat(
            [
                tensor,
                torch.zeros(
                    size=pad + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=0,
        )

    def get_tensor_padded_to_size(self, tensor, size, val=0):
        pad = torch.Size([size - tensor.shape[0]])
        return torch.cat(
            [
                tensor,
                val
                * torch.ones(
                    size=pad + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=0,
        )

    def get_mod_bg(self, mod: VERT_MODALITIES):
        map_mod_to_bg_tensor = {
            VERT_MODALITIES.OBJ_ID.value: -1 * torch.ones(size=(1,), dtype=torch.long),
            VERT_MODALITIES.OBJ_ONEHOT.value: 0
            * torch.ones(size=(1,), dtype=torch.bool),
            VERT_MODALITIES.IN_OBJ_COARSE_ONEHOT_RGB.value: 0
            * torch.ones(size=(1,), dtype=torch.bool),
            VERT_MODALITIES.IN_OBJ_COARSE_ONEHOT.value: 0
            * torch.ones(size=(1,), dtype=torch.bool),
            VERT_MODALITIES.OBJ_IN_SCENE_ID.value: -1
            * torch.ones(size=(1,), dtype=torch.long),
            VERT_MODALITIES.OBJ_IN_SCENE_ONEHOT.value: 0
            * torch.ones(size=(1,), dtype=torch.bool),
            VERT_MODALITIES.PT3D_NCDS.value: 1.0
            * torch.ones(size=(1,), dtype=torch.float),
            VERT_MODALITIES.NOCS.value: 1.0
            * torch.ones(size=(1,), dtype=torch.float),
            VERT_MODALITIES.CANOCS.value: 1.0
            * torch.ones(size=(1,), dtype=torch.float),
            VERT_MODALITIES.PT3D_NCDS_AVG.value: 1.0
            * torch.ones(size=(1,), dtype=torch.float),
            VERT_MODALITIES.RGB.value: 1.0 * torch.ones(size=(1,), dtype=torch.float),
        }
        return map_mod_to_bg_tensor[mod]


    def get_transf3d_verts_padded_and_counts(self, objs_new_tform4x4_objs: torch.Tensor, objs_ids=None, instance_deform=None):
        if objs_ids is None:
            objs_ids = torch.arange(len(self))

        objs_verts_padded, objs_lengths = self.get_vert_mod_from_objs(
            mod=VERT_MODALITIES.PT3D,
            objs_ids=objs_ids,
            padded=True,
            instance_deform=instance_deform,
        )
        objs_verts_padded = transf3d_broadcast(
            pts3d=objs_verts_padded,
            transf4x4=objs_new_tform4x4_objs[:, None],
        )
        return objs_verts_padded, objs_lengths

    def get_transf3d_verts(self, objs_new_tform4x4_objs: torch.Tensor, objs_ids=None, instance_deform=None):

        objs_verts_padded, objs_lengths = self.get_transf3d_verts_padded_and_counts(
            objs_new_tform4x4_objs=objs_new_tform4x4_objs, objs_ids=objs_ids, instance_deform=instance_deform)

        verts = self.unpad_mod(
            objs_mod=objs_verts_padded,
            objs_lengths=objs_lengths,
        )
        return verts

    def transf3d(self, objs_new_tform4x4_objs: torch.Tensor, objs_ids=None, instance_deform=None,):

        self.verts = self.get_transf3d_verts(objs_new_tform4x4_objs=objs_new_tform4x4_objs, objs_ids=objs_ids, instance_deform=instance_deform)

    def get_objscentric_tform4x4_objs(self, tform_objs=None):
        _tform_objs = []
        for i in range(len(self)):
            _tform_obj = tform_objs[i] if tform_objs is not None else None
            _tform_objs.append(
                self.get_objcentric_tform4x4_obj(obj_id=i, tform_obj=_tform_obj),
            )
        return torch.stack(_tform_objs, dim=0)

    def get_objcentric_tform4x4_obj(self, obj_id: int = None, tform_obj=None):
        if obj_id is None:
            obj_id = 0
        if tform_obj is None:
            tform_obj = torch.eye(4).to(device=self.device)

        mesh_verts_orig = self.get_vert_mod(
            mod=VERT_MODALITIES.PT3D,
            obj_id=obj_id,
            clone=True,
        )
        mesh_verts = transf3d_broadcast(
            pts3d=mesh_verts_orig.clone(),
            transf4x4=tform_obj,
        )
        tform_obj_buf = tform_obj.clone()
        transl = -(mesh_verts.max(dim=0)[0] + mesh_verts.min(dim=0)[0]) / 2.0 # this transl does not correspond to OMni6dPose
        tform_obj_buf[:3, 3] += transl

        mesh_verts = transf3d_broadcast(
            pts3d=mesh_verts_orig.clone(),
            transf4x4=tform_obj_buf,
        )
        scale = 1.0 / (max(mesh_verts.max(dim=0).values - mesh_verts.min(dim=0).values) / 2) #  mesh_verts.abs().max() #  this scale does not ensure completely mid? this could be larger
        tform_obj_buf[:3, :] *= scale

        # mesh_verts = transf3d_broadcast(pts3d=mesh.verts.clone(), transf4x4=tform_obj_buf)
        tform_obj = tform_obj_buf

        return tform_obj

    def get_vert_mod(self, mod: VERT_MODALITIES, obj_id: int, clone=False, verts3d_deform=None):
        if mod == VERT_MODALITIES.OBJ_ID:
            data = obj_id * torch.ones(self.verts_counts[obj_id], dtype=torch.long)
        elif mod == VERT_MODALITIES.OBJ_ONEHOT:
            data = torch.nn.functional.one_hot(
                obj_id * torch.ones(self.verts_counts[obj_id], dtype=torch.long),
            )
        elif (
            mod == VERT_MODALITIES.IN_OBJ_ID
        ):  #  or mod == VERT_MODALITIES.IN_SCENE_ID:
            data = torch.arange(self.verts_counts[obj_id])
        elif (
            mod == VERT_MODALITIES.IN_OBJ_ONEHOT
        ):  # or mod == VERT_MODALITIES.IN_SCENE_ONEHOT:
            data = torch.nn.functional.one_hot(torch.arange(self.verts_counts[obj_id]))
        elif (
            mod == VERT_MODALITIES.IN_OBJ_COARSE_ID
        ):  #  or mod == VERT_MODALITIES.IN_SCENE_COARSE_ID:
            data = torch.arange(self.verts_coarse_counts[obj_id])
        elif (
            mod == VERT_MODALITIES.IN_OBJ_COARSE_ONEHOT
        ):  #  or mod == VERT_MODALITIES.IN_SCENE_COARSE_ONEHOT:
            data = torch.nn.functional.one_hot(self.verts_coarse_counts[obj_id])
        elif mod == VERT_MODALITIES.IN_OBJ_COARSE_ONEHOT_RGB:
            from o3b.cv.visual.show import get_colors, show_scene

            data = self.verts - self.verts.mean(dim=0, keepdim=True)
            data = data / data.abs().max()
            verts_label_coarse_colors = get_colors(
                K=self.verts_coarse_count,
                device=self.device,
            )  # K x 3

            freq = 1.5
            bin_count = int(3)
            mod = (
                torch.cos(self.verts_coarse[obj_id] * torch.pi * 2 * freq) / 2 + 0.5
            )  # [0, 1] -> [0, 2*pi] ->  # N x 3
            bins = torch.linspace(1.0 / bin_count, 1.0, bin_count).to(mod.device)
            bins_indices = torch.bucketize(
                input=mod,
                boundaries=bins,
                out_int32=True,
            )  # , right=True)
            mod = bins[bins_indices]
            rgb_min = 0.1
            rgb_max = 0.9
            mod = mod.clamp(rgb_min, rgb_max)
            verts_label_coarse_colors = mod

            data = self.verts_label_coarse @ verts_label_coarse_colors

        elif mod == VERT_MODALITIES.PT3D:
            data = self.verts[
                self.verts_counts_acc_from_0[obj_id] : self.verts_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
        elif mod == VERT_MODALITIES.MASK:
            data = torch.ones_like(self.verts[self.verts_counts_acc_from_0[obj_id] :
                                              self.verts_counts_acc_from_0[obj_id + 1], 0], dtype=torch.bool)
        elif mod == VERT_MODALITIES.UV_PXL2D:
            data = self.verts_uvs[
                self.verts_uvs_counts_acc_from_0[
                    obj_id
                ] : self.verts_uvs_counts_acc_from_0[obj_id + 1]
            ]
        elif mod == VERT_MODALITIES.FEAT:
            data = self.feats_objects[
                self.verts_counts_acc_from_0[obj_id] : self.verts_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
        elif mod == VERT_MODALITIES.RGB:
            data = self.rgb[
                self.verts_counts_acc_from_0[obj_id] : self.verts_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
        elif mod == VERT_MODALITIES.NOCS:
            verts3d = self.verts[
                self.verts_counts_acc_from_0[obj_id] : self.verts_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
            if verts3d_deform is not None:
                verts3d = verts3d + verts3d_deform[:self.verts_counts[obj_id]]

            data = (verts3d - verts3d.min(dim=0).values[None,]) / (
                verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,]
            )
        elif mod == VERT_MODALITIES.CANOCS:
            verts3d = self.verts[
                self.verts_counts_acc_from_0[obj_id] : self.verts_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
            data = (verts3d - verts3d.min(dim=0).values[None,]) / (
                verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,]
            )
        elif mod == VERT_MODALITIES.PT3D_NCDS:
            verts3d = self.verts[
                self.verts_counts_acc_from_0[obj_id] : self.verts_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
            data = (verts3d - verts3d.min(dim=0).values[None,]) / (
                verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,]
            )
        elif mod == VERT_MODALITIES.PT3D_NCDS_AVG:
            mod, mod_lens = self.get_vert_mod_from_objs(
                mod=VERT_MODALITIES.PT3D_NCDS,
                objs_ids=None,
                padded=True,
                instance_deform=None,
            )
            mod = mod.mean(dim=0)
            freq = 1.5
            mod = (
                torch.cos(mod * torch.pi * 2 * freq) / 2 + 0.5
            )  # [0, 1] -> [0, 2*pi] ->  # N x 3

            bin_count = int(3)

            bins = torch.linspace(1.0 / bin_count, 1.0, bin_count).to(mod.device)
            bins_indices = torch.bucketize(
                input=mod,
                boundaries=bins,
                out_int32=True,
            )  # , right=True)
            mod = bins[bins_indices]

            rgb_min = 0.1
            rgb_max = 0.9
            mod = mod.clamp(rgb_min, rgb_max)
            # mod = rgb_min + (mod - rgb_min ) / (rgb_max-rgb_min) * (rgb_max-rgb_min)
            return mod
        else:
            raise NotImplementedError

        if clone:
            data = data.clone()

        return data


    def pad_mod(self, objs_mod, objs_lengths=None, objs_ids=None, concatenated_input=False):
        # not concatenated_input:
        # B*OxNxF-> BxOxNxF
        # OxNxF -> OxNxF

        if objs_lengths is None:
            objs_lengths = torch.LongTensor([obj_mod.shape[0] for obj_mod in objs_mod])

        # B*O*NxF -> BxOxNxF
        # O*NxF -> OxNxF
        if concatenated_input:
            objs_mod_ids_limits = cumsum_w0_and_wEnd(objs_lengths.flatten(), dim=0)
            objs_mod = [objs_mod[objs_mod_ids_limits[i]:objs_mod_ids_limits[i+1]]
                         for i in range(len(objs_mod_ids_limits) - 1)]

        if objs_ids is None:
            objs_ids = torch.arange(len(objs_mod))

        # NxB*OxF
        objs_mod = torch.nn.utils.rnn.pad_sequence(
            objs_mod,
            batch_first=False,
            padding_value=0,
            padding_side="right",
        )
        objs_mod = objs_mod.swapaxes(0, 1)
        objs_mod = objs_mod.clone()

        #  when is it essential to not pad sequence?,
        #  when deciding what to render and leaving out faces is more efficient
        if objs_ids.dim() == 2:
            # return BxOxNxF
            scene_count = objs_ids.shape[0]
            objs_per_scene_count = objs_ids.shape[1]

            objs_mod = objs_mod.reshape(
                scene_count,
                objs_per_scene_count,
                *objs_mod.shape[1:],
            )
            objs_lengths = objs_lengths.reshape(scene_count, objs_per_scene_count)
        else:
            # return BxNxF
            scene_count = objs_ids.shape[0]
            objs_mod = objs_mod.reshape(scene_count, *objs_mod.shape[1:])
            objs_lengths = objs_lengths.reshape(scene_count)

        return objs_mod

    def unpad_mod(self, objs_mod, objs_lengths, return_concatenated=True):
        # OxNxF -> OxNxF (-> O*NxF)
        # BxOxNxF -> B*OxNxF ( -> B*O*NxF)
        if objs_lengths.dim() == 1:
            objs_lengths = objs_lengths.reshape(-1)
            objs_mod = objs_mod.reshape(-1, *objs_mod.shape[1:])
        elif objs_lengths.dim() == 2:
            objs_lengths = objs_lengths.reshape(-1)
            objs_mod = objs_mod.reshape(-1, *objs_mod.shape[2:])

        objs_mod = objs_mod.swapaxes(0, 1).clone()  # OxNx... -> NxOx....
        objs_mod = torch.nn.utils.rnn.unpad_sequence(
            padded_sequences=objs_mod,
            lengths=objs_lengths,
            batch_first=False,
        )
        if return_concatenated:
            objs_mod = torch.cat(objs_mod, dim=0)

        return objs_mod

    def fill_objs_mod_none(self, objs_mod):
        obj_first_not_none = [obj_mod for obj_mod in objs_mod if obj_mod is not None][0]
        mod_shape = obj_first_not_none.shape[1:]
        mod_dtype = obj_first_not_none.dtype
        mod_device = obj_first_not_none.device
        mod_buf = torch.zeros((0,) + mod_shape).to(dtype=mod_dtype, device=mod_device)
        objs_mod = [obj_mod if obj_mod is not None else mod_buf for obj_mod in objs_mod]
        return objs_mod

    def get_vert_mod_from_objs(
        self,
        mod,
        objs_ids=None,
        padded=False,
        instance_deform=None,
        clone=False,
    ):
        if objs_ids is None:
            objs_ids = torch.arange(len(self))

        if isinstance(objs_ids, list):
            objs_ids = torch.LongTensor(objs_ids).to(self.device)

        if (
            mod == VERT_MODALITIES.OBJ_IN_SCENE_ID
            or mod == VERT_MODALITIES.OBJ_IN_SCENE_ONEHOT
        ):
            objs_mod, objs_lengths = self.get_vert_mod_from_objs(
                mod=VERT_MODALITIES.IN_OBJ_ID,
                objs_ids=objs_ids,
                padded=True,
                clone=clone,
            )

            if objs_lengths.dim() == 2:
                from o3b.cv.select import cumsum_w0

                objs_mod[:] = 1
                objs_mod = cumsum_w0(input=objs_mod, dim=1)
                objs_count_per_scene = objs_mod.shape[1]
            else:
                objs_mod[:] = 0
                objs_count_per_scene = 1

            if mod == VERT_MODALITIES.OBJ_IN_SCENE_ONEHOT:
                objs_mod = torch.nn.functional.one_hot(
                    input=objs_mod,
                    num_classes=objs_count_per_scene,
                )

            if not padded:
                objs_mod = self.unpad_mod(objs_mod=objs_mod, objs_lengths=objs_lengths)
                return objs_mod, None
            else:
                return objs_mod, objs_lengths

        if (
            mod == VERT_MODALITIES.IN_SCENE_ONEHOT
            or mod == VERT_MODALITIES.IN_SCENE_COARSE_ONEHOT
        ):
            if mod == VERT_MODALITIES.IN_SCENE_ONEHOT:
                # BxOxN -> BxOxNxN*0
                # BxN -> BxNxN
                objs_mod, objs_lengths = self.get_vert_mod_from_objs(
                    mod=VERT_MODALITIES.IN_SCENE_ID,
                    objs_ids=objs_ids,
                    padded=True,
                    clone=clone,
                )
            elif mod == VERT_MODALITIES.IN_SCENE_COARSE_ONEHOT:
                objs_mod, objs_lengths = self.get_vert_mod_from_objs(
                    mod=VERT_MODALITIES.IN_SCENE_COARSE_ID,
                    objs_ids=objs_ids,
                    padded=True,
                    clone=clone,
                )
            else:
                raise NotImplementedError

            num_classes = (
                objs_mod.shape[-1] * objs_mod.shape[-2]
                if objs_lengths.dim() == 2
                else objs_mod.shape[-1]
            )
            objs_mod = torch.nn.functional.one_hot(
                input=objs_mod,
                num_classes=num_classes,
            )

            if not padded:
                objs_mod = self.unpad_mod(objs_mod=objs_mod, objs_lengths=objs_lengths)
                return objs_mod, None
            else:
                return objs_mod, objs_lengths

        if (
            mod == VERT_MODALITIES.IN_SCENE_ID
            or mod == VERT_MODALITIES.IN_SCENE_COARSE_ID
        ):
            if mod == VERT_MODALITIES.IN_SCENE_ID:
                objs_mod, objs_lengths = self.get_vert_mod_from_objs(
                    mod=VERT_MODALITIES.IN_OBJ_ID,
                    objs_ids=objs_ids,
                    padded=True,
                    clone=clone,
                )
            elif mod == VERT_MODALITIES.IN_SCENE_COARSE_ID:
                objs_mod, objs_lengths = self.get_vert_mod_from_objs(
                    mod=VERT_MODALITIES.IN_OBJ_COARSE_ID,
                    objs_ids=objs_ids,
                    padded=True,
                    clone=clone,
                )
            else:
                raise NotImplementedError

            if objs_lengths.dim() == 2:
                from o3b.cv.select import cumsum_w0

                objs_lengths_acc_from_0 = cumsum_w0(input=objs_lengths, dim=1)
                objs_mod += objs_lengths_acc_from_0

            if not padded:
                objs_mod = self.unpad_mod(objs_mod=objs_mod, objs_lengths=objs_lengths)
                return objs_mod, None
            else:
                return objs_mod, objs_lengths

        if objs_ids.dim() == 1:
            # return BxNxF
            scene_count = objs_ids.shape[0]
            objs_per_scene_count = None
        elif objs_ids.dim() == 2:
            # return BxOxNxF
            scene_count = objs_ids.shape[0]
            objs_per_scene_count = objs_ids.shape[1]
        else:
            raise NotImplementedError

        objs_ids_flatten = objs_ids.clone().reshape(-1)
        if mod == VERT_MODALITIES.NOCS and instance_deform is not None:
            objs_mod = [
                self.get_vert_mod(mod=mod, obj_id=obj_id, clone=clone,
                                  verts3d_deform=instance_deform.verts_deform[i])
                if obj_id >= 0
                else None
                for i, obj_id in enumerate(objs_ids_flatten)
            ]  #  if obj_id > 0
        else:
            objs_mod = [
                self.get_vert_mod(mod=mod, obj_id=obj_id, clone=clone)
                if obj_id >= 0
                else None
                for obj_id in objs_ids_flatten
            ]  #  if obj_id > 0

        if mod == VERT_MODALITIES.PT3D and instance_deform is not None:
            if not clone:
                objs_mod = [
                    obj_mod + instance_deform.verts_deform[i, : len(obj_mod)]
                    for i, obj_mod in enumerate(objs_mod)
                    if obj_mod is not None
                ]
            else:
                objs_mod = [
                    obj_mod + instance_deform.verts_deform[i, : len(obj_mod)].clone()
                    for i, obj_mod in enumerate(objs_mod)
                    if obj_mod is not None
                ]

        objs_mod = self.fill_objs_mod_none(objs_mod)

        if not padded:
            # N x F
            objs_mod = torch.cat(objs_mod, dim=0)
            return objs_mod, None
        else:
            # from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
            objs_lengths = torch.LongTensor([obj_mod.shape[0] for obj_mod in objs_mod])
            # NxB*OxF
            objs_mod = torch.nn.utils.rnn.pad_sequence(
                objs_mod,
                batch_first=False,
                padding_value=0,
                padding_side="right",
            )
            objs_mod = objs_mod.swapaxes(0, 1)
            objs_mod = objs_mod.clone()

            #  when is it essential to not pad sequence?,
            #  when deciding what to render and leaving out faces is more efficient
            if objs_ids.dim() == 2:
                # return BxOxNxF
                objs_mod = objs_mod.reshape(
                    scene_count,
                    objs_per_scene_count,
                    *objs_mod.shape[1:],
                )
                objs_lengths = objs_lengths.reshape(scene_count, objs_per_scene_count)
            else:
                # return BxNxF
                objs_mod = objs_mod.reshape(scene_count, *objs_mod.shape[1:])
                objs_lengths = objs_lengths.reshape(scene_count)

            return objs_mod, objs_lengths

    def get_face_mod(self, mod: FACE_MODALITIES, obj_id, clone=False):
        if mod == FACE_MODALITIES.IN_OBJ_ID:  # or mod == VERT_MODALITIES.IN_SCENE_ID:
            data = torch.arange(self.faces_counts[obj_id])
        elif mod == FACE_MODALITIES.VERTS_IN_OBJ_ID:
            data = self.faces[
                self.faces_counts_acc_from_0[obj_id] : self.faces_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
        elif mod == FACE_MODALITIES.VERTS_UVS_IN_OBJ_ID:
            data = self.faces_uvs[
                self.faces_counts_acc_from_0[obj_id] : self.faces_counts_acc_from_0[
                    obj_id + 1
                ]
            ]
        else:
            raise NotImplementedError
        if clone:
            data = data.clone()
        return data

    def get_face_mod_from_objs(self, mod, objs_ids=None, padded=False, clone=False):
        if objs_ids is None:
            objs_ids = torch.arange(len(self))

        if isinstance(objs_ids, list):
            objs_ids = torch.LongTensor(objs_ids).to(self.device)

        if mod == FACE_MODALITIES.IN_SCENE_ID:
            objs_mod, objs_lengths = self.get_face_mod_from_objs(
                mod=FACE_MODALITIES.IN_OBJ_ID,
                objs_ids=objs_ids,
                padded=True,
                clone=clone,
            )
            if objs_lengths.dim() == 2:
                from o3b.cv.select import cumsum_w0

                objs_lengths_acc_from_0 = cumsum_w0(input=objs_lengths, dim=1)
                objs_mod += objs_lengths_acc_from_0.reshape(
                    *(
                        objs_lengths_acc_from_0.shape
                        + ((1,) * (objs_mod.dim() - objs_lengths.dim()))
                    ),
                )
            if not padded:
                objs_mod = self.unpad_mod(objs_mod=objs_mod, objs_lengths=objs_lengths)
                return objs_mod, None
            else:
                return objs_mod, objs_lengths

        if (
            mod == FACE_MODALITIES.VERTS_IN_SCENE_ID
            or mod == FACE_MODALITIES.VERTS_IN_SCENES_ID
            or mod == FACE_MODALITIES.VERTS_UVS_IN_SCENE_ID
            or mod == FACE_MODALITIES.VERTS_UVS_IN_SCENES_ID
        ):
            objs_mod, objs_lengths = self.get_face_mod_from_objs(
                mod=FACE_MODALITIES.VERTS_IN_OBJ_ID,
                objs_ids=objs_ids,
                padded=True,
                clone=clone,
            )

            if (
                mod == FACE_MODALITIES.VERTS_IN_SCENE_ID
                or mod == FACE_MODALITIES.VERTS_IN_SCENES_ID
            ):
                objs_verts_counts = torch.LongTensor(self.verts_counts).to(
                    device=objs_ids.device,
                )[objs_ids]
                objs_verts_counts[objs_ids < 0] = 0
            else:
                objs_verts_counts = torch.LongTensor(self.verts_uvs_counts).to(
                    device=objs_ids.device,
                )[objs_ids]
                objs_verts_counts[objs_ids < 0] = 0

            from o3b.cv.select import cumsum_w0

            objs_verts_counts_acc_from_0 = None
            if (
                mod == FACE_MODALITIES.VERTS_IN_SCENE_ID
                or mod == FACE_MODALITIES.VERTS_UVS_IN_SCENE_ID
            ) and objs_lengths.dim() == 2:
                objs_verts_counts_acc_from_0 = cumsum_w0(input=objs_verts_counts, dim=1)
            elif (
                mod == FACE_MODALITIES.VERTS_IN_SCENES_ID
                or mod == FACE_MODALITIES.VERTS_UVS_IN_SCENES_ID
            ):
                objs_verts_counts_acc_from_0 = cumsum_w0(
                    objs_verts_counts.flatten(),
                    dim=0,
                )
                objs_verts_counts_acc_from_0 = objs_verts_counts_acc_from_0.reshape(
                    *objs_verts_counts.shape,
                )
            if objs_verts_counts_acc_from_0 is not None:
                objs_mod += objs_verts_counts_acc_from_0.reshape(
                    *(
                        objs_verts_counts_acc_from_0.shape
                        + ((1,) * (objs_mod.dim() - objs_lengths.dim()))
                    ),
                )
            if not padded:
                objs_mod = self.unpad_mod(objs_mod=objs_mod, objs_lengths=objs_lengths)
                return objs_mod, None
            else:
                return objs_mod, objs_lengths

        if objs_ids.dim() == 1:
            # return BxNxF
            scene_count = objs_ids.shape[0]
            objs_per_scene_count = None
        elif objs_ids.dim() == 2:
            # return BxOxNxF
            scene_count = objs_ids.shape[0]
            objs_per_scene_count = objs_ids.shape[1]
        else:
            raise NotImplementedError

        objs_ids_flatten = objs_ids.clone().reshape(-1)
        objs_mod = [
            self.get_face_mod(mod=mod, obj_id=obj_id, clone=clone)
            if obj_id >= 0
            else None
            for obj_id in objs_ids_flatten
        ]
        objs_mod = self.fill_objs_mod_none(objs_mod)

        if not padded:
            objs_mod = torch.cat(objs_mod, dim=0)
            return objs_mod, None
        else:
            # from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
            objs_lengths = torch.LongTensor([obj_mod.shape[0] for obj_mod in objs_mod])
            objs_mod = torch.nn.utils.rnn.pad_sequence(
                objs_mod,
                batch_first=False,
                padding_value=0,
                padding_side="right",
            )
            objs_mod = objs_mod.swapaxes(0, 1)
            objs_mod = objs_mod.clone()

            #  when is it essential to not pad sequence?,
            #  when deciding what to render and leaving out faces is more efficient
            if objs_ids.dim() == 2:
                # return BxOxNxF
                objs_mod = objs_mod.reshape(
                    scene_count,
                    objs_per_scene_count,
                    *objs_mod.shape[1:],
                )
                objs_lengths = objs_lengths.reshape(scene_count, objs_per_scene_count)
            else:
                # return BxNxF
                objs_mod = objs_mod.reshape(scene_count, *objs_mod.shape[1:])
                objs_lengths = objs_lengths.reshape(scene_count)

            return objs_mod, objs_lengths

    def get_feats_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_feats_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_verts_padded_mask_with_mesh_id(self, mesh_id, device="cpu"):
        return self.get_tensor_verts_with_pad(
            tensor=torch.ones(
                size=(self.verts_counts[mesh_id],),
                dtype=torch.bool,
                device=device,
            ),
            mesh_id=mesh_id,
        )

    def get_faces_padded_mask_with_mesh_id(self, mesh_id, device="cpu"):
        return self.get_tensor_verts_with_pad(
            tensor=torch.ones(
                size=(self.faces_counts[mesh_id],),
                dtype=torch.bool,
                device=device,
            ),
            mesh_id=mesh_id,
        )

    def get_verts_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_verts_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_verts_ncds_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_verts_ncds_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_rgb_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_rgb_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    # get_rgb_padded_with_mesh_id
    # def to(self, device):
    #    if self.device != device:
    #        self.verts = [v.to(device=device) for v in self.verts]
    #        self.faces = [f.to(device=device) for f in self.faces]
    #        if self.rgb is not None:
    #            self.rgb = [i.to(device=device) for i in self.rgb]
    #        if self.feats is not None:
    #            self.feats = [f.to(device=device) for f in self.feats]
    #            self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    #           self.device = device
    def set_feats_cat_with_pad(self, feats):
        vts_ct_max = self.verts_counts_max
        device = self.device
        self.feats = torch.nn.Parameter(
            torch.cat(
                [
                    feats[i * vts_ct_max : i * vts_ct_max + self.verts_counts[i]].to(
                        device=device,
                    )
                    for i in range(len(self))
                ],
                dim=0,
            ),
            requires_grad=True,
        )
        self.feats_from_faces = torch.nn.Parameter(
            torch.cat(
                [
                    self.get_feats_with_mesh_id(mesh_id)[
                        self.get_faces_with_mesh_id(mesh_id)
                    ]
                    for mesh_id in range(len(self))
                ],
                dim=0,
            ),
        )

    def set_feats_cat(self, feats):
        self.feats = torch.nn.Parameter(feats, requires_grad=True)
        self.feats_from_faces = torch.nn.Parameter(
            torch.cat(
                [
                    self.get_feats_with_mesh_id(mesh_id)[
                        self.get_faces_with_mesh_id(mesh_id)
                    ]
                    for mesh_id in range(len(self))
                ],
                dim=0,
            ),
        )

    def get_verts_ncds_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_verts_ncds_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_rgb_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_rgb_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_verts_uvs_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_verts_uvs_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_verts_stacked_with_mesh_ids(self, mesh_ids=None, instance_deform=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        if instance_deform is None:
            return torch.stack(
                [self.get_verts_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
                dim=0,
            )
        else:
            return torch.stack(
                [
                    self.get_verts_padded_with_mesh_id(mesh_id)
                    + instance_deform.verts_deform[m]
                    for m, mesh_id in enumerate(mesh_ids)
                ],
                dim=0,
            )

    def get_verts_stacked_mask_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_verts_padded_mask_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_feats_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_feats_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_feats_coarse_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        return torch.stack(
            [self.get_feats_coarse_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_rgbs_uvs_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.rgbs_uvs[mesh_id] for mesh_id in mesh_ids],
            dim=0,
        )

    def get_faces_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_faces_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_faces_uvs_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_faces_uvs_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_faces_stacked_mask_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_faces_padded_mask_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    # def add_feats_cat(self, feats):
    #    raise Not I
    #    self.feats = [feats[self.verts_counts_acc_from_0[i] : self.verts_counts_acc_from_0[i+1]].to(device=self.device) for i in range(len(self))]
    #    self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    # def add_feats
    # def verts_stacked(self, mesh_ids: list=None):
    #    if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))

    # verts_stacked = torch.zeros(size=(len(mesh_ids), self.verts_counts_max(), 3), device=self.device)
    # for mesh_id in mesh_ids:
    #    verts_stacked[mesh_id, : len(self.verts[mesh_id])] = self.verts[mesh_id]

    #    return torch.stack([self.verts[mesh_id] for mesh_id in mesh_ids], dim=0)

    # def feats_cat(self, mesh_ids: list=None):
    #    if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))
    #
    #    return torch.cat([self.feats[mesh_id] for mesh_id in mesh_ids], dim=0)

    # def get_feats_ids_stacked(self, mesh_ids: list=None):
    #    if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.device
    #    return torch.stack([torch.arange(mesh_id*self.verts_counts_max, (mesh_id+1)*self.verts_counts_max, device=device) for mesh_id in mesh_ids], dim=0)

    # def get_verts_and_noise_ids_cat(self, mesh_ids: list=None, count_noise_ids=5):
    #   if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.device
    #    noise_ids = torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device) * self.verts_counts_acc_from_0[-1]
    #    verts_ids = [torch.arange(self.verts_counts_acc_from_0[mesh_id], self.verts_counts_acc_from_0[mesh_id+1], device=device) for mesh_id in mesh_ids]
    #    return torch.cat([torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))], dim=0)

    def get_verts_and_noise_ids_stacked(self, mesh_ids: list = None, count_noise_ids=5):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        device = self.device
        noise_ids = (
            torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device)
            * self.verts_counts_acc_from_0[-1]
        )
        verts_ids = [
            torch.arange(
                self.verts_counts_acc_from_0[mesh_id],
                self.verts_counts_acc_from_0[mesh_id] + self.verts_counts_max,
                device=device,
            )
            for mesh_id in mesh_ids
        ]
        return torch.stack(
            [torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))],
            dim=0,
        )

    def get_verts_and_noise_ids_stacked_without_acc(
        self,
        mesh_ids: list = None,
        count_noise_ids=5,
    ):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        device = self.device
        noise_ids = (
            torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device)
            * self.verts_counts_max
        )
        verts_ids = [
            torch.arange(0, self.verts_counts_max, device=device)
            for mesh_id in mesh_ids
        ]
        return torch.stack(
            [torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))],
            dim=0,
        )

    def face_normals3d(
        self,
        meshes_ids: Union[torch.LongTensor, List] = None,
        instance_deform=None,
    ):
        """
        Args:
            meshes_ids (Union[torch.LongTensor, List]): len(mesh_ids) == B
        Returns:
            normals3d (torch.Tensor): BxNx3
        """
        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        if isinstance(meshes_ids, List):
            meshes_ids = torch.LongTensor(meshes_ids)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone()

        _faces = self.get_faces_cat_with_mesh_ids(
            mesh_ids=meshes_ids,
            use_global_verts_ids=True,
        )
        _verts = self.get_verts_cat_with_mesh_ids(
            mesh_ids=meshes_ids,
            instance_deform=instance_deform,
        )

        _faces_normals = torch.cross(
            _verts[_faces][:, 2] - _verts[_faces][:, 1],
            _verts[_faces][:, 0] - _verts[_faces][:, 1],
            dim=1,
        )
        return _faces_normals

    def normals3d(
        self,
        meshes_ids: Union[torch.LongTensor, List] = None,
        instance_deform=None,
    ):
        """
        Args:
            meshes_ids (Union[torch.LongTensor, List]): len(mesh_ids) == B
        Returns:
            normals3d (torch.Tensor): BxNx3
        """

        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        if isinstance(meshes_ids, List):
            meshes_ids = torch.LongTensor(meshes_ids)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone()

        pt3dmeshes = self.get_pt3dmeshes_with_deform(
            objects_ids=meshes_ids,
            instance_deform=instance_deform,
        )
        _verts_normals_pt3d = pt3dmeshes.verts_normals_padded()
        return pt3dmeshes.verts_normals_padded()
        #
        # _faces_normals = self.face_normals3d(meshes_ids=meshes_ids, instance_deform=instance_deform)
        #
        # _verts = self.get_verts_cat_with_mesh_ids(mesh_ids=meshes_ids, instance_deform=instance_deform)
        # _faces = self.get_faces_cat_with_mesh_ids(mesh_ids=meshes_ids, use_global_verts_ids=True )
        #
        # _verts_normals = torch.zeros_like(_verts)
        #
        # # NOTE: this is already applying the area weighting as the magnitude
        # # of the cross product is 2 x area of the triangle.
        # _verts_normals = _verts_normals.index_add(
        #     0, _faces[:, 0], _faces_normals
        # )
        # _verts_normals = _verts_normals.index_add(
        #     0, _faces[:, 1], _faces_normals
        # )
        # _verts_normals = _verts_normals.index_add(
        #     0, _faces[:, 2], _faces_normals
        # )
        #
        # _verts_normals = torch.nn.functional.normalize(
        #     _verts_normals, eps=1e-6, dim=1
        # )
        #
        # _verts_normals = _verts_normals
        #
        # _verts_normals = self.get_tensor_verts_padded_from_packed(_verts_normals, meshes_ids)
        # return _verts_normals

    def get_tensor_verts_padded_from_packed(
        self,
        tensor_verts_packed,
        meshes_ids: Union[torch.LongTensor, List] = None,
    ):
        """
        Args:
            meshes_ids (Union[torch.LongTensor, List]): len(mesh_ids) == B
            tensor_verts_packed (torch.Tensor): Nx3
        Returns:
            tensor_verts_padded (torch.Tensor): BxN'x3
        """

        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        if isinstance(meshes_ids, List):
            meshes_ids = torch.LongTensor(meshes_ids)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone()

        B = len(meshes_ids)
        tensor_verts_padded = torch.zeros(
            size=(B, self.verts_counts_max, *tensor_verts_packed.shape[1:]),
            dtype=tensor_verts_packed.dtype,
            device=tensor_verts_packed.device,
        )
        index_start = 0
        for b in range(B):
            mesh_id = meshes_ids[b]
            mesh_verts_count = self.verts_counts[mesh_id]
            tensor_verts_padded[b, :mesh_verts_count] = tensor_verts_packed[
                index_start : index_start + mesh_verts_count
            ]
            index_start += mesh_verts_count

        return tensor_verts_padded

    def verts2d(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        mesh_ids: Union[torch.LongTensor, List],
        down_sample_rate=1.0,
        broadcast_batch_and_cams=False,
    ):
        """
        Args:
            cams_tform4x4_obj (torch.Tensor): Bx4x4
            cams_intr4x4 (torch.Tensor): Bx4x4
            imgs_sizes (torch.Tensor): Bx2 / 2 (height, width)
            mesh_ids (list): len(mesh_ids) == B
        Returns:
            verts2d (torch.Tensor): BxNx2
        """
        # meshes_count = mesh_ids.shape[0]
        # cams_count = cams_tform4x4_obj.shape[0]

        if isinstance(mesh_ids, List):
            mesh_ids = torch.LongTensor(mesh_ids)

        if isinstance(mesh_ids, torch.LongTensor):
            mesh_ids = mesh_ids.clone()

        meshes_count = mesh_ids.shape[0]
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f"Set `cams_tform4x4_obj.dim()` must be 3 or 4")

        if broadcast_batch_and_cams:
            mesh_ids = mesh_ids
            if cams_tform4x4_obj.dim() == 3:
                cams_tform4x4_obj = cams_tform4x4_obj[None, :]
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_tform4x4_obj = cams_tform4x4_obj.expand(
                meshes_count,
                cams_count,
                4,
                4,
            ).reshape(-1, 4, 4)
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(
                -1,
                4,
                4,
            )
            mesh_ids = mesh_ids[:, None].expand(meshes_count, cams_count).reshape(-1)
            # render_count = meshes_count * cams_count
        else:
            if meshes_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and meshes",
                )

        B = cams_tform4x4_obj.shape[0]
        # if imgs_sizes.dim() == 2:
        #    imgs_sizes = imgs_sizes[None,].expand(B, imgs_sizes.shape[0], imgs_sizes.shape[1])
        cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)
        verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=mesh_ids)
        verts2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])
        mask_verts_vsbl = self.render(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=mesh_ids,
            modalities=PROJECT_MODALITIES.MASK_VERTS_VSBL,
            down_sample_rate=down_sample_rate,
        )
        mask_verts_vsbl *= (verts2d <= (imgs_sizes[None, None] - 1)).all(dim=-1)
        mask_verts_vsbl *= (verts2d >= 0).all(dim=-1)
        verts2d[~mask_verts_vsbl] = 0
        # verts2d.clamp()

        if broadcast_batch_and_cams:
            verts2d = verts2d.reshape(meshes_count, cams_count, *verts2d.shape[1:])
            mask_verts_vsbl = mask_verts_vsbl.reshape(
                meshes_count,
                cams_count,
                *mask_verts_vsbl.shape[1:],
            )

        verts2d /= down_sample_rate

        return verts2d, mask_verts_vsbl

    def show(
        self,
        fpath: Path = None,
        return_visualization=False,
        viewpoints_count=1,
        meshes_add_translation=True,
    ):
        from o3b.cv.visual.show import show_scene

        return show_scene(
            meshes=self.get_meshes_with_ids(clone=True),
            fpath=fpath,
            return_visualization=return_visualization,
            viewpoints_count=viewpoints_count,
            meshes_add_translation=meshes_add_translation,
        )

    """
    def show(self, pts3d=[], meshes_ids=None):
        from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
        from pytorch3d.structures import Pointclouds
        if meshes_ids is None:
            meshes_ids = list(range(len(self.pt3dmeshes)))
        pcls = Pointclouds(points=pts3d)
        verts = Pointclouds(points=self.get_verts_stacked_with_mesh_ids(mesh_ids=meshes_ids))
        fig = plot_scene({
            "Meshes":
                {
                    # **{f"mesh{i + 1}": self.pt3dmeshes[i] for i in meshes_ids},
                    **{f"verts{i + 1}": verts[i] for i in meshes_ids},
                    **{f"pcl{i + 1}": pcls[i] for i in range(len(pcls))}
                }
        }, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                                   showaxeslabels=True, showticklabels=True))
        fig.show()
        input('bla')
    """

    def del_pre_rendered(self):
        self.pre_rendered_modalities.clear()
        torch.cuda.empty_cache()

    def visualize_mod(
        self,
        mod=VERT_MODALITIES.PT3D_NCDS,
        fpath=f"meshes.webm",
        H=1080,
        W=1080,
        viewpoints_count=16,
        spiral=True,
    ):
        from o3b.cv.visual.show import get_default_camera_intrinsics_from_img_size
        from o3b.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count

        # V = 3

        cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
            W=W,
            H=H,
            device=self.device,
        )
        cam_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(
            viewpoints_count=viewpoints_count,
            dist=4.0 * float(self.get_range1d()),
            spiral=spiral,
            device=self.device,
        )

        mod_rendered = self.render(
            modalities=mod,
            imgs_sizes=(H, W),
            cams_tform4x4_obj=cam_tform4x4_obj,
            cams_intr4x4=cam_intr4x4,
            broadcast_batch_and_cams=True,
        )
        # from o3b.cv.visual.show import show_imgs, save_video
        # show_imgs(mod_rendered)
        from o3b.cv.visual.video import save_video

        save_video(
            imgs=mod_rendered.permute(1, 2, 3, 0, 4).reshape(
                *mod_rendered.shape[1:-1],
                -1,
            ),
            fpath=fpath,
        )

    def visualize(self, pcl=None):
        from o3b.cv.visual.show import show_imgs
        import numpy as np

        device = self.device
        cxy = 250.0
        fxy = 500.0
        down_sample_rate = 2.0
        imgs_sizes = torch.LongTensor([512, 512]).to(device=device)

        azim = torch.linspace(
            start=eval("-np.pi / 2"),
            end=eval("np.pi / 2"),
            steps=5,
        ).to(
            device=device,
        )  # 12
        elev = torch.linspace(
            start=eval("np.pi / 4"),
            end=eval("np.pi / 4"),
            steps=1,
        ).to(
            device=device,
        )  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
        theta = torch.linspace(
            start=eval("0."),
            end=eval("0."),
            steps=1,
        ).to(
            device=device,
        )  # -torch.pi / 6, end=torch.pi / 6, steps=3

        # dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
        #    device=self.device)
        dist = torch.linspace(start=1.0, end=1.0, steps=1).to(device=device)

        azim_shape = azim.shape
        elev_shape = elev.shape
        theta_shape = theta.shape
        dist_shape = dist.shape
        in_shape = azim_shape + elev_shape + theta_shape + dist_shape
        azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
        elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
        theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
        dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
        from o3b.cv.geometry.transform import transf4x4_from_spherical

        cams_tform4x4_obj = transf4x4_from_spherical(
            azim=azim,
            elev=elev,
            theta=theta,
            dist=dist,
        )

        T = cams_tform4x4_obj.shape[0]
        M = len(self)
        # M x T x 4 x 4
        pre_rendered_cams_tform4x4_obj = (
            cams_tform4x4_obj[None, :]
            .expand(
                M,
                T,
                *cams_tform4x4_obj[0].shape,
            )
            .clone()
        )
        pre_rendered_cams_tform4x4_obj[:, :, :3, 3] = 0.0
        pre_rendered_meshes_size = (
            self.get_verts_stacked_with_mesh_ids().flatten(1).max(dim=-1)[0]
        )
        pre_rendered_meshes_dist = (pre_rendered_meshes_size * fxy) / (
            500.0 * 0.8 - cxy
        )  # u = (x / z) * fx + cx  -> z = (fx * x) / (u - cx)
        pre_rendered_meshes_dist = pre_rendered_meshes_dist[:, None].expand(M, T)
        pre_rendered_cams_tform4x4_obj[:, :, 2, 3] = pre_rendered_meshes_dist

        # 1 x 1 x 4 x 4
        pre_rendered_cams_intr4x4 = (
            torch.eye(4)[None, None].to(device=device).expand(M, 1, 4, 4)
        )
        pre_rendered_cams_intr4x4[:, :, 0, 0] = fxy
        pre_rendered_cams_intr4x4[:, :, 1, 1] = fxy
        pre_rendered_cams_intr4x4[:, :, :2, 2] = cxy

        pre_rendered_meshes_ids = torch.arange(M).to(device=device)
        rendering = []
        for m in range(M):
            rendering.append(
                self.render(
                    cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj[m : m + 1],
                    cams_intr4x4=pre_rendered_cams_intr4x4[m : m + 1],
                    imgs_sizes=imgs_sizes,
                    objects_ids=pre_rendered_meshes_ids[m : m + 1],
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    broadcast_batch_and_cams=True,
                    down_sample_rate=down_sample_rate,
                ),
            )
        rendering = torch.cat(rendering, dim=0)

        if pcl is not None:
            pxl2d_pre_rendered = (
                proj3d2d_broadcast(
                    pts3d=pcl[:, None, None],
                    proj4x4=tform4x4_broadcast(
                        pre_rendered_cams_intr4x4,
                        pre_rendered_cams_tform4x4_obj,
                    ),
                )
                / down_sample_rate
            )
            from o3b.cv.visual.mask import mask_from_pxl2d

            pxl2d_mask = mask_from_pxl2d(
                pxl2d=pxl2d_pre_rendered,
                dim_pxl=3,
                dim_pts=0,
                H=int(imgs_sizes[0] // down_sample_rate),
                W=int(imgs_sizes[1] // down_sample_rate),
            )
            rendering[pxl2d_mask[:, :, None, :, :].repeat(1, 1, 3, 1, 1)] = 1.0
            # rendering[:, :, :, ]
            # sample_pxl2d_grid(rendering.reshape(-1, C, H, W),
            #                  pxl2d=pxl2d_pre_rendered.reshape(-1, H, W, 2)).reshape(B, T, C, H, W)
        return show_imgs(rendering)

    def get_pre_rendered_feats(
        self,
        modality: PROJECT_MODALITIES,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        meshes_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
    ):
        if modality not in self.pre_rendered_modalities.keys():
            cxy = 250.0
            fxy = 500.0
            T = cams_tform4x4_obj.shape[1]
            M = len(self)
            # M x T x 4 x 4
            pre_rendered_cams_tform4x4_obj = (
                cams_tform4x4_obj[:1, :].expand(M, *cams_tform4x4_obj[0].shape).clone()
            )
            pre_rendered_cams_tform4x4_obj[:, :, :3, 3] = 0.0
            pre_rendered_meshes_size = (
                self.get_verts_stacked_with_mesh_ids().flatten(1).max(dim=-1)[0]
            )
            pre_rendered_meshes_dist = (pre_rendered_meshes_size * fxy) / (
                500.0 * 0.7 - cxy
            )  # u = (x / z) * fx + cx  -> z = (fx * x) / (u - cx)
            pre_rendered_meshes_dist = pre_rendered_meshes_dist[:, None].expand(M, T)
            pre_rendered_cams_tform4x4_obj[:, :, 2, 3] = pre_rendered_meshes_dist

            # 1 x 1 x 4 x 4
            pre_rendered_cams_intr4x4 = (
                cams_intr4x4[:1, :1].clone().expand(M, 1, *cams_intr4x4[0, 0].shape)
            )
            pre_rendered_cams_intr4x4[:, :, 0, 0] = fxy
            pre_rendered_cams_intr4x4[:, :, 1, 1] = fxy
            pre_rendered_cams_intr4x4[:, :, :2, 2] = cxy

            pre_rendered_meshes_ids = torch.arange(M).to(device=meshes_ids.device)

            rendering = []
            for m in range(M):
                rendering.append(
                    self.render(
                        cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj[m : m + 1],
                        cams_intr4x4=pre_rendered_cams_intr4x4[m : m + 1],
                        imgs_sizes=imgs_sizes,
                        objects_ids=pre_rendered_meshes_ids[m : m + 1],
                        modalities=modality,
                        broadcast_batch_and_cams=broadcast_batch_and_cams,
                        down_sample_rate=down_sample_rate,
                    ),
                )
            rendering = torch.cat(rendering, dim=0)

            self.pre_rendered_modalities[modality] = Meshes.PreRendered(
                cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj,
                cams_intr4x4=pre_rendered_cams_intr4x4,
                imgs_sizes=imgs_sizes,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                meshes_ids=pre_rendered_meshes_ids,
                down_sample_rate=down_sample_rate,
                rendering=rendering,
            )

        else:
            assert (
                self.pre_rendered_modalities[modality].cams_tform4x4_obj[
                    meshes_ids,
                    :,
                    :3,
                    :3,
                ]
                == cams_tform4x4_obj[:, :, :3, :3]
            ).all()
            assert (
                self.pre_rendered_modalities[modality].imgs_sizes == imgs_sizes
            ).all()
            assert (
                self.pre_rendered_modalities[modality].broadcast_batch_and_cams
                == broadcast_batch_and_cams
            )
            assert (
                self.pre_rendered_modalities[modality].down_sample_rate
                == down_sample_rate
            )

        pre_rendered_feats = self.pre_rendered_modalities[modality].rendering[
            meshes_ids
        ]
        pre_rendered_cam_intr4x4 = self.pre_rendered_modalities[modality].cams_intr4x4[
            meshes_ids
        ]
        pre_rendered_cam_tform4x4_obj = self.pre_rendered_modalities[
            modality
        ].cams_tform4x4_obj[meshes_ids]

        B, T, C, H, W = pre_rendered_feats.shape

        pxl2d_cams = (
            get_pxl2d(
                H=H,
                W=W,
                dtype=pre_rendered_feats.dtype,
                device=pre_rendered_feats.device,
                B=None,
            )
            * self.pre_rendered_modalities[modality].down_sample_rate
        )
        pxl2d_cams = pxl2d_cams.expand(*pre_rendered_feats.shape[:2], *pxl2d_cams.shape)
        pts3d_homog_cams = (
            transf3d_broadcast(
                pts3d=add_homog_dim(pxl2d_cams, dim=4),
                transf4x4=cams_intr4x4.pinverse()[:, :, None, None],
            )
            * cams_tform4x4_obj[:, :, 2, 3, None, None, None]
        )
        pts3d_pre_rendered = transf3d_broadcast(
            pts3d=pts3d_homog_cams,
            transf4x4=tform4x4(
                pre_rendered_cam_tform4x4_obj,
                inv_tform4x4(cams_tform4x4_obj),
            )[:, :, None, None],
        )
        pxl2d_pre_rendered = (
            proj3d2d_broadcast(
                pts3d=pts3d_pre_rendered,
                proj4x4=pre_rendered_cam_intr4x4[:, :, None, None],
            )
            / self.pre_rendered_modalities[modality].down_sample_rate
        )
        cams_features = sample_pxl2d_grid(
            pre_rendered_feats.reshape(-1, C, H, W),
            pxl2d=pxl2d_pre_rendered.reshape(-1, H, W, 2),
        ).reshape(B, T, C, H, W)

        return cams_features

    def get_pre_rendered_masks(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        meshes_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
    ):
        assert self.pre_rendered_masks_verts_vsbl_cams_tform4x4_obj == cams_tform4x4_obj
        assert self.pre_rendered_masks_verts_vsbl_cams_intr4x4 == cams_intr4x4
        assert self.pre_rendered_masks_verts_vsbl_imgs_sizes == imgs_sizes
        assert self.pre_rendered_masks_verts_vsbl_meshes_ids == meshes_ids
        assert (
            self.pre_rendered_masks_verts_vsbl_broadcast_batch_and_cams
            == broadcast_batch_and_cams
        )
        assert self.pre_rendered_masks_verts_vsbl_down_sample_rate == down_sample_rate

        if self.pre_rendered_feats is None:
            self.pre_rendered_masks_verts_vsbl_cams_tform4x4_obj = cams_tform4x4_obj
            self.pre_rendered_masks_verts_vsbl_cams_intr4x4 = cams_intr4x4
            self.pre_rendered_masks_verts_vsbl_imgs_sizes = imgs_sizes
            self.pre_rendered_feats_meshes_ids = meshes_ids
            self.pre_rendered_feats_broadcast_batch_and_cams = broadcast_batch_and_cams
            self.pre_rendered_down_sample_rate = down_sample_rate
            self.pre_rendered_feats = self.render(
                cams_tform4x4_obj=self.pre_rendered_feats_cams_tform4x4_obj,
                cams_intr4x4=self.pre_rendered_feats_cams_intr4x4,
                imgs_sizes=self.pre_rendered_feats_imgs_sizes,
                objects_ids=self.pre_rendered_feats_meshes_ids,
                modalities=PROJECT_MODALITIES.MASK_VERTS_VSBL,
                broadcast_batch_and_cams=self.pre_rendered_feats_broadcast_batch_and_cams,
                down_sample_rate=self.pre_rendered_down_sample_rate,
            )

        return self.pre_rendered_feats

    def render_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
        instance_deform: OD3D_Meshes_Deform = None,
        detach_objects=False,
        detach_deform=False,
        rgb_diffusion_alpha=0.0,
        rgb_bg=None,
        rgb_light_env=None,
        fglut_fpath = "./data/light_bsdf/bsdf_256_256.bin",
        obj_tform4x4_objs=None,
        zfar=1e3,
        znear=1e-2,
        opengl=False,
        **kwargs,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) or (B, O) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            add_clutter: bool, determines wether to add clutter features to the rendered features.
            obj_tform4x4_objs: (B, O, 4, 4,) or (O, 4, 4)
            opengl: if True, cams_tform4x4_obj is given in the OpenGL convention
                (-Z forward, +Y up) and is converted to the OpenCV convention
                (+Z forward, +Y down) the rasterizers expect.
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        device = cams_tform4x4_obj.device
        dtype = cams_tform4x4_obj.dtype
        render_count = cams_tform4x4_obj.shape[0]

        if opengl:
            # OpenGL (-Z forward, +Y up) → OpenCV (+Z forward, +Y down): flip Y and Z.
            _gl2cv = torch.diag(
                torch.tensor([1.0, -1.0, -1.0, 1.0], device=device, dtype=dtype)
            )
            cams_tform4x4_obj = (
                _gl2cv[None].expand(cams_tform4x4_obj.shape).bmm(cams_tform4x4_obj)
            )

        if self.face_blend_type == FACE_BLEND_TYPE.HARD:
            blur_radius = 0.0
            faces_per_pixel = 1
            cull_backfaces = True
        else:
            blur_radius = 2.0 * self.face_opacity_face_sdf_sigma
            faces_per_pixel = self.face_blend_count
            cull_backfaces = False

        focal_length = torch.stack(
            [cams_intr4x4[..., 0, 0], cams_intr4x4[..., 1, 1]],
            dim=-1,
        )
        principal_point = torch.stack(
            [cams_intr4x4[..., 0, 2], cams_intr4x4[..., 1, 2]],
            dim=-1,
        )

        if self.rasterizer == RASTERIZER.PYTORCH3D:
            from pytorch3d.io import IO
            from pytorch3d.io import load_ply
            from pytorch3d.io import save_ply
            from pytorch3d.renderer import MeshRasterizer
            from pytorch3d.renderer import RasterizationSettings
            from pytorch3d.renderer.cameras import PerspectiveCameras
            from pytorch3d.renderer.mesh import TexturesUV as PT3DTexturesUV
            from pytorch3d.renderer.mesh import TexturesVertex as PT3DTexturesVertex
            from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
            from pytorch3d.structures import packed_to_list as pt3d_packed_to_list

            # cams_tform4x4_obj = cams_tform4x4_obj.repeat_interleave(num_meshes, dim=0)
            # cams_intr4x4 = cams_intr4x4.repeat_interleave(num_meshes, dim=0)
            # imgs_sizes = imgs_sizes.repeat_interleave(num_meshes, dim=0)
            t3d_tform_pscl3d = torch.Tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ).to(device=device, dtype=dtype)
            # t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cams_tform4x4_obj)
            t3d_cam_tform_obj = (
                t3d_tform_pscl3d[None]
                .expand(cams_tform4x4_obj.shape)
                .bmm(cams_tform4x4_obj)
            )
            R = t3d_cam_tform_obj[..., :3, :3].permute(0, 2, 1)
            t = t3d_cam_tform_obj[..., :3, 3]

            cameras = PerspectiveCameras(
                device=device,
                R=R,
                T=t,
                focal_length=focal_length,
                principal_point=principal_point,
                in_ndc=False,
                image_size=imgs_sizes[None,].expand(render_count, 2),
            )

            # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
            #  znear=0.001, zfar=100000, fov=10
            # Define the settings for rasterization and shading. Here we set the output image to be of size
            # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
            # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
            # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
            # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
            # the difference between naive and coarse-to-fine rasterization.

            raster_settings = RasterizationSettings(
                image_size=[int(imgs_sizes[0]), int(imgs_sizes[1])],
                blur_radius=blur_radius,  # 2 * sigma
                faces_per_pixel=faces_per_pixel,
                bin_size=None,
                max_faces_per_bin=None,
                perspective_correct=self.pt3d_raster_perspective_correct,
                cull_backfaces=cull_backfaces,  # cull_backfaces=True
            )

            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            )

            pt3dmeshes = self.get_pt3dmeshes_with_deform(
                objects_ids=objects_ids,
                instance_deform=instance_deform,
                detach_objects_verts=detach_objects,
                detach_deform_verts=detach_deform,
            )
            fragments = rasterizer(pt3dmeshes)
        elif self.rasterizer == RASTERIZER.NVDIFFRAST:
            ndc_min = -1
            ndc_max = 1
            height = int(imgs_sizes[0])
            width = int(imgs_sizes[1])

            cams_persp4x4 = cams_intr4x4.clone()
            cams_persp4x4[:, 0, 2] = -cams_persp4x4[:, 0, 2]
            cams_persp4x4[:, 1, 2] = -cams_persp4x4[:, 1, 2]
            cams_persp4x4[:, 2, 2] = 0.0
            cams_persp4x4[:, 2, 3] = 1.0
            cams_persp4x4[:, 3, 2] = 1.0
            cams_persp4x4[:, 3, 3] = 0.0

            cams_persp4x4[:, 1, 2] = -(height + cams_persp4x4[:, 1, 2])
            # top = height / 2
            # bottom = -top
            # right = width / 2
            # left = -right

            top = 0
            bottom = max(height, 1)  # note: ensures no division by zero
            right = max(width, 1)  # note: ensures no division by zero
            left = 0

            zfar = 10000  # (cams_tform4x4_obj[..., 2, 3].max()) * 10.

            tx = -(right + left) / (right - left)
            ty = -(top + bottom) / (top - bottom)

            if ndc_min == -1 and ndc_max == 1:
                U = -2.0 * znear * zfar / (zfar - znear)
                V = -(zfar + znear) / (zfar - znear)
            elif ndc_min == 0 and ndc_max == 1:
                U = (znear * zfar) / (znear - zfar)
                V = zfar / (zfar - znear)
            elif ndc_min == 1 and ndc_max == 0:
                U = (znear * zfar) / (zfar - znear)
                V = znear / (zfar - znear)
            else:
                raise NotImplementedError(
                    "Perspective Projection does not support NDC range of "
                    f"[{ndc_min}, {ndc_max}]",
                )

            # The matrix is non differentiable, as NDC coordinates are a fixed standard set by the graphics api
            ndc_mat = torch.Tensor(
                [
                    [2.0 / (right - left), 0.0, 0.0, -tx],
                    [0.0, 2.0 / (top - bottom), 0.0, -ty],
                    [0.0, 0.0, U, V],
                    [0.0, 0.0, 0.0, -1.0],
                ],
            )[
                None,
            ].to(
                device=device,
            )

            # Squash matrices together to form complete perspective projection matrix which maps to NDC coordinates
            #
            from o3b.cv.visual.show import OPEN3D_CAM_TFORM_CAM

            cams_proj4x4 = (ndc_mat @ cams_persp4x4) @ (
                    OPEN3D_CAM_TFORM_CAM.clone()[None,].to(device=device)
                    @ cams_tform4x4_obj
            )

            verts, verts_counts = self.get_vert_mod_from_objs(
                mod=VERT_MODALITIES.PT3D,
                objs_ids=objects_ids,
                instance_deform=instance_deform,
                padded=True,
            )
            faces, faces_counts = self.get_face_mod_from_objs(
                mod=FACE_MODALITIES.VERTS_IN_SCENES_ID,
                objs_ids=objects_ids,
                padded=True,
            )
            from o3b.cv.select import cumsum_w0

            if objects_ids.dim() == 1:
                scenes_faces_counts = faces_counts
            elif objects_ids.dim() == 2:
                scenes_faces_counts = faces_counts.sum(dim=-1)
            else:
                raise NotImplementedError

            scenes_faces_counts_cumsum_w0 = cumsum_w0(scenes_faces_counts, dim=-1)

            ranges = torch.stack(
                [scenes_faces_counts_cumsum_w0, scenes_faces_counts],
                dim=1,
            )

            if objects_ids.dim() == 1:
                # 1x None for verts
                if obj_tform4x4_objs is not None:
                    cams_proj4x4 = cams_proj4x4[:, None] @ obj_tform4x4_objs
                else:
                    cams_proj4x4 = cams_proj4x4[:, None]
                verts_cam = (cams_proj4x4 @ add_homog_dim(verts, dim=-1)[..., None])[
                    ...,
                    0,
                ]
            elif objects_ids.dim() == 2:
                # 1x None for objects and 1x None for verts
                if obj_tform4x4_objs is not None:
                    if obj_tform4x4_objs.dim() == 3:
                        cams_proj4x4 = (
                            cams_proj4x4[:, None, None]
                            @ obj_tform4x4_objs[None, :, None]
                        )
                        logger.info(
                            "should not happen that obj_tform4x4_objs.dim() == 3",
                        )
                    elif obj_tform4x4_objs.dim() == 4:
                        cams_proj4x4 = (
                            cams_proj4x4[:, None, None] @ obj_tform4x4_objs[:, :, None]
                        )
                    else:
                        msg = f"not implemented for obj_tform4x4_objs.dim(): {obj_tform4x4_objs.dim()}"
                        raise NotImplementedError(msg)
                else:
                    cams_proj4x4 = cams_proj4x4[:, None, None]
                verts_cam = (cams_proj4x4 @ add_homog_dim(verts, dim=-1)[..., None])[
                    ...,
                    0,
                ]

            verts_cam = self.unpad_mod(objs_mod=verts_cam, objs_lengths=verts_counts)
            faces = self.unpad_mod(objs_mod=faces, objs_lengths=faces_counts)

            ranges = ranges.to(torch.int32)
            faces = faces.to(torch.int32)

            # verts_cam_new = verts_cam.clone()
            # faces_new = faces.clone()
            # ranges_new = ranges.clone()
            # TODO: doublecheck if the right faces are put together in the ranges?
            #
            # verts = self.get_verts_stacked_with_mesh_ids(mesh_ids=objects_ids[0], instance_deform=instance_deform).clone()
            # if detach_objects:
            #     verts = verts.detach()
            # if instance_deform is not None:
            #     for b, object_id in enumerate(objects_ids[0]):
            #         if not detach_deform:
            #             verts[b] += instance_deform.verts_deform[b, :len(verts[b])]
            #         else:
            #             verts[b] += instance_deform.verts_deform[b, :len(verts[b])].detach()
            #
            # verts_cam = (cams_proj4x4[:, None,] @ add_homog_dim(verts, dim=-1)[..., None])[..., 0]
            # faces = []
            # faces_count = 0
            # verts_count = 0
            # verts_count_acc_from_0 = []
            # ranges = []
            # for b, object_id in enumerate(objects_ids[0]):
            #     _faces = self.get_faces_with_mesh_id(object_id, clone=True)
            #     _faces += verts_count
            #     verts_count_acc_from_0.append(verts_count)
            #     faces.append(_faces)
            #     ranges.append(torch.LongTensor([faces_count, len(_faces)]))
            #     verts_count += len(verts_cam[b])
            #     faces_count += len(_faces)
            #
            # ranges = torch.stack(ranges, dim=0).to(device='cpu', dtype=torch.int32)
            # faces = torch.cat(faces, dim=0).to(device=device, dtype=torch.int32)
            # verts_cam = verts_cam.reshape(-1, 4)

            use_cuda = "cuda" in str(
                device,
            )  #  dr.RasterizeGLContext() if not use_cuda else
            self.glctx = (
                dr.RasterizeGLContext()
                if not use_cuda
                else dr.RasterizeCudaContext(device=device)
            )
            rast_out, _ = dr.rasterize(
                self.glctx,
                verts_cam,
                faces,
                ranges=ranges,
                resolution=[int(imgs_sizes[0]), int(imgs_sizes[1])],
            )
            # minibatch_size, height, width, 4 (u, v, z / w, triangle_id)
            # start index, triangles count

        else:
            msg = f"Unknown rasterizer {self.rasterizer}."
            raise Exception(msg)

            # return {PROJECT_MODALITIES.PT3D_NCDS: color}

        mods2d_rendered = {}
        for modality in modalities:
            if modality == PROJECT_MODALITIES.MASK:
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    mod2d_rendered = self.interpolate_and_blend_face_attributes(
                        fragments,
                        feats_from_faces=None,
                        feat_bg=None,
                        return_pix_feats=False,
                        return_pix_opacity=True,
                    )
                    # mod2d_rendered = fragments.zbuf.permute(0, 3, 1, 2) > 0.0
                elif self.rasterizer == RASTERIZER.NVDIFFRAST:
                    ones = torch.ones((verts_cam.shape[0], 1)).to(device=device)
                    mod2d_rendered, _ = dr.interpolate(ones, rast_out, faces)
                    mod2d_rendered = dr.antialias(
                        mod2d_rendered,
                        rast_out,
                        verts_cam,
                        faces,
                    )
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W
            elif modality == PROJECT_MODALITIES.WIREFRAME:
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    raise NotImplementedError
                elif self.rasterizer == RASTERIZER.NVDIFFRAST:
                    bary = rast_out[..., :3] # B x H x W x 3
                    mask = rast_out[..., 3:] > 0 # B x H x W x 1

                    edge = torch.min(torch.min(bary[..., 0:1], bary[..., 1:2]), bary[..., 2:3])
                    thickness = 0.08
                    t = 0.5 * thickness
                    wire_alpha = 1.0 - torch.clamp((edge - t) / t, 0, 1)
                    wire_alpha = wire_alpha * mask

                    import torch.nn.functional as F
                    mask = (rast_out[..., 3:] > 0).float().permute(0, 3, 1, 2)
                    dilated_mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                    outline = (dilated_mask - mask).clamp(min=0)
                    mod2d_rendered = torch.zeros_like(rast_out[..., :3])
                    mod2d_rendered += wire_alpha
                    mod2d_rendered += outline.permute(0, 2, 3, 1)
                    mod2d_rendered = mod2d_rendered.clamp(0., 1.)
                    mod2d_rendered = 1. - mod2d_rendered
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W
            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
                or modality == PROJECT_MODALITIES.ONEHOT_COARSE
            ):
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    if modality == PROJECT_MODALITIES.ONEHOT:
                        if add_other_objects:
                            num_classes = self.verts_count
                            verts_ids_from_faces = torch.cat(
                                [
                                    self.get_faces_with_mesh_id(object_id)
                                    + self.verts_counts_acc_from_0[object_id]
                                    for b, object_id in enumerate(objects_ids)
                                ],
                                dim=0,
                            )
                        else:
                            num_classes = self.verts_counts_max
                            verts_ids_from_faces = torch.cat(
                                [
                                    self.get_faces_with_mesh_id(object_id)
                                    for b, object_id in enumerate(objects_ids)
                                ],
                                dim=0,
                            )

                        if add_clutter:
                            num_classes += 1

                        verts_one_hot_from_faces = torch.nn.functional.one_hot(
                            verts_ids_from_faces,
                            num_classes=num_classes,
                        ).to(device, dtype)

                    elif modality == PROJECT_MODALITIES.ONEHOT_SMOOTH:
                        from o3b.cv.select import batched_index_select

                        verts_one_hot_from_faces = torch.cat(
                            [
                                self.get_smooth_label_from_object_id(
                                    object_id=object_id,
                                    add_other_objects=add_other_objects,
                                    add_clutter=add_clutter,
                                    device=device,
                                )[
                                    self.get_faces_with_mesh_id(object_id).to(
                                        device=device,
                                    )
                                ]
                                for b, object_id in enumerate(objects_ids)
                            ],
                            dim=0,
                        )
                    elif modality == PROJECT_MODALITIES.ONEHOT_COARSE:
                        labels_onehot = self.get_labels_onehot(
                            smooth=False,
                            coarse=True,
                            objects_ids=objects_ids,
                            add_other_objects=add_other_objects,
                            add_clutter=add_clutter,
                            sample_other_objects=False,
                            sample_clutter=False,
                            device=device,
                        )
                        verts_one_hot_from_faces = torch.cat(
                            [
                                labels_onehot[
                                    b,
                                    self.get_faces_with_mesh_id(object_id).to(
                                        device=device,
                                    ),
                                ]
                                for b, object_id in enumerate(objects_ids)
                            ],
                            dim=0,
                        )

                    if add_clutter:
                        feat_bg = self.get_label_clutter(
                            add_other_objects=add_other_objects,
                            coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                            one_hot=True,
                            device=device,
                        ).to(dtype)[0] = self.get_label_clutter(
                            add_other_objects=add_other_objects,
                            coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                            one_hot=True,
                            device=device,
                        ).to(
                            dtype,
                        )[
                            0
                        ]
                    else:
                        feat_bg = None

                    mod2d_rendered = self.interpolate_and_blend_face_attributes(
                        fragments,
                        verts_one_hot_from_faces,
                        feat_bg=feat_bg,
                        return_pix_feats=True,
                        return_pix_opacity=False,
                    )
                    # mod2d_rendered = interpolate_face_attributes(
                    #    fragments.pix_to_face,
                    #    fragments.bary_coords,
                    #    verts_one_hot_from_faces,
                    # )[:, ..., 0, :].permute(0, 3, 1, 2)

                    # if add_clutter:
                    #     mask = (fragments.zbuf.permute(0, 3, 1, 2) > 0.0).expand(
                    #         *mod2d_rendered.shape,
                    #     )
                    #     clutter_onehot = (
                    #         self.get_label_clutter(
                    #             add_other_objects=add_other_objects,
                    #             one_hot=True,
                    #             device=device,
                    #         )[:, :, None, None]
                    #         .expand(*mod2d_rendered.shape)
                    #         .to(dtype)
                    #     )
                    #     mod2d_rendered[~mask] = clutter_onehot[~mask]
                else:
                    labels_onehot = self.get_labels_onehot(
                        smooth=modality == PROJECT_MODALITIES.ONEHOT_SMOOTH,
                        coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                        objects_ids=objects_ids,
                        add_other_objects=add_other_objects,
                        add_clutter=add_clutter,
                        sample_other_objects=False,
                        sample_clutter=False,
                        device=device,
                    )
                    labels_onehot = labels_onehot.reshape(-1, labels_onehot.shape[-1])
                    mod2d_rendered, _ = dr.interpolate(labels_onehot, rast_out, faces)
                    if add_clutter:
                        feat_bg = self.get_label_clutter(
                            add_other_objects=add_other_objects,
                            coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                            one_hot=True,
                            device=device,
                        ).to(dtype)[0]
                        mod2d_rendered = torch.where(
                            rast_out[..., 3:] > 0,
                            mod2d_rendered,
                            feat_bg,
                        )

                    mod2d_rendered = dr.antialias(
                        mod2d_rendered,
                        rast_out,
                        verts_cam,
                        faces,
                    )
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W

            elif modality == PROJECT_MODALITIES.DEPTH:
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    mod2d_rendered = fragments.zbuf.permute(0, 3, 1, 2)
                elif self.rasterizer == RASTERIZER.NVDIFFRAST:
                    # mod2d_rendered_z = rast_out[..., 2:3].clone()
                    # verts_cam_z = transf3d_broadcast(
                    #     pts3d=verts,
                    #     transf4x4=cams_tform4x4_obj,
                    # ).reshape(-1, 3)[:, 2:]
                    mod2d_rendered, _ = dr.interpolate(verts_cam[..., 2:3].clone(), rast_out, faces)
                    mod2d_rendered = dr.antialias(
                        mod2d_rendered,
                        rast_out,
                        verts_cam,
                        faces,
                    )
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W

            elif modality == PROJECT_MODALITIES.MASK_VERTS_VSBL:
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    B = fragments.pix_to_face.shape[0]
                else:
                    B = rast_out.shape[0]


                # # faces_ids = faces.clone() # verts in batch
                # faces_ids_b = torch.cat(
                #     [
                #         self.get_faces_with_mesh_id(object_id)
                #         for object_id in objects_ids.flatten()
                #     ],
                #     dim=0,
                # ) # verts in object ids:
                #
                # # O = len(faces_ids)
                # # verts_ids_vsbl = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0) [fragments.pix_to_face.reshape(B, -1)].reshape(B, -1)  # .unique(dim=1)
                # verts_vsbl_mask = torch.zeros(
                #     size=(B, self.verts_counts_max),
                #     dtype=torch.bool,
                #     device=device,
                # )

                if not self.rasterizer == RASTERIZER.PYTORCH3D:
                    faces_vsbl_mask = torch.zeros_like(faces[:, 0], dtype=torch.bool)
                    verts_vsbl_mask = torch.zeros_like(verts_cam[:, 0], dtype=torch.bool)
                    faces_vsbl_ids = (rast_out[:, None, :, :, -1] - 1).to(
                        dtype=torch.long,
                    ).unique()  # - verts_count_acc_from_0[b]
                    verts_vsbl_ids = faces[faces_vsbl_ids].unique()
                    verts_vsbl_mask[verts_vsbl_ids] = 1
                    verts_vsbl_mask = self.pad_mod(verts_vsbl_mask, objs_lengths=verts_counts,
                                                   objs_ids=objects_ids, concatenated_input=True)

                else:
                    verts_vsbl_mask = (torch.zeros((B, self.verts_counts_max), dtype=torch.bool, device=device,)[None, :].
                                       repeat(B, 1))
                    for b in range(B):
                        # logger.info(f'meshes_ids {meshes_ids}')
                        if self.rasterizer == RASTERIZER.PYTORCH3D:
                            faces_ids_vsbl_b = fragments.pix_to_face[b]
                        else:
                            # TODO: this is not verified to work for a batch size larger than 1
                            faces_ids_vsbl_b = (rast_out[b, None, :, :, -1] - 1).to(
                                dtype=torch.long,
                            )  #  - verts_count_acc_from_0[b]
                        faces_ids_vsbl_b = faces_ids_vsbl_b.unique()
                        faces_ids_vsbl_b = faces_ids_vsbl_b[faces_ids_vsbl_b >= 0]
                        verts_ids_vsbl_b = faces[faces_ids_vsbl_b].unique()
                        verts_vsbl_mask[b, verts_ids_vsbl_b] = 1
                mod2d_rendered = verts_vsbl_mask

            elif (
                modality == PROJECT_MODALITIES.FEATS
                or modality == PROJECT_MODALITIES.FEATS_PBR
            ):
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    feats_from_faces = torch.cat(
                        [
                            self.get_feats_from_faces_with_mesh_id(object_id)
                            for object_id in objects_ids
                        ],
                        dim=0,
                    )

                    if add_clutter:
                        feat_bg = self.feat_clutter.clone()
                    else:
                        feat_bg = None

                    mod2d_rendered = self.interpolate_and_blend_face_attributes(
                        fragments,
                        feats_from_faces,
                        feat_bg=feat_bg,
                        return_pix_feats=True,
                        return_pix_opacity=False,
                    )

                    # interpolate_face_attributes(
                    #    fragments.pix_to_face,  # B x H x W x 1
                    #    fragments.bary_coords,  # B x H x W x 1 x 3
                    #    feats_from_faces,  # F x 3 x C
                    # )[:, ..., 0, :].permute(0, 3, 1, 2)
                    # mask = fragments.pix_to_face >= 0
                    # mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
                else:
                    feats = self.get_feats_stacked_with_mesh_ids(mesh_ids=objects_ids)
                    feats = feats.reshape(-1, feats.shape[-1])

                    mod2d_rendered, _ = dr.interpolate(feats, rast_out, faces)

                    # mod2d_rendered = mod2d_rendered.permute(0, 3, 1, 2) # from B x H x W x F to B x F x H x W
                    if modality == PROJECT_MODALITIES.FEATS_PBR:
                        if rgb_light_env is None:
                            logger.warning(
                                "no rgb light environment despite using modality FEATS_PBR.",
                            )

                        if rgb_light_env is not None:
                            normals = self.normals3d(
                                meshes_ids=objects_ids,
                                instance_deform=instance_deform,
                            )
                            viewpoint_pos = inv_tform4x4(cams_tform4x4_obj)[..., :3, 3]
                            viewpoint_pos_norm = torch.nn.functional.normalize(viewpoint_pos, dim=-1)
                            dot_normals_viewpoint = torch.einsum('bvc,bc->bv', normals, viewpoint_pos_norm)
                            normals[dot_normals_viewpoint < 0.] *= -1.

                            normals = normals.reshape(-1, normals.shape[-1])

                            normals_rendered, _ = dr.interpolate(
                                normals,
                                rast_out,
                                faces,
                            )
                            # from o3b.cv.visual.show import show_imgs
                            # show_imgs(normals_rendered.permute(0, 3, 1, 2))
                            verts3d = self.get_verts_stacked_with_mesh_ids(
                                mesh_ids=objects_ids,
                            )
                            verts3d = verts3d.reshape(-1, verts3d.shape[-1])
                            verts3d_rendered, _ = dr.interpolate(
                                verts3d,
                                rast_out,
                                faces,
                            )

                            LIGHT_MIN_RES = 16
                            MIN_ROUGHNESS = 0.08
                            MAX_ROUGHNESS = 0.5

                            k_orm_d_rendered = mod2d_rendered.clone()
                            roughness = k_orm_d_rendered[..., 1:2]  # y component
                            metallic = k_orm_d_rendered[..., 2:3]  # z component
                            kd = k_orm_d_rendered[..., 3:]

                            mat_spec = (1.0 - metallic) * 0.04 + kd * metallic
                            mat_diffuse = kd * (1.0 - metallic)

                            roughness = (
                                normals_rendered[..., 0:1].detach().clone() * 0.0
                                + roughness
                            )

                            # if specular:
                            # else:
                            #    diff_col = kd

                            # load image as float
                            from o3b.cv.io import read_image
                            from o3b.cv.render.utils.util import (
                                latlong_to_cubemap,
                                load_image,
                                safe_normalize,
                                reflect,
                                dot,
                            )
                            from o3b.cv.render.utils.ops import diffuse_cubemap
                            from o3b.cv.render.utils.cubemap import cubemap_mip

                            # latlong_img = read_image("./data/light_envs/neutral.hdr")

                            # gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent,
                            #                                      gb_geometric_normal, two_sided_shading=True, opengl=True)

                            normals_rendered = safe_normalize(normals_rendered)
                            viewpoint_dir_rendered = safe_normalize(
                                viewpoint_pos[:, None, None] - verts3d_rendered,
                            )
                            reflvec_rendered = safe_normalize(
                                reflect(viewpoint_dir_rendered, normals_rendered),
                            )

                            # HxWx3
                            if isinstance(rgb_light_env, Path) or isinstance(
                                rgb_light_env,
                                str,
                            ):
                                latlong_img = torch.tensor(
                                    load_image(rgb_light_env),
                                    dtype=torch.float32,
                                    device="cuda",
                                )
                                latlong_img = latlong_img.clamp(0.2, 1.0)

                                # 6 x H x W x 3
                                cubemap = latlong_to_cubemap(latlong_img, [512, 512])
                            else:
                                cubemap = rgb_light_env

                            # opengl or whoever defines this latlong to cubemap
                            OPENGL_OBJ_TFORM_OBJ = torch.Tensor(
                                [
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            ).to(device=device)

                            from o3b.cv.visual.show import OPEN3D_OBJ_TFORM_OBJ
                            OPENGL_OBJ_TFORM_OBJ = OPEN3D_OBJ_TFORM_OBJ.clone().to(device=device)

                            normals_rendered_gl = transf3d_broadcast(
                                normals_rendered,
                                transf4x4=OPENGL_OBJ_TFORM_OBJ,
                            )
                            reflvec_rendered_gl = transf3d_broadcast(
                                reflvec_rendered,
                                transf4x4=OPENGL_OBJ_TFORM_OBJ,
                            )
                            viewpoint_dir_rendered_gl = transf3d_broadcast(
                                viewpoint_dir_rendered,
                                transf4x4=OPENGL_OBJ_TFORM_OBJ,
                            )

                            light_specular = [cubemap]
                            while light_specular[-1].shape[1] > LIGHT_MIN_RES:
                                light_specular += [
                                    cubemap_mip.apply(light_specular[-1]),
                                ]
                            light_diffuse = diffuse_cubemap(
                                light_specular[-1],
                            )  # * 0. + 1.

                            # normals_rendered : B x H x W x 3
                            # Diffuse lookup
                            light_diffuse_rendered = dr.texture(
                                light_diffuse[None, ...],
                                normals_rendered_gl,
                                filter_mode="linear",
                                boundary_mode="cube",
                            )

                            # Roughness adjusted specular env lookup
                            miplevel = torch.where(
                                roughness < MAX_ROUGHNESS,
                                (
                                    torch.clamp(roughness, MIN_ROUGHNESS, MAX_ROUGHNESS)
                                    - MIN_ROUGHNESS
                                )
                                / (MAX_ROUGHNESS - MIN_ROUGHNESS)
                                * (len(light_specular) - 2),
                                (
                                    torch.clamp(roughness, MAX_ROUGHNESS, 1.0)
                                    - MAX_ROUGHNESS
                                )
                                / (1.0 - MAX_ROUGHNESS)
                                + len(light_specular)
                                - 2,
                            )

                            light_specular_rendered = dr.texture(
                                light_specular[0][None, ...],
                                reflvec_rendered_gl,
                                mip=list(m[None, ...] for m in light_specular[1:]),
                                mip_level_bias=miplevel[..., 0],
                                filter_mode="linear-mipmap-linear",
                                boundary_mode="cube",
                            )

                            NdotV = torch.clamp(
                                dot(viewpoint_dir_rendered_gl, normals_rendered_gl),
                                min=1e-4,
                            )
                            fg_uv = torch.cat((NdotV, roughness), dim=-1)
                            if not hasattr(self, "_FG_LUT"):
                                self._FG_LUT = torch.as_tensor(
                                    np.fromfile(
                                        fglut_fpath,
                                        dtype=np.float32,
                                    ).reshape(1, 256, 256, 2),
                                    dtype=torch.float32,
                                    device="cuda",
                                )
                            fg_lookup = dr.texture(
                                self._FG_LUT,
                                fg_uv,
                                filter_mode="linear",
                                boundary_mode="clamp",
                            )

                            eye_final = mat_diffuse * light_diffuse_rendered
                            # Compute aggregate lighting
                            reflectance_rendered = (
                                mat_spec * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
                            )
                            eye_final += light_specular_rendered * reflectance_rendered

                            mod2d_rendered = eye_final  # * mask

                            # For now no RGBA support
                            mod2d_rendered = mod2d_rendered[..., :3]

                    if add_clutter:
                        feat_bg = self.feat_clutter.clone()
                        if modality == PROJECT_MODALITIES.FEATS_PBR:
                            mod2d_rendered = torch.where(
                                rast_out[..., 3:] > 0,
                                mod2d_rendered,
                                feat_bg[3:],
                            )
                        else:
                            mod2d_rendered = torch.where(
                                rast_out[..., 3:] > 0,
                                mod2d_rendered,
                                feat_bg,
                            )

                    mod2d_rendered = dr.antialias(
                        mod2d_rendered,
                        rast_out,
                        verts_cam,
                        faces,
                    )
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W

            elif (
                modality == PROJECT_MODALITIES.PT3D_NCDS
                or modality == PROJECT_MODALITIES.NOCS
                or modality == PROJECT_MODALITIES.CANOCS
                or modality == PROJECT_MODALITIES.PT3D_NCDS_AVG
                or modality == PROJECT_MODALITIES.OBJ_IN_SCENE_ONEHOT
                or modality == PROJECT_MODALITIES.OBJ_ONEHOT
                or modality == PROJECT_MODALITIES.OBJ_IN_SCENE_ID
                or modality == PROJECT_MODALITIES.OBJ_ID
            ):
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    feats_from_faces = torch.cat(
                        [
                            self.get_verts_ncds_from_faces_with_mesh_id(object_id)
                            for object_id in objects_ids
                        ],
                        dim=0,
                    )

                    mod2d_rendered = self.interpolate_and_blend_face_attributes(
                        fragments,
                        feats_from_faces,
                        feat_bg=None,
                        return_pix_feats=True,
                        return_pix_opacity=False,
                    )

                    # mod2d_rendered = interpolate_face_attributes(
                    #     fragments.pix_to_face,
                    #     fragments.bary_coords,
                    #     feats_from_faces,
                    # )[:, ..., 0, :].permute(0, 3, 1, 2)
                    # mask = fragments.pix_to_face >= 0
                    # mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
                else:
                    feats, _ = self.get_vert_mod_from_objs(
                        mod=modality,
                        objs_ids=objects_ids,
                        instance_deform=instance_deform,
                        padded=False,
                    )
                    feats = feats.to(device=device, dtype=torch.float32)  # N x [C]
                    feat_bg = self.get_mod_bg(mod=modality).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    if feat_bg.shape[-1] != feats.shape[-1]:
                        feat_bg = feat_bg.expand(*feats.shape[1:])

                    # VERT_MODALITIES.NCDS
                    # feats = self.get_verts_ncds_stacked_with_mesh_ids(mesh_ids=objects_ids[0])
                    # feats = feats.reshape(-1, feats.shape[-1])
                    if modality == PROJECT_MODALITIES.PT3D_NCDS_AVG:
                        rast_out_discrete = rast_out.clone()
                        rast_out_discrete[..., 0][
                            rast_out_discrete[..., 0] <= 0.5
                        ] = 0.0
                        rast_out_discrete[..., 0][rast_out_discrete[..., 0] > 0.5] = 1.0
                        rast_out_discrete[..., 1][
                            rast_out_discrete[..., 1] <= 0.5
                        ] = 0.0
                        rast_out_discrete[..., 1][rast_out_discrete[..., 1] > 0.5] = 1.0
                        mod2d_rendered, _ = dr.interpolate(
                            feats,
                            rast_out_discrete,
                            faces,
                        )
                    else:
                        mod2d_rendered, _ = dr.interpolate(feats, rast_out, faces)

                    # # adding diffuse light
                    # # x: front, y: left, z: top
                    # light_dir = torch.Tensor([-1., 1., 2.]).to(device=feats.device)
                    #
                    # normals = self.normals3d(meshes_ids=objects_ids, instance_deform=instance_deform)
                    # normals = normals.reshape(-1, normals.shape[-1])
                    # normals_rendered, _ = dr.interpolate(normals, rast_out, faces)
                    # light_dir = light_dir / (light_dir.norm(dim=-1, keepdim=True) + 1e-10)

                    # diffuse = torch.einsum('bhwc,c->bhw', normals_rendered, light_dir).clamp(0, 1)
                    # mod2d_rendered = (diffuse * 0.4 + 0.6)[..., None] * mod2d_rendered

                    # add white background
                    mod2d_rendered = torch.where(
                        rast_out[..., 3:] > 0,
                        mod2d_rendered,
                        feat_bg,
                    )

                    if modality != PROJECT_MODALITIES.NOCS and modality != PROJECT_MODALITIES.CANOCS:
                        mod2d_rendered = dr.antialias(
                            mod2d_rendered,
                            rast_out,
                            verts_cam,
                            faces,
                        )
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W
            elif modality == PROJECT_MODALITIES.RGB:
                if self.rasterizer == RASTERIZER.PYTORCH3D:
                    feats_from_faces = torch.cat(
                        [
                            self.get_verts_rgbs_from_faces_with_mesh_id(object_id)
                            for object_id in objects_ids
                        ],
                        dim=0,
                    )

                    mod2d_rendered = self.interpolate_and_blend_face_attributes(
                        fragments,
                        feats_from_faces,
                        return_pix_feats=True,
                        return_pix_opacity=False,
                    )

                    # interpolate_face_attributes(
                    #    fragments.pix_to_face,  # B x H x W x 1
                    #    fragments.bary_coords,  # B x H x W x 1 x 3
                    #    feats_from_faces,  # F x 3 x C
                    # )[:, ..., 0, :].permute(0, 3, 1, 2)
                    # mask = fragments.pix_to_face >= 0
                    # mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
                else:
                    if self.rgbs_uvs is None:
                        feats = self.get_rgb_stacked_with_mesh_ids(mesh_ids=objects_ids)
                        feats = feats.reshape(-1, feats.shape[-1])

                        mod2d_rendered, _ = dr.interpolate(feats, rast_out, faces)
                    else:
                        # rast_out: (B, H, W, 4), [u, v, z / w, triangle_id]
                        # uv: (B, V', 2), [u, v]
                        # uv_idx: (B, F, 3), [v1', v2', v3']
                        # tex: (B, H, W, 3), [r, g, b], float [0, 1]

                        uv, _ = self.get_vert_mod_from_objs(
                            objs_ids=objects_ids,
                            mod=VERT_MODALITIES.UV_PXL2D,
                        )
                        uv_idx, _ = self.get_face_mod_from_objs(
                            objs_ids=objects_ids,
                            mod=FACE_MODALITIES.VERTS_UVS_IN_SCENES_ID,
                        )

                        if objects_ids.dim() == 1:
                            tex = (
                                self.rgbs_uvs[objects_ids][:, 0]
                                .permute(0, 2, 3, 1)
                                .contiguous()
                            )
                        elif objects_ids.dim() == 2:
                            tex = (
                                self.rgbs_uvs[objects_ids][:, :, 0]
                                .permute(0, 3, 4, 1, 2)
                                .flatten(3)
                                .contiguous()
                            )
                        # TODO: diff between objects_ids dim 1, and 2
                        # tex = self.rgbs_uvs[objects_ids][0, :, 0].permute(0, 2, 3, 1).contiguous()

                        texc, _ = dr.interpolate(uv, rast_out, uv_idx.to(torch.int32))
                        # B x H x W x 2

                        mod2d_rendered = dr.texture(tex, texc, filter_mode="linear")

                        if objects_ids.dim() == 2:
                            vert_obj_in_scene_onehot, _ = self.get_vert_mod_from_objs(
                                objs_ids=objects_ids,
                                mod=VERT_MODALITIES.OBJ_IN_SCENE_ONEHOT,
                            )
                            vert_obj_in_scene_onehot = (
                                vert_obj_in_scene_onehot.to(mod2d_rendered.device) * 1.0
                            )
                            mod2d_rendered_obj_in_scene_onehot, _ = dr.interpolate(
                                vert_obj_in_scene_onehot,
                                rast_out,
                                faces,
                            )
                            mod2d_rendered = (
                                mod2d_rendered.reshape(
                                    *mod2d_rendered.shape[:-1],
                                    -1,
                                    3,
                                )
                                * mod2d_rendered_obj_in_scene_onehot[..., None]
                            )
                            mod2d_rendered = mod2d_rendered.sum(dim=-2)

                        # texc, _ = dr.interpolate(uv, rast_out, uv_idx.to(torch.int32))
                        # mod2d_rendered = dr.texture(tex, texc, filter_mode='linear')
                        # uv = self.get_verts_uvs_stacked_with_mesh_ids(mesh_ids=objects_ids)
                        # uv = uv.reshape(-1, uv.shape[-1])
                        # uv_idx = self.get_faces_uvs_cat_with_mesh_ids(mesh_ids=objects_ids, use_global_verts_ids=False,
                        #                                               add_verts_offset=True)
                        # tex = self.get_rgbs_uvs_stacked_with_mesh_ids(mesh_ids=objects_ids).permute(0, 2, 3, 1).contiguous()
                        # texc, _ = dr.interpolate(uv, rast_out, uv_idx.to(torch.int32))
                        # mod2d_rendered = dr.texture(tex, texc, filter_mode='linear')

                    if rgb_diffusion_alpha > 0.0 and rgb_light_env is None:
                        normals = self.normals3d(
                            meshes_ids=objects_ids,
                            instance_deform=instance_deform,
                        )
                        normals = normals.reshape(-1, normals.shape[-1])
                        normals_rendered, _ = dr.interpolate(normals, rast_out, faces)

                        ###########
                        # # adding diffuse light v1
                        # # x: front, y: left, z: top
                        light_dir = torch.Tensor([-1., 1., 2.]).to(device=mod2d_rendered.device)
                        light_dir = torch.Tensor([1., -1., 1.]).to(device=mod2d_rendered.device)
                        light_dir = light_dir / (light_dir.norm(dim=-1, keepdim=True) + 1e-10)
                        diffuse = torch.einsum('bhwc,c->bhw', normals_rendered, light_dir).clamp(0, 1)
                        mod2d_rendered[..., :3] = (diffuse * rgb_diffusion_alpha + 1-(rgb_diffusion_alpha))[..., None] * mod2d_rendered[..., :3]

                        ##########

                    if rgb_light_env is not None:

                        normals = self.normals3d(
                            meshes_ids=objects_ids,
                            instance_deform=instance_deform,
                        )
                        normals = normals.reshape(-1, normals.shape[-1])
                        normals_rendered, _ = dr.interpolate(normals, rast_out, faces)

                        verts3d = self.get_verts_stacked_with_mesh_ids(
                            mesh_ids=objects_ids,
                        )
                        verts3d = verts3d.reshape(-1, verts3d.shape[-1])
                        verts3d_rendered, _ = dr.interpolate(verts3d, rast_out, faces)
                        viewpoint_pos = inv_tform4x4(cams_tform4x4_obj)[..., :3, 3]

                        LIGHT_MIN_RES = 16
                        MIN_ROUGHNESS = 0.08
                        MAX_ROUGHNESS = 0.5

                        roughness = 0.5
                        metallic = 0.1

                        # roughness = ks[..., 1:2]  # y component
                        # metallic = ks[..., 2:3]  # z component
                        kd = mod2d_rendered

                        mat_spec = (1.0 - metallic) * 0.04 + kd * metallic
                        mat_diffuse = kd * (1.0 - metallic)

                        roughness = (
                            normals_rendered[..., 0:1].detach().clone() * 0.0
                            + roughness
                        )

                        # if specular:
                        # else:
                        #    diff_col = kd

                        # load image as float
                        from o3b.cv.io import read_image
                        from o3b.cv.render.utils.util import (
                            latlong_to_cubemap,
                            load_image,
                            safe_normalize,
                            reflect,
                            dot,
                        )
                        from o3b.cv.render.utils.ops import diffuse_cubemap
                        from o3b.cv.render.utils.cubemap import cubemap_mip

                        # latlong_img = read_image("./data/light_envs/neutral.hdr")

                        # gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent,
                        #                                      gb_geometric_normal, two_sided_shading=True, opengl=True)

                        normals_rendered = safe_normalize(normals_rendered)
                        viewpoint_dir_rendered = safe_normalize(
                            viewpoint_pos[:, None, None] - verts3d_rendered,
                        )
                        reflvec_rendered = safe_normalize(
                            reflect(viewpoint_dir_rendered, normals_rendered),
                        )

                        # HxWx3
                        if isinstance(rgb_light_env, Path) or isinstance(
                            rgb_light_env,
                            str,
                        ):
                            latlong_img = torch.tensor(
                                load_image(rgb_light_env),
                                dtype=torch.float32,
                                device="cuda",
                            )
                            latlong_img = latlong_img.clamp(0.2, 1.0)

                            # 6 x H x W x 3
                            cubemap = latlong_to_cubemap(latlong_img, [512, 512])
                        else:
                            cubemap = rgb_light_env

                        # opengl or whoever defines this latlong to cubemap
                        OPENGL_OBJ_TFORM_OBJ = torch.Tensor(
                            [
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        ).to(device=device)

                        normals_rendered_gl = transf3d_broadcast(
                            normals_rendered,
                            transf4x4=OPENGL_OBJ_TFORM_OBJ,
                        )
                        reflvec_rendered_gl = transf3d_broadcast(
                            reflvec_rendered,
                            transf4x4=OPENGL_OBJ_TFORM_OBJ,
                        )
                        viewpoint_dir_rendered_gl = transf3d_broadcast(
                            viewpoint_dir_rendered,
                            transf4x4=OPENGL_OBJ_TFORM_OBJ,
                        )

                        light_specular = [cubemap]
                        while light_specular[-1].shape[1] > LIGHT_MIN_RES:
                            light_specular += [cubemap_mip.apply(light_specular[-1])]
                        light_diffuse = diffuse_cubemap(
                            light_specular[-1],
                        )  #  * 0. + 1.

                        # normals_rendered : B x H x W x 3
                        # Diffuse lookup
                        light_diffuse_rendered = dr.texture(
                            light_diffuse[None, ...],
                            normals_rendered_gl,
                            filter_mode="linear",
                            boundary_mode="cube",
                        )

                        # Roughness adjusted specular env lookup
                        miplevel = torch.where(
                            roughness < MAX_ROUGHNESS,
                            (
                                torch.clamp(roughness, MIN_ROUGHNESS, MAX_ROUGHNESS)
                                - MIN_ROUGHNESS
                            )
                            / (MAX_ROUGHNESS - MIN_ROUGHNESS)
                            * (len(light_specular) - 2),
                            (torch.clamp(roughness, MAX_ROUGHNESS, 1.0) - MAX_ROUGHNESS)
                            / (1.0 - MAX_ROUGHNESS)
                            + len(light_specular)
                            - 2,
                        )

                        light_specular_rendered = dr.texture(
                            light_specular[0][None, ...],
                            reflvec_rendered_gl,
                            mip=list(m[None, ...] for m in light_specular[1:]),
                            mip_level_bias=miplevel[..., 0],
                            filter_mode="linear-mipmap-linear",
                            boundary_mode="cube",
                        )

                        NdotV = torch.clamp(
                            dot(viewpoint_dir_rendered_gl, normals_rendered_gl),
                            min=1e-4,
                        )
                        fg_uv = torch.cat((NdotV, roughness), dim=-1)
                        if not hasattr(self, "_FG_LUT"):
                            self._FG_LUT = torch.as_tensor(
                                np.fromfile(
                                    "./data/light_bsdf/bsdf_256_256.bin",
                                    dtype=np.float32,
                                ).reshape(1, 256, 256, 2),
                                dtype=torch.float32,
                                device="cuda",
                            )
                        fg_lookup = dr.texture(
                            self._FG_LUT,
                            fg_uv,
                            filter_mode="linear",
                            boundary_mode="clamp",
                        )

                        eye_final = mat_diffuse * light_diffuse_rendered
                        # Compute aggregate lighting
                        reflectance_rendered = (
                            mat_spec * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
                        )
                        eye_final += light_specular_rendered * reflectance_rendered

                        mod2d_rendered[..., :3] = eye_final  #  * mask

                        # rgb =
                        # model_albedo =
                        # shaded_col = diffuse * diff_col
                        #
                        # if specular:
                        #     # Lookup FG term from lookup texture
                        #     NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
                        #     fg_uv = torch.cat((NdotV, roughness), dim=-1)
                        #     if not hasattr(self, '_FG_LUT'):
                        #         self._FG_LUT = torch.as_tensor(
                        #             np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256,
                        #                                                                                    2),
                        #             dtype=torch.float32, device='cuda')
                        #     fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
                        #
                        #     # Roughness adjusted specular env lookup
                        #     miplevel = self.get_mip(roughness)
                        #     spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(),
                        #                       mip=list(m[None, ...] for m in self.specular[1:]),
                        #                       mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear',
                        #                       boundary_mode='cube')
                        #
                        #     # Compute aggregate lighting
                        #     reflectance = spec_col * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
                        #     shaded_col += spec * reflectance

                    if rgb_bg is None:
                        if mod2d_rendered.shape[-1] == 4:
                            _rgb_bg = torch.zeros(4).to(device=mod2d_rendered.device)
                            _rgb_bg[-1] = 1.0
                        else:
                            _rgb_bg = torch.zeros(3).to(device=mod2d_rendered.device)
                    else:
                        if mod2d_rendered.shape[-1] == 4:
                            _rgb_bg = torch.ones(4).to(device=mod2d_rendered.device)
                            _rgb_bg[:3] = torch.Tensor(rgb_bg).to(
                                device=mod2d_rendered.device,
                            )
                        else:
                            _rgb_bg = torch.Tensor(rgb_bg).to(
                                device=mod2d_rendered.device,
                            )

                    # if add_clutter:
                    mod2d_rendered = torch.where(
                        rast_out[..., 3:] > 0,
                        mod2d_rendered,
                        _rgb_bg,
                    )
                    mod2d_rendered = dr.antialias(
                        mod2d_rendered,
                        rast_out,
                        verts_cam,
                        faces,
                    )
                    mod2d_rendered = mod2d_rendered.permute(
                        0,
                        3,
                        1,
                        2,
                    )  # from B x H x W x F to B x F x H x W

                    # For now no RGBA support
                    mod2d_rendered = mod2d_rendered[:, :3]
            else:
                raise ValueError(f"Unknown modality {modality}")

            mods2d_rendered[modality] = mod2d_rendered

        return mods2d_rendered

    def blend_face_attributes(
        self,
        fragments,
        opacity_face=None,
        opacity_face_sdf_sigma=None,
        blend_type: FACE_BLEND_TYPE = None,
        zfar=100,
        znear=0.01,
        eps=1e-10,
        opacity_face_sdf_gamma=None,
    ):
        if opacity_face is None:
            opacity_face = self.face_opacity
        if opacity_face_sdf_sigma is None:
            opacity_face_sdf_sigma = self.face_opacity_face_sdf_sigma
        if opacity_face_sdf_gamma is None:
            opacity_face_sdf_gamma = self.face_opacity_face_sdf_gamma
        if blend_type is None:
            blend_type = self.face_blend_type

        if blend_type == FACE_BLEND_TYPE.HARD:
            pix_face_opacity = (fragments.pix_to_face >= 0).float()
            pix_face_opacity[..., 1:] = 0.0
            pix_face_weight = pix_face_opacity  # (B, H, W, F)
            pix_opacity = pix_face_weight[..., 0]  # (B, H, W, )
        elif blend_type == FACE_BLEND_TYPE.SOFT_SIGMOID_NORMALIZED:
            pix_mask = fragments.pix_to_face >= 0

            # SIGMOID PROBABILITY
            pix_face_opacity = (
                opacity_face
                * torch.sigmoid(
                    -torch.sign(fragments.dists)
                    * (fragments.dists**2)
                    / (opacity_face_sdf_sigma**2),
                )
                * pix_mask
            )

            # NORMALIZE WITH DEPTH
            z_inv = (zfar - fragments.zbuf) / (zfar - znear) * pix_mask
            z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
            pix_face_opacity = pix_face_opacity * torch.exp(
                (z_inv - z_inv_max) / opacity_face_sdf_gamma,
            )
            bg_opacity = torch.exp((eps - z_inv_max) / opacity_face_sdf_gamma).clamp(
                min=eps,
            )
            # Normalize weights.
            # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
            pix_opacity_sum = pix_face_opacity.sum(dim=-1)[..., None] + bg_opacity
            pix_face_opacity = pix_face_opacity / pix_opacity_sum
            pix_opacity = pix_face_opacity.sum(dim=-1)

        elif blend_type == FACE_BLEND_TYPE.SOFT_GAUSSIAN_CUMULATIVE:
            pix_mask = fragments.pix_to_face >= 0

            # GAUSSIAN PROBABILITY
            pix_face_opacity = (
                opacity_face
                * torch.exp(
                    0.5
                    * -torch.sign(fragments.dists)
                    * ((fragments.dists**2) / (opacity_face_sdf_sigma**2)),
                ).clamp(0, 1)
                * pix_mask
            )

            # CUMPROD WITH ORDER
            pix_face_transparity_cum = (1.0 - pix_face_opacity).cumprod(dim=-1)

            pix_face_transparity_cum_with_preprend_one = torch.cat(
                [
                    torch.ones_like(pix_face_transparity_cum[..., :1]),
                    pix_face_transparity_cum[..., :-1],
                ],
                dim=-1,
            )
            pix_face_opacity = (
                pix_face_opacity * pix_face_transparity_cum_with_preprend_one
            )  # (B, H, W, F)
            pix_opacity = pix_face_opacity.sum(
                dim=-1,
            )  #  pix_face_transparity_cum[..., -1] #  (1. - pix_face_transparity_cum_with_preprend_one)[..., -1]  # (B, H, W, )

        else:
            raise ValueError(f"Unknown blend_type {blend_type}")
        return pix_opacity, pix_face_opacity

    def interpolate_and_blend_face_attributes(
        self,
        fragments,
        feats_from_faces,
        feat_bg=None,
        opacity_face=None,
        opacity_face_sdf_sigma=None,
        return_pix_feats=True,
        return_pix_opacity=True,
        opacity_face_sdf_gamma=None,
        blend_type: FACE_BLEND_TYPE = None,
    ):
        """
        Args:
            fragments:
                pix_to_face: (B, H, W, F) tensor of faces ids.
                bary_coords: (B, H, W, F, 3) tensor of barycentric coordinates.
                dists: (B, H, W, F) tensor of signed distances to face boundary.
            feats_from_faces: (F', 3, C) total number of faces
            feat_bg: (C,) background feature.
            blend_type: (FACE_BLEND_TYPES) the type of blending to use.
        """

        if opacity_face is None:
            opacity_face = self.face_opacity
        if opacity_face_sdf_sigma is None:
            opacity_face_sdf_sigma = self.face_opacity_face_sdf_sigma
        if opacity_face_sdf_gamma is None:
            opacity_face_sdf_gamma = self.face_opacity_face_sdf_gamma
        if blend_type is None:
            blend_type = self.face_blend_type

        pix_opacity, pix_face_weight = self.blend_face_attributes(
            fragments,
            opacity_face,
            opacity_face_sdf_sigma,
            blend_type,
            opacity_face_sdf_gamma=opacity_face_sdf_gamma,
        )

        returns = ()
        if return_pix_feats:
            if feats_from_faces is None:
                raise ValueError(
                    f"feats_from_faces must be provided if return_pix_feats is True",
                )

            from pytorch3d.renderer.mesh.utils import interpolate_face_attributes

            pix_face_feats = interpolate_face_attributes(
                fragments.pix_to_face,
                fragments.bary_coords,
                feats_from_faces,
            )  # (B, H, W, F, D)

            pix_face_feats = torch.einsum(
                "bhwf,bhwfd->bhwd",
                pix_face_weight,
                pix_face_feats,
            )

            if feat_bg is not None:
                pix_face_feats += (1.0 - pix_opacity[..., None]) * feat_bg[
                    None,
                    None,
                    None,
                    :,
                ]

            pix_face_feats = pix_face_feats.permute(0, 3, 1, 2)

            returns += (pix_face_feats,)

        if return_pix_opacity:
            pix_opacity = pix_opacity[:, None]
            returns += (pix_opacity,)

        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    def sample_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
        device=None,
        dtype=None,
        sample_clutter=False,
        sample_other_objects=False,
        instance_deform: OD3D_Meshes_Deform = None,
        detach_objects=False,
        detach_deform=False,
        obj_tform4x4_objs=None,
    ):
        """
        Sample the objects' projection.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, V, F) dict of projected modalities.
        """

        mods1d_sampled = {}
        for modality in modalities:
            if modality in mods1d_sampled.keys():
                continue

            if (
                modality == PROJECT_MODALITIES.MASK
                or modality == PROJECT_MODALITIES.MASK_VERTS_VSBL
            ):
                mask_verts_vsbl = self.render(
                    cams_tform4x4_obj=cams_tform4x4_obj,
                    cams_intr4x4=cams_intr4x4,
                    imgs_sizes=imgs_sizes,
                    objects_ids=objects_ids,
                    modalities=PROJECT_MODALITIES.MASK_VERTS_VSBL,
                    add_clutter=False,
                    add_other_objects=False,
                    instance_deform=instance_deform,
                ).to(device)

                pxl2d_verts = self.sample(
                    cams_tform4x4_obj=cams_tform4x4_obj,
                    cams_intr4x4=cams_intr4x4,
                    imgs_sizes=imgs_sizes,
                    objects_ids=objects_ids,
                    modalities=PROJECT_MODALITIES.PXL2D,
                    add_clutter=False,
                    add_other_objects=False,
                    sample_clutter=False,
                    sample_other_objects=False,
                    instance_deform=instance_deform,
                ).to(device)

                mask_verts_vsbl *= (
                    pxl2d_verts <= (imgs_sizes[None, None].to(device) - 1)
                ).all(dim=-1)
                mask_verts_vsbl *= (pxl2d_verts >= 0).all(dim=-1)

                # if add_clutter or add_other_objects:
                #     clutter_count = 1 if add_clutter else 0
                #     B = mask_verts_vsbl.shape[0]
                #     if add_other_objects:
                #         mask_verts_vsbl_all = torch.zeros((B, self.verts_count + clutter_count),
                #                                           device=mask_verts_vsbl.device, dtype=mask_verts_vsbl.dtype)
                #         for b in range(B):
                #             mask_verts_vsbl_all[b, self.verts_counts_acc_from_0[objects_ids[b]]:
                #                                    self.verts_counts_acc_from_0[objects_ids[b]+1]] = \
                #                 mask_verts_vsbl[b, :self.verts_counts[objects_ids[b]]]
                #         mask_verts_vsbl = mask_verts_vsbl_all
                #     else:
                #         mask_verts_vsbl_all = torch.zeros((B, self.verts_counts_max + clutter_count),
                #                                           device=mask_verts_vsbl.device, dtype=mask_verts_vsbl.dtype)
                #         mask_verts_vsbl_all[:, :self.verts_count_max] = mask_verts_vsbl
                #         mask_verts_vsbl = mask_verts_vsbl_all
                mods1d_sampled[modality] = mask_verts_vsbl
            elif (
                modality == PROJECT_MODALITIES.FEATS
                or modality == PROJECT_MODALITIES.FEATS_COARSE
            ):
                if modality == PROJECT_MODALITIES.FEATS:
                    if add_other_objects:
                        B = len(objects_ids)
                        F = self.feat_dim
                        mods1d_sampled[modality] = self.feats_objects.clone()[
                            None,
                            :,
                            :,
                        ].repeat(
                            B,
                            1,
                            1,
                        )  # (B, O*V, F)
                    else:
                        mods1d_sampled[modality] = self.get_feats_stacked_with_mesh_ids(
                            mesh_ids=objects_ids,
                        )  # (B, V, F)
                else:
                    if add_other_objects:
                        B = len(objects_ids)
                        F = self.feat_dim
                        feats_coarse = (
                            self.get_feats_coarse_stacked_with_mesh_ids()
                            .clone()
                            .reshape(-1, F)
                        )
                        mods1d_sampled[modality] = feats_coarse.clone()[
                            None,
                            :,
                            :,
                        ].repeat(
                            B,
                            1,
                            1,
                        )  # (B, O*V, F)
                    else:
                        mods1d_sampled[
                            modality
                        ] = self.get_feats_coarse_stacked_with_mesh_ids(
                            mesh_ids=objects_ids,
                        )  # (B, V, F)

                if add_clutter:
                    B = mods1d_sampled[modality].shape[0]
                    F = mods1d_sampled[modality].shape[-1]
                    mods1d_sampled[modality] = torch.cat(
                        [
                            mods1d_sampled[modality],
                            self.feat_clutter.clone()[None, None].expand(B, 1, F),
                        ],
                        dim=1,
                    )  # (B, V+1, F)
            elif modality == PROJECT_MODALITIES.ID:
                mods1d_sampled[modality] = self.get_labels_ids(
                    objects_ids=objects_ids,
                    add_other_objects=add_other_objects,
                    sample_clutter=sample_clutter,
                    sample_other_objects=sample_other_objects,
                    device=device,
                )

            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
                or modality == PROJECT_MODALITIES.ONEHOT_COARSE
            ):
                labels_onehot = self.get_labels_onehot(
                    smooth=modality == PROJECT_MODALITIES.ONEHOT_SMOOTH,
                    coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                    objects_ids=objects_ids,
                    add_other_objects=add_other_objects,
                    add_clutter=add_clutter,
                    sample_clutter=sample_clutter,
                    sample_other_objects=sample_other_objects,
                    device=device,
                )
                mods1d_sampled[modality] = labels_onehot.permute(0, 2, 1)

            elif modality == PROJECT_MODALITIES.PXL2D:
                cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)

                if sample_other_objects:
                    verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=None).to(
                        device,
                    )
                    verts3d = verts3d[None,].expand(len(objects_ids), *verts3d.shape)
                else:
                    verts3d = self.get_verts_stacked_with_mesh_ids(
                       mesh_ids=objects_ids,
                    ).to(device)

                if detach_objects:
                    verts3d = verts3d.detach()

                if instance_deform is not None:
                    if not detach_deform:
                        verts3d += instance_deform.verts_deform
                    else:
                        verts3d += instance_deform.verts_deform.detach()

                if sample_clutter:
                    B = verts3d.shape[0]
                    verts3d = torch.cat(
                        [
                            verts3d,
                            torch.zeros(
                                (B, 1, 3),
                                device=verts3d.device,
                                dtype=verts3d.dtype,
                            ),
                        ],
                        dim=1,
                    )

                pxl2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])
                mods1d_sampled[modality] = pxl2d
            elif (
                modality == PROJECT_MODALITIES.PT3D
                or modality == PROJECT_MODALITIES.PT3D_COARSE
            ):
                if sample_other_objects:
                    verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=None).to(
                        device,
                    )
                    verts3d = verts3d[None,].expand(len(objects_ids), *verts3d.shape)
                else:
                    verts3d = self.get_verts_stacked_with_mesh_ids(
                        mesh_ids=objects_ids,
                    ).to(device)

                if detach_objects:
                    verts3d = verts3d.detach()

                if instance_deform is not None:
                    if not detach_deform:
                        verts3d += instance_deform.verts_deform
                    else:
                        verts3d += instance_deform.verts_deform.detach()

                if modality == PROJECT_MODALITIES.PT3D_COARSE:
                    from o3b.cv.select import batched_index_select

                    verts3d = batched_index_select(
                        input=verts3d,
                        index=self.verts_ids_coarse[objects_ids],
                        dim=1,
                    )

                    if detach_objects:
                        verts3d = verts3d.detach()

                if sample_clutter:
                    B = verts3d.shape[0]
                    verts3d = torch.cat(
                        [
                            verts3d,
                            torch.zeros(
                                (B, 1, 3),
                                device=verts3d.device,
                                dtype=verts3d.dtype,
                            ),
                        ],
                        dim=1,
                    )
                mods1d_sampled[modality] = verts3d
            elif modality == PROJECT_MODALITIES.PT3D_NCDS:
                if sample_other_objects:
                    verts3d = self.get_verts_ncds_stacked_with_mesh_ids(
                        mesh_ids=None,
                    ).to(device)
                    verts3d = verts3d[None,].expand(len(objects_ids), *verts3d.shape)
                else:
                    verts3d = self.get_verts_ncds_stacked_with_mesh_ids(
                        mesh_ids=objects_ids,
                    ).to(device)

                if detach_objects:
                    verts3d = verts3d.detach()

                if instance_deform is not None:
                    if not detach_deform:
                        verts3d += instance_deform.verts_deform
                    else:
                        verts3d += instance_deform.verts_deform.detach()

                if sample_clutter:
                    B = verts3d.shape[0]
                    verts3d = torch.cat(
                        [
                            verts3d,
                            torch.zeros(
                                (B, 1, 3),
                                device=verts3d.device,
                                dtype=verts3d.dtype,
                            ),
                        ],
                        dim=1,
                    )
                mods1d_sampled[modality] = verts3d
            else:
                raise ValueError(f"Unknown modality {modality}")

        return mods1d_sampled

    def get_label_max(self, add_other_objects=True, add_clutter=True, coarse=False):
        if coarse:
            if add_other_objects:
                if add_clutter:
                    return self.verts_coarse_count * len(self)
                else:
                    return self.verts_coarse_count * len(self) - 1
            else:
                if add_clutter:
                    return self.verts_coarse_count
                else:
                    return self.verts_coarse_count - 1
        else:
            if add_other_objects:
                if add_clutter:
                    return self.verts_count
                else:
                    return self.verts_count - 1
            else:
                if add_clutter:
                    return self.verts_counts_max
                else:
                    return self.verts_counts_max - 1

    def get_label_clutter(
        self,
        add_other_objects=False,
        one_hot=False,
        device=None,
        coarse=False,
    ):
        clutter_label = self.get_label_max(
            add_other_objects=add_other_objects,
            add_clutter=True,
            coarse=coarse,
        )
        clutter_label = torch.LongTensor([clutter_label]).to(device)

        if one_hot:
            clutter_label = torch.nn.functional.one_hot(clutter_label, num_classes=-1)
        return clutter_label

    def update_feats_moving_average(
        self,
        labels,
        labels_mask,
        feats,
        alpha,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        """
        Args:
            labels (torch.Tensor): BxN (or BxVxN)
            labels_mask (torch.Tensor): BxN
            feats (torch.Tensor): BxNxF
            alpha (float): the moving average factor.
        """
        device = feats.device
        dtype = feats.dtype
        feats_count = len(self.feats_objects) + 1
        if (
            not hasattr(self, "feats_moving_average")
            or self.feats_moving_average is None
        ):
            self.feats_moving_average = torch.zeros(
                (feats_count, self.feat_dim),
                dtype=dtype,
                device=device,
            )
            import math

            # equals kaiming uniform
            bound = 1 / math.sqrt(self.feat_dim) if self.feat_dim > 0 else 0
            torch.nn.init.uniform_(self.feats_moving_average, a=-bound, b=-bound)

        # V x F
        # feats_update = torch.zeros((feats_count, self.feat_dim), dtype=dtype, device=device)

        labels = labels.clone()
        if labels.dim() == 2:
            if not add_other_objects:
                for b in range(objects_ids.shape[0]):
                    labels[b] = labels[b] + self.verts_counts_acc_from_0[objects_ids[b]]
            labels[labels >= feats_count] = feats_count - 1
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=feats_count)
        else:
            raise NotImplementedError
            # labels_onehot = labels.permute(0, 2, 1)
            # if not add_other_objects or not add_clutter:
            #     B = labels_onehot.shape[0]
            #     N = labels_onehot.shape[1]
            #     labels_onehot_ext = torch.zeros((B, N, feats_count), dtype=torch.bool, device=device)
            #
            #     if not add_other_objects:
            #         labels_onehot_ext[:, :, self.verts_counts_acc_from_0[objects_ids+1]:self.verts_counts_acc_from_0[objects_ids+2]] = labels_onehot
            #     labels_onehot = labels_onehot_ext

        feats_update = torch.einsum(
            "nf,nv->vf",
            feats[labels_mask],
            labels_onehot[labels_mask] * 1.0,
        ) / (labels_onehot[labels_mask].sum(dim=0)[:, None] + 1e-10)
        feats_update = feats_update.detach()
        feats_update_mask = labels_onehot[labels_mask].sum(dim=0) > 0
        self.feats_moving_average[feats_update_mask] = (
            alpha * self.feats_moving_average[feats_update_mask]
            + (1.0 - alpha) * feats_update[feats_update_mask]
        )

        self.feats_objects = self.feats_moving_average[:-1].clone()
        self.feat_clutter = self.feats_moving_average[-1].clone()

    def update_feats_total_average(
        self,
        labels,
        labels_mask,
        feats,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        device = feats.device
        dtype = feats.dtype
        feats_count = len(self.feats_objects) + 1
        if (
            not hasattr(self, "feats_total_average_sum")
            or self.feats_total_average_sum is None
        ):
            self.feats_total_average_sum = torch.zeros(
                (feats_count, self.feat_dim),
                dtype=dtype,
                device=device,
            )
            self.feats_total_count = torch.zeros(
                (feats_count,),
                dtype=torch.long,
                device=device,
            )

        labels = labels.clone()
        if labels.dim() == 2:
            if not add_other_objects:
                for b in range(objects_ids.shape[0]):
                    labels[b] = labels[b] + self.verts_counts_acc_from_0[objects_ids[b]]
            labels[labels >= feats_count] = feats_count - 1
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=feats_count)
        else:
            raise NotImplementedError
            # labels_onehot = labels.permute(0, 2, 1)
            # if not add_other_objects or not add_clutter:
            #     B = labels_onehot.shape[0]
            #     N = labels_onehot.shape[1]
            #     labels_onehot_ext = torch.zeros((B, N, feats_count), dtype=torch.bool, device=device)
            #
            #     if not add_other_objects:
            #         labels_onehot_ext[:, :, self.verts_counts_acc_from_0[objects_ids + 1]:self.verts_counts_acc_from_0[
            #             objects_ids + 2]] = labels_onehot
            #     labels_onehot = labels_onehot_ext

        feats_update = torch.einsum(
            "nf,nv->vf",
            feats[labels_mask],
            labels_onehot[labels_mask] * 1.0,
        ) / (labels_onehot[labels_mask].sum(dim=0)[:, None] + 1e-10)
        feats_update = feats_update.detach()
        feats_update_mask = labels_onehot[labels_mask].sum(dim=0) > 0
        self.feats_total_average_sum[feats_update_mask] = (
            self.feats_total_average_sum[feats_update_mask]
            + feats_update[feats_update_mask]
        )
        self.feats_total_count[feats_update_mask] += 1

        self.feats_objects = self.feats_total_average_sum[:-1].clone()
        self.feat_clutter = self.feats_total_average_sum[-1].clone()

    def get_labels_ids(
        self,
        objects_ids=None,
        add_other_objects=True,
        sample_clutter=False,
        sample_other_objects=True,
        device=None,
    ):
        """
        Args:
            objects_ids (torch.Tensor): B
            add_other_objects (bool): whether to add other objects.
            sample_clutter (bool): whether to sample clutter.
            sample_other_objects (bool): whether to sample other objects.
        Returns:
            labels_ids (torch.Tensor): BxV(*O)(+1) depending on sample clutter/sample other objects
                                       from 0 to V(*O)(+1) or 0 to V(+1) if not add other objects.
        """
        if objects_ids is None:
            objects_ids = list(range(len(self)))

        if device is None:
            device = self.device

        if add_other_objects:
            if not sample_other_objects:
                labels_ids = self.get_verts_and_noise_ids_stacked(
                    mesh_ids=objects_ids,
                    count_noise_ids=0,
                ).to(
                    device,
                )  # (B, V(+1))
            else:
                labels_ids = self.get_verts_and_noise_ids_stacked(
                    mesh_ids=None,
                    count_noise_ids=0,
                ).to(device)
                labels_ids = labels_ids[None,].expand(
                    len(objects_ids),
                    *labels_ids.shape,
                )
        else:
            if not sample_other_objects:
                labels_ids = self.get_verts_and_noise_ids_stacked_without_acc(
                    mesh_ids=objects_ids,
                    count_noise_ids=0,
                ).to(
                    device,
                )  # (B, V(+1))
            else:
                labels_ids = self.get_verts_and_noise_ids_stacked_without_acc(
                    mesh_ids=None,
                    count_noise_ids=0,
                ).to(
                    device,
                )
                labels_ids = labels_ids[None,].expand(
                    len(objects_ids),
                    *labels_ids.shape,
                )

        if sample_clutter:
            B = labels_ids.shape[0]
            labels_ids = torch.cat(
                [
                    labels_ids,
                    self.get_label_clutter(add_other_objects=add_other_objects)
                    .expand(B, 1)
                    .to(device),
                ],
                dim=1,
            )  # (B, V+1, F)
        return labels_ids

    def get_labels_onehot(
        self,
        smooth=False,
        objects_ids=None,
        add_other_objects=True,
        add_clutter=True,
        sample_clutter=False,
        sample_other_objects=True,
        device=None,
        coarse=False,
    ):
        """
        Args:
            objects_ids (torch.Tensor): B
            smooth (bool): whether to use smooth labels.
        Returns:
            labels_onehot (torch.Tensor): VxV
        """

        if device is None:
            device = self.device

        if objects_ids is None:
            objects_ids = list(range(len(self)))

        if coarse:
            label_max = self.get_label_max(
                add_other_objects=add_other_objects,
                add_clutter=add_clutter,
                coarse=coarse,
            )
            labels_ids = self.get_labels_ids(
                objects_ids=objects_ids,
                add_other_objects=add_other_objects,
                sample_clutter=sample_clutter,
                sample_other_objects=sample_other_objects,
                device=device,
            )  # B (x O) x V(*O)(+1)

            verts_label_coarse = self.verts_label_coarse.clone()

            if sample_clutter:
                S = verts_label_coarse.shape[0] + 1
            else:
                S = verts_label_coarse.shape[0]

            K = label_max + 1
            labels_onehot = torch.zeros((S, K), device=device)

            if add_other_objects:
                for i in range(len(self)):
                    labels_onehot[
                        self.verts_counts_acc_from_0[i] : self.verts_counts_acc_from_0[
                            i + 1
                        ],
                        i * self.verts_coarse_count : (i + 1) * self.verts_coarse_count,
                    ] = verts_label_coarse[
                        self.verts_counts_acc_from_0[i] : self.verts_counts_acc_from_0[
                            i + 1
                        ]
                    ]
            else:
                labels_onehot[
                    : verts_label_coarse.shape[0],
                    : verts_label_coarse.shape[1],
                ] = verts_label_coarse

            if sample_clutter and add_clutter:
                labels_onehot[-1, -1] = 1.0

            #  as labels come from the batch, they might contain vertices out of index, these vertices are not used
            labels_ids_out_of_range = labels_ids >= labels_onehot.shape[0]
            labels_ids[labels_ids_out_of_range] = 0
            labels_onehot = labels_onehot[labels_ids]

        else:
            if not smooth:
                labels_ids = self.get_labels_ids(
                    objects_ids=objects_ids,
                    add_other_objects=add_other_objects,
                    sample_clutter=sample_clutter,
                    sample_other_objects=sample_other_objects,
                    device=device,
                )  # BxV(*O)(+1) in range 0 to V(*O)(+1) or 0 to V(+1) if not add other objects.
                label_max = self.get_label_max(
                    add_other_objects=add_other_objects,
                    add_clutter=add_clutter,
                )
                labels_onehot = torch.eye(label_max + 1, device=device)

                labels_ids_out_of_range = labels_ids >= labels_onehot.shape[0]
                labels_ids[labels_ids_out_of_range] = 0
                labels_onehot = labels_onehot[labels_ids]
                labels_onehot[labels_ids_out_of_range] = 0
            else:
                labels_ids = self.get_labels_ids(
                    objects_ids=objects_ids,
                    add_other_objects=True,
                    sample_clutter=sample_clutter,
                    sample_other_objects=sample_other_objects,
                    device=device,
                )  # BxV(*O)(+1) in range 0 to V(*O)(+1) or 0 to V(+1) if not add other objects.
                if add_other_objects:
                    if add_clutter:
                        labels_onehot = (
                            self.get_geodesic_prob_with_noise()
                            .clone()
                            .to(device=device)
                        )

                    else:
                        labels_onehot = (
                            self.get_geodesic_prob().clone().to(device=device)
                        )

                    labels_ids_out_of_range = labels_ids >= labels_onehot.shape[0]
                    labels_ids[labels_ids_out_of_range] = 0
                    labels_onehot = labels_onehot[labels_ids]
                    labels_onehot[labels_ids_out_of_range] = 0
                else:
                    from o3b.cv.select import batched_index_select

                    labels_onehot = self.get_smooth_label_from_objects_ids(
                        objects_ids=objects_ids,
                        add_other_objects=add_other_objects,
                        add_clutter=add_clutter,
                        device=device,
                    )
                    labels_onehot = batched_index_select(
                        index=labels_ids,
                        input=labels_onehot,
                        dim=1,
                    )

        return labels_onehot

    def get_smooth_label_from_objects_ids(
        self,
        objects_ids=None,
        add_other_objects=True,
        add_clutter=True,
        device=None,
    ):
        if objects_ids is None:
            objects_ids = list(range(len(self)))

        label_max = self.get_label_max(
            add_other_objects=add_other_objects,
            add_clutter=add_clutter,
        )
        labels_onehot_smooth = torch.zeros(
            (len(objects_ids), label_max + 1, label_max + 1),
            device=device,
        )

        for b, object_id in enumerate(objects_ids):
            if add_clutter:
                labels_onehot_all = self.get_geodesic_prob_with_noise()
            else:
                labels_onehot_all = self.get_geodesic_prob()

            if add_other_objects:
                labels_onehot_smooth[
                    b,
                    : self.verts_counts[object_id],
                    : labels_onehot_all.shape[-1],
                ] = labels_onehot_all[
                    self.verts_counts_acc_from_0[
                        object_id
                    ] : self.verts_counts_acc_from_0[object_id + 1]
                ]
            else:
                labels_onehot_smooth[
                    b,
                    : self.verts_counts[object_id],
                    : self.verts_counts[object_id],
                ] = labels_onehot_all[
                    self.verts_counts_acc_from_0[
                        object_id
                    ] : self.verts_counts_acc_from_0[object_id + 1],
                    self.verts_counts_acc_from_0[
                        object_id
                    ] : self.verts_counts_acc_from_0[object_id + 1],
                ]

            if add_clutter:
                labels_onehot_smooth[b, -1, -1] = 1.0

        return labels_onehot_smooth

    def get_smooth_label_from_object_id(
        self,
        object_id,
        add_other_objects=True,
        add_clutter=True,
        device=None,
    ):
        if device is None:
            device = self.device

        labels_onehot_all = self.get_geodesic_prob().clone().to(device)

        if add_other_objects:
            labels_onehot_smooth = labels_onehot_all[
                self.verts_counts_acc_from_0[object_id] : self.verts_counts_acc_from_0[
                    object_id + 1
                ]
            ]
        else:
            labels_onehot_smooth = labels_onehot_all[
                self.verts_counts_acc_from_0[object_id] : self.verts_counts_acc_from_0[
                    object_id + 1
                ],
                self.verts_counts_acc_from_0[object_id] : self.verts_counts_acc_from_0[
                    object_id + 1
                ],
            ]

        if add_clutter:
            _labels_onehot_smooth = torch.zeros(
                (labels_onehot_smooth.shape[0], labels_onehot_smooth.shape[1] + 1),
            ).to(labels_onehot_smooth.device, labels_onehot_smooth.dtype)
            _labels_onehot_smooth[:, :-1] = labels_onehot_smooth
            labels_onehot_smooth = _labels_onehot_smooth
        return labels_onehot_smooth

    def get_rgb_meshes_as_list_of_o3d(
        self,
        category_id: torch.LongTensor = None,
        instance_deform=None,
        device=None,
    ):
        if device is None:
            device = self.device
        if category_id is None:
            category_id = list(range(len(self)))

        meshes_verts = self.get_verts_stacked_with_mesh_ids(category_id).to(
            device=device,
        )
        if instance_deform is not None:
            meshes_verts += instance_deform.verts_deform.to(device=device)

        meshes_verts = meshes_verts[:, :, [1, 2, 0]]  # permute for wandb visualization
        meshes_faces = self.get_faces_stacked_with_mesh_ids(category_id).to(
            device=device,
        )
        meshes_rgb = self.get_verts_ncds_stacked_with_mesh_ids(category_id).to(
            device=device,
        )

        o3d_meshes = []
        for b in range(len(category_id)):
            mesh = Meshes(
                verts=[meshes_verts[b]],
                faces=[meshes_faces[b]],
                rgb=[meshes_rgb[b]],
            )
            o3d_meshes.append(mesh.to_o3d())

        return o3d_meshes
