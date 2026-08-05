import logging

logger = logging.getLogger(__name__)
import os
import cv2
from o3b.cv.visual.draw import tensor_to_cv_img, cv_img_to_tensor
from o3b.cv.visual.resize import resize
import torch
import math

from o3b.cv.geometry.transform import transf3d_broadcast
from o3b.cv.visual.draw import get_colors
from pathlib import Path
from typing import List
import torchvision
import PIL

import open3d as o3d
import numpy as np
from o3b.cv.geometry.transform import tform4x4, inv_tform4x4

CAM_TFORM_OBJ = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OBJ_TFORM_OBJ_SHAPENET = torch.Tensor(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OPEN3D_CAM_TFORM_CAM = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
) # could be wrong

PYTORCH3D_CAM_TFORM_CAM = torch.Tensor(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
) # could be wrong

CAM_TFORM_OPEN3D_CAM = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
) # could be wrong

OPEN3D_OBJ_TFORM_OBJ = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OBJ_TFORM_OPEN3D_OBJ = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OPEN3D_CAM_TFORM_OBJ = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OBJ_TFORM_OPEN3D_CAM = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

def get_default_camera_intrinsics_from_img_size(
    W,
    H,
    fov_x=25,
    fov_y=None,
    dtype=None,
    device=None,
):
    import math

    fov_x_rad = fov_x / 180 * math.pi

    fx = (W / 2) / math.tan(fov_x_rad / 2)
    if fov_y is None:
        fy = fx
    else:
        fov_y_rad = fov_y / 180 * math.pi
        fy = (H / 2) / torch.tan(fov_y_rad / 2)

    cam_intr4x4 = torch.eye(4).to(dtype=dtype, device=device)  # * (W + H) / 2
    cam_intr4x4[0, 0] = fx
    cam_intr4x4[1, 1] = fy
    cam_intr4x4[0, 2] = W / 2
    cam_intr4x4[1, 2] = H / 2

    cam_intr4x4[..., -1, -1] = 1.0
    cam_intr4x4[..., -2, -2] = 1.0
    return cam_intr4x4


def pt3d_camera_from_tform4x4_intr4x4_imgs_size(
    cam_tform4x4_obj: torch.Tensor,
    cam_intr4x4: torch.Tensor,
    img_size: torch.Tensor,
):
    from pytorch3d.renderer.cameras import PerspectiveCameras

    if cam_tform4x4_obj.dim() == 2:
        cam_tform4x4_obj = cam_tform4x4_obj[None,]
    if cam_intr4x4.dim() == 2:
        cam_intr4x4 = cam_intr4x4[None,].expand(cam_tform4x4_obj.shape[0], 4, 4)
    if img_size.dim() == 1:
        img_size = img_size[None,].expand(cam_tform4x4_obj.shape[0], 2)

    t3d_tform_default = torch.Tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ).to(
        device=cam_tform4x4_obj.device,
        dtype=cam_tform4x4_obj.dtype,
    )

    from o3b.cv.geometry.transform import tform4x4_broadcast

    cam_tform4x4_obj = tform4x4_broadcast(t3d_tform_default[None,], cam_tform4x4_obj)
    # cam_tform4x4_obj = torch.bmm(t3d_tform_default[None,], cam_tform4x4_obj)
    focal_length = torch.stack([cam_intr4x4[:, 0, 0], cam_intr4x4[:, 1, 1]], dim=1)
    principal_point = torch.stack([cam_intr4x4[:, 0, 2], cam_intr4x4[:, 1, 2]], dim=1)
    # principal_point = torch.stack([img_size[:, 1] - cam_intr4x4[:, 0, 2], img_size[:, 0] - cam_intr4x4[:, 1, 2]], dim=1)

    R = cam_tform4x4_obj[:, :3, :3]
    t = cam_tform4x4_obj[:, :3, 3]
    cameras = PerspectiveCameras(
        R=R.transpose(-2, -1),
        T=t,
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        image_size=img_size,
        device=cam_tform4x4_obj.device,
    )

    return cameras


def show_mesh():
    raise NotImplementedError
    """
    from pytorch3d.structures.meshes import Meshes as PT3DMeshes
    meshes = PT3DMeshes(verts=[verts], faces=[faces])
    from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
    fig = plot_scene({
        "Meshes": {
            f"mesh{i+1}": meshes[i] for i in range(len(meshes))
        }}, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                          showaxeslabels=True, showticklabels=True))
    fig.show()
    input('bla')
    """


from typing import Union
from o3b.cv.geometry.objects3d.meshes import Meshes
from o3b.cv.visual.draw import get_colors
import open3d


def show_scene2d(
    pts2d: Union[torch.Tensor, List[torch.Tensor]] = None,
    pts2d_names: List[str] = None,
    pts2d_colors: Union[torch.Tensor, List] = None,
    pts2d_lengths: List[int] = None,
    return_visualization=False,
):
    """

    Args:
        pts2d (Union[torch.Tensor, List[torch.Tensor]]): PxNx2 or List(Npx2)
        pts2d_names (List[str]): (P,)
        pts2d_colors (Union[torch.Tensor, List]): Px2x3 or List(3)
    Returns:
        img: torch.Tensor

    """
    import matplotlib.pyplot as plt
    import itertools

    # matplotlib.use("TkAgg")
    if pts2d_lengths is not None:
        pts2d_lengths_sum = np.cumsum([0, *pts2d_lengths])

    # scatter plot 2d with legend and colors
    fig, ax = plt.subplots(len(pts2d), 1)
    # to make ax iterable
    ax = [ax] if len(pts2d) == 1 else ax
    if pts2d is not None:
        for i, pts2d_i in enumerate(pts2d):
            if pts2d_colors is not None:
                pts2d_colors_i = pts2d_colors[i]
            else:
                pts2d_colors_i = get_colors(len(pts2d))[i]
            if isinstance(pts2d_colors_i, torch.Tensor):
                c = pts2d_colors_i.detach().cpu().numpy()
            else:
                c = pts2d_colors_i
            if pts2d_lengths is not None:
                marker = itertools.cycle((".", "+", "v", "o", "*"))
                for j in range(len(pts2d_lengths)):
                    ax[i].scatter(
                        pts2d_i[pts2d_lengths_sum[j] : pts2d_lengths_sum[j + 1], 0]
                        .detach()
                        .cpu()
                        .numpy(),
                        pts2d_i[pts2d_lengths_sum[j] : pts2d_lengths_sum[j + 1], 1]
                        .detach()
                        .cpu()
                        .numpy(),
                        marker=next(marker),
                        c=c[pts2d_lengths_sum[j] : pts2d_lengths_sum[j + 1]],
                    )
            else:
                ax[i].scatter(
                    pts2d_i[:, 0].detach().cpu().numpy(),
                    pts2d_i[:, 1].detach().cpu().numpy(),
                    c=c,
                )
            if pts2d_names is not None:
                ax[i].set_title(pts2d_names[i])
            ax[i].set_axis_off()
    if return_visualization:
        fig.canvas.draw()
        try:
            img = PIL.Image.frombytes(
                "RGB",
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb(),
            )
        except:
            img = PIL.Image.frombytes(
                "RGB",
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
            )
        img_tensor = torchvision.transforms.ToTensor()(img)

        return img_tensor
    plt.show()


def show_bar_chart(
    x: Union[int, List[int]],
    height: Union[torch.Tensor, List[torch.Tensor]],
    pts2d_colors: Union[torch.Tensor, List] = None,
    return_visualization=False,
):
    import matplotlib.pyplot as plt

    if isinstance(x, int):
        x = range(x)
    fig, ax = plt.subplots()
    ax.bar(x, height.detach().cpu().numpy(), color=pts2d_colors)
    if isinstance(pts2d_colors, torch.Tensor):
        c = pts2d_colors.detach().cpu().numpy()
    else:
        c = pts2d_colors

    if return_visualization:
        fig.canvas.draw()
        try:
            img = PIL.Image.frombytes(
                "RGB",
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb(),
            )
        except:
            img = PIL.Image.frombytes(
                "RGB",
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
            )
        img_tensor = torchvision.transforms.ToTensor()(img)

        return img_tensor
    plt.show()


from o3b.data import ExtEnum


class OD3D_RENDERER(str, ExtEnum):
    OPEN3D = "open3d"
    PYTORCH3D = "pytorch3d"
    PYRENDER = "pyrender"


def plotly_fig_2_tensor(fig, width=None, height=None):
    from PIL import Image
    import io

    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png", width=width, height=height)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[:3] / 255.0
    return img


_offscreen_renderers: dict = {}


def _offscreen_renderer(width: int, height: int):
    """pyrender OffscreenRenderer for this size, created once per process.

    Building one costs ~135 ms (EGL context + framebuffer setup) against ~4 ms
    for the render itself, so a per-call renderer dominates any loop over
    frames — e.g. baking fo_mask_amodal over a dataset. pyrender takes the
    scene at render time, so one renderer serves every scene of that size.

    Keyed by pid as well: a DataLoader worker forked after the parent rendered
    would otherwise inherit the parent's GL context, which is not valid in the
    child.
    """
    import pyrender

    key = (os.getpid(), int(width), int(height))
    if key not in _offscreen_renderers:
        if "DISPLAY" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        _offscreen_renderers[key] = pyrender.OffscreenRenderer(*key[1:])
    return _offscreen_renderers[key]


def render_trimesh_to_tensor(
    mesh_trimesh,
    cam_intr4x4,
    cam_tform4x4_obj,
    H=512,
    W=512,
    rgb_bg=[0.0, 0.0, 0.0],
    ambient_light=[1.0, 1.0, 1.0, 1.0],
    znear=0.01,
    zfar=100.0,
    material = None,
    light_tform4x4_obj= None,
):
    """
    Render a trimesh mesh using given camera intrinsics and extrinsics, and return the RGB image as a torch tensor.

    Args:
        mesh_trimesh (trimesh.Trimesh): The mesh to be rendered.
        cam_intr4x4 (torch.Tensor): 3x3 camera intrinsic matrix.
        cam_tform4x4_obj (torch.Tensor): 4x4 camera extrinsic matrix.
        H: height
        W: width

    Returns:
        torch.Tensor: Rendered RGB image as a torch tensor of shape (3, H, W).
        torch.Tensor: Rendered Depth image as a torch tensor of shape (1, H, W).
    """
    # pip install pyrender
    # pip install PyOpenGL-accelerate
    import trimesh
    import pyrender
    import numpy as np
    import torch
    from pyrender import DirectionalLight, SpotLight, PointLight
    from PIL import Image

    # mesh_trimesh.show()

    # Convert trimesh mesh to pyrender mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, material=material)

    # Create scene and add mesh
    scene = pyrender.Scene(
        ambient_light=np.array(ambient_light),
        bg_color=rgb_bg,
    )
    scene.add(pyrender_mesh)

    # Create pyrender camera from intrinsic matrix
    fx, fy = cam_intr4x4[0, 0], cam_intr4x4[1, 1]
    cx, cy = cam_intr4x4[0, 2], cam_intr4x4[1, 2]
    height = H
    width = W
    camera = pyrender.IntrinsicsCamera(
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        znear=znear,
        zfar=zfar,
    )

    # FOLLOWING OPENGL convention
    pyrender_cam_tform4x4_obj = cam_tform4x4_obj.clone()
    
    #pyrender_cam_tform4x4_obj = tform4x4(OPEN3D_CAM_TFORM_CAM.clone().to(device=cam_tform4x4_obj.device), cam_tform4x4_obj,)
    # pyrender_cam_tform4x4_obj = tform4x4(cam_tform4x4_obj, OBJ_TFORM_OPEN3D_OBJ.clone().to(device=cam_tform4x4_obj.device),)

    from o3b.cv.geometry.transform import inv_tform4x4
    obj_tform4x4_pyrender_cam = inv_tform4x4(pyrender_cam_tform4x4_obj)
    obj_tform4x4_pyrender_cam_np = obj_tform4x4_pyrender_cam.detach().cpu().numpy()
    scene.add(camera, pose=obj_tform4x4_pyrender_cam_np)

    # pyrender.Viewer(scene)

    # direc_l = PointLight(color=np.ones(3), intensity=50.0)
    direc_ls = []
    direc_ls.append(PointLight(color=np.ones(3), intensity=20.0)) # 50

    for direc_l in direc_ls:
        if light_tform4x4_obj is None: # CAM_TFORM_OBJ
            light_tform4x4_obj = cam_tform4x4_obj.clone().to(device=cam_tform4x4_obj.device)
        pyrender_light_tform4x4_obj = light_tform4x4_obj
        obj_tform4x4_pyrender_light = inv_tform4x4(pyrender_light_tform4x4_obj)
        obj_tform4x4_pyrender_light_np = obj_tform4x4_pyrender_light.detach().cpu().numpy()
        direc_l_node = scene.add(direc_l, pose=obj_tform4x4_pyrender_light_np)

    # pyrender.Viewer(scene, shadows=True)

    # Offscreen renderer, reused across calls of the same size
    renderer = _offscreen_renderer(width, height)

    # Render the scene
    color, depth = renderer.render(scene)

    # Convert to torch tensor (C, H, W) format
    rgb_tensor = (
        torch.from_numpy(color.copy()).to(dtype=torch.float32).permute(2, 0, 1) / 255.0
    )
    depth_tensor = torch.from_numpy(depth.copy()).to(dtype=torch.float32)[None,]

    return rgb_tensor, depth_tensor


def show_scene(
    cams_tform4x4_world: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_intr4x4: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_imgs: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_names: List[str] = None,
    cams_imgs_resize: bool = True,
    cams_imgs_depth_scale: float = 0.30,
    cams_show_wireframe: bool = True,
    cams_show_image_encoder: bool = False,
    pts3d: Union[torch.Tensor, List[torch.Tensor]] = None,
    pts3d_names: List[str] = None,
    render_cam_tform4x4_obj: torch.Tensor = None,
    pts3d_colors: Union[torch.Tensor, List] = None,
    pts3d_normals: Union[torch.Tensor, List] = None,
    lines3d: Union[torch.Tensor, List[torch.Tensor]] = None,
    lines3d_names: List[str] = None,
    lines3d_colors: Union[torch.Tensor, List] = None,
    lines3d_radius: float = None,
    lines3d_resolution: int = 20,
    meshes: Union[Meshes, List[Meshes]] = None,
    meshes_names: List[str] = None,
    meshes_colors: Union[torch.Tensor, List] = None,
    meshes_add_translation: bool = False,
    pts3d_add_translation: bool = False,
    fpath: Path = None,
    return_visualization=False,
    viewpoints_count=1,
    viewpoint_init_dist=10.0,
    dtype=torch.float,
    H=1080,
    W=1980,
    fps=10,
    pts3d_size=3.0,
    background_color=(1.0, 1.0, 1.0),
    device="cpu",
    meshes_as_wireframe=False,
    crop_white_border=False,
    renderer=OD3D_RENDERER.OPEN3D,
    show_coordinate_frame=False,
    interactive_selection=False,
    selection_save_path: Path = None,
):
    """
    Args:
        cams_tform4x4_world (Union[torch.Tensor, List[torch.Tensor]]): (Cx4x4) or List(4x4)
        cams_intr4x4 (Union[torch.Tensor, List[torch.Tensor]]): Cx4x4 or List(4x4)
        cams_imgs (Union[torch.Tensor, List[torch.Tensor]]): Cx3xHxW or List(3xHxW)
        cams_names: (List[str]): (P,)
        pts3d (Union[torch.Tensor, List[torch.Tensor]]): PxNx3 or List(Npx3)
        pts3d_names (List[str]): (P,)
        pts3d_colors (Union[torch.Tensor, List]): Px2x3 or List(3)
        lines3d (Union[torch.Tensor, List[torch.Tensor]]): PxNx2x3 or List(Npx2x3)
        lines3d_names (List[str]): (P,)
        lines3d_colors (Union[torch.Tensor, List]): Px2x3 or List(3)
        lines3d_radius (float): cylinder radius for lines (open3d only)
        lines3d_resolution (int): cylinder resolution for lines (open3d only)
        meshes (Meshes)
        meshes_names (List[str]): (M,)
        meshes_colors (Union[torch.Tensor, List]): Mx3 or List(3)
        renderer (OD3D_RENDERER): renderer to use
    Returns:
        -
    """

    geometries = []
    meshes_x_offsets = []
    meshes_z_offset = 0.0
    meshes_y_offset = 0.0
    if meshes is not None:
        if isinstance(meshes, List):
            meshes = Meshes.read_from_meshes(meshes)

        x_offset = 0.0
        for i in range(len(meshes)):
            vertices = meshes.get_verts_with_mesh_id(mesh_id=i).clone()
            if meshes_add_translation:
                x_offset_delta_current = 1.1 * (-vertices[:, 0].min()).clamp(min=0.0)
                if (
                    vertices[:, 2].max() - vertices[:, 2].min()
                ) * 1.1 > meshes_z_offset:
                    meshes_z_offset = (
                        vertices[:, 2].max() - vertices[:, 2].min()
                    ) * 1.1
                if (
                    vertices[:, 1].max() - vertices[:, 1].min()
                ) * 1.1 > meshes_y_offset:
                    meshes_y_offset = (
                        vertices[:, 1].max() - vertices[:, 1].min()
                    ) * 1.1
                x_offset += x_offset_delta_current
                x_offset_delta_next = 1.1 * (vertices[:, 0].max())
                vertices[:, 0] += x_offset

                meshes.set_verts_with_mesh_id(mesh_id=i, value=vertices)

                meshes_x_offsets.append(x_offset.item())

                x_offset += x_offset_delta_next

            faces = meshes.get_faces_with_mesh_id(mesh_id=i)

            if (
                meshes_colors is not None
                and len(meshes_colors) >= i + 1
                and meshes_colors[i] is not None
            ):
                mesh_color = meshes_colors[i]
                # if isinstance(mesh_color, torch.Tensor):
                #    mesh_color = mesh_color.detach().cpu().numpy()

                mesh_rgb = meshes.get_verts_ncds_with_mesh_id(mesh_id=i)
                mesh_rgb[:, 0] = mesh_color[0]
                mesh_rgb[:, 1] = mesh_color[1]
                mesh_rgb[:, 2] = mesh_color[2]
            else:
                if meshes.rgb is not None:
                    mesh_rgb = meshes.get_rgb_with_mesh_id(mesh_id=i)
                else:
                    mesh_rgb = meshes.get_verts_ncds_with_mesh_id(mesh_id=i)

            if (
                meshes_names is not None
                and len(meshes_names) >= i + 1
                and meshes_names[i] is not None
            ):
                mesh_name = meshes_names[i]
            else:
                mesh_name = f"mesh{i}"

            if renderer == OD3D_RENDERER.OPEN3D:
                # mesh_engine = open3d.geometry.TriangleMesh(
                #     vertices=open3d.utility.Vector3dVector(
                #         vertices.detach().cpu().numpy(),
                #     ),
                #     triangles=open3d.utility.Vector3iVector(
                #         faces.detach().cpu().numpy(),
                #     ),
                # )
                #
                # mesh_material = open3d.visualization.rendering.MaterialRecord()
                # mesh_material.shader = 'defaultUnlit'
                # #mesh_material.shader = "defaultLitTransparency"
                # # mesh_material.shader = 'defaultLitSSR'
                # # alpha = 0.5
                # mesh_material.base_reflectance = 0.
                # mesh_material.base_roughness = 0.0
                # mesh_material.absorption_color = [0.5, 0.5, 0.5]
                # vertex_colors = open3d.utility.Vector3dVector(
                #     mesh_rgb.detach().cpu().numpy(),
                # )
                # mesh_engine.vertex_colors = vertex_colors

                mesh_engine = meshes.to_o3d(meshes_ids=[i])

                if meshes_as_wireframe:
                    mesh_engine = o3d.geometry.LineSet.create_from_triangle_mesh(
                        mesh_engine,
                    )

            elif renderer == OD3D_RENDERER.PYRENDER:
                import pyrender

                mesh_engine = meshes.to_pyrender(meshes_ids=[i])

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                from pytorch3d.renderer import TexturesVertex as PT3D_TexturesVertex
                from pytorch3d.structures import Meshes as PT3D_Meshes

                textures = PT3D_TexturesVertex(
                    verts_features=mesh_rgb.to(device)[None,],
                )
                mesh_engine = PT3D_Meshes(
                    verts=[vertices.to(device=device)],
                    faces=[faces.to(device=device)],
                    textures=textures,
                )

                mesh_engine.rgb = mesh_rgb
                mesh_material = None
            else:
                raise NotImplementedError

            geometries.append(
                {
                    "name": mesh_name,
                    "geometry": mesh_engine,
                },  # , "material": mesh_material},
            )
            # vertices: open3d.cpu.pybind.utility.Vector3dVector,
            # triangles: open3d.cpu.pybind.utility.Vector3iVector

    if lines3d is not None:
        for i, lines3d_i in enumerate(lines3d):
            # N x 2 x 3
            _lines3d_i = lines3d_i.clone()
            _lines3d_i_pts3d = _lines3d_i.reshape(-1, 3)
            _lines3d_i_pts3d_ids = torch.arange(len(_lines3d_i_pts3d)).reshape(-1, 2)
            _lines3d_count = _lines3d_i.shape[0]

            if (
                lines3d_colors is not None
                and len(lines3d_colors) >= i + 1
                and lines3d_colors[i] is not None
            ):
                lines3d_i_color = lines3d_colors[i]
            else:
                lines3d_i_color = get_colors(len(lines3d))[i]

            if (
                lines3d_names is not None
                and len(lines3d_names) >= i + 1
                and lines3d_names[i] is not None
            ):
                line3d_name = lines3d_names[i]
            else:
                line3d_name = f"lines3d_{i}"

            if renderer == OD3D_RENDERER.OPEN3D:
                if lines3d_radius is not None:
                    line3d_engine = o3d.geometry.TriangleMesh()
                    lines3d_i_color_t = (
                        lines3d_i_color
                        if isinstance(lines3d_i_color, torch.Tensor)
                        else torch.tensor(lines3d_i_color)
                    )

                    def get_line_color(idx):
                        if isinstance(lines3d_i_color, list):
                            if (
                                len(lines3d_i_color) == _lines3d_count
                                and isinstance(lines3d_i_color[0], list)
                            ):
                                return lines3d_i_color_t[idx]
                            return lines3d_i_color_t
                        if (
                            isinstance(lines3d_i_color, torch.Tensor)
                            and lines3d_i_color.dim() == 1
                        ):
                            return lines3d_i_color_t
                        if (
                            isinstance(lines3d_i_color, torch.Tensor)
                            and lines3d_i_color.dim() == 2
                            and len(lines3d_i_color) == _lines3d_count
                        ):
                            return lines3d_i_color_t[idx]
                        return lines3d_i_color_t[0]

                    for line_idx in range(_lines3d_count):
                        p0 = _lines3d_i[line_idx, 0].detach().cpu().numpy()
                        p1 = _lines3d_i[line_idx, 1].detach().cpu().numpy()
                        direction = p1 - p0
                        height = np.linalg.norm(direction)
                        if height < 1e-9:
                            continue
                        direction /= height

                        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                            radius=lines3d_radius,
                            height=height,
                            resolution=lines3d_resolution,
                        )
                        cylinder.compute_vertex_normals()

                        z_axis = np.array([0.0, 0.0, 1.0])
                        v = np.cross(z_axis, direction)
                        s = np.linalg.norm(v)
                        c = np.dot(z_axis, direction)
                        if s < 1e-9:
                            if c < 0:
                                rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
                                    np.array([1.0, 0.0, 0.0]) * math.pi,
                                )
                            else:
                                rot = np.eye(3)
                        else:
                            axis = v / s
                            angle = math.atan2(s, c)
                            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
                                axis * angle,
                            )
                        cylinder.rotate(rot, center=(0.0, 0.0, 0.0))
                        cylinder.translate((p0 + p1) / 2.0)

                        color = get_line_color(line_idx).detach().cpu().numpy()
                        cylinder.paint_uniform_color(color.tolist())
                        line3d_engine += cylinder
                else:
                    line3d_engine = open3d.geometry.LineSet()
                    line3d_engine.points = o3d.utility.Vector3dVector(
                        _lines3d_i_pts3d.detach().cpu().numpy(),
                    )
                    line3d_engine.lines = o3d.utility.Vector2iVector(
                        _lines3d_i_pts3d_ids.detach().cpu().numpy(),
                    )
                    if isinstance(lines3d_i_color, list) or lines3d_i_color.dim() == 1:
                        line3d_engine.paint_uniform_color(
                            (
                                lines3d_i_color[0],
                                lines3d_i_color[1],
                                lines3d_i_color[2],
                            ),
                        )
                    else:
                        line3d_engine.colors = o3d.utility.Vector3dVector(
                            lines3d_i_color.detach().cpu().numpy(),
                        )
            elif renderer == OD3D_RENDERER.PYRENDER:
                import pyrender

                pass
                #  0: POINTS, 1: LINES, 2: LINE_LOOP, 3: LINE_STRIP,
                #  4: TRIANGLES, 5: TRIANGLES_STRIP, 6: TRIANGLES_FAN
                # pyrender.Primitive.POINTS
                # pyrender.Primitive(positions=, normals=None, tangents=None, mode=)
                line3d_engine = None

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                line3d_engine = None
            else:
                raise NotImplementedError

            geometries.append({"name": line3d_name, "geometry": line3d_engine})

    if pts3d is not None:
        x_offset = 0.0
        for i, pts3d_i in enumerate(pts3d):
            _pts3d_i = pts3d_i.clone()
            if pts3d_add_translation:
                _pts3d_i[:, 1] += meshes_y_offset
                # _pts3d_i[:, 2] += meshes_z_offset
                if len(meshes_x_offsets) > i:
                    x_offset = meshes_x_offsets[i]
                else:
                    x_offset_delta_current = 1.1 * (-_pts3d_i[:, 0].min()).clamp(
                        min=0.0,
                    )
                    x_offset += x_offset_delta_current
                if _pts3d_i[:, 0].numel() == 0:
                    x_offset_delta_next = 0.0
                else:
                    x_offset_delta_next = 1.1 * (_pts3d_i[:, 0].max())
                _pts3d_i[:, 0] += x_offset
                x_offset += x_offset_delta_next

            if (
                pts3d_colors is not None
                and len(pts3d_colors) >= i + 1
                and pts3d_colors[i] is not None
            ):
                pts3d_i_color = pts3d_colors[i]
            else:
                pts3d_i_color = get_colors(len(pts3d))[i]

            if (
                pts3d_names is not None
                and len(pts3d_names) >= i + 1
                and pts3d_names[i] is not None
            ):
                pts3d_name = pts3d_names[i]
            else:
                pts3d_name = f"pts3d_{i}"

            if renderer == OD3D_RENDERER.OPEN3D:
                pts3d_engine = open3d.geometry.PointCloud()
                pts3d_engine.points = open3d.utility.Vector3dVector(
                    _pts3d_i.detach().cpu().numpy(),
                )
                if (
                    pts3d_normals is not None
                    and len(pts3d_normals) >= i + 1
                    and pts3d_normals[i] is not None
                ):
                    pts3d_engine.normals = open3d.utility.Vector3dVector(
                        pts3d_normals[i].detach().cpu().numpy(),
                    )
                if isinstance(pts3d_i_color, list) or pts3d_i_color.dim() == 1:
                    pts3d_engine.paint_uniform_color(
                        (pts3d_i_color[0], pts3d_i_color[1], pts3d_i_color[2]),
                    )
                else:
                    pts3d_engine.colors = o3d.utility.Vector3dVector(
                        pts3d_i_color.detach().cpu().numpy(),
                    )

            elif renderer == OD3D_RENDERER.PYRENDER:
                import pyrender

                # mode
                #  0: POINTS, 1: LINES, 2: LINE_LOOP, 3: LINE_STRIP,
                #  4: TRIANGLES, 5: TRIANGLES_STRIP, 6: TRIANGLES_FAN
                # pyrender.Primitive.POINTS

                if isinstance(pts3d_i_color, list) or pts3d_i_color.dim() == 1:
                    _pts3d_i_color = _pts3d_i.clone()
                    _pts3d_i_color[..., 0] = pts3d_i_color[0]
                    _pts3d_i_color[..., 1] = pts3d_i_color[1]
                    _pts3d_i_color[..., 2] = pts3d_i_color[2]
                else:
                    _pts3d_i_color = pts3d_i_color
                pts3d_engine = pyrender.Primitive(
                    positions=_pts3d_i.detach().cpu().numpy(),
                    color_0=_pts3d_i_color.detach().cpu().numpy(),
                    mode=0,
                )

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                from pytorch3d.structures import Pointclouds as PT3D_Pointclouds

                if isinstance(pts3d_i_color, list) or pts3d_i_color.dim() == 1:
                    _pts3d_i_rgb = _pts3d_i.clone().to(device=device)
                    _pts3d_i_rgb[:, 0] = pts3d_i_color[0]
                    _pts3d_i_rgb[:, 1] = pts3d_i_color[1]
                    _pts3d_i_rgb[:, 2] = pts3d_i_color[2]
                else:
                    _pts3d_i_rgb = pts3d_i_color.to(device=device)

                _pts3d_i_normals = (
                    [pts3d_normals[i].to(device=device)] if pts3d_normals is not None else None
                )
                pts3d_engine = PT3D_Pointclouds(
                    points=[_pts3d_i.to(device=device)],
                    normals=_pts3d_i_normals,
                    features=[_pts3d_i_rgb],
                )
            else:
                raise NotImplementedError

            geometries.append({"name": pts3d_name, "geometry": pts3d_engine})

    if show_coordinate_frame:
        if renderer == OD3D_RENDERER.OPEN3D:
            if meshes is not None:
                min = meshes.verts.min(dim=0).values
                max = meshes.verts.max(dim=0).values
                size = (max - min).min().item()
                origin = min - size / 4.0
                origin = origin.detach().cpu().numpy()
            else:
                size = 1.0
                origin = np.array([0.0, 0.0, 0.0])
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size,
                origin=origin,
            )
            geometries.append(
                {"name": "coordinate_frame", "geometry": coordinate_frame},
            )

    engine_geometries_for_cams = get_engine_geometries_for_cams(
        cams_tform4x4_world=cams_tform4x4_world,
        cams_intr4x4=cams_intr4x4,
        cams_imgs=cams_imgs,
        cams_names=cams_names,
        cams_imgs_resize=cams_imgs_resize,
        cams_imgs_depth_scale=cams_imgs_depth_scale,
        cams_show_wireframe=cams_show_wireframe,
        cams_show_image_encoder=cams_show_image_encoder,
        renderer=renderer,
        device=device,
    )

    for engine_geometry_for_cam in engine_geometries_for_cams:
        geometries.append(engine_geometry_for_cam)

    if return_visualization is False and fpath is None:
        if os.environ.get("DISPLAY"):
            if renderer == OD3D_RENDERER.OPEN3D:
                # advantage: transparent
                # open3d.visualization.draw(
                #     geometries,
                #     show_skybox=False,
                #     bg_color=[1.0, 1.0, 1.0, 1.0],
                #     raw_mode=True,
                # )

                # advantage: normals shown
                open3d.visualization.draw_geometries(
                    [geometry["geometry"] for geometry in geometries],
                )

            elif renderer == OD3D_RENDERER.PYRENDER:
                import pyrender
                from pyrender import DirectionalLight

                # Create scene and add mesh
                scene = pyrender.Scene(
                    ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
                    bg_color=background_color,
                )
                # 'f': fullscreen, 'w': wireframe, 'a': rotation
                # m = pyrender.Mesh.from_points(pts, colors=colors)
                for geometry in geometries:
                    if isinstance(geometry["geometry"], pyrender.mesh.Mesh):
                        scene.add(geometry["geometry"])
                    elif isinstance(geometry["geometry"], pyrender.camera.Camera):
                        scene.add(
                            geometry["geometry"],
                            pose=geometry["obj_tform4x4_cam"],
                        )
                    elif isinstance(geometry["geometry"], pyrender.mesh.Primitive):
                        scene.add(pyrender.Mesh([geometry["geometry"]]))

                # Create pyrender camera from intrinsic matrix
                # fx, fy = cam_intr4x4[0, 0], cam_intr4x4[1, 1]
                # cx, cy = cam_intr4x4[0, 2], cam_intr4x4[1, 2]
                # height = H
                # width = W
                # camera = pyrender.IntrinsicsCamera(float(fx), float(fy), float(cx), float(cy), znear=znear, zfar=zfar)
                # FOLLOWING OPENGL convention
                pyrender_cam_tform4x4_obj = OPEN3D_CAM_TFORM_OBJ.clone()
                # pyrender_cam_tform4x4_obj = tform4x4(
                #    OPEN3D_DEFAULT_CAM_TFORM_OBJ.clone().to(device=cam_tform4x4_obj.device), cam_tform4x4_obj, )
                from o3b.cv.geometry.transform import inv_tform4x4

                obj_tform4x4_pyrender_cam = inv_tform4x4(pyrender_cam_tform4x4_obj)
                obj_tform4x4_pyrender_cam_np = (
                    obj_tform4x4_pyrender_cam.detach().cpu().numpy()
                )
                # scene.add(camera, pose=obj_tform4x4_pyrender_cam_np)
                direc_l = DirectionalLight(color=np.ones(3), intensity=2.0)

                direc_l_node = scene.add(direc_l, pose=obj_tform4x4_pyrender_cam_np)

                pyrender.Viewer(scene, point_size=pts3d_size)
                # pyrender.Viewer(scene, shadows=True)

                # Create an offscreen renderer
                # if 'DISPLAY' not in os.environ:
                #    os.environ['PYOPENGL_PLATFORM'] = 'egl'
                # renderer = pyrender.OffscreenRenderer(width, height)
                # Render the scene
                # color, depth = renderer.render(scene)

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs

                cam_tform4x4_obj = CAM_TFORM_OBJ.clone().to(
                    dtype=dtype,
                    device=device,
                )
                cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
                    H=H,
                    W=W,
                    dtype=dtype,
                    device=device,
                )
                img_size = torch.Tensor([H, W]).to(dtype=dtype, device=device)

                pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(
                    cam_tform4x4_obj=cam_tform4x4_obj,
                    cam_intr4x4=cam_intr4x4,
                    img_size=img_size,
                )

                plotly_background_color = f"rgb({int(255*background_color[0])}, {int(255*background_color[1])}, {int(255*background_color[2])})"

                fig = plot_scene(
                    {
                        "default": {
                            geometry["name"]: geometry["geometry"]
                            for geometry in geometries
                        },
                    },
                    viewpoint_cameras=pt3d_cameras,
                    pointcloud_marker_size=pts3d_size,
                    axis_args=AxisArgs(
                        backgroundcolor=plotly_background_color,
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        showaxeslabels=True,
                        showticklabels=True,
                    ),
                )

                fig.show()
        else:
            logger.warning(
                "could not visualize with open3d, because env DISPLAY not set, try `export DISPLAY=:0.0;`",
            )
            return
    else:
        imgs = []
        from o3b.io import is_fpath_video
        from o3b.cv.geometry.transform import (
            get_cam_tform4x4_obj_for_viewpoints_count,
            inv_tform4x4,
        )

        if render_cam_tform4x4_obj is not None:
            cams_new_tform4x4_obj = render_cam_tform4x4_obj.to(
                dtype=dtype,
                device=device,
            )

            # viewpoint_init_dist = cams_new_tform4x4_obj[:, 2, 3]
        else:
            cams_new_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(
                viewpoints_count=viewpoints_count,
                dist=viewpoint_init_dist,
                spiral=is_fpath_video(fpath),
            ).to(dtype=dtype, device=device)

        # if (
        #    os.environ.get("DISPLAY")
        #    or open3d._build_config["ENABLE_HEADLESS_RENDERING"]
        # ):
        if renderer == OD3D_RENDERER.OPEN3D and os.environ.get("DISPLAY"):
            cams_new_tform4x4_obj[:, 2, 3] /= viewpoint_init_dist
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, height=H, width=W)

            opt = vis.get_render_option()
            opt.point_size = pts3d_size
            opt.mesh_show_back_face = False
            opt.show_coordinate_frame = show_coordinate_frame
            opt.background_color = np.asarray(background_color)
            # opt.background_color = np.asarray([0, 0, 0])
            # opt.mesh_show_wireframe = True #  mesh_show_wireframe
            geometries_vertices_orig = []
            for geometry in geometries:
                vis.add_geometry(geometry["geometry"])
                if isinstance(geometry["geometry"], open3d.geometry.PointCloud):
                    geometries_vertices_orig.append(
                        torch.from_numpy(np.asarray(geometry["geometry"].points))
                        .clone()
                        .to(device=device, dtype=dtype),
                    )
                elif isinstance(geometry["geometry"], open3d.geometry.TriangleMesh):
                    geometries_vertices_orig.append(
                        torch.from_numpy(np.asarray(geometry["geometry"].vertices))
                        .clone()
                        .to(device=device, dtype=dtype),
                    )
                elif isinstance(geometry["geometry"], open3d.geometry.LineSet):
                    geometries_vertices_orig.append(
                        torch.from_numpy(np.asarray(geometry["geometry"].points))
                        .clone()
                        .to(device=device, dtype=dtype),
                    )
                else:
                    geometries_vertices_orig.append(None)
                # vis.update_geometry(geometry['geometry'])
            vis.poll_events()
            vis.update_renderer()
            # view_control = vis.get_view_control()

            # open3d version 0.17.0 bug, view control does not work
            # camera_orig = view_control.convert_to_pinhole_camera_parameters()
            # cam_tform4x4_obj = torch.from_numpy(camera_orig.extrinsic).to(dtype=objs_new_tform4x4_obj.dtype, device=objs_new_tform4x4_obj.device)
            from tqdm import tqdm

            for v in tqdm(range(viewpoints_count)):
                # camera_orig.extrinsic = tform4x4_broadcast(cam_tform4x4_obj,
                #                                           objs_new_tform4x4_obj[v]).detach().cpu().numpy()
                # view_control.convert_from_pinhole_camera_parameters(camera_orig)

                for g, geometry in enumerate(geometries):
                    if geometries_vertices_orig[g] is not None:
                        vertices = geometries_vertices_orig[
                            g
                        ]  # .to(dtype=objs_new_tform4x4_obj.dtype, device=objs_new_tform4x4_obj.device)
                        vertices = transf3d_broadcast(
                            pts3d=vertices,
                            transf4x4=tform4x4(
                                OPEN3D_CAM_TFORM_CAM.to(
                                  dtype=dtype,
                                  device=device,
                                ),
                                cams_new_tform4x4_obj[v],
                                # OPEN3D_CAM_TFORM_CAM
                                #inv_tform4x4(OPEN3D_OBJ_TFORM_OBJ.to(
                                #    dtype=dtype,
                                #    device=device,
                                #)),
                            ),
                        )

                        if isinstance(
                            geometry["geometry"],
                            open3d.geometry.PointCloud,
                        ):
                            geometry["geometry"].points = open3d.utility.Vector3dVector(
                                vertices.detach().cpu().numpy(),
                            )
                        elif isinstance(
                            geometry["geometry"],
                            open3d.geometry.TriangleMesh,
                        ):
                            geometry[
                                "geometry"
                            ].vertices = open3d.utility.Vector3dVector(
                                vertices.detach().cpu().numpy(),
                            )
                        elif isinstance(
                            geometry["geometry"],
                            open3d.geometry.LineSet,
                        ):
                            geometry["geometry"].points = open3d.utility.Vector3dVector(
                                vertices.detach().cpu().numpy(),
                            )
                        vis.update_geometry(geometry["geometry"])

                vis.update_renderer()
                img = vis.capture_screen_float_buffer(do_render=True)
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                if crop_white_border:
                    from o3b.cv.visual.crop import crop_white_border_from_img

                    img = crop_white_border_from_img(img, resize_to_orig=True)
                imgs.append(img)

            vis.update_renderer()
            vis.destroy_window()

        elif renderer == OD3D_RENDERER.PYTORCH3D:
            from o3b.cv.geometry.fit.depth_from_mesh_and_box import (
                get_scale_bbox_pts3d_to_image,
            )
            from pytorch3d.structures import Pointclouds as PT3D_Pointclouds
            from pytorch3d.renderer.cameras import PerspectiveCameras

            pts3d = []  # [for g, geometry in enumerate(geometries)]
            for g, geometry in enumerate(geometries):
                if isinstance(geometry["geometry"], PT3D_Pointclouds):
                    pts3d.append(geometry["geometry"].points_packed().to(device=device))
                elif isinstance(geometry["geometry"], PT3D_Meshes):
                    pts3d.append(geometry["geometry"].verts_packed().to(device=device))
                elif isinstance(geometry["geometry"], PerspectiveCameras):
                    pts3d.append(geometry["geometry"].get_camera_center().to(device=device))
                else:
                    logger.info(f'geometry not supported {type(geometry["geometry"])}')
            pts3d = torch.cat(pts3d, dim=0)
            cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
                H=H,
                W=W,
                dtype=dtype,
                device=device,
            )

            scale = get_scale_bbox_pts3d_to_image(
                cam_intr4x4=cam_intr4x4,
                cam_tform4x4_obj=cams_new_tform4x4_obj[0],
                pts3d=pts3d,
                img_width=W,
                img_height=H,
            )
            cams_new_tform4x4_obj[:, 2, 3] *= (
                scale.max(dim=-1).values / 2.0
            )  # plot_scene inside pytorch3d does scale otherwise

            img_size = torch.Tensor([H, W]).to(dtype=dtype, device=device)

            plotly_background_color = f"rgb({int(255 * background_color[0])}, {int(255 * background_color[1])}, {int(255 * background_color[2])})"

            plotly_geometries = {
                geometry["name"]: geometry["geometry"] for geometry in geometries
            }

            from tqdm import tqdm
            from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs

            for v in tqdm(range(viewpoints_count)):
                pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(
                    cam_tform4x4_obj=cams_new_tform4x4_obj[v],
                    cam_intr4x4=cam_intr4x4,
                    img_size=img_size,
                )

                fig = plot_scene(
                    {
                        f"": plotly_geometries,  # view{v}
                    },
                    viewpoint_cameras=pt3d_cameras,
                    pointcloud_marker_size=pts3d_size,
                    axis_args=AxisArgs(
                        backgroundcolor=plotly_background_color,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showaxeslabels=False,
                        showticklabels=False,
                    ),
                )
                fig.update_layout(showlegend=False)
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                    ),
                )

                for a in range(len(fig.layout.annotations)):
                    fig.layout.annotations[a].update(
                        text="",
                    )  # a=1, remove text from subplot
                img = plotly_fig_2_tensor(fig, width=W, height=H)
                if crop_white_border:
                    from o3b.cv.visual.crop import crop_white_border_from_img

                    img = crop_white_border_from_img(img, resize_to_orig=True)
                imgs.append(img)
        elif renderer == OD3D_RENDERER.PYRENDER:
            # elif renderer == OD3D_RENDERER.PYRENDER:
            import pyrender
            from pyrender import DirectionalLight

            # Create scene and add mesh
            scene = pyrender.Scene(
                ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
                bg_color=background_color,
            )
            # 'f': fullscreen, 'w': wireframe, 'a': rotation
            # m = pyrender.Mesh.from_points(pts, colors=colors)
            for geometry in geometries:
                if isinstance(geometry["geometry"], pyrender.mesh.Mesh):
                    scene.add(geometry["geometry"])
                elif isinstance(geometry["geometry"], pyrender.camera.Camera):
                    scene.add(
                        geometry["geometry"],
                        pose=geometry["obj_tform4x4_cam"],
                    )
                elif isinstance(geometry["geometry"], pyrender.mesh.Primitive):
                    scene.add(pyrender.Mesh([geometry["geometry"]]))

            # Create pyrender camera from intrinsic matrix
            # fx, fy = cam_intr4x4[0, 0], cam_intr4x4[1, 1]
            # cx, cy = cam_intr4x4[0, 2], cam_intr4x4[1, 2]
            # height = H
            # width = W
            # camera = pyrender.IntrinsicsCamera(float(fx), float(fy), float(cx), float(cy), znear=znear, zfar=zfar)
            # FOLLOWING OPENGL convention

            imgs = []
            for cam_new_tform4x4_obj in cams_new_tform4x4_obj:
                # pyrender_cam_tform4x4_obj = OPEN3D_CAM_TFORM_OBJ.clone()
                pyrender_cam_tform4x4_obj = cam_new_tform4x4_obj.clone()
                # pyrender_cam_tform4x4_obj = tform4x4(
                #    OPEN3D_DEFAULT_CAM_TFORM_OBJ.clone().to(device=cam_tform4x4_obj.device), cam_tform4x4_obj, )
                from o3b.cv.geometry.transform import inv_tform4x4

                obj_tform4x4_pyrender_cam = inv_tform4x4(pyrender_cam_tform4x4_obj)
                obj_tform4x4_pyrender_cam_np = (
                    obj_tform4x4_pyrender_cam.detach().cpu().numpy()
                )
                # scene.add(camera, pose=obj_tform4x4_pyrender_cam_np)
                direc_l = DirectionalLight(color=np.ones(3), intensity=2.0)

                direc_l_node = scene.add(direc_l, pose=obj_tform4x4_pyrender_cam_np)

                #pyrender.Viewer(scene, point_size=pts3d_size)
                # pyrender.Viewer(scene, shadows=True)

                # Create an offscreen renderer
                if 'DISPLAY' not in os.environ:
                   os.environ['PYOPENGL_PLATFORM'] = 'egl'
                renderer = pyrender.OffscreenRenderer(W, H)
                # Render the scene
                color, depth = renderer.render(scene)

                logger.info(color.shape)
                imgs.append(color)

        else:
            logger.warning(
                "could not visualize with open3d, most likely env DISPLAY not set, try `export DISPLAY=:0.0;` (maybe without ;)",
            )

            return [
                torch.zeros(size=(3, 480, 640)).to(device=device),
            ] * viewpoints_count

        if viewpoints_count == 1:
            imgs = imgs[0]
        else:
            if viewpoints_count > 4 and not is_fpath_video(fpath):
                viewpoints_count_sqrt = math.ceil(math.sqrt(viewpoints_count))
                imgs_placeholder = torch.zeros(
                    size=(viewpoints_count_sqrt**2, 3, H, W),
                    dtype=dtype,
                    device=device,
                )
                imgs_placeholder[:viewpoints_count] = torch.stack(imgs, dim=0)
                imgs = imgs_placeholder
                imgs = imgs.reshape(
                    viewpoints_count_sqrt,
                    viewpoints_count_sqrt,
                    3,
                    H,
                    W,
                )

                logger.info(imgs.shape)
            else:
                imgs = torch.stack(imgs, dim=0)

        if fpath is not None:
            if is_fpath_video(fpath):
                from o3b.cv.visual.video import save_video

                save_video(imgs=imgs, fpath=fpath, fps=fps)
            else:
                show_imgs(rgbs=imgs, fpath=fpath, pad=0)

        if return_visualization:
            return imgs
        else:
            return 0


def get_engine_geometries_for_cams(
    cams_tform4x4_world: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_intr4x4: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_imgs: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_names: List[str] = None,
    cams_imgs_resize: bool = True,
    cams_imgs_depth_scale: float = 0.2,
    cams_show_wireframe: bool = True,
    cams_show_image_encoder: bool = False,
    renderer=OD3D_RENDERER.OPEN3D,
    device="cpu",
):
    """
    Args:
        cams_tform4x4_world (Union[torch.Tensor, List[torch.Tensor]]): (Cx4x4) or List(4x4)
        cams_intr4x4 (Union[torch.Tensor, List[torch.Tensor]]): Cx4x4 or List(4x4)
        cams_imgs (Union[torch.Tensor, List[torch.Tensor]]): Cx3xHxW or List(3xHxW)
        cams_names: (List[str]): (P,)
    Returns:
        od3d_geometries (List): list with geometries as dict
    """

    geometries = []

    if cams_tform4x4_world is not None and cams_intr4x4 is not None:
        for i in range(len(cams_tform4x4_world)):
            if isinstance(cams_intr4x4, torch.Tensor) and cams_intr4x4.dim() == 2:
                cam_intr4x4 = cams_intr4x4
            elif len(cams_intr4x4) == 1:
                cam_intr4x4 = cams_intr4x4[0]
            else:
                cam_intr4x4 = cams_intr4x4[i]

            width = int(cam_intr4x4[0, 2] * 2)
            height = int(cam_intr4x4[1, 2] * 2)

            if (
                cams_names is not None
                and len(cams_names) >= i + 1
                and cams_names[i] is not None
            ):
                cam_name = cams_names[i]
            else:
                cam_name = f"cam{i}"

            cam_wireframe_engine = None
            obj_tform4x4_cam = None

            if cams_imgs is not None and len(cams_imgs) > i:
                h, w = cams_imgs[i].shape[1:]

                width = int(w) # int(cam_intr4x4[0, 2] * 2)
                height = int(h) # int(cam_intr4x4[1, 2] * 2)


                if cams_imgs_resize:
                    cam_img = resize(cams_imgs[i], H_out=256, W_out=256)
                    h_resize = 256.0 / h
                    w_resize = 256.0 / w
                else:
                    cam_img = cams_imgs[i]
                    h_resize = 1.0
                    w_resize = 1.0

                if cam_img.dtype == torch.float:
                    cam_img = (cam_img.clone() * 255).to(dtype=torch.uint8)

                cam_intr4x4_res = cam_intr4x4.clone()
                cam_intr4x4_res[0, 0] = cam_intr4x4_res[0, 0].item() * w_resize
                cam_intr4x4_res[1, 1] = cam_intr4x4_res[1, 1].item() * h_resize
                cam_intr4x4_res[0, 2] = cam_intr4x4_res[0, 2].item() * w_resize
                cam_intr4x4_res[1, 2] = cam_intr4x4_res[1, 2].item() * h_resize

                if renderer == OD3D_RENDERER.OPEN3D:
                    cam_tform4x4_obj = cams_tform4x4_world[i].detach().cpu().numpy()
                    depth_scale = (
                        cams_imgs_depth_scale * cam_tform4x4_obj[2, 3]
                    )  #  * 10 * cam_tform4x4_obj[2, 3]
                    depth = open3d.geometry.Image(
                        (
                            (torch.ones(size=cam_img.shape[1:])).cpu().detach().numpy()
                            * 255
                        ).astype(np.uint8),
                    )
                    img = open3d.geometry.Image(
                        (
                            cam_img.permute(1, 2, 0).contiguous().cpu().detach().numpy()
                        ).astype(np.uint8),
                    )
                    fx = cam_intr4x4_res[0, 0].item()
                    fy = cam_intr4x4_res[1, 1].item()
                    cx = cam_intr4x4_res[0, 2].item()
                    cy = cam_intr4x4_res[1, 2].item()
                    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                        color=img,
                        depth=depth,
                        depth_scale=1 / depth_scale,
                        depth_trunc=3 * depth_scale,
                        convert_rgb_to_intensity=False,
                    )

                    intrinsic = open3d.camera.PinholeCameraIntrinsic(
                        w,
                        h,
                        fx,
                        fy,
                        cx,
                        cy,
                    )
                    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    cam = open3d.camera.PinholeCameraParameters()
                    cam.intrinsic = intrinsic
                    cam.extrinsic = cam_tform4x4_obj  # cams_tform4x4_world[i].detach().cpu().numpy()

                    pts3d_engine = open3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd,
                        cam.intrinsic,
                        cam.extrinsic,
                    )
                    if cams_show_wireframe:
                        cam_wireframe_engine = (
                            open3d.geometry.LineSet.create_camera_visualization(
                                view_width_px=width,
                                view_height_px=height,
                                intrinsic=cam_intr4x4[:3, :3].detach().cpu().numpy(),
                                extrinsic=cams_tform4x4_world[i].detach().cpu().numpy(),
                                scale=cams_imgs_depth_scale
                                * cams_tform4x4_world[i].detach().cpu().numpy()[2, 3],
                            )
                        )
                    if cams_show_image_encoder:
                        from o3b.cv.geometry.primitives import ImageEncoder

                        image_encoder_downsample_rate = 1.5
                        cam_wireframe_engine = ImageEncoder.init_with_cam(
                            cam_intr4x4=cam_intr4x4.detach().cpu().numpy(),
                            cam_tform4x4_obj=cams_tform4x4_world[i].detach(),
                            img_size=cams_imgs[i].shape[1:],  # H, W
                            depth_min=depth_scale * 1.05,
                            depth_max=depth_scale * 1.3,
                            downscale_factor=image_encoder_downsample_rate,
                        )
                        cam_wireframe_engine = cam_wireframe_engine.to_o3d()
                        # H, W, 3
                        cam_img = get_colors(
                            cam_img.shape[1:].numel(),
                            randperm=True,
                        ).reshape(*cam_img.shape[1:], 3)
                        cam_img *= 255.0
                        img = open3d.geometry.Image(
                            (cam_img.contiguous().cpu().detach().numpy()).astype(
                                np.uint8,
                            ),
                        )

                        feats_scale = 1.0 / 1.35
                        depth_scale /= feats_scale
                        intrinsic.intrinsic_matrix = [
                            [fx / (feats_scale / image_encoder_downsample_rate), 0, cx],
                            [0, fy / (feats_scale / image_encoder_downsample_rate), cy],
                            [0, 0, 1],
                        ]
                        cam = open3d.camera.PinholeCameraParameters()
                        cam.intrinsic = intrinsic
                        cam.extrinsic = cam_tform4x4_obj  # cams_tform4x4_world[i].detach().cpu().numpy()

                        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                            color=img,
                            depth=depth,
                            depth_scale=1 / depth_scale,
                            depth_trunc=3 * depth_scale,
                            convert_rgb_to_intensity=False,
                        )

                        pts3d_feats = open3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd,
                            cam.intrinsic,
                            cam.extrinsic,
                        )
                        pts3d_engine += pts3d_feats
                elif renderer == OD3D_RENDERER.PYRENDER:
                    import pyrender

                    # Create pyrender camera from intrinsic matrix
                    fx = cam_intr4x4_res[0, 0].item()
                    fy = cam_intr4x4_res[1, 1].item()
                    cx = cam_intr4x4_res[0, 2].item()
                    cy = cam_intr4x4_res[1, 2].item()
                    z_far = cams_tform4x4_world[i][2, 3].item() * cams_imgs_depth_scale

                    #  * 10 * cam_tform4x4_obj[2, 3]
                    # fx, fy = K[0, 0], K[1, 1]
                    # cx, cy = K[0, 2], K[1, 2]

                    # Image corners in pixel coordinates
                    # w_resize, width
                    # h_resize, height
                    img_corners = np.array(
                        [
                            [0, 0],
                            [width * w_resize - 1, 0],
                            [width * w_resize - 1, height * h_resize - 1],
                            [0, height * h_resize - 1],
                        ],
                    )

                    # Backproject to 3D at depth z_far
                    corners_3d = []
                    for u, v in img_corners:
                        x = (u - cx) * z_far / fx
                        y = (v - cy) * z_far / fy
                        corners_3d.append([x, y, z_far])
                    corners_3d = np.array(corners_3d)

                    # Camera origin
                    origin = np.array([[0, 0, 0]])
                    # Lines from origin to corners and between corners
                    lines = []
                    for corner in corners_3d:
                        lines.append(origin[0])
                        lines.append(corner)

                    # Connect corners in a loop (to form rectangle at far plane)
                    for j in range(4):
                        lines.append(corners_3d[j])
                        lines.append(corners_3d[(j + 1) % 4])

                    # Create mesh
                    import trimesh

                    # colors = np.tile(color, (len(lines), 1))
                    # frustum = trimesh.load_path(np.array(lines).reshape(-1, 2, 3))

                    segments = np.array(lines).reshape(-1, 2, 3)
                    cylinders = []
                    for start, end in segments:
                        cylinder = trimesh.creation.cylinder(
                            radius=0.002,
                            segment=[start, end],
                            sections=4,
                        )
                        cylinders.append(cylinder)
                    frustum_mesh = trimesh.util.concatenate(cylinders)

                    # texture =  cam_img.permute(1, 2, 0).contiguous().cpu().detach().numpy().astype(np.uint8)
                    # texture = cam_img.contiguous().cpu().detach().numpy()  # / 255.

                    from torchvision.transforms.functional import to_pil_image

                    texture = to_pil_image(cam_img.clone().flip(dims=(1,)))

                    # Define 4 vertices (rectangle in X-Y plane, Z=0)
                    vertices = np.array(
                        [
                            corners_3d[0],  # Bottom-left
                            corners_3d[1],  # Bottom-right
                            corners_3d[2],  # Top-right
                            corners_3d[3],  # Top-left
                            # [-1, -1, 0],  # Bottom-left
                            # [1, -1, 0],  # Bottom-right
                            # [1, 1, 0],  # Top-right
                            # [-1, 1, 0],  # Top-left
                        ],
                    )

                    # Define two triangular faces (using the 4 vertices)
                    faces = np.array(
                        [
                            [0, 1, 2],
                            [0, 2, 3],
                            [2, 1, 0],  # Back face 1 (reversed)
                            [3, 2, 0],  # Back face 2 (reversed)
                        ],
                    )

                    # Define UV coordinates (match image corners)
                    uv = np.array(
                        [
                            [0.0, 0.0],  # Bottom-left
                            [1.0, 0.0],  # Bottom-right
                            [1.0, 1.0],  # Top-right
                            [0.0, 1.0],  # Top-left
                        ],
                    )

                    # Create the texture visual
                    # material = trimesh.visual.texture.SimpleMaterial(image=texture)
                    # visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture, material=material)
                    # frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)
                    # frustum_mesh.show()
                    # Create the mesh
                    if cams_show_wireframe:
                        vertices = np.concatenate([vertices, frustum_mesh.vertices])
                        faces = np.concatenate([faces, frustum_mesh.faces + len(faces)])
                        uv = np.concatenate([uv, frustum_mesh.vertices.copy()[:, :2] * 0.0])

                    material = trimesh.visual.texture.SimpleMaterial(image=texture)
                    visual = trimesh.visual.texture.TextureVisuals(
                        uv=uv,
                        image=texture,
                        material=material,
                    )

                    frustum_mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        visual=visual,
                    )
                    # frustum_mesh.show()

                    # # FOLLOWING OPENGL convention
                    pyrender_cam_tform4x4_obj = OPEN3D_CAM_TFORM_OBJ.clone()
                    # pyrender_cam_tform4x4_obj = tform4x4(OPEN3D_DEFAULT_CAM_TFORM_OBJ.clone().to(device=cam_tform4x4_obj.device), cam_tform4x4_obj, )
                    from o3b.cv.geometry.transform import inv_tform4x4

                    obj_tform4x4_pyrender_cam = inv_tform4x4(pyrender_cam_tform4x4_obj)
                    obj_tform4x4_pyrender_cam_np = (
                        obj_tform4x4_pyrender_cam.detach().cpu().numpy()
                    )
                    obj_tform4x4_cam = obj_tform4x4_pyrender_cam_np

                    frustum_mesh = frustum_mesh
                    # inv_tform4x4(cams_tform4x4_world[i]).to(device=device, dtype=dtype)
                    dtype = cams_tform4x4_world[i].dtype
                    frustum_mesh.vertices = transf3d_broadcast(
                        torch.from_numpy(frustum_mesh.vertices).to(dtype=dtype),
                        transf4x4=inv_tform4x4(cams_tform4x4_world[i]),
                    )

                    pts3d_engine = pyrender.Mesh.from_trimesh(frustum_mesh)

                    # #fx, fy = cam_intr4x4[0, 0], cam_intr4x4[1, 1]
                    # #cx, cy = cam_intr4x4[0, 2], cam_intr4x4[1, 2]
                    # #height = H
                    # #width = W
                    #
                    # camera = pyrender.IntrinsicsCamera(float(fx), float(fy), float(cx), float(cy), znear=0.1, zfar=1.)
                    # pts3d_engine = camera
                    # # FOLLOWING OPENGL convention
                    # pyrender_cam_tform4x4_obj = OPEN3D_DEFAULT_CAM_TFORM_OBJ.clone()
                    # # pyrender_cam_tform4x4_obj = tform4x4(
                    # #    OPEN3D_DEFAULT_CAM_TFORM_OBJ.clone().to(device=cam_tform4x4_obj.device), cam_tform4x4_obj, )
                    # from o3b.cv.geometry.transform import inv_tform4x4
                    # obj_tform4x4_pyrender_cam = inv_tform4x4(pyrender_cam_tform4x4_obj)
                    # obj_tform4x4_pyrender_cam_np = obj_tform4x4_pyrender_cam.detach().cpu().numpy()
                    # obj_tform4x4_cam = obj_tform4x4_pyrender_cam_np
                    #
                    # # scene.add(camera, pose=obj_tform4x4_pyrender_cam_np)
                    #
                    # #pts3d_engine
                    # #cams_tform4x4_world[i]
                    # #cam_intr4x4
                    # #height, width
                    # #cam_img
                    # #pts3d_engine = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

                elif renderer == OD3D_RENDERER.PYTORCH3D:
                    dtype = cams_tform4x4_world[i].dtype

                    from o3b.cv.geometry.transform import (
                        depth2pts3d_grid,
                        inv_tform4x4,
                    )
                    from pytorch3d.structures import Pointclouds as PT3D_Pointclouds

                    depth = (
                        cams_imgs_depth_scale
                        * 2.5
                        * torch.ones(cam_img.shape[1:]).to(device=device, dtype=dtype)
                    )
                    pts3d_i = (
                        depth2pts3d_grid(
                            depth,
                            cam_intr4x4_res.to(device=device, dtype=dtype),
                        )
                        .flatten(1)
                        .permute(1, 0)
                    )
                    pts3d_i = transf3d_broadcast(
                        pts3d_i,
                        inv_tform4x4(
                            cams_tform4x4_world[i].to(device=device, dtype=dtype),
                        ),
                    )
                    pts3d_i_rgb = (
                        cam_img.to(device=device, dtype=dtype).flatten(1).permute(1, 0)
                        / 255.0
                    )
                    # cam_img
                    # pts3d_i =  torch.from_numpy(np.asarray(pts3d_engine.points)).to(device=device, dtype=dtype)
                    # pts3d_i_rgb = torch.from_numpy(np.asarray(pts3d_engine.colors)).to(device=device, dtype=dtype)
                    pts3d_engine = PT3D_Pointclouds(
                        points=[pts3d_i.to(device=device)],
                        features=[pts3d_i_rgb],
                    )
                    if cams_show_wireframe:
                        cam_wireframe_engine = (
                            pt3d_camera_from_tform4x4_intr4x4_imgs_size(
                                cam_tform4x4_obj=cams_tform4x4_world[i],
                                cam_intr4x4=cam_intr4x4,
                                img_size=torch.tensor([height, width]),
                            )
                        )

                else:
                    raise NotImplementedError

                geometries.append(
                    {
                        "name": f"{cam_name}_img",
                        "geometry": pts3d_engine,
                        "obj_tform4x4_cam": obj_tform4x4_cam,
                    },
                )

            else:
                if renderer == OD3D_RENDERER.OPEN3D:
                    if cams_show_wireframe:
                        cam_wireframe_engine = (
                            open3d.geometry.LineSet.create_camera_visualization(
                                view_width_px=width,
                                view_height_px=height,
                                intrinsic=cam_intr4x4[:3, :3].detach().cpu().numpy(),
                                extrinsic=cams_tform4x4_world[i].detach().cpu().numpy(),
                                scale=cams_imgs_depth_scale
                                * cams_tform4x4_world[i][2, 3],
                            )
                        )
                        cam_wireframe_engine_colors = (
                            get_colors(len(cams_tform4x4_world))[i][None,]
                            .repeat(8, 1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        cam_wireframe_engine_colors[3, 0] = 0.0
                        cam_wireframe_engine_colors[3, 1] = 1.0
                        cam_wireframe_engine_colors[3, 2] = 0.0
                        cam_wireframe_engine.colors = o3d.utility.Vector3dVector(
                            cam_wireframe_engine_colors,
                        )  # shape: (num_lines, 3)
                elif renderer == OD3D_RENDERER.PYTORCH3D:
                    if cams_show_wireframe:
                        cam_wireframe_engine = (
                            pt3d_camera_from_tform4x4_intr4x4_imgs_size(
                                cam_tform4x4_obj=cams_tform4x4_world[i],
                                cam_intr4x4=cam_intr4x4,
                                img_size=torch.tensor([height, width]),
                            )
                        )

            if (cams_show_wireframe or cams_show_image_encoder) and (
                cam_wireframe_engine is not None
            ):
                geometries.append(
                    {"name": cam_name, "geometry": cam_wireframe_engine},
                )

    return geometries


def show_pcl_via_open3d(pts3d, pts3d_colors=None):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    # vis.add_geometry(pcd)

    pcd = o3d.geometry.PointCloud()
    # from o3b.cv.geometry.transform import inv_tform4x4
    pcd.points = o3d.utility.Vector3dVector(pts3d.numpy())
    if pts3d_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pts3d_colors)
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()


def show_open3d_pcl(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()


def show_pcl(
    verts,
    cam_tform4x4_obj: torch.Tensor = None,
    cam_intr4x4: torch.Tensor = None,
    img_size: torch.Tensor = None,
):
    """

    Args:
        verts: Nx3 / BxNx3 / list(torch.Tensor Nx3)

    """

    if cam_tform4x4_obj is not None:
        pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(
            cam_tform4x4_obj=cam_tform4x4_obj,
            cam_intr4x4=cam_intr4x4,
            img_size=img_size,
        )
    else:
        pt3d_cameras = []

    if isinstance(verts, list) or verts.dim() == 3:
        if isinstance(verts, list):
            B = len(verts)
            N, _ = verts[0].shape
            device = verts[0].device
        else:
            B, N, _ = verts.shape
            device = verts.device
        colors = get_colors(B, device=device)
        rgb = colors[:, None].repeat(1, N, 1)
    else:
        N, _ = verts.shape
        colors = get_colors(1, device=verts.device)
        rgb = colors.repeat(N, 1)
        rgb = rgb[None,]
        verts = verts[None,]
        # cls = cls[None,]

    """
    # o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    rgb = rgb.reshape(-1, 3)
    verts = verts.reshape(-1, 3)
    cls = cls.reshape(-1, 3)
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.write_property('class', cls)
    geometries = [pcd]
    #o3d.visualization.draw_geometries([pcd],
    #                                  zoom=0.3412,
    #                                  front=[0.4257, -0.2125, -0.8795],
    #                                  lookat=[2.6172, 2.0475, 1.532],
    #                                  up=[-0.0694, -0.9768, 0.2024])
    viewer = o3d.visualization.Visualizer()
    o3d.visualization.gui.Label3D(color=[1., 0., 0.], position=[0., 0., 0.], scale=1., text='blub')
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.8, 0.8, 0.9])
    viewer.run()
    viewer.destroy_window()
    """
    from pytorch3d.structures import Pointclouds as PT3D_Pointclouds
    from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs

    point_cloud = PT3D_Pointclouds(points=verts, features=rgb)
    fig = plot_scene(
        {
            "Pointcloud": {
                **{f"pcl{i+1}": point_cloud[i] for i in range(len(point_cloud))},
                **{f"cam{i+1}": pt3d_cameras[i] for i in range(len(pt3d_cameras))},
            },
        },
        viewpoint_cameras=pt3d_cameras,
        axis_args=AxisArgs(
            backgroundcolor="rgb(200, 200, 230)",
            showgrid=True,
            zeroline=True,
            showline=True,
            showaxeslabels=True,
            showticklabels=True,
        ),
    )
    fig.show()
    input("bla")


def imgs_to_img(rgbs, pad=1, pad_value=0, H_out=None, W_out=None):
    # rgb: K x 3 x H x W / GH x GW x 3 x H x W

    if isinstance(rgbs, List) and isinstance(rgbs[0], List):
        H_in, W_in = rgbs[0][0].shape[-2:]
        rgbs = torch.stack(
            [
                torch.stack(
                    [resize(rgb, H_out=H_in, W_out=W_in) for rgb in rgbs_i],
                    dim=0,
                )
                for rgbs_i in rgbs
            ],
            dim=0,
        )
    elif isinstance(rgbs, List):
        H_in, W_in = rgbs[0].shape[-2:]
        rgbs = torch.stack([resize(rgb, H_out=H_in, W_out=W_in) for rgb in rgbs], dim=0)

    rgbs = torch.nn.functional.pad(rgbs, (pad, pad, pad, pad), "constant", pad_value)
    # margin = 2
    # torch.nn.functional.pad(rgbs, (1, 1), "constant", 0)

    rgbs_shape = rgbs.shape

    if len(rgbs_shape) == 5:
        GH, GW, C, H, W = rgbs_shape
        rgb = rgbs.clone()
    elif len(rgbs_shape) == 4:
        K, C, H, W = rgbs.shape
        prop_w = 4
        prop_h = 3
        GW = math.ceil(math.sqrt((K * prop_w**2) / prop_h**2))
        GH = math.ceil(K / GW)
        GTOTAL = GH * GW

        img_placeholder = torch.zeros_like(rgbs[:1]).repeat(GTOTAL - K, 1, 1, 1)

        rgb = torch.cat((rgbs, img_placeholder), dim=0)

        rgb = rgb.reshape(GH, GW, C, H, W)
    elif len(rgbs_shape) == 3:
        GH = 1
        GW = 1
        C, H, W = rgbs_shape
        rgb = rgbs.clone()[None, None]
    else:
        logger.error(
            "Visualize imgs requires the input rgb tensor to have 4 (KxCxHxW) or 5 (GHxGWxCxHxW) dimensions",
        )
        raise NotImplementedError

    rgb = rgb.permute(2, 0, 3, 1, 4)

    rgb = rgb.reshape(C, GH * H, GW * W)

    if H_out is not None and W_out is not None:
        rgb = resize(rgb, H_out=H_out, W_out=W_out)
    return rgb


def fpaths_to_rgb(fpaths: List[Path], H: int, W: int, pad=1):
    rgbs = torch.stack(
        [
            resize(torchvision.io.read_image(path=str(fpath)), H_out=H, W_out=W)
            for fpath in fpaths
        ],
        dim=0,
    )
    rgb = imgs_to_img(rgbs, pad=pad)
    return rgb


def img_to_mesh(
    img,
    cam_tform4x4_obj=None,
    cam_intr4x4=None,
    cams_imgs_resize=True,
    depth=1.0,
):
    import pyrender

    width = int(cam_intr4x4[0, 2] * 2)
    height = int(cam_intr4x4[1, 2] * 2)
    h = img.shape[-2]
    w = img.shape[-1]
    #
    # if width is None:
    #     width = img.shape[-1]
    # elif isinstance(width, torch.Tensor):
    #     width = width.item()
    # if height is None:
    #     height = img.shape[-2]
    # elif isinstance(height, torch.Tensor):
    #     height = height.item()
    # h, w = height, width

    if cams_imgs_resize:
        img = resize(img, H_out=256, W_out=256)
        h_resize = 256.0 / h
        w_resize = 256.0 / w
    else:
        img = img
        h_resize = 1.0
        w_resize = 1.0

    height = int(height * h_resize)
    width = int(width * w_resize)

    if img.dtype == torch.float:
        img = (img.clone() * 255).to(dtype=torch.uint8)

    # if img.dtype == torch.uint8:
    #     img = (img.clone() / 255.).to(dtype=torch.float)

    from torchvision.transforms.functional import to_pil_image

    # .permute(1, 2, 0) * 255).cpu().contiguous()
    # texture = to_pil_image(img.contiguous().clone().detach().permute(1, 2, 0) * 255).cpu().contiguous()
    texture = to_pil_image(img.clone().flip(dims=(1,)).clone().contiguous())

    if cam_intr4x4 is not None:
        cam_intr4x4_res = cam_intr4x4.clone()
        cam_intr4x4_res[0, 0] = cam_intr4x4_res[0, 0].item() * w_resize
        cam_intr4x4_res[1, 1] = cam_intr4x4_res[1, 1].item() * h_resize
        cam_intr4x4_res[0, 2] = cam_intr4x4_res[0, 2].item() * w_resize
        cam_intr4x4_res[1, 2] = cam_intr4x4_res[1, 2].item() * h_resize

        # Create pyrender camera from intrinsic matrix
        fx = cam_intr4x4_res[0, 0].item()
        fy = cam_intr4x4_res[1, 1].item()
        cx = cam_intr4x4_res[0, 2].item()
        cy = cam_intr4x4_res[1, 2].item()
        z_far = depth

        # fx, fy = K[0, 0], K[1, 1]
        # cx, cy = K[0, 2], K[1, 2]

        # Image corners in pixel coordinates
        # w_resize, width
        # h_resize, height
        img_corners = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
        )

        # Backproject to 3D at depth z_far
        corners_3d = []
        for u, v in img_corners:
            x = (u - cx) * z_far / fx
            y = (v - cy) * z_far / fy
            corners_3d.append([x, y, z_far])
        corners_3d = np.array(corners_3d)
        # Define 4 vertices (rectangle in X-Y plane, Z=0)
        vertices = np.array(
            [
                corners_3d[0],  # Bottom-left
                corners_3d[1],  # Bottom-right
                corners_3d[2],  # Top-right
                corners_3d[3],  # Top-left
                # [-1, -1, 0],  # Bottom-left
                # [1, -1, 0],  # Bottom-right
                # [1, 1, 0],  # Top-right
                # [-1, 1, 0],  # Top-left
            ],
        )
    else:
        height_half = height // 2
        width_half = width // 2
        vertices = np.array(
            [
                [-width_half, -height_half, 0],  # Bottom-left
                [width_half, -height_half, 0],  # Bottom-right
                [width_half, height_half, 0],  # Top-right
                [-width_half, height_half, 0],  # Top-left
            ],
        )

    # Define two triangular faces (using the 4 vertices)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],  # Back face 1 (reversed)
            [3, 2, 0],  # Back face 2 (reversed)
        ],
    )

    # Define UV coordinates (match image corners)
    uv = np.array(
        [
            [0.0, 0.0],  # Bottom-left
            [1.0, 0.0],  # Bottom-right
            [1.0, 1.0],  # Top-right
            [0.0, 1.0],  # Top-left
        ],
    )

    import trimesh

    # Create the texture visual
    # material = trimesh.visual.texture.SimpleMaterial(image=texture)
    # visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture, material=material)

    # Create the mesh
    # vertices = np.concatenate([vertices, frustum_mesh.vertices])
    # faces = np.concatenate([faces, frustum_mesh.faces + len(faces)])
    # uv = np.concatenate([uv, frustum_mesh.vertices.copy()[:, :2] * 0.])

    # material = trimesh.visual.texture.SimpleMaterial(image=texture)
    visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=texture,
    )  # , material=material)

    frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)
    # frustum_mesh.show()

    if cam_tform4x4_obj is not None:
        # # FOLLOWING OPENGL convention
        pyrender_cam_tform4x4_obj = OPEN3D_CAM_TFORM_OBJ.clone()
        # pyrender_cam_tform4x4_obj = tform4x4(OPEN3D_DEFAULT_CAM_TFORM_OBJ.clone().to(device=cam_tform4x4_obj.device), cam_tform4x4_obj, )
        from o3b.cv.geometry.transform import inv_tform4x4

        obj_tform4x4_pyrender_cam = inv_tform4x4(pyrender_cam_tform4x4_obj)
        obj_tform4x4_pyrender_cam_np = obj_tform4x4_pyrender_cam.detach().cpu().numpy()
        obj_tform4x4_cam = obj_tform4x4_pyrender_cam_np

        frustum_mesh = frustum_mesh
        # inv_tform4x4(cams_tform4x4_world[i]).to(device=device, dtype=dtype)
        dtype = cam_tform4x4_obj.dtype
        frustum_mesh.vertices = transf3d_broadcast(
            torch.from_numpy(frustum_mesh.vertices).to(dtype=dtype),
            transf4x4=inv_tform4x4(cam_tform4x4_obj).to(device="cpu"),
        )
    # from o3b.cv.visual.objectsmesh import Meshes
    return Meshes.from_trimesh(mesh_trimesh=frustum_mesh)

def show_imgs(
    rgbs,
    duration=0,
    vwriter=None,
    fpath=None,
    height=None,
    width=None,
    pad=1,
    pad_value=0.0,
):
    rgb = imgs_to_img(rgbs, pad=pad, pad_value=pad_value)
    return show_img(rgb, duration, vwriter, fpath, height, width)


def select_img_grid_cells(
    rgbs,
    pad=1,
    pad_value=0.0,
    downscale_factor=4,
    window_name="img-grid",
):
    if rgbs.dim() != 5:
        msg = "Interactive grid selection expects a tensor shaped as Rows x Columns x 3 x H x W."
        raise ValueError(msg)
    if "DISPLAY" not in os.environ:
        msg = "Interactive grid selection requires a graphical display."
        raise RuntimeError(msg)

    rows_count, cols_count, _, cell_height, cell_width = rgbs.shape
    padded_cell_height = cell_height + 2 * pad
    padded_cell_width = cell_width + 2 * pad

    rgb = imgs_to_img(rgbs, pad=pad, pad_value=pad_value)
    if downscale_factor is None or downscale_factor <= 1:
        rgb_preview = rgb
    else:
        rgb_preview = resize(rgb[None], scale_factor=1.0 / downscale_factor)[0]

    preview_img = tensor_to_cv_img(rgb_preview)
    scale_x = preview_img.shape[1] / rgb.shape[-1]
    scale_y = preview_img.shape[0] / rgb.shape[-2]

    state = {
        "cells": set(),
        "hover_cell": None,
    }

    def sync_rows_and_cols():
        rows = sorted({cell[0] for cell in state["cells"]})
        cols = sorted({cell[1] for cell in state["cells"]})
        return rows, cols

    def get_cell_from_xy(x, y):
        row = int((y / scale_y) // padded_cell_height)
        col = int((x / scale_x) // padded_cell_width)
        if row < 0 or row >= rows_count or col < 0 or col >= cols_count:
            return None
        return row, col

    def get_display_box(row, col):
        x0 = int(round(col * padded_cell_width * scale_x))
        y0 = int(round(row * padded_cell_height * scale_y))
        x1 = int(round((col + 1) * padded_cell_width * scale_x)) - 1
        y1 = int(round((row + 1) * padded_cell_height * scale_y)) - 1
        return x0, y0, x1, y1

    def draw_overlay():
        frame = preview_img.copy()

        for row, col in state["cells"]:
            x0, y0, x1, y1 = get_display_box(row, col)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 220, 40), 2)
            cv2.putText(
                frame,
                f"{row},{col}",
                (x0 + 4, min(y1 - 4, y0 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (40, 220, 40),
                1,
                cv2.LINE_AA,
            )

        if state["hover_cell"] is not None:
            x0, y0, x1, y1 = get_display_box(*state["hover_cell"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 1)

        rows, cols = sync_rows_and_cols()
        hover_text = "-" if state["hover_cell"] is None else f"{state['hover_cell'][0]},{state['hover_cell'][1]}"
        status_lines = [
            f"hover={hover_text}",
            f"rows={rows}",
            f"cols={cols}",
            "left click: toggle cell | c: clear | enter/q/esc: accept",
        ]
        for line_id, line in enumerate(status_lines):
            y = 24 + line_id * 22
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return frame

    def on_mouse(event, x, y, flags, param):
        state["hover_cell"] = get_cell_from_xy(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and state["hover_cell"] is not None:
            hover_cell = state["hover_cell"]
            if hover_cell in state["cells"]:
                state["cells"].remove(hover_cell)
            else:
                state["cells"].add(hover_cell)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            cv2.imshow(window_name, draw_overlay())
            key = cv2.waitKey(20) & 0xFF
            if key in (13, ord("q"), 27):
                break
            if key == ord("c"):
                state["cells"].clear()
    finally:
        cv2.destroyWindow(window_name)

    rows, cols = sync_rows_and_cols()
    return {
        "rows": rows,
        "cols": cols,
        "cells": sorted(state["cells"]),
    }


def depth1d_to_rgb3d(depth):
    # in: ...x 1 x H x W
    # out: ... x 3 x H x W
    #depth [tensor_to_cv_img(depth)
    from o3b.cv.visual.blend import rgb_to_range01, norm_to_range01
    import cv2

    depth_mask_valid = (depth > 0.)

    depth = depth.clone()

    if depth_mask_valid.sum() > 0:
        depth[~depth_mask_valid] = depth[depth_mask_valid].max()

    depth_01 = norm_to_range01(1. / (depth + 1e-8))
    # depth_01 = norm_to_range01(-depth)

    depth_shape = depth_01.shape

    depth_01_reshape = depth_01.reshape(-1, 1, depth_01.shape[-2], depth_01.shape[-1])
    depth_mask_valid_reshape = depth_mask_valid.reshape(-1, 1, depth_01.shape[-2], depth_01.shape[-1])

    depth_01_flatten_rgb = [
        cv_img_to_tensor(cv2.applyColorMap(tensor_to_cv_img(depth_01_reshape[d]), cv2.COLORMAP_MAGMA,)).to(depth.device) #  cv2.COLORMAP_JET,))
        for d in range(depth_01_reshape.shape[0])
    ]

    depth_01_flatten_rgb = [ (1. * depth_mask_valid_reshape[d] * _depth_rgb + (1. - 1. * depth_mask_valid_reshape[d]) * 255.) for d, _depth_rgb in enumerate(depth_01_flatten_rgb)]

    depth_01_rgb = torch.stack(depth_01_flatten_rgb, dim=0).reshape(*depth_shape[:-3], 3, *depth_shape[-2:])

    depth_01_rgb = depth_01_rgb.to(depth.device)
    return depth_01_rgb

def show_img(
    rgb,
    duration=0,
    vwriter=None,
    fpath=None,
    height=None,
    width=None,
    normalize=False,
    live=True,
):
    if rgb.dim() == 3 and rgb.shape[0] != 3 and rgb.shape[0] != 1:
        colors = get_colors(K=rgb.shape[0], device=rgb.device, last_white=True)
        rgb = (rgb.clone()[:, None] * colors[:, :, None, None]).sum(dim=0)
    # img: 3xHxW
    rgb = rgb.clone()

    if normalize:
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    if width is not None and height is not None:
        orig_width = rgb.size(2)
        orig_height = rgb.size(1)
        scale_factor = min(width / orig_width, height / orig_height)
    elif width is not None:
        orig_width = rgb.size(2)
        scale_factor = width / orig_width
    elif height is not None:
        orig_height = rgb.size(1)
        scale_factor = height / orig_height

    if width or height is not None:
        rgb = resize(
            rgb[None,],
            scale_factor=scale_factor,
        )[0]

    img = tensor_to_cv_img(rgb)

    if vwriter is not None:
        vwriter.write(img)

    # DO: check if display is available, otherwise save the image at live.png
    if fpath is None and ("DISPLAY" not in os.environ or not live):
        fpath = "live.png"
    
    if fpath is not None:
        Path(fpath).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(fpath), img)
    else:
        logging.basicConfig(level=logging.DEBUG)
        cv2.imshow("img", img)
        return cv2.waitKey(duration)


def get_img_from_plot(ax, fig, axis_off=True, margins=1, pad=1):
    try:
        count_axes = len(ax)
        single_axes = False
    except TypeError:
        count_axes = 1
        single_axes = True
    # Image from plot
    if axis_off:
        if single_axes:
            ax.axis("off")
            # To remove the huge white borders
            ax.margins(margins)
        else:
            for ax_single in ax:
                ax_single.axis("off")
                # To remove the huge white borders
                ax_single.margins(margins)
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout(pad=pad)
        if single_axes:
            ax.margins(margins)
        else:
            for ax_single in ax:
                ax_single.margins(margins)

    fig.canvas.draw()
    try:
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot: np.ndarray = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,),
        )
    except:
        image_from_plot = np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    return torch.from_numpy(image_from_plot.copy()).permute(2, 0, 1)
