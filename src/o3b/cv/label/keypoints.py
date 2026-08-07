import logging
logger = logging.getLogger(__name__)

from o3b.cv.geometry.transform import transf3d_broadcast, inv_tform4x4, tform4x4
import torch
from typing import Union, List
from o3b.cv.visual.show import get_engine_geometries_for_cams
from o3b.cv.geometry.downsample import random_sampling, random_sampling_with_fill
from pathlib import Path
import numpy as np

def label_kpts4d(fpaths_meshs: List[Path], fpaths_kpts4d: List[Path],):

    import open3d
    from o3b.cv.geometry.objects3d.meshes.meshes import Meshes
    PARALLEL_MESH_COUNT_MAX = 5
    KPOINTS3D_COUNT_MAX = 10
    PTS3D_COUNT = 30000
    VIS_POINT_SIZE = 5.

    objscentric_tform4x4_objs = []
    scene_tform4x4_objscentric = []
    pts3ds = []
    pts3ds_colors = []
    kpts3ds = []
    kpts3ds_colors = []
    kpts3ds_mask = []

    pts3d_buf = -1.1 * torch.ones((KPOINTS3D_COUNT_MAX, 3))
    pts3d_buf[:, 0] = torch.linspace(-1, 1, KPOINTS3D_COUNT_MAX)
    kpts3d_colors = torch.Tensor([[1.0, 1.0, 0.0],
                                  [0.529, 0.808, 0.980],
                                  [1.0, 0.447, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.545]])

    kpts3d_colors = kpts3d_colors.repeat(KPOINTS3D_COUNT_MAX // 5 + 1, 1)
    kpts3d_colors = kpts3d_colors[:KPOINTS3D_COUNT_MAX]

    for i, fpath_mesh in enumerate(fpaths_meshs):
        mesh = Meshes.read_from_ply_file(fpath=fpath_mesh)
        objcentric_tform4x4_obj = mesh.get_objscentric_tform4x4_objs()
        objscentric_tform4x4_objs.append(objcentric_tform4x4_obj)

        scene_tform4x4_objcentric = torch.eye(4)
        scene_tform4x4_objcentric[0, 3] = 3. * i
        scene_tform4x4_objscentric.append(scene_tform4x4_objcentric)

        mesh.transf3d(objs_new_tform4x4_objs=objcentric_tform4x4_obj)
        # mesh_open3d = mesh.to_o3d()
        # mesh_open3d.compute_vertex_normals()
        pts3d = mesh.verts.clone()
        pts3d, pts3d_ids = random_sampling_with_fill(pts3d, pts3d_count=PTS3D_COUNT, return_ids=True)

        if mesh.rgbs_uvs is not None:
            H, W = mesh.rgbs_uvs.shape[-2:]
            pts3d_colors = mesh.rgbs_uvs[:, :, (mesh.verts_uvs[:, 1] * (H-1)).long(), (mesh.verts_uvs[:, 0] * (W-1)).long()]
            pts3d_colors = pts3d_colors.permute(2, 1, 0)[:, :, 0]
            pts3d_colors = pts3d_colors[pts3d_ids]
            # mesh.get_vert_mod_from_objs("rgb")
        else:
            pts3d_colors = None

        pts3d = torch.cat([pts3d, pts3d_buf], dim=0)
        pts3d_colors = torch.cat([pts3d_colors, kpts3d_colors], dim=0)

        pts3ds.append(pts3d)
        pts3ds_colors.append(pts3d_colors)

        fpath_kpts4d = fpaths_kpts4d[i]
        kpts3d = pts3d_buf.clone() #  torch.zeros((KPOINTS3D_COUNT_MAX, 3))
        kpts3d_mask = torch.zeros((KPOINTS3D_COUNT_MAX,)).bool()
        if fpath_kpts4d.exists():
            _kpts3d = torch.load(fpath_kpts4d)[:, :3]
            _kpts3d = transf3d_broadcast(pts3d=_kpts3d, transf4x4=objcentric_tform4x4_obj)
            _kpts3d_mask = torch.load(fpath_kpts4d)[:, 3] > 0.5
            if _kpts3d_mask.sum() == 0:
                try:
                    fpath_kpts4d.unlink()
                    fpath_kpts4d.parent.rmdir()
                except Exception as e:
                    logger.info("could not remove fpath kpts4d...")
                    logger.info(e)
            kpts3d[:_kpts3d.shape[0]][_kpts3d_mask] = _kpts3d[_kpts3d_mask]
            kpts3d_mask[:_kpts3d_mask.shape[0]] = _kpts3d_mask

        kpts3ds.append(kpts3d)
        kpts3ds_mask.append(kpts3d_mask)
        kpts3ds_colors.append(kpts3d_colors[:kpts3ds[i].shape[0]] * (kpts3d_mask[:, None,] * 1. + 0.7).clamp(0., 1.))


    logger.info("")
    logger.info(
        "1) Please pick left, right, back, front, top, bottom [shift + left click]",
    )
    logger.info("   Press [shift + right click] to undo point picking")
    logger.info("2) Afther picking points, press q for close the window")

    for obj_label_id in range(len(fpaths_meshs)):
        scene_pts3d = []
        scene_pts3d_colors = []

        obj_id_min = max(obj_label_id - PARALLEL_MESH_COUNT_MAX + 1, 0)
        obj_id_max = min(obj_label_id + (PARALLEL_MESH_COUNT_MAX - (obj_label_id - obj_id_min + 1)), len(fpaths_meshs) -1 )
        logger.info(f"{obj_id_min}, {obj_id_max}")

        objs_ids = torch.arange(obj_id_min, obj_id_max + 1)
        shift_ids_count = obj_label_id - objs_ids.median()
        shift_ids_count_with_overtake = shift_ids_count + 1 * shift_ids_count.sign()
        shift_scene_tform4x4_scene = torch.eye(4).repeat(objs_ids.max() + 1, 1, 1)

        objs_ids_w_shift = objs_ids[objs_ids != obj_label_id]
        objs_ids_w_shift_with_overtake = objs_ids[(objs_ids != obj_label_id) *
                                                  ((objs_ids - obj_label_id).sign() !=
                                                   (objs_ids + shift_ids_count - obj_label_id).sign())]

        shift_scene_tform4x4_scene[objs_ids_w_shift, 0, 3] = 3. * shift_ids_count
        shift_scene_tform4x4_scene[objs_ids_w_shift_with_overtake, 0, 3] = 3. * shift_ids_count_with_overtake

        for i in range(obj_id_min, obj_id_max + 1):

            # torch.arange(obj_id_min, obj_id_max + 1)
            _scene_tform4x4_objscentric = tform4x4(shift_scene_tform4x4_scene[i], scene_tform4x4_objscentric[i])

            pts3d = transf3d_broadcast(pts3d=pts3ds[i].clone(), transf4x4=_scene_tform4x4_objscentric)
            pts3d_colors = pts3ds_colors[i].clone()
            # if i != obj_label_id: #i < (min(PARALLEL_MESH_COUNT_MAX, len(fpaths_meshs)) - 1):
            kpts3d = (kpts3ds[i][None,].clone() + 0.01 * torch.randn((20, 1, 3))).reshape(-1, 3)
            kpts3d = transf3d_broadcast(pts3d=kpts3d, transf4x4=_scene_tform4x4_objscentric)
            _kpts3d_colors = kpts3ds_colors[i].clone().repeat(20, 1)  #(dim=0, repeats=10)

            pts3d = torch.cat([pts3d[:-KPOINTS3D_COUNT_MAX], kpts3d], dim=0)
            pts3d_colors = torch.cat([pts3d_colors[:-KPOINTS3D_COUNT_MAX], _kpts3d_colors], dim=0)
            if i == obj_label_id:
                # visualize which one to lable next
                kpts3d = (kpts3ds[i][None,].clone() * 0. + 0.05 * torch.randn((20, kpts3ds[i].shape[0], 3))).reshape(-1, 3)
                kpts3d[:, 1] += 1.5
                kpts3d = transf3d_broadcast(pts3d=kpts3d, transf4x4=scene_tform4x4_objscentric[i])
                _kpts3d_colors = kpts3ds_colors[i].clone().repeat(20, 1)  #(dim=0, repeats=10)

                pts3d = torch.cat([pts3d, kpts3d], dim=0)
                pts3d_colors = torch.cat([pts3d_colors, _kpts3d_colors], dim=0)

                #pts3d = pts3d.flip(dims=(0,))
                #pts3d_colors = pts3d_colors.flip(dims=(0,))
            scene_pts3d.append(pts3d)
            scene_pts3d_colors.append(pts3d_colors)



        scene_pts3d = torch.cat(scene_pts3d, dim=0)

        scene_tform4x4_scene_shift = scene_tform4x4_objscentric[obj_label_id]
        scene_shift_tform4x4_scene = inv_tform4x4(scene_tform4x4_scene_shift)
        scene_pts3d = transf3d_broadcast(pts3d=scene_pts3d, transf4x4=scene_shift_tform4x4_scene)

        scene_pts3d_colors = torch.cat(scene_pts3d_colors, dim=0)
        # Set the point cloud data

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(scene_pts3d.detach().cpu().numpy())

        if scene_pts3d_colors is not None:
            pcd.colors = open3d.utility.Vector3dVector(scene_pts3d_colors.detach().cpu().numpy())

        vis = open3d.visualization.VisualizerWithEditing()
        vis.create_window()
        opt = vis.get_render_option()
        opt.point_size = VIS_POINT_SIZE
        opt.mesh_show_back_face = True
        #opt.show_coordinate_frame = show_coordinate_frame
        #opt.background_color = np.asarray(background_color)

        vis.add_geometry(pcd)

        vis.run()  # user picks points
        vis.destroy_window()
        scene_kpts3d_picked_ids = vis.get_picked_points()

        if len(scene_kpts3d_picked_ids) > 0:
            scene_kpts3d_picked = scene_pts3d[scene_kpts3d_picked_ids].clone()
            scene_kpts3d_picked = transf3d_broadcast(pts3d=scene_kpts3d_picked, transf4x4=scene_tform4x4_scene_shift)

            scene_kpts3d_picked_count = len(scene_kpts3d_picked_ids)
            objs_centric_kpts3d_picked = transf3d_broadcast(pts3d=scene_kpts3d_picked,
                                                            transf4x4=inv_tform4x4(scene_tform4x4_objscentric[obj_label_id].clone()))

            kpts3d_picked_mask = objs_centric_kpts3d_picked.min(dim=-1).values > -1.01

            kpts3d_picked = transf3d_broadcast(pts3d=scene_kpts3d_picked,
                                               transf4x4=tform4x4(inv_tform4x4(objscentric_tform4x4_objs[obj_label_id].clone()),
                                                                  inv_tform4x4(scene_tform4x4_objscentric[obj_label_id].clone())))

            kpts3d_orig = pts3d_buf.clone()
            kpts3d_orig = transf3d_broadcast(pts3d=kpts3d_orig,
                               transf4x4=inv_tform4x4(objscentric_tform4x4_objs[obj_label_id].clone()))
            kpts3d_orig[:scene_kpts3d_picked_count][kpts3d_picked_mask] = kpts3d_picked[kpts3d_picked_mask]

            kpts3d = pts3d_buf.clone()  # torch.zeros((KPOINTS3D_COUNT_MAX, 3))
            kpts3d_mask = torch.zeros((KPOINTS3D_COUNT_MAX,)).bool()
            kpts3d[: scene_kpts3d_picked_count] = objs_centric_kpts3d_picked #  kpts3d_picked
            kpts3d_mask[: scene_kpts3d_picked_count] = kpts3d_picked_mask
            _kpts3d_colors = kpts3d_colors[:kpts3ds[obj_label_id].shape[0]]

            kpts3ds[obj_label_id] = kpts3d
            kpts3ds_mask[obj_label_id] = kpts3d_mask
            kpts3ds_colors[obj_label_id] = _kpts3d_colors * (kpts3d_mask[:, None,] * 1. + 0.7).clamp(0., 1.)

            kpts4d_orig = torch.cat([kpts3d_orig, kpts3d_mask[:, None]], dim=-1)
            from o3b.cv.io import write_tensor
            write_tensor(obj=kpts4d_orig, fpath=fpaths_kpts4d[obj_label_id])

    # dist_kpts3d_ij = torch.ones((len(fpaths_meshs), len(fpaths_meshs)))
    # for i in range(len(fpaths_meshs)):
    #     for j in range(len(fpaths_meshs)):
    #         dist_kpts3d_ij[i, j] = ((kpts3ds[i] - kpts3ds[j]) * kpts3ds_mask[i][:, None,] * kpts3ds_mask[j][:, None,]).norm(dim=-1).mean()
    #
    # import matplotlib.pyplot as plt
    # # Plot heatmap
    # plt.imshow(dist_kpts3d_ij, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label="Distance")
    # plt.title("2D Distance Matrix")
    # plt.xlabel("Mesh j")
    # plt.ylabel("Mesh i")
    # plt.show()
