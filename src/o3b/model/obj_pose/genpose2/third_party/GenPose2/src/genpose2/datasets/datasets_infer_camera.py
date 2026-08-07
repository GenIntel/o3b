import numpy as np
import cv2
import torch
import copy
import json

from cutoop.data_loader import Dataset, ImageMetaData
from genpose2.utils.datasets_utils import aug_bbox_eval, get_2d_coord_np, crop_resize_by_warp_affine, get_affine_transform
from genpose2.utils.sgpa_utils import get_bbox
from genpose2.datasets.datasets_omni6dpose import Omni6DPoseDataSet
from cutoop.transform import pixel2xyz
from cutoop.image_meta import ViewInfo
from cutoop.data_types import CameraIntrinsicsBase
import torch.nn.functional as F

def transform_torch(M_cv, H, W, device) -> torch.Tensor:
    T = np.array([
        [2.0 / W, 0, -1],
        [0, 2.0 / H, -1],
        [0, 0, 1]
    ])
    
    ret = np.linalg.inv(T @ M_cv @ np.linalg.inv(T))

    return ret

def crop_resize_by_warp_affine_torch(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)
    trans_t = np.zeros((3, 3), dtype=np.float32)
    trans_t[:2, :3] = trans
    trans_t[2, 2] = 1
    trans_t = transform_torch(trans_t, img.shape[0], img.shape[1], img.device)
    trans_t = trans_t[:2,:]

    trans_t = torch.tensor(trans_t, dtype=torch.float32, device=img.device).unsqueeze(0)
    img = img.float()
    if len(img.shape) == 2:
        img = img.unsqueeze(0)

    grid = F.affine_grid(trans_t, size=(1, img.shape[0], img.shape[0], img.shape[1]), align_corners=False)
    # print(grid)
    img = img.unsqueeze(0)
    
    if interpolation == cv2.INTER_LINEAR:
        mode = 'bilinear'
    elif interpolation == cv2.INTER_NEAREST:
        mode = 'nearest'
    else:
        raise ValueError("Unsupported interpolation mode. Use cv2.INTER_LINEAR or cv2.INTER_NEAREST.")
    # mode = 'bilinear'
    # print(mode)
    transformed_img = F.grid_sample(img, grid, mode=mode, align_corners=False)
    # print(transformed_img.shape)
    # print(torch.sum(transformed_img))
    # print(torch.sum(img))

    transformed_img = transformed_img.squeeze(0)
    if len(transformed_img.shape) == 3 and transformed_img.shape[0] == 1:
        transformed_img = transformed_img.squeeze(0)  # (H, W)
    
    return transformed_img[:output_size[0], :output_size[1]]

def crop_resize_by_warp_affine_torch2(img, center, scale, output_size):
    """
    Manually implement crop and resize using nearest neighbor interpolation with PyTorch.
    :param img: Input image (C, H, W)
    :param center: Center point (cx, cy)
    :param scale: Scale (w, h)
    :param output_size: Output size (w, h)
    :return: Transformed image (C, H', W')
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    H, W = img.shape
    w, h = output_size
    cx, cy = center
    sx = w / scale[0]
    sy = h / scale[1]

    x = torch.arange(w, device=img.device)
    y = torch.arange(h, device=img.device)
    x, y = torch.meshgrid(x, y, indexing='xy')

    src_x = (x - w / 2) / sx + cx
    src_y = (y - h / 2) / sy + cy

    src_x = src_x.round().clamp(0, W - 1).long()
    src_y = src_y.round().clamp(0, H - 1).long()
    dst_img = img[src_y, src_x]

    return dst_img

def depth_to_pcl(depth, K, xymap, valid):
    K = torch.tensor(K.reshape(-1))
    cx, cy, fx, fy = K[2], K[5], K[0], K[4]
    depth = torch.tensor(depth.reshape(-1).astype(np.float32)).to(valid.device)[valid]
    xymap = torch.tensor(xymap, dtype=torch.float32).to(valid.device)
    x_map = xymap[0].reshape(-1)[valid]
    y_map = xymap[1].reshape(-1)[valid]
    real_x = (x_map - cx) * depth / fx
    real_y = (y_map - cy) * depth / fy
    pcl = torch.stack((real_x, real_y, depth), axis=-1)
    return pcl.type(torch.float32)

def sample_points(pcl, n_pts):
    """ Down sample the point cloud.
    TODO: use farthest point sampling

    Args:
        pcl (torch tensor or numpy array):  NumPoints x 3
        num (int): target point number
    """
    total_pts_num = pcl.shape[0]
    if total_pts_num < n_pts:
        pcl = torch.cat([torch.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
        ids = torch.cat([torch.tile(torch.arange(total_pts_num), (n_pts // total_pts_num, )), torch.arange(n_pts % total_pts_num)], axis=0)
    else:
        ids = torch.randperm(total_pts_num)[:n_pts]
        pcl = pcl[ids]
    return ids, pcl
    

class InferDataset(object):
    def __init__(self, data: dict, img_size: int=224, device='cuda', n_pts=1024):
        """
        Args:
            data (dict): dictionary containing depth, color, mask, and meta data
                depth (np.ndarray): depth image
                color (np.ndarray): color image
                mask (np.ndarray): mask image
                meta (dict): camera intrinsics
            img_size (int): size of the image to be used for the network
            device (str): device to be used for the network
            n_pts (int): number of points to be used for the network
        """
        self._depth: np.ndarray = data['depth']
        self._color: np.ndarray = data['color']
        # print(data)
        if 'mask' in data:
            self._mask: np.ndarray = data['mask']
        elif 'masks' in data:
            self._masks: torch.Tensor = data['masks']
            self.obj_ids = data['obj_ids']
        else:
            raise ValueError("Data must contain either 'mask' or 'masks' key")
        if isinstance(data['meta'], dict):
            camera_intrinsics = data['meta']['camera']['intrinsics']
            camera_intrinsics = CameraIntrinsicsBase(
                fx=camera_intrinsics['fx'],
                fy=camera_intrinsics['fy'],
                cx=camera_intrinsics['cx'],
                cy=camera_intrinsics['cy'],
                width=camera_intrinsics['width'],
                height=camera_intrinsics['height']
            )
            camera = ViewInfo(None, None, camera_intrinsics, None, None, None, None, None)
            self._meta: ImageMetaData = ImageMetaData(None, camera, None, None, None, None, None, None, None, None)
        else:
            self._meta: ImageMetaData = data['meta']

        self._img_size = img_size
        self._device = device
        self._n_pts = n_pts

    
    @classmethod
    def alternetive_init(cls, prefix: str, img_size: int=224, device='cuda', n_pts=1024):
        prefix = prefix
        depth = Dataset.load_depth(prefix + 'depth.exr')
        color = Dataset.load_color(prefix + 'color.png')
        mask = Dataset.load_mask(prefix + 'mask.exr')
        with open(prefix + 'meta.json', 'r') as f:
            meta = json.load(f)
        # meta = Dataset.load_meta(prefix + 'meta.json')
        return cls({'depth': depth, 'color': color, 'mask': mask, 'meta': meta}, img_size=img_size, device=device, n_pts=n_pts)


    def get_per_object(self, obj_idx, ind=0):
        max_depth = 4.0
        if hasattr(self, '_mask'):
            # print(self._mask.shape)
            object_mask = np.equal(self._mask, obj_idx)
            if not object_mask.any():
                assert False, f"Object {obj_idx} not found in mask"
            self._depth[self._depth > max_depth] = 0
            if not (self._mask.shape[:2] == self._depth.shape[:2] == self._color.shape[:2]):
                assert False, "depth, mask, and rgb should have the same shape"
            intrinsics = self._meta.camera.intrinsics
            intrinsic_matrix = np.array([
                [intrinsics.fx, 0,             intrinsics.cx], 
                [0,             intrinsics.fy, intrinsics.cy], 
                [0,             0,             0]
                ], dtype=np.float32)
            
            img_width, img_height = self._color.shape[1], self._color.shape[0]
            scale_x = img_width / intrinsics.width
            scale_y = img_height / intrinsics.height
            intrinsic_matrix[0] *= scale_x
            intrinsic_matrix[1] *= scale_y

            coord_2d = get_2d_coord_np(img_width, img_height).transpose(1, 2, 0)

            ys, xs = np.argwhere(object_mask).transpose(1, 0)
            rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
            rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], img_height, img_width)

            # here resize and crop to a fixed size 224 x 224
            bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
            bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)

            # crop and resize
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)

            roi_rgb_ = crop_resize_by_warp_affine(
                self._color, bbox_center, scale, self._img_size, interpolation=cv2.INTER_LINEAR
            )
            roi_rgb = Omni6DPoseDataSet.rgb_transform(roi_rgb_)

            mask_target = self._mask.copy().astype(np.float32)
            mask_target[self._mask != obj_idx] = 0.0
            mask_target[self._mask == obj_idx] = 1.0
            roi_mask = crop_resize_by_warp_affine(
                mask_target, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_mask = np.expand_dims(roi_mask, axis=0)

            roi_depth = crop_resize_by_warp_affine(
                self._depth, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_depth = np.expand_dims(roi_depth, axis=0)
            depth_valid = roi_depth > 0
            if np.sum(depth_valid) <= 1.0:
                # from ipdb import set_trace; set_trace()
                return None
                assert False, "No valid depth values"

            roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
            if np.sum(roi_m_d_valid) <= 1.0:
                # from ipdb import set_trace; set_trace()
                return None
                assert False, "No valid depth values"

            valid = (np.squeeze(roi_depth, axis=0) > 0) * (np.squeeze(roi_mask, axis=0) > 0)
            xs, ys = np.argwhere(valid).transpose(1, 0)
            valid = valid.reshape(-1)
            pcl_in = Omni6DPoseDataSet.depth_to_pcl(roi_depth, intrinsic_matrix, roi_coord_2d, valid)

            if len(pcl_in) < 10:
                return None
                assert False, f"Not enough points for pose estimation. {len(pcl_in)} points found"
            ids, pcl_in = Omni6DPoseDataSet.sample_points(pcl_in, self._n_pts)
            xs, ys = xs[ids], ys[ids]

            data = {}
            data['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
            data['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()
            data['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
            data['roi_xs'] = torch.as_tensor(np.ascontiguousarray(xs), dtype=torch.int64).contiguous()
            data['roi_ys'] = torch.as_tensor(np.ascontiguousarray(ys), dtype=torch.int64).contiguous()
            data['roi_center_dir'] = torch.as_tensor(pixel2xyz(img_height, img_height, bbox_center, intrinsics), dtype=torch.float32).contiguous()

        else:
            object_mask = self._masks[ind]
            if not object_mask.any():
                assert False, f"Object {obj_idx} not found in mask"
            if not (self._masks[ind].shape[:2] == self._depth.shape[:2] == self._color.shape[:2]):
                assert False, "depth, masks, and rgb should have the same shape"
            self._depth[self._depth > max_depth] = 0
            intrinsics = self._meta.camera.intrinsics
            intrinsic_matrix = np.array([
                [intrinsics.fx, 0,             intrinsics.cx], 
                [0,             intrinsics.fy, intrinsics.cy], 
                [0,             0,             0]
                ], dtype=np.float32)
            
            img_width, img_height = self._color.shape[1], self._color.shape[0]
            scale_x = img_width / intrinsics.width
            scale_y = img_height / intrinsics.height
            intrinsic_matrix[0] *= scale_x
            intrinsic_matrix[1] *= scale_y

            coord_2d = get_2d_coord_np(img_width, img_height).transpose(1, 2, 0)
            indices = torch.nonzero(object_mask, as_tuple=False)
            ys, xs = indices[:, 0], indices[:, 1]
            rmin, rmax, cmin, cmax = torch.min(ys).cpu().numpy(), torch.max(ys).cpu().numpy(), torch.min(xs).cpu().numpy(), torch.max(xs).cpu().numpy()
            rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], img_height, img_width)

            # here resize and crop to a fixed size 224 x 224
            bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
            bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)

            # crop and resize
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)

            roi_rgb_ = crop_resize_by_warp_affine(
                self._color, bbox_center, scale, self._img_size, interpolation=cv2.INTER_LINEAR
            )
            roi_rgb = Omni6DPoseDataSet.rgb_transform(roi_rgb_)

            # mask_target = torch.zeros_like(self._masks[ind], dtype=torch.float32)
            # mask_target[self._masks[ind] != obj_idx] = 0.0
            # mask_target[self._masks[ind] == obj_idx] = 1.0
            mask_target = self._masks[ind].clone().to(dtype=torch.float32)
            roi_mask = crop_resize_by_warp_affine_torch2(
                mask_target, bbox_center, scale, self._img_size
            )
            roi_mask = torch.unsqueeze(roi_mask, dim=0)

            roi_depth = crop_resize_by_warp_affine(
                self._depth, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
            )

            roi_depth = np.expand_dims(roi_depth, axis=0)
            depth_valid = roi_depth > 0
            if np.sum(depth_valid) <= 1.0:
                # from ipdb import set_trace; set_trace()
                return None
                assert False, "No valid depth values"
            roi_m_d_valid = (roi_mask > 0) & torch.tensor(depth_valid).to(roi_mask.device)
            # print(roi_mask)
            # print(torch.sum(roi_m_d_valid))
            if torch.sum(roi_m_d_valid) <= 1.0:
                # from ipdb import set_trace; set_trace()
                return None
                assert False, "No valid depth values"

            valid = torch.tensor((np.squeeze(roi_depth, axis=0) > 0)).to(roi_mask.device) & (np.squeeze(roi_mask, axis=0) > 0)
            # xs, ys = np.argwhere(valid).transpose(1, 0)
            indices = torch.nonzero(valid, as_tuple=False)
            ys, xs = indices[:, 0], indices[:, 1]
            valid = valid.reshape(-1)
            pcl_in = depth_to_pcl(roi_depth, intrinsic_matrix, roi_coord_2d, valid)

            if len(pcl_in) < 10:
                return None
                assert False, f"Not enough points for pose estimation. {len(pcl_in)} points found"
            ids, pcl_in = sample_points(pcl_in, self._n_pts)
            xs, ys = xs[ids], ys[ids]

            data = {}
            data['pcl_in'] = torch.as_tensor(pcl_in, dtype=torch.float32).contiguous()
            data['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()
            data['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
            data['roi_xs'] = torch.as_tensor(xs, dtype=torch.int64).contiguous()
            data['roi_ys'] = torch.as_tensor(ys, dtype=torch.int64).contiguous()
            data['roi_center_dir'] = torch.as_tensor(pixel2xyz(img_height, img_height, bbox_center, intrinsics), dtype=torch.float32).contiguous()

        return data
    

    def get_objects(self, only_idx=False):
        if hasattr(self, 'data'):
            return self.data
        
        if hasattr(self, 'mask'):
            obj_idx = np.unique(self._mask)
            objects = {}
            if only_idx:
                objects['idx'] = []
            for ind, idx in enumerate(obj_idx):
                if idx == 0:
                    continue
                if np.sum(self._mask == idx) < 10:
                    continue
                obj = self.get_per_object(idx)
                if obj is None:
                    continue
                if only_idx:
                    objects['idx'].append(idx)
                    continue
                for key, value in obj.items():
                    if key not in objects:
                        objects[key] = []
                    objects[key].append(value)

            data = {}
            if only_idx:
                data['idx'] = np.array(objects['idx'])          # [bs, 1]
                return data

            for key, value in objects.items():
                objects[key] = torch.stack(value, dim=0)
                
            PC_da = objects['pcl_in'].to(self._device)
            data['pts'] = PC_da                         # [bs, 1024, 3]
            data['pts_color'] = PC_da                   # [bs, 1024, 3]
            data['roi_rgb'] = objects['roi_rgb'].to(self._device)   # [bs, 3, imgsize, imgsize]
            assert data['roi_rgb'].shape[-1] == data['roi_rgb'].shape[-2]
            assert data['roi_rgb'].shape[-1] % 14 == 0

            data['roi_xs'] = objects['roi_xs'].to(self._device)     # [bs, 1024]
            data['roi_ys'] = objects['roi_ys'].to(self._device)     # [bs, 1024]
            data['roi_center_dir'] = objects['roi_center_dir'].to(self._device)     # [bs, 3]

            """ zero center """
            num_pts = data['pts'].shape[1]
            zero_mean = torch.mean(data['pts'][:, :, :3], dim=1)
            data['zero_mean_pts'] = copy.deepcopy(data['pts'])
            data['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
            data['pts_center'] = zero_mean
            
        else:
            obj_idx = self.obj_ids
            objects = {}
            if only_idx:
                objects['idx'] = []
            for ind, idx in enumerate(obj_idx):
                # print(ind, idx, type(self._masks[ind]), self._masks[ind].any())
                if torch.sum(self._masks[ind]) < 10:
                    continue
                obj = self.get_per_object(idx, ind)
                if obj is None:
                    continue
                if only_idx:
                    objects['idx'].append(idx)
                    continue
                for key, value in obj.items():
                    if key not in objects:
                        objects[key] = []
                    objects[key].append(value)        

            data = {}
            if only_idx:
                data['idx'] = np.array(objects['idx'])            # [bs, 1]
                return data
            
            for key, value in objects.items():
                objects[key] = torch.stack(value, dim=0)
                
            PC_da = objects['pcl_in'].to(self._device)
            data['pts'] = PC_da                         # [bs, 1024, 3]
            data['pts_color'] = PC_da                   # [bs, 1024, 3]
            data['roi_rgb'] = objects['roi_rgb'].to(self._device)   # [bs, 3, imgsize, imgsize]
            assert data['roi_rgb'].shape[-1] == data['roi_rgb'].shape[-2]
            assert data['roi_rgb'].shape[-1] % 14 == 0

            data['roi_xs'] = objects['roi_xs'].to(self._device)     # [bs, 1024]
            data['roi_ys'] = objects['roi_ys'].to(self._device)     # [bs, 1024]
            data['roi_center_dir'] = objects['roi_center_dir'].to(self._device)     # [bs, 3]

            """ zero center """
            num_pts = data['pts'].shape[1]
            zero_mean = torch.mean(data['pts'][:, :, :3], dim=1)
            data['zero_mean_pts'] = copy.deepcopy(data['pts'])
            data['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
            data['pts_center'] = zero_mean

        self.data = data
        return data
    

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, depth):
        self._depth = depth

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def cam_intrinsics(self):
        return self._meta.camera.intrinsics
    
    @cam_intrinsics.setter
    def cam_intrinsics(self, intrinsics):
        self._meta.camera.intrinsics = intrinsics

