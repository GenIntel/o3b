import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import torch
import random
import gc
import cv2
import pyrealsense2 as rs
import pyrealsense2 as rs
import numpy as np
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ipdb import set_trace
from sklearn.cluster import DBSCAN
from genpose2.networks.posenet_agent import PoseNet
from genpose2.networks.reward import sort_poses_by_energy
from genpose2.utils.metrics import get_rot_matrix
from genpose2.utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from genpose2.utils.misc import average_quaternion_batch, get_pose_dim, get_pose_representation
from genpose2.utils.so3_visualize import visualize_so3
from cutoop.eval_utils import DetectMatch, Metrics
from genpose2.configs.config import get_config
from genpose2.datasets.datasets_infer import InferDataset

from flask import Flask, request
flask_app = Flask(__name__)


class GenPose2:
    def __init__(self, score_model_path:str, energy_model_path:str, scale_model_path:str):
        ''' load config '''
        self.cfg = self._get_config(score_model_path, energy_model_path, scale_model_path)

        ''' set random seed '''
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        ''' load score model '''
        self.cfg.agent_type = 'score'
        self.score_agent = PoseNet(self.cfg)
        self.score_agent.load_ckpt(model_dir=self.cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        self.score_agent.eval()

        ''' load energy model '''
        self.cfg.agent_type = 'energy'
        self.energy_agent = PoseNet(self.cfg)
        self.energy_agent.load_ckpt(model_dir=self.cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
        self.energy_agent.eval()

        ''' load scale model '''
        self.cfg.agent_type = 'scale'
        self.scale_agent = PoseNet(self.cfg)
        self.scale_agent.load_ckpt(model_dir=self.cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
        self.scale_agent.eval()

    def _get_config(self, score_model_path:str, energy_model_path:str, scale_model_path:str):
        cfg = get_config()
        cfg.pretrained_score_model_path = score_model_path
        cfg.pretrained_energy_model_path = energy_model_path
        cfg.pretrained_scale_model_path = scale_model_path
        cfg.sampler_mode=['ode']
        cfg.T0 = 0.55
        cfg.seed = 0
        cfg.eval_repeat_num = 50
        cfg.dino = 'pointwise'
        return cfg
        
    def _inference_score(self, data:InferDataset, score_agent:PoseNet, prev_pose=None, T0=None):
        all_pred_pose = []
        all_score_feature = []

        for i, batch_sample in enumerate([data.get_objects()]):
            if prev_pose is not None:
                _prev_pose = prev_pose[i].clone()
                _prev_pose[:, -3:] -= batch_sample['pts_center']    
            pred_results = score_agent.pred_func(
                data=batch_sample, 
                repeat_num=self.cfg.eval_repeat_num, 
                T0=self.cfg.T0 if T0 is None else T0,
                init_x=None if prev_pose is None else _prev_pose,
                return_average_res=False,
                return_process=False
            )
            pred_pose, _ = pred_results
            all_pred_pose.append(pred_pose)
            all_score_feature.append({
                'pts_feat': batch_sample['pts_feat'].cpu(),
                'rgb_feat': (None if batch_sample['rgb_feat'] is None else batch_sample['rgb_feat'].cpu()),
            })
            if i % 4 == 3:
                gc.collect()
        
        return all_pred_pose, all_score_feature

    def _inference_energy(self, data:InferDataset, energy_agent:PoseNet, all_pred_pose):
        all_pred_energy = []

        for i, batch_sample in enumerate([data.get_objects()]):
            pred_energy = energy_agent.get_energy(
                data=batch_sample, 
                pose_samples=all_pred_pose[i], 
                T=None, 
                mode='test', 
                extract_feature=True
            )
            all_pred_energy.append(pred_energy.cpu())
            if i % 4 == 3:
                gc.collect()
        
        return all_pred_energy

    def _aggregate_pose(self, all_pred_pose, all_pred_energy):
        if all_pred_energy is None:
            all_pred_energy = [torch.ones(*(all_pred_pose[i].shape[:2]), 2) 
                            for i in range(len(all_pred_pose))]

        all_aggregated_pose = []
        
        for i, (pred_pose, pred_energy) in enumerate(zip(all_pred_pose, all_pred_energy)):
            sorted_pose, sorted_energy = sort_poses_by_energy(pred_pose, pred_energy)
            bs = pred_pose.shape[0]
            retain_num = int(self.cfg.eval_repeat_num * self.cfg.retain_ratio)
            good_pose = sorted_pose[:, :retain_num, :]
            rot_matrix = get_rot_matrix(good_pose[:, :, :-3].reshape(bs * retain_num, -1), self.cfg.pose_mode)
            quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
            aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)
            if self.cfg.clustering:
                for j in range(bs):
                    # https://math.stackexchange.com/a/90098
                    # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
                    pairwise_distance = 1 - torch.sum(quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1), dim=2) ** 2
                    dbscan = DBSCAN(eps=self.cfg.clustering_eps, min_samples=int(self.cfg.clustering_minpts * retain_num)).fit(pairwise_distance.cpu().cpu().numpy())
                    labels = dbscan.labels_
                    if np.any(labels >= 0):
                        bins = np.bincount(labels[labels >= 0])
                        best_label = np.argmax(bins)
                        aggregated_quat_wxyz[j] = average_quaternion_batch(quat_wxyz[j, labels == best_label].unsqueeze(0))[0]
            aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
            aggregated_pose = torch.zeros(bs, 4, 4)
            aggregated_pose[:, 3, 3] = 1
            aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
            aggregated_pose[:, :3, 3] = aggregated_trans
            all_aggregated_pose.append(aggregated_pose)
            if i % 10 == 9:
                gc.collect()
        
        return all_aggregated_pose

    def _inference_scale(self, data:InferDataset, scale_agent:PoseNet, all_score_feature, all_aggregated_pose):
        if self.cfg.pretrained_scale_model_path is None:
            all_final_length = []

            for i, test_batch in enumerate([data.get_objects()]):
                pcl: torch.Tensor = test_batch['pcl_in'] # [bs, 1024, 3]
                rotation: torch.Tensor = all_aggregated_pose[i][:, :3, :3] # [bs, 3, 3]
                rotation_t = torch.transpose(rotation, 1, 2) # [bs, 3, 3]
                translation: torch.Tensor = all_aggregated_pose[i][:, :3, 3] # [bs, 3]

                n_pts = pcl.shape[1]
                pcl = pcl - translation.unsqueeze(1) # [bs, 1024, 3]
                pcl = pcl.reshape(-1, 3, 1) # [bs * 1024, 3, 1]
                rotation_t = torch.repeat_interleave(rotation_t, n_pts, dim=0) # [bs * 1024, 3, 3]
                pcl = torch.bmm(rotation_t, pcl).reshape(-1, n_pts, 3) # [bs, 1024, 3]

                bbox_length, _ = torch.max(torch.abs(pcl), dim=1)
                bbox_length *= 2
                all_final_length.append(bbox_length.cpu())

                if i % 10 == 9:
                    gc.collect()

            return all_aggregated_pose, all_final_length
        
        all_final_pose = []
        all_final_length = []

        for i, batch_sample in enumerate([data.get_objects()]):
            batch_sample.update({key: (None if value is None else value.to(self.cfg.device)) 
                                for key, value in all_score_feature[i].items()})
            batch_sample['axes'] = all_aggregated_pose[i][:, :3, :3].to(self.cfg.device)
            cal_mat, length = scale_agent.pred_scale_func(batch_sample)
            final_pose = all_aggregated_pose[i].clone()
            final_pose[:, :3, :3] = cal_mat.cpu()
            all_final_pose.append(final_pose.cpu())
            all_final_length.append(length.cpu())
            if i % 4 == 3:
                gc.collect()
        
        return all_final_pose, all_final_length

    def visualize_pose_distribution(self, data:InferDataset, all_pred_pose):
        for i, test_batch in enumerate([data.get_objects()]):
            pred_pose = all_pred_pose[i][:, :, :-3]
            pose_rot = get_rot_matrix(pred_pose.reshape(pred_pose.shape[0] * self.cfg.eval_repeat_num, -1), self.cfg.pose_mode) \
                        .reshape(pred_pose.shape[0], self.cfg.eval_repeat_num, 3, 3)
            avg_pose_rot = get_rot_matrix(average_quaternion_batch(matrix_to_quaternion(pose_rot)), 'quat_wxyz')
            for j in range(pred_pose.shape[0]):
                index = i * self.cfg.batch_size + j
                visualize_so3(
                    save_path='./so3_distribution.png', 
                    pred_rotations=pose_rot[j].cpu().numpy(),
                    pred_rotation=avg_pose_rot[j].cpu().numpy(),
                    gt_rotation=None,
                    # probabilities=confidence
                )
                # all_dm.draw_image(index=index)
                set_trace()

    def inference(self, data:InferDataset, prev_pose=None, tracking=False, tracking_T0=0.15):
        prev_pose_genpose2 = None
        if prev_pose is not None:
            prev_pose_genpose2 = []
            for item in prev_pose:
                pose_genpose2 = torch.zeros(item.shape[0], get_pose_dim(self.cfg.pose_mode), device=self.cfg.device)
                pose_genpose2[:, :-3] = get_pose_representation(item[:, :3, :3], self.cfg.pose_mode)
                pose_genpose2[:, -3:] = item[:, :3, 3]
                prev_pose_genpose2.append(pose_genpose2)
        
        infer_T0 = tracking_T0 if tracking and prev_pose is not None else None
        all_pred_pose, all_score_feature = self._inference_score(data, self.score_agent, prev_pose_genpose2, infer_T0)
        if self.cfg.pretrained_energy_model_path is not None:
            all_pred_energy = self._inference_energy(data, self.energy_agent, all_pred_pose)
            all_aggregated_pose = self._aggregate_pose(all_pred_pose, all_pred_energy)
        else:
            all_aggregated_pose = self._aggregate_pose(all_pred_pose, None)

        all_final_pose, all_final_length = self._inference_scale(data, self.scale_agent, all_score_feature, all_aggregated_pose)

        return all_final_pose, all_final_length


def create_genpose2(score_model_path:str, energy_model_path:str, scale_model_path:str):
    return GenPose2(score_model_path, energy_model_path, scale_model_path)

def visualize_pose(data:InferDataset, all_final_pose, all_final_length, visualize_pts=False, visualize_image=False):
    import open3d as o3d
    color_img = cv2.cvtColor(data.color, cv2.COLOR_RGB2BGR)
    all_final_pose = all_final_pose[0].cpu().numpy()
    all_final_length = all_final_length[0].cpu().numpy()

    for index, (obj_pose, obj_length) in enumerate(zip(all_final_pose, all_final_length)):
        if visualize_pts:
            pts = data.get_objects()['pts'].cpu().numpy()[index]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.visualization.draw_geometries([pcd])
        color_img = DetectMatch._draw_image(
            vis_img=color_img,
            pred_affine=obj_pose,
            pred_size=obj_length,
            gt_affine=None,
            gt_size=None,
            gt_sym_label=None,
            camera_intrinsics=data.cam_intrinsics,
            draw_pred=True,
            draw_gt=False,
            draw_label=False,
            draw_pred_axes_length=0.1,
            draw_gt_axes_length=None,
            thickness=True,
        )
    
    if visualize_image:
        cv2.namedWindow('rgb')
        cv2.imshow('rgb', color_img)
        cv2.waitKey() 
        cv2.destroyAllWindows()
    return color_img


def main():
    ######################################## PARAMETERS ########################################
    DATA_PATH = 'data/Omni6DPose/ROPE/000007'                 # Path to the data
    TRACKING = True                                           # Tracking mode

    # Tracking parameter, if the relative pose between the current frame and the previous frame
    # is large, such as low video FPS or fast object motion, you can set a larger value. The default
    # TRACKING_TO is set to 0.15.
    TRACKING_T0 = 0.15

    SCORE_MODEL_PATH='results/ckpts/ScoreNet/scorenet.pth'     # Path to the score model
    ENERGY_MODEL_PATH='results/ckpts/EnergyNet/energynet.pth'  # Path to the energy model
    SCALE_MODEL_PATH='results/ckpts/ScaleNet/scalenet.pth'     # Path to the scale model
    PREV_POSE = None                                           # Previous pose
    ######################################## PARAMETERS ########################################

    ''' load data '''
    # Get data from image file
    color_images = sorted(glob.glob(DATA_PATH + '/*_color.png'))
    GenPose2 = create_genpose2(
        score_model_path=SCORE_MODEL_PATH, 
        energy_model_path=ENERGY_MODEL_PATH,
        scale_model_path=SCALE_MODEL_PATH,
    )
    
    cv2.namedWindow('rgb')
    for index, color_image in enumerate(tqdm(color_images)):
        data_prefix = color_image.replace('color.png', '')
        data = InferDataset.alternetive_init(data_prefix, img_size=GenPose2.cfg.img_size, device=GenPose2.cfg.device, n_pts=GenPose2.cfg.num_points)
        pose, length = GenPose2.inference(data, PREV_POSE, TRACKING, TRACKING_T0)
        color_image_w_pose = visualize_pose(data, pose, length, visualize_image=False)
        PREV_POSE = pose
        cv2.imshow('rgb', color_image_w_pose)
        cv2.waitKey(1) 

    cv2.destroyAllWindows()    


if __name__ == '__main__':
    main()


