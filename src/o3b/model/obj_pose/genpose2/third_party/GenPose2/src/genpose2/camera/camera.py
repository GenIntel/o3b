import numpy as np
# import rospy
# import pyrealsense2 as rs
import OpenEXR
import Imath
import cv2

# from std_msgs.msg import Time, Bool
import os
# os.path.join(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from .images import *
from .constants import *
from .realsense_helper import get_profiles
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pickle
import torch
import time
import timeit
import argparse
import json

from PIL import Image as PILImage
import torch
import sys
sys.path.insert(0, './segment-anything-2-real-time')
from sam2.build_sam import build_sam2_camera_predictor
import matplotlib.pyplot as plt
from termcolor import cprint

def show_mask_and_video(mask, color_image, obj_id=None, random_color=False):
    mask = mask.cpu().numpy()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Clear previous frame and plot the current one
    plt.clf()  # Clears the figure for the new frame
    ax = plt.gca()  # Get current axes for plotting
    plt.imshow(color_image)  # Display the color image
    ax.imshow(mask_image)
    plt.draw()  # Redraw the figure to update the mask on the image
    plt.pause(0.01)  # Pause to update the plot for each frame


def show_mask_and_video_binary(mask):
    # Ensure mask is in NumPy format and reshape if needed
    mask = mask.reshape(HEIGHT, WIDTH) # Convert 0/1 mask to 0/255

    # Display mask using OpenCV
    cv2.imshow("Binary Mask", mask)
    cv2.waitKey(1)  # Refresh the frame


import torch.nn.functional as F
def convert_mask_for_ros(mask: torch.Tensor) -> torch.Tensor:
    """ Fully GPU-based mask processing to speed up erosion and conversion. """
    if not mask.is_cuda:
        mask = mask.cuda()  # Ensure mask is on GPU

    # Convert to uint8 and add batch + channel dims for compatibility with PyTorch erosion
    mask = mask.squeeze().unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, H, W)

    erode = True

    if erode:
        # Create erosion kernel directly on GPU
        kernel_size = 15
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)

        # Apply morphological erosion using 2D convolution (fast and parallelized on GPU)
        mask_eroded = F.conv2d(mask, kernel, padding=kernel_size//2)
        final_mask = (mask_eroded > (kernel_size**2 // 2)).to(torch.uint8) * 255  # Thresholding
    else:
        final_mask = mask * 255
    # Convert back to binary mask and flatten (remains on GPU)
    # binary_mask = mask_eroded.view(-1) * 255  # Flatten and scale

    return final_mask.to(torch.uint8).cpu().numpy()  # Still on GPU


@torch.jit.script
def depth_image_to_point_cloud_GPU_with_seg(camera_tensor, seg_camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)
    seg_buffer = seg_camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)
    seg = seg_buffer.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]
    seg_valid = seg[valid].unsqueeze(1)

    points_with_seg = torch.cat((points, seg_valid), dim=1)

    return points_with_seg


def save_exr(filename, image):
    """
    保存 2-D uint8 或 float32 单通道掩码为 .exr
    不转换 dtype
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image 必须是 2-D 单通道数组")

    height, width = image.shape

    # print(image.dtype)
    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        image = image.astype(np.uint32)
        pixel_type = Imath.PixelType(Imath.PixelType.UINT)
    elif image.dtype in [np.float32, np.float64]:
        # print("float")
        image = image.astype(np.float32)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    else:
        raise TypeError("only uint8 or float32")

    header = OpenEXR.Header(width, height)
    header['channels'] = {'Y': Imath.Channel(pixel_type)}

    exr = OpenEXR.OutputFile(filename, header)
    
    exr.writePixels({'Y': image.tobytes()})
    exr.close()

class RealSenseRobotStream(object):
    def __init__(self, cam_serial_num, robot_cam_num, rotation_angle = 0, mode='rgbd', use_sam=False, sam2_path=None):
        import open3d as o3d
        self.mode = mode
        self.cam_serial_num = cam_serial_num

        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.voxel_size = 0.005
        self.first_frame = True
        self.use_sam = use_sam
        self.reset_mask = None
        self.sam2_path = sam2_path
        self.if_init = False

        # Initializing ROS Node

        # Disabling scientific notations
        np.set_printoptions(suppress=True)

        # Setting rotation settings
        self.rotation_angle = rotation_angle

        # Starting the realsense camera stream
        self._start_realsense(PROCESSING_PRESET) 

        # Initialize SAM
        # if use_sam:
        #     sam2_checkpoint = self.sam2_path or "./segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"
        #     model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        #     device = 'cuda'
        #     self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)
        #     self.if_init = False

            
        print(f"Started the Realsense pipeline for camera: {self.cam_serial_num}!")

    def reset_mask_callback(self, msg):
        self.reset_mask = msg.data

    def _start_realsense(self, processing_preset):
        
        import pyrealsense2 as rs
        config = rs.config()
        pipeline = rs.pipeline()
        config.enable_device(self.cam_serial_num)

        # Enabling camera streams
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, CAM_FPS)
        if self.mode == 'rgbd':
            config.enable_stream(rs.stream.depth, WIDTH, HEIGHT,rs.format.z16, CAM_FPS)


        # Starting the pipeline
        cfg = pipeline.start(config)
        device = cfg.get_device()

        # if self.cam_serial_num == "211222065122": # D415
        #     device.hardware_reset()

        # if self.mode == 'rgbd':
        #     # Setting the depth mode to high accuracy mode
            # depth_sensor = device.first_depth_sensor()
            # depth_sensor.set_option(rs.option.visual_preset, processing_preset) # High accuracy post-processing mode
        self.realsense = pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])
        print(f"Camera {self.cam_serial_num} intrinsics matrix: {self.intrinsics_matrix}")
        self.distortion_coefficients = np.array(intrinsics.coeffs) # k1, k2, p1, p2, k3

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)
        # if self.cam_serial_num in ["251622062545"]: # D415
        #     self.align = rs.align(rs.stream.color)
        # elif self.cam_serial_num == "230322275548":
        #     self.align = rs.align(rs.stream.color)

        if self.cam_serial_num in ["251622062545"]: # D415
            sensor = profile.get_device().query_sensors()[1]
            sensor.set_option(rs.option.auto_exposure_priority, True)
            print(sensor.get_option(rs.option.exposure)) 
        elif self.cam_serial_num == "230322275548":
            sensor = profile.get_device().query_sensors()[0]
            print(sensor.get_option(rs.option.exposure)) 
        else:
            sensor = profile.get_device().query_sensors()[1]
            sensor.set_option(rs.option.auto_exposure_priority, True)
            print(sensor.get_option(rs.option.exposure))
    
    def reset_mask_selection(self):
        """Prepare the tracker for a brand‑new object selection."""
        self.if_init = False            # make get_mask() open the click‑UI again
        if hasattr(self, "predictor"):
            # safest: wipe any internal state; if `reset()` is unavailable,
            # just re‑create the predictor here.
            try:
                self.predictor.reset()
            except AttributeError:
                # fall‑back: re‑create
                sam2_checkpoint = self.sam2_path or "./segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
                device = 'cuda'
                self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)
        else:
            sam2_checkpoint = self.sam2_path or "./segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            device = 'cuda'
            self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)
                

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            if self.mode == 'rgbd':
                aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            if self.mode == 'rgbd':
                if self.cam_serial_num == "f1230963": 
                    depth_image = (
                        np.asanyarray(aligned_depth_frame.get_data()) // 4
                    )  # L515 camera need to divide by 4 to get metric in meter   
                elif self.cam_serial_num in ["251622062545"]:
                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                else:
                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if self.mode == 'rgbd':
                return color_image, depth_image
            else:
                return color_image
            
    def stream(self, test=False, show_mask=False, save_path=None):
        print("Starting stream!\n")
        start_time = time.time()
        prev_reset = False
        count = 0
        while True:
            t0 = time.time()
            if self.mode == 'rgbd':
                t1 = time.time()
                color_image, depth_image = self.get_rgb_depth_images()
                t2 = time.time()
                color_image, depth_image = rotate_image(color_image, self.rotation_angle), rotate_image(depth_image, self.rotation_angle)
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                t3 = time.time()
            elif self.mode == 'rgb':
                color_image = self.get_rgb_depth_images()
                color_image = rotate_image(color_image, self.rotation_angle)
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            masks, obj_id = self.get_mask(color_image)
            # binary_mask = convert_mask_for_ros(mask)
            if show_mask:
                show_mask_and_video(masks[0], color_image, obj_id=obj_id)
                # show_mask_and_video_binary(binary_mask)

            intr = self.intrinsics_matrix
            height, width = depth_image.shape
            meta = {
                'camera':{
                    'intrinsics':{
                        'fx': intr[0, 0],
                        'fy': intr[1, 1],
                        'cx': intr[0, 2],
                        'cy': intr[1, 2],
                        'width': width,
                        'height': height,
                    }
                }
            }

            masks = torch.as_tensor(masks[:,0,:,:], dtype=torch.bool)
            # print(masks.shape)
            # mask = torch.zeros(1, 360, 640, dtype=torch.int32)
            # for i in range(masks.shape[0]):
            #     mask[0][masks[i, 0]] = i+1

            if save_path:
                cv2.imwrite(os.path.join(save_path, f"{count:04d}_color.png"), color_image)
                depth_save_image = depth_image / 1000.0
                save_exr(os.path.join(save_path, f"{count:04d}_depth.exr"), depth_save_image)
                mask_image = torch.zeros(1, 360, 640, dtype=torch.float32)
                for i in range(masks.shape[0]):
                    mask_image[0][masks[i]] = i+1
                # mask_image = masks
                mask_image = (mask_image[0] / torch.max(mask_image[0]) * 255).cpu().numpy().astype(np.uint8)
                # mask_image = mask_image.cpu().numpy().astype(np.float32)
                save_exr(os.path.join(save_path, f"{count:04d}_mask.exr"), mask_image)
                with open(os.path.join(save_path, f"{count:04d}_meta.json"), 'w') as f:
                    json.dump(meta, f, indent=4)
                mask_image_png = mask_image / mask_image.max() * 255
                cv2.imwrite(os.path.join(save_path, f"{count:04d}_mask.png"), mask_image_png)
                count += 1
                
            t4 = time.time()

            # self.rate.sleep()
            cprint(f"get rgbd freq: {1 / (t2-t1)}", "blue")
            cprint(f"other operation freq: {1 / (t3-t2)}", "blue")
            cprint(f"get mask freq: {1/(t4-t3+0.00000001)}", "blue")
            cprint(f"Send image frequency: {1/(time.time()-t0)}", "green", "on_black")

            cur_reset = self.reset_mask
            if cur_reset and not prev_reset:
                cprint(">>> Ready to select a NEW mask – click on the object", "yellow")
                self.reset_mask_selection()
            prev_reset = cur_reset

            # if count == 130:
            #     break
            if test:
                yield color_image, depth_image, masks, meta, obj_id
        
    
    def colorize_depth(self, depth_image, min_depth=0, max_depth=3000):
        """
        Converts a depth image to a colored representation.
        
        Args:
            depth_image: NumPy array of the depth image.
            min_depth: Minimum depth value to normalize (default 0mm).
            max_depth: Maximum depth value to normalize (default 3000mm = 3m).
        
        Returns:
            A colorized depth image.
        """
        # Convert depth to 8-bit format
        depth_normalized = np.clip(depth_image, min_depth, max_depth)  # Clip values
        depth_normalized = ((depth_normalized - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        return depth_colored

    def show_rgbd_in_3d(color_image, depth_image, intrinsics):
        """
        Converts RGB-D images into a 3D point cloud and visualizes it.

        Args:
            color_image: RGB image as a NumPy array.
            depth_image: Depth image as a NumPy array (same size as color_image).
            intrinsics: 3x3 intrinsic camera matrix.
        """
        import open3d as o3d
        height, width = depth_image.shape

        # Create Open3D images
        color_o3d = o3d.geometry.Image(color_image)
        depth_o3d = o3d.geometry.Image(depth_image)

        # Convert intrinsics to Open3D format
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Create RGB-D image in Open3D
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d)

        # Flip the point cloud to align with Open3D coordinate system
        pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],  # Flip Y axis
                    [0, 0, -1, 0],  # Flip Z axis
                    [0, 0, 0, 1]])

        # Visualize the 3D point cloud
        o3d.visualization.draw_geometries([pcd])

    def create_point_cloud(self, colors, depths, cam_intrinsics, voxel_size=0.005, repr_frame="camera"):
        """
        color, depth => point cloud
        """
        import open3d as o3d
        h, w = depths.shape
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )

        # if isinstance(depths, o3d.geometry.Image):
        #     raise TypeError("Expected a NumPy array for depths, but got an Open3D Image.")
        depths_np = np.asarray(depths).astype(np.float32)
        print(f"Depth min: {depths_np.min()}, max: {depths_np.max()}")

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale=1000.0, convert_rgb_to_intensity=False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(voxel_size)
        # points = np.array(cloud.points).astype(np.float32)
        # colors = np.array(cloud.colors).astype(np.float32)

        if len(cloud.points) == 0:
            print("⚠️ Warning: No valid 3D points generated.")
            return None

        # Update point cloud
        self.pcd.points = cloud.points
        self.pcd.colors = cloud.colors

        if self.first_frame:
            self.vis.add_geometry(self.pcd)
            self.first_frame = False
        else:
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    # def get_mask(self, color_image, points=np.array([[400, 200]], dtype=np.float32)):
    #     ann_frame_idx = 0  # the frame index we interact with
    #     ann_obj_id = 0
    #     if not self.if_init:
    #         predictor.load_first_frame(color_image)
    #         self.if_init = True
    #         # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(ann_frame_idx, ann_obj_id, points)
    #         _, out_obj_ids, out_mask_logits = predictor.add_new_points(
    #             frame_idx=ann_frame_idx,
    #             obj_id=ann_obj_id,
    #             points=points,
    #             labels=np.array([1], np.int32),
    #         )
    #     else:
    #         out_obj_ids, out_mask_logits = predictor.track(color_image)
    #     obj_id = out_obj_ids[0]
    #     mask = (out_mask_logits[0] > 0.0)

    #     return mask, obj_id
    

    def get_mask(self, color_image):
        """Allows the user to select multiple points and gets the segmentation mask."""
        
        
        # Segmentation logic
        
        # if not hasattr(get_mask, "if_init"):
        #     predictor.load_first_frame(color_image)
        #     get_mask.if_init = True
        #     _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        #         frame_idx=ann_frame_idx,
        #         obj_id=ann_obj_id,
        #         points=points,
        #         labels=np.ones(len(points), np.int32),  # Assign all points as positive labels
        #     )
        # else:
        #     out_obj_ids, out_mask_logits = predictor.track(color_image)
        if not hasattr(self, "predictor"):
            mask = np.zeros((1, 1, HEIGHT, WIDTH))
            # print(mask)
            mask = torch.as_tensor(mask, dtype=torch.bool)
            obj_id = []
            # print(mask.shape, obj_id)
            return mask, obj_id

        if not self.if_init:
            self.predictor.load_first_frame(color_image)

            # Show image and let user select points
            ann_frame_idx = 0  # The frame index we interact with
            ann_obj_id = 0
            while True:
                selected_points = []
                def click_event(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        selected_points.append([x, y])
                        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                        cv2.imshow("Select Points", param)

                image_copy = color_image.copy()
                cv2.imshow("Select Points", image_copy)
                cv2.setMouseCallback("Select Points", click_event, image_copy)
                p = cv2.waitKey(0)
                cv2.destroyAllWindows()

                points = np.array(selected_points, dtype=np.float32)
                if points.size == 0:
                    continue
                ann_obj_id += 1

                # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(ann_frame_idx, ann_obj_id, points)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=np.array([1] * points.shape[0], dtype=np.int32),
                )

                if p == ord('q'):
                    break

            self.if_init = True
        else:
            out_obj_ids, out_mask_logits = self.predictor.track(color_image)

        # obj_id = out_obj_ids[0]
        # mask = (out_mask_logits[0] > 0.0)
        obj_id = out_obj_ids
        masks = (out_mask_logits > 0.0)
        # print(np.unique(mask.cpu().numpy()))
        # print(mask.shape)
        # print(mask)
        # print(obj_id)

        return masks, obj_id



if __name__ == '__main__':
    # context = rs.context()
    # devices = context.devices
    # for i, device in enumerate(devices):
    #     print(f"Device {i}: {device.get_info(rs.camera_info.serial_number)}")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cam_num", type=str, default="back")

    args = parser.parse_args()

    from PIL import Image as PILImage
    import torch
    from sam2.build_sam import build_sam2_camera_predictor
    import matplotlib.pyplot as plt

    if args.cam_num == "back":
        cam_serial_num = "211222065122" # 415 back
    elif args.cam_num == "side":
        cam_serial_num = "332122060334" # 415 side
    elif args.cam_num == "fengwu":
        cam_serial_num = "251622062545" # 415 fengwu
    else:   
        raise NotImplementedError()

    # robot_cam_num = 1
    robot_cam_num = int(cam_serial_num[0])
    rotation_angle = 0
    mode = 'rgbd'
    rs_streamer = RealSenseRobotStream(cam_serial_num, robot_cam_num, rotation_angle, mode ,use_sam=(args.cam_num=="fengwu"))
    # device = 'cuda'

    # Set up the plot ahead of time
    plt.figure(figsize=(6, 4))

    save_path = "./saved_images/"
    os.makedirs(save_path, exist_ok=True)

    # Start the loop for displaying the frames
    while True:
        color_image, depth_image = rs_streamer.stream(show_mask=False, save_path=save_path)

        break
