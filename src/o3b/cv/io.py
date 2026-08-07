import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import wandb
from PIL import Image
from torchvision import transforms
import h5py
import numpy as np
import time 
from pathlib import Path
import h5py

import math
import shutil
import re
from o3b.multiprocessing import get_num_workers_suggested
from datasets import load_from_disk as hf_load_from_disk

def read_sharded_dataset(sharded_dataset_path):
    logger.info(f"sharded dataset path exists {sharded_dataset_path}...")
    if not "parquet" in str(sharded_dataset_path):
        sharded_dataset = hf_load_from_disk(str(sharded_dataset_path))
    else:
        parquet_files = sorted(list(Path(sharded_dataset_path).glob("*.parquet")))
        parquet_files = [str(f) for f in parquet_files]
        
        logger.info(parquet_files)
        from datasets import load_dataset as hf_load_dataset
        sharded_dataset = hf_load_dataset(f"{str(sharded_dataset_path)}/*.parquet", cache_dir=str(sharded_dataset_cache_path))

        # self.sharded_dataset = hf_load_dataset("parquet", data_files=f"{str(sharded_dataset_path)}/*.parquet", cache_dir=str(sharded_dataset_cache_path), streaming=True)
        #self.sharded_dataset = hf_load_dataset("parquet", data_files=parquet_files, cache_dir=str(sharded_dataset_cache_path))
        sharded_dataset = sharded_dataset["train"]
        
        # # note: worked on roycoffee/slurm for small dataset, but always says generating train split when loading...
        # # note: this creates some cache files, which can generate problems?
        # # t:[Errno 16] Device or resource busy: '.nfs00000000087f385e000001d7'
        # # du -hs /home/sommerl/.cache/huggingface/datasets/parquet
        # # perhaps this is not a good cache place. 
        # from datasets import Dataset as HFDataset
        # self.sharded_dataset = HFDataset.from_parquet(path_or_paths=f"{str(sharded_dataset_path)}/*.parquet")
        # self.sharded_dataset = HFDataset.from_parquet(path_or_paths=parquet_files, cache_dir=str(sharded_dataset_cache_path))
        
    
    logger.info(f"read sharded dataset.")
    return sharded_dataset

def write_sharded_dataset(sharded_dataset, sharded_dataset_path):
    logger.info("saving huggingface dataset...")
    if isinstance(sharded_dataset_path, str):
        sharded_dataset_path = Path(sharded_dataset_path)
    
    if sharded_dataset_path.exists():
        shutil.rmtree(str(sharded_dataset_path))
    
    write_batch_size_max = 1048 # 256, 1048
    num_shards = int(math.ceil(len(sharded_dataset) / write_batch_size_max))
    logger.info(f"number of shards to save: {num_shards} with batch size {write_batch_size_max}...")

    if not "parquet" in str(sharded_dataset_path):
        num_proc = get_num_workers_suggested()
        logger.info(f"number of processes {num_proc}") 
        sharded_dataset.save_to_disk(str(sharded_dataset_path), 
                                        num_shards=num_shards,
                                        num_proc=num_proc)
    else:
        if not sharded_dataset_path.exists():
            sharded_dataset_path.mkdir(parents=True, exist_ok=True)
        output_path_template = str(sharded_dataset_path) + "/{index:05d}.parquet"
        for index in range(num_shards):
            shard = sharded_dataset.shard(index=index, num_shards=num_shards, contiguous=True)
            shard.to_parquet(output_path_template.format(index=index)) # , compression="zstd")
        # without: 1.3GB, snappy: 447M, with zstd: 356M
    logger.info("saved huggingface dataset.")

def parse_subset_fraction(path: str) -> float:
    """
    Parse subset fraction encoded as _sXX_.

    Examples:
    _s01_   -> 0.1
    _s001_  -> 0.01
    _s0001_ -> 0.001
    _s10_   -> 1.0
    """
    if isinstance(path, Path):
        path = str(path)
    
    match = re.search(r'_s(\d+)', path)
    if not match:
        raise ValueError("No subset fraction token '_sXX' found")

    digits = match.group(1)
    value = int(digits)

    if value >= 10:
        return 1.0

    return value / (10 ** (len(digits)-1))


def subset_fraction_to_string(f: float) -> str:
    """
    Convert a fraction to '_sXX_' by rounding to the nearest power-of-ten subset.
    Digits 1–9 are rounded to 0 or 10.
    """
    if not (0 < f <= 1.0):
        raise ValueError("Subset fraction must be in (0, 1]")

    # Full dataset
    if f >= 0.95:
        return "_s10"

    # Determine order of magnitude
    exponent = int(round(-math.log10(f)))

    # Clamp to valid range
    exponent = max(1, exponent)

    return f"_s{'0' * (exponent)}1"

# def subset_fraction_to_string(f: float) -> str:
#     """
#     Convert subset fraction to '_sXX' string.

#     Valid inputs:
#     1.0, 0.1, 0.01, 0.001, ...
#     """
#     if not (0 < f <= 1.0):
#         raise ValueError("Subset fraction must be in (0, 1]")

#     # Full dataset
#     if f == 1.0:
#         return "_s10"

#     # Use Decimal to avoid float precision issues
#     d = Decimal(str(f)).normalize()

#     # Ensure it's exactly a power-of-10 fraction
#     exponent = -d.as_tuple().exponent
#     if d != Decimal(1) / (10 ** exponent):
#         raise ValueError(f"Unsupported subset fraction: {f}")

#     return f"_s{'0' * (exponent - 1)}1"

def multiply_subset_in_path(path: str, factor: float) -> str:
    if isinstance(path, Path):
        path = str(path)
    original = parse_subset_fraction(path)
    factor = factor

    new_fraction = original * factor

    if new_fraction > 1:
        raise ValueError("Resulting subset fraction exceeds 1.0")

    new_token = subset_fraction_to_string(new_fraction)

    return re.sub(r'_s\d+', new_token, path)

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_pts3d_colors(fpath: Path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(fpath))
    return torch.from_numpy(np.asarray(pcd.colors)).to(torch.float)


def read_pts3d(fpath: Path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(fpath))
    return torch.from_numpy(np.asarray(pcd.points)).to(torch.float)


def read_pts3d_with_colors_and_normals(fpath: Path, device="cpu"):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(fpath))
    pts3d = torch.from_numpy(np.asarray(pcd.points)).to(
        dtype=torch.float,
        device=device,
    )
    pts3d_colors = torch.from_numpy(np.asarray(pcd.colors)).to(
        dtype=torch.float,
        device=device,
    )
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
        )
    pts3d_normals = torch.from_numpy(np.asarray(pcd.normals)).to(
        dtype=torch.float,
        device=device,
    )
    return pts3d, pts3d_colors, pts3d_normals


def write_pts3d_with_colors(
    pts3d: torch.Tensor,
    pts3d_colors: torch.Tensor,
    fpath: Path,
):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data
    pcd.points = o3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())

    o3d.io.write_point_cloud(filename=str(fpath), pointcloud=pcd)


def write_pts3d_with_colors_and_normals(
    pts3d: torch.Tensor,
    pts3d_colors: torch.Tensor,
    pts3d_normals: torch.Tensor,
    fpath: Path,
):
    import open3d as o3d
    fpath.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data
    pcd.points = o3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())

    if pts3d_colors is None:
        pts3d_colors = torch.zeros_like(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())

    if pts3d_normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())

    if pts3d_normals is None or not pcd.has_normals():
        pts3d_normals = torch.ones_like(pts3d) * 0.333
        pcd.normals = o3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
        )
    else:
        pcd.normals = o3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())

    o3d.io.write_point_cloud(filename=str(fpath), pointcloud=pcd)


def read_co3d_depth_image(path: Path):
    img = Image.open(path)

    img = (
        np.frombuffer(np.array(img, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((img.size[1], img.size[0]))
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ],
    )

    img = transform(img)
    return img

def read_depth_image(path: Path, factor=1000.):
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    depth = torch.from_numpy(depth / factor)[None,]
    return depth

def write_depth_image(img: torch.Tensor, path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if img.dim() == 3:
        depth = img[0]
    else:
        depth = img
    depth = depth.clone().detach().cpu()
    depth = (depth * 1000).to(torch.long)
    depth = depth.clamp(0, 65535.0).to(torch.uint16)
    # 65m highest, 0.001
    cv2.imwrite(str(path), depth.detach().cpu().numpy().astype(np.uint16))

def write_mask_image(img: torch.Tensor, path: Path):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
        ],
    )
    img = transform((img * 255).to(torch.uint8))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def read_image_exr(fpath: Path):
    """Load the mask image.

    :return: uint8 array of shape (Height, Width), whose values are related
        to the objects' mask ids (:attr:`.image_meta.ObjectPoseInfo.mask_id`).
    """
    import cv2
    import numpy as np
    import torch

    # The OPENCV_IO_ENABLE_OPENEXR gate is armed once in o3b/__init__.py; do not
    # flip it here — cv2 latches the codec state on the first EXR decode, so a
    # read that lands while it is off disables EXR for the rest of the process.
    img = cv2.imread(str(fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read EXR image {fpath}")
    if len(img.shape) == 3:
        img = img[:, :, 2]
    img = np.array(img * 255, dtype=np.uint8)

    img = torch.from_numpy(img)[None,]
    return img

def read_tensor(fpath: Path):
    return torch.load(fpath)

def write_tensor(obj, fpath: Path):
    fpath.parent.mkdir(exist_ok=True, parents=True)
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
    return torch.save(obj=obj, f=fpath)

def read_image(path: Path):
    img = Image.open(path)

    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
        ],
    )

    # Convert the PIL image to Torch tensor
    img = transform(img)
    return img


def write_image(img: torch.Tensor, path: Path):
    path = Path(path)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
        ],
    )
    img = transform(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def image_as_wandb_image(img, caption="Caption Blub"):
    img = wandb.Image(
        img.permute(1, 2, 0).detach().cpu().numpy(),
        caption=caption,
    )
    return img


def watch_model_in_wandb(model, log="gradient", config=None, log_freq=1000):
    """
    args:
        model: (torch.Module) The model to hook, can be a tuple
        log: (str) One of "gradients", "parameters", "all", or None
        config: dict
    """
    wandb.watch(model, log=log, log_freq=log_freq)


def extract_frames_from_video(fpath_video: Path, path_frames: Path, fps=5):
    # vidcap.set(cv2.CAP_PROP_FPS, fps)
    # delete all files in directory path_frmaes
    from o3b.io import rm_dir

    rm_dir(path_frames)
    path_frames.mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(str(fpath_video))
    vid_fps = vidcap.get(cv2.CAP_PROP_FPS)

    # if not path_frames.exists():

    success, image = vidcap.read()
    count_cap = 0
    count_store = 0
    while success:
        if count_cap % int(vid_fps / fps) == 0:
            count_store += 1
            cv2.imwrite(
                str(path_frames.joinpath(f"frame_{count_store}.jpg")),
                image,
            )  # save frame as JPEG file
        # print('Read a new frame: ', success)
        success, image = vidcap.read()
        count_cap += 1
        cv2.waitKey(1)


def write_webm_videos_side_by_side(
    out_fpath: Path,
    in_fpaths=List[Path],
    W=1280,
    padding_size=10,
    padding_color=(0.89, 0.89, 0.89),
):
    from moviepy import VideoFileClip, clips_array, ColorClip

    # Load the webm videos
    video_clips = [VideoFileClip(str(fpath)) for fpath in in_fpaths]

    # Set the desired width and padding size

    # Resize videos to have the same height (keeping the aspect ratio)
    for i in range(len(video_clips)):
        video_clips[i] = video_clips[i].resized(
            height=(W / 2) * video_clips[i].size[1] / video_clips[i].size[0],
        )
    # video1 = video1.resize(height=(final_width / 2) * video1.size[1] / video1.size[0])
    # video2 = video2.resize(height=(final_width / 2) * video2.size[1] / video2.size[0])

    # Create white-gray padding
    padding = ColorClip(
        (padding_size, video_clips[0].h),
        color=(padding_color[0] * 255, padding_color[1] * 255, padding_color[2] * 255),
        duration=video_clips[0].duration,
    )  # .()

    # Combine videos and padding side by side
    video_clips_with_pad = []  #  = [ for video in video_clips]
    for i, video in enumerate(video_clips):
        video_clips_with_pad.append(video)
        if i < len(video_clips) - 1:
            video_clips_with_pad.append(padding)
    final_clip = clips_array([video_clips_with_pad])

    # Write the combined video to a file
    final_clip.write_videofile(str(out_fpath), codec="libvpx", bitrate="5000k")

    # Close the video clips
    for i in range(len(video_clips)):
        video_clips[i].close()



def read_mask_from_video(fpath_video: Path, frame_timestamp: float):
    mask = read_image_from_video(fpath_video, frame_timestamp)
    mask = mask > 125
    mask = mask[:1, :, :]  # return only one channel
    return mask

def read_image_from_video(fpath_video: Path, frame_timestamp: float):

    if fpath_video.suffix in [".mkv", ".mp4", ".avi", ".mov"]:
        import cv2 
        capture = cv2.VideoCapture(str(fpath_video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
        #capture.set(cv2.CAP_PROP_POS_FRAMES, frame_timestamp)
        capture.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp * 1000)

        ret, image = capture.read()
        capture.release()
        
        if not ret:
            logger.debug(f"Failed to read RGB frame {frame_timestamp} from video {fpath_video}")
            return None 
        image = image[..., ::-1].copy()  # BGR to RGB
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.to(torch.uint8) 


def read_depth_from_video(h5_path: Path, h5_frame_number: int, depth_scale: float = 1.0):
    if not h5_path.exists():
        raise FileNotFoundError(f"Depth video {h5_path} does not exist.")
    with h5py.File(h5_path, "r") as h5file:
        depth_map = h5file[str(h5_frame_number)][:].astype(np.float32)
        depth_map *= depth_scale
        depth_mask = (np.isfinite(depth_map) & (depth_map > 0.0)) # .astype(np.float32)
        # remove infinite values
        depth_map[~depth_mask] = 0.0
        depth_map = depth_map[None,]
        depth_map = torch.from_numpy(depth_map)
    return depth_map
