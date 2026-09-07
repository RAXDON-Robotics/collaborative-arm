"""
Script to convert h5 files to lerobot format.

example:
python utils/convert_h5_to_lerobot_v30.py \
  --config.h5-raw-dir data/piper_place_and_place_0722/ \
  --config.repo-id piper/piper_place_and_place_0722
"""

import tyro
import shutil
import h5py
import tqdm
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class CovertConfig:
  h5_raw_dir: Path
  repo_id: str
  # True: single arm, False: dual arm
  single_arm: bool = True
  # fix camera names use your camera config
  # cam_names: list[str] = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
  cam_names: List[str] = field(default_factory=lambda: ["cam_high", "cam_right_wrist"])
  has_velocity: bool = False
  has_effort: bool = False
  fps: int = 30

  robot_type: str = "RAXDON_Y1"
  push_to_hub: bool = False
  
  use_video: bool = True
  tolerance_s: float = 0.0001
  image_writer_processes: int = 4
  image_writer_threads: int = 5
  
class LoadH5Dataset:
  def __init__(self, config: CovertConfig) -> None:
    """
    Initialize the dataset for loading and processing HDF5 files contaning Expert demonstration data.
    
    Arguments:
      data_dir (Path): The directory containing the HDF5 files.
    """
    self.config = config
    self.data_dir  = config.h5_raw_dir.resolve()
    if not self.data_dir.exists():
      raise ValueError(f"Data directory {self.data_dir} does not exist.")
    
    hdf5_files = list(self.data_dir.glob("*.hdf5"))
    if len(hdf5_files) == 0:
      raise ValueError(f"Data directory {self.data_dir} is empty.")
    
    # sort
    self.datasets = sorted(
            hdf5_files,
            key=lambda p: int(p.stem.rsplit("_", 1)[1])
        )
    # print("hdf5_files: ", self.datasets)

    # load first dataset to evaluate config
    with h5py.File(self.datasets[0], "r") as ep:
        # evaluate state and action shape
        state = ep['/observation/state'][()]
        action = ep['/action'][()]
        if self.config.single_arm:
           assert state.shape[-1] == 7 and action.shape[-1] == 7
        else:
           assert state.shape[-1] == 14 and action.shape[-1] == 14

        # whether has velocity
        if config.has_velocity:
           assert "/observation/qvel" in ep

        # whether has effort
        if config.has_effort:
           assert "/observation/effort" in ep

        # evaluate camera
        for cam in self.config.cam_names:
           assert f"/observation/images/{cam}" in ep
    
  def __len__(self) -> int:
    """"Return the number of collected episodes."""
    return len(self.datasets)
  
  def get_item(self, index: int) -> dict:
    """Return the dataset at the given index."""
    episode_data = {}
    
    # load hdf5 file
    with h5py.File(self.datasets[index], 'r') as root:
      """observation"""
      # state(joint position)
      episode_data["state"] = root['/observation/state'][()]
      
      # joint velocity
      if self.config.has_velocity:
         episode_data["velocity"] = root['/observation/velocity'][()]
      
      # joint effort
      if self.config.has_effort:
         episode_data["effort"] = root['/observation/effort'][()]
      
      # cameras
      cameras = {}
      images_grp = root['/observation/images/']
      for cam_name in images_grp.keys():
        img_bytes_seq = images_grp[cam_name][()]
        # img_bytes_seq 可能是一个包含多帧字节的 NumPy 数组
        frames = []
        for frame_bytes in img_bytes_seq:
            buf = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode frame from camera '{cam_name}'.")
            frames.append(img)
        cameras[cam_name] = frames
        
      episode_data["cameras"] = cameras
      
      # action
      episode_data["action"] = root['/action'][()]
      
      # episode length
      episode_data["episode_length"] = len(root['/observation/state'][()])
      
      # task description
      episode_data["task_description"] = root.attrs['task']
    
    return episode_data
  
def create_empty_dataset(config: CovertConfig) -> LeRobotDataset:
    if config.single_arm:
      motors = ["Left_J1", "Left_J2", "Left_J3", "Left_J4", "Left_J5", "Left_J6", "Left_Gripper"]
    else:
      motors = ["Left_J1", "Left_J2", "Left_J3", "Left_J4", "Left_J5", "Left_J6", "Left_Gripper",
                "Right_J1", "Right_J2", "Right_J3", "Right_J4", "Right_J5", "Right_J6", "Right_Gripper"]
    
    cameras = config.cam_names

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if config.has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if config.has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }
  
    # images
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if config.use_video else "image",
            "shape": (3, 480, 640),
            "names": ["channels", 
                      "height",
                      "width"]
        }
  
    # 创建一个空的LeRobotDataset对象
    return LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=config.fps,
        robot_type=config.robot_type,
        features=features,
        use_videos=config.use_video,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads * len(cameras),
    )
  
def h5_to_lerobot(config: CovertConfig, dataset: LeRobotDataset) -> LeRobotDataset:
  h5_dataset = LoadH5Dataset(config)

  for i in tqdm.tqdm(range(len(h5_dataset))):
    episode = h5_dataset.get_item(i)
    
    state = episode["state"]
    action =episode["action"]

    if config.has_velocity:
      velocity = episode["velocity"]
    
    if config.has_effort:
      effort = episode["effort"]

    cameras = episode["cameras"]
    episode_length = episode["episode_length"]
    task = episode["task_description"]
    
    for j in range(episode_length):
      frame = {
        "observation.state": state[j],
        "action": action[j],
      }

      if config.has_velocity:
        frame["observation.velocity"] = velocity[j]

      if config.has_effort:
        frame["observation.effort"] = effort[j]

      frame["task"] = task
      
      for camera, image_arr in cameras.items():
        frame[f"observation.images.{camera}"] = image_arr[j]
        
      dataset.add_frame(frame)
      
    dataset.save_episode()
    
  return dataset
  
def convert_h5_to_lerobot(config: CovertConfig):
  # 如果repo_id存在，删除它
  if (HF_LEROBOT_HOME/config.repo_id).exists():
    print(f"{config.repo_id} already exists, removing...")
    shutil.rmtree(HF_LEROBOT_HOME/config.repo_id)
    
  # 创建LeRobotDataset
  dataset: LeRobotDataset = create_empty_dataset(config)
  
  # 开始转换
  dataset: LeRobotDataset = h5_to_lerobot(config, dataset)

  # 是否上传到HuggingFace
  if config.push_to_hub:
    dataset.push_to_hub(config.repo_id)

if __name__ == "__main__":
  tyro.cli(convert_h5_to_lerobot)
  