""""
  Save datasets to hdf5 format file.
"""
import os
import h5py
import time
import numpy as np
from utils.data_collection_cfg import DataCollectionCfg

class H5Dataset:
  def __init__(self, cfg: DataCollectionCfg) -> None:
    self.cfg = cfg
    self.hdf5_cfg = cfg.hdf5_cfg
    
    # dataset dir
    dataset_dir = os.path.abspath(self.hdf5_cfg.dataset_dir)
    self.dataset_dir = os.path.join(dataset_dir, self.hdf5_cfg.task_name)
    
    if self.hdf5_cfg.start_episode is None:
      self.episode_idx = self.get_auto_index()
    else:
      self.episode_idx = self.get_auto_index(self.hdf5_cfg.start_episode)
    print("HDF5: start episode index is ", self.episode_idx)
    
  def get_auto_index(self, start_episode=0, dataset_name_prefix = '', data_suffix = 'hdf5'):
    """
      获取数据集目录中下一个可用的片段索引
      
      参数：
          dataset_dir: 数据集目录
          dataset_name_prefix: 数据集名称前缀
          data_suffix: 数据文件扩展名
          
      返回：
          int: 下一个可用的片段索引
    """
    # 设置最大索引值为1000，防止无限循环
    max_idx = 1000
    
    # 如果数据集目录不存在，创建该目录
    if not os.path.isdir(self.dataset_dir):
        os.makedirs(self.dataset_dir)
    
    # 从0到1000遍历可能的索引值
    for i in range(max_idx+1):
        # 构造可能的文件名，例如：'episode_0.hdf5', 'episode_1.hdf5' 等
        # 检查该文件是否已经存在
        if not os.path.isfile(os.path.join(self.dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            # 如果文件不存在，返回这个索引值
            return i
    
    # 如果遍历完1000个索引都没找到可用的，抛出异常
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")
  
  def save_to_hdf5(self, timesteps, actions):
    # extract timesteps and actions to data dict
    dataset_path = os.path.join(self.dataset_dir, f'episode_{self.episode_idx}.hdf5')
    ts_copy = timesteps.copy()
    action_copy = actions.copy()
    
    len_episode = len(action_copy)
    if (len_episode == 0): 
      print("H5 not save, this episode data is empty!")
      return
    
    print(f"episode length: {len_episode}. Start save to hdf5 format file!")
    data_dict = {
      "/observation/state": [],
      "/action": [],
    }
    
    if self.cfg.robotic_arm_cfg.use_joint_effort:
      data_dict["/observation/effort"] = []
    
    if self.cfg.robotic_arm_cfg.use_joint_velocity:
      data_dict["/observation/velocity"] = []
    
    for cam_name in self.cfg.camera_cfg.camera_names:
      data_dict[f"/observation/images/{cam_name}"] = []
    
    while (action_copy):
      action = action_copy.pop(0)
      ts = ts_copy.pop(0)
      data_dict["/observation/state"].append(ts.observation["state"])
      
      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        data_dict["/observation/velocity"].append(ts.observation["velocity"])
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        data_dict["/observation/effort"].append(ts.observation["effort"])
      
      data_dict["/action"].append(action)
      
      for cam_name in self.cfg.camera_cfg.camera_names:
        data_dict[f"/observation/images/{cam_name}"].append(np.frombuffer(ts.observation["cameras"][cam_name], dtype='uint8'))
        
    # save in hdf5 format
    start_time = time.time()
    with h5py.File(dataset_path, "w") as root:
      # root.attrs["sim"] = False
      root.attrs["task"] = self.cfg.task_description
      
      # save observation group
      obs = root.create_group("observation")
      obs.create_dataset("state", data=np.array(data_dict["/observation/state"], dtype=np.float32))
      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        obs.create_dataset("velocity", data=np.array(data_dict["/observation/velocity"], dtype=np.float32))
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        obs.create_dataset("effort", data=np.array(data_dict["/observation/effort"], dtype=np.float32))
      
      # save images group
      images = obs.create_group("images")
      vlen_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
      for cam_name in self.cfg.camera_cfg.camera_names:
        images.create_dataset(cam_name, 
                              data=data_dict[f"/observation/images/{cam_name}"], 
                              dtype=vlen_dtype)
      
      # save action data
      root.create_dataset("action", data=np.array(data_dict["/action"], dtype=np.float32))
    
    self.episode_idx += 1  
    print(f"Save episode to {dataset_path}.")
    print(f"Save time: {time.time() - start_time:.3f} secs.")