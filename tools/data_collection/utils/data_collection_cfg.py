from dataclasses import dataclass, field, MISSING
from omegaconf import OmegaConf
from typing import Optional, List
from pathlib import Path

@dataclass
class Hdf5Cfg:
  dataset_dir: Optional[str] = None
  """Directory to save the dataset."""
  
  task_name: Optional[str] = None
  """Task name."""
  
  start_episode: Optional[int] = None

@dataclass
class RoboticArmCfg:
  """Configuration for robotic arm."""
  collection_type: Optional[str] = None
  """data collection type, can be 'one_master' or 'two_master' 
      or 'one_master_slave' or 'two_master_slave' """
  
  master_arm_right_topic: Optional[str] = None
  """Right master arm joint states topic."""
  
  puppet_arm_right_topic: Optional[str] = None
  """Right puppet arm joint states topic."""
  
  enable_left_arm: bool = False
  """Whether to enable left arm."""
  
  master_arm_left_topic: Optional[str] = None
  """Left master arm joint states topic."""
  
  puppet_arm_left_topic: Optional[str] = None
  """Left puppet arm joint states topic."""
  
  use_joint_velocity: bool = False
  """Whether to save joint velocity."""
  
  use_joint_effort: bool = False
  """Whether to save joint effort."""

@dataclass
class CameraCfg:
  """Configuration for camera."""
  
  camera_names: List[str] = field(default_factory=list)
  """The camera name in saved data"""
  
  img_right_topic: Optional[str] = None
  """right camera rgb image topic"""
  
  img_left_topic: Optional[str] = None
  """left camera rgb image topic"""
  
  img_high_topic: Optional[str] = None
  """high camera rgb image topic"""
  
  img_low_topic: Optional[str] = None
  """low camera rgb image topic"""
  
  depth_right_topic: Optional[str] = None
  """right camera depth image topic"""
  
  depth_left_topic: Optional[str] = None
  """left camera depth image topic"""
  
  depth_high_topic: Optional[str] = None
  """high camera depth image topic"""
  
  depth_low_topic: Optional[str] = None
  """low camera depth image topic"""

@dataclass
class DataCollectionCfg:
  """Configuration for robotic arm data collection."""

  task_description: str = MISSING
  
  num_episodes: int = 1
  """The number of collect expert demonstrations.default is 1."""
  
  save_rate: int = 30
  """The frequency of save data. default is 30."""

  skip_static_data: bool = True
  """Whether to skip static data."""

  
  hdf5_cfg: Hdf5Cfg = Hdf5Cfg()
  """Configuration for save data as hdf5."""
  
  robotic_arm_cfg: RoboticArmCfg = RoboticArmCfg()
  """Configuration for robotic arm."""
  
  camera_cfg: CameraCfg = CameraCfg()
  """Configuration for camera."""

def build_config():
  """Parse configuration from command line arguments and YAML file."""
  # Step 1: Load base config from dataclass
  base_cfg = OmegaConf.structured(DataCollectionCfg)
  
  # Step 2: Load config from YAML file
  cli_cfg = OmegaConf.from_cli()
  if "config_file" in cli_cfg:
    yaml_cfg = OmegaConf.load(cli_cfg["config_file"])
    # print("yaml_cfg: ", yaml_cfg)
    base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
  else:
    raise ValueError("arg config_file=your_path must provided.")
  
  # Step 3: Merge CLI args
  if "config_file" in cli_cfg:
    del cli_cfg["config_file"]

  # print("cli_cfg: ", cli_cfg)
  cfg = OmegaConf.merge(base_cfg, cli_cfg)
  
  return cfg