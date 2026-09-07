''''
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/common/robot_devices/control_utils.py
'''

import logging
import torch
import time
from torch import nn
from pprint import pformat
from dataclasses import asdict
from typing import Any
from lerobot.utils.utils import init_logging, get_safe_torch_device
from lerobot.policies.factory import make_policy, make_pre_post_processors

from eval_config import EvalConfig, predict_action
from utils.rerun_visualizer import RerunLogger, visualization_data
from real_robot_env import RealRobotEnv
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
import rospy
from contextlib import nullcontext

def eval_policy(
    cfg: EvalConfig,
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Get initial pose from the first step of the dataset
    from_idx = dataset.meta.episodes["dataset_from_index"][cfg.episode_index]
    step = dataset[from_idx]
    init_pose = step["observation.state"][:].cpu().numpy()
    
    # ros robot env
    if init_pose.shape[-1] >=14:
        # dual arm
        single_arm = False
    elif init_pose.shape[-1] >= 7:
        # single arm
        single_arm = True
    else :
        raise ValueError(f"Invalid state shape {init_pose.shape[-1]}")
    
    cam_names = []
    for key, _ in step.items():
        if key.startswith("observation.images."):
            cam_name = key[len("observation.images."):]
            cam_names.append(cam_name)
    
    env = RealRobotEnv(single_arm, cam_names)
    time.sleep(1)

    #===============init robot=====================
    input("Press key [enter] to control robot to init pose!")

    # "The initial positions of the robot take the initial positions during data recording."
    print("wait robot to init pose!")
    env.step(init_pose)
    time.sleep(3)

    index = 0
    while True and not rospy.is_shutdown():
        loop_start_time = time.perf_counter()

        # get observation
        observation = env.get_observation()
        if observation is None:
            time.sleep(max(0, (1.0 / cfg.control_rate) - (time.perf_counter() - loop_start_time)))
            continue

        # predict action
        action = predict_action(
            observation, 
            policy, 
            get_safe_torch_device(policy.config.device),
            preprocessor,
            postprocessor,
            policy.config.use_amp,
            step["task"],
            use_dataset=False
        )
        action = action.cpu().numpy()
        
        time.sleep(max(0, (1.0 / cfg.control_rate) - (time.perf_counter() - loop_start_time)))
        # execute action
        env.step(action)

        if cfg.visualization:
            visualization_data(index, observation, observation["observation.state"], action, rerun_logger)
            index += 1
        
        # time.sleep(max(0, (1.0 / cfg.control_rate) - (time.perf_counter() - loop_start_time)))

@parser.wrap()
def eval_main(cfg: EvalConfig):
    logging.info(pformat(asdict(cfg)))
    
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making Dataset.")
    dataset = LeRobotDataset(repo_id = cfg.repo_id)
    
    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, policy, dataset, preprocessor, postprocessor)
    
    logging.info("End of eval")


if __name__ == "__main__":
  init_logging()
  eval_main()