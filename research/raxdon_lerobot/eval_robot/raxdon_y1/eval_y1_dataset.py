''''
examples:
    python eval_robot/raxdon_y1/eval_y1_dataset.py \
        --policy.path=outputs/train/2025-12-07/20-24-59_diffusion/checkpoints/last/pretrained_model/ \
        --repo_id=v30/folded_orange_towel_1122 \
        --episode_index=2
'''

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from eval_config import EvalConfig, predict_action
# from eval_robot.raxdon_y1.utils.rerun_visualizer import RerunLogger, visualization_data
from utils.rerun_visualizer import RerunLogger, visualization_data

def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            # print("value.shape: ", value.shape)
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation

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

    # dataset.meta.episodes["dataset_from_index"][episode_index]
    from_idx = dataset.meta.episodes["dataset_from_index"][cfg.episode_index]
    step = dataset[from_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][cfg.episode_index]

    ground_truth_actions = []
    predicted_actions = []
        
    #===============init robot=====================
    input("Press key [enter] to start eval dataset: ")

    index = 0
    for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
        loop_start_time = time.perf_counter()

        step = dataset[step_idx]
        observation = extract_observation(step)

        action = predict_action(
            observation, 
            policy, 
            get_safe_torch_device(policy.config.device),
            preprocessor,
            postprocessor,
            policy.config.use_amp,
            step["task"],
            use_dataset=True
        )
        action = action.cpu().numpy()
        print(f"Iteration {step_idx - from_idx} predict_action speed: {(time.perf_counter() - loop_start_time) * 1000} ms")

        ground_truth_actions.append(step["action"].numpy())
        predicted_actions.append(action)

        if cfg.visualization:
            visualization_data(index, observation, observation["observation.state"], action, rerun_logger)
            index += 1

        time.sleep(0.01)
    
    ground_truth_actions = np.array(ground_truth_actions)
    predicted_actions = np.array(predicted_actions)

    # Get the number of timesteps and action dimensions
    _, n_dims = ground_truth_actions.shape

    # Create a figure with subplots for each action dimension
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims), sharex=True)
    fig.suptitle('Ground Truth vs Predicted Actions')

    # Plot each dimension
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes

        ax.plot(ground_truth_actions[:, i], label='Ground Truth', color='blue')
        ax.plot(predicted_actions[:, i], label='Predicted', color='red', linestyle='--')
        ax.set_ylabel(f'Dim {i+1}')
        ax.legend()

    # Set common x-label
    axes[-1].set_xlabel('Timestep')

    plt.tight_layout()
    # plt.show()

    time.sleep(1)
    plt.savefig('dp.png')

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
