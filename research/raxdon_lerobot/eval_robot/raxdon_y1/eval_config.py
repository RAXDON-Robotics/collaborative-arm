"""
  The eval configure for dataset or real robot eval pipeline.
  
  Reference:
    lerobot/scripts/eval.py
"""

import logging
from dataclasses import dataclass, field
import torch
import numpy as np
from typing import Any
from copy import copy
from contextlib import nullcontext
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

@dataclass
class EvalConfig:
  repo_id: str
  policy: PreTrainedConfig | None = None
  episode_index: int = 0
  control_rate: int = 30  # hz
  visualization: bool = False
  rename_map: dict[str, str] = field(default_factory=dict)
  
  def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )


  @classmethod
  def __get_path_fields__(cls) -> list[str]:
      """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
      return ["policy"]

# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
):

    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "images" in name and not use_dataset:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
                # print(f"observation {name} : ", observation[name].type(torch.float32))
            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""

        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        print("select_action shape: ", action.shape)
        action = postprocessor(action)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action