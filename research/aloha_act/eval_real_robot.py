"""
eval your tranined act model in real robot arm. 

example:
  python3 eval_real_robot.py \
    --task_name ${task_name} \
    --ckpt_dir output/act_ckpt/act-${task_name}/ \
    --ckpt_name policy_epoch_8000_seed_0.ckpt \
    --policy_class ACT \
    --chunk_size 25 \
    --control_rate 18 \
    --seed 0

"""

import argparse
import os
import torch
import pickle
import time
import rospy
import numpy as np
from einops import rearrange
from collections import deque

from policy import ACTPolicy, CNNMLPPolicy
from utils import set_seed
from task_config import TASK_CONFIGS
from real_robot_env import RealRobotEnv

def make_policy(policy_class, policy_config):
    """
    根据策略类别创建策略对象
    """
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def load_model(args):
  set_seed(args['seed'])

  # 获取任务参数
  task_name = args['task_name']
  task_config = TASK_CONFIGS[task_name]
  camera_names = task_config['camera_names']
  state_dim = task_config["state_dim"]
  
  # policy config
  policy_class = args['policy_class']
  if policy_class == 'ACT':
      policy_config = {
          'num_queries': args['chunk_size'],    # 查询数量,对应动作序列长度
          'kl_weight': args['kl_weight'],       # KL散度权重,用于VAE训练
          'hidden_dim': args['hidden_dim'],     # 隐藏层维度
          'dim_feedforward': args['dim_feedforward'],  # 前馈网络维度
          'backbone': args['backbone'],                 # 主干网络类型
          'enc_layers': args['enc_layers'],             # 编码器层数
          'dec_layers': args['dec_layers'],             # 解码器层数
          'nheads': args['nheads'],                     # 注意力头数
          'camera_names': camera_names,         # 使用的相机名称列表
          'state_dim': state_dim,
      }
  elif policy_class == 'CNNMLP':
      policy_config = {'lr': args['lr'], 'backbone' : args['backbone'], 'num_queries': 1,
                        'camera_names': camera_names,}
  else:
      raise NotImplementedError

  # create policy
  policy = make_policy(policy_class, policy_config)
  # load policy model parameters
  ckpt_dir = os.path.abspath(args['ckpt_dir']) # 取绝对路径
  ckpt_path = os.path.join(ckpt_dir, args['ckpt_name'])
  loading_status = policy.load_state_dict(torch.load(ckpt_path))
  if not loading_status:
    raise ValueError('Failed to load policy model parameters, ckpt path not exist')
  
  policy.cuda()
  policy.eval()
  print(f'Loaded: {ckpt_path}')

  return policy
  
def get_image(image_list: list):
    """
    获取当前时间步的相机图像，并转换为模型输入格式
    """
    curr_images = []
    for image in image_list:
        curr_image = rearrange(image, 'h w c -> c h w')
        curr_images.append(curr_image)
    # 将多个相机图像堆叠为单个数组
    curr_image = np.stack(curr_images, axis=0)  # shape: (num_cameras, C, H, W)
    
    # 归一化图像并转换为PyTorch张量，然后移至GPU
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda()
    
    # 添加批次维度
    curr_image = curr_image.unsqueeze(0)  # shape: (1, num_cameras, C, H, W)
    
    return curr_image
  
def model_inference(args, policy, env: RealRobotEnv):
  ckpt_dir = os.path.abspath(args['ckpt_dir']) # 取绝对路径
  stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
  with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

  # 输入数据预处理
  pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
  # 将模型输出的动作转换回原始尺度
  post_process = lambda a: a * stats['action_std'] + stats['action_mean']

  # 设置查询频率
  query_frequency = args['chunk_size']
  temporal_agg = args['temporal_agg']
  task_name = args['task_name']
  state_dim = TASK_CONFIGS[task_name]["state_dim"]
  
  if temporal_agg:
      # 如果使用时间聚合，则每步查询一次
      query_frequency = 1
      # 保存原始的查询数量
      num_queries = args['chunk_size']

      # 使用双端队列来存储历史预测的动作序列
      # maxlen 参数自动管理队列大小，当队列满时，最老的元素会被自动移除
      action_queue = deque(maxlen=num_queries) # maxlen 设置为预测序列的长度
      
  input("Press key [enter] to start model inference: ")
  # TODO: robot go to dataset init position
  print("wait robot to init pose")
#   init_position = [0, 0, 0, 0, 0, 0, 0]
#   env.step(init_position)
  time.sleep(3)
  
  t = 0
  rate = rospy.Rate(args['control_rate'])
  with torch.inference_mode():
    while not rospy.is_shutdown():
        start_time = time.time()
        observation = env.get_observation()
        
        # wait input data
        if observation is None:
            rate.sleep()
            continue
        
        ### process qpos and image_list
        qpos = pre_process(observation['state'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        
        curr_image = get_image(observation["images"]) # shape: (1, num_cameras, C, H, W)
        print(f"pre_process_time : {(time.time() - start_time) * 1000}")

        ### query policy
        if args['policy_class'] == "ACT":
            # query_frequency间隔推理一次输出 动作快
            if t % query_frequency == 0:
                time1 = time.time()
                all_actions = policy(qpos, curr_image)

                print(f"policy time: {(time.time() - time1) * 1000}")
                # 将本次推理得到的整个动作序列（或其张量）加入队列
                # 这里可以选择只存 CPU 张量以节省 GPU 内存，使用时再 .cuda()
                if temporal_agg:
                    action_queue.append(all_actions.cpu())
            
            if temporal_agg:
                # 从队列中收集所有历史预测
                # 队列中的每个元素都是一个完整的预测序列 (num_queries, action_dim)
                actions_for_curr_step = []
                # 计算每个历史预测在当前时间步 t 应该贡献哪个动作
                # 例如，10步前的预测，其第10个动作对应当前t
                for i, past_actions in enumerate(action_queue):
                    past_actions = past_actions.cuda() # 移动到 GPU
                    time_offset = len(action_queue) - 1 - i # 距离现在的时间步数
                    if time_offset < past_actions.shape[1]: # 确保索引有效
                        actions_for_curr_step.append(past_actions[0, time_offset]) # 取 batch 0, 对应时间步的动作
                
                if actions_for_curr_step:
                    actions_for_curr_step = torch.stack(actions_for_curr_step) # shape: (num_history, action_dim)
                    
                    # 计算指数衰减权重 (权重数量等于队列中的历史预测数量)
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # 队列为空的边缘情况，通常不会发生
                    print("action_queue is empty")
                    rate.sleep()
                    continue

            else:
                # 如果不使用时间聚合，直接选择当前时间步对应的动作
                raw_action = all_actions[:, t % query_frequency]
        elif args['policy_class'] == "CNNMLP":
            raw_action = policy(qpos, curr_image)
        else:
            raise NotImplementedError

        ### post-process actions
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)
        target_qpos = action

        # 计算耗时
        inference_time = (time.time() - start_time) * 1000
        # print(f"inference time: {inference_time} ms")
        if inference_time > 50 :
            print(f"delay inference time: {inference_time} ms")
        
        rate.sleep()
        ### step the environment
        env.step(target_qpos)
        t += 1

def init_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
  parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
  parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
  parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
  parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
  parser.add_argument('--control_rate', action='store', type=int, help='publish_rate',
                        default=50, required=False)
  
  # for ACT policy
  # VAE中的KL权重
  parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
  # 动作块的长度
  parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=50, required=False)
  # transformer的encoder输入维度
  parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
  # transformer 前馈网络FFN的维度
  parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
  parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
  parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
  parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
  parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
  # 时序聚合 
  parser.add_argument('--temporal_agg', action='store_true')
  
  args = vars(parser.parse_args())
  print(f"args: {args}")

  return args
  
if __name__ == '__main__':
  # 1. init arguments
  args = init_arguments()
  # 2. load trained model
  policy = load_model(args)
  # 3. make real robot environment
  env = RealRobotEnv(args)
  # 4. inference
  model_inference(args, policy, env)