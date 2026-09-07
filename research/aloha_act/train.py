#!/usr/bin/env python3
"""
train your act model.

example:
  python3 train.py \
    --task_name ${task_name} \
    --ckpt_dir ./output/act_ckpt/act-${task_name}/ \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --seed 0

"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import load_data # 数据加载函数
from utils import compute_dict_mean, set_seed, detach_dict # 辅助函数
from policy import ACTPolicy, CNNMLPPolicy
from task_config import TASK_CONFIGS

import IPython
e = IPython.embed

def main(args):
    """
    主函数，用于解析命令行参数和执行训练或评估
    """
    print("args : ", args)
    # 设置随机种子，确保结果可复现
    set_seed(1)
    # 解析命令行参数
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # 获取任务参数
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']
    state_dim = task_config["state_dim"]
    if policy_class == 'ACT':
        # 策略配置参数
        policy_config = {
            'lr': args['lr'],                     # 学习率
            'num_queries': args['chunk_size'],    # 查询数量,对应动作序列长度
            'kl_weight': args['kl_weight'],       # KL散度权重,用于VAE训练
            'hidden_dim': args['hidden_dim'],     # 隐藏层维度
            'dim_feedforward': args['dim_feedforward'],  # 前馈网络维度
            'lr_backbone': args['lr_backbone'],           # 主干网络学习率
            'backbone': args['backbone'],                 # 主干网络类型
            'enc_layers': args['enc_layers'],             # 编码器层数
            'dec_layers': args['dec_layers'],             # 解码器层数
            'nheads': args['nheads'],                     # 注意力头数
            'camera_names': camera_names,         # 使用的相机名称列表
            'state_dim': state_dim,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': args['lr_backbone'], 'backbone' : args['backbone'], 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
    }

    # Trainning
    # 1. create dataloader
    train_dataloader, val_dataloader, stats = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # 2. save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 3.train model
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # 4.save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


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


def make_optimizer(policy_class, policy):
    """
    根据策略类别创建优化器
    """
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    """
    获取当前时间步的相机图像，并转换为模型输入格式
    """
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    # 将多个相机图像堆叠为单个数组
    curr_image = np.stack(curr_images, axis=0)  # shape: (num_cameras, C, H, W)
    
    # 归一化图像并转换为PyTorch张量，然后移至GPU
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda()
    
    # 添加批次维度
    curr_image = curr_image.unsqueeze(0)  # shape: (1, num_cameras, C, H, W)
    
    return curr_image

def forward_pass(data, policy):
    """
        Args:  
            data: ['image_data', 'qpos_data', 'action_data', 'is_pad']
                image_data shape: (batch_size, C, H, W)
                qpos_data shape: (batch_size, state_dim)
                action_data shape: (batch_size, max_action_len, state_dim)
                is_pad shape: (batch_size, max_action_len)
    
        Returns:
            loss: scalar
    """
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # policy: ACTPolicy or CNNMLPPolicy
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    """train behavior clone policy."""
    
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    # create policy
    policy = make_policy(policy_class, policy_config) # ACTPolicy or CNNMLPPolicy
    policy.cuda()
    # create optimizer
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    # 训练num_epochs个epoch, 每个epoch包括训练和验证
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # data: (image_data, qpos_data, action_data, is_pad)
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy) # forward_dict: {'loss': loss, 'action': action}
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            # 更新最小验证损失
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        # 计算平均损失
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # 100个epoch保存一次
        if epoch % 500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """
    绘制训练和验证曲线
    """
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 保存ckpt的目录
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # 策略类: act or cnnmlp
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    # 任务名称
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # batch_size
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    # seed
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    # 训练轮数
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    # 学习率
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    # backbone的学习率
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=4e-5, required=False)

    # for ACT policy
    # VAE中的KL权重
    parser.add_argument('--kl_weight', action='store', type=int, help='KL 权重', required=False)
    # 动作块的长度
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # transformer的encoder输入维度
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # transformer 前馈网络FFN的维度
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    
    main(vars(parser.parse_args()))