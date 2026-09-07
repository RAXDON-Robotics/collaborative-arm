"""
策略模型模块

该模块实现了两种策略模型：
1. ACT (Action Chunking with Transformers): 基于Transformer的动作分块策略
2. CNNMLP: 基于CNN+MLP的基线策略
"""

import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    """基于Transformer的动作分块策略模型
    
    使用条件变分自编码器(CVAE)和Transformer来学习动作序列
    """
    def __init__(self, args_override):
        """初始化ACT策略模型
        
        Args:
            args_override: 模型配置参数
        """
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # DETRVAE or CNNMLP
        self.optimizer = optimizer  # Adam优化器
        self.kl_weight = args_override['kl_weight']  # KL散度权重
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """
        Args:
            qpos: 机器人关节位置, shape=(batch_size, state_dim)
            image: 相机图像, shape=(batch_size, num_cameras, C, H, W)
            actions: 目标动作序列（训练时）, shape=(batch_size, max_action_len, state_dim)
            is_pad: 填充标记（训练时）, shape=(batch_size, max_action_len)
            
        Returns:
            训练时: 包含各种损失的字典
            推理时: 预测的动作序列
        """
        env_state = None
        # 图像归一化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            # 只使用前num_queries个动作
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # forward
            # a_hat: 预测的动作序列, shape=(batch_size, num_queries, action_dim)
            # is_pad_hat: 预测的填充标记, shape=(batch_size, num_queries, 1)
            # (mu, logvar): VAE的均值和方差
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # 1. 计算预测动作和真实动作的L1损失
            # reduction='none'是指不进行reduce操作
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            # 2. 只计算未填充的部分的L1损失
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            # 3. 将L1损失和KL散度loss组合成总loss
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            # shape: (batch_size, num_queries, action_dim)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """基于CNN+MLP的基线策略模型"""
    def __init__(self, args_override):
        """初始化CNNMLP策略模型
        
        Args:
            args_override: 模型配置参数
        """
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """前向传播
        
        Args:
            qpos: 机器人关节位置, shape=(batch_size, 14)
            image: 相机图像, shape=(batch_size, C, H, W)
            actions: 目标动作（训练时）, shape=(batch_size, 14)
            is_pad: 未使用
            
        Returns:
            训练时: 包含MSE损失的字典
            推理时: 预测的动作
        """
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    """计算VAE中的KL散度
    Args:
        mu: 隐变量的均值, shape=[batch_size, latent_dim]
        logvar: 隐变量的对数方差, shape=[batch_size, latent_dim]
    
    Returns:
        tuple: (总KL散度, 维度wise的KL散度, 平均KL散度)
    """
    # 1. 确保输入维度正确
    batch_size = mu.size(0)
    assert batch_size != 0
    # 2. 处理4D输入（如果有）
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1)) # 展平到2D
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1)) # 展平到2D

    # 3. 计算KL散度
    # KL(N(mu,sigma)||N(0,1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # 4. 计算三种KL散度统计量
    total_kld = klds.sum(1).mean(0, True) # 总KL散度, shape=[1]
    dimension_wise_kld = klds.mean(0)     # 每个维度的平均KL散度, shape=[latent_dim]
    mean_kld = klds.mean(1).mean(0, True) # 平均KL散度, shape=[1]

    return total_kld, dimension_wise_kld, mean_kld
