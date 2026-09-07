# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
位置编码的各种实现方式
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from detr.util.misc import NestedTensor

import IPython
e = IPython.embed

class PositionEmbeddingSine(nn.Module):
    """
    这是位置编码的标准版本，与"Attention is all you need"论文中使用的版本非常相似
    并进行了扩展以适用于图像处理
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        参数初始化
        num_pos_feats: 位置特征的维度
        temperature: 用于位置编码的温度参数
        normalize: 是否对位置编码进行归一化
        scale: 缩放因子,仅在normalize为True时使用
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        """
        前向传播函数
        生成正弦位置编码
        tensor: 输入张量
        返回: 位置编码张量
        """
        # 输入tensor维度: [B, num_channels, H/32, W/32]
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        # 创建一个与输入张量形状相同的全1掩码
        not_mask = torch.ones_like(x[0, [0]])
        # 在高度维度上计算累积和，得到y方向的位置编码基础值
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 在宽度维度上计算累积和，得到x方向的位置编码基础值
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            # 添加一个小的epsilon值避免除零错误
            eps = 1e-6
            # 对y方向的位置编码进行归一化，使其范围在[0, scale]之间
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # 对x方向的位置编码进行归一化，使其范围在[0, scale]之间
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成位置编码的频率因子
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 计算不同频率的温度缩放因子，用于生成不同频率的正弦波
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 将x和y方向的位置值除以温度因子，准备进行正弦/余弦编码
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 对x方向进行正弦和余弦编码：偶数位置使用sin，奇数位置使用cos
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 对y方向进行正弦和余弦编码：偶数位置使用sin，奇数位置使用cos
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 将x和y方向的编码拼接并调整维度顺序，得到最终的位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # [B, hidden_dim, H/32, W/32]
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    可学习的绝对位置编码
    通过神经网络学习位置的表示方式
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        """
        初始化函数
        num_pos_feats: 位置特征的维度, 默认为256
        创建行和列的嵌入层, 最大支持50x50的特征图
        """
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置参数函数
        使用均匀分布初始化行列嵌入的权重
        """
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        """
        前向传播函数
        tensor_list: 输入的嵌套张量
        返回: 学习到的位置编码
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    """
    构建位置编码的函数
    args: 配置参数
    返回: 位置编码模块的实例
    
    支持两种类型的位置编码：
    1. sine: 基于正弦函数的位置编码
    2. learned: 可学习的位置编码
    """
    # 将维度平均分配给x和y方向，使得位置编码能够捕获2D图像的空间信息
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
