# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from detr.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        """
        初始化FrozenBatchNorm2d
        Args:
            n: 特征通道数
        """
        super(FrozenBatchNorm2d, self).__init__()
        # 注册固定的模型参数（不会在训练中更新）
        self.register_buffer("weight", torch.ones(n))  # gamma
        self.register_buffer("bias", torch.zeros(n))   # beta
        self.register_buffer("running_mean", torch.zeros(n))  # 均值
        self.register_buffer("running_var", torch.ones(n))    # 方差

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 移除num_batches_tracked参数，因为在冻结的BN中不需要
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """
        前向传播函数
        执行标准的批归一化计算，但使用固定的统计信息
        """
        # 重塑权重和偏置的维度以便于广播
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        # 计算归一化的scale和bias
        scale = w * (rv + eps).rsqrt()  # 标准差的倒数乘以gamma
        bias = b - rm * scale           # 减去归一化的均值后加上beta
        return x * scale + bias         # 应用归一化


class BackboneBase(nn.Module):
    """
    基础主干网络类，处理特征提取的通用逻辑
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        """
        Args:
            backbone: 主干网络模型（如ResNet）
            train_backbone: 是否训练主干网络
            num_channels: 输出通道数
            return_interm_layers: 是否返回中间层特征
        """
        super().__init__()
        # 注释掉的代码用于选择性地训练后面的层
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)

        # 根据是否需要中间层特征来设置返回层
        if return_interm_layers:
            # 返回所有中间层的特征图
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            # 只返回最后一层的特征图
            return_layers = {'layer4': "0"}
        # 使用IntermediateLayerGetter来获取指定层的输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        """前向传播，返回特征图"""
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """使用冻结BatchNorm的ResNet主干网络"""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        """
        Args:
            name: 主干网络名称（如'resnet50')
            train_backbone: 是否训练主干网络
            return_interm_layers: 是否返回中间层特征
            dilation: 是否使用空洞卷积
        """
        # 从torchvision.models中获取预训练的ResNet模型
        backbone = (getattr(torchvision.models, name))(
            # 设置空洞卷积的使用情况
            replace_stride_with_dilation=[False, False, dilation],
            # 是否使用预训练模型，使用冻结的BN层
            pretrained=is_main_process(), 
            norm_layer=FrozenBatchNorm2d)
        
        # 根据ResNet类型设置输出通道数
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """
    将主干网络和位置编码组合在一起
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """
        前向传播函数
        Args:
            tensor_list: 输入的特征图列表
        Returns:
            out: 输出的特征图列表
            pos: 输出的位置编码列表
        """
        # tensor_list: [B, 3, H, W]
        # xs: 字典，键是层名称，值是特征图, 
        # 例如：{"0": [B, num_channels, H/32, W/32]}
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        # out: List[[B, num_channels, H/32, W/32]]
        # pos: List[Tensor] = [[B, hidden_dim, H/32, W/32]]
        return out, pos


def build_backbone(args):
    """
    构建主干网络
    Args:
        args: 模型配置参数
    Returns:
        model: 主干网络模型
    """
    # 构建位置编码模块
    position_embedding = build_position_encoding(args)
    # 判断是否训练backbone，如果学习率大于0则训练
    train_backbone = args.lr_backbone > 0
    # 是否返回中间层特征
    return_interm_layers = args.masks
    # 构建主干网络（通常是ResNet）
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 将backbone和位置编码组合在一起
    model = Joiner(backbone, position_embedding)
    # 设置输出通道数
    model.num_channels = backbone.num_channels
    return model
