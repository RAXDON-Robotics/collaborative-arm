# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed


def get_args_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # 优化器参数
    parser.add_argument('--lr', default=1e-4, type=float)                 # 主网络学习率
    parser.add_argument('--lr_backbone', default=1e-5, type=float)        # 骨干网络学习率
    parser.add_argument('--batch_size', default=2, type=int)              # 批次大小
    parser.add_argument('--weight_decay', default=1e-4, type=float)       # L2正则化系数
    parser.add_argument('--epochs', default=300, type=int)                # 训练轮数
    parser.add_argument('--lr_drop', default=200, type=int)               # 学习率衰减轮数
    parser.add_argument('--clip_max_norm', default=0.1, type=float,       # 梯度裁剪阈值
                        help='gradient clipping max norm')

    # 模型参数
    # * 骨干网络参数
    parser.add_argument('--backbone', default='resnet18', type=str,       # CNN骨干网络类型
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',                # 是否使用空洞卷积
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str,  # 位置编码类型：正弦或可学习
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list,          # 相机名称列表
                        help="A list of camera names")

    # * Transformer参数
    parser.add_argument('--enc_layers', default=4, type=int,              # Transformer编码器层数
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,              # Transformer解码器层数
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,      # 前馈网络隐藏层维度
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,            # Transformer隐藏层维度
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,             # Dropout比率
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,                  # 注意力头数量
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int,           # 查询向量数量
                        help="Number of query slots")                      # 对应动作序列长度
    parser.add_argument('--pre_norm', action='store_true')                # 是否在注意力之前进行层归一化

    # * 分割相关参数
    parser.add_argument('--masks', action='store_true',                   # 是否训练分割头
                        help="Train segmentation head if the flag is provided")

    # 以下参数来自imitate_episodes.py，仅用于避免报错，实际未使用
    parser.add_argument('--eval', action='store_true')                    # 是否为评估模式
    parser.add_argument('--onscreen_render', action='store_true')         # 是否在屏幕上渲染
    parser.add_argument('--ckpt_dir', action='store', type=str,           # 检查点保存目录
                        help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str,       # 策略类型（ACT/CNNMLP）
                        help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str,          # 任务名称
                        help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int,               # 随机种子
                        help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int,         # 训练轮数
                        help='num_epochs', required=False)
    parser.add_argument('--kl_weight', action='store', type=int,          # KL散度权重
                        help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int,         # 动作序列分块大小
                        help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')            # 是否使用时序聚合
    parser.add_argument('--state_dim', default=14, type=int,             
                        help="observation state dimmension")
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--control_rate', action='store', type=int, help='publish_rate',
                        default=50, required=False)

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args) # DETRVAE or CNNMLP
    model.cuda()  # 将模型移至GPU
    
    # 创建两个参数组：非backbone参数和backbone参数
    param_dicts = [
        # 组1：非backbone参数
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        # 组2：backbone参数（使用较小的学习率, 因为是预训练模型）
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # 使用AdamW优化器，设置学习率和权重衰减
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer
