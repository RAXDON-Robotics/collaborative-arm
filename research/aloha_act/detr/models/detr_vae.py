"""
DETR模型和损失函数类的实现。
这个文件实现了带有VAE(变分自编码器)的DETR模型,用于机器人操作任务。
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    """VAE的重参数化技巧
    将均值和对数方差转换为隐变量采样
    Args:
        mu: 均值
        logvar: 对数方差
    Returns:
        采样得到的隐变量
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """生成正弦位置编码表, 帮助Transformer模型理解序列中的位置关系
    Args:
        n_position: 位置数量
        d_hid: 隐藏层维度
    Returns:
        位置编码表
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # 为每个位置生成编码向量
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # 偶数维度使用sin
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # 奇数维度使用cos
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # 增加一个维度： [1, n_position, d_hid]
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """带有VAE的DETR模型, 用于物体检测和机器人动作生成"""
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """初始化模型
        Args:
            backbones: 主干网络模块, 用于特征提取
            transformer: transformer架构模块
            encoder: 编码器模块 
            state_dim: 机器人状态维度
            num_queries: 查询数量, 对应动作序列长度
            camera_names: 相机名称列表
        """
        super().__init__()
        self.num_queries = num_queries
        
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)  # 动作预测头
        self.is_pad_head = nn.Linear(hidden_dim, 1)  # padding预测头
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 查询嵌入
        
        if backbones is not None:
            # 使用视觉backbone的情况
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1) # 图像特征投影
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)  # 关节位置投影
        else:
            # 只使用状态信息的情况
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)  # 关节位置投影
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # VAE编码器额外参数
        self.latent_dim = 32  # 隐变量z维度
        self.cls_embed = nn.Embedding(1, hidden_dim)  # CLS token嵌入
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)  # 动作序列投影
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)   # 关节位置投影
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)  # 隐变量投影(均值和方差)
        # 位置编码表：CLS token + qpos + action序列
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

        # VAE解码器额外参数
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # 隐变量输出投影
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # 为latent_input和proprio_input增加额外位置编码

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """前向传播
        Args:
            qpos: 机器人关节位置 [batch_size, state_dim]
            image: 图像输入 [batch_size, num_cam, channel, height, width]
            env_state: 环境状态
            actions: 动作序列 [batch_size, num_queries, state_dim]
            is_pad: padding掩码 [batch_size, num_queries]
        Returns:
            a_hat: 预测的动作
            is_pad_hat: 预测的padding
            [mu, logvar]: VAE的均值和方差
        """
        is_training = actions is not None
        bs, _ = qpos.shape
        
        # 1.VAE
        if is_training:
            # training: 从动作序列获取隐变量z
            # 将动作序列投影到嵌入空间，并与CLS token拼接
            action_embed = self.encoder_action_proj(actions) # (batch_size, num_queries, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos) # (batch_size, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1) # (batch_size, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (batch_size, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (batch_size, 1+1+num_queries, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (1+1+num_queries, batch_size, hidden_dim)
            
            # 处理padding掩码
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # (batch_size, 2)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1) # (batch_size, 2+num_queries)
            
            # 获取位置编码
            # 位置编码是独立生成的, 只依赖于序列长度和隐藏维度
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2) # [1+1+num_queries, 1, hidden_dim] 
            
            # 编码器前向传播
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad) # shape: (1+1+num_queries, batch_size, hidden_dim)
            encoder_output = encoder_output[0]  # 只取CLS token输出, shape: (batch_size, hidden_dim)
            latent_info = self.latent_proj(encoder_output) # shape: (batch_size, latent_dim*2)
            mu = latent_info[:, :self.latent_dim] # 均值
            logvar = latent_info[:, self.latent_dim:] # 对数方差
            latent_sample = reparametrize(mu, logvar) # shape: (batch_size, latent_dim)
            latent_input = self.latent_out_proj(latent_sample) # shape: (batch_size, hidden_dim)
        else:
            # 测试时使用零向量作为隐变量
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # 2.Transformer
        # 使用视觉backbone的情况
        if self.backbones is not None:
            # 处理多相机图像特征
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                # backbones is list, but only use the first one
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]  # 取最后一层特征, shape: [B, num_channels, H/32, W/32]
                pos = pos[0] # 取最后一层位置编码, shape: [B, hidden_dim, H/32, W/32]
                all_cam_features.append(self.input_proj(features)) # input_proj后的shape: [B, hidden_dim, H/32, W/32]
                all_cam_pos.append(pos)
            
            # 关节位置投影, shape: [B, hidden_dim]
            proprio_input = self.input_proj_robot_state(qpos)
            
            # 合并所有相机特征
            src = torch.cat(all_cam_features, axis=3) # shape: [B, hidden_dim, H/32, W/32*num_cam]
            pos = torch.cat(all_cam_pos, axis=3) # shape: [B, hidden_dim, H/32, W/32*num_cam]
            
            # transformer前向传播
            # shape: [B, num_queries, hidden_dim]
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, 
                                proprio_input, self.additional_pos_embed.weight)[0]
        # 只使用状态信息的情况
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        
        # 预测动作和padding
        a_hat = self.action_head(hs) # shape: [B, num_queries, state_dim]
        is_pad_hat = self.is_pad_head(hs) # shape: [B, num_queries, 1]
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    """CNN+MLP基线模型"""
    def __init__(self, backbones, state_dim, camera_names):
        """初始化模型
        Args:
            backbones: 主干网络列表
            state_dim: 状态维度
            camera_names: 相机名称列表
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)
        
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """前向传播
        Args:
            qpos: 机器人关节位置
            image: 图像输入
            env_state: 环境状态
            actions: 动作序列
        Returns:
            预测的动作
        """
        is_training = actions is not None
        bs, _ = qpos.shape
        
        # 处理多相机图像特征
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
            
        # 展平特征
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)
        
        # 合并特征和机器人状态
        features = torch.cat([flattened_features, qpos], axis=1)
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    """构建MLP网络
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        hidden_depth: 隐藏层数量
    Returns:
        MLP网络
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    """构建编码器
    Args:
        args: 配置参数
    Returns:
        编码器模型
    """
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    """构建完整的DETR-VAE模型
    Args:
        args: 配置参数
    Returns:
        DETR-VAE模型
    """
    state_dim = args.state_dim # TODO hardcode
    
    # From image
    backbones = []
    # image backbone(resnet) + position encoding
    backbone = build_backbone(args)
    backbones.append(backbone)

    # transformer
    transformer = build_transformer(args)

    # encoder用于VAE的编码过程
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    # 计算模型中可训练参数的总数
    # sum(): 对迭代器中的所有元素求和
    # p.numel(): 返回张量p中的元素总数
    # model.parameters(): 返回模型中所有参数的迭代器
    # if p.requires_grad: 只计算需要梯度的参数（即可训练的参数）
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    """构建CNN-MLP基线模型
    Args:
        args: 配置参数
    Returns:
        CNN-MLP模型
    """
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
