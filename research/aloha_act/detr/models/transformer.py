# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer类。

复制自torch.nn.Transformer并进行修改：
    * 将位置编码传递给MHattention
    * 删除编码器末尾的额外LN层
    * 解码器返回所有解码层的激活栈
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed


class Transformer(nn.Module):
    """Transformer模型类
    包含编码器和解码器，用于处理输入序列并生成输出
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """初始化Transformer模型
        Args:
            d_model: 模型的特征维度
            nhead: 多头注意力机制的头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络的维度
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在注意力层前进行归一化
            return_intermediate_dec: 是否返回解码器的中间结果
        """
        super().__init__()

        # 初始化编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 初始化解码器
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 重置参数
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """使用Xavier初始化重置模型参数"""
        for p in self.parameters():
            # dim > 1 是因为我们只对权重矩阵进行初始化，而不对偏置向量进行初始化
            # 权重矩阵通常是二维或更高维的，而偏置向量是一维的
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
        """前向传播函数
        Args:
            src: 图像特征图, shape: [B, hidden_dim, H/32, W/32*num_cam]
            mask: 输入掩码, None
            query_embed: 查询嵌入, shape: [num_queries, hidden_dim]
            pos_embed: 位置编码, shape: [B, hidden_dim, H/32, W/32*num_cam]
            latent_input: 隐变量输入, shape: [B, hidden_dim]
            proprio_input: 关节位置, shape: [B, hidden_dim]
            additional_pos_embed: 为 latent_input 和 proprio_input 提供额外的位置编码， shape: [2, hidden_dim]
        Returns:
            hs: 解码器输出
        """
        # 检查输入维度是否包含高度和宽度
        if len(src.shape) == 4: # 包含H和W
            # 将NxCxHxW展平为HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1) # shape: [HW, B, hidden_dim]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1) # shape: [HW, B, hidden_dim]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # shape: [num_queries, B, hidden_dim]

            # 处理额外位置嵌入
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # shape: [2, B, hidden_dim]
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0) # shape: [HW+2, B, hidden_dim]

            # 组合隐变量和本体感知输入
            addition_input = torch.stack([latent_input, proprio_input], axis=0) # shape: [2, B, hidden_dim]
            src = torch.cat([addition_input, src], axis=0) # shape: [HW+2, B, hidden_dim]
        else:
            assert len(src.shape) == 3
            # 将NxHWxC展平为HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed) # shape: [num_queries, B, hidden_dim]
        # 编码器前向传播
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # shape: [HW+2, B, hidden_dim]
        # 解码器前向传播
        # shape: [num_layers / 1, num_queries, B, hidden_dim], 
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2) # shape: [num_layers/1, B, num_queries, hidden_dim]
        
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 创建多层编码器
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """TransformerEncoder的前向传播函数
    
        Args:
            src: 输入序列
                shape: (seq_len, batch_size, hidden_dim)
                例如: (102, 8, 512) - 102个token, 8个批次, 512维特征
                
            mask: 注意力掩码，用于控制每个位置能看到哪些其他位置
                shape: (seq_len, seq_len)
                可选参数, 默认为None, 表示所有位置都可以相互关注
                
            src_key_padding_mask: 填充掩码，标记哪些位置是填充的
                shape: (batch_size, seq_len)
                例如: (8, 102) - 8个批次, 每个序列102个位置的填充标记
                True表示该位置是填充的, 应该被忽略
                
            pos: 位置编码
                shape: (seq_len, batch_size, hidden_dim)
                例如: (102, 8, 512)
                为每个token添加位置信息
        
        Returns:
            output: 经过Transformer编码器处理后的序列
                shape: 与输入src相同 (seq_len, batch_size, hidden_dim)
                例如: (102, 8, 512)
                每个token位置都包含了与其他token交互后的信息
        """
        output = src

        # 依次通过每个编码器层
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        # 对输出进行归一化
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 创建多层解码器
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """TransformerDecoder的前向传播函数
        Args:
            tgt: 目标序列，初始为全零张量, shape: [num_queries, batch_size, hidden_dim]
            memory: 编码器输出, shape: [HW+2, batch_size, hidden_dim]
            tgt_mask: 目标序列的注意力掩码, 用于控制自注意力的可见范围, 这里为None
            memory_mask: 记忆的注意力掩码, 用于控制交叉注意力的可见范围, 这里为None
            tgt_key_padding_mask: 目标序列的填充掩码, 这里为None
            memory_key_padding_mask: 记忆的填充掩码, 这里为None
            pos: 编码器输出的位置编码, shape: [HW+2, batch_size, hidden_dim]
            query_pos: 查询的位置编码, shape: [num_queries, batch_size, hidden_dim]
        Returns:
            如果return_intermediate为True: shape: [num_layers, num_queries, batch_size, hidden_dim]
            否则: shape: [1, num_queries, batch_size, hidden_dim]
        """
        output = tgt # shape: [num_queries, batch_size, hidden_dim]

        # 用于存储中间层的输出
        intermediate = []

        # 依次通过每个解码器层
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 是否对输出进行归一化
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # 是否返回中间层的输出
        if self.return_intermediate:
            return torch.stack(intermediate) # shape: [num_layers, num_queries, batch_size, hidden_dim]

        return output.unsqueeze(0) # shape: [1, num_queries, batch_size, hidden_dim]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层
    实现了标准Transformer编码器层的功能,包括自注意力机制和前馈网络
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """初始化Transformer编码器层
        Args:
            d_model: 输入特征维度
            nhead: 注意力头数量
            dim_feedforward: 前馈网络隐藏层维度
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在注意力层前进行归一化
        """
        super().__init__()
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 第一个线性层
        self.dropout = nn.Dropout(dropout)                  # dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 第二个线性层

        # 层归一化和dropout层
        self.norm1 = nn.LayerNorm(d_model)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 第二个归一化层
        self.dropout1 = nn.Dropout(dropout)  # 第一个dropout层
        self.dropout2 = nn.Dropout(dropout)  # 第二个dropout层

        # 激活函数和归一化顺序标志
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """将位置编码加到输入张量上
        Args:
            tensor: 输入张量
            pos: 位置编码
        Returns:
            位置编码与输入张量的和
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """后归一化的前向传播
        Args:
            src: 输入序列
            src_mask: 注意力掩码
            src_key_padding_mask: padding掩码
            pos: 位置编码
        Returns:
            处理后的特征
        """
        # 1. 自注意力层
        q = k = self.with_pos_embed(src, pos)  # 将位置编码加到查询和键上
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        
        # 2. 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 残差连接
        src = src + self.dropout2(src2)
        # 后归一化
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """前归一化的前向传播
        Args:
            src: 输入序列
                shape: (seq_len, batch_size, hidden_dim)
                例如: (102, 8, 512) - 102个token, 8个批次, 512维特征
            
            src_mask: 注意力掩码，用于控制每个位置能看到哪些其他位置
                shape: (seq_len, seq_len)
                可选参数, 默认为None, 表示所有位置都可以相互关注
            
            src_key_padding_mask: 填充掩码，标记哪些位置是填充的
                shape: (batch_size, seq_len)
                例如: (8, 102) - 8个批次, 每个序列102个位置的填充标记
                True表示该位置是填充的, 应该被忽略
            
            pos: 位置编码
                shape: (seq_len, batch_size, hidden_dim)
                例如: (102, 8, 512)
                为每个token添加位置信息
        Returns:
            处理后的特征
        """
        # 1. 自注意力层
        # 前归一化
        src2 = self.norm1(src)
        # 将位置编码加到查询和键上
        q = k = self.with_pos_embed(src2, pos)
        # value 使用原始的输入序列src2（没有加位置编码）
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接
        src = src + self.dropout1(src2) # shape: (102, 8, 512)
        # 预归一化
        src2 = self.norm2(src)
        
        # 2. 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # 残差连接
        src = src + self.dropout2(src2)
        
        return src # shape: (102, 8, 512)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """前向传播函数
        根据normalize_before标志选择前归一化或后归一化的处理流程。
        对于深层 Transformer(层数>6),推荐使用前归一化。
        对于浅层 Transformer,两种方式都可以，后归一化可能性能略好。
        如果遇到训练不稳定的问题，可以尝试切换到前归一化。
        Args:
            src: 输入序列
            src_mask: 注意力掩码
            src_key_padding_mask: padding掩码
            pos: 位置编码
        Returns:
            处理后的特征
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # nn.LayerNorm用于对输入进行层归一化，有助于稳定网络训练
        # 它对最后一个维度进行归一化，保持输入的形状不变
        # 参数d_model指定了要归一化的特征数量
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        # 是否在注意力层前进行归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """将位置编码加到输入张量上
        Args:
            tensor: 输入张量 
            pos: 位置编码
        Returns:
            tensor + pos 或原始 tensor(如果pos为None)
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """后归一化的前向传播
        Args:
            Args:
            tgt: 目标序列, shape: (num_queries, batch_size, hidden_dim)
            memory: 编码器的输出, shape: (src_len, batch_size, hidden_dim)
            tgt_mask: 目标序列的注意力掩码, 这里为None
            memory_mask: 记忆的注意力掩码, 这里为None
            tgt_key_padding_mask: 目标序列的填充掩码, 这里为None
            memory_key_padding_mask: 记忆的填充掩码, 这里为None
            pos: 编码器输出的位置编码, shape: (src_len, batch_size, hidden_dim)
            query_pos: 查询的位置编码, [num_queries, batch_size, hidden_dim]
        """
        # 1. 自注意力层
        # q和k都是目标序列加上位置编码
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接和归一化
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 2. 交叉注意力层
        # query是目标序列，key是编码器输出
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # 残差连接和归一化
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # 3. 前馈神经网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 残差连接
        tgt = tgt + self.dropout3(tgt2)
        # 后归一化
        tgt = self.norm3(tgt)
        
        return tgt # shape: (num_queries, batch_size, hidden_dim)

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """前归一化的前向传播
        Args:
            tgt: 目标序列, shape: (num_queries, batch_size, hidden_dim)
            memory: 编码器的输出, shape: (src_len, batch_size, hidden_dim)
            tgt_mask: 目标序列的注意力掩码, 这里为None
            memory_mask: 记忆的注意力掩码, 这里为None
            tgt_key_padding_mask: 目标序列的填充掩码, 这里为None
            memory_key_padding_mask: 记忆的填充掩码, 这里为None
            pos: 编码器输出的位置编码, shape: (src_len, batch_size, hidden_dim)
            query_pos: 查询的位置编码, [num_queries, batch_size, hidden_dim]
        """
        # 前归一化
        tgt2 = self.norm1(tgt)
        # Masked multihead_attn
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接和归一化
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # 交叉注意力 multihead_attn
        # 查询(Q)：来自解码器的输出 + 查询位置编码
        # 键(K)：来自编码器的输出 + 编码器位置编码
        # 值(V)：来自编码器的输出
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # 残差连接和归一化
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # 残差连接
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt # shape: (num_queries, batch_size, hidden_dim)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """根据配置选择使用前归一化还是后归一化的前向传播"""
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    # nn.ModuleList用于创建一个模块列表
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
