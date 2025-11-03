"""
=============================================
TRANSFORMER MODEL WITH DSA OPTIMIZATION
集成DSA算法的Transformer模型
=============================================

模型特性：
1. 集成Dynamic Sparse Attention算法
2. 可配置的稀疏度策略
3. 标准Transformer架构兼容
4. 高效的训练和推理

架构设计：
- Multi-Head Attention with DSA
- Feed-Forward Networks
- Layer Normalization
- Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dsa_algorithm import DSAModule, DSALoss


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class DSAMultiHeadAttention(nn.Module):
    """
    集成DSA的多头注意力机制
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 dsa_enabled=True, dsa_config=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dsa_enabled = dsa_enabled

        assert self.head_dim * num_heads == embed_dim

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # DSA模块
        if dsa_enabled:
            dsa_config = dsa_config or {}
            self.dsa = DSAModule(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sparsity_ratio=dsa_config.get('sparsity_ratio', 0.1),
                adaptive_threshold=dsa_config.get('adaptive_threshold', True),
                min_connections=dsa_config.get('min_connections', 5)
            )
            # 设置额外的DSA参数
            if hasattr(self.dsa, 'sparsity_scheduler'):
                self.dsa.sparsity_scheduler.target_sparsity = dsa_config.get('target_sparsity', 0.05)
                self.dsa.sparsity_scheduler.warmup_epochs = dsa_config.get('warmup_epochs', 10)
                self.dsa.sparsity_scheduler.schedule_type = dsa_config.get('schedule_type', 'linear')
        else:
            self.dsa = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, training=True):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码
            training: 是否为训练模式

        Returns:
            output: 输出张量 [batch_size, seq_len, embed_dim]
            attention_stats: 注意力统计信息
        """
        batch_size, seq_len, embed_dim = x.size()
        residual = x

        # 计算Q, K, V
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        if self.dsa_enabled and self.dsa is not None:
            # 使用DSA优化attention
            attention_output, attention_weights, stats = self.dsa(
                query, key, value, mask, training
            )
        else:
            # 标准attention
            attention_output, attention_weights, stats = self._standard_attention(
                query, key, value, mask
            )

        # 输出投影
        output = self.out_proj(attention_output)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        # 添加统计信息
        stats['dsa_enabled'] = self.dsa_enabled

        return output, stats

    def _standard_attention(self, query, key, value, mask=None):
        """标准scaled dot-product attention"""
        batch_size, seq_len, embed_dim = query.size()
        num_heads = self.num_heads
        head_dim = self.head_dim

        # 重塑为多头形式
        q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # 计算attention分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # 应用掩码
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用attention权重
        attention_output = torch.matmul(attention_weights, v)

        # 重塑回原始形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )

        # 统计信息
        stats = {
            'sparsity_ratio': 0.0,
            'active_connections': seq_len * seq_len,
            'total_connections': seq_len * seq_len,
            'retained_info_ratio': 1.0,
            'compression_ratio': 1.0
        }

        return attention_output, attention_weights, stats


class DSATransformerEncoderLayer(nn.Module):
    """
    带DSA的Transformer编码器层
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1,
                 dsa_enabled=True, dsa_config=None):
        super().__init__()

        # DSA Multi-Head Attention
        self.self_attention = DSAMultiHeadAttention(
            embed_dim, num_heads, dropout, dsa_enabled, dsa_config
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, training=True):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: 注意力掩码
            training: 是否为训练模式

        Returns:
            output: [batch_size, seq_len, embed_dim]
            layer_stats: 层统计信息
        """
        # Self-attention with DSA
        attn_output, attn_stats = self.self_attention(x, mask, training)

        # Feed-forward
        ffn_output = self.ffn(attn_output)

        # Residual connection and layer norm
        output = self.layer_norm(ffn_output + attn_output)

        # 合并统计信息
        layer_stats = {
            'attention_stats': attn_stats,
            'layer_output_norm': output.norm(dim=-1).mean().item()
        }

        return output, layer_stats


class TransformerWithDSA(nn.Module):
    """
    集成DSA算法的完整Transformer模型

    特性：
    - 可配置的DSA优化
    - 支持回归和分类任务
    - 完整的训练和推理支持
    - 详细的统计信息收集
    """

    def __init__(self, input_dim, embed_dim, num_heads, num_layers,
                 ff_dim, output_dim, dropout=0.1,
                 dsa_enabled=True, dsa_config=None,
                 task_type='regression'):
        """
        Args:
            input_dim: 输入特征维度
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            ff_dim: 前馈网络隐藏层维度
            output_dim: 输出维度
            dropout: Dropout率
            dsa_enabled: 是否启用DSA
            dsa_config: DSA配置参数
            task_type: 任务类型 ('regression' 或 'classification')
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dsa_enabled = dsa_enabled
        self.task_type = task_type

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim)

        # Transformer编码器层
        self.layers = nn.ModuleList([
            DSATransformerEncoderLayer(
                embed_dim, num_heads, ff_dim, dropout,
                dsa_enabled, dsa_config
            )
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

        # 任务特定的激活函数
        if task_type == 'classification':
            self.output_activation = nn.Softmax(dim=-1)
        elif task_type == 'binary_classification':
            self.output_activation = nn.Sigmoid()
        else:  # regression
            self.output_activation = nn.Identity()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, mask=None, training=True, return_attention=False):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            training: 是否为训练模式
            return_attention: 是否返回attention权重

        Returns:
            output: 模型输出 [batch_size, seq_len, output_dim]
            stats: 模型统计信息
        """
        # 输入投影
        x = self.input_projection(x)

        # 位置编码
        x = self.pos_encoding(x)

        # 存储所有层的统计信息
        all_stats = {
            'layer_stats': [],
            'global_stats': {}
        }

        # 通过所有Transformer层
        for i, layer in enumerate(self.layers):
            x, layer_stats = layer(x, mask, training)
            all_stats['layer_stats'].append({
                'layer': i,
                **layer_stats
            })

        # 输出投影
        output = self.output_projection(x)
        output = self.output_activation(output)

        # 计算全局统计信息
        all_stats['global_stats'] = self._compute_global_stats(
            all_stats['layer_stats']
        )

        if return_attention:
            return output, all_stats
        else:
            return output

    def _compute_global_stats(self, layer_stats):
        """计算全局统计信息"""
        if not layer_stats:
            return {}

        # 聚合所有层的稀疏度信息
        sparsity_ratios = []
        compression_ratios = []
        retained_infos = []

        for stats in layer_stats:
            attn_stats = stats.get('attention_stats', {})
            sparsity_ratios.append(attn_stats.get('sparsity_ratio', 0))
            compression_ratios.append(attn_stats.get('compression_ratio', 1))
            retained_infos.append(attn_stats.get('retained_info_ratio', 1))

        return {
            'avg_sparsity': np.mean(sparsity_ratios) if sparsity_ratios else 0,
            'max_sparsity': np.max(sparsity_ratios) if sparsity_ratios else 0,
            'min_sparsity': np.min(sparsity_ratios) if sparsity_ratios else 0,
            'avg_compression': np.mean(compression_ratios) if compression_ratios else 1,
            'avg_retained_info': np.mean(retained_infos) if retained_infos else 1,
            'dsa_enabled': self.dsa_enabled,
            'num_layers': len(layer_stats)
        }

    def get_model_size(self):
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dsa_enabled': self.dsa_enabled,
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        }

    def enable_dsa(self, dsa_config=None):
        """启用DSA优化"""
        for layer in self.layers:
            if hasattr(layer.self_attention, 'dsa') and layer.self_attention.dsa is not None:
                if dsa_config:
                    # 更新DSA配置
                    for key, value in dsa_config.items():
                        if hasattr(layer.self_attention.dsa, key):
                            setattr(layer.self_attention.dsa, key, value)
        self.dsa_enabled = True

    def disable_dsa(self):
        """禁用DSA优化"""
        self.dsa_enabled = False

    def update_dsa_sparsity(self, epoch_loss):
        """更新DSA稀疏度（如果启用）"""
        if self.dsa_enabled:
            for layer in self.layers:
                if hasattr(layer.self_attention, 'dsa') and layer.self_attention.dsa is not None:
                    layer.self_attention.dsa.update_sparsity(epoch_loss)


def create_default_dsa_config():
    """创建默认的DSA配置"""
    return {
        'sparsity_ratio': 0.1,          # 初始稀疏度10%
        'adaptive_threshold': True,      # 使用自适应阈值
        'min_connections': 5,            # 最少连接数
        'target_sparsity': 0.05,         # 目标稀疏度5%
        'warmup_epochs': 10,             # 预热轮数
        'schedule_type': 'adaptive'      # 自适应调度
    }


def test_transformer_dsa():
    """测试Transformer with DSA"""
    print("Testing Transformer with DSA...")

    # 配置参数
    batch_size = 4
    seq_len = 50
    input_dim = 10
    embed_dim = 128
    num_heads = 8
    num_layers = 2
    ff_dim = 256
    output_dim = 1

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_dim)

    # 创建模型
    model = TransformerWithDSA(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        output_dim=output_dim,
        dsa_enabled=True,
        dsa_config=create_default_dsa_config(),
        task_type='regression'
    )

    print(f"Model size: {model.get_model_size()}")

    # 测试前向传播
    output, stats = model(x, training=True, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Global stats: {stats['global_stats']}")

    # 测试无DSA模式
    model.disable_dsa()
    output_no_dsa, stats_no_dsa = model(x, training=True, return_attention=True)

    print(f"No-DSA stats: {stats_no_dsa['global_stats']}")

    print("Transformer with DSA test passed!")


if __name__ == "__main__":
    test_transformer_dsa()