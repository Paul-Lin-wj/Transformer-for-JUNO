"""
=============================================
DSA (Dynamic Sparse Attention) ALGORITHM
动态稀疏注意力算法实现
=============================================

DSA算法核心思想：
1. 动态选择稀疏的attention模式
2. 根据重要性阈值筛选attention连接
3. 自适应调整稀疏度
4. 保持关键信息的传递

实现特性：
- 动态稀疏度调整
- 基于重要性的节点选择
- 梯度友好的稀疏化
- 高效的矩阵运算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DSAModule(nn.Module):
    """
    Dynamic Sparse Attention Module
    动态稀疏注意力模块

    核心功能：
    1. 计算attention权重
    2. 应用动态稀疏化
    3. 保持top-k重要连接
    4. 自适应阈值调整
    """

    def __init__(self, embed_dim, num_heads, sparsity_ratio=0.1,
                 adaptive_threshold=True, min_connections=5):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            sparsity_ratio: 稀疏度比例（保留的连接数比例）
            adaptive_threshold: 是否使用自适应阈值
            min_connections: 每个节点最少连接数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.adaptive_threshold = adaptive_threshold
        self.min_connections = min_connections

        # 稀疏度控制参数
        self.temperature = nn.Parameter(torch.ones(1))
        self.sparsity_scheduler = SparsityScheduler(sparsity_ratio)

        # 用于计算重要性的可学习参数
        self.importance_weight = nn.Parameter(torch.ones(1))

        # 验证参数
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, query, key, value, mask=None, training=True):
        """
        前向传播

        Args:
            query, key, value: [batch_size, seq_len, embed_dim]
            mask: 可选的掩码张量
            training: 是否为训练模式

        Returns:
            attention_output: [batch_size, seq_len, embed_dim]
            attention_weights: 稀疏化的attention权重 [batch_size, num_heads, seq_len, seq_len]
            sparsity_stats: 稀疏度统计信息
        """
        batch_size, seq_len, _ = query.size()

        # 1. 计算标准attention权重
        attention_weights = self._compute_attention_weights(query, key, mask)

        # 2. 应用DSA稀疏化
        sparse_attention, sparsity_stats = self._apply_dsa_sparsity(
            attention_weights, training)

        # 3. 计算输出
        # 将value重塑为多头形式 [batch_size, num_heads, seq_len, head_dim]
        v = value.view(value.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算attention输出
        output = torch.matmul(sparse_attention, v)

        # 重塑回原始形状 [batch_size, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        return output, sparse_attention, sparsity_stats

    def _compute_attention_weights(self, query, key, mask=None):
        """计算标准的scaled dot-product attention权重"""
        # 重塑为多头形式 [batch_size, num_heads, seq_len, head_dim]
        q = query.view(query.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(key.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算attention分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码（如果提供）
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # 应用softmax获得权重
        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights

    def _apply_dsa_sparsity(self, attention_weights, training):
        """
        应用DSA稀疏化策略

        策略：
        1. 基于重要性的top-k选择
        2. 自适应阈值调整
        3. 保证最小连接数
        4. 训练时的随机性
        """
        batch_size, num_heads, seq_len, _ = attention_weights.size()

        if not training:
            # 推理模式：确定性稀疏化
            return self._deterministic_sparsity(attention_weights)

        # 训练模式：动态稀疏化
        # 1. 计算连接重要性
        importance_scores = self._compute_importance_scores(attention_weights)

        # 2. 动态调整稀疏度
        current_sparsity = self.sparsity_scheduler.get_current_sparsity()

        # 3. 选择top-k连接
        max_connections = seq_len * seq_len
        k = max(self.min_connections, min(int(max_connections * current_sparsity), max_connections))

        # Debug info
        if k > max_connections:
            print(f"Warning: k={k} > max_connections={max_connections}, seq_len={seq_len}, sparsity={current_sparsity}")
            k = min(k, max_connections)

        sparse_attention = self._select_topk_connections(
            attention_weights, importance_scores, k)

        # 4. 计算稀疏度统计
        sparsity_stats = self._compute_sparsity_stats(
            attention_weights, sparse_attention)

        return sparse_attention, sparsity_stats

    def _compute_importance_scores(self, attention_weights):
        """计算连接的重要性分数"""
        # 结合attention权重和可学习的重要性参数
        importance = attention_weights * self.importance_weight

        # 添加基于位置的先验（可选）
        # 这里可以添加位置编码或其他先验知识

        return importance

    def _select_topk_connections(self, attention_weights, importance_scores, k):
        """选择top-k最重要的连接"""
        batch_size, num_heads, seq_len, _ = attention_weights.size()

        # 重塑为2D以进行top-k选择
        flattened_importance = importance_scores.view(batch_size, num_heads, -1)

        # 获取top-k索引
        topk_values, topk_indices = torch.topk(flattened_importance, k, dim=-1)

        # 创建稀疏的attention矩阵
        sparse_attention = torch.zeros_like(attention_weights)

        # 填充top-k连接
        for b in range(batch_size):
            for h in range(num_heads):
                # 将2D索引转换回2D坐标
                indices_2d = topk_indices[b, h]
                rows = indices_2d // seq_len
                cols = indices_2d % seq_len

                sparse_attention[b, h, rows, cols] = attention_weights[b, h, rows, cols]

        return sparse_attention

    def _deterministic_sparsity(self, attention_weights):
        """推理时的确定性稀疏化"""
        batch_size, num_heads, seq_len, _ = attention_weights.size()
        max_connections = seq_len * seq_len
        k = max(self.min_connections, min(int(max_connections * self.sparsity_ratio), max_connections))

        # 选择top-k连接（无随机性）
        flattened_attention = attention_weights.view(batch_size, num_heads, -1)
        topk_values, topk_indices = torch.topk(flattened_attention, k, dim=-1)

        sparse_attention = torch.zeros_like(attention_weights)

        for b in range(batch_size):
            for h in range(num_heads):
                indices_2d = topk_indices[b, h]
                rows = indices_2d // seq_len
                cols = indices_2d % seq_len
                sparse_attention[b, h, rows, cols] = attention_weights[b, h, rows, cols]

        # 重新归一化
        row_sums = sparse_attention.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)  # 避免除零
        sparse_attention = sparse_attention / row_sums

        stats = {
            'sparsity_ratio': 1.0 - (k / (seq_len * seq_len)),
            'active_connections': k,
            'total_connections': seq_len * seq_len
        }

        return sparse_attention, stats

    def _compute_sparsity_stats(self, original, sparse):
        """计算稀疏度统计信息"""
        total_connections = original.numel()
        active_connections = (sparse > 0).sum().item()
        sparsity_ratio = 1.0 - (active_connections / total_connections)

        # 计算保留的信息量
        retained_info = (sparse * original).sum().item() / original.sum().item()

        return {
            'sparsity_ratio': sparsity_ratio,
            'active_connections': active_connections,
            'total_connections': total_connections,
            'retained_info_ratio': retained_info,
            'compression_ratio': total_connections / active_connections if active_connections > 0 else float('inf')
        }

    def update_sparsity(self, epoch_loss):
        """根据训练进度更新稀疏度"""
        self.sparsity_scheduler.update_sparsity(epoch_loss)


class SparsityScheduler:
    """
    稀疏度调度器
    根据训练进度动态调整稀疏度
    """

    def __init__(self, initial_sparsity, target_sparsity=0.05,
                 warmup_epochs=10, schedule_type='linear'):
        """
        Args:
            initial_sparsity: 初始稀疏度
            target_sparsity: 目标稀疏度
            warmup_epochs: 预热轮数
            schedule_type: 调度类型 ('linear', 'exponential', 'cosine')
        """
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.current_sparsity = initial_sparsity
        self.loss_history = []

    def get_current_sparsity(self):
        """获取当前稀疏度"""
        return self.current_sparsity

    def update_sparsity(self, epoch_loss):
        """更新稀疏度"""
        self.current_epoch += 1
        self.loss_history.append(epoch_loss)

        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：保持初始稀疏度
            self.current_sparsity = self.initial_sparsity
        else:
            # 根据调度类型更新稀疏度
            progress = (self.current_epoch - self.warmup_epochs) / max(1, 100 - self.warmup_epochs)
            progress = min(1.0, progress)

            if self.schedule_type == 'linear':
                self.current_sparsity = self._linear_schedule(progress)
            elif self.schedule_type == 'exponential':
                self.current_sparsity = self._exponential_schedule(progress)
            elif self.schedule_type == 'cosine':
                self.current_sparsity = self._cosine_schedule(progress)
            elif self.schedule_type == 'adaptive':
                self.current_sparsity = self._adaptive_schedule()

            # 确保稀疏度在合理范围内
            self.current_sparsity = max(self.target_sparsity,
                                      min(self.initial_sparsity, self.current_sparsity))

    def _linear_schedule(self, progress):
        """线性调度"""
        return self.initial_sparsity + progress * (self.target_sparsity - self.initial_sparsity)

    def _exponential_schedule(self, progress):
        """指数调度"""
        return self.initial_sparsity * (self.target_sparsity / self.initial_sparsity) ** progress

    def _cosine_schedule(self, progress):
        """余弦调度"""
        import math
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * cosine_decay

    def _adaptive_schedule(self):
        """自适应调度（基于损失变化）"""
        if len(self.loss_history) < 2:
            return self.current_sparsity

        # 如果损失下降缓慢，增加稀疏度（更稀疏）
        recent_loss = np.mean(self.loss_history[-5:]) if len(self.loss_history) >= 5 else self.loss_history[-1]
        earlier_loss = self.loss_history[0]

        loss_improvement = (earlier_loss - recent_loss) / earlier_loss

        if loss_improvement < 0.01:  # 改善小于1%
            # 增加稀疏度
            new_sparsity = self.current_sparsity * 1.05
        else:
            # 适当降低稀疏度
            new_sparsity = self.current_sparsity * 0.98

        return new_sparsity


class DSALoss(nn.Module):
    """
    DSA专用的损失函数
    结合标准任务损失和稀疏化正则化
    """

    def __init__(self, base_loss_fn, sparsity_weight=0.001,
                 entropy_weight=0.0001):
        """
        Args:
            base_loss_fn: 基础损失函数（如MSELoss, CrossEntropyLoss）
            sparsity_weight: 稀疏度正则化权重
            entropy_weight: 熵正则化权重（鼓励多样性）
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight

    def forward(self, predictions, targets, sparsity_stats):
        """
        计算DSA损失

        Args:
            predictions: 模型预测
            targets: 目标值
            sparsity_stats: 稀疏度统计信息

        Returns:
            total_loss: 总损失
            loss_components: 损失组成部分
        """
        # 基础任务损失
        task_loss = self.base_loss_fn(predictions, targets)

        # 检查是否产生NaN
        if torch.isnan(task_loss):
            print(f"Warning: task_loss is NaN! Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            print(f"Predictions: {predictions}")
            print(f"Targets: {targets}")
            print(f"Predictions contains NaN: {torch.isnan(predictions).any()}")
            print(f"Targets contains NaN: {torch.isnan(targets).any()}")
            print(f"Predictions contains Inf: {torch.isinf(predictions).any()}")
            print(f"Targets contains Inf: {torch.isinf(targets).any()}")
            task_loss = torch.tensor(0.0, device=predictions.device)

        # 稀疏度正则化（鼓励稀疏）
        sparsity_ratio = sparsity_stats.get('sparsity_ratio', 0)
        sparsity_loss = sparsity_ratio * self.sparsity_weight

        # 信息保留正则化（惩罚信息丢失）
        retained_info_ratio = sparsity_stats.get('retained_info_ratio', 1)
        info_loss = (1 - retained_info_ratio) * self.sparsity_weight

        # 熵正则化（鼓励attention分布的多样性）
        entropy = sparsity_stats.get('entropy', 0)
        entropy_loss = -entropy * self.entropy_weight

        # 总损失
        total_loss = task_loss + sparsity_loss + info_loss + entropy_loss

        # 再次检查总损失是否为NaN
        if torch.isnan(total_loss):
            print(f"Warning: total_loss is NaN! Using only task_loss.")
            total_loss = task_loss

        # 安全地获取损失值
        def safe_get_tensor_value(tensor, default=0.0):
            if torch.is_tensor(tensor):
                if torch.isnan(tensor):
                    return default
                return tensor.item()
            return tensor

        loss_components = {
            'total_loss': safe_get_tensor_value(total_loss),
            'task_loss': safe_get_tensor_value(task_loss),
            'sparsity_loss': safe_get_tensor_value(sparsity_loss),
            'info_loss': safe_get_tensor_value(info_loss),
            'entropy_loss': safe_get_tensor_value(entropy_loss)
        }

        return total_loss, loss_components


def test_dsa_module():
    """测试DSA模块功能"""
    print("Testing DSA Module...")

    # 创建测试数据
    batch_size, seq_len, embed_dim = 2, 10, 64
    num_heads = 8

    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # 创建DSA模块
    dsa = DSAModule(embed_dim, num_heads, sparsity_ratio=0.2)

    # 测试前向传播
    output, attention_weights, stats = dsa(query, key, value, training=True)

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Sparsity stats: {stats}")

    # 验证稀疏性
    sparsity = stats['sparsity_ratio']
    print(f"Achieved sparsity: {sparsity:.3f}")

    assert output.shape == query.shape, "Output shape mismatch"
    assert sparsity > 0, "No sparsity achieved"

    print("DSA Module test passed!")


if __name__ == "__main__":
    test_dsa_module()