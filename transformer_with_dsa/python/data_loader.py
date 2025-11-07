"""
=============================================
DATA LOADER FOR TRANSFORMER WITH DSA
DSA优化Transformer的数据加载器
=============================================

功能特性：
1. 支持多种数据格式
2. 自动数据预处理
3. 批量数据加载
4. 训练/验证集分割
5. 数据增强选项
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import logging


class TransformerDataset(Dataset):
    """
    Transformer模型的数据集类
    """

    def __init__(self, data, targets, seq_len=None,
                 normalize=True, augment=False):
        """
        Args:
            data: 输入数据 [N, input_dim] 或 [N, seq_len, input_dim]
            targets: 目标数据 [N, target_dim]
            seq_len: 序列长度（如果数据是2D的）
            normalize: 是否归一化数据
            augment: 是否使用数据增强
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.augment = augment

        # 如果数据是2D的，转换为序列数据
        if len(self.data.shape) == 2:
            if seq_len is None:
                # 使用整个数据作为序列
                self.data = self.data.unsqueeze(1)  # [N, 1, input_dim]
            else:
                # 滑动窗口创建序列
                self.data = self._create_sequences(self.data, seq_len)
        # 如果数据已经是3D的（N, seq_len, features），保持不变

        # 数据归一化
        if normalize:
            self._normalize_data()

        self.logger = logging.getLogger(__name__)

    def _create_sequences(self, data, seq_len):
        """
        从2D数据创建序列数据

        Args:
            data: [N, input_dim]
            seq_len: 序列长度

        Returns:
            sequences: [N-seq_len+1, seq_len, input_dim]
        """
        if data.size(0) <= seq_len:
            # 数据长度不足，使用padding
            padding = seq_len - data.size(0)
            padded_data = F.pad(data, (0, 0, 0, padding), mode='constant', value=0)
            return padded_data.unsqueeze(0)  # [1, seq_len, input_dim]
        else:
            # 创建滑动窗口序列
            sequences = []
            for i in range(data.size(0) - seq_len + 1):
                seq = data[i:i+seq_len]
                sequences.append(seq)
            return torch.stack(sequences)  # [N-seq_len+1, seq_len, input_dim]

    def _normalize_data(self):
        """数据归一化"""
        # 保存原始统计信息
        self.data_mean = self.data.mean(dim=0, keepdim=True)
        self.data_std = self.data.std(dim=0, keepdim=True) + 1e-8

        # 归一化
        self.data = (self.data - self.data_mean) / self.data_std

        # 目标值也归一化
        self.target_mean = self.targets.mean(dim=0, keepdim=True)
        self.target_std = self.targets.std(dim=0, keepdim=True) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]

        # 数据增强
        if self.augment and torch.rand(1) < 0.5:
            data = self._augment_data(data)

        return data, target

    def _augment_data(self, data):
        """数据增强"""
        # 随机噪声
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(data) * 0.01
            data = data + noise

        # 随机缩放
        if torch.rand(1) < 0.3:
            scale = 0.9 + 0.2 * torch.rand(1).item()
            data = data * scale

        return data

    def denormalize_target(self, normalized_target):
        """反归一化目标值"""
        return normalized_target * self.target_std + self.target_mean


class DataLoaderWithDSA:
    """
    专为DSA优化Transformer设计的数据加载器
    支持4:1的数据集和测试集分割
    """

    def __init__(self, dataset_path, input_dim, target_dim=1,
                 seq_len=None, batch_size=32, train_ratio=0.8,
                 normalize=True, augment_train=False,
                 num_workers=4, random_seed=42, max_files=None):
        """
        Args:
            dataset_path: 数据集路径
            input_dim: 输入特征维度
            target_dim: 目标维度
            seq_len: 序列长度
            batch_size: 批大小
            train_ratio: 训练集比例（在训练+验证集中的比例）
            normalize: 是否归一化
            augment_train: 是否对训练集进行数据增强
            num_workers: 数据加载进程数
            random_seed: 随机种子
            max_files: 最大加载文件数（None表示加载所有文件）
        """
        self.dataset_path = dataset_path
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.augment_train = augment_train
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.max_files = max_files

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # 加载数据
        self.data, self.targets = self._load_dataset()
        self.logger.info(f"Debug: After _load_dataset - data shape: {self.data.shape}, targets shape: {self.targets.shape}")

        # 对于序列数据，直接使用整个轨迹作为输入（与原始Transformer相同）
        if len(self.data.shape) == 3:
            # 每个事件作为一个样本（不使用滑动窗口）
            all_sequences = []
            all_targets = []

            for event_idx in range(self.data.shape[0]):
                sequence = self.data[event_idx]  # [seq_len, features]
                target = self.targets[event_idx]  # [target_dim]

                # 直接使用整个序列
                self.logger.info(f"Event {event_idx}: sequence length = {sequence.shape[0]}")
                all_sequences.append(sequence)
                all_targets.append(target)

            # 转换为numpy数组
            all_sequences = np.array(all_sequences)
            all_targets = np.array(all_targets)

            self.logger.info(f"Created {len(all_sequences)} samples from {self.data.shape[0]} events (1:1)")

            # 第一步：4:1分割为训练+验证集 和 测试集
            train_val_data, test_data, train_val_targets, test_targets = \
                train_test_split(all_sequences, all_targets, train_size=0.8,  # 4:1分割
                               random_state=random_seed)

            # 第二步：将训练+验证集按train_ratio分割为训练集和验证集
            self.train_data, self.val_data, self.train_targets, self.val_targets = \
                train_test_split(train_val_data, train_val_targets, train_size=train_ratio,
                               random_state=random_seed)

            # 保存测试数据
            self.test_data = test_data
            self.test_targets = test_targets

        else:
            # 对于2D数据
            # 第一步：4:1分割为训练+验证集 和 测试集
            train_val_data, test_data, train_val_targets, test_targets = \
                train_test_split(self.data, self.targets, train_size=0.8,  # 4:1分割
                               random_state=random_seed)

            # 第二步：将训练+验证集按train_ratio分割为训练集和验证集
            self.train_data, self.val_data, self.train_targets, self.val_targets = \
                train_test_split(train_val_data, train_val_targets, train_size=train_ratio,
                               random_state=random_seed)

            # 保存测试数据
            self.test_data = test_data
            self.test_targets = test_targets

        # 创建数据集
        self.train_dataset = TransformerDataset(
            self.train_data, self.train_targets, seq_len,
            normalize=normalize, augment=augment_train
        )

        self.val_dataset = TransformerDataset(
            self.val_data, self.val_targets, seq_len,
            normalize=normalize, augment=False
        )

        self.test_dataset = TransformerDataset(
            self.test_data, self.test_targets, seq_len,
            normalize=normalize, augment=False
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Dataset loaded: {len(self.train_data)} training samples, "
                        f"{len(self.val_data)} validation samples, "
                        f"{len(self.test_data)} test samples")

    def _load_dataset(self):
        """
        加载数据集
        支持多种格式：.npy, .pkl, .csv, .json
        如果是目录，则加载目录中所有的.pt文件
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # 如果是目录，查找所有.pt文件并合并
        if os.path.isdir(self.dataset_path):
            pt_files = []
            for file in os.listdir(self.dataset_path):
                file_path = os.path.join(self.dataset_path, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext in ['.pt', '.pth']:
                        pt_files.append(file_path)

            if not pt_files:
                raise FileNotFoundError(f"No .pt/.pth files found in directory: {self.dataset_path}")

            # 按文件名排序以确保一致性
            pt_files.sort()

            # 如果设置了最大文件数，限制加载数量
            if self.max_files is not None and self.max_files > 0:
                pt_files = pt_files[:self.max_files]
                self.logger.info(f"Limiting to first {len(pt_files)} files (max_files={self.max_files})")

            self.logger.info(f"Found {len(pt_files)} .pt/.pth files")
            self.logger.info("Loading and merging data files:")

            all_data = []
            all_targets = []

            for i, file_path in enumerate(pt_files):
                self.logger.info(f"  [{i+1}/{len(pt_files)}] {os.path.basename(file_path)}")
                loaded_data = torch.load(file_path, map_location='cpu')

                if isinstance(loaded_data, dict):
                    data = loaded_data.get('x_data', loaded_data.get('data', loaded_data.get('x', None)))
                    targets = loaded_data.get('y_data', loaded_data.get('targets', loaded_data.get('y', None)))
                else:
                    data = loaded_data
                    targets = None

                if data is not None:
                    if isinstance(data, torch.Tensor):
                        data = data.numpy()
                    all_data.append(data)

                    if targets is not None:
                        if isinstance(targets, torch.Tensor):
                            targets = targets.numpy()
                        all_targets.append(targets)

            # 合并所有数据
            if all_data:
                data = np.concatenate(all_data, axis=0)
                if all_targets:
                    targets = np.concatenate(all_targets, axis=0)
                else:
                    # 如果没有targets，使用data的最后几列作为targets
                    targets = data[:, -self.target_dim:]

                self.logger.info(f"Total merged data shape: {data.shape}")
                self.logger.info(f"Total merged targets shape: {targets.shape}")

                # 打印数据统计信息
                self.logger.info("Data Statistics:")
                self.logger.info(f"  Data shape: {data.shape}")
                self.logger.info(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")
                self.logger.info(f"  Data mean: {data.mean():.6f}")
                self.logger.info(f"  Data std: {data.std():.6f}")
                self.logger.info(f"  Targets shape: {targets.shape}")
                self.logger.info(f"  Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
                self.logger.info(f"  Targets mean: {targets.mean():.6f}")
                self.logger.info(f"  Targets std: {targets.std():.6f}")

                # 检查目标数据结构（入口点和出口点）
                if targets.shape[1] >= 6:
                    entry_points = targets[:, :3]
                    exit_points = targets[:, 3:6]
                    self.logger.info("Entry Points Statistics:")
                    self.logger.info(f"  X range: [{entry_points[:, 0].min():.6f}, {entry_points[:, 0].max():.6f}]")
                    self.logger.info(f"  Y range: [{entry_points[:, 1].min():.6f}, {entry_points[:, 1].max():.6f}]")
                    self.logger.info(f"  Z range: [{entry_points[:, 2].min():.6f}, {entry_points[:, 2].max():.6f}]")
                    self.logger.info("Exit Points Statistics:")
                    self.logger.info(f"  X range: [{exit_points[:, 0].min():.6f}, {exit_points[:, 0].max():.6f}]")
                    self.logger.info(f"  Y range: [{exit_points[:, 1].min():.6f}, {exit_points[:, 1].max():.6f}]")
                    self.logger.info(f"  Z range: [{exit_points[:, 2].min():.6f}, {exit_points[:, 2].max():.6f}]")
            else:
                raise ValueError("No valid data found in .pt files")

            # 返回加载的数据
            return data, targets

        # 如果是单个文件
        else:
            file_ext = os.path.splitext(self.dataset_path)[1].lower()

            try:
                if file_ext == '.npy':
                    data = np.load(self.dataset_path)
                    if len(data.shape) == 1:
                        data = data.reshape(-1, 1)
                    targets = data[:, :self.target_dim] if data.shape[1] >= self.target_dim else data[:, :1]
                    data = data[:, :self.input_dim] if data.shape[1] >= self.input_dim else \
                           np.pad(data, ((0, 0), (0, max(0, self.input_dim - data.shape[1]))))

                elif file_ext in ['.pt', '.pth']:
                    loaded_data = torch.load(self.dataset_path, map_location='cpu')
                    if isinstance(loaded_data, dict):
                        # 处理字典格式 - 首先尝试原始transformer的keys
                        data = loaded_data.get('x_data', loaded_data.get('data', loaded_data.get('x', None)))
                        targets = loaded_data.get('y_data', loaded_data.get('targets', loaded_data.get('y', None)))

                        # 如果data或targets是None，尝试其他key
                        if data is None:
                            for key in loaded_data:
                                if isinstance(loaded_data[key], torch.Tensor) and loaded_data[key].numel() > 0:
                                    # 优先选择3D张量作为数据（序列数据）
                                    if len(loaded_data[key].shape) == 3 and loaded_data[key].shape[-1] <= self.input_dim * 10:
                                        data = loaded_data[key]
                                        break
                            # 如果还是None，选择最大的张量
                            if data is None:
                                max_tensor = None
                                max_size = 0
                                for key in loaded_data:
                                    if isinstance(loaded_data[key], torch.Tensor) and loaded_data[key].numel() > max_size:
                                        max_size = loaded_data[key].numel()
                                        max_tensor = loaded_data[key]
                                data = max_tensor

                        if targets is None:
                            for key in loaded_data:
                                if isinstance(loaded_data[key], torch.Tensor) and loaded_data[key].numel() > 0:
                                    # 优先选择2D张量作为目标
                                    if len(loaded_data[key].shape) == 2 and loaded_data[key].shape[-1] >= self.target_dim:
                                        targets = loaded_data[key]
                                        break
                                    # 或者选择1D或2D张量
                                    elif len(loaded_data[key].shape) <= 2 and loaded_data[key].shape[-1] >= self.target_dim:
                                        targets = loaded_data[key]
                                        break

                        # 转换为numpy
                        if isinstance(data, torch.Tensor):
                            data = data.numpy()
                        if isinstance(targets, torch.Tensor):
                            targets = targets.numpy()

                        # 如果还是None，使用数据的最后一部分作为targets
                        if targets is None and isinstance(data, np.ndarray):
                            # 对于轨迹数据，targets通常是最后6个值（入射点3个+出射点3个）
                            targets = data[:, -self.target_dim:] if data.shape[1] >= self.target_dim else data[:, :self.target_dim]
                    else:
                        data = loaded_data.numpy() if isinstance(loaded_data, torch.Tensor) else loaded_data
                        targets = data[:, :self.target_dim]

                    if len(data.shape) == 1:
                        data = data.reshape(-1, 1)
                    if len(targets.shape) == 1:
                        targets = targets.reshape(-1, 1)

                elif file_ext == '.pkl':
                    with open(self.dataset_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                    if isinstance(loaded_data, dict):
                        data = loaded_data.get('data', loaded_data.get('x', loaded_data))
                        targets = loaded_data.get('targets', loaded_data.get('y', loaded_data))
                    else:
                        data = loaded_data
                        targets = data[:, :self.target_dim]

                elif file_ext == '.csv':
                    df = pd.read_csv(self.dataset_path)
                    data = df.values
                    targets = data[:, :self.target_dim] if data.shape[1] >= self.target_dim else data[:, :1]
                    data = data[:, :self.input_dim] if data.shape[1] >= self.input_dim else \
                           np.pad(data, ((0, 0), (0, max(0, self.input_dim - data.shape[1]))))

                elif file_ext == '.json':
                    with open(self.dataset_path, 'r') as f:
                        json_data = json.load(f)
                    data = np.array(json_data.get('data', json_data))
                    targets = np.array(json_data.get('targets', json_data))

                else:
                    raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .npy, .pkl, .csv, .json, .pt, .pth")

                # 确保数据形状正确
                self.logger.info(f"Debug: Initial data shape: {data.shape}, targets shape: {targets.shape}")

                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                if len(targets.shape) == 1:
                    targets = targets.reshape(-1, 1)

                # 对于3D数据 (N, seq_len, features)，确保特征维度正确
                if len(data.shape) == 3:
                    # data已经是序列格式 [N, seq_len, features]
                    self.logger.info(f"Debug: Processing 3D data with shape: {data.shape}")
                    # 检查特征维度
                    if data.shape[2] > self.input_dim:
                        data = data[:, :, :self.input_dim]
                    elif data.shape[2] < self.input_dim:
                        # 填充特征维度
                        padding = np.zeros((data.shape[0], data.shape[1], self.input_dim - data.shape[2]))
                        data = np.concatenate([data, padding], axis=2)
                elif len(data.shape) == 2:
                    # 如果2D数据的第二个维度是input_dim，将其作为特征
                    self.logger.info(f"Debug: 2D data shape before processing: {data.shape}")
                    if data.shape[1] >= self.input_dim:
                        data = data[:, :self.input_dim]
                    else:
                        # 填充到input_dim
                        padding = np.zeros((data.shape[0], self.input_dim - data.shape[1]))
                        data = np.concatenate([data, padding], axis=1)

                # 处理targets维度
                if len(targets.shape) == 3:
                    # 对于3D targets，压缩序列维度
                    targets = targets[:, 0, :]  # 取第一个时间步

                if targets.shape[1] > self.target_dim:
                    targets = targets[:, :self.target_dim]
                elif targets.shape[1] < self.target_dim:
                    padding = np.zeros((targets.shape[0], self.target_dim - targets.shape[1]))
                    targets = np.concatenate([targets, padding], axis=1)

                # 检查数据有效性
                if np.isnan(data).any():
                    self.logger.warning(f"Data contains NaN values after loading. Shape: {data.shape}")
                    # 替换NaN值为0
                    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                if np.isnan(targets).any():
                    self.logger.warning(f"Targets contain NaN values after loading. Shape: {targets.shape}")
                    # 替换NaN值为0
                    targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

                self.logger.info(f"Debug: Before return - data shape: {data.shape}, targets shape: {targets.shape}")
                return data, targets

            except Exception as e:
                self.logger.error(f"Error loading dataset: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # 创建合成数据作为后备
                self.logger.warning("Creating synthetic dataset as fallback")
                synthetic_data, synthetic_targets = self._create_synthetic_data()
                return synthetic_data, synthetic_targets

    def _create_synthetic_data(self, num_samples=1000):
        """创建合成数据集作为后备"""
        # 创建时间序列数据
        t = np.linspace(0, 10, num_samples)

        # 多个频率的正弦波 + 噪声
        data = np.column_stack([
            np.sin(2 * np.pi * 0.5 * t),
            np.sin(2 * np.pi * 1.0 * t + np.pi/4),
            np.sin(2 * np.pi * 2.0 * t + np.pi/2),
            np.cos(2 * np.pi * 0.7 * t),
            0.1 * np.random.randn(num_samples)
        ])

        # 目标是下一个时间点的值
        targets = data[1:, :self.target_dim]
        data = data[:-1]

        # 确保维度正确
        if data.shape[1] > self.input_dim:
            data = data[:, :self.input_dim]
        elif data.shape[1] < self.input_dim:
            padding = np.zeros((data.shape[0], self.input_dim - data.shape[1]))
            data = np.concatenate([data, padding], axis=1)

        return data, targets

    def get_train_loader(self, shuffle=True):
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

    def get_val_loader(self, shuffle=False):
        """获取验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

    def get_test_loader(self, shuffle=False):
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'test_samples': len(self.test_dataset),
            'input_dim': self.input_dim,
            'target_dim': self.target_dim,
            'seq_len': self.seq_len or self.train_dataset.data.shape[1],
            'batch_size': self.batch_size,
            'data_shape': self.train_dataset.data.shape,
            'target_shape': self.train_dataset.targets.shape,
            'normalized': self.normalize
        }

    def get_statistics(self):
        """获取数据统计信息"""
        train_stats = {
            'train_mean': self.train_dataset.data.mean().item(),
            'train_std': self.train_dataset.data.std().item(),
            'train_min': self.train_dataset.data.min().item(),
            'train_max': self.train_dataset.data.max().item(),
            'target_mean': self.train_dataset.targets.mean().item(),
            'target_std': self.train_dataset.targets.std().item()
        }

        val_stats = {
            'val_mean': self.val_dataset.data.mean().item(),
            'val_std': self.val_dataset.data.std().item(),
            'val_min': self.val_dataset.data.min().item(),
            'val_max': self.val_dataset.data.max().item(),
            'val_target_mean': self.val_dataset.targets.mean().item(),
            'val_target_std': self.val_dataset.targets.std().item()
        }

        return {**train_stats, **val_stats}


def test_data_loader():
    """测试数据加载器"""
    print("Testing DataLoader with DSA...")

    # 创建临时测试数据
    test_data_path = 'test_dataset.npy'
    test_data = np.random.randn(500, 10)
    np.save(test_data_path, test_data)

    try:
        # 创建数据加载器
        data_loader = DataLoaderWithDSA(
            dataset_path=test_data_path,
            input_dim=5,
            target_dim=1,
            seq_len=20,
            batch_size=16,
            train_ratio=0.8,
            normalize=True,
            augment_train=True
        )

        # 获取数据集信息
        info = data_loader.get_dataset_info()
        print(f"Dataset info: {info}")

        # 获取统计信息
        stats = data_loader.get_statistics()
        print(f"Data statistics: {stats}")

        # 测试数据加载
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # 获取一个batch
        for batch_idx, (data, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: data shape {data.shape}, targets shape {targets.shape}")
            if batch_idx >= 2:  # 只测试前几个batch
                break

        print("DataLoader test passed!")

    finally:
        # 清理测试文件
        if os.path.exists(test_data_path):
            os.remove(test_data_path)


if __name__ == "__main__":
    test_data_loader()